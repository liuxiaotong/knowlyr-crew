"""测试 SG Bridge — SSH 转发、Circuit Breaker、模型选择."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crew.sg_bridge import (
    CBState,
    CircuitBreaker,
    ModelTier,
    SGBridge,
    SGBridgeConfig,
    SGBridgeError,
    SGBridgeTimeout,
    SGBridgeUnavailable,
    _get_employee_soul,
    _strip_model_command,
    load_sg_bridge_config,
    reset_sg_bridge,
    reset_soul_cache,
    select_model_tier,
    sg_dispatch,
)


def _run(coro):
    return asyncio.run(coro)


# ── 配置 ──


class TestSGBridgeConfig:
    """配置加载."""

    def test_defaults(self):
        cfg = SGBridgeConfig()
        assert cfg.ssh_host == "43.106.24.105"
        assert cfg.ssh_user == "root"
        assert cfg.enabled is True
        assert cfg.claude_timeout == 180
        assert cfg.cb_failure_threshold == 3

    def test_load_from_yaml(self, tmp_path):
        crew_dir = tmp_path / ".crew"
        crew_dir.mkdir()
        (crew_dir / "sg_bridge.yaml").write_text(
            "enabled: true\nssh_host: 1.2.3.4\nssh_user: deploy\nclaude_timeout: 60\n",
            encoding="utf-8",
        )
        cfg = load_sg_bridge_config(tmp_path)
        assert cfg.enabled is True
        assert cfg.ssh_host == "1.2.3.4"
        assert cfg.ssh_user == "deploy"
        assert cfg.claude_timeout == 60

    def test_load_missing_file(self, tmp_path):
        cfg = load_sg_bridge_config(tmp_path)
        assert cfg.enabled is False

    def test_load_invalid_yaml(self, tmp_path):
        crew_dir = tmp_path / ".crew"
        crew_dir.mkdir()
        (crew_dir / "sg_bridge.yaml").write_text("not: [valid: yaml: stuff", encoding="utf-8")
        cfg = load_sg_bridge_config(tmp_path)
        # 应该返回默认配置（disabled）而非崩溃
        assert cfg.enabled is False


# ── 模型选择 ──


class TestModelSelection:
    """模型分层路由."""

    def setup_method(self):
        self.config = SGBridgeConfig()

    def test_default_is_sonnet(self):
        assert select_model_tier("帮我看看这段代码", self.config) == ModelTier.SONNET

    def test_opus_keyword(self):
        assert select_model_tier("这个架构需要重新设计", self.config) == ModelTier.OPUS

    def test_opus_manual_command(self):
        assert select_model_tier("/model opus 请分析一下", self.config) == ModelTier.OPUS

    def test_haiku_short_greeting(self):
        assert select_model_tier("你好", self.config) == ModelTier.HAIKU

    def test_haiku_not_for_long_message(self):
        # 超过 haiku_max_length 的消息即使含闲聊关键词也不走 haiku
        assert select_model_tier("你好，帮我看看这个项目的整体架构设计有没有问题", self.config) == ModelTier.OPUS

    def test_empty_message(self):
        assert select_model_tier("", self.config) == ModelTier.SONNET

    def test_manual_haiku(self):
        assert select_model_tier("/model haiku 你好呀", self.config) == ModelTier.HAIKU

    def test_manual_sonnet(self):
        assert select_model_tier("/model sonnet 帮我看代码", self.config) == ModelTier.SONNET

    def test_strip_model_command(self):
        assert _strip_model_command("/model opus 分析架构") == "分析架构"
        assert _strip_model_command("你好") == "你好"
        assert _strip_model_command("/model haiku hi") == "hi"

    def test_case_insensitive(self):
        assert select_model_tier("OK", self.config) == ModelTier.HAIKU


# ── Circuit Breaker ──


class TestCircuitBreaker:
    """Circuit Breaker 状态转换."""

    def test_initial_state(self):
        cb = CircuitBreaker()
        assert cb.state == CBState.CLOSED
        assert cb.allow_request() is True

    def test_single_failure_stays_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        assert cb.state == CBState.CLOSED
        assert cb.allow_request() is True

    def test_threshold_failures_opens(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CBState.OPEN
        assert cb.allow_request() is False

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # 重置了，再失败 2 次不够阈值
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CBState.CLOSED

    def test_open_to_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        cb.record_failure()
        assert cb._state == CBState.OPEN
        # recovery_timeout=0 意味着立即恢复
        time.sleep(0.01)
        assert cb.state == CBState.HALF_OPEN
        assert cb.allow_request() is True

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        cb.record_failure()
        time.sleep(0.01)
        _ = cb.state  # 触发 OPEN → HALF_OPEN
        cb.record_success()
        assert cb.state == CBState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        cb.record_failure()
        time.sleep(0.01)
        _ = cb.state  # 触发 OPEN → HALF_OPEN
        cb.record_failure()
        assert cb._state == CBState.OPEN

    def test_reset(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb._state == CBState.OPEN
        cb.reset()
        assert cb.state == CBState.CLOSED
        assert cb.allow_request() is True


# ── SGBridge 核心 ──


class TestSGBridge:
    """SG Bridge SSH 转发."""

    def test_ssh_base_args(self):
        config = SGBridgeConfig(ssh_host="1.2.3.4", ssh_user="deploy", ssh_port=2222)
        bridge = SGBridge(config)
        args = bridge._ssh_base_args()
        assert "ssh" in args
        assert "-p" in args
        idx = args.index("-p")
        assert args[idx + 1] == "2222"

    def test_ssh_target(self):
        config = SGBridgeConfig(ssh_host="1.2.3.4", ssh_user="deploy")
        bridge = SGBridge(config)
        assert bridge._ssh_target() == "deploy@1.2.3.4"

    @patch("crew.sg_bridge.asyncio.create_subprocess_exec")
    def test_health_check_success(self, mock_exec):
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"ok\n", b""))
        proc.returncode = 0
        mock_exec.return_value = proc

        bridge = SGBridge(SGBridgeConfig())
        assert _run(bridge.health_check()) is True

    @patch("crew.sg_bridge.asyncio.create_subprocess_exec")
    def test_health_check_failure(self, mock_exec):
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"", b"Connection refused"))
        proc.returncode = 255
        mock_exec.return_value = proc

        bridge = SGBridge(SGBridgeConfig())
        assert _run(bridge.health_check()) is False

    @patch("crew.sg_bridge.asyncio.create_subprocess_exec")
    def test_execute_claude_success(self, mock_exec):
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"Hello! How can I help?", b""))
        proc.returncode = 0
        mock_exec.return_value = proc

        bridge = SGBridge(SGBridgeConfig())
        reply = _run(bridge.execute_claude("你好"))
        assert reply == "Hello! How can I help?"

    @patch("crew.sg_bridge.asyncio.create_subprocess_exec")
    def test_execute_claude_timeout(self, mock_exec):
        proc = AsyncMock()
        proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
        proc.kill = MagicMock()
        mock_exec.return_value = proc

        bridge = SGBridge(SGBridgeConfig(claude_timeout=1))
        with pytest.raises(SGBridgeTimeout):
            _run(bridge.execute_claude("长任务"))

    @patch("crew.sg_bridge.asyncio.create_subprocess_exec")
    def test_execute_claude_connection_refused(self, mock_exec):
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"", b"Connection refused"))
        proc.returncode = 255
        mock_exec.return_value = proc

        bridge = SGBridge(SGBridgeConfig())
        with pytest.raises(SGBridgeUnavailable, match="连接被拒"):
            _run(bridge.execute_claude("test"))

    @patch("crew.sg_bridge.asyncio.create_subprocess_exec")
    def test_execute_claude_empty_output(self, mock_exec):
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"", b""))
        proc.returncode = 0
        mock_exec.return_value = proc

        bridge = SGBridge(SGBridgeConfig())
        with pytest.raises(SGBridgeError, match="空输出"):
            _run(bridge.execute_claude("test"))

    @patch("crew.sg_bridge.asyncio.create_subprocess_exec")
    def test_execute_claude_with_opus(self, mock_exec):
        """验证 opus 模型参数传递."""
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"opus reply", b""))
        proc.returncode = 0
        mock_exec.return_value = proc

        bridge = SGBridge(SGBridgeConfig())
        reply = _run(bridge.execute_claude("分析架构", model_tier=ModelTier.OPUS))
        assert reply == "opus reply"

        # 检查命令中包含 --model opus
        call_args = mock_exec.call_args[0]
        cmd_str = " ".join(str(a) for a in call_args)
        assert "--model" in cmd_str
        assert "opus" in cmd_str

    @patch("crew.sg_bridge.asyncio.create_subprocess_exec")
    def test_execute_claude_uses_stdin(self, mock_exec):
        """验证消息通过 stdin 传递（避免多层 SSH 引号问题）."""
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"ok", b""))
        proc.returncode = 0
        mock_exec.return_value = proc

        bridge = SGBridge(SGBridgeConfig())
        _run(bridge.execute_claude("what's up"))

        # 消息通过 stdin 传递，不在远程命令中
        call_args = mock_exec.call_args
        remote_cmd = call_args[0][-1]  # 最后一个参数是远程命令
        assert "what's up" not in remote_cmd  # 消息不在命令行
        # communicate 收到 stdin input
        comm_kwargs = proc.communicate.call_args
        assert comm_kwargs[1]["input"] == b"what's up"


# ── sg_dispatch 高层 API ──


class TestSGDispatch:
    """sg_dispatch 集成测试."""

    def setup_method(self):
        reset_sg_bridge()

    def teardown_method(self):
        reset_sg_bridge()

    @patch("crew.sg_bridge.load_sg_bridge_config")
    def test_disabled_bridge_raises(self, mock_load):
        mock_load.return_value = SGBridgeConfig(enabled=False)
        with pytest.raises(SGBridgeError, match="未启用"):
            _run(sg_dispatch("test"))

    @patch("crew.sg_bridge.asyncio.create_subprocess_exec")
    @patch("crew.sg_bridge.load_sg_bridge_config")
    def test_success_path(self, mock_load, mock_exec):
        mock_load.return_value = SGBridgeConfig(enabled=True)

        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"SG reply OK", b""))
        proc.returncode = 0
        mock_exec.return_value = proc

        reply = _run(sg_dispatch("帮我看看代码"))
        assert reply == "SG reply OK"

    @patch("crew.sg_bridge.asyncio.create_subprocess_exec")
    @patch("crew.sg_bridge.load_sg_bridge_config")
    def test_fallback_on_timeout(self, mock_load, mock_exec):
        mock_load.return_value = SGBridgeConfig(enabled=True, claude_timeout=1)

        proc = AsyncMock()
        proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
        proc.kill = MagicMock()
        mock_exec.return_value = proc

        with pytest.raises(SGBridgeTimeout):
            _run(sg_dispatch("长任务"))

    @patch("crew.sg_bridge.asyncio.create_subprocess_exec")
    @patch("crew.sg_bridge.load_sg_bridge_config")
    def test_circuit_breaker_triggers_after_failures(self, mock_load, mock_exec):
        mock_load.return_value = SGBridgeConfig(
            enabled=True,
            cb_failure_threshold=2,
            claude_timeout=1,
        )

        proc = AsyncMock()
        proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
        proc.kill = MagicMock()
        mock_exec.return_value = proc

        # 前 2 次超时，触发熔断
        for _ in range(2):
            with pytest.raises(SGBridgeTimeout):
                _run(sg_dispatch("test"))
            reset_sg_bridge.__wrapped__ = None  # 不重置单例

        # 第 3 次应该直接被熔断器拒绝（不会实际执行 SSH）
        # 需要复用同一个 bridge 实例
        bridge = _get_bridge_for_test(mock_load)
        assert bridge is not None
        assert not bridge.circuit_breaker.allow_request()

    @patch("crew.sg_bridge.asyncio.create_subprocess_exec")
    @patch("crew.sg_bridge.load_sg_bridge_config")
    def test_model_tier_in_dispatch(self, mock_load, mock_exec):
        """验证 sg_dispatch 正确选择模型."""
        mock_load.return_value = SGBridgeConfig(enabled=True)

        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"opus analysis", b""))
        proc.returncode = 0
        mock_exec.return_value = proc

        # 包含 opus 关键词
        reply = _run(sg_dispatch("请做一下架构复盘"))
        assert reply == "opus analysis"

        # 验证命令包含 --model opus
        call_args = mock_exec.call_args[0]
        cmd_str = " ".join(str(a) for a in call_args)
        assert "--model" in cmd_str and "opus" in cmd_str


def _get_bridge_for_test(mock_load):
    """测试辅助：获取当前 bridge 单例."""
    from crew.sg_bridge import _bridge_instance
    return _bridge_instance


# ── Soul 动态加载 ──


class TestEmployeeSoul:
    """从 crew 发现层动态获取员工 soul."""

    def setup_method(self):
        reset_soul_cache()

    def teardown_method(self):
        reset_soul_cache()

    @patch("crew.discovery.discover_employees")
    def test_chat_profile_preferred_over_body(self, mock_discover):
        """有 chat-profile.md 时优先加载精简版."""
        mock_emp = MagicMock()
        mock_emp.body = "完整版 body 非常长" * 100
        # 模拟 source_path 下有 chat-profile.md
        mock_emp.source_path = Path(tmp_dir := "/tmp/_test_chat_profile")
        Path(tmp_dir).mkdir(exist_ok=True)
        (Path(tmp_dir) / "chat-profile.md").write_text(
            "你是墨言，有猫叫阿灰。", encoding="utf-8"
        )
        mock_result = MagicMock()
        mock_result.get.return_value = mock_emp
        mock_discover.return_value = mock_result

        soul = _get_employee_soul("ceo-assistant")
        assert "阿灰" in soul
        assert len(soul) < 100  # chat-profile 精简版
        # 清理
        (Path(tmp_dir) / "chat-profile.md").unlink()
        Path(tmp_dir).rmdir()

    @patch("crew.discovery.discover_employees")
    def test_fallback_to_body_without_chat_profile(self, mock_discover):
        """没有 chat-profile.md 时 fallback 到完整 body."""
        mock_emp = MagicMock()
        mock_emp.body = "你是姜墨言，有一只灰色英短叫阿灰。"
        mock_emp.source_path = Path("/tmp/_nonexistent_dir")
        mock_result = MagicMock()
        mock_result.get.return_value = mock_emp
        mock_discover.return_value = mock_result

        soul = _get_employee_soul("ceo-assistant")
        assert "阿灰" in soul
        mock_discover.assert_called_once()

    @patch("crew.discovery.discover_employees")
    def test_soul_cached_with_ttl(self, mock_discover):
        """第二次调用走缓存，不再调 discover."""
        mock_emp = MagicMock()
        mock_emp.body = "cached soul"
        mock_emp.source_path = None
        mock_result = MagicMock()
        mock_result.get.return_value = mock_emp
        mock_discover.return_value = mock_result

        _get_employee_soul("ceo-assistant")
        _get_employee_soul("ceo-assistant")
        # 只调了一次 discover
        assert mock_discover.call_count == 1

    @patch("crew.discovery.discover_employees")
    def test_soul_not_found_returns_empty(self, mock_discover):
        """员工不存在时返回空字符串."""
        mock_result = MagicMock()
        mock_result.get.return_value = None
        mock_discover.return_value = mock_result

        soul = _get_employee_soul("nonexistent")
        assert soul == ""

    @patch("crew.discovery.discover_employees")
    def test_soul_exception_returns_empty(self, mock_discover):
        """discover 抛异常时优雅降级."""
        mock_discover.side_effect = RuntimeError("boom")

        soul = _get_employee_soul("ceo-assistant")
        assert soul == ""

    @patch("crew.discovery.discover_employees")
    def test_soul_exception_uses_stale_cache(self, mock_discover):
        """discover 抛异常但有旧缓存时，返回旧缓存."""
        # 第一次正常
        mock_emp = MagicMock()
        mock_emp.body = "stale soul"
        mock_emp.source_path = None
        mock_result = MagicMock()
        mock_result.get.return_value = mock_emp
        mock_discover.return_value = mock_result
        _get_employee_soul("ceo-assistant")

        # 强制缓存过期
        from crew.sg_bridge import _soul_cache
        name, (body, _ts) = next(iter(_soul_cache.items()))
        _soul_cache[name] = (body, 0)  # 过期

        # 第二次 discover 失败
        mock_discover.side_effect = RuntimeError("network error")
        soul = _get_employee_soul("ceo-assistant")
        assert soul == "stale soul"


# ── webhook_feishu 集成 ──


class TestWebhookFeishuSGIntegration:
    """验证 _feishu_dispatch 中 SG Bridge 的 fallback 行为."""

    def setup_method(self):
        reset_sg_bridge()

    def teardown_method(self):
        reset_sg_bridge()

    @patch("crew.sg_bridge.load_sg_bridge_config")
    def test_sg_disabled_falls_through(self, mock_load):
        """SG Bridge 未启用时，应走现有 crew 引擎."""
        mock_load.return_value = SGBridgeConfig(enabled=False)
        # 只验证 sg_dispatch 抛出 SGBridgeError
        with pytest.raises(SGBridgeError):
            _run(sg_dispatch("test"))

    @patch("crew.sg_bridge._get_employee_soul", return_value="")
    @patch("crew.sg_bridge.asyncio.create_subprocess_exec")
    @patch("crew.sg_bridge.load_sg_bridge_config")
    def test_sg_success_returns_reply(self, mock_load, mock_exec, _mock_soul):
        """SG 成功时返回回复."""
        mock_load.return_value = SGBridgeConfig(enabled=True)

        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"SG reply", b""))
        proc.returncode = 0
        mock_exec.return_value = proc

        reply = _run(sg_dispatch("你好", employee_name="ceo-assistant"))
        assert "SG reply" in reply

    @patch("crew.sg_bridge._get_employee_soul", return_value="你是墨言，有一只猫叫阿灰。")
    @patch("crew.sg_bridge.asyncio.create_subprocess_exec")
    @patch("crew.sg_bridge.load_sg_bridge_config")
    def test_sg_soul_injected_via_stdin(self, mock_load, mock_exec, _mock_soul):
        """验证 soul 通过 stdin <identity> 块注入."""
        mock_load.return_value = SGBridgeConfig(enabled=True)

        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"meow", b""))
        proc.returncode = 0
        mock_exec.return_value = proc

        _run(sg_dispatch("你的猫叫什么", employee_name="ceo-assistant"))

        # stdin 应包含 <identity> 块和 soul 内容
        comm_kwargs = proc.communicate.call_args
        stdin_data = comm_kwargs[1]["input"].decode("utf-8")
        assert "<identity>" in stdin_data
        assert "阿灰" in stdin_data
        assert "你的猫叫什么" in stdin_data
