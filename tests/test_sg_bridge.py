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
    _strip_model_command,
    load_sg_bridge_config,
    reset_sg_bridge,
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
        assert cfg.claude_timeout == 120
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
    def test_execute_claude_escapes_quotes(self, mock_exec):
        """验证消息中的单引号被正确转义."""
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"ok", b""))
        proc.returncode = 0
        mock_exec.return_value = proc

        bridge = SGBridge(SGBridgeConfig())
        _run(bridge.execute_claude("what's up"))

        call_args = mock_exec.call_args[0]
        remote_cmd = call_args[-1]  # 最后一个参数是远程命令
        assert "'\\''" in remote_cmd  # 转义后的单引号


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

    @patch("crew.sg_bridge.asyncio.create_subprocess_exec")
    @patch("crew.sg_bridge.load_sg_bridge_config")
    def test_sg_success_returns_reply(self, mock_load, mock_exec):
        """SG 成功时返回回复."""
        mock_load.return_value = SGBridgeConfig(enabled=True)

        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"SG reply", b""))
        proc.returncode = 0
        mock_exec.return_value = proc

        reply = _run(sg_dispatch("你好", employee_name="ceo-assistant"))
        assert "SG reply" in reply
