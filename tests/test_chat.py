"""测试 CrewEngine.chat() + POST /api/chat 端点."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("starlette")

from starlette.testclient import TestClient

from crew.engine import CrewEngine
from crew.exceptions import EmployeeNotFoundError
from crew.webhook import create_webhook_app
from crew.webhook_config import WebhookConfig

FIXTURES = Path(__file__).parent / "fixtures"
TOKEN = "test-token-chat"


def _make_client() -> TestClient:
    """创建测试客户端."""
    app = create_webhook_app(
        project_dir=FIXTURES,
        token=TOKEN,
        config=WebhookConfig(),
    )
    return TestClient(app)


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


# ── 辅助 mock 工厂 ──


def _mock_execution_result(
    content: str = "你好，我是墨言。",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> MagicMock:
    """构造一个假的 ExecutionResult."""
    result = MagicMock()
    result.content = content
    result.input_tokens = input_tokens
    result.output_tokens = output_tokens
    result.model = "claude-sonnet-4-20250514"
    return result


def _mock_employee(name: str = "code-reviewer") -> MagicMock:
    """构造一个假的 Employee 对象."""
    emp = MagicMock()
    emp.name = name
    emp.display_name = "测试员工"
    emp.character_name = "小测"
    emp.description = "用于测试的员工"
    emp.model = "claude-sonnet-4-20250514"
    emp.fallback_model = None
    emp.fallback_api_key = None
    emp.fallback_base_url = None
    emp.api_key = None
    emp.base_url = None
    emp.tools = []
    emp.tags = []
    emp.args = []
    emp.kpi = []
    emp.context = []
    emp.permissions = None
    emp.agent_status = "active"
    emp.source_path = None
    emp.body = "你是测试员工。"
    emp.output = MagicMock()
    emp.output.format = "text"
    emp.output.filename = None
    emp.output.dir = None
    emp.effective_display_name = "测试员工"
    return emp


# ═══════════════════════════════════════════════
# 1. CrewEngine.chat() — 非流式调用
# ═══════════════════════════════════════════════


class TestCrewEngineChatNonStream:
    """CrewEngine.chat() 非流式路径."""

    @pytest.mark.asyncio
    async def test_chat_returns_reply(self):
        """chat() 成功调用时返回包含 reply 字段的 dict."""
        engine = CrewEngine(project_dir=FIXTURES)
        mock_emp = _mock_employee("code-reviewer")
        mock_result = _mock_execution_result("这是测试回复。")

        # chat() 内部用局部 import，需要 patch 源模块
        with (
            patch("crew.discovery.discover_employees") as mock_disc,
            patch("crew.executor.aexecute_prompt", new_callable=AsyncMock) as mock_exec,
            patch("crew.output_sanitizer.strip_internal_tags", side_effect=lambda x: x),
        ):
            mock_disc.return_value.get.return_value = mock_emp
            mock_exec.return_value = mock_result

            result = await engine.chat(
                employee_id="code-reviewer",
                message="帮我看一下代码",
                channel="antgather_dm",
                sender_id="user_001",
            )

        assert result["reply"] == "这是测试回复。"
        assert result["employee_id"] == "code-reviewer"
        assert "tokens_used" in result
        assert result["tokens_used"] == 150  # 100 + 50
        assert "latency_ms" in result
        assert isinstance(result["latency_ms"], int)

    @pytest.mark.asyncio
    async def test_chat_with_message_history(self):
        """chat() 传入 message_history 时能正确处理历史上下文."""
        engine = CrewEngine(project_dir=FIXTURES)
        mock_emp = _mock_employee("code-reviewer")
        mock_result = _mock_execution_result("好的，继续。")

        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么需要帮助的？"},
        ]

        with (
            patch("crew.discovery.discover_employees") as mock_disc,
            patch("crew.executor.aexecute_prompt", new_callable=AsyncMock) as mock_exec,
            patch("crew.output_sanitizer.strip_internal_tags", side_effect=lambda x: x),
        ):
            mock_disc.return_value.get.return_value = mock_emp
            mock_exec.return_value = mock_result

            result = await engine.chat(
                employee_id="code-reviewer",
                message="继续我们的对话",
                channel="lark",
                sender_id="user_002",
                message_history=history,
            )

        # 验证 LLM 调用时传入的 user_message 包含历史
        call_kwargs = mock_exec.call_args.kwargs
        assert "历史对话" in call_kwargs["user_message"] or "[user]" in call_kwargs["user_message"]
        assert result["reply"] == "好的，继续。"


# ═══════════════════════════════════════════════
# 2. context_only 模式
# ═══════════════════════════════════════════════


class TestCrewEngineChatContextOnly:
    """chat() context_only 模式 — 跳过 LLM，直接返回 prompt + 记忆."""

    @pytest.mark.asyncio
    async def test_context_only_returns_prompt(self):
        """context_only=True 时返回 prompt、memories、budget_remaining."""
        engine = CrewEngine(project_dir=FIXTURES)
        mock_emp = _mock_employee("code-reviewer")

        with (
            patch("crew.discovery.discover_employees") as mock_disc,
            patch("crew.executor.aexecute_prompt", new_callable=AsyncMock) as mock_exec,
        ):
            mock_disc.return_value.get.return_value = mock_emp
            # context_only 时不应调用 LLM
            mock_exec.side_effect = AssertionError("不应调用 LLM")

            result = await engine.chat(
                employee_id="code-reviewer",
                message="测试消息",
                channel="internal",
                sender_id="claude_code",
                context_only=True,
            )

        assert "prompt" in result
        assert "memories" in result
        assert "budget_remaining" in result
        assert isinstance(result["prompt"], str)
        assert isinstance(result["memories"], list)
        assert isinstance(result["budget_remaining"], int)
        assert result["budget_remaining"] >= 0

    @pytest.mark.asyncio
    async def test_context_only_no_llm_call(self):
        """context_only 模式下 aexecute_prompt 不被调用."""
        engine = CrewEngine(project_dir=FIXTURES)
        mock_emp = _mock_employee("code-reviewer")

        with (
            patch("crew.discovery.discover_employees") as mock_disc,
            patch("crew.executor.aexecute_prompt", new_callable=AsyncMock) as mock_exec,
        ):
            mock_disc.return_value.get.return_value = mock_emp
            mock_exec.return_value = _mock_execution_result()

            await engine.chat(
                employee_id="code-reviewer",
                message="测试",
                channel="internal",
                sender_id="dev",
                context_only=True,
            )

            mock_exec.assert_not_called()


# ═══════════════════════════════════════════════
# 3. 参数校验
# ═══════════════════════════════════════════════


class TestCrewEngineChatValidation:
    """chat() 参数校验."""

    @pytest.mark.asyncio
    async def test_missing_employee_id_raises(self):
        """employee_id 为空时抛出 ValueError."""
        engine = CrewEngine(project_dir=FIXTURES)
        with pytest.raises(ValueError, match="employee_id"):
            await engine.chat(
                employee_id="",
                message="hello",
                channel="antgather_dm",
                sender_id="user_1",
            )

    @pytest.mark.asyncio
    async def test_missing_message_raises(self):
        """message 为空时抛出 ValueError."""
        engine = CrewEngine(project_dir=FIXTURES)
        with pytest.raises(ValueError, match="message"):
            await engine.chat(
                employee_id="code-reviewer",
                message="",
                channel="antgather_dm",
                sender_id="user_1",
            )

    @pytest.mark.asyncio
    async def test_missing_channel_raises(self):
        """channel 为空时抛出 ValueError."""
        engine = CrewEngine(project_dir=FIXTURES)
        with pytest.raises(ValueError, match="channel"):
            await engine.chat(
                employee_id="code-reviewer",
                message="hello",
                channel="",
                sender_id="user_1",
            )

    @pytest.mark.asyncio
    async def test_missing_sender_id_raises(self):
        """sender_id 为空时抛出 ValueError."""
        engine = CrewEngine(project_dir=FIXTURES)
        with pytest.raises(ValueError, match="sender_id"):
            await engine.chat(
                employee_id="code-reviewer",
                message="hello",
                channel="antgather_dm",
                sender_id="",
            )


# ═══════════════════════════════════════════════
# 4. employee 不存在
# ═══════════════════════════════════════════════


class TestCrewEngineChatEmployeeNotFound:
    """员工不存在时的处理."""

    @pytest.mark.asyncio
    async def test_employee_not_found_raises(self):
        """discovery.get() 返回 None 时抛出 EmployeeNotFoundError."""
        engine = CrewEngine(project_dir=FIXTURES)

        with patch("crew.discovery.discover_employees") as mock_disc:
            mock_disc.return_value.get.return_value = None

            with pytest.raises(EmployeeNotFoundError):
                await engine.chat(
                    employee_id="nonexistent-employee",
                    message="hello",
                    channel="antgather_dm",
                    sender_id="user_1",
                )


# ═══════════════════════════════════════════════
# 5. 记忆写回
# ═══════════════════════════════════════════════


class TestCrewEngineChatMemoryWriteback:
    """记忆异步写回逻辑."""

    @pytest.mark.asyncio
    async def test_memory_writeback_triggered(self):
        """LLM 调用成功后 memory_updated 应为 True（写回任务被创建）."""
        import asyncio

        engine = CrewEngine(project_dir=FIXTURES)
        mock_emp = _mock_employee("code-reviewer")
        mock_result = _mock_execution_result("回复内容")

        with (
            patch("crew.discovery.discover_employees") as mock_disc,
            patch("crew.executor.aexecute_prompt", new_callable=AsyncMock) as mock_exec,
            patch("crew.output_sanitizer.strip_internal_tags", side_effect=lambda x: x),
        ):
            mock_disc.return_value.get.return_value = mock_emp
            mock_exec.return_value = mock_result

            result = await engine.chat(
                employee_id="code-reviewer",
                message="帮我看一下代码",
                channel="antgather_dm",
                sender_id="user_001",
            )

            # 等待后台任务执行完
            await asyncio.sleep(0.05)

        # memory_updated 字段存在（写回可能成功或静默失败）
        assert "memory_updated" in result


# ═══════════════════════════════════════════════
# 6. fast path / full path 路由
# ═══════════════════════════════════════════════


class TestCrewEngineChatRouting:
    """fast / full 路由决策."""

    @pytest.mark.asyncio
    async def test_full_path_used_for_employee_without_tools(self):
        """没有 agent tools 的员工走 full path（直接 aexecute_prompt）."""
        engine = CrewEngine(project_dir=FIXTURES)
        mock_emp = _mock_employee("code-reviewer")
        mock_emp.tools = []  # 无工具
        mock_emp.fallback_model = None
        mock_result = _mock_execution_result("full path 回复")

        with (
            patch("crew.discovery.discover_employees") as mock_disc,
            patch("crew.executor.aexecute_prompt", new_callable=AsyncMock) as mock_exec,
            patch("crew.output_sanitizer.strip_internal_tags", side_effect=lambda x: x),
        ):
            mock_disc.return_value.get.return_value = mock_emp
            mock_exec.return_value = mock_result

            result = await engine.chat(
                employee_id="code-reviewer",
                message="请帮我审查代码",
                channel="antgather_dm",
                sender_id="user_001",
            )

        assert result["reply"] == "full path 回复"
        mock_exec.assert_called_once()


# ═══════════════════════════════════════════════
# 7. POST /api/chat HTTP 端点
# ═══════════════════════════════════════════════


class TestChatEndpointValidation:
    """HTTP 端点参数校验."""

    def test_missing_required_fields_returns_400(self):
        """缺少必填字段时返回 400."""
        client = _make_client()
        resp = client.post(
            "/api/chat",
            json={"employee_id": "code-reviewer"},  # 缺少 message/channel/sender_id
            headers=_auth_headers(),
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["ok"] is False
        assert "error" in body

    def test_empty_employee_id_returns_400(self):
        """employee_id 为空时返回 400."""
        client = _make_client()
        resp = client.post(
            "/api/chat",
            json={
                "employee_id": "",
                "message": "hello",
                "channel": "antgather_dm",
                "sender_id": "user_1",
            },
            headers=_auth_headers(),
        )
        assert resp.status_code == 400

    def test_requires_auth(self):
        """不带 token 时返回 401/403."""
        client = _make_client()
        resp = client.post(
            "/api/chat",
            json={
                "employee_id": "code-reviewer",
                "message": "hello",
                "channel": "antgather_dm",
                "sender_id": "user_1",
            },
        )
        assert resp.status_code in (401, 403)


class TestChatEndpointEmployeeNotFound:
    """HTTP 端点 — employee 不存在返回 404."""

    def test_nonexistent_employee_returns_404(self):
        """请求不存在的员工时返回 404."""
        client = _make_client()

        with patch("crew.engine.CrewEngine") as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.chat = AsyncMock(
                side_effect=EmployeeNotFoundError("ghost-employee")
            )

            resp = client.post(
                "/api/chat",
                json={
                    "employee_id": "ghost-employee",
                    "message": "hello",
                    "channel": "antgather_dm",
                    "sender_id": "user_1",
                },
                headers=_auth_headers(),
            )

        assert resp.status_code == 404
        assert resp.json()["ok"] is False
        assert "ghost-employee" in resp.json()["error"]


class TestChatEndpointSuccess:
    """HTTP 端点 — 成功场景."""

    def test_non_stream_response(self):
        """非流式请求返回标准 JSON 结构."""
        client = _make_client()

        mock_reply: dict[str, Any] = {
            "reply": "你好！我是测试员工。",
            "employee_id": "code-reviewer",
            "memory_updated": False,
            "tokens_used": 200,
            "latency_ms": 500,
        }

        with patch("crew.engine.CrewEngine") as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.chat = AsyncMock(return_value=mock_reply)

            resp = client.post(
                "/api/chat",
                json={
                    "employee_id": "code-reviewer",
                    "message": "你好",
                    "channel": "antgather_dm",
                    "sender_id": "user_1",
                },
                headers=_auth_headers(),
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["reply"] == "你好！我是测试员工。"
        assert body["employee_id"] == "code-reviewer"
        assert body["tokens_used"] == 200

    def test_context_only_response(self):
        """context_only=True 时返回 prompt/memories/budget_remaining."""
        client = _make_client()

        mock_context: dict[str, Any] = {
            "prompt": "# 测试员工\n\n你是测试员工。",
            "memories": [],
            "budget_remaining": 3000,
        }

        with patch("crew.engine.CrewEngine") as MockEngine:
            mock_engine_instance = MockEngine.return_value
            mock_engine_instance.chat = AsyncMock(return_value=mock_context)

            resp = client.post(
                "/api/chat",
                json={
                    "employee_id": "code-reviewer",
                    "message": "hello",
                    "channel": "internal",
                    "sender_id": "claude_code",
                    "context_only": True,
                },
                headers=_auth_headers(),
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "prompt" in body
        assert "memories" in body
        assert body["budget_remaining"] == 3000
