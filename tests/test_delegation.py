"""员工委派能力测试."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from crew.models import ToolCall, ToolExecutionResult
from crew.tool_schema import _TOOL_SCHEMAS, employee_tools_to_schemas, is_finish_tool

# ── helpers ──

# 本地 import 需要 patch 源模块
_P_DISCOVER = "crew.discovery.discover_employees"
_P_ENGINE = "crew.engine.CrewEngine"
_P_EXEC_PROMPT = "crew.executor.aexecute_prompt"
_P_EXEC_TOOLS = "crew.executor.aexecute_with_tools"
_P_DETECT = "crew.providers.detect_provider"
_P_DELEGATE = "crew.webhook._delegate_employee"
_P_TOOLS_LOOP = "crew.webhook._execute_employee_with_tools"


def _run(coro):
    return asyncio.run(coro)


def _make_ctx(project_dir: Path | None = None) -> Any:
    ctx = MagicMock()
    ctx.project_dir = project_dir or Path("/tmp/test-project")
    ctx.registry = MagicMock()
    ctx.registry.create.return_value = MagicMock(task_id="test-task-id")
    return ctx


def _make_tool_result(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        content=content,
        tool_calls=tool_calls or [],
        model="test-model",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stop_reason="end_turn" if not tool_calls else "tool_use",
    )


def _make_emp(
    *,
    tools=None,
    model="test-model",
    name="test",
    desc="test",
    agent_id=None,
    base_url="",
    api_key="",
):
    emp = MagicMock()
    emp.tools = tools or []
    emp.model = model
    emp.character_name = name
    emp.effective_display_name = name
    emp.description = desc
    emp.args = []
    emp.agent_id = agent_id
    emp.base_url = base_url
    emp.api_key = api_key
    emp.fallback_model = ""
    emp.fallback_api_key = ""
    emp.fallback_base_url = ""
    emp.permissions = None
    return emp


# ── Step 1: tool_schema tests ──


class TestDelegateSchema:
    def test_delegate_in_schemas(self):
        assert "delegate" in _TOOL_SCHEMAS

    def test_delegate_schema_structure(self):
        schema = _TOOL_SCHEMAS["delegate"]
        assert schema["name"] == "delegate"
        props = schema["input_schema"]["properties"]
        assert "employee_name" in props
        assert "task" in props
        assert schema["input_schema"]["required"] == ["employee_name", "task"]

    def test_employee_tools_to_schemas_includes_delegate(self):
        # defer=False: delegate 直接在 schemas 中
        schemas, _ = employee_tools_to_schemas(["delegate"], defer=False)
        names = {s["name"] for s in schemas}
        assert "delegate" in names
        assert "submit" in names

    def test_employee_tools_to_schemas_defers_delegate(self):
        # defer=True: delegate 在 deferred 集合中，schemas 只有 load_tools + submit
        schemas, deferred = employee_tools_to_schemas(["delegate"])
        names = {s["name"] for s in schemas}
        assert "delegate" not in names
        assert "delegate" in deferred
        assert "load_tools" in names
        assert "submit" in names

    def test_employee_tools_to_schemas_without_delegate(self):
        schemas, _ = employee_tools_to_schemas(["file_read"], defer=False)
        names = {s["name"] for s in schemas}
        assert "delegate" not in names
        assert "file_read" in names
        assert "submit" in names

    def test_delegate_not_finish_tool(self):
        assert not is_finish_tool("delegate")


# ── Step 2a: _delegate_employee tests ──


class TestDelegateEmployee:
    @patch(_P_EXEC_PROMPT, new_callable=AsyncMock)
    @patch(_P_ENGINE)
    @patch(_P_DISCOVER)
    def test_success(self, mock_disc, mock_engine_cls, mock_exec):
        from crew.executor import ExecutionResult

        emp = _make_emp()
        discovery = MagicMock()
        discovery.get.return_value = emp
        mock_disc.return_value = discovery

        mock_engine = MagicMock()
        mock_engine.prompt.return_value = "prompt"
        mock_engine_cls.return_value = mock_engine

        mock_exec.return_value = ExecutionResult(
            content="审查结果：代码质量良好",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
        )

        from crew.webhook import _delegate_employee

        result = _run(_delegate_employee(_make_ctx(), "code-reviewer", "审查 auth.py"))
        assert "审查结果" in result
        mock_exec.assert_called_once()

    @patch(_P_DISCOVER)
    def test_employee_not_found(self, mock_disc):
        discovery = MagicMock()
        discovery.get.return_value = None
        discovery.employees = {"code-reviewer": MagicMock(), "doc-writer": MagicMock()}
        mock_disc.return_value = discovery

        from crew.webhook import _delegate_employee

        result = _run(_delegate_employee(_make_ctx(), "nonexistent", "task"))
        assert "错误" in result
        assert "未找到员工" in result

    @patch(_P_EXEC_PROMPT, new_callable=AsyncMock)
    @patch(_P_ENGINE)
    @patch(_P_DISCOVER)
    def test_execution_error(self, mock_disc, mock_engine_cls, mock_exec):
        emp = _make_emp()
        discovery = MagicMock()
        discovery.get.return_value = emp
        mock_disc.return_value = discovery
        mock_engine_cls.return_value = MagicMock()

        mock_exec.side_effect = RuntimeError("API error")

        from crew.webhook import _delegate_employee

        result = _run(_delegate_employee(_make_ctx(), "code-reviewer", "task"))
        assert "委派执行失败" in result


# ── Step 2b: _execute_employee_with_tools tests ──


class TestExecuteEmployeeWithTools:
    def test_no_tool_calls_returns_text(self):
        """LLM 直接返回文本，不调用任何工具."""
        with (
            patch(_P_DISCOVER) as mock_disc,
            patch(_P_ENGINE) as mock_engine_cls,
            patch(_P_EXEC_TOOLS, new_callable=AsyncMock) as mock_exec,
            patch(_P_DETECT) as mock_detect,
        ):
            from crew.providers import Provider

            emp = _make_emp(tools=["delegate"])
            discovery = MagicMock()
            discovery.get.return_value = emp
            discovery.employees = {"test": emp}
            mock_disc.return_value = discovery

            mock_engine = MagicMock()
            mock_engine.prompt.return_value = "system prompt"
            mock_engine_cls.return_value = mock_engine

            mock_detect.return_value = Provider.MOONSHOT
            mock_exec.return_value = _make_tool_result(content="直接回复内容")

            from crew.webhook import _execute_employee_with_tools

            result = _run(_execute_employee_with_tools(_make_ctx(), "test", {"task": "hello"}))
            assert result["output"] == "直接回复内容"
            assert result["employee"] == "test"
            mock_exec.assert_called_once()

    def test_delegate_then_submit_openai(self):
        """OpenAI 格式：LLM 先委派再提交结果."""
        with (
            patch(_P_DISCOVER) as mock_disc,
            patch(_P_ENGINE) as mock_engine_cls,
            patch(_P_EXEC_TOOLS, new_callable=AsyncMock) as mock_exec,
            patch(_P_DETECT) as mock_detect,
            patch(_P_DELEGATE, new_callable=AsyncMock) as mock_delegate,
        ):
            from crew.providers import Provider

            emp = _make_emp(tools=["delegate"], name="Boss")
            discovery = MagicMock()
            discovery.get.return_value = emp
            discovery.employees = {"boss": emp}
            mock_disc.return_value = discovery

            mock_engine = MagicMock()
            mock_engine.prompt.return_value = "prompt"
            mock_engine_cls.return_value = mock_engine

            mock_detect.return_value = Provider.MOONSHOT

            mock_exec.side_effect = [
                _make_tool_result(
                    content="我来委派",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="delegate",
                            arguments={
                                "employee_name": "code-reviewer",
                                "task": "审查代码",
                            },
                        )
                    ],
                ),
                _make_tool_result(
                    content="汇总结果",
                    tool_calls=[
                        ToolCall(
                            id="call_2",
                            name="submit",
                            arguments={
                                "result": "最终汇总：代码质量良好",
                            },
                        )
                    ],
                ),
            ]

            mock_delegate.return_value = "代码审查完毕，没有问题"

            from crew.webhook import _execute_employee_with_tools

            result = _run(_execute_employee_with_tools(_make_ctx(), "boss", {"task": "审查代码"}))
            assert result["output"] == "最终汇总：代码质量良好"
            assert mock_delegate.call_count == 1
            call_args = mock_delegate.call_args[0]
            assert call_args[1] == "code-reviewer"
            assert call_args[2] == "审查代码"
            assert mock_exec.call_count == 2

    def test_anthropic_message_format(self):
        """Anthropic 格式 tool_use/tool_result 消息."""
        with (
            patch(_P_DISCOVER) as mock_disc,
            patch(_P_ENGINE) as mock_engine_cls,
            patch(_P_EXEC_TOOLS, new_callable=AsyncMock) as mock_exec,
            patch(_P_DETECT) as mock_detect,
            patch(_P_DELEGATE, new_callable=AsyncMock) as mock_delegate,
        ):
            from crew.providers import Provider

            emp = _make_emp(tools=["delegate"], model="claude-sonnet-4-20250514")
            discovery = MagicMock()
            discovery.get.return_value = emp
            discovery.employees = {"boss": emp}
            mock_disc.return_value = discovery

            mock_engine = MagicMock()
            mock_engine.prompt.return_value = "prompt"
            mock_engine_cls.return_value = mock_engine

            mock_detect.return_value = Provider.ANTHROPIC

            mock_exec.side_effect = [
                _make_tool_result(
                    content="委派中",
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            name="delegate",
                            arguments={
                                "employee_name": "doc-writer",
                                "task": "写文档",
                            },
                        )
                    ],
                ),
                _make_tool_result(content="文档已完成"),
            ]
            mock_delegate.return_value = "文档内容"

            from crew.webhook import _execute_employee_with_tools

            result = _run(_execute_employee_with_tools(_make_ctx(), "boss", {"task": "写文档"}))
            assert result["output"] == "文档已完成"

            # 验证 Anthropic 格式 messages
            second_call_msgs = mock_exec.call_args_list[1].kwargs["messages"]
            assistant_msg = second_call_msgs[1]
            assert assistant_msg["role"] == "assistant"
            assert any(b.get("type") == "tool_use" for b in assistant_msg["content"])

            tool_msg = second_call_msgs[2]
            assert tool_msg["role"] == "user"
            assert any(b.get("type") == "tool_result" for b in tool_msg["content"])

    def test_max_iterations(self):
        """达到最大轮次限制时停止."""
        with (
            patch(_P_DISCOVER) as mock_disc,
            patch(_P_ENGINE) as mock_engine_cls,
            patch(_P_EXEC_TOOLS, new_callable=AsyncMock) as mock_exec,
            patch(_P_DETECT) as mock_detect,
            patch(_P_DELEGATE, new_callable=AsyncMock) as mock_delegate,
            patch("crew.webhook._MAX_TOOL_ROUNDS", 2),
        ):
            from crew.providers import Provider

            emp = _make_emp(tools=["delegate"], name="Boss")
            discovery = MagicMock()
            discovery.get.return_value = emp
            discovery.employees = {"boss": emp}
            mock_disc.return_value = discovery

            mock_engine = MagicMock()
            mock_engine.prompt.return_value = "prompt"
            mock_engine_cls.return_value = mock_engine

            mock_detect.return_value = Provider.MOONSHOT
            mock_delegate.return_value = "done"

            mock_exec.return_value = _make_tool_result(
                content="继续委派",
                tool_calls=[
                    ToolCall(
                        id="tc",
                        name="delegate",
                        arguments={
                            "employee_name": "x",
                            "task": "y",
                        },
                    )
                ],
            )

            from crew.webhook import _execute_employee_with_tools

            _run(_execute_employee_with_tools(_make_ctx(), "boss", {"task": "loop"}))
            assert mock_exec.call_count == 2

    def test_unknown_tool_returns_error(self):
        """未知工具返回错误信息."""
        with (
            patch(_P_DISCOVER) as mock_disc,
            patch(_P_ENGINE) as mock_engine_cls,
            patch(_P_EXEC_TOOLS, new_callable=AsyncMock) as mock_exec,
            patch(_P_DETECT) as mock_detect,
        ):
            from crew.providers import Provider

            emp = _make_emp(tools=["delegate"])
            discovery = MagicMock()
            discovery.get.return_value = emp
            discovery.employees = {"test": emp}
            mock_disc.return_value = discovery

            mock_engine = MagicMock()
            mock_engine.prompt.return_value = "prompt"
            mock_engine_cls.return_value = mock_engine

            mock_detect.return_value = Provider.MOONSHOT

            mock_exec.side_effect = [
                _make_tool_result(
                    content="",
                    tool_calls=[ToolCall(id="tc1", name="bash", arguments={"command": "ls"})],
                ),
                _make_tool_result(content="好的"),
            ]

            from crew.webhook import _execute_employee_with_tools

            result = _run(_execute_employee_with_tools(_make_ctx(), "test", {"task": "test"}))
            # 第二轮返回文本
            assert result["output"] == "好的"
            # 验证 tool result 消息包含错误提示
            second_msgs = mock_exec.call_args_list[1].kwargs["messages"]
            tool_msg = next(m for m in second_msgs if m.get("role") == "tool")
            assert "权限拒绝" in tool_msg["content"] or "不可用" in tool_msg["content"]


# ── Step 2c: routing tests ──


class TestExecuteEmployeeRouting:
    @patch(_P_TOOLS_LOOP, new_callable=AsyncMock)
    @patch(_P_DISCOVER)
    def test_routes_to_tools_when_agent_tool_in_tools(self, mock_disc, mock_tools):
        emp = _make_emp(tools=["file_read", "delegate"])
        discovery = MagicMock()
        discovery.get.return_value = emp
        mock_disc.return_value = discovery

        mock_tools.return_value = {"employee": "boss", "output": "ok"}

        from crew.webhook import _execute_employee

        result = _run(_execute_employee(_make_ctx(), "boss", {"task": "test"}))
        mock_tools.assert_called_once()
        assert result["output"] == "ok"

    @patch(_P_EXEC_PROMPT, new_callable=AsyncMock)
    @patch(_P_ENGINE)
    @patch(_P_DISCOVER)
    def test_routes_to_prompt_when_no_agent_tools(self, mock_disc, mock_engine_cls, mock_exec):
        from crew.executor import ExecutionResult

        emp = _make_emp(tools=["file_read", "bash"])
        discovery = MagicMock()
        discovery.get.return_value = emp
        mock_disc.return_value = discovery
        mock_engine_cls.return_value = MagicMock()

        mock_exec.return_value = ExecutionResult(
            content="prompt result",
            model="test-model",
            input_tokens=10,
            output_tokens=20,
            stop_reason="end_turn",
        )

        from crew.webhook import _execute_employee

        result = _run(_execute_employee(_make_ctx(), "worker", {"task": "test"}))
        assert result["output"] == "prompt result"
        mock_exec.assert_called_once()

    @patch(_P_EXEC_PROMPT, new_callable=AsyncMock)
    @patch(_P_ENGINE)
    @patch(_P_DISCOVER)
    def test_routes_to_prompt_when_no_tools(self, mock_disc, mock_engine_cls, mock_exec):
        from crew.executor import ExecutionResult

        emp = _make_emp(tools=None)
        discovery = MagicMock()
        discovery.get.return_value = emp
        mock_disc.return_value = discovery
        mock_engine_cls.return_value = MagicMock()

        mock_exec.return_value = ExecutionResult(
            content="ok",
            model="m",
            input_tokens=1,
            output_tokens=1,
            stop_reason="end_turn",
        )

        from crew.webhook import _execute_employee

        result = _run(_execute_employee(_make_ctx(), "worker", {"task": "test"}))
        assert result["output"] == "ok"
