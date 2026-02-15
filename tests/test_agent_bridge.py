"""agent_bridge.py 测试 — 使用 mock LLM."""

from unittest.mock import MagicMock, patch

import pytest

from crew.models import ToolCall, ToolExecutionResult


class TestCreateCrewAgent:
    """测试 create_crew_agent 工厂函数."""

    def _make_tool_result(self, tool_name, tool_id="tc_1", args=None, content=""):
        """构造一个包含工具调用的 ToolExecutionResult."""
        return ToolExecutionResult(
            content=content,
            tool_calls=[ToolCall(id=tool_id, name=tool_name, arguments=args or {})],
            model="mock-model",
            input_tokens=100,
            output_tokens=50,
            stop_reason="tool_use",
        )

    def _make_text_result(self, content="任务完成"):
        """构造一个纯文本的 ToolExecutionResult (无工具调用)."""
        return ToolExecutionResult(
            content=content,
            tool_calls=[],
            model="mock-model",
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
        )

    @patch("crew.agent_bridge.execute_with_tools")
    @patch("crew.discovery.discover_employees")
    def test_first_call_sends_task(self, mock_discover, mock_exec):
        """首次调用应包含任务描述."""
        from crew.agent_bridge import create_crew_agent

        # mock employee
        mock_emp = MagicMock()
        mock_emp.name = "code-reviewer"
        mock_emp.tools = ["file_read", "bash"]
        mock_emp.tags = []
        mock_emp.context = []
        mock_emp.model = ""
        mock_emp.character_name = ""
        mock_emp.effective_display_name = "代码审查"
        mock_emp.description = "审查代码"
        mock_emp.output = MagicMock(format="markdown", filename="", dir=".crew/logs")
        mock_emp.args = []
        mock_emp.agent_id = None
        mock_emp.avatar_prompt = ""
        mock_emp.research_instructions = ""
        mock_emp.body = "你是代码审查员"
        mock_emp.source_path = None
        mock_emp.source_layer = "builtin"

        mock_discovery = MagicMock()
        mock_discovery.get.return_value = mock_emp
        mock_discover.return_value = mock_discovery

        mock_exec.return_value = self._make_tool_result(
            "file_read", args={"path": "src/main.py"}
        )

        agent = create_crew_agent("code-reviewer", "审查 src/main.py")
        action = agent("沙箱就绪")

        assert action["tool"] == "file_read"
        assert action["params"]["path"] == "src/main.py"

        # 验证 messages 中包含任务描述
        call_kwargs = mock_exec.call_args
        messages = call_kwargs.kwargs["messages"]
        assert "审查 src/main.py" in messages[0]["content"]

    @patch("crew.agent_bridge.execute_with_tools")
    @patch("crew.discovery.discover_employees")
    def test_submit_tool_ends_task(self, mock_discover, mock_exec):
        """submit 工具应正确返回."""
        from crew.agent_bridge import create_crew_agent

        mock_emp = MagicMock()
        mock_emp.name = "test"
        mock_emp.tools = ["file_read"]
        mock_emp.body = "test"
        mock_emp.args = []
        mock_emp.agent_id = None

        mock_discovery = MagicMock()
        mock_discovery.get.return_value = mock_emp
        mock_discover.return_value = mock_discovery

        mock_exec.return_value = self._make_tool_result(
            "submit", args={"result": "审查完成，发现 2 个问题"}
        )

        agent = create_crew_agent("test", "测试任务")
        action = agent("沙箱就绪")

        assert action["tool"] == "submit"
        assert "审查完成" in action["params"]["result"]

    @patch("crew.agent_bridge.execute_with_tools")
    @patch("crew.discovery.discover_employees")
    def test_text_only_response_submits(self, mock_discover, mock_exec):
        """纯文本响应（无工具调用）应自动转为 submit."""
        from crew.agent_bridge import create_crew_agent

        mock_emp = MagicMock()
        mock_emp.name = "test"
        mock_emp.tools = []
        mock_emp.body = "test"
        mock_emp.args = []
        mock_emp.agent_id = None

        mock_discovery = MagicMock()
        mock_discovery.get.return_value = mock_emp
        mock_discover.return_value = mock_discovery

        mock_exec.return_value = self._make_text_result("分析结果：代码质量良好")

        agent = create_crew_agent("test", "测试")
        action = agent("沙箱就绪")

        assert action["tool"] == "submit"
        assert "代码质量良好" in action["params"]["result"]

    @patch("crew.agent_bridge.execute_with_tools")
    @patch("crew.discovery.discover_employees")
    def test_bash_mapped_to_shell(self, mock_discover, mock_exec):
        """bash 工具应映射为 shell."""
        from crew.agent_bridge import create_crew_agent

        mock_emp = MagicMock()
        mock_emp.name = "test"
        mock_emp.tools = ["bash"]
        mock_emp.body = "test"
        mock_emp.args = []
        mock_emp.agent_id = None

        mock_discovery = MagicMock()
        mock_discovery.get.return_value = mock_emp
        mock_discover.return_value = mock_discovery

        mock_exec.return_value = self._make_tool_result(
            "bash", args={"command": "pytest tests/"}
        )

        agent = create_crew_agent("test", "跑测试")
        action = agent("沙箱就绪")

        assert action["tool"] == "shell"
        assert action["params"]["command"] == "pytest tests/"

    @patch("crew.agent_bridge.execute_with_tools")
    @patch("crew.discovery.discover_employees")
    def test_on_step_callback(self, mock_discover, mock_exec):
        """on_step 回调应被调用."""
        from crew.agent_bridge import create_crew_agent

        mock_emp = MagicMock()
        mock_emp.name = "test"
        mock_emp.tools = ["file_read"]
        mock_emp.body = "test"
        mock_emp.args = []
        mock_emp.agent_id = None

        mock_discovery = MagicMock()
        mock_discovery.get.return_value = mock_emp
        mock_discover.return_value = mock_discovery

        mock_exec.return_value = self._make_tool_result(
            "file_read", args={"path": "foo.py"}
        )

        steps = []
        agent = create_crew_agent("test", "测试", on_step=lambda n, t, p: steps.append((n, t, p)))
        agent("沙箱就绪")

        assert len(steps) == 1
        assert steps[0] == (1, "file_read", {"path": "foo.py"})

    @patch("crew.agent_bridge.execute_with_tools")
    @patch("crew.discovery.discover_employees")
    def test_llm_error_returns_submit(self, mock_discover, mock_exec):
        """LLM 调用失败应返回 submit + 错误信息."""
        from crew.agent_bridge import create_crew_agent

        mock_emp = MagicMock()
        mock_emp.name = "test"
        mock_emp.tools = []
        mock_emp.body = "test"
        mock_emp.args = []
        mock_emp.agent_id = None

        mock_discovery = MagicMock()
        mock_discovery.get.return_value = mock_emp
        mock_discover.return_value = mock_discovery

        mock_exec.side_effect = RuntimeError("API 超时")

        agent = create_crew_agent("test", "测试")
        action = agent("沙箱就绪")

        assert action["tool"] == "submit"
        assert "LLM 调用失败" in action["params"]["result"]

    def test_unknown_employee_raises(self):
        """未知员工应抛出 ValueError."""
        from crew.agent_bridge import create_crew_agent

        with pytest.raises(ValueError, match="未找到员工"):
            create_crew_agent("nonexistent-employee-xyz", "test")


class TestToolExecutionResultModel:
    """测试 ToolExecutionResult 数据模型."""

    def test_has_tool_calls(self):
        r = ToolExecutionResult(
            content="",
            tool_calls=[ToolCall(id="1", name="test")],
        )
        assert r.has_tool_calls is True

    def test_no_tool_calls(self):
        r = ToolExecutionResult(content="hello")
        assert r.has_tool_calls is False
