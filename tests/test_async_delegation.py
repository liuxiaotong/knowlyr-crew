"""异步委派 & 会议编排工具测试."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crew.task_registry import TaskRegistry


# ── helpers ──


def _make_ctx(tmp_path: Path):
    """构造最小 _AppContext."""
    from crew.webhook import _AppContext
    from crew.webhook_config import WebhookConfig

    registry = TaskRegistry(persist_path=tmp_path / "tasks.jsonl")
    config = WebhookConfig()
    return _AppContext(
        project_dir=Path(__file__).resolve().parent.parent,
        config=config,
        registry=registry,
    )


# ── TaskRegistry 新方法 ──


class TestRegistryFilters:
    """list_by_status / list_by_type."""

    def test_list_by_status(self):
        reg = TaskRegistry()
        r1 = reg.create(trigger="d", target_type="employee", target_name="a")
        r2 = reg.create(trigger="d", target_type="employee", target_name="b")
        reg.update(r1.task_id, "running")
        reg.update(r2.task_id, "completed", result={"ok": True})

        assert len(reg.list_by_status("running")) == 1
        assert reg.list_by_status("running")[0].task_id == r1.task_id
        assert len(reg.list_by_status("completed")) == 1
        assert len(reg.list_by_status("failed")) == 0

    def test_list_by_type(self):
        reg = TaskRegistry()
        reg.create(trigger="d", target_type="employee", target_name="a")
        reg.create(trigger="d", target_type="meeting", target_name="b")
        reg.create(trigger="d", target_type="employee", target_name="c")

        assert len(reg.list_by_type("employee")) == 2
        assert len(reg.list_by_type("meeting")) == 1
        assert len(reg.list_by_type("pipeline")) == 0

    def test_list_by_status_limit(self):
        reg = TaskRegistry()
        for i in range(5):
            reg.create(trigger="d", target_type="employee", target_name=f"e-{i}")
        assert len(reg.list_by_status("pending", limit=3)) == 3


# ── delegate_async ──


class TestDelegateAsync:
    """_tool_delegate_async."""

    @pytest.mark.asyncio
    async def test_returns_task_id(self, tmp_path):
        from crew.webhook import _tool_delegate_async

        ctx = _make_ctx(tmp_path)

        with patch("crew.webhook._execute_task", new_callable=AsyncMock):
            result = await _tool_delegate_async(
                {"employee_name": "code-reviewer", "task": "审查 PR"},
                ctx=ctx,
            )

        assert "任务 ID:" in result
        assert "code-reviewer" in result

    @pytest.mark.asyncio
    async def test_creates_task_record(self, tmp_path):
        from crew.webhook import _tool_delegate_async

        ctx = _make_ctx(tmp_path)

        with patch("crew.webhook._execute_task", new_callable=AsyncMock):
            result = await _tool_delegate_async(
                {"employee_name": "doc-writer", "task": "写文档"},
                ctx=ctx,
            )

        tasks = ctx.registry.list_recent(n=10)
        assert len(tasks) == 1
        assert tasks[0].target_type == "employee"
        assert tasks[0].target_name == "doc-writer"
        assert tasks[0].trigger == "delegate_async"

    @pytest.mark.asyncio
    async def test_unknown_employee(self, tmp_path):
        from crew.webhook import _tool_delegate_async

        ctx = _make_ctx(tmp_path)
        result = await _tool_delegate_async(
            {"employee_name": "no-such-emp", "task": "test"},
            ctx=ctx,
        )
        assert "错误" in result
        assert "no-such-emp" in result

    @pytest.mark.asyncio
    async def test_no_ctx(self):
        from crew.webhook import _tool_delegate_async

        result = await _tool_delegate_async({"employee_name": "x", "task": "t"}, ctx=None)
        assert "错误" in result


# ── check_task ──


class TestCheckTask:
    """_tool_check_task."""

    @pytest.mark.asyncio
    async def test_pending(self, tmp_path):
        from crew.webhook import _tool_check_task

        ctx = _make_ctx(tmp_path)
        record = ctx.registry.create(
            trigger="delegate_async", target_type="employee", target_name="code-reviewer",
        )
        result = await _tool_check_task({"task_id": record.task_id}, ctx=ctx)
        assert "pending" in result
        assert "code-reviewer" in result

    @pytest.mark.asyncio
    async def test_completed(self, tmp_path):
        from crew.webhook import _tool_check_task

        ctx = _make_ctx(tmp_path)
        record = ctx.registry.create(
            trigger="delegate_async", target_type="employee", target_name="test-engineer",
        )
        ctx.registry.update(record.task_id, "completed", result={"content": "all tests pass"})
        result = await _tool_check_task({"task_id": record.task_id}, ctx=ctx)
        assert "completed" in result
        assert "all tests pass" in result

    @pytest.mark.asyncio
    async def test_meeting_result(self, tmp_path):
        from crew.webhook import _tool_check_task

        ctx = _make_ctx(tmp_path)
        record = ctx.registry.create(
            trigger="organize_meeting", target_type="meeting", target_name="test-meeting",
        )
        ctx.registry.update(
            record.task_id, "completed",
            result={"synthesis": "决议：采用方案 A", "rounds": []},
        )
        result = await _tool_check_task({"task_id": record.task_id}, ctx=ctx)
        assert "会议综合结论" in result
        assert "方案 A" in result

    @pytest.mark.asyncio
    async def test_failed(self, tmp_path):
        from crew.webhook import _tool_check_task

        ctx = _make_ctx(tmp_path)
        record = ctx.registry.create(
            trigger="delegate_async", target_type="employee", target_name="x",
        )
        ctx.registry.update(record.task_id, "failed", error="timeout")
        result = await _tool_check_task({"task_id": record.task_id}, ctx=ctx)
        assert "failed" in result
        assert "timeout" in result

    @pytest.mark.asyncio
    async def test_not_found(self, tmp_path):
        from crew.webhook import _tool_check_task

        ctx = _make_ctx(tmp_path)
        result = await _tool_check_task({"task_id": "no-such-id"}, ctx=ctx)
        assert "未找到" in result


# ── list_tasks ──


class TestListTasks:
    """_tool_list_tasks."""

    @pytest.mark.asyncio
    async def test_empty(self, tmp_path):
        from crew.webhook import _tool_list_tasks

        ctx = _make_ctx(tmp_path)
        result = await _tool_list_tasks({}, ctx=ctx)
        assert "暂无" in result

    @pytest.mark.asyncio
    async def test_with_tasks(self, tmp_path):
        from crew.webhook import _tool_list_tasks

        ctx = _make_ctx(tmp_path)
        ctx.registry.create(trigger="d", target_type="employee", target_name="a")
        ctx.registry.create(trigger="d", target_type="meeting", target_name="b")

        result = await _tool_list_tasks({}, ctx=ctx)
        assert "共 2 条" in result

    @pytest.mark.asyncio
    async def test_filter_by_status(self, tmp_path):
        from crew.webhook import _tool_list_tasks

        ctx = _make_ctx(tmp_path)
        r = ctx.registry.create(trigger="d", target_type="employee", target_name="a")
        ctx.registry.create(trigger="d", target_type="employee", target_name="b")
        ctx.registry.update(r.task_id, "completed", result={"ok": True})

        result = await _tool_list_tasks({"status": "pending"}, ctx=ctx)
        assert "共 1 条" in result

    @pytest.mark.asyncio
    async def test_filter_by_type(self, tmp_path):
        from crew.webhook import _tool_list_tasks

        ctx = _make_ctx(tmp_path)
        ctx.registry.create(trigger="d", target_type="employee", target_name="a")
        ctx.registry.create(trigger="d", target_type="meeting", target_name="b")

        result = await _tool_list_tasks({"type": "meeting"}, ctx=ctx)
        assert "共 1 条" in result


# ── organize_meeting ──


class TestOrganizeMeeting:
    """_tool_organize_meeting."""

    @pytest.mark.asyncio
    async def test_returns_meeting_id(self, tmp_path):
        from crew.webhook import _tool_organize_meeting

        ctx = _make_ctx(tmp_path)

        with patch("crew.webhook._execute_task", new_callable=AsyncMock):
            result = await _tool_organize_meeting(
                {
                    "employees": ["code-reviewer", "test-engineer"],
                    "topic": "测试策略",
                },
                ctx=ctx,
            )

        assert "会议 ID:" in result
        assert "code-reviewer" in result
        assert "test-engineer" in result
        assert "测试策略" in result

    @pytest.mark.asyncio
    async def test_creates_meeting_record(self, tmp_path):
        from crew.webhook import _tool_organize_meeting

        ctx = _make_ctx(tmp_path)

        with patch("crew.webhook._execute_task", new_callable=AsyncMock):
            await _tool_organize_meeting(
                {
                    "employees": ["code-reviewer", "doc-writer"],
                    "topic": "文档规范",
                    "rounds": 3,
                },
                ctx=ctx,
            )

        tasks = ctx.registry.list_by_type("meeting")
        assert len(tasks) == 1
        assert tasks[0].args["topic"] == "文档规范"
        assert tasks[0].args["rounds"] == "3"
        assert tasks[0].args["employees"] == "code-reviewer,doc-writer"

    @pytest.mark.asyncio
    async def test_missing_employees(self, tmp_path):
        from crew.webhook import _tool_organize_meeting

        ctx = _make_ctx(tmp_path)
        result = await _tool_organize_meeting(
            {"employees": ["no-such-person"], "topic": "test"},
            ctx=ctx,
        )
        assert "错误" in result
        assert "no-such-person" in result

    @pytest.mark.asyncio
    async def test_empty_params(self, tmp_path):
        from crew.webhook import _tool_organize_meeting

        ctx = _make_ctx(tmp_path)
        result = await _tool_organize_meeting({"employees": [], "topic": ""}, ctx=ctx)
        assert "错误" in result

    @pytest.mark.asyncio
    async def test_string_employees(self, tmp_path):
        """employees 可以是逗号分隔字符串."""
        from crew.webhook import _tool_organize_meeting

        ctx = _make_ctx(tmp_path)

        with patch("crew.webhook._execute_task", new_callable=AsyncMock):
            result = await _tool_organize_meeting(
                {"employees": "code-reviewer,test-engineer", "topic": "测试"},
                ctx=ctx,
            )

        assert "会议 ID:" in result


# ── check_meeting ──


class TestCheckMeeting:
    """_tool_check_meeting 是 check_task 别名."""

    @pytest.mark.asyncio
    async def test_delegates_to_check_task(self, tmp_path):
        from crew.webhook import _tool_check_meeting

        ctx = _make_ctx(tmp_path)
        record = ctx.registry.create(
            trigger="organize_meeting", target_type="meeting", target_name="test",
        )
        result = await _tool_check_meeting({"task_id": record.task_id}, ctx=ctx)
        assert record.task_id in result


# ── _execute_meeting ──


class TestExecuteMeeting:
    """会议执行逻辑."""

    @pytest.mark.asyncio
    async def test_runs_discussion(self, tmp_path):
        from crew.webhook import _execute_meeting

        ctx = _make_ctx(tmp_path)

        mock_result = MagicMock()
        mock_result.content = "模拟回答"

        mock_plan = MagicMock()
        mock_plan.rounds = [
            MagicMock(
                round_number=1,
                name="开场",
                participant_prompts=[
                    MagicMock(prompt="prompt1 {previous_rounds}", employee_name="code-reviewer"),
                    MagicMock(prompt="prompt2 {previous_rounds}", employee_name="test-engineer"),
                ],
            ),
        ]
        mock_plan.synthesis_prompt = "综合 {previous_rounds}"

        with (
            patch("crew.discussion.create_adhoc_discussion") as mock_create,
            patch("crew.discussion.render_discussion_plan", return_value=mock_plan),
            patch("crew.executor.aexecute_prompt", new_callable=AsyncMock, return_value=mock_result),
        ):
            result = await _execute_meeting(
                ctx, task_id="test-001",
                employees=["code-reviewer", "test-engineer"],
                topic="测试策略",
            )

        assert "rounds" in result
        assert "synthesis" in result
        assert result["synthesis"] == "模拟回答"
        assert len(result["rounds"]) == 1
        assert len(result["rounds"][0]["outputs"]) == 2

    @pytest.mark.asyncio
    async def test_handles_execution_error(self, tmp_path):
        """参会者执行失败不影响整体."""
        from crew.webhook import _execute_meeting

        ctx = _make_ctx(tmp_path)

        success_result = MagicMock()
        success_result.content = "正常回答"

        mock_plan = MagicMock()
        mock_plan.rounds = [
            MagicMock(
                round_number=1,
                name="开场",
                participant_prompts=[
                    MagicMock(prompt="p1 {previous_rounds}", employee_name="emp-a"),
                    MagicMock(prompt="p2 {previous_rounds}", employee_name="emp-b"),
                ],
            ),
        ]
        mock_plan.synthesis_prompt = "综合 {previous_rounds}"

        call_count = 0

        async def _mock_execute(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("模拟失败")
            return success_result

        with (
            patch("crew.discussion.create_adhoc_discussion"),
            patch("crew.discussion.render_discussion_plan", return_value=mock_plan),
            patch("crew.executor.aexecute_prompt", side_effect=_mock_execute),
        ):
            result = await _execute_meeting(
                ctx, task_id="test-002",
                employees=["emp-a", "emp-b"],
                topic="test",
            )

        outputs = result["rounds"][0]["outputs"]
        assert "执行失败" in outputs[0]["content"]
        assert outputs[1]["content"] == "正常回答"


# ── _execute_task meeting 分支 ──


class TestExecuteTaskMeeting:
    """_execute_task 的 meeting 分支."""

    @pytest.mark.asyncio
    async def test_meeting_branch(self, tmp_path):
        from crew.webhook import _execute_task

        ctx = _make_ctx(tmp_path)
        record = ctx.registry.create(
            trigger="organize_meeting",
            target_type="meeting",
            target_name="test-meeting",
            args={
                "employees": "code-reviewer,test-engineer",
                "topic": "测试",
                "goal": "",
                "rounds": "2",
            },
        )

        mock_meeting_result = {"rounds": [], "synthesis": "done"}
        with patch("crew.webhook._execute_meeting", new_callable=AsyncMock, return_value=mock_meeting_result):
            await _execute_task(ctx, record.task_id)

        updated = ctx.registry.get(record.task_id)
        assert updated.status == "completed"
        assert updated.result == mock_meeting_result


# ── tool_schema 注册 ──


class TestToolSchemaRegistration:
    """验证新工具已注册."""

    def test_schemas_exist(self):
        from crew.tool_schema import _TOOL_SCHEMAS

        for name in ("delegate_async", "check_task", "list_tasks", "organize_meeting", "check_meeting"):
            assert name in _TOOL_SCHEMAS, f"{name} not in _TOOL_SCHEMAS"

    def test_in_agent_tools(self):
        from crew.tool_schema import AGENT_TOOLS

        for name in ("delegate_async", "check_task", "list_tasks", "organize_meeting", "check_meeting"):
            assert name in AGENT_TOOLS, f"{name} not in AGENT_TOOLS"

    def test_handlers_registered(self):
        from crew.webhook import _TOOL_HANDLERS

        for name in ("delegate_async", "check_task", "list_tasks", "organize_meeting", "check_meeting"):
            assert name in _TOOL_HANDLERS, f"{name} not in _TOOL_HANDLERS"
