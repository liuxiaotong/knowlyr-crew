"""姜墨言 8 项能力增强 — 10 个新工具测试."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crew.task_registry import TaskRegistry


# ── helpers ──


def _make_ctx(tmp_path: Path, *, with_scheduler: bool = False):
    """构造最小 _AppContext."""
    from crew.webhook import _AppContext
    from crew.webhook_config import WebhookConfig

    registry = TaskRegistry(persist_path=tmp_path / "tasks.jsonl")
    config = WebhookConfig()

    ctx = _AppContext(
        project_dir=Path(__file__).resolve().parent.parent,
        config=config,
        registry=registry,
    )
    if with_scheduler:
        ctx.scheduler = MagicMock()
    return ctx


# ── 1. run_pipeline ──


class TestRunPipeline:
    """_tool_run_pipeline."""

    @pytest.mark.asyncio
    async def test_returns_task_id(self, tmp_path):
        from crew.webhook import _tool_run_pipeline

        ctx = _make_ctx(tmp_path)
        mock_pipelines = {"security-audit": MagicMock()}

        with (
            patch("crew.webhook._execute_task", new_callable=AsyncMock),
            patch("crew.pipeline.discover_pipelines", return_value=mock_pipelines),
        ):
            result = await _tool_run_pipeline(
                {"name": "security-audit"},
                ctx=ctx,
            )
        assert "任务 ID" in result
        assert "security-audit" in result

    @pytest.mark.asyncio
    async def test_unknown_pipeline(self, tmp_path):
        from crew.webhook import _tool_run_pipeline

        ctx = _make_ctx(tmp_path)
        with patch("crew.pipeline.discover_pipelines", return_value={}):
            result = await _tool_run_pipeline(
                {"name": "nonexistent"},
                ctx=ctx,
            )
        assert "未找到" in result or "错误" in result

    @pytest.mark.asyncio
    async def test_no_context(self):
        from crew.webhook import _tool_run_pipeline

        result = await _tool_run_pipeline({"name": "x"}, ctx=None)
        assert "错误" in result


# ── 2. delegate_chain ──


class TestDelegateChain:
    """_tool_delegate_chain."""

    @pytest.mark.asyncio
    async def test_returns_task_id(self, tmp_path):
        from crew.webhook import _tool_delegate_chain

        ctx = _make_ctx(tmp_path)
        steps = [
            {"employee_name": "code-reviewer", "task": "审查代码"},
            {"employee_name": "doc-writer", "task": "根据审查结果写文档: {prev}"},
        ]

        mock_discovery = {"code-reviewer": MagicMock(), "doc-writer": MagicMock()}
        with (
            patch("crew.webhook._execute_task", new_callable=AsyncMock),
            patch("crew.discovery.discover_employees", return_value=mock_discovery),
        ):
            result = await _tool_delegate_chain(
                {"steps": steps},
                ctx=ctx,
            )
        assert "任务 ID" in result
        assert "code-reviewer" in result
        assert "doc-writer" in result

    @pytest.mark.asyncio
    async def test_missing_employee(self, tmp_path):
        from crew.webhook import _tool_delegate_chain

        ctx = _make_ctx(tmp_path)
        steps = [{"employee_name": "nonexistent", "task": "做事"}]

        with patch("crew.discovery.discover_employees", return_value={}):
            result = await _tool_delegate_chain({"steps": steps}, ctx=ctx)
        assert "错误" in result

    @pytest.mark.asyncio
    async def test_empty_steps(self, tmp_path):
        from crew.webhook import _tool_delegate_chain

        ctx = _make_ctx(tmp_path)
        result = await _tool_delegate_chain({"steps": []}, ctx=ctx)
        assert "错误" in result


class TestExecuteChain:
    """_execute_chain."""

    @pytest.mark.asyncio
    async def test_sequential_execution(self, tmp_path):
        from crew.webhook import _execute_chain

        ctx = _make_ctx(tmp_path)
        steps = [
            {"employee_name": "code-reviewer", "task": "审查代码"},
            {"employee_name": "doc-writer", "task": "写文档: {prev}"},
        ]

        mock_emp = MagicMock()
        mock_emp.model = None
        mock_result = MagicMock()
        mock_result.content = "审查通过"

        with (
            patch("crew.discovery.discover_employees", return_value={
                "code-reviewer": mock_emp,
                "doc-writer": mock_emp,
            }),
            patch("crew.engine.CrewEngine") as MockEngine,
            patch("crew.executor.aexecute_prompt", new_callable=AsyncMock, return_value=mock_result),
        ):
            MockEngine.return_value.prompt.return_value = "system prompt"
            result = await _execute_chain(ctx, "task-123", steps)

        assert "steps" in result
        assert len(result["steps"]) == 2
        assert result["final_output"] == "审查通过"

    @pytest.mark.asyncio
    async def test_prev_substitution(self, tmp_path):
        from crew.webhook import _execute_chain

        ctx = _make_ctx(tmp_path)
        steps = [
            {"employee_name": "code-reviewer", "task": "审查"},
            {"employee_name": "doc-writer", "task": "用这个: {prev}"},
        ]

        mock_emp = MagicMock()
        mock_emp.model = None
        results = iter([MagicMock(content="第一步结果"), MagicMock(content="第二步结果")])

        with (
            patch("crew.discovery.discover_employees", return_value={
                "code-reviewer": mock_emp,
                "doc-writer": mock_emp,
            }),
            patch("crew.engine.CrewEngine") as MockEngine,
            patch("crew.executor.aexecute_prompt", new_callable=AsyncMock, side_effect=lambda **kw: next(results)),
        ):
            MockEngine.return_value.prompt.return_value = "system prompt"
            result = await _execute_chain(ctx, "task-456", steps)

        assert result["final_output"] == "第二步结果"


# ── 3. schedule_task / list_schedules / cancel_schedule ──


class TestScheduleTask:
    """_tool_schedule_task."""

    @pytest.mark.asyncio
    async def test_creates_schedule(self, tmp_path):
        from crew.webhook import _tool_schedule_task

        ctx = _make_ctx(tmp_path, with_scheduler=True)
        ctx.scheduler.schedules = []
        ctx.scheduler.add_schedule = AsyncMock()

        with patch("croniter.croniter") as mock_croniter:
            mock_croniter.is_valid.return_value = True
            result = await _tool_schedule_task(
                {"name": "daily-briefing", "cron": "0 9 * * *",
                 "employee_name": "ceo-assistant", "task": "发简报"},
                ctx=ctx,
            )
        assert "daily-briefing" in result
        assert "已创建" in result
        ctx.scheduler.add_schedule.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invalid_cron(self, tmp_path):
        from crew.webhook import _tool_schedule_task

        ctx = _make_ctx(tmp_path, with_scheduler=True)
        ctx.scheduler.schedules = []

        with patch("croniter.croniter") as mock_croniter:
            mock_croniter.is_valid.return_value = False
            result = await _tool_schedule_task(
                {"name": "bad", "cron": "invalid", "employee_name": "a", "task": "b"},
                ctx=ctx,
            )
        assert "无效" in result or "错误" in result

    @pytest.mark.asyncio
    async def test_duplicate_name(self, tmp_path):
        from crew.webhook import _tool_schedule_task

        ctx = _make_ctx(tmp_path, with_scheduler=True)
        mock_schedule = MagicMock()
        mock_schedule.name = "existing"
        ctx.scheduler.schedules = [mock_schedule]

        with patch("croniter.croniter") as mock_croniter:
            mock_croniter.is_valid.return_value = True
            result = await _tool_schedule_task(
                {"name": "existing", "cron": "0 9 * * *",
                 "employee_name": "a", "task": "b"},
                ctx=ctx,
            )
        assert "已存在" in result

    @pytest.mark.asyncio
    async def test_missing_params(self, tmp_path):
        from crew.webhook import _tool_schedule_task

        ctx = _make_ctx(tmp_path, with_scheduler=True)
        result = await _tool_schedule_task({"name": "only-name"}, ctx=ctx)
        assert "错误" in result


class TestListSchedules:
    """_tool_list_schedules."""

    @pytest.mark.asyncio
    async def test_lists_schedules(self, tmp_path):
        from crew.webhook import _tool_list_schedules

        ctx = _make_ctx(tmp_path, with_scheduler=True)
        ctx.scheduler.get_next_runs.return_value = [
            {"name": "daily", "cron": "0 9 * * *", "target_type": "employee",
             "target_name": "ceo-assistant", "next_run": "2026-02-17T09:00:00",
             "missed_count": 0},
        ]
        result = await _tool_list_schedules({}, ctx=ctx)
        assert "daily" in result
        assert "1" in result  # 共 1 个

    @pytest.mark.asyncio
    async def test_empty(self, tmp_path):
        from crew.webhook import _tool_list_schedules

        ctx = _make_ctx(tmp_path, with_scheduler=True)
        ctx.scheduler.get_next_runs.return_value = []
        result = await _tool_list_schedules({}, ctx=ctx)
        assert "暂无" in result


class TestCancelSchedule:
    """_tool_cancel_schedule."""

    @pytest.mark.asyncio
    async def test_cancels(self, tmp_path):
        from crew.webhook import _tool_cancel_schedule

        ctx = _make_ctx(tmp_path, with_scheduler=True)
        ctx.scheduler.remove_schedule = AsyncMock(return_value=True)
        result = await _tool_cancel_schedule({"name": "daily"}, ctx=ctx)
        assert "已取消" in result

    @pytest.mark.asyncio
    async def test_not_found(self, tmp_path):
        from crew.webhook import _tool_cancel_schedule

        ctx = _make_ctx(tmp_path, with_scheduler=True)
        ctx.scheduler.remove_schedule = AsyncMock(return_value=False)
        result = await _tool_cancel_schedule({"name": "nonexistent"}, ctx=ctx)
        assert "未找到" in result


# ── 4. agent_file_read / agent_file_grep ──


class TestAgentFileRead:
    """_tool_agent_file_read."""

    @pytest.mark.asyncio
    async def test_reads_file(self, tmp_path):
        from crew.webhook import _tool_agent_file_read

        ctx = _make_ctx(tmp_path)
        ctx.project_dir = tmp_path

        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3\n")

        result = await _tool_agent_file_read({"path": "test.py"}, ctx=ctx)
        assert "line1" in result
        assert "test.py" in result
        assert "3 行" in result

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, tmp_path):
        from crew.webhook import _tool_agent_file_read

        ctx = _make_ctx(tmp_path)
        ctx.project_dir = tmp_path

        result = await _tool_agent_file_read(
            {"path": "../../../etc/passwd"},
            ctx=ctx,
        )
        assert "错误" in result

    @pytest.mark.asyncio
    async def test_nonexistent_file(self, tmp_path):
        from crew.webhook import _tool_agent_file_read

        ctx = _make_ctx(tmp_path)
        ctx.project_dir = tmp_path

        result = await _tool_agent_file_read({"path": "nope.py"}, ctx=ctx)
        assert "不存在" in result

    @pytest.mark.asyncio
    async def test_line_range(self, tmp_path):
        from crew.webhook import _tool_agent_file_read

        ctx = _make_ctx(tmp_path)
        ctx.project_dir = tmp_path

        test_file = tmp_path / "big.py"
        test_file.write_text("\n".join(f"line{i}" for i in range(100)))

        result = await _tool_agent_file_read(
            {"path": "big.py", "start_line": 10, "end_line": 15},
            ctx=ctx,
        )
        assert "line9" in result  # 0-indexed in content, but line 10 is "line9"
        assert "显示 10-15" in result


class TestAgentFileGrep:
    """_tool_agent_file_grep."""

    @pytest.mark.asyncio
    async def test_finds_matches(self, tmp_path):
        from crew.webhook import _tool_agent_file_grep

        ctx = _make_ctx(tmp_path)
        ctx.project_dir = tmp_path

        test_file = tmp_path / "code.py"
        test_file.write_text("def hello():\n    return 'world'\n")

        result = await _tool_agent_file_grep({"pattern": "hello"}, ctx=ctx)
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_no_matches(self, tmp_path):
        from crew.webhook import _tool_agent_file_grep

        ctx = _make_ctx(tmp_path)
        ctx.project_dir = tmp_path

        test_file = tmp_path / "code.py"
        test_file.write_text("def hello():\n    pass\n")

        result = await _tool_agent_file_grep({"pattern": "nonexistent_xyz"}, ctx=ctx)
        assert "未找到" in result

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, tmp_path):
        from crew.webhook import _tool_agent_file_grep

        ctx = _make_ctx(tmp_path)
        ctx.project_dir = tmp_path

        result = await _tool_agent_file_grep(
            {"pattern": "root", "path": "../../../etc"},
            ctx=ctx,
        )
        assert "错误" in result


# ── 5. query_data ──


class TestQueryData:
    """_tool_query_data."""

    @pytest.mark.asyncio
    async def test_success(self):
        from crew.webhook import _tool_query_data

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"users": 100}'

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await _tool_query_data(
                {"metric": "users", "period": "week"},
            )
        assert "100" in result

    @pytest.mark.asyncio
    async def test_api_error(self):
        from crew.webhook import _tool_query_data

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await _tool_query_data({"metric": "users"})
        assert "失败" in result


# ── 6. find_free_time ──


class TestFindFreeTime:
    """_tool_find_free_time."""

    @pytest.mark.asyncio
    async def test_no_feishu(self, tmp_path):
        from crew.webhook import _tool_find_free_time

        ctx = _make_ctx(tmp_path)
        ctx.feishu_token_mgr = None
        result = await _tool_find_free_time(
            {"user_ids": ["ou_xxx"]}, ctx=ctx,
        )
        assert "错误" in result or "未配置" in result

    @pytest.mark.asyncio
    async def test_no_user_ids(self, tmp_path):
        from crew.webhook import _tool_find_free_time

        ctx = _make_ctx(tmp_path)
        ctx.feishu_token_mgr = MagicMock()
        result = await _tool_find_free_time({"user_ids": []}, ctx=ctx)
        assert "错误" in result

    @pytest.mark.asyncio
    async def test_freebusy_error(self, tmp_path):
        from crew.webhook import _tool_find_free_time

        ctx = _make_ctx(tmp_path)
        ctx.feishu_token_mgr = MagicMock()

        with patch("crew.feishu.get_freebusy", new_callable=AsyncMock,
                    return_value={"error": "token expired"}):
            result = await _tool_find_free_time(
                {"user_ids": "ou_xxx,ou_yyy", "days": 3},
                ctx=ctx,
            )
        assert "失败" in result


# ── 7. execute_task chain branch ──


class TestExecuteTaskChain:
    """_execute_task with target_type=chain."""

    @pytest.mark.asyncio
    async def test_chain_branch(self, tmp_path):
        from crew.webhook import _execute_task

        ctx = _make_ctx(tmp_path)
        steps = [{"employee_name": "code-reviewer", "task": "do stuff"}]
        record = ctx.registry.create(
            trigger="chain",
            target_type="chain",
            target_name="code-reviewer",
            args={"steps_json": json.dumps(steps)},
        )

        mock_result = {"steps": [{"employee": "code-reviewer", "content": "done"}], "final_output": "done"}

        with patch("crew.webhook._execute_chain", new_callable=AsyncMock, return_value=mock_result):
            await _execute_task(ctx, record.task_id)

        updated = ctx.registry.get(record.task_id)
        assert updated.status == "completed"


# ── 8. CronScheduler add/remove ──


class TestCronSchedulerDynamic:
    """CronScheduler.add_schedule / remove_schedule."""

    @pytest.mark.asyncio
    async def test_add_schedule(self):
        from crew.cron_config import CronConfig, CronSchedule
        from crew.cron_scheduler import CronScheduler

        config = CronConfig(schedules=[])
        scheduler = CronScheduler(config=config, execute_fn=AsyncMock())
        scheduler._running = True

        new_sched = CronSchedule(
            name="test-job",
            cron="0 9 * * *",
            target_type="employee",
            target_name="ceo-assistant",
            args={"task": "hello"},
        )
        await scheduler.add_schedule(new_sched)

        assert len(scheduler.schedules) == 1
        assert scheduler.schedules[0].name == "test-job"
        assert len(scheduler._tasks) == 1

        # cleanup
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_remove_schedule(self):
        from crew.cron_config import CronConfig, CronSchedule
        from crew.cron_scheduler import CronScheduler

        sched = CronSchedule(
            name="removable",
            cron="0 9 * * *",
            target_type="employee",
            target_name="ceo-assistant",
            args={"task": "hello"},
        )
        config = CronConfig(schedules=[sched])
        scheduler = CronScheduler(config=config, execute_fn=AsyncMock())
        scheduler._running = True

        # Simulate a task
        mock_task = MagicMock()
        mock_task.get_name.return_value = "cron-removable"
        mock_task.cancel = MagicMock()
        scheduler._tasks.append(mock_task)

        removed = await scheduler.remove_schedule("removable")
        assert removed is True
        assert len(scheduler.schedules) == 0
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self):
        from crew.cron_config import CronConfig
        from crew.cron_scheduler import CronScheduler

        config = CronConfig(schedules=[])
        scheduler = CronScheduler(config=config, execute_fn=AsyncMock())

        removed = await scheduler.remove_schedule("ghost")
        assert removed is False


# ── 9. feishu get_freebusy ──


class TestFeishuFreebusy:
    """feishu.get_freebusy."""

    @pytest.mark.asyncio
    async def test_success(self):
        from crew.feishu import FeishuTokenManager, get_freebusy

        token_mgr = MagicMock(spec=FeishuTokenManager)
        token_mgr.get_token = AsyncMock(return_value="test-token")

        mock_data = {
            "code": 0,
            "data": {
                "freebusy_list": [
                    {"user_id": "ou_xxx", "busy": [{"start_time": "1000", "end_time": "2000"}]},
                ],
            },
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_data

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await get_freebusy(token_mgr, ["ou_xxx"], 1000, 2000)

        assert "freebusy_list" in result

    @pytest.mark.asyncio
    async def test_api_error(self):
        from crew.feishu import FeishuTokenManager, get_freebusy

        token_mgr = MagicMock(spec=FeishuTokenManager)
        token_mgr.get_token = AsyncMock(return_value="test-token")

        mock_data = {"code": 40003, "msg": "invalid token"}
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_data
        mock_resp.status_code = 200

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await get_freebusy(token_mgr, ["ou_xxx"], 1000, 2000)

        assert "error" in result


# ── 10. Schema registration ──


class TestSchemaRegistration:
    """新工具 schema 注册."""

    def test_all_new_tools_in_agent_tools(self):
        from crew.tool_schema import AGENT_TOOLS

        new_tools = [
            "run_pipeline", "delegate_chain",
            "schedule_task", "list_schedules", "cancel_schedule",
            "agent_file_read", "agent_file_grep",
            "query_data", "find_free_time",
        ]
        for tool in new_tools:
            assert tool in AGENT_TOOLS, f"{tool} not in AGENT_TOOLS"

    def test_all_new_tools_have_schemas(self):
        from crew.tool_schema import _TOOL_SCHEMAS

        new_tools = [
            "run_pipeline", "delegate_chain",
            "schedule_task", "list_schedules", "cancel_schedule",
            "agent_file_read", "agent_file_grep",
            "query_data", "find_free_time",
        ]
        for tool in new_tools:
            assert tool in _TOOL_SCHEMAS, f"{tool} not in _TOOL_SCHEMAS"
            schema = _TOOL_SCHEMAS[tool]
            assert "name" in schema
            assert "description" in schema
            assert "input_schema" in schema

    def test_all_new_tools_in_handlers(self):
        from crew.webhook import _TOOL_HANDLERS

        new_tools = [
            "run_pipeline", "delegate_chain",
            "schedule_task", "list_schedules", "cancel_schedule",
            "agent_file_read", "agent_file_grep",
            "query_data", "find_free_time",
        ]
        for tool in new_tools:
            assert tool in _TOOL_HANDLERS, f"{tool} not in _TOOL_HANDLERS"
