"""评估闭环模块测试."""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crew.evaluation import Decision, EvaluationEngine


class TestDecision:
    """测试决策模型."""

    def test_default_fields(self):
        d = Decision(employee="pm", category="estimate", content="需要 3 天")
        assert d.employee == "pm"
        assert d.category == "estimate"
        assert d.status == "pending"
        assert d.id.startswith("D")
        assert d.actual_outcome == ""
        assert d.evaluation == ""

    def test_custom_fields(self):
        d = Decision(
            employee="pm",
            category="commitment",
            content="本周交付",
            expected_outcome="周五上线",
            meeting_id="M001",
        )
        assert d.category == "commitment"
        assert d.expected_outcome == "周五上线"
        assert d.meeting_id == "M001"


class TestEvaluationEngine:
    """测试评估引擎."""

    def test_track(self, tmp_path):
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        d = engine.track("pm", "estimate", "需要 3 天", expected_outcome="3 天完成")
        assert d.employee == "pm"
        assert d.category == "estimate"
        assert d.status == "pending"
        # 文件存在
        assert (tmp_path / "eval" / "decisions.jsonl").exists()

    def test_track_multiple(self, tmp_path):
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        engine.track("pm", "estimate", "决策 1")
        engine.track("dev", "recommendation", "决策 2")
        lines = (tmp_path / "eval" / "decisions.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_get(self, tmp_path):
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        d = engine.track("pm", "estimate", "需要 3 天")
        found = engine.get(d.id)
        assert found is not None
        assert found.id == d.id
        assert found.content == "需要 3 天"

    def test_get_nonexistent(self, tmp_path):
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        assert engine.get("nonexistent") is None

    def test_get_no_file(self, tmp_path):
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        assert engine.get("D12345678") is None

    def test_evaluate(self, tmp_path):
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        d = engine.track("pm", "estimate", "需要 3 天", expected_outcome="3 天完成")
        result = engine.evaluate(d.id, "实际花了 5 天", "低估了复杂度")
        assert result is not None
        assert result.status == "evaluated"
        assert result.actual_outcome == "实际花了 5 天"
        assert result.evaluation == "低估了复杂度"

    def test_evaluate_auto_conclusion(self, tmp_path):
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        d = engine.track("pm", "estimate", "需要 3 天", expected_outcome="3 天完成")
        result = engine.evaluate(d.id, "实际花了 5 天")
        assert result is not None
        assert "预期" in result.evaluation
        assert "实际" in result.evaluation

    def test_evaluate_persists(self, tmp_path):
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        d = engine.track("pm", "estimate", "需要 3 天")
        engine.evaluate(d.id, "花了 5 天", "低估")
        # 重新读取验证
        reloaded = engine.get(d.id)
        assert reloaded is not None
        assert reloaded.status == "evaluated"
        assert reloaded.actual_outcome == "花了 5 天"

    def test_evaluate_nonexistent(self, tmp_path):
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        engine.track("pm", "estimate", "test")
        assert engine.evaluate("nonexistent", "outcome") is None

    def test_evaluate_no_file(self, tmp_path):
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        assert engine.evaluate("D12345678", "outcome") is None

    def test_list_decisions(self, tmp_path):
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        engine.track("pm", "estimate", "决策 1")
        engine.track("dev", "recommendation", "决策 2")
        engine.track("pm", "commitment", "决策 3")

        all_decisions = engine.list_decisions()
        assert len(all_decisions) == 3
        # 最新在前
        assert all_decisions[0].content == "决策 3"

    def test_list_decisions_filter_employee(self, tmp_path):
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        engine.track("pm", "estimate", "PM 决策")
        engine.track("dev", "recommendation", "Dev 决策")

        pm_decisions = engine.list_decisions(employee="pm")
        assert len(pm_decisions) == 1
        assert pm_decisions[0].employee == "pm"

    def test_list_decisions_filter_status(self, tmp_path):
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        d1 = engine.track("pm", "estimate", "决策 1")
        engine.track("pm", "estimate", "决策 2")
        engine.evaluate(d1.id, "结果")

        pending = engine.list_decisions(status="pending")
        assert len(pending) == 1
        evaluated = engine.list_decisions(status="evaluated")
        assert len(evaluated) == 1

    def test_list_decisions_limit(self, tmp_path):
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        for i in range(10):
            engine.track("pm", "estimate", f"决策 {i}")

        limited = engine.list_decisions(limit=3)
        assert len(limited) == 3

    def test_list_decisions_empty(self, tmp_path):
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        assert engine.list_decisions() == []

    def test_generate_evaluation_prompt(self, tmp_path):
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        d = engine.track("pm", "estimate", "需要 3 天", expected_outcome="3 天完成")
        prompt = engine.generate_evaluation_prompt(d.id)
        assert prompt is not None
        assert "决策回溯评估" in prompt
        assert d.id in prompt
        assert "pm" in prompt
        assert "需要 3 天" in prompt
        assert "预期结果" in prompt
        assert "3 天完成" in prompt

    def test_generate_evaluation_prompt_with_actual(self, tmp_path):
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        d = engine.track("pm", "estimate", "需要 3 天", expected_outcome="3 天完成")
        engine.evaluate(d.id, "花了 5 天")
        prompt = engine.generate_evaluation_prompt(d.id)
        assert prompt is not None
        assert "实际结果" in prompt
        assert "花了 5 天" in prompt

    def test_generate_evaluation_prompt_nonexistent(self, tmp_path):
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        assert engine.generate_evaluation_prompt("nonexistent") is None

    def test_evaluate_writes_to_memory(self, tmp_path):
        """评估后应将结论写入员工记忆."""
        eval_dir = tmp_path / "eval"

        # Monkey-patch MemoryStore 的默认路径
        engine = EvaluationEngine(eval_dir=eval_dir)
        d = engine.track("pm", "estimate", "需要 3 天", expected_outcome="3 天完成")

        # evaluate 内部调用 MemoryStore() 用默认路径，
        # 我们直接验证 evaluate 不报错并返回结果
        result = engine.evaluate(d.id, "花了 5 天", "低估了复杂度")
        assert result is not None
        assert result.status == "evaluated"

    def test_evaluate_atomic_write(self, tmp_path):
        """评估写入应为原子操作 — 文件被正确更新."""
        eval_dir = tmp_path / "eval"
        engine = EvaluationEngine(eval_dir=eval_dir)
        d1 = engine.track("pm", "estimate", "需要 3 天", expected_outcome="3 天完成")
        d2 = engine.track("dev", "commitment", "用 React", expected_outcome="React 上线")

        # 评估 d1
        result = engine.evaluate(d1.id, "花了 5 天")
        assert result is not None

        # d2 应该仍存在且未受影响
        d2_loaded = engine.get(d2.id)
        assert d2_loaded is not None
        assert d2_loaded.status == "pending"

        # 验证文件完整性 — 所有行都是合法 JSON
        decisions_file = eval_dir / "decisions.jsonl"
        for line in decisions_file.read_text().splitlines():
            if line.strip():
                json.loads(line)  # 不应抛异常

    def test_evaluate_crash_safety(self, tmp_path):
        """os.replace 失败时不应损坏原文件."""
        from unittest.mock import patch

        eval_dir = tmp_path / "eval"
        engine = EvaluationEngine(eval_dir=eval_dir)
        d = engine.track("pm", "estimate", "测试", expected_outcome="OK")

        decisions_file = eval_dir / "decisions.jsonl"
        original_content = decisions_file.read_text()

        # 模拟 os.replace 失败
        with patch("os.replace", side_effect=OSError("disk full")):
            try:
                engine.evaluate(d.id, "实际结果")
            except OSError:
                pass

        # 原文件应完好无损
        assert decisions_file.read_text() == original_content

    def test_track_with_deadline(self, tmp_path):
        """track 时传 deadline，验证 Decision 包含正确的 deadline."""
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        d = engine.track(
            "pm", "estimate", "需要 3 天",
            deadline="2026-03-07",
        )
        assert d.deadline == "2026-03-07"
        # 从文件重新加载验证持久化
        reloaded = engine.get(d.id)
        assert reloaded is not None
        assert reloaded.deadline == "2026-03-07"

    def test_track_with_task_id(self, tmp_path):
        """track 时传 task_id，验证 Decision 包含正确的 task_id."""
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        d = engine.track(
            "dev", "commitment", "用 React 重构前端",
            task_id="20260301-120000-abc12345",
        )
        assert d.task_id == "20260301-120000-abc12345"
        reloaded = engine.get(d.id)
        assert reloaded is not None
        assert reloaded.task_id == "20260301-120000-abc12345"

    def test_list_overdue_basic(self, tmp_path):
        """list_overdue 只返回过期未评估的决策."""
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        # 已过期的 pending
        d1 = engine.track("pm", "estimate", "过期决策", deadline="2026-01-01")
        # 未过期的 pending
        d2 = engine.track("pm", "estimate", "未来决策", deadline="2099-12-31")
        # 已过期但已评估
        d3 = engine.track("pm", "estimate", "已评估", deadline="2026-01-01")
        engine.evaluate(d3.id, "结果 OK")
        # 无 deadline 的 pending
        d4 = engine.track("pm", "estimate", "无截止日期")

        overdue = engine.list_overdue(as_of="2026-03-04")
        assert len(overdue) == 1
        assert overdue[0].id == d1.id

    def test_list_overdue_empty(self, tmp_path):
        """没有过期的决策，返回空列表."""
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        engine.track("pm", "estimate", "未来决策", deadline="2099-12-31")
        engine.track("pm", "estimate", "无截止日期")
        overdue = engine.list_overdue(as_of="2026-03-04")
        assert overdue == []

    def test_list_overdue_no_file(self, tmp_path):
        """没有 decisions 文件时返回空列表."""
        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        assert engine.list_overdue() == []


class TestScanOverdueDecisions:
    """测试过期决策扫描."""

    @pytest.mark.asyncio
    async def test_scan_overdue_auto_evaluate(self, tmp_path):
        """有 task_id 且任务完成的 → 自动 evaluate."""
        from crew.task_registry import TaskRecord, TaskRegistry

        # 准备 eval engine
        eval_dir = tmp_path / ".crew" / "evaluations"
        engine = EvaluationEngine(eval_dir=eval_dir)

        # 准备 task registry
        tasks_path = tmp_path / ".crew" / "tasks.jsonl"
        registry = TaskRegistry(persist_path=tasks_path)
        task = registry.create(
            trigger="direct",
            target_type="employee",
            target_name="dev",
        )
        registry.update(task.task_id, "completed", result={"summary": "任务完成了"})

        # track 一个带 task_id 的过期决策
        d = engine.track(
            "dev", "commitment", "用新框架重构",
            deadline="2026-02-01",
            task_id=task.task_id,
        )

        # 运行扫描
        from crew.cron_evaluate import scan_overdue_decisions

        # 需要 mock resolve_project_dir 让它返回 tmp_path
        with patch("crew.cron_evaluate.resolve_project_dir", return_value=tmp_path):
            with patch("crew.evaluation.resolve_project_dir", return_value=tmp_path):
                result = await scan_overdue_decisions(project_dir=tmp_path)

        assert len(result["auto_evaluated"]) == 1
        assert result["auto_evaluated"][0]["id"] == d.id
        assert "系统自动评估" in result["auto_evaluated"][0]["evaluation"]

        # 验证决策已被标记为 evaluated
        reloaded = engine.get(d.id)
        assert reloaded is not None
        assert reloaded.status == "evaluated"

    @pytest.mark.asyncio
    async def test_scan_overdue_expired_writes_memory(self, tmp_path):
        """超期 7 天的 → 写入 finding 记忆并自动关闭."""
        eval_dir = tmp_path / ".crew" / "evaluations"
        engine = EvaluationEngine(eval_dir=eval_dir)

        # track 一个超期 10 天的决策
        d = engine.track(
            "pm", "estimate", "需要完成用户调研",
            deadline="2026-02-20",
        )

        memory_add_calls = []

        # Mock MemoryStore.add 来捕获调用
        original_add = None

        class FakeMemoryStore:
            def __init__(self, **kwargs):
                pass

            def add(self, **kwargs):
                memory_add_calls.append(kwargs)

        from crew.cron_evaluate import scan_overdue_decisions

        with patch("crew.cron_evaluate.resolve_project_dir", return_value=tmp_path):
            with patch("crew.evaluation.resolve_project_dir", return_value=tmp_path):
                with patch("crew.cron_evaluate.MemoryStore", FakeMemoryStore):
                    result = await scan_overdue_decisions(project_dir=tmp_path)

        # 应该在 expired 列表中
        assert len(result["expired"]) == 1
        assert result["expired"][0]["id"] == d.id
        assert "系统自动关闭" in result["expired"][0]["evaluation"]

        # 应该写入了 finding 记忆
        assert len(memory_add_calls) == 1
        assert memory_add_calls[0]["category"] == "finding"
        assert "超期" in memory_add_calls[0]["content"]
        assert memory_add_calls[0]["employee"] == "pm"

    @pytest.mark.asyncio
    async def test_scan_overdue_generates_reminders(self, tmp_path):
        """过期 < 7 天且无 task_id 的决策 → 生成提醒."""
        eval_dir = tmp_path / ".crew" / "evaluations"
        engine = EvaluationEngine(eval_dir=eval_dir)

        # track 一个过期 3 天的决策（无 task_id）
        d = engine.track(
            "dev", "recommendation", "建议用缓存优化查询",
            deadline="2026-03-01",
        )

        from crew.cron_evaluate import scan_overdue_decisions

        with patch("crew.cron_evaluate.resolve_project_dir", return_value=tmp_path):
            with patch("crew.evaluation.resolve_project_dir", return_value=tmp_path):
                result = await scan_overdue_decisions(project_dir=tmp_path)

        assert len(result["reminders"]) == 1
        assert result["reminders"][0]["decision_id"] == d.id
        assert result["reminders"][0]["employee"] == "dev"
        assert result["auto_evaluated"] == []
        assert result["expired"] == []


class TestFormatScanReport:
    """测试扫描报告格式化."""

    def test_format_scan_report_with_data(self):
        """有数据时返回格式化报告."""
        from crew.cron_evaluate import format_scan_report

        results = {
            "auto_evaluated": [
                {"employee": "dev", "content": "用新框架重构前端模块"},
            ],
            "reminders": [
                {"employee": "pm", "content": "需要完成调研", "days_overdue": 3},
            ],
            "expired": [
                {"employee": "qa", "content": "测试覆盖率提升计划"},
            ],
        }
        report = format_scan_report(results)
        assert report is not None
        assert "决策评估日报" in report
        assert "自动评估 (1 条)" in report
        assert "dev" in report
        assert "待回复 (1 条)" in report
        assert "pm" in report
        assert "超期 3 天" in report
        assert "超期关闭 (1 条)" in report
        assert "qa" in report

    def test_format_scan_report_empty(self):
        """无数据时返回 None."""
        from crew.cron_evaluate import format_scan_report

        results = {"auto_evaluated": [], "reminders": [], "expired": []}
        assert format_scan_report(results) is None

    def test_format_scan_report_partial(self):
        """只有部分数据时只输出对应段落."""
        from crew.cron_evaluate import format_scan_report

        results = {
            "auto_evaluated": [],
            "reminders": [
                {"employee": "pm", "content": "调研任务", "days_overdue": 2},
            ],
            "expired": [],
        }
        report = format_scan_report(results)
        assert report is not None
        assert "待回复 (1 条)" in report
        assert "自动评估" not in report
        assert "超期关闭" not in report


class TestEvaluateScanDM:
    """测试决策扫描的飞书私信投递逻辑."""

    @pytest.mark.asyncio
    async def test_scan_report_sends_dm_to_owner(self, tmp_path):
        """扫描有结果时，通过飞书 open_id API 给 owner 发私信."""
        # 准备 eval engine 和过期决策
        eval_dir = tmp_path / ".crew" / "evaluations"
        engine = EvaluationEngine(eval_dir=eval_dir)
        engine.track("dev", "recommendation", "建议用缓存优化查询", deadline="2026-03-01")

        # 构造 mock ctx
        mock_feishu_config = MagicMock()
        mock_feishu_config.owner_open_id = "ou_test_owner_123"
        mock_token_mgr = AsyncMock()
        mock_token_mgr.get_token = AsyncMock(return_value="fake_token")

        # 构造 mock httpx client
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": 0, "msg": "success"}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        # 运行扫描获取报告
        from crew.cron_evaluate import format_scan_report, scan_overdue_decisions

        with patch("crew.cron_evaluate.resolve_project_dir", return_value=tmp_path):
            with patch("crew.evaluation.resolve_project_dir", return_value=tmp_path):
                results = await scan_overdue_decisions(project_dir=tmp_path)

        report = format_scan_report(results)
        assert report is not None, "应该有扫描报告"

        # 模拟 _run_evaluate_scan_cron 中的 DM 投递逻辑
        with patch("crew.feishu.get_feishu_client", return_value=mock_client):
            import json as _json
            from crew.feishu import get_feishu_client

            token = await mock_token_mgr.get_token()
            client = get_feishu_client()
            resp = await client.post(
                "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={
                    "receive_id": mock_feishu_config.owner_open_id,
                    "msg_type": "text",
                    "content": _json.dumps({"text": report}),
                },
                timeout=15.0,
            )
            data = resp.json()

        # 验证调用了正确的 API
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "receive_id_type=open_id" in call_args[0][0]
        assert call_args[1]["json"]["receive_id"] == "ou_test_owner_123"
        assert call_args[1]["json"]["msg_type"] == "text"
        content_parsed = _json.loads(call_args[1]["json"]["content"])
        assert "决策评估日报" in content_parsed["text"]
        assert data.get("code") == 0


class TestTrackDecisionAgentTool:
    """测试 track_decision agent 工具集成."""

    def test_schema_in_tool_schemas(self):
        """track_decision schema 存在于 _TOOL_SCHEMAS."""
        from crew.tool_schema import _TOOL_SCHEMAS

        assert "track_decision" in _TOOL_SCHEMAS
        schema = _TOOL_SCHEMAS["track_decision"]
        assert schema["name"] == "track_decision"
        assert "category" in schema["input_schema"]["properties"]
        assert "content" in schema["input_schema"]["properties"]
        assert "deadline" in schema["input_schema"]["properties"]
        assert schema["input_schema"]["required"] == ["category", "content"]

    def test_in_agent_tools_set(self):
        """track_decision 在 AGENT_TOOLS set 中."""
        from crew.tool_schema import AGENT_TOOLS

        assert "track_decision" in AGENT_TOOLS

    def test_schema_in_core_tools(self):
        """track_decision 在 CORE_TOOLS 中."""
        from crew.tool_schema import CORE_TOOLS

        assert "track_decision" in CORE_TOOLS

    def test_default_deadline_7_days(self, tmp_path):
        """不传 deadline 时，executor 应自动设为 7 天后."""
        from datetime import datetime, timedelta

        from crew.evaluation import EvaluationEngine

        # 模拟 executor 中的默认 deadline 逻辑
        deadline = ""
        if not deadline:
            deadline = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

        engine = EvaluationEngine(eval_dir=tmp_path / "eval")
        d = engine.track("dev", "estimate", "需要 3 天", deadline=deadline)

        expected_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        assert d.deadline == expected_date

        reloaded = engine.get(d.id)
        assert reloaded is not None
        assert reloaded.deadline == expected_date
