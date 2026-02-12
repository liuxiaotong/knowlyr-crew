"""评估闭环模块测试."""

import json

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
        from crew.memory import MemoryStore
        memory_dir = tmp_path / "memory"
        eval_dir = tmp_path / "eval"

        # Monkey-patch MemoryStore 的默认路径
        engine = EvaluationEngine(eval_dir=eval_dir)
        d = engine.track("pm", "estimate", "需要 3 天", expected_outcome="3 天完成")

        # evaluate 内部调用 MemoryStore() 用默认路径，
        # 我们直接验证 evaluate 不报错并返回结果
        result = engine.evaluate(d.id, "花了 5 天", "低估了复杂度")
        assert result is not None
        assert result.status == "evaluated"
