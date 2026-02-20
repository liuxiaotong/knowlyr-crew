"""测试 Token 成本追踪 — cost.py."""

from crew.cost import (
    COST_PER_1K,
    PROXY_PRICE_OVERRIDES,
    enrich_result_with_cost,
    estimate_cost,
    parse_quality_score,
    query_cost_summary,
)


class TestEstimateCost:
    """成本估算测试."""

    def test_known_model(self):
        """已知模型应按单价计算."""
        cost = estimate_cost("claude-opus-4-6", input_tokens=1000, output_tokens=500)
        expected = (1000 * 0.005 + 500 * 0.025) / 1000
        assert abs(cost - expected) < 1e-6

    def test_proxy_pricing(self):
        """代理商价格应覆盖官方价格."""
        official = estimate_cost("claude-opus-4-6", 1000, 1000)
        proxy = estimate_cost("claude-opus-4-6", 1000, 1000, base_url="https://aiberm.com/v1")
        assert proxy < official
        # aiberm: (1000 * 0.00095 + 1000 * 0.00475) / 1000 = 0.0057
        assert abs(proxy - 0.0057) < 1e-6

    def test_proxy_unknown_model_falls_back(self):
        """代理商不在覆盖表中的模型用官方价格."""
        cost_proxy = estimate_cost("kimi-k2.5", 1000, 1000, base_url="https://aiberm.com/v1")
        cost_direct = estimate_cost("kimi-k2.5", 1000, 1000)
        assert cost_proxy == cost_direct

    def test_cheap_model(self):
        """便宜模型成本应更低."""
        opus_cost = estimate_cost("claude-opus-4-6", 1000, 1000)
        kimi_cost = estimate_cost("kimi-k2.5", 1000, 1000)
        assert kimi_cost < opus_cost

    def test_unknown_model_uses_default(self):
        """未知模型用默认价格."""
        cost = estimate_cost("some-unknown-model", 1000, 1000)
        assert cost > 0

    def test_zero_tokens(self):
        """0 token 成本为 0."""
        assert estimate_cost("claude-opus-4-6", 0, 0) == 0.0


class TestEnrichResult:
    """结果追加成本字段测试."""

    def test_adds_cost_usd(self):
        """应追加 cost_usd 字段."""
        result = {
            "model": "kimi-k2.5",
            "input_tokens": 5000,
            "output_tokens": 2000,
        }
        enriched = enrich_result_with_cost(result)
        assert "cost_usd" in enriched
        assert enriched["cost_usd"] > 0

    def test_no_tokens_no_cost(self):
        """无 token 数据不追加成本."""
        result = {"model": "kimi-k2.5"}
        enriched = enrich_result_with_cost(result)
        assert "cost_usd" not in enriched

    def test_preserves_existing_fields(self):
        """不覆盖已有字段."""
        result = {
            "employee": "test",
            "model": "kimi-k2.5",
            "input_tokens": 100,
            "output_tokens": 50,
        }
        enriched = enrich_result_with_cost(result)
        assert enriched["employee"] == "test"

    def test_uses_base_url_for_proxy_pricing(self):
        """有 base_url 时使用代理价格."""
        result = {
            "model": "claude-opus-4-6",
            "input_tokens": 1000,
            "output_tokens": 1000,
            "base_url": "https://aiberm.com/v1",
        }
        enriched = enrich_result_with_cost(result)
        # 代理价: (1000 * 0.00095 + 1000 * 0.00475) / 1000 = 0.0057
        assert abs(enriched["cost_usd"] - 0.0057) < 1e-6


class TestQueryCostSummary:
    """成本汇总查询测试."""

    def test_empty_registry(self):
        """空注册表返回零."""
        from crew.task_registry import TaskRegistry

        registry = TaskRegistry()
        summary = query_cost_summary(registry)
        assert summary["total_tasks"] == 0
        assert summary["total_cost_usd"] == 0

    def test_with_completed_tasks(self):
        """有完成任务时应汇总."""
        from crew.task_registry import TaskRegistry

        registry = TaskRegistry()
        record = registry.create(
            trigger="test",
            target_type="employee",
            target_name="code-reviewer",
        )
        registry.update(
            record.task_id,
            "completed",
            result={
                "employee": "code-reviewer",
                "model": "kimi-k2.5",
                "input_tokens": 5000,
                "output_tokens": 2000,
                "cost_usd": 0.058,
            },
        )
        summary = query_cost_summary(registry)
        assert summary["total_tasks"] == 1
        assert summary["total_cost_usd"] > 0
        assert "code-reviewer" in summary["by_employee"]

    def test_filter_by_employee(self):
        """按员工过滤."""
        from crew.task_registry import TaskRegistry

        registry = TaskRegistry()
        for emp in ["code-reviewer", "test-engineer"]:
            record = registry.create(
                trigger="test",
                target_type="employee",
                target_name=emp,
            )
            registry.update(
                record.task_id,
                "completed",
                result={
                    "employee": emp,
                    "model": "kimi-k2.5",
                    "input_tokens": 1000,
                    "output_tokens": 500,
                },
            )
        summary = query_cost_summary(registry, employee="code-reviewer")
        assert summary["total_tasks"] == 1

    def test_filter_by_source_work(self):
        """source=work 过滤掉 feishu 闲聊."""
        from crew.task_registry import TaskRegistry

        registry = TaskRegistry()
        # 正式任务
        r1 = registry.create(trigger="direct", target_type="employee", target_name="cr")
        registry.update(
            r1.task_id,
            "completed",
            result={
                "employee": "cr",
                "model": "kimi-k2.5",
                "input_tokens": 1000,
                "output_tokens": 500,
            },
        )
        # 飞书闲聊
        r2 = registry.create(trigger="feishu", target_type="employee", target_name="cr")
        registry.update(
            r2.task_id,
            "completed",
            result={
                "employee": "cr",
                "model": "kimi-k2.5",
                "input_tokens": 1000,
                "output_tokens": 500,
            },
        )
        # source=work 应只有 1 条
        summary = query_cost_summary(registry, source="work")
        assert summary["total_tasks"] == 1
        # source=chat 应只有 1 条
        summary_chat = query_cost_summary(registry, source="chat")
        assert summary_chat["total_tasks"] == 1
        # 无过滤应有 2 条
        summary_all = query_cost_summary(registry)
        assert summary_all["total_tasks"] == 2

    def test_by_trigger_in_output(self):
        """返回值中包含 by_trigger 分组."""
        from crew.task_registry import TaskRegistry

        registry = TaskRegistry()
        for trigger in ["direct", "feishu", "github"]:
            r = registry.create(trigger=trigger, target_type="employee", target_name="cr")
            registry.update(
                r.task_id,
                "completed",
                result={
                    "employee": "cr",
                    "model": "kimi-k2.5",
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
            )
        summary = query_cost_summary(registry)
        assert "by_trigger" in summary
        assert "direct" in summary["by_trigger"]
        assert "feishu" in summary["by_trigger"]
        assert "github" in summary["by_trigger"]


class TestParseQualityScore:
    """质量评分解析测试."""

    def test_parse_valid_score(self):
        """应从输出中解析评分."""
        output = '一些代码审查结果...\n{"score": 72, "critical": 3, "warning": 5, "suggestion": 4}'
        result = parse_quality_score(output)
        assert result is not None
        assert result["score"] == 72
        assert result["critical"] == 3

    def test_parse_from_middle(self):
        """评分块在中间也能解析."""
        output = '开头\n{"score": 85}\n结尾'
        result = parse_quality_score(output)
        assert result is not None
        assert result["score"] == 85

    def test_no_score_returns_none(self):
        """无评分块返回 None."""
        output = "这是普通输出，没有评分。"
        assert parse_quality_score(output) is None

    def test_empty_output(self):
        """空输出返回 None."""
        assert parse_quality_score("") is None
        assert parse_quality_score(None) is None

    def test_invalid_json_ignored(self):
        """无效 JSON 被忽略."""
        output = '{"score": invalid}'
        assert parse_quality_score(output) is None

    def test_prefers_last_match(self):
        """优先取最后一个匹配."""
        output = '{"score": 50}\n修改后\n{"score": 80}'
        result = parse_quality_score(output)
        assert result["score"] == 80


class TestCostPriceTable:
    """价格表完整性测试."""

    def test_all_entries_have_input_output(self):
        """每个模型都有 input 和 output 价格."""
        for model, prices in COST_PER_1K.items():
            assert "input" in prices, f"{model} 缺少 input 价格"
            assert "output" in prices, f"{model} 缺少 output 价格"
            assert prices["input"] >= 0
            assert prices["output"] >= 0

    def test_major_models_present(self):
        """主要模型都在价格表中."""
        assert "claude-opus-4-6" in COST_PER_1K
        assert "kimi-k2.5" in COST_PER_1K
        assert "gpt-4o" in COST_PER_1K
        assert "moonshot-v1-32k" in COST_PER_1K
        assert "moonshot-v1-8k" in COST_PER_1K

    def test_proxy_overrides_present(self):
        """代理商覆盖表包含 aiberm."""
        assert "https://aiberm.com/v1" in PROXY_PRICE_OVERRIDES
        aiberm = PROXY_PRICE_OVERRIDES["https://aiberm.com/v1"]
        assert "claude-opus-4-6" in aiberm
        assert aiberm["claude-opus-4-6"]["input"] < COST_PER_1K["claude-opus-4-6"]["input"]
