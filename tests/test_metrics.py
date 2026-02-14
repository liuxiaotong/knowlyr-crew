"""测试运行时指标收集器."""

from crew.metrics import MetricsCollector, get_collector


class TestMetricsCollector:
    """MetricsCollector 单元测试."""

    def test_initial_state(self):
        m = MetricsCollector()
        snap = m.snapshot()
        assert snap["calls"]["total"] == 0
        assert snap["calls"]["success"] == 0
        assert snap["calls"]["failed"] == 0
        assert snap["tokens"]["input"] == 0
        assert snap["tokens"]["output"] == 0
        assert snap["by_employee"] == {}
        assert snap["by_provider"] == {}

    def test_record_success(self):
        m = MetricsCollector()
        m.record_call(employee="bot", provider="anthropic", input_tokens=100, output_tokens=50)
        snap = m.snapshot()
        assert snap["calls"]["total"] == 1
        assert snap["calls"]["success"] == 1
        assert snap["calls"]["failed"] == 0
        assert snap["tokens"]["input"] == 100
        assert snap["tokens"]["output"] == 50

    def test_record_failure(self):
        m = MetricsCollector()
        m.record_call(provider="openai", success=False)
        snap = m.snapshot()
        assert snap["calls"]["total"] == 1
        assert snap["calls"]["success"] == 0
        assert snap["calls"]["failed"] == 1

    def test_by_employee(self):
        m = MetricsCollector()
        m.record_call(employee="code-reviewer", input_tokens=200, output_tokens=100)
        m.record_call(employee="code-reviewer", input_tokens=300, output_tokens=150)
        m.record_call(employee="test-engineer", input_tokens=100, output_tokens=50)
        snap = m.snapshot()
        assert snap["by_employee"]["code-reviewer"]["calls"] == 2
        assert snap["by_employee"]["code-reviewer"]["tokens"] == 750
        assert snap["by_employee"]["test-engineer"]["calls"] == 1
        assert snap["by_employee"]["test-engineer"]["tokens"] == 150

    def test_by_provider(self):
        m = MetricsCollector()
        m.record_call(provider="anthropic", success=True)
        m.record_call(provider="anthropic", success=True)
        m.record_call(provider="anthropic", success=False)
        m.record_call(provider="openai", success=True)
        snap = m.snapshot()
        assert snap["by_provider"]["anthropic"]["calls"] == 3
        assert snap["by_provider"]["anthropic"]["success"] == 2
        assert snap["by_provider"]["anthropic"]["failed"] == 1
        assert snap["by_provider"]["openai"]["calls"] == 1

    def test_uptime(self):
        m = MetricsCollector()
        snap = m.snapshot()
        assert snap["uptime_seconds"] >= 0

    def test_reset(self):
        m = MetricsCollector()
        m.record_call(employee="bot", provider="anthropic", input_tokens=100, output_tokens=50)
        m.reset()
        snap = m.snapshot()
        assert snap["calls"]["total"] == 0
        assert snap["tokens"]["input"] == 0
        assert snap["by_employee"] == {}
        assert snap["by_provider"] == {}

    def test_no_employee_no_provider(self):
        m = MetricsCollector()
        m.record_call(input_tokens=10, output_tokens=5)
        snap = m.snapshot()
        assert snap["calls"]["total"] == 1
        assert snap["by_employee"] == {}
        assert snap["by_provider"] == {}

    def test_global_singleton(self):
        c1 = get_collector()
        c2 = get_collector()
        assert c1 is c2
