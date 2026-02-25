"""测试 EventCollector — 统一埋点事件收集器."""

import pytest

from crew.event_collector import EventCollector, _reset_singleton, get_event_collector


@pytest.fixture()
def tmp_db(tmp_path):
    """每个测试使用独立的临时 SQLite 文件."""
    return tmp_path / "test_events.db"


@pytest.fixture()
def collector(tmp_db):
    """创建带临时 DB 的 EventCollector."""
    ec = EventCollector(db_path=tmp_db)
    yield ec
    ec.close()


class TestEventCollectorRecord:
    """测试事件写入."""

    def test_record_basic(self, collector):
        """基本写入应成功."""
        collector.record(
            event_type="tool_call",
            event_name="list_employees",
            duration_ms=12.3,
            success=True,
            source="mcp",
        )
        rows = collector.query(limit=10)
        assert len(rows) == 1
        row = rows[0]
        assert row["event_type"] == "tool_call"
        assert row["event_name"] == "list_employees"
        assert row["duration_ms"] == 12.3
        assert row["success"] is True
        assert row["source"] == "mcp"

    def test_record_failure(self, collector):
        """失败事件应记录 error_type."""
        collector.record(
            event_type="tool_call",
            event_name="run_employee",
            duration_ms=5.0,
            success=False,
            error_type="ValueError",
            source="mcp",
        )
        rows = collector.query()
        assert len(rows) == 1
        assert rows[0]["success"] is False
        assert rows[0]["error_type"] == "ValueError"

    def test_record_with_metadata(self, collector):
        """metadata 字段应正确存储为 JSON."""
        collector.record(
            event_type="employee_run",
            event_name="product-manager",
            duration_ms=100.0,
            success=True,
            metadata={"agent_id": "AI42", "args": {"topic": "test"}},
        )
        rows = collector.query()
        assert len(rows) == 1
        meta = rows[0]["metadata"]
        assert isinstance(meta, dict)
        assert meta["agent_id"] == "AI42"

    def test_record_multiple(self, collector):
        """多条写入应全部持久化."""
        for i in range(5):
            collector.record(
                event_type="tool_call",
                event_name=f"tool_{i}",
                duration_ms=float(i),
                success=True,
            )
        rows = collector.query(limit=10)
        assert len(rows) == 5

    def test_record_no_optional_fields(self, collector):
        """只传必填字段应成功."""
        collector.record(event_type="tool_call", event_name="test")
        rows = collector.query()
        assert len(rows) == 1
        assert rows[0]["duration_ms"] is None
        assert rows[0]["error_type"] is None
        assert rows[0]["source"] is None


class TestEventCollectorQuery:
    """测试事件查询."""

    def test_query_by_event_type(self, collector):
        """按 event_type 过滤."""
        collector.record(event_type="tool_call", event_name="a", success=True)
        collector.record(event_type="employee_run", event_name="b", success=True)
        rows = collector.query(event_type="tool_call")
        assert len(rows) == 1
        assert rows[0]["event_name"] == "a"

    def test_query_by_event_name(self, collector):
        """按 event_name 过滤."""
        collector.record(event_type="tool_call", event_name="list_employees", success=True)
        collector.record(event_type="tool_call", event_name="run_employee", success=True)
        rows = collector.query(event_name="run_employee")
        assert len(rows) == 1
        assert rows[0]["event_name"] == "run_employee"

    def test_query_since_until(self, collector):
        """时间范围过滤."""
        # 手动插入带已知时间戳的记录

        conn = collector._get_conn()
        conn.execute(
            "INSERT INTO events (event_type, event_name, timestamp, success) VALUES (?, ?, ?, ?)",
            ("tool_call", "old", "2026-01-01T00:00:00+00:00", 1),
        )
        conn.execute(
            "INSERT INTO events (event_type, event_name, timestamp, success) VALUES (?, ?, ?, ?)",
            ("tool_call", "new", "2026-02-20T00:00:00+00:00", 1),
        )
        conn.commit()

        rows = collector.query(since="2026-02-01T00:00:00+00:00")
        names = [r["event_name"] for r in rows]
        assert "new" in names
        assert "old" not in names

        rows = collector.query(until="2026-01-31T23:59:59+00:00")
        names = [r["event_name"] for r in rows]
        assert "old" in names
        assert "new" not in names

    def test_query_limit(self, collector):
        """limit 应限制返回条数."""
        for i in range(10):
            collector.record(event_type="tool_call", event_name=f"t{i}", success=True)
        rows = collector.query(limit=3)
        assert len(rows) == 3

    def test_query_order_desc(self, collector):
        """查询结果应按 id 倒序（最新在前）."""
        collector.record(event_type="tool_call", event_name="first", success=True)
        collector.record(event_type="tool_call", event_name="second", success=True)
        rows = collector.query()
        assert rows[0]["event_name"] == "second"
        assert rows[1]["event_name"] == "first"

    def test_query_empty(self, collector):
        """空表应返回空列表."""
        rows = collector.query()
        assert rows == []

    def test_query_combined_filters(self, collector):
        """多条件组合过滤."""
        collector.record(event_type="tool_call", event_name="a", success=True)
        collector.record(event_type="tool_call", event_name="b", success=True)
        collector.record(event_type="employee_run", event_name="a", success=True)
        rows = collector.query(event_type="tool_call", event_name="a")
        assert len(rows) == 1


class TestEventCollectorAggregate:
    """测试聚合统计."""

    def test_aggregate_basic(self, collector):
        """基本聚合应返回正确的 count / success / fail."""
        collector.record(
            event_type="tool_call", event_name="list_employees", duration_ms=10.0, success=True
        )
        collector.record(
            event_type="tool_call", event_name="list_employees", duration_ms=20.0, success=True
        )
        collector.record(
            event_type="tool_call",
            event_name="list_employees",
            duration_ms=5.0,
            success=False,
            error_type="E",
        )
        agg = collector.aggregate(event_type="tool_call")
        assert len(agg) == 1
        row = agg[0]
        assert row["event_name"] == "list_employees"
        assert row["count"] == 3
        assert row["success_count"] == 2
        assert row["fail_count"] == 1
        # avg: (10+20+5)/3 ≈ 11.7
        assert abs(row["avg_duration_ms"] - 11.7) < 0.1

    def test_aggregate_multiple_names(self, collector):
        """多个 event_name 应分组聚合."""
        for _ in range(3):
            collector.record(event_type="tool_call", event_name="a", duration_ms=1.0, success=True)
        collector.record(event_type="tool_call", event_name="b", duration_ms=2.0, success=True)
        agg = collector.aggregate(event_type="tool_call")
        assert len(agg) == 2
        # 按 count 倒序
        assert agg[0]["event_name"] == "a"
        assert agg[0]["count"] == 3
        assert agg[1]["event_name"] == "b"
        assert agg[1]["count"] == 1

    def test_aggregate_with_since(self, collector):
        """since 过滤应生效."""

        conn = collector._get_conn()
        conn.execute(
            "INSERT INTO events (event_type, event_name, timestamp, duration_ms, success) VALUES (?, ?, ?, ?, ?)",
            ("tool_call", "old_tool", "2026-01-01T00:00:00+00:00", 1.0, 1),
        )
        conn.commit()
        collector.record(
            event_type="tool_call", event_name="new_tool", duration_ms=2.0, success=True
        )

        agg = collector.aggregate(event_type="tool_call", since="2026-02-01T00:00:00+00:00")
        names = [r["event_name"] for r in agg]
        assert "new_tool" in names
        assert "old_tool" not in names

    def test_aggregate_no_filter(self, collector):
        """无过滤条件应聚合所有事件."""
        collector.record(event_type="tool_call", event_name="a", success=True)
        collector.record(event_type="employee_run", event_name="b", success=True)
        agg = collector.aggregate()
        assert len(agg) == 2

    def test_aggregate_empty(self, collector):
        """空表应返回空列表."""
        agg = collector.aggregate()
        assert agg == []


class TestEventCollectorSingleton:
    """测试全局单例."""

    def test_singleton(self, tmp_path, monkeypatch):
        """get_event_collector 应返回同一实例."""
        _reset_singleton()
        monkeypatch.setenv("CREW_EVENTS_DB", str(tmp_path / "singleton.db"))
        ec1 = get_event_collector()
        ec2 = get_event_collector()
        assert ec1 is ec2
        _reset_singleton()

    def test_env_db_path(self, tmp_path, monkeypatch):
        """CREW_EVENTS_DB 环境变量应被尊重."""
        _reset_singleton()
        db_file = tmp_path / "custom.db"
        monkeypatch.setenv("CREW_EVENTS_DB", str(db_file))
        ec = get_event_collector()
        ec.record(event_type="test", event_name="x", success=True)
        assert db_file.exists()
        _reset_singleton()


class TestEventCollectorThreadSafety:
    """测试线程安全."""

    def test_concurrent_writes(self, collector):
        """并发写入不应丢数据."""
        import threading

        errors = []

        def writer(n):
            try:
                for i in range(20):
                    collector.record(
                        event_type="tool_call",
                        event_name=f"thread_{n}",
                        duration_ms=float(i),
                        success=True,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        rows = collector.query(limit=200)
        assert len(rows) == 80  # 4 threads * 20 writes
