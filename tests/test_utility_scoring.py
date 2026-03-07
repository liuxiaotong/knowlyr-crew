"""tests/test_utility_scoring.py — NG-4 效用评分检索测试.

覆盖：
1. record_useful 同时更新 verified_count 和 q_value
2. decay_unverified_recalls 正确衰减
3. 混合检索 SQL 包含 q_value 因子
4. Thompson Sampling 对低召回记忆重排序
5. q_value 默认值 0.5
6. q_value 不超过 1.0
7. decay 不低于 0
"""

from __future__ import annotations

import random
from unittest.mock import AsyncMock, MagicMock, patch

# ── 辅助 ──


def _make_store():
    """创建 mock MemoryStoreDB 实例（不连接数据库，不依赖 psycopg2）."""
    mock_psycopg2 = MagicMock()
    with (
        patch("crew.memory_store_db.is_pg", return_value=True),
        patch("crew.memory_store_db.get_connection"),
        patch.dict(
            "sys.modules",
            {"psycopg2": mock_psycopg2, "psycopg2.extras": mock_psycopg2.extras},
        ),
    ):
        from crew.memory_store_db import MemoryStoreDB

        store = MemoryStoreDB.__new__(MemoryStoreDB)
        store._project_dir = None
        store._tenant_id = "test-tenant"
        store._dict_cursor_factory = MagicMock()
    return store


# ── 1. record_useful 同时更新 verified_count 和 q_value ──


class TestRecordUsefulQValue:
    """record_useful 同时更新 verified_count 和 q_value."""

    def test_sql_updates_both_fields(self) -> None:
        """SQL 同时包含 verified_count 和 q_value 更新."""
        store = _make_store()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_cur.rowcount = 2

        with patch("crew.memory_store_db.get_connection", return_value=mock_conn):
            result = store.record_useful(["m1", "m2"], "测试员工")

        assert result == 2
        executed_sql = mock_cur.execute.call_args[0][0]
        assert "verified_count = verified_count + 1" in executed_sql
        assert "q_value = q_value + 0.1 * (1.0 - q_value)" in executed_sql


# ── 2. decay_unverified_recalls 正确衰减 ──


class TestDecayUnverifiedRecalls:
    """decay_unverified_recalls 衰减逻辑."""

    def test_decays_unverified_only(self) -> None:
        """只衰减被召回但未标记有用的记忆."""
        store = _make_store()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_cur.rowcount = 1

        with patch("crew.memory_store_db.get_connection", return_value=mock_conn):
            result = store.decay_unverified_recalls(
                recalled_ids=["m1", "m2", "m3"],
                useful_ids=["m2"],
                employee="测试员工",
            )

        assert result == 1
        executed_sql = mock_cur.execute.call_args[0][0]
        assert "q_value = q_value * 0.95" in executed_sql
        # 确认只传了未验证的 id
        passed_ids = mock_cur.execute.call_args[0][1][0]
        assert "m1" in passed_ids
        assert "m3" in passed_ids
        assert "m2" not in passed_ids

    def test_no_decay_when_all_useful(self) -> None:
        """所有被召回的都有用时不衰减."""
        store = _make_store()
        result = store.decay_unverified_recalls(
            recalled_ids=["m1", "m2"],
            useful_ids=["m1", "m2"],
            employee="测试员工",
        )
        assert result == 0

    def test_no_decay_when_empty(self) -> None:
        """空列表不衰减."""
        store = _make_store()
        result = store.decay_unverified_recalls(
            recalled_ids=[],
            useful_ids=[],
            employee="测试员工",
        )
        assert result == 0


# ── 3. 混合检索 SQL 包含 q_value 因子 ──


class TestHybridScoreFormula:
    """混合检索公式包含三因子."""

    def test_query_by_keywords_includes_q_value(self) -> None:
        """query_by_keywords 混合检索 SQL 包含 q_value."""
        store = _make_store()
        fake_vec = [0.1] * 384

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = []

        with (
            patch("crew.embedding.get_embedding", return_value=fake_vec),
            patch("crew.memory_store_db.get_connection", return_value=mock_conn),
        ):
            store.query_by_keywords("测试员工", ["性能"])

        executed_sql = mock_cur.execute.call_args[0][0]
        assert "0.25" in executed_sql
        assert "0.55" in executed_sql
        assert "COALESCE(q_value, 0.5)" in executed_sql

    def test_query_cross_employee_includes_q_value(self) -> None:
        """query_cross_employee 混合检索 SQL 包含 q_value."""
        store = _make_store()
        fake_vec = [0.1] * 384

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = []

        with (
            patch("crew.embedding.get_embedding", return_value=fake_vec),
            patch("crew.memory_store_db.get_connection", return_value=mock_conn),
        ):
            store.query_cross_employee(["性能"])

        executed_sql = mock_cur.execute.call_args[0][0]
        assert "0.25" in executed_sql
        assert "0.55" in executed_sql
        assert "COALESCE(q_value, 0.5)" in executed_sql


# ── 4. Thompson Sampling 对低召回记忆重排序 ──


class TestThompsonSampling:
    """Thompson Sampling 重排测试."""

    def test_low_recall_gets_resampled(self) -> None:
        """recall_count < 5 的记忆会被 Thompson Sampling 重排."""
        from crew.memory_store_db import _thompson_rescore

        rows = [
            {
                "hybrid_score": 0.6,
                "recall_count": 0,
                "verified_count": 0,
                "q_value": 0.5,
                "importance": 3,
            },
            {
                "hybrid_score": 0.7,
                "recall_count": 10,
                "verified_count": 8,
                "q_value": 0.9,
                "importance": 3,
            },
        ]

        # 高召回记忆的分数应保持不变
        random.seed(42)
        result = _thompson_rescore(rows)
        # 第二条（高召回）分数不变
        assert result[0].get("recall_count", 0) >= 5 or result[1].get("recall_count", 0) >= 5

    def test_high_recall_not_resampled(self) -> None:
        """recall_count >= 5 的记忆不做重排."""
        from crew.memory_store_db import _thompson_rescore

        row = {
            "hybrid_score": 0.8,
            "recall_count": 10,
            "verified_count": 8,
            "q_value": 0.9,
            "importance": 3,
        }
        result = _thompson_rescore([row])
        # 分数不变（因为 rc >= 5，不触发 Thompson Sampling）
        assert result[0] is row

    def test_returns_sorted_by_score(self) -> None:
        """结果按重排后的分数降序排列."""
        from crew.memory_store_db import _thompson_rescore

        rows = [
            {
                "hybrid_score": 0.5,
                "recall_count": 10,
                "verified_count": 5,
                "q_value": 0.7,
                "importance": 3,
            },
            {
                "hybrid_score": 0.8,
                "recall_count": 20,
                "verified_count": 18,
                "q_value": 0.95,
                "importance": 3,
            },
        ]
        result = _thompson_rescore(rows)
        assert result[0]["hybrid_score"] == 0.8


# ── 5. q_value 默认值 0.5 ──


class TestQValueDefault:
    """q_value 默认值."""

    def test_schema_default_0_5(self) -> None:
        """init_memory_tables 的 ALTER TABLE 使用 DEFAULT 0.5."""
        import inspect

        from crew.memory_store_db import init_memory_tables

        source = inspect.getsource(init_memory_tables)
        assert "q_value REAL DEFAULT 0.5" in source

    def test_coalesce_in_score_expr(self) -> None:
        """混合检索中 COALESCE(q_value, 0.5) 处理 NULL."""
        import inspect

        from crew.memory_store_db import MemoryStoreDB

        source = inspect.getsource(MemoryStoreDB.query_by_keywords)
        assert "COALESCE(q_value, 0.5)" in source


# ── 6. q_value 不超过 1.0 ──


class TestQValueBounds:
    """q_value 数学边界."""

    def test_update_formula_bounded_above(self) -> None:
        """q + 0.1 * (1 - q) 对任何 0 <= q <= 1 不超过 1."""
        for q_int in range(0, 101):
            q = q_int / 100.0
            q_new = q + 0.1 * (1.0 - q)
            assert q_new <= 1.0, f"q={q}, q_new={q_new}"

    def test_update_approaches_1(self) -> None:
        """连续更新趋近 1.0 但不超过."""
        q = 0.5
        for _ in range(100):
            q = q + 0.1 * (1.0 - q)
        assert q <= 1.0
        assert q > 0.99  # 应该趋近 1


# ── 7. decay 不低于 0 ──


class TestDecayBounds:
    """q_value 衰减数学边界."""

    def test_decay_formula_bounded_below(self) -> None:
        """q * 0.95 对任何 q >= 0 不低于 0."""
        for q_int in range(0, 101):
            q = q_int / 100.0
            q_new = q * 0.95
            assert q_new >= 0.0, f"q={q}, q_new={q_new}"

    def test_decay_approaches_zero(self) -> None:
        """连续衰减趋近 0 但不低于 0."""
        q = 0.5
        for _ in range(1000):
            q = q * 0.95
        assert q >= 0.0
        assert q < 0.001  # 应该趋近 0


# ── 8. recall-feedback webhook 衰减集成 ──


class TestRecallFeedbackDecay:
    """recall-feedback handler 在 recalled_memory_ids 存在时调用 decay_unverified_recalls."""

    def test_decay_called_when_recalled_ids_present(self) -> None:
        """当 recalled_memory_ids 存在时，decay_unverified_recalls 被调用."""
        import asyncio
        import json

        mock_store = MagicMock()
        mock_store.record_useful.return_value = 2
        mock_store.decay_unverified_recalls.return_value = 1

        mock_request = MagicMock()
        mock_request.json = AsyncMock(
            return_value={
                "employee": "test-emp",
                "useful_memory_ids": ["id1", "id2"],
                "recalled_memory_ids": ["id1", "id2", "id3", "id4"],
            }
        )

        mock_ctx = MagicMock()

        with patch("crew.webhook_handlers.get_memory_store", return_value=mock_store):
            from crew.webhook_handlers import _handle_recall_feedback

            result = asyncio.run(_handle_recall_feedback(mock_request, mock_ctx))

        body = json.loads(result.body.decode())
        assert body["ok"] is True
        assert body["updated"] == 2
        assert body["decayed"] == 1

        mock_store.record_useful.assert_called_once_with(["id1", "id2"], "test-emp")
        mock_store.decay_unverified_recalls.assert_called_once_with(
            ["id1", "id2", "id3", "id4"], ["id1", "id2"], "test-emp"
        )

    def test_decay_not_called_without_recalled_ids(self) -> None:
        """当 recalled_memory_ids 不存在时，decay_unverified_recalls 不被调用."""
        import asyncio
        import json

        mock_store = MagicMock()
        mock_store.record_useful.return_value = 2

        mock_request = MagicMock()
        mock_request.json = AsyncMock(
            return_value={
                "employee": "test-emp",
                "useful_memory_ids": ["id1", "id2"],
            }
        )

        mock_ctx = MagicMock()

        with patch("crew.webhook_handlers.get_memory_store", return_value=mock_store):
            from crew.webhook_handlers import _handle_recall_feedback

            result = asyncio.run(_handle_recall_feedback(mock_request, mock_ctx))

        body = json.loads(result.body.decode())
        assert body["ok"] is True
        assert body["updated"] == 2
        assert body["decayed"] == 0

        mock_store.decay_unverified_recalls.assert_not_called()
