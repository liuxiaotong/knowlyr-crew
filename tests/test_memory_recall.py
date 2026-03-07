"""Tests for memory recall: query_by_keywords and query_cross_employee."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

from crew.memory import MemoryEntry

# ── Helpers ──


def _make_entry(
    id: str = "abc123",
    employee: str = "测试员工",
    category: str = "finding",
    content: str = "测试内容",
    keywords: list[str] | None = None,
    visibility: str = "open",
    **kwargs,
) -> MemoryEntry:
    """创建测试用 MemoryEntry."""
    return MemoryEntry(
        id=id,
        employee=employee,
        category=category,
        content=content,
        keywords=keywords or [],
        visibility=visibility,
        **kwargs,
    )


def _make_store():
    """创建一个 mock 出的 MemoryStoreDB 实例，跳过 psycopg2 依赖."""
    with patch("crew.memory_store_db.is_pg", return_value=True), \
         patch("crew.memory_store_db.get_connection"):
        # 在导入阶段 mock psycopg2
        mock_psycopg2 = MagicMock()
        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2, "psycopg2.extras": mock_psycopg2.extras}):
            from crew.memory_store_db import MemoryStoreDB
            store = MemoryStoreDB.__new__(MemoryStoreDB)
            store._project_dir = None
            store._tenant_id = "test-tenant"
            store._dict_cursor_factory = MagicMock()
            return store


# ── Feature 1: query_by_keywords ──


class TestQueryByKeywords:
    """测试 query_by_keywords 的排序逻辑."""

    @patch("crew.memory_store_db.get_connection")
    @patch("crew.memory_store_db.is_pg", return_value=True)
    def test_multi_keyword_match_ranked_higher(self, mock_is_pg, mock_conn):
        """匹配更多关键词的记忆排在前面."""
        store = _make_store()

        row_2_matches = {
            "id": "mem001", "employee": "测试员工",
            "created_at": "2026-03-01T00:00:00", "category": "finding",
            "content": "API 性能优化方案", "source_session": "",
            "confidence": 1.0, "superseded_by": "", "ttl_days": 0,
            "importance": 3, "last_accessed": None, "tags": [],
            "shared": False, "visibility": "open", "trigger_condition": "",
            "applicability": [], "origin_employee": "", "verified_count": 0,
            "classification": "internal", "domain": [],
            "keywords": ["API", "性能", "优化"], "linked_memories": [],
            "match_count": 2,
        }
        row_1_match = {
            **row_2_matches,
            "id": "mem002", "content": "数据库索引方案",
            "keywords": ["数据库", "索引"], "match_count": 1,
        }

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [row_2_matches, row_1_match]
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cm)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_cm.cursor.return_value = mock_cursor
        mock_conn.return_value = mock_cm

        results = store.query_by_keywords(
            employee="测试员工", keywords=["API", "性能"],
        )

        assert len(results) == 2
        assert results[0].id == "mem001"
        assert results[1].id == "mem002"

        # 验证 SQL 包含 match_count 排序
        executed_sql = mock_cursor.execute.call_args[0][0]
        assert "match_count" in executed_sql
        assert "ORDER BY match_count DESC" in executed_sql

    @patch("crew.memory_store_db.get_connection")
    @patch("crew.memory_store_db.is_pg", return_value=True)
    def test_empty_keywords_fallback_to_query(self, mock_is_pg, mock_conn):
        """空关键词列表时 fallback 到 query."""
        store = _make_store()

        with patch.object(store, "query", return_value=[]) as mock_query:
            results = store.query_by_keywords(employee="测试员工", keywords=[])
            mock_query.assert_called_once()
            assert results == []

    @patch("crew.memory_store_db.get_connection")
    @patch("crew.memory_store_db.is_pg", return_value=True)
    def test_category_filter(self, mock_is_pg, mock_conn):
        """category 过滤正确传入 SQL."""
        store = _make_store()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cm)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_cm.cursor.return_value = mock_cursor
        mock_conn.return_value = mock_cm

        store.query_by_keywords(
            employee="测试员工", keywords=["API"], category="pattern",
        )

        executed_sql = mock_cursor.execute.call_args[0][0]
        assert "category = %s" in executed_sql
        executed_params = mock_cursor.execute.call_args[0][1]
        assert "pattern" in executed_params

    @patch("crew.memory_store_db.get_connection")
    @patch("crew.memory_store_db.is_pg", return_value=True)
    def test_ilike_pattern_in_sql(self, mock_is_pg, mock_conn):
        """SQL 中对每个关键词生成 ILIKE 条件."""
        store = _make_store()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cm)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_cm.cursor.return_value = mock_cursor
        mock_conn.return_value = mock_cm

        store.query_by_keywords(
            employee="测试员工", keywords=["API", "性能", "缓存"],
        )

        executed_sql = mock_cursor.execute.call_args[0][0]
        # 应该有 3 个 ILIKE 条件用于 match_count
        assert executed_sql.count("ILIKE") >= 3


# ── Feature 2: query_cross_employee ──


class TestQueryCrossEmployee:
    """测试跨员工查询."""

    @patch("crew.memory_store_db.get_connection")
    @patch("crew.memory_store_db.is_pg", return_value=True)
    def test_cross_employee_filters_visibility_open(self, mock_is_pg, mock_conn):
        """跨员工查询仅返回 visibility=open 的记忆."""
        store = _make_store()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cm)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_cm.cursor.return_value = mock_cursor
        mock_conn.return_value = mock_cm

        store.query_cross_employee(
            keywords=["API", "性能"], exclude_employee="赵云帆",
        )

        executed_sql = mock_cursor.execute.call_args[0][0]
        assert "visibility = 'open'" in executed_sql

    @patch("crew.memory_store_db.get_connection")
    @patch("crew.memory_store_db.is_pg", return_value=True)
    def test_cross_employee_excludes_specified(self, mock_is_pg, mock_conn):
        """正确排除指定员工."""
        store = _make_store()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cm)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_cm.cursor.return_value = mock_cursor
        mock_conn.return_value = mock_cm

        store.query_cross_employee(
            keywords=["记忆"], exclude_employee="赵云帆",
        )

        executed_sql = mock_cursor.execute.call_args[0][0]
        assert "employee != %s" in executed_sql
        executed_params = mock_cursor.execute.call_args[0][1]
        assert "赵云帆" in executed_params

    def test_cross_employee_empty_keywords_returns_empty(self):
        """空关键词返回空列表."""
        store = _make_store()
        results = store.query_cross_employee(keywords=[])
        assert results == []

    @patch("crew.memory_store_db.get_connection")
    @patch("crew.memory_store_db.is_pg", return_value=True)
    def test_cross_employee_no_exclude(self, mock_is_pg, mock_conn):
        """不排除任何员工时 SQL 不包含 employee != 条件."""
        store = _make_store()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cm)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_cm.cursor.return_value = mock_cursor
        mock_conn.return_value = mock_cm

        store.query_cross_employee(keywords=["API"], exclude_employee="")

        executed_sql = mock_cursor.execute.call_args[0][0]
        assert "employee != %s" not in executed_sql


# ── task_context keyword extraction ──


class TestTaskContextExtraction:
    """测试 task_context 关键词提取逻辑."""

    def test_extract_keywords_from_task_context(self):
        """从 task_context 提取关键词：按空格/标点分词，取 2 字以上."""
        task_context = "API 性能优化 数据库索引 a"
        tokens = re.split(r"[\s,;，；。！？、\-/\\:：\[\]()（）\"']+", task_context)
        keywords = [t for t in tokens if len(t) >= 2][:10]

        assert "API" in keywords
        assert "性能优化" in keywords
        assert "数据库索引" in keywords
        assert "a" not in keywords

    def test_extract_chinese_keywords(self):
        """中文标点分词."""
        task_context = "记忆系统，关键词匹配；跨员工查询"
        tokens = re.split(r"[\s,;，；。！？、\-/\\:：\[\]()（）\"']+", task_context)
        keywords = [t for t in tokens if len(t) >= 2][:10]

        assert "记忆系统" in keywords
        assert "关键词匹配" in keywords
        assert "跨员工查询" in keywords

    def test_max_10_keywords(self):
        """最多提取 10 个关键词."""
        task_context = " ".join([f"词语{i}" for i in range(20)])
        tokens = re.split(r"[\s,;，；。！？、\-/\\:：\[\]()（）\"']+", task_context)
        keywords = [t for t in tokens if len(t) >= 2][:10]

        assert len(keywords) == 10
