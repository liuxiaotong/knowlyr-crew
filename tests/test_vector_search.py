"""tests/test_vector_search.py — 混合检索测试.

覆盖：
- _get_query_embedding 方法
- query_by_keywords 混合检索路径 / 纯关键词降级
- query_cross_employee 混合检索路径 / 纯关键词降级
- add() 写入时 embedding 生成

所有测试 mock 数据库和 embedding，不依赖 pgvector / sentence-transformers。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


# ── _get_query_embedding ──


class TestGetQueryEmbedding:
    """测试查询向量生成."""

    def test_returns_vector_when_available(self) -> None:
        """embedding 可用时返回向量."""
        store = _make_store()
        fake_vec = [0.1] * 384

        with patch("crew.embedding.get_embedding", return_value=fake_vec):
            result = store._get_query_embedding(["性能", "优化"])
            assert result == fake_vec

    def test_returns_none_when_unavailable(self) -> None:
        """embedding 不可用时返回 None."""
        store = _make_store()

        with patch("crew.embedding.get_embedding", return_value=None):
            result = store._get_query_embedding(["性能"])
            assert result is None

    def test_returns_none_on_import_error(self) -> None:
        """embedding 模块导入失败时返回 None."""
        store = _make_store()

        with patch.dict("sys.modules", {"crew.embedding": None}):
            result = store._get_query_embedding(["性能"])
            assert result is None

    def test_returns_none_on_exception(self) -> None:
        """embedding 生成异常时返回 None."""
        store = _make_store()

        with patch("crew.embedding.get_embedding", side_effect=RuntimeError("boom")):
            result = store._get_query_embedding(["性能"])
            assert result is None


# ── query_by_keywords 降级 ──


class TestQueryByKeywordsFallback:
    """测试 query_by_keywords 在无 embedding 时降级为纯关键词匹配."""

    def test_fallback_to_keyword_only(self) -> None:
        """embedding 不可用时，走纯关键词路径."""
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        mock_row = {
            "id": "abc123",
            "employee": "test",
            "created_at": "2026-01-01T00:00:00",
            "category": "finding",
            "content": "test content",
            "source_session": "",
            "confidence": 1.0,
            "superseded_by": "",
            "ttl_days": 0,
            "importance": 3,
            "last_accessed": None,
            "tags": [],
            "shared": False,
            "visibility": "open",
            "trigger_condition": "",
            "applicability": [],
            "origin_employee": "test",
            "verified_count": 0,
            "classification": "internal",
            "domain": [],
            "keywords": ["性能"],
            "linked_memories": [],
            "match_count": 1,
        }

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [mock_row]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(store, "_get_query_embedding", return_value=None),
            patch("crew.memory_store_db.get_connection", return_value=mock_conn),
        ):
            results = store.query_by_keywords("test", ["性能"])

        assert len(results) == 1
        assert results[0].id == "abc123"

    def test_empty_keywords_delegates_to_query(self) -> None:
        """空关键词时委托给 query()."""
        store = _make_store()
        store._resolve_to_character_name = lambda x: x
        store.query = MagicMock(return_value=[])

        store.query_by_keywords("test", [])
        store.query.assert_called_once()


# ── query_cross_employee 降级 ──


class TestQueryCrossEmployeeFallback:
    """测试 query_cross_employee 在无 embedding 时降级为纯关键词匹配."""

    def test_empty_keywords_returns_empty(self) -> None:
        """空关键词返回空列表."""
        store = _make_store()
        results = store.query_cross_employee([])
        assert results == []

    def test_fallback_to_keyword_only(self) -> None:
        """embedding 不可用时走纯关键词路径."""
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        mock_row = {
            "id": "def456",
            "employee": "other",
            "created_at": "2026-01-01T00:00:00",
            "category": "pattern",
            "content": "shared pattern",
            "source_session": "",
            "confidence": 1.0,
            "superseded_by": "",
            "ttl_days": 0,
            "importance": 3,
            "last_accessed": None,
            "tags": [],
            "shared": True,
            "visibility": "open",
            "trigger_condition": "",
            "applicability": [],
            "origin_employee": "other",
            "verified_count": 0,
            "classification": "internal",
            "domain": [],
            "keywords": ["优化"],
            "linked_memories": [],
            "match_count": 1,
        }

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [mock_row]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(store, "_get_query_embedding", return_value=None),
            patch("crew.memory_store_db.get_connection", return_value=mock_conn),
        ):
            results = store.query_cross_employee(["优化"], exclude_employee="me")

        assert len(results) == 1
        assert results[0].id == "def456"


# ── add() embedding 生成 ──


class TestAddWithEmbedding:
    """测试 add() 写入时 embedding 生成."""

    def test_add_generates_embedding(self) -> None:
        """add() 调用 get_embedding 并写入向量."""
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        fake_vec = [0.1] * 384
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)

        with (
            patch("crew.embedding.get_embedding", return_value=fake_vec),
            patch("crew.embedding.build_embedding_text", return_value="test content 性能"),
            patch("crew.memory_store_db.get_connection", return_value=mock_conn),
        ):
            entry = store.add(
                employee="test",
                category="finding",
                content="test content",
                keywords=["性能"],
            )

        assert entry.id  # 成功创建
        # 验证 INSERT 被调用且包含 embedding 参数
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args
        # 最后一个参数应该是 embedding_vector
        insert_params = call_args[0][1]
        assert insert_params[-1] == fake_vec

    def test_add_without_embedding(self) -> None:
        """embedding 不可用时，add() 照常写入（embedding=None）."""
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)

        with (
            patch("crew.embedding.get_embedding", return_value=None),
            patch("crew.embedding.build_embedding_text", return_value="test content"),
            patch("crew.memory_store_db.get_connection", return_value=mock_conn),
        ):
            entry = store.add(
                employee="test",
                category="finding",
                content="test content",
            )

        assert entry.id
        insert_params = mock_cursor.execute.call_args[0][1]
        assert insert_params[-1] is None

    def test_add_embedding_exception_does_not_block(self) -> None:
        """embedding 生成抛异常时，add() 不阻塞（embedding=None）."""
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = lambda s: s
        mock_conn.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "crew.embedding.get_embedding",
                side_effect=RuntimeError("model crash"),
            ),
            patch("crew.embedding.build_embedding_text", return_value="text"),
            patch("crew.memory_store_db.get_connection", return_value=mock_conn),
        ):
            entry = store.add(
                employee="test",
                category="finding",
                content="test content",
            )

        assert entry.id
        insert_params = mock_cursor.execute.call_args[0][1]
        assert insert_params[-1] is None
