"""测试 MemoryStoreDB.query(search_text=...) 和 _semantic_memory_search 降级逻辑."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

from crew.memory import MemoryEntry

# ── MemoryStoreDB.query search_text SQL 构建测试 ──


class TestMemoryStoreDBSearchText:
    """验证 MemoryStoreDB.query 的 search_text 参数签名和 SQL 构建逻辑."""

    def test_query_accepts_search_text_parameter(self):
        """MemoryStoreDB.query 签名中包含 search_text 参数."""
        from crew.memory_store_db import MemoryStoreDB

        sig = inspect.signature(MemoryStoreDB.query)
        assert "search_text" in sig.parameters
        param = sig.parameters["search_text"]
        assert param.default is None

    def test_search_text_generates_ilike_condition(self):
        """search_text 有值时应生成 ILIKE 条件（通过 mock 验证 SQL）."""
        from crew.memory_store_db import MemoryStoreDB

        with (
            patch("crew.memory_store_db.is_pg", return_value=True),
            patch("crew.memory_store_db.get_connection") as mock_conn,
            patch.object(MemoryStoreDB, "__init__", lambda self, **kw: None),
        ):
            store = MemoryStoreDB.__new__(MemoryStoreDB)
            store._tenant_id = "test-tenant"
            store._project_dir = None

            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_conn.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_conn.return_value.__enter__.return_value.cursor.return_value = mock_cursor
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)

            store._dict_cursor_factory = MagicMock()
            store.resolve_to_character_name = lambda e: e

            store.query(employee="赵云帆", search_text="部署")

            call_args = mock_cursor.execute.call_args
            sql = call_args[0][0]
            params = call_args[0][1]

            assert "ILIKE" in sql
            assert "%部署%" in params

    def test_search_text_none_no_ilike(self):
        """search_text=None 时不应生成 ILIKE 条件."""
        from crew.memory_store_db import MemoryStoreDB

        with (
            patch("crew.memory_store_db.is_pg", return_value=True),
            patch("crew.memory_store_db.get_connection") as mock_conn,
            patch.object(MemoryStoreDB, "__init__", lambda self, **kw: None),
        ):
            store = MemoryStoreDB.__new__(MemoryStoreDB)
            store._tenant_id = "test-tenant"
            store._project_dir = None
            store._dict_cursor_factory = MagicMock()
            store.resolve_to_character_name = lambda e: e

            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_conn.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_conn.return_value.__enter__.return_value.cursor.return_value = mock_cursor
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)

            store.query(employee="赵云帆", search_text=None)

            call_args = mock_cursor.execute.call_args
            sql = call_args[0][0]
            assert "ILIKE" not in sql


# ── 辅助函数 ──


def _make_entry(content: str, entry_id: str = "abc123", category: str = "finding") -> MemoryEntry:
    return MemoryEntry(
        id=entry_id,
        employee="赵云帆",
        created_at="2026-03-07T00:00:00",
        category=category,
        content=content,
    )


def _make_store_with_search_text(query_return):
    """创建一个 store mock，其 query 方法有 search_text 参数."""
    call_log = []

    class FakeStore:
        def query(
            self, employee=None, category=None, limit=20, search_text=None, classification_max=None
        ):
            call_log.append(
                {
                    "employee": employee,
                    "category": category,
                    "limit": limit,
                    "search_text": search_text,
                    "classification_max": classification_max,
                }
            )
            return query_return

    store = FakeStore()
    return store, call_log


def _make_store_without_search_text(query_return):
    """创建一个 store mock，其 query 方法没有 search_text 参数（模拟文件版 MemoryStore）."""
    call_log = []

    class FakeStore:
        def query(self, employee=None, category=None, limit=20, classification_max=None):
            call_log.append(
                {
                    "employee": employee,
                    "category": category,
                    "limit": limit,
                    "classification_max": classification_max,
                }
            )
            return query_return

    store = FakeStore()
    return store, call_log


# ── _semantic_memory_search 降级逻辑测试 ──


class TestSemanticMemorySearchFallback:
    """验证 _semantic_memory_search 降级到 search_text 的逻辑."""

    def test_fallback_uses_search_text_for_db_store(self):
        """当 store 没有 memory_dir 且 query 支持 search_text 时，调用 search_text."""
        from crew.webhook_handlers import _semantic_memory_search

        expected = [_make_entry("部署流程已更新")]
        store, call_log = _make_store_with_search_text(expected)

        result = _semantic_memory_search(store, "赵云帆", "部署", None, 5)

        assert result == expected
        assert len(call_log) == 1
        assert call_log[0]["search_text"] == "部署"
        assert call_log[0]["limit"] == 5

    def test_fallback_chinese_substring_for_file_store(self):
        """文件版 store（无 search_text 参数）对中文用子串匹配."""
        from crew.webhook_handlers import _semantic_memory_search

        entries = [
            _make_entry("部署流程已更新", entry_id="a1"),
            _make_entry("数据库迁移需要注意", entry_id="a2"),
            _make_entry("自动部署配置完成", entry_id="a3"),
        ]
        store, _ = _make_store_without_search_text(entries)

        result = _semantic_memory_search(store, "赵云帆", "部署", None, 5)

        assert len(result) == 2
        contents = [e.content for e in result]
        assert "部署流程已更新" in contents
        assert "自动部署配置完成" in contents

    def test_fallback_english_token_matching(self):
        """文件版 store 对英文按空格分词匹配."""
        from crew.webhook_handlers import _semantic_memory_search

        entries = [
            _make_entry("deploy pipeline updated", entry_id="a1"),
            _make_entry("database migration notes", entry_id="a2"),
            _make_entry("deploy config ready", entry_id="a3"),
        ]
        store, _ = _make_store_without_search_text(entries)

        result = _semantic_memory_search(store, "赵云帆", "deploy pipeline", None, 5)

        assert len(result) >= 1
        # "deploy pipeline updated" 匹配 2 个 token，应排最前
        assert result[0].content == "deploy pipeline updated"


class TestMcpLocalSemanticSearch:
    """验证 mcp_server._local_semantic_memory_search 降级逻辑."""

    def test_fallback_uses_search_text_for_db_store(self):
        """mcp_server 降级路径也应使用 search_text."""
        from crew.mcp_server import _local_semantic_memory_search

        expected = [_make_entry("部署流程已更新")]
        store, call_log = _make_store_with_search_text(expected)

        result = _local_semantic_memory_search(
            store, "赵云帆", "部署", None, 5, classification_max="internal"
        )

        assert result == expected
        assert len(call_log) == 1
        assert call_log[0]["search_text"] == "部署"
        assert call_log[0]["classification_max"] == "internal"

    def test_fallback_chinese_for_file_store(self):
        """mcp_server 文件版降级也用中文子串匹配."""
        from crew.mcp_server import _local_semantic_memory_search

        entries = [
            _make_entry("部署流程已更新", entry_id="a1"),
            _make_entry("数据库迁移需要注意", entry_id="a2"),
        ]
        store, _ = _make_store_without_search_text(entries)

        result = _local_semantic_memory_search(store, "赵云帆", "部署", None, 5)

        assert len(result) == 1
        assert result[0].content == "部署流程已更新"
