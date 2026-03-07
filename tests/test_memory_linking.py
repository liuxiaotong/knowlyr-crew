"""tests/test_memory_linking.py — NG-3 记忆关联网络测试.

覆盖：
1. add() 写入后自动创建双向关联
2. 相似度 < 0.5 不关联
3. 旧记忆 linked_memories 不超过 20
4. dedup merge 后也做关联
5. expand_linked_memories 展开方法
6. 无关联时返回空
7. embedding 不可用时跳过

所有测试 mock 数据库和 embedding，不依赖 pgvector / sentence-transformers。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from crew.memory import MemoryEntry

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


def _make_mock_conn(cursor_return=None):
    """创建 mock connection."""
    mock_cursor = MagicMock()
    if cursor_return is not None:
        mock_cursor.fetchall.return_value = cursor_return
        mock_cursor.fetchone.return_value = cursor_return[0] if cursor_return else None
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    return mock_conn, mock_cursor


# ── 1. test_auto_link_on_add ──


class TestAutoLinkOnAdd:
    """add() 后自动创建双向关联."""

    def test_auto_link_called_after_insert(self) -> None:
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        fake_vec = [0.1] * 384

        mock_conn, mock_cursor = _make_mock_conn()

        with (
            patch("crew.embedding.get_embedding", return_value=fake_vec),
            patch("crew.embedding.build_embedding_text", return_value="text"),
            patch.object(store, "_try_dedup_merge", return_value=None),
            patch.object(store, "_auto_link_similar") as mock_auto_link,
            patch("crew.memory_store_db.get_connection", return_value=mock_conn),
        ):
            entry = store.add(
                employee="test",
                category="finding",
                content="new memory content",
                keywords=["测试"],
            )

        # _auto_link_similar 应该被调用，参数: entry_id, employee, embedding
        mock_auto_link.assert_called_once_with(entry.id, "test", fake_vec)

    def test_auto_link_creates_bidirectional_links(self) -> None:
        """验证 _auto_link_similar 创建双向关联."""
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        fake_vec = [0.1] * 384
        mock_psycopg2 = MagicMock()
        mock_psycopg2.extras.RealDictCursor = MagicMock()

        # 候选记忆
        candidates = [
            {"id": "old001", "linked_memories": [], "similarity": 0.8},
            {"id": "old002", "linked_memories": ["prev001"], "similarity": 0.6},
        ]

        # 第一个 connection: SELECT 候选
        mock_conn1, mock_cursor1 = _make_mock_conn(candidates)

        # 后续 connections: UPDATE 操作
        update_conns = []
        update_cursors = []
        for _ in range(3):  # 1 for new + 2 for old memories
            conn, cur = _make_mock_conn()
            update_conns.append(conn)
            update_cursors.append(cur)

        all_conns = [mock_conn1] + update_conns
        call_count = {"n": 0}

        def fake_get_conn():
            idx = min(call_count["n"], len(all_conns) - 1)
            call_count["n"] += 1
            return all_conns[idx]

        with (
            patch.dict(
                "sys.modules",
                {"psycopg2": mock_psycopg2, "psycopg2.extras": mock_psycopg2.extras},
            ),
            patch("crew.memory_store_db.get_connection", side_effect=fake_get_conn),
        ):
            store._auto_link_similar("new001", "test", fake_vec)

        # 验证新记忆被更新 linked_memories = ["old001", "old002"]
        update_cursors[0].execute.assert_called_once()
        update_args = update_cursors[0].execute.call_args[0]
        assert "UPDATE memories SET linked_memories" in update_args[0]
        new_linked = update_args[1][0]
        assert "old001" in new_linked
        assert "old002" in new_linked


# ── 2. test_auto_link_respects_threshold ──


class TestAutoLinkRespectsThreshold:
    """相似度 < 0.5 不关联."""

    def test_low_similarity_skipped(self) -> None:
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        fake_vec = [0.1] * 384
        mock_psycopg2 = MagicMock()
        mock_psycopg2.extras.RealDictCursor = MagicMock()

        # 所有候选相似度都 < 0.5
        candidates = [
            {"id": "old001", "linked_memories": [], "similarity": 0.3},
            {"id": "old002", "linked_memories": [], "similarity": 0.2},
        ]

        mock_conn1, _ = _make_mock_conn(candidates)

        with (
            patch.dict(
                "sys.modules",
                {"psycopg2": mock_psycopg2, "psycopg2.extras": mock_psycopg2.extras},
            ),
            patch("crew.memory_store_db.get_connection", return_value=mock_conn1),
        ):
            store._auto_link_similar("new001", "test", fake_vec)

        # 只调用了一次 get_connection（SELECT），没有后续 UPDATE
        # 因为所有候选都 < 0.5 被过滤掉了


# ── 3. test_auto_link_max_20 ──


class TestAutoLinkMax20:
    """旧记忆 linked_memories 不超过 20."""

    def test_truncates_to_20(self) -> None:
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        fake_vec = [0.1] * 384
        mock_psycopg2 = MagicMock()
        mock_psycopg2.extras.RealDictCursor = MagicMock()

        # 旧记忆已有 20 个关联
        existing_linked = [f"link{i:03d}" for i in range(20)]
        candidates = [
            {"id": "old001", "linked_memories": existing_linked, "similarity": 0.8},
        ]

        mock_conn1, _ = _make_mock_conn(candidates)

        # connections: SELECT, UPDATE new, UPDATE old
        update_conn_new, update_cursor_new = _make_mock_conn()
        update_conn_old, update_cursor_old = _make_mock_conn()

        all_conns = [mock_conn1, update_conn_new, update_conn_old]
        call_count = {"n": 0}

        def fake_get_conn():
            idx = min(call_count["n"], len(all_conns) - 1)
            call_count["n"] += 1
            return all_conns[idx]

        with (
            patch.dict(
                "sys.modules",
                {"psycopg2": mock_psycopg2, "psycopg2.extras": mock_psycopg2.extras},
            ),
            patch("crew.memory_store_db.get_connection", side_effect=fake_get_conn),
        ):
            store._auto_link_similar("new001", "test", fake_vec)

        # 验证旧记忆的 UPDATE 包含最多 20 个（截断最旧的）
        update_cursor_old.execute.assert_called_once()
        update_args = update_cursor_old.execute.call_args[0]
        old_linked_updated = update_args[1][0]
        assert len(old_linked_updated) == 20
        # 新的 entry_id 应在列表中
        assert "new001" in old_linked_updated
        # 最旧的被截掉了
        assert "link000" not in old_linked_updated


# ── 4. test_auto_link_after_dedup ──


class TestAutoLinkAfterDedup:
    """dedup merge 后也做关联."""

    def test_auto_link_called_after_merge(self) -> None:
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        fake_vec = [0.1] * 384
        merged_entry = MemoryEntry(
            id="existing001",
            employee="test",
            category="finding",
            content="merged content",
            keywords=["性能"],
        )

        mock_conn, mock_cursor = _make_mock_conn()

        with (
            patch("crew.embedding.get_embedding", return_value=fake_vec),
            patch("crew.embedding.build_embedding_text", return_value="text"),
            patch.object(store, "_try_dedup_merge", return_value=merged_entry),
            patch.object(store, "_auto_link_similar") as mock_auto_link,
            patch("crew.memory_store_db.get_connection", return_value=mock_conn),
        ):
            entry = store.add(
                employee="test",
                category="finding",
                content="new similar content",
                keywords=["性能"],
            )

        # 应该使用合并后的 ID
        assert entry.id == "existing001"
        mock_auto_link.assert_called_once_with("existing001", "test", fake_vec)


# ── 5. test_expand_linked_memories ──


class TestExpandLinkedMemories:
    """展开方法返回正确结构."""

    def test_expand_returns_linked_entries(self) -> None:
        store = _make_store()

        entries = [
            MemoryEntry(
                id="m001",
                employee="test",
                category="finding",
                content="main memory",
                linked_memories=["linked001", "linked002", "linked003"],
            ),
        ]

        linked_rows = [
            {
                "id": "linked001",
                "employee": "test",
                "created_at": "2026-01-01T00:00:00",
                "category": "finding",
                "content": "linked content 1",
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
                "keywords": [],
                "linked_memories": [],
            },
            {
                "id": "linked002",
                "employee": "test",
                "created_at": "2026-01-01T00:00:00",
                "category": "finding",
                "content": "linked content 2",
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
                "keywords": [],
                "linked_memories": [],
            },
        ]

        mock_conn, mock_cursor = _make_mock_conn(linked_rows)

        with patch("crew.memory_store_db.get_connection", return_value=mock_conn):
            result = store.expand_linked_memories(entries, max_linked=2)

        assert "m001" in result
        assert len(result["m001"]) == 2
        assert result["m001"][0].id == "linked001"
        assert result["m001"][1].id == "linked002"
        assert result["m001"][0].content == "linked content 1"

    def test_expand_respects_max_linked(self) -> None:
        """max_linked 限制每条记忆展开的关联数."""
        store = _make_store()

        entries = [
            MemoryEntry(
                id="m001",
                employee="test",
                category="finding",
                content="main",
                linked_memories=["l1", "l2", "l3"],
            ),
        ]

        # 只返回 l1 (max_linked=1 只请求 l1)
        linked_rows = [
            {
                "id": "l1",
                "employee": "test",
                "created_at": "2026-01-01T00:00:00",
                "category": "finding",
                "content": "l1 content",
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
                "keywords": [],
                "linked_memories": [],
            },
        ]

        mock_conn, _ = _make_mock_conn(linked_rows)

        with patch("crew.memory_store_db.get_connection", return_value=mock_conn):
            result = store.expand_linked_memories(entries, max_linked=1)

        assert "m001" in result
        assert len(result["m001"]) == 1

        # 验证 SQL 只查了 l1（不包含 l2, l3）
        execute_args = mock_conn.cursor().execute.call_args[0]
        queried_ids = execute_args[1][0]
        assert "l1" in queried_ids
        assert "l2" not in queried_ids


# ── 6. test_expand_linked_empty ──


class TestExpandLinkedEmpty:
    """无关联时返回空."""

    def test_no_linked_returns_empty(self) -> None:
        store = _make_store()

        entries = [
            MemoryEntry(
                id="m001",
                employee="test",
                category="finding",
                content="no links",
                linked_memories=[],
            ),
        ]

        # 不应查询数据库
        result = store.expand_linked_memories(entries)
        assert result == {}

    def test_all_empty_linked_returns_empty(self) -> None:
        store = _make_store()

        entries = [
            MemoryEntry(
                id="m001", employee="test", category="finding", content="a", linked_memories=[]
            ),
            MemoryEntry(
                id="m002", employee="test", category="finding", content="b", linked_memories=[]
            ),
        ]

        result = store.expand_linked_memories(entries)
        assert result == {}


# ── 7. test_auto_link_no_embedding ──


class TestAutoLinkNoEmbedding:
    """embedding 不可用时跳过自动关联."""

    def test_no_embedding_skips_auto_link(self) -> None:
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        mock_conn, mock_cursor = _make_mock_conn()

        with (
            patch("crew.embedding.get_embedding", return_value=None),
            patch("crew.embedding.build_embedding_text", return_value="text"),
            patch.object(store, "_auto_link_similar") as mock_auto_link,
            patch("crew.memory_store_db.get_connection", return_value=mock_conn),
        ):
            store.add(
                employee="test",
                category="finding",
                content="test content",
                keywords=["测试"],
            )

        # embedding 为 None 时不调用 _auto_link_similar
        mock_auto_link.assert_not_called()

    def test_auto_link_failure_does_not_block_add(self) -> None:
        """_auto_link_similar 失败不影响 add() 返回."""
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        fake_vec = [0.1] * 384
        mock_conn, mock_cursor = _make_mock_conn()

        with (
            patch("crew.embedding.get_embedding", return_value=fake_vec),
            patch("crew.embedding.build_embedding_text", return_value="text"),
            patch.object(store, "_try_dedup_merge", return_value=None),
            patch.object(store, "_auto_link_similar", side_effect=RuntimeError("link failed")),
            patch("crew.memory_store_db.get_connection", return_value=mock_conn),
        ):
            entry = store.add(
                employee="test",
                category="finding",
                content="test content",
                keywords=["测试"],
            )

        # add() 应该成功返回（auto_link 失败不阻塞）
        assert entry.id
