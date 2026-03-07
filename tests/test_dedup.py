"""tests/test_dedup.py — NG-2 语义去重测试.

覆盖：
1. add() 轻量去重（高相似合并 / 低相似新建 / 不同category / correction跳过 / 无embedding）
2. connect() 语义合并 / 语义关联 / 降级关键词

所有测试 mock 数据库和 embedding，不依赖 pgvector / sentence-transformers。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from crew.memory import MemoryEntry
from crew.memory_pipeline import ReflectResult

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


def _make_mock_row(
    entry_id: str = "existing001",
    content: str = "existing content",
    category: str = "finding",
    keywords: list | None = None,
    employee: str = "test",
    **overrides,
) -> dict:
    """创建模拟数据库行."""
    row = {
        "id": entry_id,
        "employee": employee,
        "created_at": "2026-01-01T00:00:00",
        "category": category,
        "content": content,
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
        "origin_employee": employee,
        "verified_count": 0,
        "classification": "internal",
        "domain": [],
        "keywords": keywords or ["性能"],
        "linked_memories": [],
        "embedding": [0.1] * 384,
    }
    row.update(overrides)
    return row


def _mock_conn_with_rows(rows_sequence: list[list[dict] | dict | None]):
    """创建带有多次 fetchone/fetchall 调用结果的 mock connection.

    rows_sequence: 每次 cursor.execute 后的 fetchone/fetchall 返回值列表
    """
    mock_cursor = MagicMock()
    # 为每次 execute 设置不同的返回值
    fetch_one_results = []
    fetch_all_results = []
    for item in rows_sequence:
        if item is None:
            fetch_one_results.append(None)
            fetch_all_results.append([])
        elif isinstance(item, list):
            fetch_all_results.append(item)
            fetch_one_results.append(item[0] if item else None)
        else:
            fetch_one_results.append(item)
            fetch_all_results.append([item])

    mock_cursor.fetchone = MagicMock(side_effect=fetch_one_results)
    mock_cursor.fetchall = MagicMock(side_effect=fetch_all_results)
    mock_cursor.rowcount = 1

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    return mock_conn


# ── 1. add() 去重测试 ──


class TestAddDedupHighSimilarity:
    """test_add_dedup_high_similarity — 相似度 >= 0.90 触发合并."""

    def test_merge_triggered(self) -> None:
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        fake_vec = [0.1] * 384

        # 直接 mock _try_dedup_merge 返回合并结果
        merged_entry = MemoryEntry(
            id="existing001",
            employee="test",
            category="finding",
            content="existing content\n---\nnew similar content",
            keywords=["性能"],
        )

        mock_conn_insert = MagicMock()
        mock_cursor_insert = MagicMock()
        mock_conn_insert.cursor.return_value = mock_cursor_insert
        mock_conn_insert.__enter__ = lambda s: s
        mock_conn_insert.__exit__ = MagicMock(return_value=False)

        with (
            patch("crew.embedding.get_embedding", return_value=fake_vec),
            patch("crew.embedding.build_embedding_text", return_value="text"),
            patch.object(store, "_try_dedup_merge", return_value=merged_entry),
            patch("crew.memory_store_db.get_connection", return_value=mock_conn_insert),
        ):
            entry = store.add(
                employee="test",
                category="finding",
                content="new similar content",
                keywords=["性能"],
            )

        # 应该返回已有记忆的 ID（合并到旧记忆）
        assert entry.id == "existing001"
        assert "existing content" in entry.content
        assert "new similar content" in entry.content
        # INSERT 不应被调用（合并直接返回）
        mock_cursor_insert.execute.assert_not_called()


class TestAddDedupLowSimilarity:
    """test_add_dedup_low_similarity — 相似度 < 0.90 正常新建."""

    def test_insert_when_low_similarity(self) -> None:
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        fake_vec = [0.1] * 384

        # _try_dedup_merge 返回 None → 正常新建
        mock_conn_insert = MagicMock()
        mock_cursor_insert = MagicMock()
        mock_conn_insert.cursor.return_value = mock_cursor_insert
        mock_conn_insert.__enter__ = lambda s: s
        mock_conn_insert.__exit__ = MagicMock(return_value=False)

        with (
            patch("crew.embedding.get_embedding", return_value=fake_vec),
            patch("crew.embedding.build_embedding_text", return_value="text"),
            patch.object(store, "_try_dedup_merge", return_value=None),
            patch("crew.memory_store_db.get_connection", return_value=mock_conn_insert),
        ):
            entry = store.add(
                employee="test",
                category="finding",
                content="different content",
                keywords=["不同"],
            )

        # 应该新建（ID 不是已有的）
        assert entry.id != "existing001"
        # INSERT 应该被调用
        mock_cursor_insert.execute.assert_called_once()


class TestAddDedupDifferentCategory:
    """test_add_dedup_different_category — 不同 category 不合并."""

    def test_no_merge_on_category_mismatch(self) -> None:
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        fake_vec = [0.1] * 384

        # _try_dedup_merge 返回 None（category 不匹配）
        mock_conn_insert = MagicMock()
        mock_cursor_insert = MagicMock()
        mock_conn_insert.cursor.return_value = mock_cursor_insert
        mock_conn_insert.__enter__ = lambda s: s
        mock_conn_insert.__exit__ = MagicMock(return_value=False)

        with (
            patch("crew.embedding.get_embedding", return_value=fake_vec),
            patch("crew.embedding.build_embedding_text", return_value="text"),
            patch.object(store, "_try_dedup_merge", return_value=None),
            patch("crew.memory_store_db.get_connection", return_value=mock_conn_insert),
        ):
            entry = store.add(
                employee="test",
                category="finding",
                content="finding content",
                keywords=["性能"],
            )

        # 不应合并
        assert entry.id != "existing001"


class TestAddDedupCorrectionSkipped:
    """test_add_dedup_correction_skipped — correction 类型不去重."""

    def test_correction_always_inserts(self) -> None:
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        fake_vec = [0.1] * 384

        # 只需要一个 INSERT connection（correction 跳过去重）
        mock_conn_insert = MagicMock()
        mock_cursor_insert = MagicMock()
        mock_conn_insert.cursor.return_value = mock_cursor_insert
        mock_conn_insert.__enter__ = lambda s: s
        mock_conn_insert.__exit__ = MagicMock(return_value=False)

        with (
            patch("crew.embedding.get_embedding", return_value=fake_vec),
            patch("crew.embedding.build_embedding_text", return_value="text"),
            patch("crew.memory_store_db.get_connection", return_value=mock_conn_insert),
        ):
            entry = store.add(
                employee="test",
                category="correction",
                content="correction content",
                keywords=["纠正"],
            )

        # correction 直接插入，不查去重
        assert entry.id  # 新 ID
        mock_cursor_insert.execute.assert_called_once()


class TestAddDedupNoEmbedding:
    """test_add_dedup_no_embedding — embedding 不可用时正常新建."""

    def test_no_embedding_skips_dedup(self) -> None:
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        mock_conn_insert = MagicMock()
        mock_cursor_insert = MagicMock()
        mock_conn_insert.cursor.return_value = mock_cursor_insert
        mock_conn_insert.__enter__ = lambda s: s
        mock_conn_insert.__exit__ = MagicMock(return_value=False)

        with (
            patch("crew.embedding.get_embedding", return_value=None),
            patch("crew.embedding.build_embedding_text", return_value="text"),
            patch("crew.memory_store_db.get_connection", return_value=mock_conn_insert),
        ):
            entry = store.add(
                employee="test",
                category="finding",
                content="test content",
                keywords=["测试"],
            )

        assert entry.id
        # 直接 INSERT，embedding=None
        mock_cursor_insert.execute.assert_called_once()
        insert_params = mock_cursor_insert.execute.call_args[0][1]
        assert insert_params[-1] is None  # embedding_vector


# ── 1b. _try_dedup_merge 直接测试 ──


class TestTryDedupMerge:
    """直接测试 _try_dedup_merge 方法的逻辑."""

    def test_merge_on_high_similarity(self) -> None:
        """cosine similarity >= 0.90 且同 category 触发合并."""
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        existing_row = _make_mock_row()
        dist_row = (0.05,)  # similarity = 0.95

        mock_psycopg2 = MagicMock()
        mock_psycopg2.extras.RealDictCursor = MagicMock()

        # 3 次 get_connection: 查候选、查距离、UPDATE
        mock_conn1 = _mock_conn_with_rows([existing_row])
        mock_conn1.cursor.return_value = MagicMock(
            fetchone=MagicMock(return_value=existing_row),
            fetchall=MagicMock(return_value=[existing_row]),
        )
        mock_conn2 = _mock_conn_with_rows([dist_row])
        mock_conn3 = _mock_conn_with_rows([])

        call_count = {"n": 0}
        connections = [mock_conn1, mock_conn2, mock_conn3]

        def fake_get_conn():
            idx = min(call_count["n"], len(connections) - 1)
            call_count["n"] += 1
            return connections[idx]

        with (
            patch.dict(
                "sys.modules",
                {
                    "psycopg2": mock_psycopg2,
                    "psycopg2.extras": mock_psycopg2.extras,
                },
            ),
            patch("crew.memory_store_db.get_connection", side_effect=fake_get_conn),
            patch("crew.embedding.get_embedding", return_value=[0.2] * 384),
            patch("crew.embedding.build_embedding_text", return_value="merged text"),
        ):
            result = store._try_dedup_merge(
                "test",
                [0.1] * 384,
                "new content",
                ["性能"],
                "finding",
            )

        assert result is not None
        assert result.id == "existing001"
        assert "new content" in result.content

    def test_no_merge_on_low_similarity(self) -> None:
        """cosine similarity < 0.90 返回 None."""
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        existing_row = _make_mock_row()
        dist_row = (0.5,)  # similarity = 0.5

        mock_psycopg2 = MagicMock()
        mock_psycopg2.extras.RealDictCursor = MagicMock()

        mock_conn1 = _mock_conn_with_rows([existing_row])
        mock_conn1.cursor.return_value = MagicMock(
            fetchone=MagicMock(return_value=existing_row),
        )
        mock_conn2 = _mock_conn_with_rows([dist_row])

        call_count = {"n": 0}
        connections = [mock_conn1, mock_conn2]

        def fake_get_conn():
            idx = min(call_count["n"], len(connections) - 1)
            call_count["n"] += 1
            return connections[idx]

        with (
            patch.dict(
                "sys.modules",
                {
                    "psycopg2": mock_psycopg2,
                    "psycopg2.extras": mock_psycopg2.extras,
                },
            ),
            patch("crew.memory_store_db.get_connection", side_effect=fake_get_conn),
        ):
            result = store._try_dedup_merge(
                "test",
                [0.1] * 384,
                "new content",
                ["性能"],
                "finding",
            )

        assert result is None

    def test_no_merge_on_category_mismatch(self) -> None:
        """同相似度但不同 category 返回 None."""
        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        existing_row = _make_mock_row(category="decision")
        dist_row = (0.05,)  # similarity = 0.95 但 category 不同

        mock_psycopg2 = MagicMock()
        mock_psycopg2.extras.RealDictCursor = MagicMock()

        mock_conn1 = _mock_conn_with_rows([existing_row])
        mock_conn1.cursor.return_value = MagicMock(
            fetchone=MagicMock(return_value=existing_row),
        )
        mock_conn2 = _mock_conn_with_rows([dist_row])

        call_count = {"n": 0}
        connections = [mock_conn1, mock_conn2]

        def fake_get_conn():
            idx = min(call_count["n"], len(connections) - 1)
            call_count["n"] += 1
            return connections[idx]

        with (
            patch.dict(
                "sys.modules",
                {
                    "psycopg2": mock_psycopg2,
                    "psycopg2.extras": mock_psycopg2.extras,
                },
            ),
            patch("crew.memory_store_db.get_connection", side_effect=fake_get_conn),
        ):
            result = store._try_dedup_merge(
                "test",
                [0.1] * 384,
                "new content",
                ["性能"],
                "finding",
            )

        assert result is None


# ── 2. connect() 语义测试 ──


class TestConnectSemanticMerge:
    """test_connect_semantic_merge — connect() 语义合并."""

    def test_merge_on_high_similarity(self) -> None:
        note = ReflectResult(
            store=True,
            content="优化数据库查询性能",
            category="finding",
            keywords=["数据库", "性能"],
            tags=["backend"],
        )

        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        existing_entry = MemoryEntry(
            id="sem001",
            employee="test",
            category="finding",
            content="数据库查询优化方法",
            keywords=["数据库", "查询"],
            linked_memories=[],
        )

        # mock _find_candidates_by_semantic 返回高相似度候选
        with (
            patch(
                "crew.memory_pipeline._find_candidates_by_semantic",
                return_value=[(existing_entry, 0.90)],
            ),
            patch.object(store, "update", return_value=True),
            patch.object(store, "update_keywords", return_value=True),
            # mock embedding 更新
            patch("crew.embedding.get_embedding", return_value=[0.2] * 384),
            patch("crew.embedding.build_embedding_text", return_value="text"),
            patch("crew.memory_pipeline.get_connection")
            if False
            else patch(
                "crew.database.get_connection",
                return_value=MagicMock(
                    __enter__=lambda s: s,
                    __exit__=MagicMock(return_value=False),
                    cursor=MagicMock(return_value=MagicMock()),
                ),
            ),
        ):
            from crew.memory_pipeline import connect

            result = connect(note, "test", store)

        assert result.action == "merge"
        assert result.merged_entry_id == "sem001"
        assert "数据库查询优化方法" in result.entry.content
        assert "优化数据库查询性能" in result.entry.content


class TestConnectSemanticLink:
    """test_connect_semantic_link — connect() 语义关联."""

    def test_link_on_medium_similarity(self) -> None:
        note = ReflectResult(
            store=True,
            content="前端性能优化",
            category="finding",
            keywords=["前端", "性能"],
            tags=[],
        )

        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        existing_entry = MemoryEntry(
            id="sem002",
            employee="test",
            category="finding",
            content="后端性能调优",
            keywords=["后端", "性能"],
            linked_memories=[],
        )

        new_entry = MemoryEntry(
            id="new001",
            employee="test",
            category="finding",
            content="前端性能优化",
            keywords=["前端", "性能"],
            linked_memories=[],
        )

        with (
            patch(
                "crew.memory_pipeline._find_candidates_by_semantic",
                return_value=[(existing_entry, 0.6)],
            ),
            patch(
                "crew.memory_pipeline._store_new",
                return_value=new_entry,
            ),
            patch.object(store, "update_linked_memories", return_value=True),
            patch.object(store, "update_keywords", return_value=True),
        ):
            from crew.memory_pipeline import connect

            result = connect(note, "test", store)

        assert result.action == "link"
        assert "sem002" in result.entry.linked_memories


class TestConnectFallbackKeywords:
    """test_connect_fallback_keywords — embedding 不可用时降级关键词."""

    def test_fallback_to_keyword_matching(self) -> None:
        note = ReflectResult(
            store=True,
            content="数据库优化",
            category="finding",
            keywords=["数据库", "优化", "索引"],
            tags=[],
        )

        store = _make_store()
        store._resolve_to_character_name = lambda x: x

        # 语义查找返回空（不可用）
        existing_entry = MemoryEntry(
            id="kw001",
            employee="test",
            category="finding",
            content="数据库索引优化",
            keywords=["数据库", "优化", "索引"],
            linked_memories=[],
        )

        with (
            patch(
                "crew.memory_pipeline._find_candidates_by_semantic",
                return_value=[],  # 语义不可用
            ),
            patch(
                "crew.memory_pipeline._find_candidates_by_keywords",
                return_value=[existing_entry],
            ),
            patch.object(store, "update", return_value=True),
            patch.object(store, "update_keywords", return_value=True),
        ):
            from crew.memory_pipeline import connect

            result = connect(note, "test", store)

        # 关键词 Jaccard = 3/3 = 1.0 >= 0.7 且同 category → merge
        assert result.action == "merge"
        assert result.merged_entry_id == "kw001"
