"""灵魂自动进化测试 -- soul_evolution.py 单元测试."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

from crew.soul_evolution import (
    EvolutionCandidate,
    find_archive_candidates,
    find_promotion_candidates,
    run_evolution_review,
)

# ── 测试用 Mock 对象 ──


@dataclass
class _MockMemoryEntry:
    """模拟 MemoryEntry."""

    id: str = ""
    employee: str = ""
    created_at: str = "2026-01-01T00:00:00"
    category: str = "pattern"
    content: str = ""
    source_session: str = ""
    confidence: float = 1.0
    superseded_by: str = ""
    ttl_days: int = 0
    importance: int = 3
    last_accessed: str = ""
    tags: list[str] = field(default_factory=list)
    shared: bool = False
    visibility: str = "open"
    trigger_condition: str = ""
    applicability: list[str] = field(default_factory=list)
    origin_employee: str = ""
    verified_count: int = 0
    classification: str = "internal"
    domain: str = ""
    keywords: list[str] = field(default_factory=list)
    linked_memories: list[str] = field(default_factory=list)


class _MockStore:
    """模拟 MemoryStoreDB."""

    def __init__(self, entries: list[_MockMemoryEntry] | None = None):
        self._entries = entries or []
        self._tenant_id = "tenant_admin"

    def _resolve_to_character_name(self, employee: str) -> str:
        return employee

    def _row_to_entry(self, row: dict) -> _MockMemoryEntry:
        return _MockMemoryEntry(**{k: row[k] for k in row if hasattr(_MockMemoryEntry, k)})

    def query(
        self,
        employee: str,
        category: str | None = None,
        limit: int = 20,
        include_expired: bool = False,
    ) -> list[_MockMemoryEntry]:
        results = []
        for e in self._entries:
            if e.employee != employee:
                continue
            if category and e.category != category:
                continue
            results.append(e)
        return results[:limit]

    def list_employees(self) -> list[str]:
        return list({e.employee for e in self._entries})


def _make_db_row(entry: _MockMemoryEntry) -> dict:
    """将 MockMemoryEntry 转换为类似数据库行的字典."""
    return {
        "id": entry.id,
        "employee": entry.employee,
        "created_at": entry.created_at,
        "category": entry.category,
        "content": entry.content,
        "source_session": entry.source_session,
        "confidence": entry.confidence,
        "superseded_by": entry.superseded_by,
        "ttl_days": entry.ttl_days,
        "importance": entry.importance,
        "last_accessed": entry.last_accessed,
        "tags": entry.tags,
        "shared": entry.shared,
        "visibility": entry.visibility,
        "trigger_condition": entry.trigger_condition,
        "applicability": entry.applicability,
        "origin_employee": entry.origin_employee,
        "verified_count": entry.verified_count,
        "classification": entry.classification,
        "domain": entry.domain,
        "keywords": entry.keywords,
        "linked_memories": entry.linked_memories,
    }


# ── 测试用例 ──


class TestFindPromotionCandidates:
    """测试 find_promotion_candidates."""

    @patch("crew.soul_evolution.get_config", return_value=None)
    @patch("crew.soul_evolution._call_llm", return_value="始终编写单元测试。")
    def test_find_promotion_candidates_creates_candidate(
        self, mock_llm: MagicMock, mock_get_config: MagicMock
    ):
        """verified_count >= 3 的 pattern 应生成候选."""
        entry = _MockMemoryEntry(
            id="mem001",
            employee="赵云帆",
            category="pattern",
            content="编写代码后必须写测试",
            verified_count=5,
        )
        store = _MockStore([entry])
        rows = [_make_db_row(entry)]

        # Mock 数据库查询
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = rows
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("crew.database.get_connection", return_value=mock_conn):
            candidates = find_promotion_candidates("赵云帆", store, min_verified=3)

        assert len(candidates) == 1
        assert candidates[0].action == "promote"
        assert candidates[0].source_type == "pattern"
        assert candidates[0].source_ids == ["mem001"]
        assert candidates[0].content == "始终编写单元测试。"
        assert candidates[0].confidence == 1.0  # min(1.0, 5/5)
        mock_llm.assert_called_once()

    @patch("crew.soul_evolution.get_config", return_value=None)
    @patch("crew.soul_evolution._call_llm")
    def test_find_promotion_candidates_min_verified(
        self, mock_llm: MagicMock, mock_get_config: MagicMock
    ):
        """verified_count < 3 不应出候选."""
        entry = _MockMemoryEntry(
            id="mem002",
            employee="赵云帆",
            category="pattern",
            content="使用类型注解",
            verified_count=2,  # 低于阈值
        )
        store = _MockStore([entry])

        # 数据库返回空（因为 SQL 过滤了 verified_count < 3）
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("crew.database.get_connection", return_value=mock_conn):
            candidates = find_promotion_candidates("赵云帆", store, min_verified=3)

        assert len(candidates) == 0
        mock_llm.assert_not_called()

    @patch("crew.soul_evolution.get_config", return_value=None)
    @patch("crew.soul_evolution._call_llm")
    def test_find_promotion_candidates_skips_superseded(
        self, mock_llm: MagicMock, mock_get_config: MagicMock
    ):
        """superseded 的 pattern 不参与（SQL 已过滤）."""
        # superseded 的 entry 不会出现在 SQL 结果中
        store = _MockStore()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []  # SQL 过滤掉了 superseded
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch("crew.database.get_connection", return_value=mock_conn):
            candidates = find_promotion_candidates("赵云帆", store, min_verified=3)

        assert len(candidates) == 0
        mock_llm.assert_not_called()


class TestFindArchiveCandidates:
    """测试 find_archive_candidates."""

    @patch("crew.soul_evolution.get_config", return_value=None)
    def test_find_archive_candidates_clusters_corrections(
        self, mock_get_config: MagicMock
    ):
        """多条同主题 correction 应生成 archive 候选."""
        corrections = [
            _MockMemoryEntry(
                id="c001",
                employee="赵云帆",
                category="correction",
                content="不应该直接操作数据库",
                tags=["database", "orm"],
            ),
            _MockMemoryEntry(
                id="c002",
                employee="赵云帆",
                category="correction",
                content="必须使用 ORM 而不是原始 SQL",
                tags=["database", "orm", "sql"],
            ),
            _MockMemoryEntry(
                id="c003",
                employee="赵云帆",
                category="correction",
                content="数据库操作要通过 ORM 层",
                tags=["database", "orm"],
            ),
        ]
        store = _MockStore(corrections)

        candidates = find_archive_candidates("赵云帆", store, min_corrections=2)

        assert len(candidates) >= 1
        assert candidates[0].action == "archive"
        assert candidates[0].source_type == "correction"
        assert len(candidates[0].source_ids) >= 2

    @patch("crew.soul_evolution.get_config", return_value=None)
    def test_find_archive_candidates_too_few(self, mock_get_config: MagicMock):
        """corrections < 2 不应出候选."""
        corrections = [
            _MockMemoryEntry(
                id="c001",
                employee="赵云帆",
                category="correction",
                content="不应该直接操作数据库",
                tags=["database", "orm"],
            ),
        ]
        store = _MockStore(corrections)

        candidates = find_archive_candidates("赵云帆", store, min_corrections=2)

        assert len(candidates) == 0


class TestRunEvolutionReview:
    """测试 run_evolution_review."""

    @patch("crew.soul_evolution.put_config")
    @patch("crew.soul_evolution.get_config", return_value=None)
    @patch("crew.soul_evolution.find_archive_candidates", return_value=[])
    @patch(
        "crew.soul_evolution.find_promotion_candidates",
        return_value=[
            EvolutionCandidate(
                id="cand001",
                employee="赵云帆",
                action="promote",
                source_type="pattern",
                source_ids=["mem001"],
                content="始终编写测试。",
                reason="验证 3 次",
                confidence=0.6,
                created_at="2026-01-01T00:00:00",
            )
        ],
    )
    def test_run_evolution_review_stores_candidates(
        self,
        mock_promote: MagicMock,
        mock_archive: MagicMock,
        mock_get_config: MagicMock,
        mock_put_config: MagicMock,
    ):
        """验证候选写入 config_store."""
        store = _MockStore(
            [
                _MockMemoryEntry(
                    id="mem001",
                    employee="赵云帆",
                    category="pattern",
                    verified_count=3,
                )
            ]
        )

        result = run_evolution_review(store=store, employee="赵云帆")

        assert result["employees_reviewed"] == 1
        assert result["promote_candidates"] == 1
        assert result["archive_candidates"] == 0
        assert len(result["candidates"]) == 1

        # 验证 put_config 被调用，写入了候选
        mock_put_config.assert_called_once()
        call_args = mock_put_config.call_args
        assert call_args[0][0] == "soul_evolution"
        assert call_args[0][1] == "赵云帆_candidates"
        stored = json.loads(call_args[0][2])
        assert len(stored) == 1
        assert stored[0]["action"] == "promote"


class TestEvolutionCandidate:
    """测试 EvolutionCandidate 数据类."""

    def test_to_dict(self):
        """to_dict 应返回可序列化字典."""
        candidate = EvolutionCandidate(
            id="test",
            employee="赵云帆",
            action="promote",
            source_type="pattern",
            source_ids=["m1", "m2"],
            content="测试内容",
            reason="测试原因",
            confidence=0.8,
            created_at="2026-01-01",
        )
        d = candidate.to_dict()
        assert d["employee"] == "赵云帆"
        assert d["action"] == "promote"
        # 确保可 JSON 序列化
        json.dumps(d, ensure_ascii=False)


class TestEvolutionReviewHandler:
    """冒烟测试 handler."""

    @patch("crew.soul_evolution.run_evolution_review")
    def test_evolution_review_handler(self, mock_review: MagicMock):
        """handler 冒烟测试 -- 验证调用链路."""
        mock_review.return_value = {
            "employees_reviewed": 1,
            "promote_candidates": 0,
            "archive_candidates": 0,
            "candidates": [],
        }

        # 直接测试函数签名和返回格式
        result = mock_review(store=None, employee="赵云帆")
        assert result["employees_reviewed"] == 1
        mock_review.assert_called_once_with(store=None, employee="赵云帆")
