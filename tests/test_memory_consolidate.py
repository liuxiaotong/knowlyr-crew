"""Tests for memory consolidation (Phase 4: 碎片聚合)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from crew.memory import MemoryEntry
from crew.memory_consolidate import (
    find_clusters,
    run_consolidation,
    synthesize_cluster,
)

# ── Fixtures ──


def _make_entry(
    id: str = "abc123",
    employee: str = "测试员工",
    category: str = "finding",
    content: str = "测试内容",
    keywords: list[str] | None = None,
    linked_memories: list[str] | None = None,
    importance: int = 3,
    **kwargs,
) -> MemoryEntry:
    """创建测试用 MemoryEntry."""
    return MemoryEntry(
        id=id,
        employee=employee,
        category=category,
        content=content,
        keywords=keywords or [],
        linked_memories=linked_memories or [],
        importance=importance,
        **kwargs,
    )


def _make_mock_store() -> MagicMock:
    """创建 mock MemoryStoreDB."""
    store = MagicMock()
    store._tenant_id = "test-tenant"
    store._resolve_to_character_name = MagicMock(side_effect=lambda x: x)

    # 默认 add 返回新 entry
    def mock_add(employee, category, content, tags=None, **kw):
        return _make_entry(
            id="new_pattern_1",
            employee=employee,
            category=category,
            content=content,
        )

    store.add = MagicMock(side_effect=mock_add)
    store.update = MagicMock(return_value=True)
    store.update_keywords = MagicMock(return_value=True)
    store.update_linked_memories = MagicMock(return_value=True)
    store.list_employees = MagicMock(return_value=["测试员工"])
    return store


# ── Test: find_clusters ──


class TestFindClusters:
    def test_find_clusters_groups_overlapping_findings(self):
        """5 条有重叠关键词的 findings 分成正确的 cluster."""
        findings = [
            _make_entry(id="f1", keywords=["API", "限流", "阈值"]),
            _make_entry(id="f2", keywords=["API", "限流", "超时"]),
            _make_entry(id="f3", keywords=["API", "限流", "监控"]),
            _make_entry(id="f4", keywords=["数据库", "迁移", "PostgreSQL"]),
            _make_entry(id="f5", keywords=["API", "限流", "告警"]),
        ]
        store = _make_mock_store()
        store.query = MagicMock(return_value=findings)

        clusters = find_clusters("测试员工", store, min_cluster_size=3)

        # API 限流相关的 4 条应聚成一个 cluster
        assert len(clusters) >= 1
        api_cluster = max(clusters, key=len)
        api_ids = {e.id for e in api_cluster}
        assert "f1" in api_ids
        assert "f2" in api_ids
        assert "f3" in api_ids
        assert "f5" in api_ids
        # 数据库相关的不应在 API cluster 中
        assert "f4" not in api_ids

    def test_find_clusters_ignores_non_findings(self):
        """pattern/decision 不参与聚类（query 已按 category=finding 过滤）."""
        # find_clusters 内部调 store.query(category="finding")
        # 所以即使 store 中有 pattern，也不会被查出来
        findings = [
            _make_entry(id="f1", category="finding", keywords=["API", "限流"]),
            _make_entry(id="f2", category="finding", keywords=["API", "限流"]),
        ]
        store = _make_mock_store()
        store.query = MagicMock(return_value=findings)

        clusters = find_clusters("测试员工", store, min_cluster_size=3)

        # 只有 2 条，不满足 min_cluster_size=3
        assert len(clusters) == 0

        # 验证 query 被正确调用（category=finding）
        call_kwargs = store.query.call_args
        assert call_kwargs[1]["category"] == "finding"

    def test_find_clusters_min_size(self):
        """少于 min_cluster_size 的不返回."""
        findings = [
            _make_entry(id="f1", keywords=["API", "限流", "阈值"]),
            _make_entry(id="f2", keywords=["API", "限流", "超时"]),
            _make_entry(id="f3", keywords=["数据库", "迁移", "PostgreSQL"]),
        ]
        store = _make_mock_store()
        store.query = MagicMock(return_value=findings)

        # min_cluster_size=3，API 相关只有 2 条，不够
        clusters = find_clusters("测试员工", store, min_cluster_size=3)
        assert len(clusters) == 0

        # min_cluster_size=2，API 相关 2 条可以成 cluster
        clusters = find_clusters("测试员工", store, min_cluster_size=2)
        assert len(clusters) >= 1


# ── Test: synthesize_cluster ──


class TestSynthesizeCluster:
    @patch("crew.memory_consolidate._mark_superseded", return_value=3)
    @patch("crew.memory_consolidate._update_importance")
    @patch("crew.memory_consolidate._call_llm")
    def test_synthesize_cluster_creates_pattern(self, mock_llm, mock_importance, mock_superseded):
        """mock LLM，验证新 pattern 被创建 + 原 findings 被 superseded."""
        mock_llm.return_value = json.dumps(
            {
                "content": "API 限流策略：阈值 100 req/min，超时后降级，需告警监控",
                "keywords": ["API", "限流", "策略"],
                "tags": ["infra", "API"],
            }
        )

        entries = [
            _make_entry(id="f1", content="API 限流阈值为 100", keywords=["API", "限流"]),
            _make_entry(id="f2", content="限流超时后降级处理", keywords=["API", "超时"]),
            _make_entry(id="f3", content="需要限流告警", keywords=["API", "告警"]),
        ]
        store = _make_mock_store()

        result = synthesize_cluster(entries, "测试员工", store)

        # 验证返回新 pattern
        assert result is not None
        assert result.importance == 4
        assert result.linked_memories == ["f1", "f2", "f3"]

        # 验证 store.add 被调用创建 pattern
        store.add.assert_called_once()
        add_kwargs = store.add.call_args[1]
        assert add_kwargs["category"] == "pattern"
        assert "限流" in add_kwargs["content"]

        # 验证 superseded 被调用
        mock_superseded.assert_called_once()
        call_args = mock_superseded.call_args
        assert set(call_args[0][0]) == {"f1", "f2", "f3"}

        # 验证 linked_memories 被更新
        store.update_linked_memories.assert_called_once()

    @patch("crew.memory_consolidate._mark_superseded")
    @patch("crew.memory_consolidate._update_importance")
    @patch("crew.memory_consolidate._call_llm")
    def test_synthesize_cluster_llm_failure(self, mock_llm, mock_importance, mock_superseded):
        """LLM 失败时返回 None，不影响原数据."""
        mock_llm.side_effect = RuntimeError("API error")

        entries = [
            _make_entry(id="f1", keywords=["API"]),
            _make_entry(id="f2", keywords=["API"]),
            _make_entry(id="f3", keywords=["API"]),
        ]
        store = _make_mock_store()

        result = synthesize_cluster(entries, "测试员工", store)

        assert result is None
        # 验证没有写库
        store.add.assert_not_called()
        mock_superseded.assert_not_called()


# ── Test: run_consolidation ──


class TestRunConsolidation:
    @patch("crew.memory_consolidate.synthesize_cluster")
    @patch("crew.memory_consolidate.find_clusters")
    def test_run_consolidation_dry_run(self, mock_find, mock_synth):
        """dry_run 模式不写库."""
        cluster = [
            _make_entry(id="f1", keywords=["API"]),
            _make_entry(id="f2", keywords=["API"]),
            _make_entry(id="f3", keywords=["API"]),
        ]
        mock_find.return_value = [cluster]
        store = _make_mock_store()

        result = run_consolidation(store=store, dry_run=True)

        assert result["employees_processed"] == 1
        assert result["clusters_found"] == 1
        assert result["patterns_created"] == 1
        assert result["findings_superseded"] == 3
        # dry_run 不调 synthesize
        mock_synth.assert_not_called()

    @patch("crew.memory_consolidate.synthesize_cluster")
    @patch("crew.memory_consolidate.find_clusters")
    def test_run_consolidation_full(self, mock_find, mock_synth):
        """完整流程冒烟测试."""
        cluster = [
            _make_entry(id="f1", keywords=["API"]),
            _make_entry(id="f2", keywords=["API"]),
            _make_entry(id="f3", keywords=["API"]),
        ]
        mock_find.return_value = [cluster]
        mock_synth.return_value = _make_entry(
            id="new_pattern",
            category="pattern",
            content="合成的 pattern",
            importance=4,
            linked_memories=["f1", "f2", "f3"],
        )
        store = _make_mock_store()

        result = run_consolidation(store=store, dry_run=False)

        assert result["employees_processed"] == 1
        assert result["clusters_found"] == 1
        assert result["patterns_created"] == 1
        assert result["findings_superseded"] == 3
        # 非 dry_run 调了 synthesize
        mock_synth.assert_called_once()

    @patch("crew.memory_consolidate.synthesize_cluster")
    @patch("crew.memory_consolidate.find_clusters")
    def test_run_consolidation_single_employee(self, mock_find, mock_synth):
        """指定单个员工."""
        mock_find.return_value = []
        store = _make_mock_store()

        result = run_consolidation(store=store, employee="赵云帆")

        assert result["employees_processed"] == 1
        # 指定了 employee 时不调 list_employees
        store.list_employees.assert_not_called()
        mock_find.assert_called_once_with("赵云帆", store)

    @patch("crew.memory_consolidate.synthesize_cluster")
    @patch("crew.memory_consolidate.find_clusters")
    def test_run_consolidation_synth_failure(self, mock_find, mock_synth):
        """synthesize 失败时 patterns_created 不增加."""
        cluster = [
            _make_entry(id="f1", keywords=["API"]),
            _make_entry(id="f2", keywords=["API"]),
            _make_entry(id="f3", keywords=["API"]),
        ]
        mock_find.return_value = [cluster]
        mock_synth.return_value = None  # 合成失败
        store = _make_mock_store()

        result = run_consolidation(store=store, dry_run=False)

        assert result["clusters_found"] == 1
        assert result["patterns_created"] == 0
        assert result["findings_superseded"] == 0
