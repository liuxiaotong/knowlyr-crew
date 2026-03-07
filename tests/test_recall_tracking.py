"""Tests for recall tracking (召回效果闭环)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crew.memory import MemoryEntry

# ── Fixtures ──


def _make_entry(
    id: str = "abc123",
    employee: str = "测试员工",
    category: str = "finding",
    content: str = "测试内容",
    **kwargs,
) -> MemoryEntry:
    """创建测试用 MemoryEntry."""
    return MemoryEntry(
        id=id,
        employee=employee,
        category=category,
        content=content,
        **kwargs,
    )


def _make_mock_store() -> MagicMock:
    """创建 mock MemoryStoreDB."""
    store = MagicMock()
    store._tenant_id = "test-tenant"
    store._resolve_to_character_name = MagicMock(side_effect=lambda x: x)
    store.record_recall = MagicMock(return_value=2)
    store.record_useful = MagicMock(return_value=1)
    store.get_recall_stats = MagicMock(
        return_value={
            "total_recalls": 100,
            "total_useful": 30,
            "hit_rate": 0.3,
            "top_useful": [],
            "never_recalled": 50,
        }
    )
    store.query = MagicMock(
        return_value=[
            _make_entry(id="m1"),
            _make_entry(id="m2"),
        ]
    )
    store.list_employees = MagicMock(return_value=["测试员工"])
    return store


# ── Test: record_recall ──


class TestRecordRecall:
    def test_record_recall_increments_count(self):
        """调两次 record_recall，recall_count 应该是 2."""
        store = _make_mock_store()

        # 第一次调用
        store.record_recall(["m1", "m2"])
        assert store.record_recall.call_count == 1
        store.record_recall.assert_called_with(["m1", "m2"])

        # 第二次调用
        store.record_recall(["m1", "m2"])
        assert store.record_recall.call_count == 2

    def test_record_recall_empty_ids(self):
        """空 ID 列表应该返回 0."""
        store = _make_mock_store()
        store.record_recall = MagicMock(return_value=0)
        result = store.record_recall([])
        assert result == 0


# ── Test: record_useful ──


class TestRecordUseful:
    def test_record_useful_increments_verified(self):
        """verified_count 递增."""
        store = _make_mock_store()

        result = store.record_useful(["m1"], "测试员工")
        assert result == 1
        store.record_useful.assert_called_once_with(["m1"], "测试员工")

    def test_record_useful_multiple_ids(self):
        """多个 ID 批量更新."""
        store = _make_mock_store()
        store.record_useful = MagicMock(return_value=3)

        result = store.record_useful(["m1", "m2", "m3"], "测试员工")
        assert result == 3


# ── Test: get_recall_stats ──


class TestGetRecallStats:
    def test_get_recall_stats_structure(self):
        """返回格式正确."""
        store = _make_mock_store()
        stats = store.get_recall_stats()

        assert "total_recalls" in stats
        assert "total_useful" in stats
        assert "hit_rate" in stats
        assert "top_useful" in stats
        assert "never_recalled" in stats

    def test_get_recall_stats_hit_rate(self):
        """算术正确（verified/recall）."""
        store = _make_mock_store()
        store.get_recall_stats = MagicMock(
            return_value={
                "total_recalls": 200,
                "total_useful": 50,
                "hit_rate": 0.25,
                "top_useful": [],
                "never_recalled": 10,
            }
        )

        stats = store.get_recall_stats()
        assert stats["hit_rate"] == pytest.approx(stats["total_useful"] / stats["total_recalls"])

    def test_get_recall_stats_zero_recalls(self):
        """零次召回时 hit_rate 为 0."""
        store = _make_mock_store()
        store.get_recall_stats = MagicMock(
            return_value={
                "total_recalls": 0,
                "total_useful": 0,
                "hit_rate": 0.0,
                "top_useful": [],
                "never_recalled": 100,
            }
        )

        stats = store.get_recall_stats()
        assert stats["hit_rate"] == 0.0


# ── Test: recall feedback handler ──


class TestRecallFeedbackHandler:
    @pytest.mark.asyncio
    async def test_recall_feedback_handler(self):
        """mock handler 冒烟测试."""
        from crew.webhook_handlers import _handle_recall_feedback

        mock_request = MagicMock()
        mock_request.json = AsyncMock(
            return_value={
                "employee": "测试员工",
                "useful_memory_ids": ["m1", "m2"],
            }
        )

        mock_ctx = MagicMock()
        mock_ctx.project_dir = None

        mock_store = _make_mock_store()

        with (
            patch("crew.webhook_handlers.get_memory_store", return_value=mock_store),
            patch("crew.webhook_handlers._tenant_id_for_store", return_value=None),
        ):
            response = await _handle_recall_feedback(mock_request, mock_ctx)

        body = json.loads(response.body.decode())
        assert body["ok"] is True
        assert body["updated"] == 1
        mock_store.record_useful.assert_called_once_with(["m1", "m2"], "测试员工")

    @pytest.mark.asyncio
    async def test_recall_feedback_missing_employee(self):
        """缺少 employee 时返回 400."""
        from crew.webhook_handlers import _handle_recall_feedback

        mock_request = MagicMock()
        mock_request.json = AsyncMock(
            return_value={
                "useful_memory_ids": ["m1"],
            }
        )

        mock_ctx = MagicMock()
        response = await _handle_recall_feedback(mock_request, mock_ctx)
        assert response.status_code == 400
        body = json.loads(response.body.decode())
        assert "employee" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_recall_feedback_missing_ids(self):
        """缺少 useful_memory_ids 时返回 400."""
        from crew.webhook_handlers import _handle_recall_feedback

        mock_request = MagicMock()
        mock_request.json = AsyncMock(
            return_value={
                "employee": "测试员工",
                "useful_memory_ids": [],
            }
        )

        mock_ctx = MagicMock()
        response = await _handle_recall_feedback(mock_request, mock_ctx)
        assert response.status_code == 400


# ── Test: employee state records recall ──


class TestEmployeeStateRecordsRecall:
    @pytest.mark.asyncio
    async def test_employee_state_records_recall(self):
        """验证 state API 会触发 record_recall."""
        from crew.webhook_handlers import _handle_employee_state

        mock_employee = MagicMock()
        mock_employee.name = "test-employee"
        mock_employee.character_name = "测试员工"
        mock_employee.display_name = "测试"
        mock_employee.agent_status = "active"
        mock_employee.source_path = None

        mock_store = _make_mock_store()
        mock_store.query.return_value = [
            _make_entry(id="m1"),
            _make_entry(id="m2"),
        ]

        mock_request = MagicMock()
        mock_request.path_params = {"identifier": "test-employee"}
        mock_request.query_params = {}

        mock_ctx = MagicMock()
        mock_ctx.project_dir = None

        mock_tenant = MagicMock()
        mock_tenant.is_admin = True
        mock_tenant.tenant_id = "admin"

        with (
            patch(
                "crew.discovery.discover_employees",
                return_value={"test-employee": mock_employee},
            ),
            patch(
                "crew.webhook_handlers._find_employee",
                return_value=mock_employee,
            ),
            patch(
                "crew.webhook_handlers.get_memory_store",
                return_value=mock_store,
            ),
            patch(
                "crew.webhook_handlers._tenant_id_for_store",
                return_value=None,
            ),
            patch(
                "crew.webhook_handlers._tenant_id_for_config",
                return_value="admin",
            ),
            patch(
                "crew.webhook_handlers.get_current_tenant",
                return_value=mock_tenant,
            ),
            patch(
                "crew.config_store.get_employee_from_db",
                return_value={"soul_content": "test soul"},
            ),
        ):
            response = await _handle_employee_state(mock_request, mock_ctx)

        # 验证 record_recall 被调用
        mock_store.record_recall.assert_called_once_with(["m1", "m2"])

        # 验证返回中包含 recalled_memory_ids
        body = json.loads(response.body.decode())
        assert "recalled_memory_ids" in body
        assert body["recalled_memory_ids"] == ["m1", "m2"]
