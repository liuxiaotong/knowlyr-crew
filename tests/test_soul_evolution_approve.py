"""灵魂进化审批执行测试 -- approve/reject candidate 单元测试."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from crew.soul_evolution import (
    _update_soul_archive,
    _update_soul_promote,
    approve_candidate,
    reject_candidate,
)

# ── 辅助 mock ──


def _make_config_store(data: dict[str, str] | None = None):
    """构造 mock config_store get/put 函数.

    data 是 {key: json_string} 字典，get_config 从中读，put_config 写入。
    """
    store = dict(data or {})

    def _get(ns: str, key: str) -> str | None:
        return store.get(f"{ns}:{key}")

    def _put(ns: str, key: str, value: str) -> None:
        store[f"{ns}:{key}"] = value

    return store, _get, _put


def _sample_candidate(
    candidate_id: str = "abc123",
    action: str = "promote",
    content: str = "始终在回复中引用来源",
    employee: str = "姜墨言",
) -> dict:
    return {
        "id": candidate_id,
        "employee": employee,
        "action": action,
        "source_type": "pattern" if action == "promote" else "correction",
        "source_ids": ["mem_001", "mem_002"],
        "content": content,
        "reason": "测试候选",
        "confidence": 0.8,
        "created_at": "2026-03-01T00:00:00",
    }


# ── 测试 _update_soul_promote ──


class TestUpdateSoulPromote:
    """_update_soul_promote — mock HTTP 验证行为准则追加."""

    def test_approve_promote_updates_soul(self):
        """mock soul API，验证行为准则被追加."""
        soul_text = "# 姜墨言\n\n## 行为准则\n\n- 永远诚实\n- 尊重用户\n\n## 自检清单\n\n- 检查回复"

        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.json.return_value = {"soul": soul_text}

        mock_put_resp = MagicMock()
        mock_put_resp.status_code = 200

        with (
            patch("crew.soul_evolution.os.environ.get", return_value="test-token"),
            patch("crew.soul_evolution._HAS_HTTPX", True),
            patch("crew.soul_evolution.httpx") as mock_httpx,
        ):
            mock_httpx.get.return_value = mock_get_resp
            mock_httpx.put.return_value = mock_put_resp

            result = _update_soul_promote("jiang-moyan", "始终引用来源")

        assert result is True

        # 验证 PUT 调用的 soul 内容包含新规则
        put_call = mock_httpx.put.call_args
        updated_soul = put_call.kwargs.get("json", put_call[1].get("json", {})).get("soul", "")
        assert "- 始终引用来源" in updated_soul
        assert "- 永远诚实" in updated_soul
        assert "- 尊重用户" in updated_soul

    def test_promote_no_token_returns_false(self):
        """没有 API token 时返回 False."""
        with patch("crew.soul_evolution.os.environ.get", return_value=None):
            result = _update_soul_promote("jiang-moyan", "test rule")
        assert result is False


# ── 测试 _update_soul_archive ──


class TestUpdateSoulArchive:
    """_update_soul_archive — mock HTTP 验证归档注释添加."""

    def test_approve_archive_adds_note(self):
        """mock soul API，验证归档注释被添加."""
        soul_text = "# 姜墨言\n\n## 行为准则\n\n- 永远诚实\n\n## 自检清单\n\n- 检查回复"

        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.json.return_value = {"soul": soul_text}

        mock_put_resp = MagicMock()
        mock_put_resp.status_code = 200

        with (
            patch("crew.soul_evolution.os.environ.get", return_value="test-token"),
            patch("crew.soul_evolution._HAS_HTTPX", True),
            patch("crew.soul_evolution.httpx") as mock_httpx,
        ):
            mock_httpx.get.return_value = mock_get_resp
            mock_httpx.put.return_value = mock_put_resp

            result = _update_soul_archive("jiang-moyan", "旧规则已不适用")

        assert result is True

        put_call = mock_httpx.put.call_args
        updated_soul = put_call.kwargs.get("json", put_call[1].get("json", {})).get("soul", "")
        assert "<!-- archived: 旧规则已不适用 -->" in updated_soul
        # 归档注释应在自检清单之前
        archive_pos = updated_soul.find("<!-- archived:")
        selfcheck_pos = updated_soul.find("## 自检清单")
        assert archive_pos < selfcheck_pos


# ── 测试 approve_candidate ──


class TestApproveCandidate:
    """approve_candidate 完整流程测试."""

    def test_approve_moves_to_approved_list(self):
        """候选从 candidates 移到 approved."""
        candidate = _sample_candidate()
        config_data, mock_get, mock_put = _make_config_store(
            {"soul_evolution:姜墨言_candidates": json.dumps([candidate])}
        )

        mock_store = MagicMock()
        mock_store.add = MagicMock()
        mock_store.update = MagicMock()

        with (
            patch("crew.soul_evolution.get_config", side_effect=mock_get),
            patch("crew.soul_evolution.put_config", side_effect=mock_put),
            patch("crew.soul_evolution._employee_slug_from_name", return_value=None),
        ):
            result = approve_candidate("abc123", "姜墨言", store=mock_store)

        assert result["ok"] is True
        assert result["action"] == "promote"
        assert result["content"] == "始终在回复中引用来源"
        assert result["soul_updated"] is False  # no slug found

        # candidates 应该为空
        remaining = json.loads(config_data["soul_evolution:姜墨言_candidates"])
        assert len(remaining) == 0

        # approved 列表应包含该候选
        approved = json.loads(config_data["soul_evolution:姜墨言_approved"])
        assert len(approved) == 1
        assert approved[0]["id"] == "abc123"
        assert "approved_at" in approved[0]

    def test_approve_records_decision_memory(self):
        """生成 decision 类记忆."""
        candidate = _sample_candidate()
        config_data, mock_get, mock_put = _make_config_store(
            {"soul_evolution:姜墨言_candidates": json.dumps([candidate])}
        )

        mock_store = MagicMock()
        mock_store.add = MagicMock()
        mock_store.update = MagicMock()

        with (
            patch("crew.soul_evolution.get_config", side_effect=mock_get),
            patch("crew.soul_evolution.put_config", side_effect=mock_put),
            patch("crew.soul_evolution._employee_slug_from_name", return_value=None),
        ):
            approve_candidate("abc123", "姜墨言", store=mock_store)

        # 验证 store.add 被调用且 category='decision'
        mock_store.add.assert_called_once()
        call_kwargs = mock_store.add.call_args
        assert call_kwargs.kwargs.get("category") == "decision"
        assert "promote" in call_kwargs.kwargs.get("content", "")

    def test_approve_nonexistent_candidate(self):
        """不存在的 candidate_id 返回错误."""
        candidate = _sample_candidate(candidate_id="other123")
        config_data, mock_get, mock_put = _make_config_store(
            {"soul_evolution:姜墨言_candidates": json.dumps([candidate])}
        )

        with (
            patch("crew.soul_evolution.get_config", side_effect=mock_get),
            patch("crew.soul_evolution.put_config", side_effect=mock_put),
        ):
            result = approve_candidate("nonexistent", "姜墨言")

        assert result["ok"] is False
        assert "不存在" in result["error"]

    def test_approve_no_candidates_key(self):
        """员工没有候选时返回错误."""
        config_data, mock_get, mock_put = _make_config_store()

        with (
            patch("crew.soul_evolution.get_config", side_effect=mock_get),
            patch("crew.soul_evolution.put_config", side_effect=mock_put),
        ):
            result = approve_candidate("abc123", "姜墨言")

        assert result["ok"] is False

    def test_approve_archive_marks_superseded(self):
        """archive 候选审批后标记 source corrections superseded_by=archived."""
        candidate = _sample_candidate(action="archive", content="旧规则不适用")
        config_data, mock_get, mock_put = _make_config_store(
            {"soul_evolution:姜墨言_candidates": json.dumps([candidate])}
        )

        mock_store = MagicMock()
        mock_store.add = MagicMock()
        mock_store.update = MagicMock()

        with (
            patch("crew.soul_evolution.get_config", side_effect=mock_get),
            patch("crew.soul_evolution.put_config", side_effect=mock_put),
            patch("crew.soul_evolution._employee_slug_from_name", return_value=None),
        ):
            result = approve_candidate("abc123", "姜墨言", store=mock_store)

        assert result["ok"] is True
        assert result["action"] == "archive"

        # 验证 source_ids 被标记 superseded
        assert mock_store.update.call_count == 2  # mem_001, mem_002


# ── 测试 reject_candidate ──


class TestRejectCandidate:
    """reject_candidate 测试."""

    def test_reject_moves_to_rejected_list(self):
        """候选从 candidates 移到 rejected."""
        candidate = _sample_candidate()
        config_data, mock_get, mock_put = _make_config_store(
            {"soul_evolution:姜墨言_candidates": json.dumps([candidate])}
        )

        with (
            patch("crew.soul_evolution.get_config", side_effect=mock_get),
            patch("crew.soul_evolution.put_config", side_effect=mock_put),
        ):
            result = reject_candidate("abc123", "姜墨言")

        assert result["ok"] is True
        assert result["candidate_id"] == "abc123"

        # candidates 应为空
        remaining = json.loads(config_data["soul_evolution:姜墨言_candidates"])
        assert len(remaining) == 0

        # rejected 列表应包含该候选
        rejected = json.loads(config_data["soul_evolution:姜墨言_rejected"])
        assert len(rejected) == 1
        assert rejected[0]["id"] == "abc123"
        assert "rejected_at" in rejected[0]

    def test_reject_nonexistent_candidate(self):
        """不存在的 candidate_id 返回错误."""
        candidate = _sample_candidate(candidate_id="other123")
        config_data, mock_get, mock_put = _make_config_store(
            {"soul_evolution:姜墨言_candidates": json.dumps([candidate])}
        )

        with (
            patch("crew.soul_evolution.get_config", side_effect=mock_get),
            patch("crew.soul_evolution.put_config", side_effect=mock_put),
        ):
            result = reject_candidate("nonexistent", "姜墨言")

        assert result["ok"] is False


# ── 测试 handler 冒烟 ──


class TestHandlerSmoke:
    """handler 冒烟测试."""

    @pytest.mark.asyncio
    async def test_approve_handler_smoke(self):
        """_handle_evolution_approve handler 冒烟测试."""
        from crew.webhook_handlers import _handle_evolution_approve

        mock_request = MagicMock()
        mock_request.headers = {"x-admin-token": "test-admin-token"}

        async def _json():
            return {"candidate_id": "abc123", "employee": "姜墨言"}

        mock_request.json = _json

        mock_ctx = MagicMock()

        with (
            patch(
                "os.environ.get",
                side_effect=lambda k, d="": "test-admin-token" if k == "ADMIN_TOKEN" else d,
            ),
            patch(
                "crew.webhook_handlers.get_memory_store",
                return_value=MagicMock(),
            ),
            patch(
                "crew.soul_evolution.approve_candidate",
                return_value={
                    "ok": True,
                    "action": "promote",
                    "content": "test",
                    "soul_updated": False,
                },
            ),
        ):
            resp = await _handle_evolution_approve(mock_request, mock_ctx)

        assert resp.status_code == 200
