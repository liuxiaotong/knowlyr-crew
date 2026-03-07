"""Tests for knowledge dashboard: get_knowledge_stats and API endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_store():
    """创建一个 mock 出的 MemoryStoreDB 实例，跳过 psycopg2 依赖."""
    with patch("crew.memory_store_db.is_pg", return_value=True), \
         patch("crew.memory_store_db.get_connection"):
        mock_psycopg2 = MagicMock()
        with patch.dict("sys.modules", {"psycopg2": mock_psycopg2, "psycopg2.extras": mock_psycopg2.extras}):
            from crew.memory_store_db import MemoryStoreDB
            store = MemoryStoreDB.__new__(MemoryStoreDB)
            store._project_dir = None
            store._tenant_id = "test-tenant"
            store._dict_cursor_factory = MagicMock()
            return store


# ── Feature 3: get_knowledge_stats ──


class TestGetKnowledgeStats:
    """测试 get_knowledge_stats 返回的数据结构."""

    @patch("crew.memory_store_db.get_connection")
    @patch("crew.memory_store_db.is_pg", return_value=True)
    def test_stats_structure(self, mock_is_pg, mock_conn):
        """验证返回的数据结构包含所有必要字段."""
        store = _make_store()

        # 设置 mock cursor 返回不同查询的结果
        mock_cursor = MagicMock()
        call_count = [0]

        def fetchall_side_effect():
            call_count[0] += 1
            idx = call_count[0]
            if idx == 1:
                return [
                    {"employee": "赵云帆", "category": "finding", "cnt": 10},
                    {"employee": "赵云帆", "category": "pattern", "cnt": 5},
                    {"employee": "卫子昂", "category": "finding", "cnt": 8},
                ]
            elif idx == 2:
                return [
                    {"kw": "API", "cnt": 15},
                    {"kw": "性能", "cnt": 12},
                ]
            elif idx == 3:
                return [{"kw": "配置", "cnt": 3}]
            elif idx == 4:
                return [{"keyword": "部署", "findings": 5, "patterns": 0}]
            elif idx == 5:
                return [
                    {"week": "2026-W09", "cnt": 20},
                    {"week": "2026-W10", "cnt": 25},
                ]
            return []

        mock_cursor.fetchall = MagicMock(side_effect=fetchall_side_effect)

        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cm)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_cm.cursor.return_value = mock_cursor
        mock_conn.return_value = mock_cm

        stats = store.get_knowledge_stats()

        # 验证顶层 keys
        assert "employee_stats" in stats
        assert "top_keywords" in stats
        assert "correction_hotspots" in stats
        assert "knowledge_gaps" in stats
        assert "weekly_trend" in stats

        # 验证 employee_stats 结构
        emp_stats = stats["employee_stats"]
        assert len(emp_stats) == 2
        zhao = next(e for e in emp_stats if e["employee"] == "赵云帆")
        assert zhao["total"] == 15
        assert zhao["by_category"]["finding"] == 10
        assert zhao["by_category"]["pattern"] == 5

        # 验证 top_keywords
        assert len(stats["top_keywords"]) == 2
        assert stats["top_keywords"][0]["keyword"] == "API"
        assert stats["top_keywords"][0]["count"] == 15

        # 验证 correction_hotspots
        assert len(stats["correction_hotspots"]) == 1
        assert stats["correction_hotspots"][0]["keyword"] == "配置"
        assert stats["correction_hotspots"][0]["correction_count"] == 3

        # 验证 knowledge_gaps
        assert len(stats["knowledge_gaps"]) == 1
        assert stats["knowledge_gaps"][0]["keyword"] == "部署"
        assert stats["knowledge_gaps"][0]["findings"] == 5
        assert stats["knowledge_gaps"][0]["patterns"] == 0

        # 验证 weekly_trend
        assert len(stats["weekly_trend"]) == 2
        assert stats["weekly_trend"][0]["week"] == "2026-W09"
        assert stats["weekly_trend"][0]["count"] == 20

    @patch("crew.memory_store_db.get_connection")
    @patch("crew.memory_store_db.is_pg", return_value=True)
    def test_stats_empty_db(self, mock_is_pg, mock_conn):
        """空数据库时返回空列表."""
        store = _make_store()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []

        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_cm)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_cm.cursor.return_value = mock_cursor
        mock_conn.return_value = mock_cm

        stats = store.get_knowledge_stats()

        assert stats["employee_stats"] == []
        assert stats["top_keywords"] == []
        assert stats["correction_hotspots"] == []
        assert stats["knowledge_gaps"] == []
        assert stats["weekly_trend"] == []


# ── Feature 3: API endpoint ──


class TestKnowledgeDashboardEndpoint:
    """测试 /api/knowledge/dashboard 端点."""

    def test_handler_exists(self):
        """验证 handler 可以被导入."""
        from crew.webhook_handlers import _handle_knowledge_dashboard

        assert callable(_handle_knowledge_dashboard)

    def test_route_registered(self):
        """验证路由已注册."""
        starlette = pytest.importorskip("starlette")
        from pathlib import Path

        from crew.webhook import create_webhook_app
        from crew.webhook_config import WebhookConfig

        app = create_webhook_app(
            project_dir=Path("/tmp/test-knowledge"),
            token="test-token",
            config=WebhookConfig(),
        )
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/api/knowledge/dashboard" in routes
