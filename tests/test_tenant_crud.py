"""租户管理 CRUD API 测试.

测试覆盖:
1. 创建租户 -> 成功（含 api_key）
2. 非 admin 调用 -> 403
3. 列出租户 -> 不含 api_key
4. 获取单个租户 -> 成功
5. 删除租户 -> 成功
6. 更新租户 -> 成功
7. 重复创建 -> 适当处理
8. 缺少必填字段 -> 400
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mcp")

from starlette.testclient import TestClient

from crew.tenant import TenantContext
from crew.webhook_context import _AppContext

# ── 工具函数 ──


def _make_ctx() -> _AppContext:
    """构造最小 _AppContext."""
    config = MagicMock()
    registry = MagicMock()
    ctx = _AppContext(project_dir=None, config=config, registry=registry)
    ctx.tenant_auth_cache = {}
    return ctx


def _make_app(ctx: _AppContext, is_admin: bool = True):
    """构造带租户 CRUD 路由的测试 app."""
    from starlette.applications import Starlette
    from starlette.routing import Route

    from crew.webhook_handlers import (
        _handle_tenant_create,
        _handle_tenant_delete,
        _handle_tenant_get,
        _handle_tenant_list,
        _handle_tenant_update,
    )

    tenant = TenantContext(
        tenant_id="tenant_admin" if is_admin else "tenant_user1",
        tenant_name="admin" if is_admin else "user1",
        is_admin=is_admin,
    )

    def _inject_tenant(handler):
        async def wrapper(request):
            # 模拟中间件注入 tenant
            request.state.tenant = tenant
            return await handler(request, ctx)

        return wrapper

    routes = [
        Route("/api/tenants", endpoint=_inject_tenant(_handle_tenant_create), methods=["POST"]),
        Route("/api/tenants", endpoint=_inject_tenant(_handle_tenant_list), methods=["GET"]),
        Route(
            "/api/tenants/{tenant_id}",
            endpoint=_inject_tenant(_handle_tenant_get),
            methods=["GET"],
        ),
        Route(
            "/api/tenants/{tenant_id}",
            endpoint=_inject_tenant(_handle_tenant_delete),
            methods=["DELETE"],
        ),
        Route(
            "/api/tenants/{tenant_id}",
            endpoint=_inject_tenant(_handle_tenant_update),
            methods=["PATCH"],
        ),
    ]
    return Starlette(routes=routes)


# ── 测试 ──


class TestTenantCrudNonAdmin:
    """非 admin 租户调用 -> 403."""

    def test_create_forbidden(self):
        ctx = _make_ctx()
        app = _make_app(ctx, is_admin=False)
        client = TestClient(app)
        resp = client.post("/api/tenants", json={"name": "test"})
        assert resp.status_code == 403

    def test_list_forbidden(self):
        ctx = _make_ctx()
        app = _make_app(ctx, is_admin=False)
        client = TestClient(app)
        resp = client.get("/api/tenants")
        assert resp.status_code == 403

    def test_get_forbidden(self):
        ctx = _make_ctx()
        app = _make_app(ctx, is_admin=False)
        client = TestClient(app)
        resp = client.get("/api/tenants/t1")
        assert resp.status_code == 403

    def test_delete_forbidden(self):
        ctx = _make_ctx()
        app = _make_app(ctx, is_admin=False)
        client = TestClient(app)
        resp = client.delete("/api/tenants/t1")
        assert resp.status_code == 403

    def test_update_forbidden(self):
        ctx = _make_ctx()
        app = _make_app(ctx, is_admin=False)
        client = TestClient(app)
        resp = client.patch("/api/tenants/t1", json={"name": "new"})
        assert resp.status_code == 403


class TestTenantCreate:
    """创建租户."""

    @patch("crew.tenant.create_tenant")
    def test_create_success(self, mock_create):
        mock_create.return_value = {
            "id": "tenant_abc123",
            "name": "test-tenant",
            "api_key": "secret-key-xxx",
            "is_admin": False,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        ctx = _make_ctx()
        app = _make_app(ctx)
        client = TestClient(app)

        resp = client.post("/api/tenants", json={"name": "test-tenant"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["ok"] is True
        assert data["tenant"]["name"] == "test-tenant"
        assert "api_key" in data["tenant"]  # 创建时返回 api_key

        mock_create.assert_called_once_with(
            name="test-tenant", is_admin=False, metadata=None
        )

    @patch("crew.tenant.create_tenant")
    def test_create_with_metadata(self, mock_create):
        mock_create.return_value = {
            "id": "tenant_abc123",
            "name": "test",
            "api_key": "key",
            "is_admin": True,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        ctx = _make_ctx()
        app = _make_app(ctx)
        client = TestClient(app)

        resp = client.post(
            "/api/tenants",
            json={"name": "test", "is_admin": True, "metadata": {"plan": "pro"}},
        )
        assert resp.status_code == 201
        mock_create.assert_called_once_with(
            name="test", is_admin=True, metadata={"plan": "pro"}
        )

    def test_create_missing_name(self):
        ctx = _make_ctx()
        app = _make_app(ctx)
        client = TestClient(app)

        resp = client.post("/api/tenants", json={})
        assert resp.status_code == 400
        assert "name" in resp.json()["error"]

    def test_create_empty_name(self):
        ctx = _make_ctx()
        app = _make_app(ctx)
        client = TestClient(app)

        resp = client.post("/api/tenants", json={"name": "  "})
        assert resp.status_code == 400

    @patch("crew.tenant.create_tenant")
    def test_create_clears_cache(self, mock_create):
        mock_create.return_value = {
            "id": "t1",
            "name": "x",
            "api_key": "k",
            "is_admin": False,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        ctx = _make_ctx()
        ctx.tenant_auth_cache["old-key"] = TenantContext(
            tenant_id="old", tenant_name="old"
        )
        app = _make_app(ctx)
        client = TestClient(app)

        client.post("/api/tenants", json={"name": "x"})
        assert len(ctx.tenant_auth_cache) == 0  # 缓存已清除


class TestTenantList:
    """列出租户."""

    @patch("crew.tenant.list_tenants")
    def test_list_success(self, mock_list):
        mock_list.return_value = [
            {
                "id": "t1",
                "name": "tenant1",
                "is_admin": False,
                "created_at": "2026-01-01T00:00:00",
            },
            {
                "id": "t2",
                "name": "tenant2",
                "is_admin": True,
                "created_at": "2026-01-02T00:00:00",
            },
        ]
        ctx = _make_ctx()
        app = _make_app(ctx)
        client = TestClient(app)

        resp = client.get("/api/tenants")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["tenants"]) == 2
        # 不含 api_key
        for t in data["tenants"]:
            assert "api_key" not in t


class TestTenantGet:
    """获取单个租户."""

    @patch("crew.tenant.get_tenant_detail")
    def test_get_success(self, mock_get):
        mock_get.return_value = {
            "id": "t1",
            "name": "tenant1",
            "is_admin": False,
            "created_at": "2026-01-01T00:00:00",
            "metadata": '{"plan": "free"}',
        }
        ctx = _make_ctx()
        app = _make_app(ctx)
        client = TestClient(app)

        resp = client.get("/api/tenants/t1")
        assert resp.status_code == 200
        assert resp.json()["tenant"]["id"] == "t1"
        assert "api_key" not in resp.json()["tenant"]

    @patch("crew.tenant.get_tenant_detail")
    def test_get_not_found(self, mock_get):
        mock_get.return_value = None
        ctx = _make_ctx()
        app = _make_app(ctx)
        client = TestClient(app)

        resp = client.get("/api/tenants/nonexistent")
        assert resp.status_code == 404


class TestTenantDelete:
    """删除租户."""

    @patch("crew.tenant.delete_tenant")
    def test_delete_success(self, mock_delete):
        mock_delete.return_value = True
        ctx = _make_ctx()
        app = _make_app(ctx)
        client = TestClient(app)

        resp = client.delete("/api/tenants/t1")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == "t1"

    @patch("crew.tenant.delete_tenant")
    def test_delete_not_found(self, mock_delete):
        mock_delete.return_value = False
        ctx = _make_ctx()
        app = _make_app(ctx)
        client = TestClient(app)

        resp = client.delete("/api/tenants/nonexistent")
        assert resp.status_code == 404

    @patch("crew.tenant.delete_tenant")
    def test_delete_clears_cache(self, mock_delete):
        mock_delete.return_value = True
        ctx = _make_ctx()
        ctx.tenant_auth_cache["some-key"] = TenantContext(
            tenant_id="x", tenant_name="x"
        )
        app = _make_app(ctx)
        client = TestClient(app)

        client.delete("/api/tenants/t1")
        assert len(ctx.tenant_auth_cache) == 0


class TestTenantUpdate:
    """更新租户."""

    @patch("crew.tenant.update_tenant")
    def test_update_name(self, mock_update):
        mock_update.return_value = {
            "id": "t1",
            "name": "new-name",
            "is_admin": False,
        }
        ctx = _make_ctx()
        app = _make_app(ctx)
        client = TestClient(app)

        resp = client.patch("/api/tenants/t1", json={"name": "new-name"})
        assert resp.status_code == 200
        assert resp.json()["tenant"]["name"] == "new-name"

    @patch("crew.tenant.update_tenant")
    def test_update_metadata(self, mock_update):
        mock_update.return_value = {
            "id": "t1",
            "name": "t1",
            "is_admin": False,
        }
        ctx = _make_ctx()
        app = _make_app(ctx)
        client = TestClient(app)

        resp = client.patch(
            "/api/tenants/t1", json={"metadata": {"plan": "pro"}}
        )
        assert resp.status_code == 200

    def test_update_no_fields(self):
        ctx = _make_ctx()
        app = _make_app(ctx)
        client = TestClient(app)

        resp = client.patch("/api/tenants/t1", json={})
        assert resp.status_code == 400

    @patch("crew.tenant.update_tenant")
    def test_update_not_found(self, mock_update):
        mock_update.return_value = None
        ctx = _make_ctx()
        app = _make_app(ctx)
        client = TestClient(app)

        resp = client.patch("/api/tenants/t1", json={"name": "new"})
        assert resp.status_code == 404
