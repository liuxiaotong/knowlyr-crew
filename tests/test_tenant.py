"""多租户核心模块测试 — tenant.py.

测试覆盖:
1. TenantContext 数据模型
2. MultiTenantAuthMiddleware 认证逻辑
3. get_current_tenant 依赖注入
4. 向后兼容性（无 DB 时 fallback 到管理员 token）
"""

import pytest

pytest.importorskip("mcp")

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from crew.tenant import (
    DEFAULT_ADMIN_TENANT_ID,
    DEFAULT_ADMIN_TENANT_NAME,
    MultiTenantAuthMiddleware,
    TenantContext,
    get_current_tenant,
)

ADMIN_TOKEN = "admin-secret-token"
USER_TOKEN = "user-token-abc"


# ── TenantContext 模型测试 ──


class TestTenantContext:
    def test_default_not_admin(self):
        ctx = TenantContext(tenant_id="t1", tenant_name="test")
        assert ctx.tenant_id == "t1"
        assert ctx.tenant_name == "test"
        assert ctx.is_admin is False

    def test_admin_context(self):
        ctx = TenantContext(tenant_id="admin", tenant_name="admin", is_admin=True)
        assert ctx.is_admin is True


# ── get_current_tenant 测试 ──


class TestGetCurrentTenant:
    def test_returns_admin_when_no_state(self):
        """无 state 时返回默认管理员."""

        class FakeRequest:
            pass

        tenant = get_current_tenant(FakeRequest())
        assert tenant.tenant_id == DEFAULT_ADMIN_TENANT_ID
        assert tenant.tenant_name == DEFAULT_ADMIN_TENANT_NAME
        assert tenant.is_admin is True

    def test_returns_tenant_from_state(self):
        """从 request.state 获取租户."""

        class FakeState:
            tenant = TenantContext(tenant_id="t1", tenant_name="user1")

        class FakeRequest:
            state = FakeState()

        tenant = get_current_tenant(FakeRequest())
        assert tenant.tenant_id == "t1"
        assert tenant.tenant_name == "user1"
        assert tenant.is_admin is False


# ── MultiTenantAuthMiddleware 测试（无 DB，fallback 模式）──


def _make_tenant_app():
    """构建测试 app."""

    async def hello(request: Request):
        tenant = get_current_tenant(request)
        return JSONResponse({
            "msg": "ok",
            "tenant_id": tenant.tenant_id,
            "is_admin": tenant.is_admin,
        })

    async def health(request: Request):
        return JSONResponse({"status": "ok"})

    app = Starlette(
        routes=[
            Route("/health", endpoint=health, methods=["GET"]),
            Route("/hello", endpoint=hello, methods=["GET"]),
        ],
    )
    return app


class TestMultiTenantAuth:
    """MultiTenantAuthMiddleware 认证测试（无 DB fallback 模式）."""

    def test_no_token_returns_401(self):
        app = _make_tenant_app()
        app.add_middleware(
            MultiTenantAuthMiddleware,
            admin_token=ADMIN_TOKEN,
            skip_paths=["/health"],
        )
        client = TestClient(app)
        resp = client.get("/hello")
        assert resp.status_code == 401

    def test_wrong_token_returns_401(self):
        app = _make_tenant_app()
        app.add_middleware(
            MultiTenantAuthMiddleware,
            admin_token=ADMIN_TOKEN,
            skip_paths=["/health"],
        )
        client = TestClient(app)
        resp = client.get("/hello", headers={"Authorization": "Bearer wrong-token"})
        assert resp.status_code == 401

    def test_admin_token_passes(self):
        app = _make_tenant_app()
        app.add_middleware(
            MultiTenantAuthMiddleware,
            admin_token=ADMIN_TOKEN,
            skip_paths=["/health"],
        )
        client = TestClient(app)
        resp = client.get("/hello", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == DEFAULT_ADMIN_TENANT_ID
        assert data["is_admin"] is True

    def test_health_bypasses_auth(self):
        app = _make_tenant_app()
        app.add_middleware(
            MultiTenantAuthMiddleware,
            admin_token=ADMIN_TOKEN,
            skip_paths=["/health"],
        )
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_missing_bearer_prefix_returns_401(self):
        app = _make_tenant_app()
        app.add_middleware(
            MultiTenantAuthMiddleware,
            admin_token=ADMIN_TOKEN,
            skip_paths=["/health"],
        )
        client = TestClient(app)
        resp = client.get("/hello", headers={"Authorization": ADMIN_TOKEN})
        assert resp.status_code == 401

    def test_cache_invalidation(self):
        app = _make_tenant_app()
        middleware = MultiTenantAuthMiddleware(
            app,
            admin_token=ADMIN_TOKEN,
        )
        # 手动填入缓存
        ctx = TenantContext(tenant_id="t1", tenant_name="test")
        middleware._cache["some-key"] = ctx
        assert "some-key" in middleware._cache

        # 清除特定 key
        middleware.invalidate_cache("some-key")
        assert "some-key" not in middleware._cache

        # 清除全部
        middleware._cache["a"] = ctx
        middleware._cache["b"] = ctx
        middleware.invalidate_cache()
        assert len(middleware._cache) == 0

    def test_skip_path_prefix_match(self):
        """skip_paths 支持前缀匹配."""
        app = _make_tenant_app()
        app.add_middleware(
            MultiTenantAuthMiddleware,
            admin_token=ADMIN_TOKEN,
            skip_paths=["/health", "/static"],
        )
        client = TestClient(app)
        # /health 精确匹配
        resp = client.get("/health")
        assert resp.status_code == 200
