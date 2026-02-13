"""BearerTokenMiddleware 测试."""

import pytest

pytest.importorskip("mcp")

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from crew.auth import BearerTokenMiddleware

TOKEN = "test-secret-token"


def _make_app(*, with_auth: bool = True):
    """构建带/不带认证的测试 app."""

    async def hello(request: Request):
        return JSONResponse({"msg": "ok"})

    async def health(request: Request):
        return JSONResponse({"status": "ok"})

    app = Starlette(
        routes=[
            Route("/health", endpoint=health),
            Route("/hello", endpoint=hello),
        ],
    )
    if with_auth:
        app.add_middleware(BearerTokenMiddleware, token=TOKEN)
    return app


class TestWithAuth:
    """认证启用时的行为."""

    def setup_method(self):
        self.client = TestClient(_make_app(with_auth=True))

    def test_no_header_returns_401(self):
        resp = self.client.get("/hello")
        assert resp.status_code == 401
        assert resp.json() == {"error": "unauthorized"}

    def test_wrong_token_returns_401(self):
        resp = self.client.get("/hello", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

    def test_correct_token_returns_200(self):
        resp = self.client.get("/hello", headers={"Authorization": f"Bearer {TOKEN}"})
        assert resp.status_code == 200
        assert resp.json() == {"msg": "ok"}

    def test_missing_bearer_prefix_returns_401(self):
        resp = self.client.get("/hello", headers={"Authorization": TOKEN})
        assert resp.status_code == 401

    def test_health_bypasses_auth(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_health_with_token_also_works(self):
        resp = self.client.get("/health", headers={"Authorization": f"Bearer {TOKEN}"})
        assert resp.status_code == 200


class TestWithoutAuth:
    """认证未启用时的行为（无中间件）."""

    def setup_method(self):
        self.client = TestClient(_make_app(with_auth=False))

    def test_no_header_passes_through(self):
        resp = self.client.get("/hello")
        assert resp.status_code == 200
        assert resp.json() == {"msg": "ok"}

    def test_health_still_works(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
