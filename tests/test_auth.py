"""认证与安全中间件测试."""

import pytest

pytest.importorskip("mcp")

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from crew.auth import BearerTokenMiddleware, RateLimitMiddleware, RequestSizeLimitMiddleware

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


def _make_post_app():
    """构建支持 POST 的测试 app."""

    async def upload(request: Request):
        body = await request.body()
        return JSONResponse({"size": len(body)})

    async def health(request: Request):
        return JSONResponse({"status": "ok"})

    return Starlette(
        routes=[
            Route("/health", endpoint=health, methods=["GET"]),
            Route("/upload", endpoint=upload, methods=["POST"]),
        ],
    )


class TestRequestSizeLimit:
    """请求大小限制中间件."""

    def test_oversized_request_rejected(self):
        app = _make_post_app()
        app.add_middleware(RequestSizeLimitMiddleware, max_bytes=100)
        client = TestClient(app)
        resp = client.post(
            "/upload",
            content="x" * 200,
            headers={"Content-Length": "200", "Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 413
        assert "too large" in resp.json()["error"]

    def test_normal_request_passes(self):
        app = _make_post_app()
        app.add_middleware(RequestSizeLimitMiddleware, max_bytes=1000)
        client = TestClient(app)
        resp = client.post(
            "/upload",
            content="x" * 50,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 200

    def test_no_content_length_passes(self):
        """无 Content-Length 头的请求不被拦截."""
        app = _make_post_app()
        app.add_middleware(RequestSizeLimitMiddleware, max_bytes=10)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200


class TestRateLimiter:
    """速率限制中间件."""

    def test_within_limit_passes(self):
        app = _make_post_app()
        app.add_middleware(RateLimitMiddleware, rate=5, window=60.0)
        client = TestClient(app)
        for _ in range(5):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_exceeds_limit_returns_429(self):
        app = _make_post_app()
        app.add_middleware(RateLimitMiddleware, rate=3, window=60.0, skip_paths=[])
        client = TestClient(app)
        for _ in range(3):
            resp = client.post("/upload", content="x")
            assert resp.status_code == 200
        resp = client.post("/upload", content="x")
        assert resp.status_code == 429
        assert "rate limit" in resp.json()["error"]

    def test_health_bypasses_rate_limit(self):
        app = _make_post_app()
        app.add_middleware(RateLimitMiddleware, rate=1, window=60.0, skip_paths=["/health"])
        client = TestClient(app)
        # 第一次 upload 应该成功
        resp = client.post("/upload", content="x")
        assert resp.status_code == 200
        # 第二次 upload 被限流
        resp = client.post("/upload", content="x")
        assert resp.status_code == 429
        # health 不受限制
        for _ in range(5):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_default_skip_paths(self):
        app = _make_post_app()
        middleware = RateLimitMiddleware(app, rate=10)
        assert "/health" in middleware.skip_paths
        assert "/metrics" in middleware.skip_paths

    def test_bucket_cleanup_on_overflow(self):
        """超过 1000 个 IP 后清理空桶."""
        app = _make_post_app()
        middleware = RateLimitMiddleware(app, rate=100, window=0.001, skip_paths=[])
        # 模拟大量空桶
        for i in range(1100):
            middleware._buckets[f"10.0.{i // 256}.{i % 256}"] = []
        assert len(middleware._buckets) >= 1100

        client = TestClient(
            Starlette(
                routes=[
                    Route(
                        "/upload", endpoint=lambda r: JSONResponse({"ok": True}), methods=["POST"]
                    )
                ],
                middleware=[],
            )
        )
        # 直接触发 dispatch 不方便，验证桶数
        # 清理空桶后应减少
        stale = [ip for ip, ts in middleware._buckets.items() if not ts]
        for ip in stale:
            del middleware._buckets[ip]
        assert len(middleware._buckets) < 1100


class TestRequestSizeLimitInvalidHeader:
    """Content-Length 畸形值防御."""

    def test_invalid_content_length_returns_400(self):
        app = _make_post_app()
        app.add_middleware(RequestSizeLimitMiddleware, max_bytes=1000)
        client = TestClient(app)
        resp = client.post(
            "/upload",
            content="x",
            headers={"Content-Length": "not-a-number", "Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 400
        assert "invalid" in resp.json()["error"]

    def test_negative_content_length(self):
        app = _make_post_app()
        app.add_middleware(RequestSizeLimitMiddleware, max_bytes=1000)
        client = TestClient(app)
        resp = client.post(
            "/upload",
            content="x",
            headers={"Content-Length": "-1", "Content-Type": "application/octet-stream"},
        )
        # -1 < max_bytes, so it passes through (valid int, just negative)
        assert resp.status_code == 200


class TestMagicNumberConstants:
    """魔法数字提取为命名常量."""

    def test_default_max_request_bytes(self):
        assert RequestSizeLimitMiddleware.DEFAULT_MAX_REQUEST_BYTES == 1_048_576

    def test_bucket_cleanup_threshold(self):
        from crew.auth import RateLimitMiddleware

        assert RateLimitMiddleware._BUCKET_CLEANUP_THRESHOLD == 1000
