"""Bearer token 认证中间件."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """校验 Authorization: Bearer <token>."""

    def __init__(self, app, *, token: str):
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next):
        # /health 免验证（部署探活）
        if request.url.path == "/health":
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != self.token:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        return await call_next(request)
