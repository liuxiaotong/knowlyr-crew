"""Bearer token 认证中间件."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """校验 Authorization: Bearer <token>."""

    def __init__(self, app, *, token: str, skip_paths: list[str] | None = None):
        super().__init__(app)
        self.token = token
        self.skip_paths = skip_paths or ["/health"]

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.skip_paths:
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != self.token:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        return await call_next(request)
