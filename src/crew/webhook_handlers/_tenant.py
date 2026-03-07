"""租户管理 CRUD 处理器."""

from __future__ import annotations

from typing import Any

from crew.webhook_handlers._common import (
    _AppContext,
    _error_response,
    _ok_response,
    _require_admin_tenant,
    logger,
)


async def _handle_tenant_create(request: Any, ctx: _AppContext) -> Any:
    """创建租户 — POST /api/tenants."""
    tenant = _require_admin_tenant(request)
    if not tenant:
        return _error_response("admin tenant required", 403)

    from crew.tenant import create_tenant

    try:
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body", 400)

    name = (body.get("name") or "").strip()
    if not name:
        return _error_response("name is required", 400)

    is_admin = bool(body.get("is_admin", False))
    metadata = body.get("metadata")

    try:
        result = create_tenant(name=name, is_admin=is_admin, metadata=metadata)
    except Exception:
        logger.exception("创建租户失败")
        return _error_response("创建租户失败", 500)

    # 清除中间件缓存
    if ctx.tenant_auth_cache is not None:
        ctx.tenant_auth_cache.clear()

    return _ok_response({"tenant": result}, status_code=201)


async def _handle_tenant_list(request: Any, ctx: _AppContext) -> Any:
    """列出所有租户 — GET /api/tenants."""
    tenant = _require_admin_tenant(request)
    if not tenant:
        return _error_response("admin tenant required", 403)

    from crew.tenant import list_tenants

    tenants = list_tenants()
    return _ok_response({"tenants": tenants})


async def _handle_tenant_get(request: Any, ctx: _AppContext) -> Any:
    """获取单个租户 — GET /api/tenants/{tenant_id}."""
    tenant = _require_admin_tenant(request)
    if not tenant:
        return _error_response("admin tenant required", 403)

    from crew.tenant import get_tenant_detail

    tenant_id = request.path_params["tenant_id"]
    detail = get_tenant_detail(tenant_id)
    if not detail:
        return _error_response("tenant not found", 404)

    return _ok_response({"tenant": detail})


async def _handle_tenant_delete(request: Any, ctx: _AppContext) -> Any:
    """删除租户 — DELETE /api/tenants/{tenant_id}."""
    tenant = _require_admin_tenant(request)
    if not tenant:
        return _error_response("admin tenant required", 403)

    from crew.tenant import delete_tenant

    tenant_id = request.path_params["tenant_id"]
    try:
        deleted = delete_tenant(tenant_id)
    except Exception:
        logger.exception("删除租户失败")
        return _error_response("删除租户失败", 500)

    if not deleted:
        return _error_response("tenant not found", 404)

    # 清除中间件缓存
    if ctx.tenant_auth_cache is not None:
        ctx.tenant_auth_cache.clear()

    return _ok_response({"deleted": tenant_id})


async def _handle_tenant_update(request: Any, ctx: _AppContext) -> Any:
    """更新租户 — PATCH /api/tenants/{tenant_id}."""
    tenant = _require_admin_tenant(request)
    if not tenant:
        return _error_response("admin tenant required", 403)

    from crew.tenant import update_tenant

    tenant_id = request.path_params["tenant_id"]

    try:
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body", 400)

    name = body.get("name")
    metadata = body.get("metadata")

    if name is None and metadata is None:
        return _error_response("at least one of name or metadata is required", 400)

    try:
        result = update_tenant(tenant_id, name=name, metadata=metadata)
    except ValueError as e:
        return _error_response(str(e), 400)
    except Exception:
        logger.exception("更新租户失败")
        return _error_response("更新租户失败", 500)

    if result is None:
        return _error_response("tenant not found", 404)

    # 清除中间件缓存（名称变了需要刷新）
    if ctx.tenant_auth_cache is not None:
        ctx.tenant_auth_cache.clear()

    return _ok_response({"tenant": result})

