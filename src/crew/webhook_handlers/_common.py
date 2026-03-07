"""共享工具函数、常量和公共上下文引用."""

from __future__ import annotations

import asyncio
import logging
import os
import re as _re
from pathlib import Path
from typing import Any

from starlette.responses import JSONResponse, StreamingResponse  # noqa: F401

from crew.memory import get_memory_store  # noqa: F401
from crew.tenant import TenantContext, get_current_tenant  # noqa: F401
from crew.webhook_context import _EMPLOYEE_UPDATABLE_FIELDS, _AppContext  # noqa: F401

logger = logging.getLogger(__name__)

# 后台任务引用集合 — 防止 GC 提前回收 + 异常日志
_background_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]


def _task_done_callback(task: asyncio.Task) -> None:  # type: ignore[type-arg]
    """后台 task 完成回调：记录异常日志 + 从引用集合移除."""
    _background_tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("后台任务异常: %s", exc, exc_info=exc)






def _tenant_id_for_store(request: Any) -> str | None:
    """从请求获取租户 ID，用于传给 MemoryStoreDB 等接受 None 的 store.

    admin 租户返回 None（让 store 使用默认的 tenant_admin，向后兼容）。
    非 admin 租户返回具体 tenant_id 做数据隔离。
    """
    tenant = get_current_tenant(request)
    return None if tenant.is_admin else tenant.tenant_id


def _tenant_data_dir(request: Any, subdir: str) -> Path | None:
    """返回租户隔离的数据子目录，admin 返回 None（使用默认路径）.

    用法: _tenant_data_dir(request, "memory_archive") → Path("/data/tenants/{tid}/memory_archive") 或 None
    """
    tid = _tenant_id_for_store(request)
    return Path(f"/data/tenants/{tid}/{subdir}") if tid else None


def _tenant_base_dir(request: Any) -> Path:
    """返回租户数据根目录：非 admin 为 /data/tenants/{tid}/，admin 为 /data/.

    用于需要多个子目录的 store（如 MemoryFeedbackManager 同时需要 feedback + stats 目录）。
    """
    tid = _tenant_id_for_store(request)
    return Path(f"/data/tenants/{tid}") if tid else Path("/data")


def _tenant_id_for_config(request: Any) -> str:
    """从请求获取租户 ID，用于传给 config_store 等需要字符串的函数.

    始终返回具体的 tenant_id 字符串。
    """
    return get_current_tenant(request).tenant_id


def _require_admin_token(request: Any) -> str | None:
    """校验管理员 token，返回错误消息或 None（通过）.

    要求请求 header 中携带 X-Admin-Token，与环境变量 ADMIN_TOKEN 比对。
    如果 ADMIN_TOKEN 未配置，拒绝所有请求（fail-closed）。
    """
    import hmac

    admin_token = os.environ.get("ADMIN_TOKEN", "")
    if not admin_token:
        return "ADMIN_TOKEN not configured, admin operations are disabled"

    provided = request.headers.get("x-admin-token", "")
    if not provided or not hmac.compare_digest(provided, admin_token):
        return "forbidden: valid X-Admin-Token header required"

    return None


def _safe_int(value: str | None, default: int = 0) -> int:
    """安全转换为 int，转换失败时返回默认值."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


_MAX_QUERY_LIMIT = 1000


def _safe_limit(raw: str | int | None, default: int = 20) -> int:
    """安全解析 limit 参数，不超过 _MAX_QUERY_LIMIT."""
    if raw is None:
        return default
    try:
        val = int(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return default
    return max(1, min(val, _MAX_QUERY_LIMIT))


def _sanitized_error(e: Exception, public_msg: str = "内部错误") -> str:
    """对外返回安全错误信息，内部异常只写日志."""
    logger.exception("Handler error: %s", e)
    return public_msg


def _find_employee(result: Any, identifier: str) -> Any:
    """按 agent_id 或 name（字符串）查找员工."""
    # 先按 name 查找
    emp = result.get(identifier)
    if emp is not None:
        return emp
    # 再按 agent_id 查找（agent_id 现在是 "AI3050" 格式的字符串）
    for emp in result.employees.values():
        if emp.agent_id == identifier:
            return emp
    return None


def _ok_response(data: dict | None = None, status_code: int = 200) -> Any:
    """统一成功响应格式 — 后续逐步迁移各端点使用."""

    body: dict[str, Any] = {"ok": True}
    if data:
        body.update(data)
    return JSONResponse(body, status_code=status_code)


def _error_response(message: str, status_code: int = 400) -> Any:
    """统一错误响应格式 — 后续逐步迁移各端点使用."""

    return JSONResponse({"ok": False, "error": message}, status_code=status_code)


def _write_yaml_field(emp_dir: Path, updates: dict) -> None:
    """更新 employee.yaml 中的指定字段."""
    import tempfile

    import yaml

    config_path = emp_dir / "employee.yaml"
    if not config_path.exists():
        return
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        return
    config.update(updates)
    content = yaml.dump(config, allow_unicode=True, sort_keys=False, default_flow_style=False)
    fd, tmp = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
    fd_closed = False
    try:
        os.write(fd, content.encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
        fd_closed = True
        os.replace(tmp, config_path)
    except OSError:
        if not fd_closed:
            os.close(fd)
        Path(tmp).unlink(missing_ok=True)
        raise


def _parse_sender_name(extra_context: str | None) -> str | None:
    """从 extra_context 解析发送者名（格式: '当前对话用户: XXX'）."""
    if not extra_context:
        return None
    m = _re.search(r"当前对话用户[:：]\s*(\S+?)(?:（|$|\n)", extra_context)
    return m.group(1) if m else None


def _extract_task_description(text: str) -> str:
    """从 user_message 提取简洁的 task_description，过滤 soul prompt 污染.

    规则：
    - 以"你是"开头或超过 500 字 → 疑似 soul prompt，尝试提取 ## 任务 内容
    - 否则取前 200 字
    """
    if not text:
        return ""
    is_soul_prompt = text.startswith("你是") or len(text) > 500
    if is_soul_prompt:
        # 尝试提取 ## 任务 后的内容
        m = _re.search(r"##\s*(?:本次)?任务\s*\n+(.+)", text, _re.DOTALL)
        if m:
            task_text = m.group(1).strip()
            # 截取到下一个 ## 标题或文本结尾
            next_section = _re.search(r"\n##\s", task_text)
            if next_section:
                task_text = task_text[: next_section.start()].strip()
            if task_text:
                return task_text[:200]
        # 提取不到则用前 200 字，但不用 soul prompt 的"你是XXX"开头
        return text[:200]
    return text[:200]


# ── 租户管理 CRUD ──


def _require_admin_tenant(request: Any) -> TenantContext | None:
    """检查当前租户是否为 admin，返回 TenantContext 或 None（非 admin）."""
    tenant = get_current_tenant(request)
    return tenant if tenant.is_admin else None



async def _health(request: Any) -> Any:
    """健康检查."""

    return JSONResponse({"status": "ok", "service": "crew-webhook"})


async def _metrics(request: Any) -> Any:
    """运行时指标."""

    from crew.metrics import get_collector

    return JSONResponse(get_collector().snapshot())

