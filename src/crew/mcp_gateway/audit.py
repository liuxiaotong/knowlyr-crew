"""MCP Gateway 审计日志 — 记录所有外部 MCP 工具调用.

审计日志写入数据库 ``mcp_audit_log`` 表，包含:
- 调用者（employee_name / user_id）
- 目标 MCP server + tool
- 参数摘要（脱敏后）
- 结果状态 + 耗时
- 异常信息（脱敏后，不泄露内部路径/凭据）
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── 脱敏规则 ──

_SENSITIVE_PATTERNS = [
    (
        re.compile(
            r"(token|password|secret|key|credential)[\"']?\s*[:=]\s*[\"']?[^\s\"',}{]+",
            re.IGNORECASE,
        ),
        r"\1=***REDACTED***",
    ),
    (re.compile(r"/home/[a-z_][a-z0-9_-]*/"), "/home/***/"),
    (re.compile(r"/root/"), "/***root***/"),
    (re.compile(r"Bearer\s+\S+", re.IGNORECASE), "Bearer ***REDACTED***"),
]


def sanitize_error(msg: str) -> str:
    """脱敏异常信息，去除敏感路径和凭据."""
    result = msg
    for pattern, replacement in _SENSITIVE_PATTERNS:
        result = pattern.sub(replacement, result)
    # 截断过长消息
    if len(result) > 2000:
        result = result[:2000] + "...(truncated)"
    return result


def sanitize_args(args: dict[str, Any]) -> dict[str, Any]:
    """脱敏工具参数，隐藏可能包含凭据的字段（递归处理嵌套结构）."""
    sensitive_keys = {"token", "password", "secret", "key", "credential", "api_key", "access_token"}

    def _sanitize_value(key: str, value: Any) -> Any:
        if key.lower() in sensitive_keys:
            return "***REDACTED***"
        return _sanitize_any(value)

    def _sanitize_any(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _sanitize_value(k, v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sanitize_any(item) for item in value]
        if isinstance(value, str) and len(value) > 500:
            return value[:200] + f"...(truncated, {len(value)} chars)"
        return value

    return {k: _sanitize_value(k, v) for k, v in args.items()}


# ── 审计表初始化 ──


def init_audit_table() -> None:
    """创建 mcp_audit_log 表（幂等）."""
    from crew.database import get_connection, is_pg

    if is_pg():
        sql = """\
CREATE TABLE IF NOT EXISTS mcp_audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tenant_id TEXT NOT NULL DEFAULT '',
    employee_name TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL DEFAULT '',
    server_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    namespaced_name TEXT NOT NULL,
    args_sanitized TEXT DEFAULT '{}',
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT DEFAULT '',
    duration_ms DOUBLE PRECISION DEFAULT 0,
    metadata TEXT DEFAULT '{}'
)"""
    else:
        sql = """\
CREATE TABLE IF NOT EXISTS mcp_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    tenant_id TEXT NOT NULL DEFAULT '',
    employee_name TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL DEFAULT '',
    server_name TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    namespaced_name TEXT NOT NULL,
    args_sanitized TEXT DEFAULT '{}',
    success INTEGER NOT NULL DEFAULT 1,
    error_message TEXT DEFAULT '',
    duration_ms REAL DEFAULT 0,
    metadata TEXT DEFAULT '{}'
)"""

    with get_connection() as conn:
        cur = conn.cursor() if is_pg() else conn
        cur.execute(sql)
        if not is_pg():
            conn.commit()

    # 索引
    idx_sqls = [
        "CREATE INDEX IF NOT EXISTS idx_mcp_audit_timestamp ON mcp_audit_log(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_mcp_audit_server ON mcp_audit_log(server_name)",
        "CREATE INDEX IF NOT EXISTS idx_mcp_audit_employee ON mcp_audit_log(employee_name)",
        "CREATE INDEX IF NOT EXISTS idx_mcp_audit_tenant ON mcp_audit_log(tenant_id)",
        "CREATE INDEX IF NOT EXISTS idx_mcp_audit_user ON mcp_audit_log(user_id)",
    ]
    with get_connection() as conn:
        cur = conn.cursor() if is_pg() else conn
        for idx_sql in idx_sqls:
            try:
                cur.execute(idx_sql)
            except Exception:
                pass  # 索引已存在
        if not is_pg():
            conn.commit()


def log_tool_call(
    *,
    employee_name: str,
    tenant_id: str = "",
    user_id: str = "",
    server_name: str,
    tool_name: str,
    namespaced_name: str,
    args: dict[str, Any] | None = None,
    success: bool = True,
    error_message: str = "",
    duration_ms: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> None:
    """记录一次 MCP 工具调用到审计日志."""
    from crew.database import get_connection, is_pg

    args_sanitized = json.dumps(sanitize_args(args or {}), ensure_ascii=False, default=str)
    error_safe = sanitize_error(error_message) if error_message else ""
    meta_json = json.dumps(metadata or {}, ensure_ascii=False, default=str)

    if is_pg():
        sql = """\
INSERT INTO mcp_audit_log (tenant_id, employee_name, user_id, server_name, tool_name, namespaced_name, args_sanitized, success, error_message, duration_ms, metadata)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        params = (
            tenant_id,
            employee_name,
            user_id,
            server_name,
            tool_name,
            namespaced_name,
            args_sanitized,
            success,
            error_safe,
            duration_ms,
            meta_json,
        )
    else:
        sql = """\
INSERT INTO mcp_audit_log (tenant_id, employee_name, user_id, server_name, tool_name, namespaced_name, args_sanitized, success, error_message, duration_ms, metadata)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        params = (
            tenant_id,
            employee_name,
            user_id,
            server_name,
            tool_name,
            namespaced_name,
            args_sanitized,
            int(success),
            error_safe,
            duration_ms,
            meta_json,
        )

    try:
        with get_connection() as conn:
            cur = conn.cursor() if is_pg() else conn
            cur.execute(sql, params)
            if not is_pg():
                conn.commit()
    except Exception as e:
        # 审计日志写入失败不应影响主流程
        logger.warning("审计日志写入失败: %s", e)
