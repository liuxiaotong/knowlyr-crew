"""MCP Gateway 凭据管理 — 用户级 MCP 凭据的数据库加密存储.

每个用户的 OAuth token 存储在 ``user_mcp_credentials`` 表中，
使用 Fernet (AES-128-CBC + HMAC-SHA256) 加密
（通过环境变量 ``MCP_CREDENTIAL_KEY`` 提供密钥）。
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

# ── 加密工具 ──

_CREDENTIAL_KEY_ENV = "MCP_CREDENTIAL_KEY"


def _get_fernet() -> Fernet:
    """获取 Fernet 实例（从环境变量派生密钥）.

    生产环境（ENV=production）缺少 MCP_CREDENTIAL_KEY 时直接报错。
    """
    raw = os.environ.get(_CREDENTIAL_KEY_ENV, "")
    if not raw:
        env = os.environ.get("ENV", "").lower()
        if env == "production":
            raise RuntimeError(f"{_CREDENTIAL_KEY_ENV} 未设置，生产环境不允许使用默认密钥")
        logger.warning("%s 未设置，使用默认密钥（仅限开发环境！）", _CREDENTIAL_KEY_ENV)
        raw = "dev-only-insecure-key-do-not-use-in-prod"
    # SHA-256 派生 32 字节，取前 32 字节做 url-safe base64 作为 Fernet key
    derived = hashlib.sha256(raw.encode()).digest()
    fernet_key = base64.urlsafe_b64encode(derived)
    return Fernet(fernet_key)


def _encrypt(plaintext: str) -> str:
    """Fernet 加密，返回 base64 编码的密文."""
    f = _get_fernet()
    return f.encrypt(plaintext.encode("utf-8")).decode("ascii")


def _decrypt(ciphertext_b64: str) -> str:
    """解密 Fernet 密文."""
    f = _get_fernet()
    return f.decrypt(ciphertext_b64.encode("ascii")).decode("utf-8")


# ── 数据库操作 ──


def _init_credentials_table() -> None:
    """创建 user_mcp_credentials 表（幂等）."""
    from crew.database import get_connection, is_pg

    if is_pg():
        sql = """\
CREATE TABLE IF NOT EXISTS user_mcp_credentials (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    server_name TEXT NOT NULL,
    credential_type TEXT NOT NULL DEFAULT 'oauth_token',
    encrypted_value TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, server_name)
)"""
    else:
        sql = """\
CREATE TABLE IF NOT EXISTS user_mcp_credentials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    server_name TEXT NOT NULL,
    credential_type TEXT NOT NULL DEFAULT 'oauth_token',
    encrypted_value TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, server_name)
)"""

    with get_connection() as conn:
        cur = conn.cursor() if is_pg() else conn
        cur.execute(sql)
        if not is_pg():
            conn.commit()


def store_credential(
    user_id: str,
    server_name: str,
    token: str,
    credential_type: str = "oauth_token",
    metadata: dict[str, Any] | None = None,
) -> None:
    """存储用户的 MCP server 凭据（加密）."""
    from crew.database import get_connection, is_pg

    encrypted = _encrypt(token)
    meta_json = json.dumps(metadata or {}, ensure_ascii=False)

    if is_pg():
        sql = """\
INSERT INTO user_mcp_credentials (user_id, server_name, credential_type, encrypted_value, metadata, updated_at)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (user_id, server_name) DO UPDATE SET
    encrypted_value = EXCLUDED.encrypted_value,
    credential_type = EXCLUDED.credential_type,
    metadata = EXCLUDED.metadata,
    updated_at = EXCLUDED.updated_at
"""
        params = (
            user_id,
            server_name,
            credential_type,
            encrypted,
            meta_json,
            datetime.now(timezone.utc),
        )
    else:
        sql = """\
INSERT INTO user_mcp_credentials (user_id, server_name, credential_type, encrypted_value, metadata, updated_at)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT (user_id, server_name) DO UPDATE SET
    encrypted_value = excluded.encrypted_value,
    credential_type = excluded.credential_type,
    metadata = excluded.metadata,
    updated_at = excluded.updated_at
"""
        params = (
            user_id,
            server_name,
            credential_type,
            encrypted,
            meta_json,
            datetime.now(timezone.utc).isoformat(),
        )

    with get_connection() as conn:
        cur = conn.cursor() if is_pg() else conn
        cur.execute(sql, params)
        if not is_pg():
            conn.commit()

    logger.info("凭据已存储: user=%s server=%s", user_id, server_name)


def get_credential(user_id: str, server_name: str) -> str | None:
    """获取用户的 MCP server 凭据（解密），不存在返回 None."""
    from crew.database import get_connection, is_pg

    if is_pg():
        sql = "SELECT encrypted_value FROM user_mcp_credentials WHERE user_id = %s AND server_name = %s"
    else:
        sql = (
            "SELECT encrypted_value FROM user_mcp_credentials WHERE user_id = ? AND server_name = ?"
        )

    with get_connection() as conn:
        cur = conn.cursor() if is_pg() else conn
        cur.execute(sql, (user_id, server_name))
        row = cur.fetchone()

    if row is None:
        return None

    encrypted = row[0] if isinstance(row, (tuple, list)) else row["encrypted_value"]
    try:
        return _decrypt(encrypted)
    except Exception:
        logger.error("凭据解密失败: user=%s server=%s", user_id, server_name)
        return None


def delete_credential(user_id: str, server_name: str) -> bool:
    """删除用户的 MCP server 凭据."""
    from crew.database import get_connection, is_pg

    if is_pg():
        sql = "DELETE FROM user_mcp_credentials WHERE user_id = %s AND server_name = %s"
    else:
        sql = "DELETE FROM user_mcp_credentials WHERE user_id = ? AND server_name = ?"

    with get_connection() as conn:
        cur = conn.cursor() if is_pg() else conn
        cur.execute(sql, (user_id, server_name))
        if is_pg():
            deleted = cur.rowcount > 0
        else:
            deleted = conn.total_changes > 0
            conn.commit()

    return deleted


def list_credentials(user_id: str) -> list[dict[str, Any]]:
    """列出用户的所有 MCP server 凭据（不返回明文 token）."""
    from crew.database import get_connection, is_pg

    if is_pg():
        sql = "SELECT server_name, credential_type, metadata, created_at, updated_at FROM user_mcp_credentials WHERE user_id = %s"
    else:
        sql = "SELECT server_name, credential_type, metadata, created_at, updated_at FROM user_mcp_credentials WHERE user_id = ?"

    with get_connection() as conn:
        cur = conn.cursor() if is_pg() else conn
        cur.execute(sql, (user_id,))
        rows = cur.fetchall()

    results = []
    for row in rows:
        if isinstance(row, (tuple, list)):
            results.append(
                {
                    "server_name": row[0],
                    "credential_type": row[1],
                    "metadata": json.loads(row[2]) if row[2] else {},
                    "created_at": str(row[3]),
                    "updated_at": str(row[4]),
                }
            )
        else:
            results.append(
                {
                    "server_name": row["server_name"],
                    "credential_type": row["credential_type"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "created_at": str(row["created_at"]),
                    "updated_at": str(row["updated_at"]),
                }
            )

    return results
