"""MCP Gateway 凭据管理 — 用户级 MCP 凭据的数据库加密存储.

每个用户的 OAuth token 存储在 ``user_mcp_credentials`` 表中，
使用 Fernet (AES-128-CBC + HMAC-SHA256) 加密
（通过环境变量 ``MCP_CREDENTIAL_KEY`` 提供密钥）。
"""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timezone
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

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
    # HKDF-SHA256 派生 32 字节，url-safe base64 编码作为 Fernet key
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=b"mcp-credential-v1", info=b"fernet-key")
    derived = hkdf.derive(raw.encode())
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
    mcp_server TEXT NOT NULL,
    access_token TEXT NOT NULL DEFAULT '',
    refresh_token TEXT NOT NULL DEFAULT '',
    token_expires_at TIMESTAMP,
    scopes TEXT NOT NULL DEFAULT '',
    tenant_id TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, mcp_server)
)"""
    else:
        sql = """\
CREATE TABLE IF NOT EXISTS user_mcp_credentials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    mcp_server TEXT NOT NULL,
    access_token TEXT NOT NULL DEFAULT '',
    refresh_token TEXT NOT NULL DEFAULT '',
    token_expires_at TIMESTAMP,
    scopes TEXT NOT NULL DEFAULT '',
    tenant_id TEXT NOT NULL DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, mcp_server)
)"""

    with get_connection() as conn:
        cur = conn.cursor() if is_pg() else conn
        cur.execute(sql)
        if not is_pg():
            conn.commit()


def store_credential(
    user_id: str,
    mcp_server: str,
    access_token: str,
    *,
    refresh_token: str = "",
    token_expires_at: datetime | None = None,
    scopes: str = "",
    tenant_id: str = "",
) -> None:
    """存储用户的 MCP server 凭据（加密）."""
    from crew.database import get_connection, is_pg

    enc_access = _encrypt(access_token)
    enc_refresh = _encrypt(refresh_token) if refresh_token else ""
    expires_val = (
        token_expires_at
        if is_pg()
        else (token_expires_at.isoformat() if token_expires_at else None)
    )
    now = datetime.now(timezone.utc)

    if is_pg():
        sql = """\
INSERT INTO user_mcp_credentials (user_id, mcp_server, access_token, refresh_token, token_expires_at, scopes, tenant_id, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (user_id, mcp_server) DO UPDATE SET
    access_token = EXCLUDED.access_token,
    refresh_token = EXCLUDED.refresh_token,
    token_expires_at = EXCLUDED.token_expires_at,
    scopes = EXCLUDED.scopes,
    tenant_id = EXCLUDED.tenant_id,
    updated_at = EXCLUDED.updated_at
"""
        params = (user_id, mcp_server, enc_access, enc_refresh, expires_val, scopes, tenant_id, now)
    else:
        sql = """\
INSERT INTO user_mcp_credentials (user_id, mcp_server, access_token, refresh_token, token_expires_at, scopes, tenant_id, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (user_id, mcp_server) DO UPDATE SET
    access_token = excluded.access_token,
    refresh_token = excluded.refresh_token,
    token_expires_at = excluded.token_expires_at,
    scopes = excluded.scopes,
    tenant_id = excluded.tenant_id,
    updated_at = excluded.updated_at
"""
        params = (
            user_id,
            mcp_server,
            enc_access,
            enc_refresh,
            expires_val,
            scopes,
            tenant_id,
            now.isoformat(),
        )

    with get_connection() as conn:
        cur = conn.cursor() if is_pg() else conn
        cur.execute(sql, params)
        if not is_pg():
            conn.commit()

    logger.info("凭据已存储: user=%s server=%s tenant=%s", user_id, mcp_server, tenant_id)


def get_credential(user_id: str, mcp_server: str) -> dict[str, Any] | None:
    """获取用户的 MCP server 凭据（解密），不存在返回 None."""
    from crew.database import get_connection, is_pg

    if is_pg():
        sql = "SELECT access_token, refresh_token, token_expires_at, scopes, tenant_id FROM user_mcp_credentials WHERE user_id = %s AND mcp_server = %s"
    else:
        sql = "SELECT access_token, refresh_token, token_expires_at, scopes, tenant_id FROM user_mcp_credentials WHERE user_id = ? AND mcp_server = ?"

    with get_connection() as conn:
        cur = conn.cursor() if is_pg() else conn
        cur.execute(sql, (user_id, mcp_server))
        row = cur.fetchone()

    if row is None:
        return None
    if isinstance(row, (tuple, list)):
        enc_a, enc_r, exp, scp, tid = row
    else:
        enc_a, enc_r, exp, scp, tid = (
            row["access_token"],
            row["refresh_token"],
            row["token_expires_at"],
            row["scopes"],
            row["tenant_id"],
        )
    try:
        da = _decrypt(enc_a) if enc_a else ""
    except Exception:
        logger.error("access_token 解密失败: user=%s server=%s", user_id, mcp_server)
        return None
    dr = ""
    if enc_r:
        try:
            dr = _decrypt(enc_r)
        except Exception:
            logger.warning("refresh_token 解密失败: user=%s server=%s", user_id, mcp_server)
    return {
        "access_token": da,
        "refresh_token": dr,
        "token_expires_at": str(exp) if exp else None,
        "scopes": scp or "",
        "tenant_id": tid or "",
    }


def delete_credential(user_id: str, mcp_server: str) -> bool:
    """删除用户的 MCP server 凭据."""
    from crew.database import get_connection, is_pg

    if is_pg():
        sql = "DELETE FROM user_mcp_credentials WHERE user_id = %s AND mcp_server = %s"
    else:
        sql = "DELETE FROM user_mcp_credentials WHERE user_id = ? AND mcp_server = ?"

    with get_connection() as conn:
        cur = conn.cursor() if is_pg() else conn
        cur.execute(sql, (user_id, mcp_server))
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
        sql = "SELECT mcp_server, scopes, tenant_id, token_expires_at, created_at, updated_at FROM user_mcp_credentials WHERE user_id = %s"
    else:
        sql = "SELECT mcp_server, scopes, tenant_id, token_expires_at, created_at, updated_at FROM user_mcp_credentials WHERE user_id = ?"

    with get_connection() as conn:
        cur = conn.cursor() if is_pg() else conn
        cur.execute(sql, (user_id,))
        rows = cur.fetchall()

    results = []
    for row in rows:
        if isinstance(row, (tuple, list)):
            results.append(
                {
                    "mcp_server": row[0],
                    "scopes": row[1] or "",
                    "tenant_id": row[2] or "",
                    "token_expires_at": str(row[3]) if row[3] else None,
                    "created_at": str(row[4]),
                    "updated_at": str(row[5]),
                }
            )
        else:
            results.append(
                {
                    "mcp_server": row["mcp_server"],
                    "scopes": row["scopes"] or "",
                    "tenant_id": row["tenant_id"] or "",
                    "token_expires_at": str(row["token_expires_at"])
                    if row["token_expires_at"]
                    else None,
                    "created_at": str(row["created_at"]),
                    "updated_at": str(row["updated_at"]),
                }
            )

    return results


def has_credential(user_id: str, mcp_server: str) -> bool:
    """检查用户是否有指定 MCP server 的凭据."""
    from crew.database import get_connection, is_pg

    if is_pg():
        sql = "SELECT 1 FROM user_mcp_credentials WHERE user_id = %s AND mcp_server = %s LIMIT 1"
    else:
        sql = "SELECT 1 FROM user_mcp_credentials WHERE user_id = ? AND mcp_server = ? LIMIT 1"
    with get_connection() as conn:
        cur = conn.cursor() if is_pg() else conn
        cur.execute(sql, (user_id, mcp_server))
        return cur.fetchone() is not None
