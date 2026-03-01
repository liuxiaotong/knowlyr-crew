"""配置存储 — 员工灵魂、讨论会、流水线配置的数据库持久化.

将配置从文件系统迁移到数据库，支持版本管理和远程访问。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from crew.database import get_connection, is_pg

logger = logging.getLogger(__name__)


# ── Schema 定义 ──

_PG_CREATE_EMPLOYEE_SOULS = """\
CREATE TABLE IF NOT EXISTS employee_souls (
    employee_name VARCHAR(255) PRIMARY KEY,
    content TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(255),
    version INTEGER NOT NULL DEFAULT 1,
    metadata TEXT
)
"""

_PG_CREATE_DISCUSSIONS = """\
CREATE TABLE IF NOT EXISTS discussions (
    name VARCHAR(255) PRIMARY KEY,
    yaml_content TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
)
"""

_PG_CREATE_PIPELINES = """\
CREATE TABLE IF NOT EXISTS pipelines (
    name VARCHAR(255) PRIMARY KEY,
    yaml_content TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
)
"""

_PG_CREATE_SOUL_HISTORY = """\
CREATE TABLE IF NOT EXISTS employee_soul_history (
    id SERIAL PRIMARY KEY,
    employee_name VARCHAR(255) NOT NULL,
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    updated_by VARCHAR(255),
    UNIQUE(employee_name, version)
)
"""

_PG_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_soul_history_employee ON employee_soul_history(employee_name)",
    "CREATE INDEX IF NOT EXISTS idx_discussions_updated ON discussions(updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_pipelines_updated ON pipelines(updated_at)",
]


def init_config_tables() -> None:
    """初始化配置存储表（仅 PG 模式）."""
    if not is_pg():
        logger.debug("SQLite 模式，跳过配置表初始化")
        return

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(_PG_CREATE_EMPLOYEE_SOULS)
        cur.execute(_PG_CREATE_DISCUSSIONS)
        cur.execute(_PG_CREATE_PIPELINES)
        cur.execute(_PG_CREATE_SOUL_HISTORY)
        for sql in _PG_CREATE_INDEXES:
            cur.execute(sql)

    logger.info("配置存储表初始化完成")


# ── Employee Soul 操作 ──


def get_soul(employee_name: str) -> dict[str, Any] | None:
    """读取员工灵魂配置.

    Returns:
        包含 content, version, updated_at, updated_by 的字典，不存在返回 None
    """
    if not is_pg():
        return None

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT content, version, updated_at, updated_by, metadata FROM employee_souls WHERE employee_name = %s",
            (employee_name,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "employee_name": employee_name,
            "content": row[0],
            "version": row[1],
            "updated_at": row[2].isoformat() if row[2] else None,
            "updated_by": row[3],
            "metadata": json.loads(row[4]) if row[4] else {},
        }


def update_soul(
    employee_name: str,
    content: str,
    updated_by: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """更新员工灵魂配置（自动版本递增 + 历史记录）.

    Returns:
        更新后的配置字典
    """
    if not is_pg():
        raise RuntimeError("配置存储仅支持 PG 模式")

    now = datetime.now(timezone.utc)
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

    with get_connection() as conn:
        cur = conn.cursor()

        # 获取当前版本
        cur.execute(
            "SELECT version, content FROM employee_souls WHERE employee_name = %s",
            (employee_name,),
        )
        row = cur.fetchone()

        if row:
            # 更新现有记录
            old_version = row[0]
            old_content = row[1]
            new_version = old_version + 1

            # 保存历史版本
            cur.execute(
                """
                INSERT INTO employee_soul_history (employee_name, version, content, updated_at, updated_by)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (employee_name, version) DO NOTHING
                """,
                (employee_name, old_version, old_content, now, updated_by),
            )

            # 更新当前版本
            cur.execute(
                """
                UPDATE employee_souls
                SET content = %s, version = %s, updated_at = %s, updated_by = %s, metadata = %s
                WHERE employee_name = %s
                """,
                (content, new_version, now, updated_by, metadata_json, employee_name),
            )
        else:
            # 插入新记录
            new_version = 1
            cur.execute(
                """
                INSERT INTO employee_souls (employee_name, content, version, updated_at, updated_by, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (employee_name, content, new_version, now, updated_by, metadata_json),
            )

    return {
        "employee_name": employee_name,
        "version": new_version,
        "updated_at": now.isoformat(),
        "updated_by": updated_by,
    }


def list_souls() -> list[dict[str, Any]]:
    """列出所有员工灵魂配置（不含 content）."""
    if not is_pg():
        return []

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT employee_name, version, updated_at, updated_by FROM employee_souls ORDER BY employee_name"
        )
        rows = cur.fetchall()
        return [
            {
                "employee_name": row[0],
                "version": row[1],
                "updated_at": row[2].isoformat() if row[2] else None,
                "updated_by": row[3],
            }
            for row in rows
        ]


# ── Discussion 操作 ──


def get_discussion(name: str) -> dict[str, Any] | None:
    """读取讨论会配置."""
    if not is_pg():
        return None

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT yaml_content, description, created_at, updated_at, metadata FROM discussions WHERE name = %s",
            (name,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "name": name,
            "yaml_content": row[0],
            "description": row[1],
            "created_at": row[2].isoformat() if row[2] else None,
            "updated_at": row[3].isoformat() if row[3] else None,
            "metadata": json.loads(row[4]) if row[4] else {},
        }


def create_discussion(
    name: str,
    yaml_content: str,
    description: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """创建讨论会配置."""
    if not is_pg():
        raise RuntimeError("配置存储仅支持 PG 模式")

    now = datetime.now(timezone.utc)
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO discussions (name, yaml_content, description, created_at, updated_at, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (name, yaml_content, description, now, now, metadata_json),
        )

    return {
        "name": name,
        "description": description,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }


def update_discussion(
    name: str,
    yaml_content: str,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """更新讨论会配置."""
    if not is_pg():
        raise RuntimeError("配置存储仅支持 PG 模式")

    now = datetime.now(timezone.utc)

    with get_connection() as conn:
        cur = conn.cursor()

        # 构建更新语句
        updates = ["yaml_content = %s", "updated_at = %s"]
        params: list[Any] = [yaml_content, now]

        if description is not None:
            updates.append("description = %s")
            params.append(description)

        if metadata is not None:
            updates.append("metadata = %s")
            params.append(json.dumps(metadata, ensure_ascii=False))

        params.append(name)

        cur.execute(
            f"UPDATE discussions SET {', '.join(updates)} WHERE name = %s",
            tuple(params),
        )

    return {"name": name, "updated_at": now.isoformat()}


def list_discussions() -> list[dict[str, Any]]:
    """列出所有讨论会配置（不含 yaml_content）."""
    if not is_pg():
        return []

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT name, description, created_at, updated_at FROM discussions ORDER BY name"
        )
        rows = cur.fetchall()
        return [
            {
                "name": row[0],
                "description": row[1],
                "created_at": row[2].isoformat() if row[2] else None,
                "updated_at": row[3].isoformat() if row[3] else None,
            }
            for row in rows
        ]


# ── Pipeline 操作 ──


def get_pipeline(name: str) -> dict[str, Any] | None:
    """读取流水线配置."""
    if not is_pg():
        return None

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT yaml_content, description, created_at, updated_at, metadata FROM pipelines WHERE name = %s",
            (name,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "name": name,
            "yaml_content": row[0],
            "description": row[1],
            "created_at": row[2].isoformat() if row[2] else None,
            "updated_at": row[3].isoformat() if row[3] else None,
            "metadata": json.loads(row[4]) if row[4] else {},
        }


def create_pipeline(
    name: str,
    yaml_content: str,
    description: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """创建流水线配置."""
    if not is_pg():
        raise RuntimeError("配置存储仅支持 PG 模式")

    now = datetime.now(timezone.utc)
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO pipelines (name, yaml_content, description, created_at, updated_at, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (name, yaml_content, description, now, now, metadata_json),
        )

    return {
        "name": name,
        "description": description,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }


def update_pipeline(
    name: str,
    yaml_content: str,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """更新流水线配置."""
    if not is_pg():
        raise RuntimeError("配置存储仅支持 PG 模式")

    now = datetime.now(timezone.utc)

    with get_connection() as conn:
        cur = conn.cursor()

        # 构建更新语句
        updates = ["yaml_content = %s", "updated_at = %s"]
        params: list[Any] = [yaml_content, now]

        if description is not None:
            updates.append("description = %s")
            params.append(description)

        if metadata is not None:
            updates.append("metadata = %s")
            params.append(json.dumps(metadata, ensure_ascii=False))

        params.append(name)

        cur.execute(
            f"UPDATE pipelines SET {', '.join(updates)} WHERE name = %s",
            tuple(params),
        )

    return {"name": name, "updated_at": now.isoformat()}


def list_pipelines() -> list[dict[str, Any]]:
    """列出所有流水线配置（不含 yaml_content）."""
    if not is_pg():
        return []

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT name, description, created_at, updated_at FROM pipelines ORDER BY name")
        rows = cur.fetchall()
        return [
            {
                "name": row[0],
                "description": row[1],
                "created_at": row[2].isoformat() if row[2] else None,
                "updated_at": row[3].isoformat() if row[3] else None,
            }
            for row in rows
        ]
