"""配置存储 — 员工灵魂、讨论会、流水线配置的数据库持久化.

将配置从文件系统迁移到数据库，支持版本管理和远程访问。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from crew.database import get_connection, is_pg
from crew.tenant import DEFAULT_ADMIN_TENANT_ID

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

_PG_CREATE_MEMORIES = """\
CREATE TABLE IF NOT EXISTS memories (
    id VARCHAR(12) PRIMARY KEY,
    employee VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    category VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    source_session VARCHAR(255) DEFAULT '',
    confidence FLOAT DEFAULT 1.0,
    superseded_by VARCHAR(12) DEFAULT '',
    ttl_days INTEGER DEFAULT 0,
    importance INTEGER DEFAULT 3,
    last_accessed TIMESTAMP,
    tags TEXT[],
    shared BOOLEAN DEFAULT FALSE,
    visibility VARCHAR(20) DEFAULT 'open',
    trigger_condition TEXT DEFAULT '',
    applicability TEXT[],
    origin_employee VARCHAR(255) DEFAULT '',
    verified_count INTEGER DEFAULT 0
)
"""

_PG_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_soul_history_employee ON employee_soul_history(employee_name)",
    "CREATE INDEX IF NOT EXISTS idx_discussions_updated ON discussions(updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_pipelines_updated ON pipelines(updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_memories_employee ON memories(employee)",
    "CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)",
    "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_memories_shared ON memories(shared)",
    "CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN(tags)",
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
        cur.execute(_PG_CREATE_MEMORIES)
        for sql in _PG_CREATE_INDEXES:
            cur.execute(sql)

    logger.info("配置存储表初始化完成")


# ── Employee Soul 操作 ──


def get_soul(employee_name: str, tenant_id: str = DEFAULT_ADMIN_TENANT_ID) -> dict[str, Any] | None:
    """读取员工灵魂配置.

    Returns:
        包含 content, version, updated_at, updated_by 的字典，不存在返回 None
    """
    if not is_pg():
        return None

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT content, version, updated_at, updated_by, metadata FROM employee_souls WHERE employee_name = %s AND tenant_id = %s",
            (employee_name, tenant_id),
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
    tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
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
            "SELECT version, content FROM employee_souls WHERE employee_name = %s AND tenant_id = %s",
            (employee_name, tenant_id),
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
                INSERT INTO employee_soul_history (employee_name, version, content, updated_at, updated_by, tenant_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, employee_name, version) DO NOTHING
                """,
                (employee_name, old_version, old_content, now, updated_by, tenant_id),
            )

            # 更新当前版本
            cur.execute(
                """
                UPDATE employee_souls
                SET content = %s, version = %s, updated_at = %s, updated_by = %s, metadata = %s
                WHERE employee_name = %s AND tenant_id = %s
                """,
                (content, new_version, now, updated_by, metadata_json, employee_name, tenant_id),
            )
        else:
            # 插入新记录
            new_version = 1
            cur.execute(
                """
                INSERT INTO employee_souls (employee_name, content, version, updated_at, updated_by, metadata, tenant_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (employee_name, content, new_version, now, updated_by, metadata_json, tenant_id),
            )

    return {
        "employee_name": employee_name,
        "version": new_version,
        "updated_at": now.isoformat(),
        "updated_by": updated_by,
    }


def list_souls(tenant_id: str = DEFAULT_ADMIN_TENANT_ID) -> list[dict[str, Any]]:
    """列出所有员工灵魂配置（不含 content）."""
    if not is_pg():
        return []

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT employee_name, version, updated_at, updated_by FROM employee_souls WHERE tenant_id = %s ORDER BY employee_name",
            (tenant_id,),
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


def _generate_unique_agent_id() -> str:
    """生成唯一的 agent_id（AI + 4位随机数字）."""
    import random
    from pathlib import Path

    # 确定头像目录
    static_dir = Path(__file__).parent.parent.parent / "static" / "avatars"
    static_dir.mkdir(parents=True, exist_ok=True)

    # 尝试生成唯一 ID
    for _ in range(100):
        num = random.randint(1000, 9999)
        agent_id = f"AI{num}"
        avatar_path = static_dir / f"{agent_id}.webp"
        if not avatar_path.exists():
            return agent_id

    raise RuntimeError("无法生成唯一的 agent_id（100 次尝试后仍冲突）")


def _create_employee_files(
    name: str,
    character_name: str,
    metadata: dict[str, Any],
    soul_content: str,
) -> None:
    """创建员工的文件系统结构（employee.yaml + soul.md）."""
    from pathlib import Path

    import yaml

    # 确定员工目录
    emp_dir = Path(__file__).parent.parent.parent / "private" / "employees" / name
    emp_dir.mkdir(parents=True, exist_ok=True)

    # 创建 employee.yaml
    yaml_content = {
        "name": name,
        "character_name": character_name,
        "display_name": metadata.get("display_name", ""),
        "description": metadata.get("description", ""),
        "model": metadata.get("model", "claude-sonnet-4-6"),
        "model_tier": metadata.get("model_tier", "claude"),
        "tags": metadata.get("tags", []),
        "agent_id": metadata.get("agent_id"),
        "agent_status": metadata.get("agent_status", "active"),
        "avatar_prompt": metadata.get("avatar_prompt", ""),
    }

    yaml_path = emp_dir / "employee.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    # 创建 soul.md
    soul_path = emp_dir / "soul.md"
    with open(soul_path, "w", encoding="utf-8") as f:
        f.write(soul_content)

    logger.info("已创建员工文件: %s", emp_dir)


def create_employee(
    name: str,
    character_name: str,
    display_name: str = "",
    description: str = "",
    model: str = "claude-sonnet-4-6",
    model_tier: str = "claude",
    tags: list[str] | None = None,
    soul_content: str = "",
    agent_status: str = "active",
    avatar_prompt: str = "",
    tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
) -> dict[str, Any]:
    """创建新员工（数据库 + 文件系统）.

    Args:
        name: 员工标识（slug，仅 [a-z0-9-]）
        character_name: 角色名/中文名
        display_name: 显示名称（可选）
        description: 职责描述
        model: 使用的模型
        model_tier: 模型档位
        tags: 标签列表
        soul_content: 初始 soul 配置
        agent_status: 员工状态（active/frozen/inactive）
        avatar_prompt: 头像生成 prompt（可选）

    Returns:
        创建结果字典（包含 agent_id, employee_name, version, created_at）

    Raises:
        ValueError: 员工已存在
        RuntimeError: agent_id 生成失败或配置存储不可用
    """
    if not is_pg():
        raise RuntimeError("配置存储仅支持 PG 模式")

    # 1. 检查员工是否已存在（同一租户内）
    if get_soul(character_name, tenant_id=tenant_id):
        raise ValueError(f"employee already exists: {character_name}")

    # 2. 生成唯一的 agent_id
    agent_id = _generate_unique_agent_id()

    # 3. 准备元数据
    metadata = {
        "name": name,
        "agent_id": agent_id,
        "display_name": display_name,
        "description": description,
        "model": model,
        "model_tier": model_tier,
        "tags": tags or [],
        "agent_status": agent_status,
        "avatar_prompt": avatar_prompt,
    }

    # 4. 写入数据库
    result = update_soul(
        employee_name=character_name,
        content=soul_content,
        updated_by="create_employee",
        metadata=metadata,
        tenant_id=tenant_id,
    )

    # 5. 创建文件系统
    try:
        _create_employee_files(name, character_name, metadata, soul_content)
    except Exception as e:
        logger.error("创建员工文件失败: %s", e, exc_info=True)
        # 不抛出异常，因为数据库已写入

    # 6. 返回结果（添加 agent_id）
    result["agent_id"] = agent_id
    return result


# ── Discussion 操作 ──


def get_discussion(name: str, tenant_id: str = DEFAULT_ADMIN_TENANT_ID) -> dict[str, Any] | None:
    """读取讨论会配置."""
    if not is_pg():
        return None

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT yaml_content, description, created_at, updated_at, metadata FROM discussions WHERE name = %s AND tenant_id = %s",
            (name, tenant_id),
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
    tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
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
            INSERT INTO discussions (name, yaml_content, description, created_at, updated_at, metadata, tenant_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (name, yaml_content, description, now, now, metadata_json, tenant_id),
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
    tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
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

        params.extend([name, tenant_id])

        cur.execute(
            f"UPDATE discussions SET {', '.join(updates)} WHERE name = %s AND tenant_id = %s",
            tuple(params),
        )

    return {"name": name, "updated_at": now.isoformat()}


def list_discussions(tenant_id: str = DEFAULT_ADMIN_TENANT_ID) -> list[dict[str, Any]]:
    """列出所有讨论会配置（不含 yaml_content）."""
    if not is_pg():
        return []

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT name, description, created_at, updated_at FROM discussions WHERE tenant_id = %s ORDER BY name",
            (tenant_id,),
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


def get_pipeline(name: str, tenant_id: str = DEFAULT_ADMIN_TENANT_ID) -> dict[str, Any] | None:
    """读取流水线配置."""
    if not is_pg():
        return None

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT yaml_content, description, created_at, updated_at, metadata FROM pipelines WHERE name = %s AND tenant_id = %s",
            (name, tenant_id),
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
    tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
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
            INSERT INTO pipelines (name, yaml_content, description, created_at, updated_at, metadata, tenant_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (name, yaml_content, description, now, now, metadata_json, tenant_id),
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
    tenant_id: str = DEFAULT_ADMIN_TENANT_ID,
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

        params.extend([name, tenant_id])

        cur.execute(
            f"UPDATE pipelines SET {', '.join(updates)} WHERE name = %s AND tenant_id = %s",
            tuple(params),
        )

    return {"name": name, "updated_at": now.isoformat()}


def list_pipelines(tenant_id: str = DEFAULT_ADMIN_TENANT_ID) -> list[dict[str, Any]]:
    """列出所有流水线配置（不含 yaml_content）."""
    if not is_pg():
        return []

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT name, description, created_at, updated_at FROM pipelines WHERE tenant_id = %s ORDER BY name",
            (tenant_id,),
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
