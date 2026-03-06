"""Migration: 给核心表添加 tenant_id 字段 (Phase 1 多租户改造).

幂等执行：重复运行不会出错。

表清单:
  - memories: 加 tenant_id, 默认 'tenant_admin'
  - employee_souls: 加 tenant_id, 改 PK 为 (tenant_id, employee_name)
  - employee_soul_history: 加 tenant_id
  - discussions: 加 tenant_id, 改 PK 为 (tenant_id, name)
  - pipelines: 加 tenant_id, 改 PK 为 (tenant_id, name)
  - events: 加 tenant_id
  - entries: 加 tenant_id
  - memory_vectors: 加 tenant_id

用法::

    python scripts/migrate_add_tenant_id.py
    # 或
    CREW_DATABASE_URL=postgresql://... python scripts/migrate_add_tenant_id.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 确保项目 src 在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crew.database import get_connection, is_pg  # noqa: E402
from crew.tenant import init_tenant_tables  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TENANT = "tenant_admin"


def _column_exists(cur, table: str, column: str) -> bool:
    """检查列是否存在（PG information_schema）."""
    cur.execute(
        """
        SELECT 1 FROM information_schema.columns
        WHERE table_name = %s AND column_name = %s
        """,
        (table, column),
    )
    return cur.fetchone() is not None


def _constraint_exists(cur, constraint_name: str) -> bool:
    """检查约束是否存在."""
    cur.execute(
        """
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = %s
        """,
        (constraint_name,),
    )
    return cur.fetchone() is not None


def migrate() -> None:
    """执行 tenant_id 迁移."""
    if not is_pg():
        logger.error("此脚本仅支持 PostgreSQL 模式")
        sys.exit(1)

    # 0. 初始化 tenants 表
    init_tenant_tables()
    logger.info("tenants 表已就绪")

    with get_connection() as conn:
        cur = conn.cursor()

        # ── 1. memories 表 ──
        if not _column_exists(cur, "memories", "tenant_id"):
            cur.execute(
                f"ALTER TABLE memories ADD COLUMN tenant_id VARCHAR(64) NOT NULL DEFAULT '{DEFAULT_TENANT}'"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_tenant ON memories(tenant_id)")
            logger.info("memories: 已添加 tenant_id")
        else:
            logger.info("memories: tenant_id 已存在，跳过")

        # ── 2. employee_souls 表 ──
        if not _column_exists(cur, "employee_souls", "tenant_id"):
            cur.execute(
                f"ALTER TABLE employee_souls ADD COLUMN tenant_id VARCHAR(64) NOT NULL DEFAULT '{DEFAULT_TENANT}'"
            )
            # 改 PK: 先删旧 PK，再建联合唯一约束
            cur.execute(
                "ALTER TABLE employee_souls DROP CONSTRAINT IF EXISTS employee_souls_pkey"
            )
            cur.execute(
                "ALTER TABLE employee_souls ADD PRIMARY KEY (tenant_id, employee_name)"
            )
            logger.info("employee_souls: 已添加 tenant_id 并改 PK")
        else:
            logger.info("employee_souls: tenant_id 已存在，跳过")

        # ── 3. employee_soul_history 表 ──
        if not _column_exists(cur, "employee_soul_history", "tenant_id"):
            cur.execute(
                f"ALTER TABLE employee_soul_history ADD COLUMN tenant_id VARCHAR(64) NOT NULL DEFAULT '{DEFAULT_TENANT}'"
            )
            # 更新唯一约束
            cur.execute(
                "ALTER TABLE employee_soul_history DROP CONSTRAINT IF EXISTS employee_soul_history_employee_name_version_key"
            )
            cur.execute(
                """
                ALTER TABLE employee_soul_history
                ADD CONSTRAINT employee_soul_history_tenant_employee_version_key
                UNIQUE (tenant_id, employee_name, version)
                """
            )
            logger.info("employee_soul_history: 已添加 tenant_id")
        else:
            logger.info("employee_soul_history: tenant_id 已存在，跳过")

        # ── 4. discussions 表 ──
        if not _column_exists(cur, "discussions", "tenant_id"):
            cur.execute(
                f"ALTER TABLE discussions ADD COLUMN tenant_id VARCHAR(64) NOT NULL DEFAULT '{DEFAULT_TENANT}'"
            )
            cur.execute("ALTER TABLE discussions DROP CONSTRAINT IF EXISTS discussions_pkey")
            cur.execute("ALTER TABLE discussions ADD PRIMARY KEY (tenant_id, name)")
            logger.info("discussions: 已添加 tenant_id 并改 PK")
        else:
            logger.info("discussions: tenant_id 已存在，跳过")

        # ── 5. pipelines 表 ──
        if not _column_exists(cur, "pipelines", "tenant_id"):
            cur.execute(
                f"ALTER TABLE pipelines ADD COLUMN tenant_id VARCHAR(64) NOT NULL DEFAULT '{DEFAULT_TENANT}'"
            )
            cur.execute("ALTER TABLE pipelines DROP CONSTRAINT IF EXISTS pipelines_pkey")
            cur.execute("ALTER TABLE pipelines ADD PRIMARY KEY (tenant_id, name)")
            logger.info("pipelines: 已添加 tenant_id 并改 PK")
        else:
            logger.info("pipelines: tenant_id 已存在，跳过")

        # ── 6. events 表 ──
        if not _column_exists(cur, "events", "tenant_id"):
            cur.execute(
                f"ALTER TABLE events ADD COLUMN tenant_id VARCHAR(64) NOT NULL DEFAULT '{DEFAULT_TENANT}'"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_events_tenant ON events(tenant_id)")
            logger.info("events: 已添加 tenant_id")
        else:
            logger.info("events: tenant_id 已存在，跳过")

        # ── 7. entries 表 ──
        if not _column_exists(cur, "entries", "tenant_id"):
            cur.execute(
                f"ALTER TABLE entries ADD COLUMN tenant_id VARCHAR(64) NOT NULL DEFAULT '{DEFAULT_TENANT}'"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_entries_tenant ON entries(tenant_id)")
            logger.info("entries: 已添加 tenant_id")
        else:
            logger.info("entries: tenant_id 已存在，跳过")

        # ── 8. memory_vectors 表 ──
        if not _column_exists(cur, "memory_vectors", "tenant_id"):
            cur.execute(
                f"ALTER TABLE memory_vectors ADD COLUMN tenant_id VARCHAR(64) NOT NULL DEFAULT '{DEFAULT_TENANT}'"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_mv_tenant ON memory_vectors(tenant_id)"
            )
            logger.info("memory_vectors: 已添加 tenant_id")
        else:
            logger.info("memory_vectors: tenant_id 已存在，跳过")

    logger.info("Migration 完成！所有表已添加 tenant_id")


if __name__ == "__main__":
    migrate()
