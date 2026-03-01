"""Memory 数据库存储 — PostgreSQL 版本的 MemoryStore.

将记忆从 JSONL 文件迁移到数据库，提供与文件版本相同的接口。
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from crew.database import get_connection, is_pg

logger = logging.getLogger(__name__)


# ── Schema 定义 ──

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
    "CREATE INDEX IF NOT EXISTS idx_memories_employee ON memories(employee)",
    "CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)",
    "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_memories_shared ON memories(shared)",
    "CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN(tags)",
]


def init_memory_tables() -> None:
    """初始化 memories 表（仅 PG 模式）."""
    if not is_pg():
        logger.debug("SQLite 模式，跳过 memories 表初始化")
        return

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(_PG_CREATE_MEMORIES)
        for sql in _PG_CREATE_INDEXES:
            cur.execute(sql)

    logger.info("memories 表初始化完成")


# ── MemoryStoreDB 实现 ──


class MemoryStoreDB:
    """数据库版 MemoryStore.

    提供与文件版本相同的接口，但数据存储在 PostgreSQL 中。
    """

    def __init__(self, project_dir: Any = None):
        """初始化数据库存储.

        Args:
            project_dir: 项目目录（兼容参数，数据库版不使用）
        """
        self._project_dir = project_dir
        if not is_pg():
            raise RuntimeError("MemoryStoreDB 仅支持 PostgreSQL 模式")

    def _resolve_to_character_name(self, employee: str) -> str:
        """将 slug 或花名统一转换为花名（character_name）.

        与文件版本保持一致的逻辑。
        """
        if not employee or not isinstance(employee, str):
            return employee
        # 已经是中文名（含 CJK 字符）则直接返回
        if any("\u4e00" <= c <= "\u9fff" for c in employee):
            return employee
        try:
            from crew.discovery import discover_employees

            discovery = discover_employees(project_dir=self._project_dir)
            emp = discovery.get(employee)
            if emp and emp.character_name and isinstance(emp.character_name, str):
                return emp.character_name
        except Exception:
            pass
        return employee

    def add(
        self,
        employee: str,
        category: Literal["decision", "estimate", "finding", "correction", "pattern"],
        content: str,
        source_session: str = "",
        confidence: float = 1.0,
        ttl_days: int = 0,
        tags: list[str] | None = None,
        shared: bool = False,
        visibility: Literal["open", "private"] = "open",
        trigger_condition: str = "",
        applicability: list[str] | None = None,
        origin_employee: str = "",
    ) -> dict[str, Any]:
        """添加一条记忆.

        Returns:
            记忆字典（包含所有字段）
        """
        employee = self._resolve_to_character_name(employee)

        # 生成 ID
        entry_id = uuid.uuid4().hex[:12]
        created_at = datetime.now(timezone.utc)

        # pattern 默认共享
        if category == "pattern":
            shared = True

        # 准备数据
        tags_list = tags or []
        applicability_list = applicability or []
        origin_emp = origin_employee or employee

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO memories (
                    id, employee, created_at, category, content,
                    source_session, confidence, superseded_by, ttl_days,
                    importance, last_accessed, tags, shared, visibility,
                    trigger_condition, applicability, origin_employee, verified_count
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                """,
                (
                    entry_id, employee, created_at, category, content,
                    source_session, confidence, "", ttl_days,
                    3, None, tags_list, shared, visibility,
                    trigger_condition, applicability_list, origin_emp, 0
                ),
            )

        return {
            "id": entry_id,
            "employee": employee,
            "created_at": created_at.isoformat(),
            "category": category,
            "content": content,
            "source_session": source_session,
            "confidence": confidence,
            "superseded_by": "",
            "ttl_days": ttl_days,
            "importance": 3,
            "last_accessed": None,
            "tags": tags_list,
            "shared": shared,
            "visibility": visibility,
            "trigger_condition": trigger_condition,
            "applicability": applicability_list,
            "origin_employee": origin_emp,
            "verified_count": 0,
        }

    def query(
        self,
        employee: str,
        category: str | None = None,
        limit: int = 20,
        min_confidence: float = 0.0,
        include_expired: bool = False,
        max_visibility: str = "private",
        sort_by: str = "created_at",
        min_importance: int = 0,
        update_access: bool = False,
    ) -> list[dict[str, Any]]:
        """查询员工记忆.

        Args:
            employee: 员工名称
            category: 按类别过滤（可选）
            limit: 最大返回条数
            min_confidence: 最低置信度
            include_expired: 是否包含已过期条目
            max_visibility: 可见性上限
            sort_by: 排序方式
            min_importance: 最低重要性
            update_access: 是否更新 last_accessed

        Returns:
            记忆列表
        """
        employee = self._resolve_to_character_name(employee)

        # 构建查询
        conditions = ["employee = %s", "superseded_by = ''"]
        params: list[Any] = [employee]

        if category:
            conditions.append("category = %s")
            params.append(category)

        if min_confidence > 0:
            conditions.append("confidence >= %s")
            params.append(min_confidence)

        if min_importance > 0:
            conditions.append("importance >= %s")
            params.append(min_importance)

        if max_visibility != "private":
            conditions.append("visibility = 'open'")

        if not include_expired:
            # 过滤过期记忆：ttl_days > 0 且 created_at + ttl_days < now
            conditions.append(
                "(ttl_days = 0 OR created_at + (ttl_days || ' days')::interval > NOW())"
            )

        # 排序
        if sort_by == "importance":
            order_by = "importance DESC, created_at DESC"
        elif sort_by == "confidence":
            order_by = "confidence DESC, created_at DESC"
        else:
            order_by = "created_at DESC"

        # 执行查询
        with get_connection() as conn:
            cur = conn.cursor()
            sql = f"""
                SELECT id, employee, created_at, category, content,
                       source_session, confidence, superseded_by, ttl_days,
                       importance, last_accessed, tags, shared, visibility,
                       trigger_condition, applicability, origin_employee, verified_count
                FROM memories
                WHERE {' AND '.join(conditions)}
                ORDER BY {order_by}
                LIMIT %s
            """
            params.append(limit)
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        # 转换为字典
        results = []
        entry_ids = []
        for row in rows:
            entry = {
                "id": row[0],
                "employee": row[1],
                "created_at": row[2].isoformat() if row[2] else None,
                "category": row[3],
                "content": row[4],
                "source_session": row[5],
                "confidence": row[6],
                "superseded_by": row[7],
                "ttl_days": row[8],
                "importance": row[9],
                "last_accessed": row[10].isoformat() if row[10] else None,
                "tags": row[11] or [],
                "shared": row[12],
                "visibility": row[13],
                "trigger_condition": row[14],
                "applicability": row[15] or [],
                "origin_employee": row[16],
                "verified_count": row[17],
            }
            results.append(entry)
            entry_ids.append(row[0])

        # 更新访问时间
        if update_access and entry_ids:
            self._update_last_accessed(employee, entry_ids)

        return results

    def _update_last_accessed(self, employee: str, entry_ids: list[str]) -> None:
        """批量更新记忆的 last_accessed 时间戳."""
        if not entry_ids:
            return

        now = datetime.now(timezone.utc)
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE memories
                SET last_accessed = %s
                WHERE id = ANY(%s) AND employee = %s
                """,
                (now, entry_ids, employee),
            )

    def query_shared(
        self,
        tags: list[str] | None = None,
        exclude_employee: str = "",
        limit: int = 10,
        min_confidence: float = 0.3,
    ) -> list[dict[str, Any]]:
        """查询跨员工的共享记忆.

        Args:
            tags: 按标签过滤（任一匹配即可）
            exclude_employee: 排除指定员工
            limit: 最大返回条数
            min_confidence: 最低置信度

        Returns:
            共享记忆列表
        """
        exclude_employee = self._resolve_to_character_name(exclude_employee)

        conditions = ["shared = TRUE", "superseded_by = ''"]
        params: list[Any] = []

        if exclude_employee:
            conditions.append("employee != %s")
            params.append(exclude_employee)

        if min_confidence > 0:
            conditions.append("confidence >= %s")
            params.append(min_confidence)

        # 过滤过期记忆
        conditions.append(
            "(ttl_days = 0 OR created_at + (ttl_days || ' days')::interval > NOW())"
        )

        # 标签过滤（任一匹配）
        if tags:
            conditions.append("tags && %s")
            params.append(tags)

        with get_connection() as conn:
            cur = conn.cursor()
            sql = f"""
                SELECT id, employee, created_at, category, content,
                       source_session, confidence, superseded_by, ttl_days,
                       importance, last_accessed, tags, shared, visibility,
                       trigger_condition, applicability, origin_employee, verified_count
                FROM memories
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at DESC
                LIMIT %s
            """
            params.append(limit)
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "employee": row[1],
                "created_at": row[2].isoformat() if row[2] else None,
                "category": row[3],
                "content": row[4],
                "source_session": row[5],
                "confidence": row[6],
                "superseded_by": row[7],
                "ttl_days": row[8],
                "importance": row[9],
                "last_accessed": row[10].isoformat() if row[10] else None,
                "tags": row[11] or [],
                "shared": row[12],
                "visibility": row[13],
                "trigger_condition": row[14],
                "applicability": row[15] or [],
                "origin_employee": row[16],
                "verified_count": row[17],
            })

        return results

    def query_patterns(
        self,
        employee: str = "",
        applicability: list[str] | None = None,
        limit: int = 10,
        min_confidence: float = 0.3,
    ) -> list[dict[str, Any]]:
        """查询可复用的工作模式（跨员工）.

        Args:
            employee: 当前员工（排除自己的）
            applicability: 按适用范围标签过滤
            limit: 最大返回条数
            min_confidence: 最低置信度

        Returns:
            pattern 列表
        """
        conditions = ["category = 'pattern'", "superseded_by = ''"]
        params: list[Any] = []

        if employee:
            employee = self._resolve_to_character_name(employee)
            conditions.append("employee != %s")
            params.append(employee)

        if min_confidence > 0:
            conditions.append("confidence >= %s")
            params.append(min_confidence)

        # 过滤过期记忆
        conditions.append(
            "(ttl_days = 0 OR created_at + (ttl_days || ' days')::interval > NOW())"
        )

        # 适用范围过滤
        if applicability:
            conditions.append("applicability && %s")
            params.append(applicability)

        with get_connection() as conn:
            cur = conn.cursor()
            sql = f"""
                SELECT id, employee, created_at, category, content,
                       source_session, confidence, superseded_by, ttl_days,
                       importance, last_accessed, tags, shared, visibility,
                       trigger_condition, applicability, origin_employee, verified_count
                FROM memories
                WHERE {' AND '.join(conditions)}
                ORDER BY verified_count DESC, created_at DESC
                LIMIT %s
            """
            params.append(limit)
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "employee": row[1],
                "created_at": row[2].isoformat() if row[2] else None,
                "category": row[3],
                "content": row[4],
                "source_session": row[5],
                "confidence": row[6],
                "superseded_by": row[7],
                "ttl_days": row[8],
                "importance": row[9],
                "last_accessed": row[10].isoformat() if row[10] else None,
                "tags": row[11] or [],
                "shared": row[12],
                "visibility": row[13],
                "trigger_condition": row[14],
                "applicability": row[15] or [],
                "origin_employee": row[16],
                "verified_count": row[17],
            })

        return results

    def delete(self, entry_id: str, employee: str | None = None) -> bool:
        """删除指定的记忆条目.

        Args:
            entry_id: 记忆条目 ID
            employee: 员工名（可选）

        Returns:
            True 如果删除成功，False 如果未找到
        """
        with get_connection() as conn:
            cur = conn.cursor()
            if employee:
                employee = self._resolve_to_character_name(employee)
                cur.execute(
                    "DELETE FROM memories WHERE id = %s AND employee = %s",
                    (entry_id, employee),
                )
            else:
                cur.execute("DELETE FROM memories WHERE id = %s", (entry_id,))

            return cur.rowcount > 0

    def update_confidence(self, entry_id: str, confidence: float) -> bool:
        """更新记忆的置信度.

        Args:
            entry_id: 记忆条目 ID
            confidence: 新的置信度

        Returns:
            True 如果更新成功，False 如果未找到
        """
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE memories SET confidence = %s WHERE id = %s",
                (confidence, entry_id),
            )
            return cur.rowcount > 0

    def count(self, employee: str) -> int:
        """返回员工的有效记忆条数."""
        employee = self._resolve_to_character_name(employee)

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT COUNT(*)
                FROM memories
                WHERE employee = %s
                  AND superseded_by = ''
                  AND (ttl_days = 0 OR created_at + (ttl_days || ' days')::interval > NOW())
                """,
                (employee,),
            )
            row = cur.fetchone()
            return row[0] if row else 0

    def list_employees(self) -> list[str]:
        """列出有记忆的员工."""
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT employee FROM memories ORDER BY employee")
            rows = cur.fetchall()
            return [row[0] for row in rows]
