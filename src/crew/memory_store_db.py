"""Memory 数据库存储 — PostgreSQL 版本的 MemoryStore.

将记忆从 JSONL 文件迁移到数据库，提供与文件版本相同的接口。
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from crew.database import get_connection, is_pg
from crew.memory import MemoryEntry
from crew.tenant import DEFAULT_ADMIN_TENANT_ID

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

        # 幂等添加 classification 列（信息分级）
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE memories ADD COLUMN classification VARCHAR(20) DEFAULT 'internal';
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
        """)
        # 幂等添加 domain 列（职能域标签）
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE memories ADD COLUMN domain TEXT[] DEFAULT '{}';
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
        """)
        # 索引
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_classification ON memories(classification)"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories USING GIN(domain)")

    logger.info("memories 表初始化完成")


# ── MemoryStoreDB 实现 ──


class MemoryStoreDB:
    """数据库版 MemoryStore.

    提供与文件版本相同的接口，但数据存储在 PostgreSQL 中。
    """

    def __init__(self, project_dir: Any = None, tenant_id: str | None = None):
        """初始化数据库存储.

        Args:
            project_dir: 项目目录（兼容参数，数据库版不使用）
            tenant_id: 租户 ID（None 则使用默认管理员租户，向后兼容）
        """
        self._project_dir = project_dir
        self._tenant_id = tenant_id or DEFAULT_ADMIN_TENANT_ID
        if not is_pg():
            raise RuntimeError("MemoryStoreDB 仅支持 PostgreSQL 模式")
        import psycopg2.extras

        self._dict_cursor_factory = psycopg2.extras.RealDictCursor

    def resolve_to_character_name(self, employee: str) -> str:
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

    def _row_to_entry(self, row: dict) -> MemoryEntry:
        """将数据库行（RealDictCursor dict）转换为 MemoryEntry."""
        return MemoryEntry(
            id=row["id"],
            employee=row["employee"],
            created_at=row["created_at"].isoformat()
            if hasattr(row["created_at"], "isoformat")
            else str(row["created_at"]),
            category=row["category"],
            content=row["content"],
            source_session=row.get("source_session") or "",
            confidence=float(row.get("confidence", 1.0)),
            superseded_by=row.get("superseded_by") or "",
            ttl_days=int(row.get("ttl_days", 0)),
            importance=int(row.get("importance", 3)),
            last_accessed=row["last_accessed"].isoformat()
            if row.get("last_accessed") and hasattr(row["last_accessed"], "isoformat")
            else (str(row["last_accessed"]) if row.get("last_accessed") else ""),
            tags=list(row.get("tags") or []),
            shared=bool(row.get("shared", False)),
            visibility=row.get("visibility") or "open",
            trigger_condition=row.get("trigger_condition") or "",
            applicability=list(row.get("applicability") or []),
            origin_employee=row.get("origin_employee") or "",
            classification=row.get("classification") or "internal",
            domain=list(row.get("domain") or []),
        )

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
        classification: Literal["public", "internal", "restricted", "confidential"] = "internal",
        domain: list[str] | None = None,
    ) -> MemoryEntry:
        """添加一条记忆.

        Returns:
            MemoryEntry 对象
        """
        employee = self.resolve_to_character_name(employee)

        # 防御性截断（S4：以防 API 层校验被绕过）
        content = content[:5000] if len(content) > 5000 else content

        # 12 位 hex = 48 bit 熵，碰撞概率 ~1/2.8e14，当前数据量安全
        entry_id = uuid.uuid4().hex[:12]
        created_at = datetime.now(timezone.utc)

        # pattern 默认共享
        if category == "pattern":
            shared = True

        # 准备数据
        tags_list = tags or []
        applicability_list = applicability or []
        origin_emp = origin_employee or employee
        domain_list = domain or []

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO memories (
                    id, employee, created_at, category, content,
                    source_session, confidence, superseded_by, ttl_days,
                    importance, last_accessed, tags, shared, visibility,
                    trigger_condition, applicability, origin_employee, verified_count,
                    classification, domain, tenant_id
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s
                )
                """,
                (
                    entry_id,
                    employee,
                    created_at,
                    category,
                    content,
                    source_session,
                    confidence,
                    "",
                    ttl_days,
                    3,
                    None,
                    tags_list,
                    shared,
                    visibility,
                    trigger_condition,
                    applicability_list,
                    origin_emp,
                    0,
                    classification,
                    domain_list,
                    self._tenant_id,
                ),
            )

            # [W10] 容量管理：每个租户每个员工最多保留 500 条
            MAX_ENTRIES = 500
            cur.execute(
                "SELECT count(*) FROM memories WHERE employee = %s AND tenant_id = %s AND (superseded_by = '' OR superseded_by IS NULL)",
                (employee, self._tenant_id),
            )
            count = cur.fetchone()[0]
            if count > MAX_ENTRIES:
                # 删除最旧的超出部分
                cur.execute(
                    """
                    DELETE FROM memories WHERE id IN (
                        SELECT id FROM memories
                        WHERE employee = %s AND tenant_id = %s AND (superseded_by = '' OR superseded_by IS NULL)
                        ORDER BY created_at ASC
                        LIMIT %s
                    )
                    """,
                    (employee, self._tenant_id, count - MAX_ENTRIES),
                )

        return MemoryEntry(
            id=entry_id,
            employee=employee,
            created_at=created_at.isoformat(),
            category=category,
            content=content,
            source_session=source_session,
            confidence=confidence,
            superseded_by="",
            ttl_days=ttl_days,
            importance=3,
            last_accessed="",
            tags=tags_list,
            shared=shared,
            visibility=visibility,
            trigger_condition=trigger_condition,
            applicability=applicability_list,
            origin_employee=origin_emp,
            classification=classification,
            domain=domain_list,
        )

    # 信息分级等级序（用于 classification_max 过滤）
    _CLASSIFICATION_LEVELS = {
        "public": 0,
        "internal": 1,
        "restricted": 2,
        "confidential": 3,
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
        classification_max: str | None = None,
        allowed_domains: list[str] | None = None,
        include_confidential: bool = False,
    ) -> list[MemoryEntry]:
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
            classification_max: 最高信息分级（可选，按等级过滤）
            allowed_domains: 允许的职能域（可选，restricted 级别需域匹配）
            include_confidential: 是否包含 confidential 级别记忆（默认 False）

        Returns:
            记忆列表（MemoryEntry）
        """
        employee = self.resolve_to_character_name(employee)

        # 构建查询（租户隔离）
        conditions = [
            "employee = %s",
            "tenant_id = %s",
            "(superseded_by = '' OR superseded_by IS NULL)",
        ]
        params: list[Any] = [employee, self._tenant_id]

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

        # 信息分级过滤
        if not include_confidential:
            conditions.append("COALESCE(classification, 'internal') != 'confidential'")

        if classification_max is not None:
            max_level = self._CLASSIFICATION_LEVELS.get(classification_max, 1)
            allowed_classifications = [
                k for k, v in self._CLASSIFICATION_LEVELS.items() if v <= max_level
            ]
            placeholders = ", ".join(["%s"] * len(allowed_classifications))
            conditions.append(f"COALESCE(classification, 'internal') IN ({placeholders})")
            params.extend(allowed_classifications)

        # [W7] restricted 域匹配移到 SQL 层
        if allowed_domains is not None:
            domain_placeholders = ", ".join(["%s"] * len(allowed_domains))
            conditions.append(
                f"(COALESCE(classification, 'internal') != 'restricted' OR domain && ARRAY[{domain_placeholders}]::text[])"
            )
            params.extend(allowed_domains)

        # 排序
        if sort_by == "importance":
            order_by = "importance DESC, created_at DESC"
        elif sort_by == "confidence":
            order_by = "confidence DESC, created_at DESC"
        else:
            order_by = "created_at DESC"

        # 执行查询
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
            sql = f"""
                SELECT id, employee, created_at, category, content,
                       source_session, confidence, superseded_by, ttl_days,
                       importance, last_accessed, tags, shared, visibility,
                       trigger_condition, applicability, origin_employee, verified_count,
                       classification, domain
                FROM memories
                WHERE {" AND ".join(conditions)}
                ORDER BY {order_by}
                LIMIT %s
            """
            params.append(limit)
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        # 转换为 MemoryEntry
        results: list[MemoryEntry] = []
        entry_ids: list[str] = []
        for row in rows:
            entry = self._row_to_entry(row)
            results.append(entry)
            entry_ids.append(entry.id)

        # Phase 4：审计日志
        logger.info(
            "memory_access: employee=%s classification_max=%s allowed_domains=%s "
            "include_confidential=%s returned=%d channel=unknown",
            employee,
            classification_max or "none",
            allowed_domains or "none",
            include_confidential,
            len(results),
        )

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
                WHERE id = ANY(%s) AND employee = %s AND tenant_id = %s
                """,
                (now, entry_ids, employee, self._tenant_id),
            )

    def query_shared(
        self,
        tags: list[str] | None = None,
        exclude_employee: str = "",
        limit: int = 10,
        min_confidence: float = 0.3,
    ) -> list[MemoryEntry]:
        """查询跨员工的共享记忆.

        Args:
            tags: 按标签过滤（任一匹配即可）
            exclude_employee: 排除指定员工
            limit: 最大返回条数
            min_confidence: 最低置信度

        Returns:
            共享记忆列表（MemoryEntry）
        """
        exclude_employee = self.resolve_to_character_name(exclude_employee)

        conditions = [
            "shared = TRUE",
            "tenant_id = %s",
            "(superseded_by = '' OR superseded_by IS NULL)",
        ]
        params: list[Any] = [self._tenant_id]

        if exclude_employee:
            conditions.append("employee != %s")
            params.append(exclude_employee)

        if min_confidence > 0:
            conditions.append("confidence >= %s")
            params.append(min_confidence)

        # 过滤过期记忆
        conditions.append("(ttl_days = 0 OR created_at + (ttl_days || ' days')::interval > NOW())")

        # 标签过滤（任一匹配）
        if tags:
            conditions.append("tags && %s")
            params.append(tags)

        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
            sql = f"""
                SELECT id, employee, created_at, category, content,
                       source_session, confidence, superseded_by, ttl_days,
                       importance, last_accessed, tags, shared, visibility,
                       trigger_condition, applicability, origin_employee, verified_count,
                       classification, domain
                FROM memories
                WHERE {" AND ".join(conditions)}
                ORDER BY created_at DESC
                LIMIT %s
            """
            params.append(limit)
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        results: list[MemoryEntry] = []
        for row in rows:
            results.append(self._row_to_entry(row))

        return results

    def query_patterns(
        self,
        employee: str = "",
        applicability: list[str] | None = None,
        limit: int = 10,
        min_confidence: float = 0.3,
    ) -> list[MemoryEntry]:
        """查询可复用的工作模式（跨员工）.

        Args:
            employee: 当前员工（排除自己的）
            applicability: 按适用范围标签过滤
            limit: 最大返回条数
            min_confidence: 最低置信度

        Returns:
            pattern 列表（MemoryEntry）
        """
        conditions = [
            "category = 'pattern'",
            "tenant_id = %s",
            "(superseded_by = '' OR superseded_by IS NULL)",
        ]
        params: list[Any] = [self._tenant_id]

        if employee:
            employee = self.resolve_to_character_name(employee)
            conditions.append("employee != %s")
            params.append(employee)

        if min_confidence > 0:
            conditions.append("confidence >= %s")
            params.append(min_confidence)

        # 过滤过期记忆
        conditions.append("(ttl_days = 0 OR created_at + (ttl_days || ' days')::interval > NOW())")

        # 适用范围过滤
        if applicability:
            conditions.append("applicability && %s")
            params.append(applicability)

        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
            sql = f"""
                SELECT id, employee, created_at, category, content,
                       source_session, confidence, superseded_by, ttl_days,
                       importance, last_accessed, tags, shared, visibility,
                       trigger_condition, applicability, origin_employee, verified_count,
                       classification, domain
                FROM memories
                WHERE {" AND ".join(conditions)}
                ORDER BY verified_count DESC, created_at DESC
                LIMIT %s
            """
            params.append(limit)
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        results: list[MemoryEntry] = []
        for row in rows:
            results.append(self._row_to_entry(row))

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
                employee = self.resolve_to_character_name(employee)
                cur.execute(
                    "DELETE FROM memories WHERE id = %s AND employee = %s AND tenant_id = %s",
                    (entry_id, employee, self._tenant_id),
                )
            else:
                cur.execute(
                    "DELETE FROM memories WHERE id = %s AND tenant_id = %s",
                    (entry_id, self._tenant_id),
                )

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
                "UPDATE memories SET confidence = %s WHERE id = %s AND tenant_id = %s",
                (confidence, entry_id, self._tenant_id),
            )
            return cur.rowcount > 0

    def count(self, employee: str) -> int:
        """返回员工的有效记忆条数."""
        employee = self.resolve_to_character_name(employee)

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT COUNT(*)
                FROM memories
                WHERE employee = %s AND tenant_id = %s
                  AND (superseded_by = '' OR superseded_by IS NULL)
                  AND (ttl_days = 0 OR created_at + (ttl_days || ' days')::interval > NOW())
                """,
                (employee, self._tenant_id),
            )
            row = cur.fetchone()
            return row[0] if row else 0

    def load_employee_entries(self, employee: str) -> list[MemoryEntry]:
        """加载指定员工的全部记忆条目（不做过滤）。

        公开接口，与文件版 MemoryStore.load_employee_entries() 对齐。
        """
        employee = self._resolve_to_character_name(employee)

        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
            cur.execute(
                """
                SELECT id, employee, created_at, category, content,
                       source_session, confidence, superseded_by, ttl_days,
                       importance, last_accessed, tags, shared, visibility,
                       trigger_condition, applicability, origin_employee, verified_count,
                       classification, domain
                FROM memories
                WHERE employee = %s AND tenant_id = %s
                ORDER BY created_at ASC
                """,
                (employee, self._tenant_id),
            )
            rows = cur.fetchall()

        return [self._row_to_entry(row) for row in rows]

    @staticmethod
    def is_expired(entry: MemoryEntry) -> bool:
        """检查记忆是否已过期（公开接口，与文件版对齐）."""
        ttl = entry.ttl_days
        if ttl <= 0:
            return False
        try:
            created = datetime.fromisoformat(entry.created_at)
            age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400
            return age_days > ttl
        except (ValueError, TypeError):
            return False

    def list_employees(self) -> list[str]:
        """列出有记忆的员工（当前租户）."""
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT DISTINCT employee FROM memories WHERE tenant_id = %s ORDER BY employee",
                (self._tenant_id,),
            )
            rows = cur.fetchall()
            return [row[0] for row in rows]

    def correct(
        self,
        employee: str,
        old_id: str,
        new_content: str,
        source_session: str = "",
    ) -> MemoryEntry | None:
        """纠正一条记忆：标记旧记忆为 superseded，创建新记忆.

        Args:
            employee: 员工名称
            old_id: 要纠正的记忆 ID
            new_content: 纠正后的内容
            source_session: 来源 session

        Returns:
            新创建的纠正记忆，如果旧记忆不存在返回 None
        """
        employee = self.resolve_to_character_name(employee)

        # 12 位 hex = 48 bit 熵，碰撞概率 ~1/2.8e14，当前数据量安全
        new_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc)

        with get_connection() as conn:
            cur = conn.cursor()
            # 标记旧记忆为 superseded
            cur.execute(
                """
                UPDATE memories
                SET superseded_by = %s, confidence = 0.0
                WHERE id = %s AND employee = %s AND tenant_id = %s
                """,
                (new_id, old_id, employee, self._tenant_id),
            )
            if cur.rowcount == 0:
                return None

            # 创建纠正记忆
            cur.execute(
                """
                INSERT INTO memories (
                    id, employee, created_at, category, content,
                    source_session, confidence, superseded_by, ttl_days,
                    importance, last_accessed, tags, shared, visibility,
                    trigger_condition, applicability, origin_employee, verified_count,
                    classification, domain, tenant_id
                ) VALUES (
                    %s, %s, %s, 'correction', %s,
                    %s, 1.0, '', 0,
                    3, NULL, '{}', FALSE, 'open',
                    '', '{}', %s, 0,
                    'internal', '{}', %s
                )
                """,
                (new_id, employee, now, new_content, source_session, employee, self._tenant_id),
            )

        return MemoryEntry(
            id=new_id,
            employee=employee,
            created_at=now.isoformat(),
            category="correction",
            content=new_content,
            source_session=source_session,
            confidence=1.0,
        )

    def format_for_prompt(
        self,
        employee: str,
        limit: int = 10,
        query: str = "",
        employee_tags: list[str] | None = None,
        max_visibility: str = "open",
        team_members: list[str] | None = None,
        classification_max: str | None = None,
        allowed_domains: list[str] | None = None,
        include_confidential: bool = False,
    ) -> str:
        """格式化记忆为可注入 prompt 的文本.

        Args:
            employee: 员工名称
            limit: 最大条数
            query: 查询上下文（暂不支持语义搜索，降级为普通查询）
            employee_tags: 员工标签（用于匹配共享记忆）
            max_visibility: 可见性上限
            team_members: 同团队成员名列表
            classification_max: 最高信息分级（可选）
            allowed_domains: 允许的职能域（可选）
            include_confidential: 是否包含 confidential 级别

        Returns:
            Markdown 格式的记忆文本，无记忆时返回空字符串
        """
        employee = self.resolve_to_character_name(employee)
        parts: list[str] = []

        # 个人记忆（透传 classification 参数）
        entries = self.query(
            employee,
            limit=limit,
            max_visibility=max_visibility,
            classification_max=classification_max,
            allowed_domains=allowed_domains,
            include_confidential=include_confidential,
        )
        if entries:
            parts.append(self._format_entries(entries))

        # 跨员工共享记忆
        shared_entries = self.query_shared(
            tags=employee_tags,
            exclude_employee=employee,
            limit=max(3, limit // 3),
        )
        if shared_entries:
            lines = []
            for entry in shared_entries:
                tag_str = f" [{', '.join(entry.tags)}]" if entry.tags else ""
                cat = self._category_label(entry.category)
                conf = f" (置信度: {entry.confidence:.0%})" if entry.confidence < 1.0 else ""
                trigger = (
                    f" [触发: {entry.trigger_condition}]"
                    if entry.category == "pattern" and entry.trigger_condition
                    else ""
                )
                lines.append(
                    f"- [{cat}]{conf}{tag_str}{trigger} ({entry.employee}) {entry.content}"
                )
            parts.append("\n### 团队共享经验\n\n" + "\n".join(lines))

        # 同团队成员的公开记忆
        if team_members:
            team_entries = self.query_team(
                team_members,
                exclude_employee=employee,
                limit=max(3, limit // 3),
            )
            if team_entries:
                lines = []
                for entry in team_entries:
                    cat = self._category_label(entry.category)
                    conf = f" (置信度: {entry.confidence:.0%})" if entry.confidence < 1.0 else ""
                    lines.append(f"- [{cat}]{conf} ({entry.employee}) {entry.content}")
                parts.append("\n### 队友近况\n\n" + "\n".join(lines))

        return "\n".join(parts)

    @staticmethod
    def _category_label(category: str) -> str:
        """将类别转为中文标签."""
        return {
            "decision": "决策",
            "estimate": "估算",
            "finding": "发现",
            "correction": "纠正",
            "pattern": "模式",
        }.get(category, category)

    @staticmethod
    def _format_entries(entries: list[MemoryEntry]) -> str:
        """格式化记忆条目列表为 Markdown."""
        lines = []
        for entry in entries:
            category_label = {
                "decision": "决策",
                "estimate": "估算",
                "finding": "发现",
                "correction": "纠正",
                "pattern": "模式",
            }.get(entry.category, entry.category)
            conf = f" (置信度: {entry.confidence:.0%})" if entry.confidence < 1.0 else ""
            proxied = " ⚠️模拟讨论记录，非实际工作" if "proxied" in entry.tags else ""
            trigger = (
                f" [触发: {entry.trigger_condition}]"
                if entry.category == "pattern" and entry.trigger_condition
                else ""
            )
            lines.append(f"- [{category_label}]{conf}{proxied}{trigger} {entry.content}")
        return "\n".join(lines)

    def query_team(
        self,
        members: list[str],
        exclude_employee: str = "",
        limit: int = 5,
        min_confidence: float = 0.3,
    ) -> list[MemoryEntry]:
        """查询指定团队成员的公开记忆（不要求 shared=True）.

        Args:
            members: 团队成员名列表
            exclude_employee: 排除指定员工（通常是当前员工自身）
            limit: 最大返回条数
            min_confidence: 最低有效置信度
        """
        members = [self.resolve_to_character_name(m) for m in members]
        exclude_employee = self.resolve_to_character_name(exclude_employee)

        member_set = set(members) - {exclude_employee}
        if not member_set:
            return []

        conditions = [
            "employee = ANY(%s)",
            "tenant_id = %s",
            "(superseded_by = '' OR superseded_by IS NULL)",
            "visibility = 'open'",
            "(ttl_days = 0 OR created_at + (ttl_days || ' days')::interval > NOW())",
        ]
        params: list[Any] = [list(member_set), self._tenant_id]

        if min_confidence > 0:
            conditions.append("confidence >= %s")
            params.append(min_confidence)

        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
            sql = f"""
                SELECT id, employee, created_at, category, content,
                       source_session, confidence, superseded_by, ttl_days,
                       importance, last_accessed, tags, shared, visibility,
                       trigger_condition, applicability, origin_employee, verified_count,
                       classification, domain
                FROM memories
                WHERE {" AND ".join(conditions)}
                ORDER BY created_at DESC
                LIMIT %s
            """
            params.append(limit)
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        return [self._row_to_entry(row) for row in rows]

    def verify_pattern(self, pattern_id: str) -> bool:
        """验证一条 pattern（verified_count +1）.

        Returns:
            True 如果找到并更新，False 如果未找到
        """
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE memories
                SET verified_count = verified_count + 1
                WHERE id = %s AND category = 'pattern' AND tenant_id = %s
                """,
                (pattern_id, self._tenant_id),
            )
            return cur.rowcount > 0

    def update(
        self,
        entry_id: str,
        employee: str,
        *,
        content: str | None = None,
        tags: list[str] | None = None,
        confidence: float | None = None,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
    ) -> bool:
        """更新一条记忆.

        Args:
            entry_id: 记忆 ID
            employee: 员工名称
            content: 新内容（None 不更新）
            tags: 完全替换标签列表（None 不更新）
            confidence: 新置信度（None 不更新）
            add_tags: 追加标签
            remove_tags: 移除标签

        Returns:
            True 更新成功，False 未找到
        """
        employee = self.resolve_to_character_name(employee)

        # 先查出当前状态
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
            cur.execute(
                "SELECT id, tags FROM memories WHERE id = %s AND employee = %s AND tenant_id = %s",
                (entry_id, employee, self._tenant_id),
            )
            row = cur.fetchone()
            if not row:
                return False

            # 构建 SET 子句
            sets: list[str] = []
            params: list[Any] = []

            if content is not None:
                sets.append("content = %s")
                params.append(content)

            if confidence is not None:
                sets.append("confidence = %s")
                params.append(confidence)

            # 标签处理
            current_tags = list(row.get("tags") or [])

            if tags is not None:
                # 完全替换
                current_tags = tags

            if add_tags:
                current_tags = list(set(current_tags + add_tags))

            if remove_tags:
                current_tags = [t for t in current_tags if t not in remove_tags]

            if tags is not None or add_tags or remove_tags:
                sets.append("tags = %s")
                params.append(current_tags)

            if not sets:
                return True  # 没有需要更新的字段

            params.extend([entry_id, employee, self._tenant_id])
            cur.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id = %s AND employee = %s AND tenant_id = %s",
                tuple(params),
            )
            return cur.rowcount > 0

    def add_from_session(
        self,
        *,
        employee: str,
        session_id: str,
        summary: str,
        category: Literal["decision", "estimate", "finding", "correction", "pattern"] = "finding",
    ) -> MemoryEntry:
        """根据会话摘要写入记忆."""
        return self.add(
            employee=employee,
            category=category,
            content=summary,
            source_session=session_id,
        )
