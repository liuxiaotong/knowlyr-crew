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

        # Phase 3-1: 幂等添加 keywords 列
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE memories ADD COLUMN keywords text[] DEFAULT '{}';
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
        """)
        # Phase 3-1: 幂等添加 linked_memories 列
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE memories ADD COLUMN linked_memories text[] DEFAULT '{}';
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
        """)
        # 注意：keywords 列的查询使用 unnest + ILIKE 子串匹配，GIN 索引无法加速。
        # 当前数据量不需要索引；后续量大时可考虑 pg_trgm 或改为精确匹配再加 GIN。
        cur.execute("DROP INDEX IF EXISTS idx_memories_keywords")

        # Phase 4: 幂等添加 recall_count 列（召回效果闭环）
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE memories ADD COLUMN recall_count INTEGER DEFAULT 0;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$;
        """)

        # NG-1: pgvector 向量语义检索
        # 安装 pgvector 扩展（需要 superuser 或已预装）
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception:
            # pgvector 扩展未安装时不阻塞初始化
            conn.rollback()
            logger.warning("pgvector 扩展不可用，向量检索功能将被禁用")

        # 幂等添加 embedding 列（vector(384) = all-MiniLM-L6-v2 输出维度）
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE memories ADD COLUMN embedding vector(384);
            EXCEPTION
                WHEN duplicate_column THEN NULL;
                WHEN undefined_object THEN NULL;
            END $$;
        """)

        # ivfflat 索引：需要表中至少有足够行才能建 lists=100 的索引
        # 用 IF NOT EXISTS 保持幂等
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_embedding
                ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            """)
        except Exception:
            # 索引创建失败（如 vector 类型不存在或数据不足），不阻塞
            conn.rollback()
            logger.warning("embedding 索引创建失败，将在数据回填后重试")

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

    def _get_query_embedding(self, keywords: list[str]) -> list[float] | None:
        """为查询关键词生成 embedding 向量.

        Args:
            keywords: 搜索关键词列表

        Returns:
            384 维浮点向量，或 None（不可用时静默降级）
        """
        try:
            from crew.embedding import build_embedding_text, get_embedding

            query_text = build_embedding_text(" ".join(keywords))
            return get_embedding(query_text)
        except Exception:
            return None

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
            keywords=list(row.get("keywords") or []),
            linked_memories=list(row.get("linked_memories") or []),
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
        keywords: list[str] | None = None,
    ) -> MemoryEntry:
        """添加一条记忆.

        Returns:
            MemoryEntry 对象
        """
        employee = self._resolve_to_character_name(employee)

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
        # 支持 keywords 原子写入，避免先 INSERT 空再 UPDATE 的两步操作
        keywords_list = keywords or []

        # NG-1: 生成 embedding 向量（失败不阻塞写入）
        embedding_vector = None
        try:
            from crew.embedding import build_embedding_text, get_embedding

            emb_text = build_embedding_text(content, keywords_list)
            embedding_vector = get_embedding(emb_text)
        except Exception as e:
            logger.debug("embedding 生成跳过: %s", e)

        # NG-2: 轻量去重 — correction 类型不做去重（每次纠正都应独立记录）
        if embedding_vector is not None and category != "correction":
            merged = self._try_dedup_merge(
                employee,
                embedding_vector,
                content,
                keywords_list,
                category,
            )
            if merged is not None:
                return merged

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO memories (
                    id, employee, created_at, category, content,
                    source_session, confidence, superseded_by, ttl_days,
                    importance, last_accessed, tags, shared, visibility,
                    trigger_condition, applicability, origin_employee, verified_count,
                    classification, domain, tenant_id,
                    keywords, linked_memories, embedding
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
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
                    keywords_list,  # keywords: 原子写入传入值
                    [],  # linked_memories
                    embedding_vector,  # NG-1: embedding 向量（可能为 None）
                ),
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
            keywords=keywords_list,
            linked_memories=[],
        )

    def _try_dedup_merge(
        self,
        employee: str,
        embedding_vector: list[float],
        content: str,
        keywords: list[str],
        category: str,
    ) -> MemoryEntry | None:
        """NG-2: 去重合并 — 查同员工同租户 top-1 最相似记忆，>=0.90 则合并.

        Args:
            employee: 员工名称（已 resolve）
            embedding_vector: 新记忆的 embedding 向量
            content: 新记忆内容
            keywords: 新记忆关键词
            category: 新记忆类别

        Returns:
            合并后的 MemoryEntry（如果触发合并），否则 None
        """
        try:
            with get_connection() as conn:
                import psycopg2.extras

                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute(
                    """
                    SELECT id, content, keywords, category, embedding,
                           employee, created_at, source_session, confidence,
                           superseded_by, ttl_days, importance, last_accessed,
                           tags, shared, visibility, trigger_condition,
                           applicability, origin_employee, verified_count,
                           classification, domain, linked_memories
                    FROM memories
                    WHERE employee = %s AND tenant_id = %s
                      AND (superseded_by = '' OR superseded_by IS NULL)
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s
                    LIMIT 1
                    """,
                    (employee, self._tenant_id, str(embedding_vector)),
                )
                row = cur.fetchone()

            if row is None:
                return None

            # cosine_distance = 1 - cosine_similarity
            # embedding <=> 返回 cosine distance，需要 Python 端计算 similarity
            # 但 pgvector <=> 已经返回了距离值，不在 row 中
            # 重新查询获取距离
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT embedding <=> %s AS dist FROM memories WHERE id = %s AND tenant_id = %s",
                    (str(embedding_vector), row["id"], self._tenant_id),
                )
                dist_row = cur.fetchone()

            if dist_row is None:
                return None

            cosine_distance = float(dist_row[0])
            similarity = 1.0 - cosine_distance

            if similarity < 0.90 or row["category"] != category:
                return None

            # 触发合并：UPDATE 旧记忆
            merged_content = f"{row['content']}\n---\n{content}"
            if len(merged_content) > 3000:
                merged_content = f"{row['content'][:1500]}\n---\n{content}"

            old_keywords = list(row.get("keywords") or [])
            merged_keywords = list(set(old_keywords + keywords))

            # 重新生成 embedding（用合并后内容）
            new_embedding = None
            try:
                from crew.embedding import build_embedding_text, get_embedding

                emb_text = build_embedding_text(merged_content, merged_keywords)
                new_embedding = get_embedding(emb_text)
            except Exception:
                new_embedding = embedding_vector  # fallback: 用新记忆的向量

            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE memories
                    SET content = %s, keywords = %s, embedding = %s
                    WHERE id = %s AND employee = %s AND tenant_id = %s
                    """,
                    (
                        merged_content,
                        merged_keywords,
                        new_embedding,
                        row["id"],
                        employee,
                        self._tenant_id,
                    ),
                )

            logger.info(
                "dedup: merged into %s (similarity=%.2f)",
                row["id"],
                similarity,
            )

            # 返回更新后的 entry
            return self._row_to_entry(
                {
                    **row,
                    "content": merged_content,
                    "keywords": merged_keywords,
                }
            )

        except Exception as e:
            logger.debug("dedup 查询失败，回退到正常写入: %s", e)
            return None

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
        search_text: str | None = None,
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
            search_text: 文本搜索（可选，ILIKE 子串匹配，支持中文）

        Returns:
            记忆列表（MemoryEntry）
        """
        employee = self._resolve_to_character_name(employee)

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

        # 文本搜索（ILIKE 子串匹配，对中文友好）
        # 按空格拆分为多个词，每个词用 AND 连接，支持 "skip_paths 误删" 这类多词查询
        if search_text:
            tokens = search_text.strip().split()
            for token in tokens:
                conditions.append("content ILIKE %s")
                params.append(f"%{token}%")

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
        # 当有 search_text 时，按 category 权重排序（知识类优先）
        if search_text:
            order_by = (
                "CASE category "
                "WHEN 'pattern' THEN 1 "
                "WHEN 'decision' THEN 2 "
                "WHEN 'correction' THEN 3 "
                "WHEN 'estimate' THEN 4 "
                "WHEN 'finding' THEN 5 "
                "ELSE 6 END, "
                "importance DESC, created_at DESC"
            )
        elif sort_by == "importance":
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
                       classification, domain,
                       keywords, linked_memories
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
        exclude_employee = self._resolve_to_character_name(exclude_employee)

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
                       classification, domain,
                       keywords, linked_memories
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
            employee = self._resolve_to_character_name(employee)
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
                       classification, domain,
                       keywords, linked_memories
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
                employee = self._resolve_to_character_name(employee)
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
        employee = self._resolve_to_character_name(employee)

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
                       classification, domain,
                       keywords, linked_memories
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
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
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
        employee = self._resolve_to_character_name(employee)

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
            query: 查询上下文（有值时走 search_text 语义搜索，返回相关记忆）
            employee_tags: 员工标签（用于匹配共享记忆）
            max_visibility: 可见性上限
            team_members: 同团队成员名列表
            classification_max: 最高信息分级（可选）
            allowed_domains: 允许的职能域（可选）
            include_confidential: 是否包含 confidential 级别

        Returns:
            Markdown 格式的记忆文本，无记忆时返回空字符串
        """
        employee = self._resolve_to_character_name(employee)
        parts: list[str] = []

        # 个人记忆（透传 classification 参数）
        # L2 优化：有 query 时走 search_text 语义搜索，返回最相关 Top-N
        entries = self.query(
            employee,
            limit=limit,
            max_visibility=max_visibility,
            classification_max=classification_max,
            allowed_domains=allowed_domains,
            include_confidential=include_confidential,
            search_text=query if query else None,
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
        members = [self._resolve_to_character_name(m) for m in members]
        exclude_employee = self._resolve_to_character_name(exclude_employee)

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
                       classification, domain,
                       keywords, linked_memories
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
        employee = self._resolve_to_character_name(employee)

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

    def update_keywords(self, entry_id: str, employee: str, keywords: list[str]) -> bool:
        """更新记忆的结构化关键词.

        Args:
            entry_id: 记忆 ID
            employee: 员工名称
            keywords: 新的关键词列表（完全替换）

        Returns:
            True 更新成功，False 未找到
        """
        employee = self._resolve_to_character_name(employee)
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE memories SET keywords = %s WHERE id = %s AND employee = %s AND tenant_id = %s",
                (keywords, entry_id, employee, self._tenant_id),
            )
            return cur.rowcount > 0

    def update_linked_memories(self, entry_id: str, employee: str, linked_ids: list[str]) -> bool:
        """更新记忆的关联记忆 ID 列表.

        Args:
            entry_id: 记忆 ID
            employee: 员工名称
            linked_ids: 关联记忆 ID 列表（完全替换）

        Returns:
            True 更新成功，False 未找到
        """
        employee = self._resolve_to_character_name(employee)
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE memories SET linked_memories = %s WHERE id = %s AND employee = %s AND tenant_id = %s",
                (linked_ids, entry_id, employee, self._tenant_id),
            )
            return cur.rowcount > 0

    def query_by_keywords(
        self,
        employee: str,
        keywords: list[str],
        limit: int = 10,
        category: str | None = None,
    ) -> list[MemoryEntry]:
        """按关键词匹配记忆，支持混合检索（关键词 + 向量语义）.

        混合排序公式：final_score = 0.3 * keyword_score + 0.7 * cosine_similarity
        当 embedding 不可用时，退化为纯关键词匹配（向后兼容）。
        """
        employee = self._resolve_to_character_name(employee)

        if not keywords:
            return self.query(employee, category=category, limit=limit)

        # 截断到最多 10 个关键词
        keywords = keywords[:10]

        # 尝试生成查询向量
        query_embedding = self._get_query_embedding(keywords)

        # 构建 match_count 表达式：对每个搜索关键词累加是否匹配
        match_parts: list[str] = []
        params: list[Any] = []

        for kw in keywords:
            match_parts.append(
                "(CASE WHEN EXISTS (SELECT 1 FROM unnest(keywords) AS k WHERE k ILIKE %s) THEN 1 ELSE 0 END)"
            )
            params.append(f"%{kw}%")

        match_count_expr = " + ".join(match_parts)
        num_keywords = len(keywords)

        conditions = [
            "employee = %s",
            "tenant_id = %s",
            "(superseded_by = '' OR superseded_by IS NULL)",
            "(ttl_days = 0 OR created_at + (ttl_days || ' days')::interval > NOW())",
        ]
        params.extend([employee, self._tenant_id])

        if category:
            conditions.append("category = %s")
            params.append(category)

        if query_embedding is not None:
            # 混合检索：关键词匹配 OR 向量相似度足够高
            # 不要求必须命中关键词 — 语义相似也可召回
            or_conditions = []
            for kw in keywords:
                or_conditions.append(
                    "EXISTS (SELECT 1 FROM unnest(keywords) AS k WHERE k ILIKE %s)"
                )
                params.append(f"%{kw}%")
            # cosine distance < 0.5 即 cosine_similarity > 0.5
            or_conditions.append("(embedding IS NOT NULL AND embedding <=> %s < 0.5)")
            params.append(str(query_embedding))
            conditions.append(f"({' OR '.join(or_conditions)})")

            # 混合评分：0.3 * keyword_norm + 0.7 * cosine_sim
            score_expr = (
                f"(0.3 * (({match_count_expr}) / {num_keywords}.0)"
                f" + 0.7 * CASE WHEN embedding IS NOT NULL"
                f" THEN 1.0 - (embedding <=> %s) ELSE 0 END)"
            )
            # match_count_expr 的参数需要再传一次（用于 SELECT 中的评分计算）
            score_params: list[Any] = []
            for kw in keywords:
                score_params.append(f"%{kw}%")
            score_params.append(str(query_embedding))

            # params 开头有 N 个 match_parts 参数（用于纯关键词分支的 SELECT），
            # 混合分支的 SELECT 用 score_params 代替，所以跳过它们
            where_params = params[num_keywords:]

            with get_connection() as conn:
                cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
                sql = f"""
                    SELECT id, employee, created_at, category, content,
                           source_session, confidence, superseded_by, ttl_days,
                           importance, last_accessed, tags, shared, visibility,
                           trigger_condition, applicability, origin_employee, verified_count,
                           classification, domain,
                           keywords, linked_memories,
                           {score_expr} AS hybrid_score
                    FROM memories
                    WHERE {" AND ".join(conditions)}
                    ORDER BY hybrid_score DESC, importance DESC, created_at DESC
                    LIMIT %s
                """
                all_params = tuple(score_params + where_params + [limit])
                cur.execute(sql, all_params)
                rows = cur.fetchall()

            return [self._row_to_entry(row) for row in rows]

        else:
            # 纯关键词匹配（降级模式）
            or_conditions = []
            for kw in keywords:
                or_conditions.append(
                    "EXISTS (SELECT 1 FROM unnest(keywords) AS k WHERE k ILIKE %s)"
                )
                params.append(f"%{kw}%")
            conditions.append(f"({' OR '.join(or_conditions)})")

            params.append(limit)

            with get_connection() as conn:
                cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
                sql = f"""
                    SELECT id, employee, created_at, category, content,
                           source_session, confidence, superseded_by, ttl_days,
                           importance, last_accessed, tags, shared, visibility,
                           trigger_condition, applicability, origin_employee, verified_count,
                           classification, domain,
                           keywords, linked_memories,
                           ({match_count_expr}) AS match_count
                    FROM memories
                    WHERE {" AND ".join(conditions)}
                    ORDER BY match_count DESC, importance DESC, created_at DESC
                    LIMIT %s
                """
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()

            return [self._row_to_entry(row) for row in rows]

    def query_cross_employee(
        self,
        keywords: list[str],
        exclude_employee: str = "",
        limit: int = 5,
        category: str | None = None,
    ) -> list[MemoryEntry]:
        """跨员工按关键词匹配记忆，支持混合检索（仅 visibility=open）.

        不限制员工，但排除指定员工（通常是当前员工自身）。
        混合排序公式：final_score = 0.3 * keyword_score + 0.7 * cosine_similarity
        当 embedding 不可用时，退化为纯关键词匹配。
        """
        if not keywords:
            return []

        exclude_employee = self._resolve_to_character_name(exclude_employee)
        keywords = keywords[:10]

        # 尝试生成查询向量
        query_embedding = self._get_query_embedding(keywords)

        # 构建 match_count 表达式
        match_parts: list[str] = []
        params: list[Any] = []

        for kw in keywords:
            match_parts.append(
                "(CASE WHEN EXISTS (SELECT 1 FROM unnest(keywords) AS k WHERE k ILIKE %s) THEN 1 ELSE 0 END)"
            )
            params.append(f"%{kw}%")

        match_count_expr = " + ".join(match_parts)
        num_keywords = len(keywords)

        conditions = [
            "tenant_id = %s",
            "(superseded_by = '' OR superseded_by IS NULL)",
            "(ttl_days = 0 OR created_at + (ttl_days || ' days')::interval > NOW())",
            "visibility = 'open'",
        ]
        params.append(self._tenant_id)

        if exclude_employee:
            conditions.append("employee != %s")
            params.append(exclude_employee)

        if category:
            conditions.append("category = %s")
            params.append(category)

        if query_embedding is not None:
            # 混合检索
            or_conditions = []
            for kw in keywords:
                or_conditions.append(
                    "EXISTS (SELECT 1 FROM unnest(keywords) AS k WHERE k ILIKE %s)"
                )
                params.append(f"%{kw}%")
            or_conditions.append("(embedding IS NOT NULL AND embedding <=> %s < 0.5)")
            params.append(str(query_embedding))
            conditions.append(f"({' OR '.join(or_conditions)})")

            score_expr = (
                f"(0.3 * (({match_count_expr}) / {num_keywords}.0)"
                f" + 0.7 * CASE WHEN embedding IS NOT NULL"
                f" THEN 1.0 - (embedding <=> %s) ELSE 0 END)"
            )
            score_params: list[Any] = []
            for kw in keywords:
                score_params.append(f"%{kw}%")
            score_params.append(str(query_embedding))

            # params 开头有 N 个 match_parts 参数（用于纯关键词分支的 SELECT），
            # 混合分支的 SELECT 用 score_params 代替，所以跳过它们
            where_params = params[num_keywords:]

            with get_connection() as conn:
                cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
                sql = f"""
                    SELECT id, employee, created_at, category, content,
                           source_session, confidence, superseded_by, ttl_days,
                           importance, last_accessed, tags, shared, visibility,
                           trigger_condition, applicability, origin_employee, verified_count,
                           classification, domain,
                           keywords, linked_memories,
                           {score_expr} AS hybrid_score
                    FROM memories
                    WHERE {" AND ".join(conditions)}
                    ORDER BY hybrid_score DESC, importance DESC, created_at DESC
                    LIMIT %s
                """
                all_params = tuple(score_params + where_params + [limit])
                cur.execute(sql, all_params)
                rows = cur.fetchall()

            return [self._row_to_entry(row) for row in rows]

        else:
            # 纯关键词匹配（降级模式）
            or_conditions = []
            for kw in keywords:
                or_conditions.append(
                    "EXISTS (SELECT 1 FROM unnest(keywords) AS k WHERE k ILIKE %s)"
                )
                params.append(f"%{kw}%")
            conditions.append(f"({' OR '.join(or_conditions)})")

            params.append(limit)

            with get_connection() as conn:
                cur = conn.cursor(cursor_factory=self._dict_cursor_factory)
                sql = f"""
                    SELECT id, employee, created_at, category, content,
                           source_session, confidence, superseded_by, ttl_days,
                           importance, last_accessed, tags, shared, visibility,
                           trigger_condition, applicability, origin_employee, verified_count,
                           classification, domain,
                           keywords, linked_memories,
                           ({match_count_expr}) AS match_count
                    FROM memories
                    WHERE {" AND ".join(conditions)}
                    ORDER BY match_count DESC, importance DESC, created_at DESC
                    LIMIT %s
                """
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()

            return [self._row_to_entry(row) for row in rows]

    def get_knowledge_stats(self) -> dict:
        """返回团队知识结构统计.

        Returns:
            {
                "employee_stats": [{"employee": "xxx", "total": N, "by_category": {...}}],
                "top_keywords": [{"keyword": "xxx", "count": N}],  # Top 30
                "correction_hotspots": [{"keyword": "xxx", "correction_count": N}],
                "knowledge_gaps": [{"keyword": "xxx", "findings": N, "patterns": 0}],
                "weekly_trend": [{"week": "2026-W10", "count": N}],  # 最近 12 周
            }
        """
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)

            # 1. employee_stats: 每个员工的记忆数和分类统计
            cur.execute(
                """
                SELECT employee, category, COUNT(*) as cnt
                FROM memories
                WHERE tenant_id = %s
                  AND (superseded_by = '' OR superseded_by IS NULL)
                GROUP BY employee, category
                ORDER BY employee, category
                """,
                (self._tenant_id,),
            )
            rows = cur.fetchall()
            emp_map: dict[str, dict] = {}
            for row in rows:
                emp = row["employee"]
                if emp not in emp_map:
                    emp_map[emp] = {"employee": emp, "total": 0, "by_category": {}}
                emp_map[emp]["by_category"][row["category"]] = row["cnt"]
                emp_map[emp]["total"] += row["cnt"]
            employee_stats = list(emp_map.values())

            # 2. top_keywords: 展开 keywords 数组统计频次 Top 30
            cur.execute(
                """
                SELECT kw, COUNT(*) as cnt
                FROM memories, unnest(keywords) AS kw
                WHERE tenant_id = %s
                  AND (superseded_by = '' OR superseded_by IS NULL)
                GROUP BY kw
                ORDER BY cnt DESC
                LIMIT 30
                """,
                (self._tenant_id,),
            )
            top_keywords = [{"keyword": r["kw"], "count": r["cnt"]} for r in cur.fetchall()]

            # 3. correction_hotspots: correction 类别中高频关键词
            cur.execute(
                """
                SELECT kw, COUNT(*) as cnt
                FROM memories, unnest(keywords) AS kw
                WHERE tenant_id = %s
                  AND category = 'correction'
                  AND (superseded_by = '' OR superseded_by IS NULL)
                GROUP BY kw
                ORDER BY cnt DESC
                LIMIT 20
                """,
                (self._tenant_id,),
            )
            correction_hotspots = [
                {"keyword": r["kw"], "correction_count": r["cnt"]} for r in cur.fetchall()
            ]

            # 4. knowledge_gaps: 有 finding 但没有 pattern 的关键词
            cur.execute(
                """
                WITH finding_kws AS (
                    SELECT kw, COUNT(*) as findings
                    FROM memories, unnest(keywords) AS kw
                    WHERE tenant_id = %s
                      AND category = 'finding'
                      AND (superseded_by = '' OR superseded_by IS NULL)
                    GROUP BY kw
                ),
                pattern_kws AS (
                    SELECT kw, COUNT(*) as patterns
                    FROM memories, unnest(keywords) AS kw
                    WHERE tenant_id = %s
                      AND category = 'pattern'
                      AND (superseded_by = '' OR superseded_by IS NULL)
                    GROUP BY kw
                )
                SELECT f.kw AS keyword, f.findings, COALESCE(p.patterns, 0) AS patterns
                FROM finding_kws f
                LEFT JOIN pattern_kws p ON f.kw = p.kw
                WHERE COALESCE(p.patterns, 0) = 0
                ORDER BY f.findings DESC
                LIMIT 20
                """,
                (self._tenant_id, self._tenant_id),
            )
            knowledge_gaps = [
                {"keyword": r["keyword"], "findings": r["findings"], "patterns": r["patterns"]}
                for r in cur.fetchall()
            ]

            # 5. weekly_trend: 最近 12 周每周新增记忆数
            cur.execute(
                """
                SELECT to_char(created_at, 'IYYY-"W"IW') AS week, COUNT(*) as cnt
                FROM memories
                WHERE tenant_id = %s
                  AND created_at >= NOW() - INTERVAL '12 weeks'
                GROUP BY week
                ORDER BY week
                """,
                (self._tenant_id,),
            )
            weekly_trend = [{"week": r["week"], "count": r["cnt"]} for r in cur.fetchall()]

        return {
            "employee_stats": employee_stats,
            "top_keywords": top_keywords,
            "correction_hotspots": correction_hotspots,
            "knowledge_gaps": knowledge_gaps,
            "weekly_trend": weekly_trend,
        }

    def record_recall(self, memory_ids: list[str]) -> int:
        """批量增加 recall_count.

        每次这些记忆被召回注入上下文时调用。

        Returns:
            更新的行数
        """
        if not memory_ids:
            return 0
        now = datetime.now(timezone.utc)
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE memories
                SET recall_count = recall_count + 1,
                    last_accessed = %s
                WHERE id = ANY(%s) AND tenant_id = %s
                """,
                (now, memory_ids, self._tenant_id),
            )
            return cur.rowcount

    def record_useful(self, memory_ids: list[str], employee: str) -> int:
        """标记这些记忆在任务中被实际使用.

        Returns:
            更新的行数
        """
        if not memory_ids:
            return 0
        employee = self._resolve_to_character_name(employee)
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE memories
                SET verified_count = verified_count + 1
                WHERE id = ANY(%s) AND tenant_id = %s AND employee = %s
                """,
                (memory_ids, self._tenant_id, employee),
            )
            return cur.rowcount

    def get_recall_stats(self) -> dict:
        """召回效果统计.

        Returns:
            包含 total_recalls, total_useful, hit_rate, top_useful,
            never_recalled 的字典
        """
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=self._dict_cursor_factory)

            # 汇总统计
            cur.execute(
                """
                SELECT COALESCE(SUM(recall_count), 0) AS total_recalls,
                       COALESCE(SUM(verified_count), 0) AS total_useful
                FROM memories
                WHERE tenant_id = %s
                """,
                (self._tenant_id,),
            )
            row = cur.fetchone()
            total_recalls = int(row["total_recalls"])
            total_useful = int(row["total_useful"])
            hit_rate = total_useful / total_recalls if total_recalls > 0 else 0.0

            # Top 10 最有用的记忆
            cur.execute(
                """
                SELECT id, employee, content, category, verified_count, recall_count
                FROM memories
                WHERE tenant_id = %s AND verified_count > 0
                ORDER BY verified_count DESC
                LIMIT 10
                """,
                (self._tenant_id,),
            )
            top_useful = [dict(r) for r in cur.fetchall()]

            # 从未被召回的记忆数
            cur.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM memories
                WHERE tenant_id = %s
                  AND recall_count = 0
                  AND (superseded_by = '' OR superseded_by IS NULL)
                """,
                (self._tenant_id,),
            )
            never_recalled = int(cur.fetchone()["cnt"])

        return {
            "total_recalls": total_recalls,
            "total_useful": total_useful,
            "hit_rate": round(hit_rate, 4),
            "top_useful": top_useful,
            "never_recalled": never_recalled,
        }

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
