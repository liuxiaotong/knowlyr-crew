"""持久化记忆 — 每个员工独立的经验存储."""

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from crew.paths import file_lock, resolve_project_dir

logger = logging.getLogger(__name__)


class MemoryConfig(BaseModel):
    """记忆系统配置."""

    default_ttl_days: int = Field(default=0, description="默认 TTL 天数 (0=永不过期)")
    max_entries_per_employee: int = Field(default=0, description="每员工最大记忆条数 (0=不限)")
    confidence_half_life_days: float = Field(default=90.0, description="置信度衰减半衰期（天）")
    auto_index: bool = Field(default=True, description="写入时自动更新语义索引")


class MemoryEntry(BaseModel):
    """单条记忆."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12], description="唯一 ID")
    employee: str = Field(description="员工标识符")
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(), description="创建时间"
    )
    category: Literal["decision", "estimate", "finding", "correction", "pattern"] = Field(
        description="记忆类别"
    )
    content: str = Field(description="记忆内容")
    source_session: str = Field(default="", description="来源 session ID")
    confidence: float = Field(default=1.0, description="置信度（被纠正后降低）")
    superseded_by: str = Field(default="", description="被哪条记忆覆盖")
    # Enhancement 1: 衰减 + 容量
    ttl_days: int = Field(default=0, description="生存期天数，0=永不过期")
    # 重要性与访问追踪
    importance: int = Field(
        default=3, description="重要性 1-5（5=最高，如决策/待办；1=最低，如闲聊背景）"
    )
    last_accessed: str = Field(default="", description="最后被加载到上下文的时间（ISO 格式）")
    # Enhancement 3: 跨员工共享
    tags: list[str] = Field(default_factory=list, description="语义标签")
    shared: bool = Field(default=False, description="是否加入共享记忆池")
    # 可见性控制
    visibility: Literal["open", "private"] = Field(
        default="open", description="可见性: open=公开, private=仅私聊可见"
    )
    # Pattern 专有字段（仅 category="pattern" 时使用）
    trigger_condition: str = Field(default="", description="触发条件：什么场景下该用此模式")
    applicability: list[str] = Field(default_factory=list, description="适用范围：角色/领域标签")
    origin_employee: str = Field(default="", description="来源员工：谁发现的此模式")
    verified_count: int = Field(default=0, description="被验证次数：其他员工确认有效的次数")


class MemoryStore:
    """员工记忆存储管理器.

    存储结构:
      {memory_dir}/
        {employee_name}.jsonl   — 每个员工一个文件
        config.json             — 可选配置
    """

    def __init__(
        self,
        memory_dir: Path | None = None,
        *,
        project_dir: Path | None = None,
        config: MemoryConfig | None = None,
    ):
        self.memory_dir = (
            memory_dir
            if memory_dir is not None
            else resolve_project_dir(project_dir) / ".crew" / "memory"
        )
        self.config = config or self._load_config()
        self._semantic_index = None  # lazy

    def _load_config(self) -> MemoryConfig:
        """从 config.json 加载配置，不存在则返回默认值."""
        config_path = self.memory_dir / "config.json"
        if config_path.is_file():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
                return MemoryConfig(**data)
            except Exception as e:
                logger.debug("记忆配置加载失败，使用默认值: %s", e)
        return MemoryConfig()

    def _get_semantic_index(self):
        """懒初始化语义索引."""
        if self._semantic_index is None:
            try:
                from crew.memory_search import SemanticMemoryIndex

                self._semantic_index = SemanticMemoryIndex(self.memory_dir)
            except Exception as e:
                logger.debug("语义索引初始化失败: %s", e)
        return self._semantic_index

    def _auto_index(self, entry: MemoryEntry) -> None:
        """写入时自动索引（best-effort）."""
        if not self.config.auto_index:
            return
        try:
            idx = self._get_semantic_index()
            if idx is not None:
                idx.index(entry)
        except Exception as e:
            logger.debug("自动索引失败（不影响写入）: %s", e)

    def _auto_remove_index(self, entry_id: str) -> None:
        """从索引中删除条目（best-effort）."""
        if not self.config.auto_index:
            return
        try:
            idx = self._get_semantic_index()
            if idx is not None:
                idx.remove(entry_id)
        except Exception as e:
            logger.debug("自动删除索引失败（不影响操作）: %s", e)

    def _load_employee_entries(self, employee: str) -> list[MemoryEntry]:
        """加载指定员工的全部记忆条目（不做过滤）。"""
        path = self._employee_file(employee)
        if not path.exists():
            return []
        entries: list[MemoryEntry] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(MemoryEntry(**json.loads(line)))
            except (json.JSONDecodeError, ValueError):
                continue
        return entries

    def _apply_decay(self, entry: MemoryEntry) -> MemoryEntry:
        """计算时间衰减后的有效置信度（纯函数，不改原始数据）."""
        half_life = self.config.confidence_half_life_days
        if half_life <= 0:
            return entry
        try:
            created = datetime.fromisoformat(entry.created_at)
            age_days = (datetime.now() - created).total_seconds() / 86400
            if age_days <= 0:
                return entry
            decay_factor = 0.5 ** (age_days / half_life)
            effective = entry.confidence * decay_factor
            return entry.model_copy(update={"confidence": effective})
        except (ValueError, TypeError):
            return entry

    def _is_expired(self, entry: MemoryEntry) -> bool:
        """检查记忆是否已过期."""
        ttl = entry.ttl_days
        if ttl <= 0:
            return False
        try:
            created = datetime.fromisoformat(entry.created_at)
            age_days = (datetime.now() - created).total_seconds() / 86400
            return age_days > ttl
        except (ValueError, TypeError):
            return False

    def _enforce_capacity(self, employee: str) -> int:
        """超限时裁剪低有效置信度条目，返回裁剪数量."""
        max_entries = self.config.max_entries_per_employee
        if max_entries <= 0:
            return 0

        path = self._employee_file(employee)
        if not path.exists():
            return 0

        removed_ids: list[str] = []
        pruned = 0
        with file_lock(path):
            lines = path.read_text(encoding="utf-8").splitlines()
            entries_with_lines: list[tuple[MemoryEntry, str]] = []
            other_lines: list[str] = []

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = MemoryEntry(**json.loads(stripped))
                    entries_with_lines.append((entry, stripped))
                except (json.JSONDecodeError, ValueError):
                    other_lines.append(stripped)

            active = [(e, ln) for e, ln in entries_with_lines if not e.superseded_by]
            superseded = [(e, ln) for e, ln in entries_with_lines if e.superseded_by]

            if len(active) <= max_entries:
                return 0

            # 按有效置信度排序，保留高分
            scored = [(self._apply_decay(e).confidence, e, ln) for e, ln in active]
            scored.sort(key=lambda x: x[0], reverse=True)
            keep = scored[:max_entries]
            pruned = len(scored) - max_entries
            if pruned > 0:
                removed_ids = [entry.id for _, entry, _ in scored[max_entries:]]

            kept_lines = other_lines + [ln for _, ln in superseded] + [ln for _, _, ln in keep]
            path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")

        if removed_ids:
            for entry_id in removed_ids:
                self._auto_remove_index(entry_id)
        return pruned

    def _ensure_dir(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def _employee_file(self, employee: str) -> Path:
        safe_name = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]", "_", employee)
        if not safe_name:
            safe_name = "employee"
        return self.memory_dir / f"{safe_name}.jsonl"

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
    ) -> MemoryEntry:
        """添加一条记忆."""
        self._ensure_dir()
        effective_ttl = ttl_days if ttl_days > 0 else self.config.default_ttl_days
        entry = MemoryEntry(
            employee=employee,
            category=category,
            content=content,
            source_session=source_session,
            confidence=confidence,
            ttl_days=effective_ttl,
            tags=tags or [],
            shared=shared if category != "pattern" else True,  # pattern 默认共享
            visibility=visibility,
            trigger_condition=trigger_condition,
            applicability=applicability or [],
            origin_employee=origin_employee or employee,
        )
        path = self._employee_file(employee)
        with file_lock(path):
            with path.open("a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")
        self._auto_index(entry)
        self._enforce_capacity(employee)
        return entry

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
    ) -> list[MemoryEntry]:
        """查询员工记忆.

        Args:
            employee: 员工名称
            category: 按类别过滤（可选）
            limit: 最大返回条数
            min_confidence: 最低置信度（对比衰减后的有效值）
            include_expired: 是否包含已过期条目
            max_visibility: 可见性上限 — "private" 返回全部, "open" 只返回公开记忆
            sort_by: 排序方式 — "created_at"(默认,最新在前) | "importance"(重要性优先) | "confidence"(置信度优先)
            min_importance: 最低重要性 (0=不过滤, 1-5)
            update_access: 是否更新 last_accessed 时间戳（API 调用时设为 True）

        Returns:
            记忆列表（按 sort_by 排序，置信度为衰减后的有效值）
        """
        path = self._employee_file(employee)
        if not path.exists():
            return []

        entries: list[MemoryEntry] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = MemoryEntry(**json.loads(line))
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug("跳过损坏的记忆条目: %s", e)
                continue

            # 跳过已被覆盖的
            if entry.superseded_by:
                continue

            # TTL 过期检查
            if not include_expired and self._is_expired(entry):
                continue

            if category and entry.category != category:
                continue

            # 重要性过滤
            if min_importance > 0 and entry.importance < min_importance:
                continue

            # 可见性过滤: max_visibility="open" 时跳过 private 记忆
            if max_visibility != "private" and entry.visibility == "private":
                continue

            # 应用衰减后检查置信度
            decayed = self._apply_decay(entry)
            if decayed.confidence < min_confidence:
                continue

            entries.append(decayed)

        # 排序
        if sort_by == "importance":
            entries.sort(key=lambda e: (e.importance, e.created_at), reverse=True)
        elif sort_by == "confidence":
            entries.sort(key=lambda e: (e.confidence, e.created_at), reverse=True)
        else:
            entries.reverse()  # JSONL 按时间正序追加，reverse 即最新在前

        result = entries if limit <= 0 else entries[:limit]

        # 更新 last_accessed（best-effort，不阻塞查询）
        if update_access and result:
            self._update_last_accessed(employee, [e.id for e in result])

        return result

    def _update_last_accessed(self, employee: str, entry_ids: list[str]) -> None:
        """批量更新记忆的 last_accessed 时间戳（best-effort）."""
        path = self._employee_file(employee)
        if not path.exists():
            return
        now = datetime.now().isoformat()
        id_set = set(entry_ids)
        try:
            with file_lock(path):
                lines = path.read_text(encoding="utf-8").splitlines()
                new_lines: list[str] = []
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        entry = MemoryEntry(**json.loads(stripped))
                        if entry.id in id_set:
                            entry.last_accessed = now
                            new_lines.append(entry.model_dump_json())
                        else:
                            new_lines.append(stripped)
                    except (json.JSONDecodeError, ValueError):
                        new_lines.append(stripped)
                path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        except Exception as e:
            logger.debug("更新 last_accessed 失败（不影响查询）: %s", e)

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
        path = self._employee_file(employee)
        if not path.exists():
            return None

        with file_lock(path):
            # 读取所有条目
            lines = path.read_text(encoding="utf-8").splitlines()
            found = False
            new_lines: list[str] = []
            new_entry: MemoryEntry | None = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = MemoryEntry(**json.loads(line))
                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug("跳过损坏的条目（纠正模式）: %s", e)
                    new_lines.append(line)
                    continue

                if entry.id == old_id:
                    found = True
                    # 创建纠正记忆
                    new_entry = MemoryEntry(
                        employee=employee,
                        category="correction",
                        content=new_content,
                        source_session=source_session,
                        confidence=1.0,
                    )
                    # 标记旧记忆
                    entry.superseded_by = new_entry.id
                    entry.confidence = 0.0
                    new_lines.append(entry.model_dump_json())
                else:
                    new_lines.append(line)

            if not found:
                return None

            # 追加新记忆
            assert new_entry is not None
            new_lines.append(new_entry.model_dump_json())

            # 重写文件
            path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        # 锁外维护索引（best-effort）
        if new_entry is not None:
            self._auto_remove_index(old_id)
            self._auto_index(new_entry)

        return new_entry

    def format_for_prompt(
        self,
        employee: str,
        limit: int = 10,
        query: str = "",
        employee_tags: list[str] | None = None,
        max_visibility: str = "open",
        team_members: list[str] | None = None,
    ) -> str:
        """格式化记忆为可注入 prompt 的文本.

        Args:
            employee: 员工名称
            limit: 最大条数
            query: 查询上下文（有值时使用语义搜索优先返回相关记忆）
            employee_tags: 员工标签（用于匹配共享记忆）
            max_visibility: 可见性上限 — "private" 返回全部, "open" 只返回公开记忆
            team_members: 同团队成员名列表（注入队友的公开记忆）

        Returns:
            Markdown 格式的记忆文本，无记忆时返回空字符串
        """
        parts: list[str] = []

        # 尝试语义搜索
        own_found = False
        if query:
            try:
                from crew.memory_search import SemanticMemoryIndex

                with SemanticMemoryIndex(self.memory_dir) as index:
                    if index.has_index(employee):
                        results = index.search(employee, query, limit=limit)
                        if results:
                            entries_map = {e.id: e for e in self._load_employee_entries(employee)}
                            filtered: list[MemoryEntry] = []
                            for entry_id, _content, _score in results:
                                entry = entries_map.get(entry_id)
                                if entry is None or entry.superseded_by:
                                    continue
                                if self._is_expired(entry):
                                    continue
                                if max_visibility != "private" and entry.visibility == "private":
                                    continue
                                filtered.append(self._apply_decay(entry))
                            if filtered:
                                parts.append(self._format_entries(filtered))
                                own_found = True
            except Exception as e:
                logger.debug("语义搜索降级: %s", e)

        if not own_found:
            entries = self.query(employee, limit=limit, max_visibility=max_visibility)
            if entries:
                parts.append(self._format_entries(entries))

        # 跨员工共享记忆（标签匹配）
        shared_text = self._get_shared_memories(
            employee,
            query=query,
            employee_tags=employee_tags,
            limit=max(3, limit // 3),
        )
        if shared_text:
            parts.append(f"\n### 团队共享经验\n\n{shared_text}")

        # 同团队成员的公开记忆（不要求 shared=True）
        if team_members:
            team_entries = self.query_team(
                team_members,
                exclude_employee=employee,
                limit=max(3, limit // 3),
            )
            if team_entries:
                lines = []
                for entry in team_entries:
                    cat = {
                        "decision": "决策",
                        "estimate": "估算",
                        "finding": "发现",
                        "correction": "纠正",
                        "pattern": "模式",
                    }.get(entry.category, entry.category)
                    conf = f" (置信度: {entry.confidence:.0%})" if entry.confidence < 1.0 else ""
                    lines.append(f"- [{cat}]{conf} ({entry.employee}) {entry.content}")
                parts.append("\n### 队友近况\n\n" + "\n".join(lines))

        return "\n".join(parts)

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
            # proxied 记忆来自代理推理，不是员工本人的真实工作
            proxied = " ⚠️模拟讨论记录，非实际工作" if "proxied" in entry.tags else ""
            # pattern 额外显示触发条件
            trigger = (
                f" [触发: {entry.trigger_condition}]"
                if entry.category == "pattern" and entry.trigger_condition
                else ""
            )
            lines.append(f"- [{category_label}]{conf}{proxied}{trigger} {entry.content}")
        return "\n".join(lines)

    def _get_shared_memories(
        self,
        employee: str,
        query: str = "",
        employee_tags: list[str] | None = None,
        limit: int = 5,
    ) -> str:
        """获取其他员工的共享记忆."""
        shared_entries = self.query_shared(
            tags=employee_tags,
            exclude_employee=employee,
            limit=limit,
        )
        if not shared_entries:
            return ""

        lines = []
        for entry in shared_entries:
            tag_str = f" [{', '.join(entry.tags)}]" if entry.tags else ""
            category_label = {
                "decision": "决策",
                "estimate": "估算",
                "finding": "发现",
                "correction": "纠正",
                "pattern": "模式",
            }.get(entry.category, entry.category)
            conf = f" (置信度: {entry.confidence:.0%})" if entry.confidence < 1.0 else ""
            trigger = (
                f" [触发: {entry.trigger_condition}]"
                if entry.category == "pattern" and entry.trigger_condition
                else ""
            )
            lines.append(
                f"- [{category_label}]{conf}{tag_str}{trigger} ({entry.employee}) {entry.content}"
            )
        return "\n".join(lines)

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
            min_confidence: 最低有效置信度

        Returns:
            共享记忆列表（最新在前）
        """
        if not self.memory_dir.is_dir():
            return []

        tag_set = set(tags) if tags else None
        all_shared: list[MemoryEntry] = []

        for jsonl_file in self.memory_dir.glob("*.jsonl"):
            employee_name = jsonl_file.stem
            if employee_name == exclude_employee:
                continue

            for line in jsonl_file.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = MemoryEntry(**json.loads(stripped))
                except (json.JSONDecodeError, ValueError):
                    continue

                if not entry.shared:
                    continue
                if entry.superseded_by:
                    continue
                if self._is_expired(entry):
                    continue

                # 标签过滤：有标签要求时取交集
                if tag_set and not (tag_set & set(entry.tags)):
                    continue

                decayed = self._apply_decay(entry)
                if decayed.confidence < min_confidence:
                    continue

                all_shared.append(decayed)

        # 按创建时间降序
        all_shared.sort(key=lambda e: e.created_at, reverse=True)
        return all_shared[:limit]

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
        if not self.memory_dir.is_dir():
            return []

        member_set = set(members) - {exclude_employee}
        results: list[MemoryEntry] = []

        for jsonl_file in self.memory_dir.glob("*.jsonl"):
            employee_name = jsonl_file.stem
            if employee_name not in member_set:
                continue

            for line in jsonl_file.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = MemoryEntry(**json.loads(stripped))
                except (json.JSONDecodeError, ValueError):
                    continue

                if entry.superseded_by:
                    continue
                if self._is_expired(entry):
                    continue
                if entry.visibility == "private":
                    continue

                decayed = self._apply_decay(entry)
                if decayed.confidence < min_confidence:
                    continue
                results.append(decayed)

        results.sort(key=lambda e: e.created_at, reverse=True)
        return results[:limit]

    def query_patterns(
        self,
        employee: str = "",
        applicability: list[str] | None = None,
        limit: int = 10,
        min_confidence: float = 0.3,
    ) -> list[MemoryEntry]:
        """查询可复用的工作模式（跨员工）.

        Args:
            employee: 当前员工（排除自己的，优先看别人的）
            applicability: 按适用范围标签过滤（任一匹配即可）
            limit: 最大返回条数
            min_confidence: 最低有效置信度

        Returns:
            pattern 列表（按 verified_count 降序，再按时间降序）
        """
        if not self.memory_dir.is_dir():
            return []

        app_set = set(applicability) if applicability else None
        all_patterns: list[MemoryEntry] = []

        for jsonl_file in self.memory_dir.glob("*.jsonl"):
            for line in jsonl_file.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = MemoryEntry(**json.loads(stripped))
                except (json.JSONDecodeError, ValueError):
                    continue

                if entry.category != "pattern":
                    continue
                if entry.superseded_by:
                    continue
                if self._is_expired(entry):
                    continue

                # 适用范围过滤
                if app_set and not (app_set & set(entry.applicability)):
                    continue

                decayed = self._apply_decay(entry)
                if decayed.confidence < min_confidence:
                    continue

                all_patterns.append(decayed)

        # 排序：verified_count 降序 → 时间降序
        all_patterns.sort(
            key=lambda e: (e.verified_count, e.created_at),
            reverse=True,
        )
        return all_patterns[:limit]

    def verify_pattern(self, pattern_id: str) -> bool:
        """验证一条 pattern（verified_count +1）.

        Returns:
            True 如果找到并更新，False 如果未找到
        """
        for jsonl_file in self.memory_dir.glob("*.jsonl"):
            with file_lock(jsonl_file):
                lines = jsonl_file.read_text(encoding="utf-8").splitlines()
                new_lines: list[str] = []
                found = False
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        entry = MemoryEntry(**json.loads(stripped))
                    except (json.JSONDecodeError, ValueError):
                        new_lines.append(stripped)
                        continue

                    if entry.id == pattern_id and entry.category == "pattern":
                        entry.verified_count += 1
                        new_lines.append(entry.model_dump_json())
                        found = True
                    else:
                        new_lines.append(stripped)

                if found:
                    jsonl_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                    return True

        return False

    def count(self, employee: str) -> int:
        """返回员工的有效记忆条数（排除已覆盖和已过期的），无需加载全部内容."""
        path = self._employee_file(employee)
        if not path.exists():
            return 0
        n = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = MemoryEntry(**json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
            if entry.superseded_by:
                continue
            if self._is_expired(entry):
                continue
            n += 1
        return n

    def list_employees(self) -> list[str]:
        """列出有记忆的员工."""
        if not self.memory_dir.is_dir():
            return []
        return sorted(f.stem for f in self.memory_dir.glob("*.jsonl"))

    def delete(self, entry_id: str, employee: str | None = None) -> bool:
        """删除指定的记忆条目.

        Args:
            entry_id: 记忆条目 ID
            employee: 员工名（可选，提供后只在该员工文件中查找，提高效率）

        Returns:
            True 如果删除成功，False 如果未找到
        """
        # 如果指定了员工，只在该员工文件中查找
        if employee:
            employees = [employee]
        else:
            # 否则遍历所有员工文件
            employees = self.list_employees()

        for emp in employees:
            path = self._employee_file(emp)
            if not path.exists():
                continue

            with file_lock(path):
                # 读取所有条目
                lines = path.read_text(encoding="utf-8").splitlines()
                found = False
                new_lines: list[str] = []

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = MemoryEntry(**json.loads(line))
                        if entry.id == entry_id:
                            found = True
                            # 跳过这条记录（删除）
                            continue
                        new_lines.append(line)
                    except (json.JSONDecodeError, ValueError):
                        # 保留损坏的行
                        new_lines.append(line)

                if found:
                    # 重写文件
                    path.write_text("\n".join(new_lines) + "\n" if new_lines else "", encoding="utf-8")
                    self._auto_remove_index(entry_id)
                    return True

        return False
