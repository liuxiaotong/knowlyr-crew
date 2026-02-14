"""持久化记忆 — 每个员工独立的经验存储."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from crew.paths import file_lock, resolve_project_dir

logger = logging.getLogger(__name__)


class MemoryEntry(BaseModel):
    """单条记忆."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12], description="唯一 ID")
    employee: str = Field(description="员工标识符")
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(), description="创建时间"
    )
    category: Literal["decision", "estimate", "finding", "correction"] = Field(
        description="记忆类别"
    )
    content: str = Field(description="记忆内容")
    source_session: str = Field(default="", description="来源 session ID")
    confidence: float = Field(default=1.0, description="置信度（被纠正后降低）")
    superseded_by: str = Field(default="", description="被哪条记忆覆盖")


class MemoryStore:
    """员工记忆存储管理器.

    存储结构:
      {memory_dir}/
        {employee_name}.jsonl   — 每个员工一个文件
    """

    def __init__(self, memory_dir: Path | None = None, *, project_dir: Path | None = None):
        self.memory_dir = memory_dir if memory_dir is not None else resolve_project_dir(project_dir) / ".crew" / "memory"

    def _ensure_dir(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def _employee_file(self, employee: str) -> Path:
        return self.memory_dir / f"{employee}.jsonl"

    def add(
        self,
        employee: str,
        category: Literal["decision", "estimate", "finding", "correction"],
        content: str,
        source_session: str = "",
        confidence: float = 1.0,
    ) -> MemoryEntry:
        """添加一条记忆."""
        self._ensure_dir()
        entry = MemoryEntry(
            employee=employee,
            category=category,
            content=content,
            source_session=source_session,
            confidence=confidence,
        )
        with self._employee_file(employee).open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")
        return entry

    def add_from_session(
        self,
        *,
        employee: str,
        session_id: str,
        summary: str,
        category: Literal["decision", "estimate", "finding", "correction"] = "finding",
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
    ) -> list[MemoryEntry]:
        """查询员工记忆.

        Args:
            employee: 员工名称
            category: 按类别过滤（可选）
            limit: 最大返回条数
            min_confidence: 最低置信度

        Returns:
            记忆列表（最新在前）
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

            if category and entry.category != category:
                continue

            if entry.confidence < min_confidence:
                continue

            entries.append(entry)

        entries.reverse()
        return entries[:limit]

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

        return new_entry

    def format_for_prompt(
        self,
        employee: str,
        limit: int = 10,
        query: str = "",
    ) -> str:
        """格式化记忆为可注入 prompt 的文本.

        Args:
            employee: 员工名称
            limit: 最大条数
            query: 查询上下文（有值时使用语义搜索优先返回相关记忆）

        Returns:
            Markdown 格式的记忆文本，无记忆时返回空字符串
        """
        # 尝试语义搜索
        if query:
            try:
                from crew.memory_search import SemanticMemoryIndex

                with SemanticMemoryIndex(self.memory_dir) as index:
                    if index.has_index(employee):
                        results = index.search(employee, query, limit=limit)
                        if results:
                            lines = []
                            for _id, content, score in results:
                                lines.append(f"- {content}")
                            return "\n".join(lines)
            except Exception as e:
                logger.debug("语义搜索降级: %s", e)

        entries = self.query(employee, limit=limit)
        if not entries:
            return ""

        lines = []
        for entry in entries:
            category_label = {
                "decision": "决策",
                "estimate": "估算",
                "finding": "发现",
                "correction": "纠正",
            }.get(entry.category, entry.category)
            conf = f" (置信度: {entry.confidence:.0%})" if entry.confidence < 1.0 else ""
            lines.append(f"- [{category_label}]{conf} {entry.content}")

        return "\n".join(lines)

    def list_employees(self) -> list[str]:
        """列出有记忆的员工."""
        if not self.memory_dir.is_dir():
            return []
        return sorted(
            f.stem for f in self.memory_dir.glob("*.jsonl")
        )
