"""记忆草稿管理 — 存储待审核的记忆提案."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MemoryDraft(BaseModel):
    """记忆草稿."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    employee: str = Field(description="员工名称")
    category: Literal["decision", "estimate", "finding", "correction", "pattern"] = Field(
        description="记忆类别"
    )
    content: str = Field(description="记忆内容")
    tags: list[str] = Field(default_factory=list, description="标签")
    confidence: float = Field(default=1.0, description="置信度")
    source_trajectory_id: str = Field(default="", description="来源轨迹 ID")
    status: Literal["pending", "approved", "rejected"] = Field(
        default="pending", description="审核状态"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(), description="创建时间"
    )
    reviewed_at: str = Field(default="", description="审核时间")
    reviewed_by: str = Field(default="", description="审核人")
    reject_reason: str = Field(default="", description="拒绝原因")


class MemoryDraftStore:
    """记忆草稿存储管理器."""

    def __init__(self, drafts_dir: Path | None = None):
        """初始化草稿存储.

        Args:
            drafts_dir: 草稿存储目录，默认 /data/memory_drafts
        """
        self.drafts_dir = drafts_dir or Path("/data/memory_drafts")
        self.drafts_dir.mkdir(parents=True, exist_ok=True)

    def create_draft(
        self,
        employee: str,
        category: str,
        content: str,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        source_trajectory_id: str = "",
    ) -> MemoryDraft:
        """创建新的记忆草稿.

        Args:
            employee: 员工名称
            category: 记忆类别
            content: 记忆内容
            tags: 标签列表
            confidence: 置信度
            source_trajectory_id: 来源轨迹 ID

        Returns:
            创建的草稿对象
        """
        draft = MemoryDraft(
            employee=employee,
            category=category,
            content=content,
            tags=tags or [],
            confidence=confidence,
            source_trajectory_id=source_trajectory_id,
        )

        # 写入文件
        draft_file = self.drafts_dir / f"{draft.id}.json"
        draft_file.write_text(draft.model_dump_json(indent=2), encoding="utf-8")

        logger.info("创建记忆草稿: id=%s employee=%s category=%s", draft.id, employee, category)
        return draft

    def get_draft(self, draft_id: str) -> MemoryDraft | None:
        """获取草稿详情.

        Args:
            draft_id: 草稿 ID

        Returns:
            草稿对象，不存在返回 None
        """
        draft_file = self.drafts_dir / f"{draft_id}.json"
        if not draft_file.exists():
            return None

        try:
            data = json.loads(draft_file.read_text(encoding="utf-8"))
            return MemoryDraft(**data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("草稿文件损坏: %s, error=%s", draft_file, e)
            return None

    def list_drafts(
        self,
        status: str | None = None,
        employee: str | None = None,
        limit: int = 100,
    ) -> list[MemoryDraft]:
        """列出草稿.

        Args:
            status: 按状态过滤（pending/approved/rejected）
            employee: 按员工过滤
            limit: 最大返回数量

        Returns:
            草稿列表（按创建时间倒序）
        """
        drafts: list[MemoryDraft] = []

        for draft_file in self.drafts_dir.glob("*.json"):
            try:
                data = json.loads(draft_file.read_text(encoding="utf-8"))
                draft = MemoryDraft(**data)

                # 过滤
                if status and draft.status != status:
                    continue
                if employee and draft.employee != employee:
                    continue

                drafts.append(draft)
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug("跳过损坏的草稿文件: %s, error=%s", draft_file, e)
                continue

        # 按创建时间倒序
        drafts.sort(key=lambda d: d.created_at, reverse=True)
        return drafts[:limit]

    def approve_draft(
        self,
        draft_id: str,
        reviewed_by: str = "system",
    ) -> MemoryDraft | None:
        """批准草稿.

        Args:
            draft_id: 草稿 ID
            reviewed_by: 审核人

        Returns:
            更新后的草稿对象，不存在返回 None
        """
        draft = self.get_draft(draft_id)
        if draft is None:
            return None

        if draft.status != "pending":
            logger.warning("草稿已审核，无法重复批准: id=%s status=%s", draft_id, draft.status)
            return draft

        draft.status = "approved"
        draft.reviewed_at = datetime.now().isoformat()
        draft.reviewed_by = reviewed_by

        # 更新文件
        draft_file = self.drafts_dir / f"{draft_id}.json"
        draft_file.write_text(draft.model_dump_json(indent=2), encoding="utf-8")

        logger.info("批准记忆草稿: id=%s employee=%s", draft_id, draft.employee)
        return draft

    def reject_draft(
        self,
        draft_id: str,
        reason: str = "",
        reviewed_by: str = "system",
    ) -> MemoryDraft | None:
        """拒绝草稿.

        Args:
            draft_id: 草稿 ID
            reason: 拒绝原因
            reviewed_by: 审核人

        Returns:
            更新后的草稿对象，不存在返回 None
        """
        draft = self.get_draft(draft_id)
        if draft is None:
            return None

        if draft.status != "pending":
            logger.warning("草稿已审核，无法重复拒绝: id=%s status=%s", draft_id, draft.status)
            return draft

        draft.status = "rejected"
        draft.reviewed_at = datetime.now().isoformat()
        draft.reviewed_by = reviewed_by
        draft.reject_reason = reason

        # 更新文件
        draft_file = self.drafts_dir / f"{draft_id}.json"
        draft_file.write_text(draft.model_dump_json(indent=2), encoding="utf-8")

        logger.info("拒绝记忆草稿: id=%s employee=%s reason=%s", draft_id, draft.employee, reason)
        return draft

    def delete_draft(self, draft_id: str) -> bool:
        """删除草稿.

        Args:
            draft_id: 草稿 ID

        Returns:
            True 如果删除成功，False 如果不存在
        """
        draft_file = self.drafts_dir / f"{draft_id}.json"
        if not draft_file.exists():
            return False

        draft_file.unlink()
        logger.info("删除记忆草稿: id=%s", draft_id)
        return True

    def count_by_status(self) -> dict[str, int]:
        """统计各状态的草稿数量.

        Returns:
            状态计数字典: {"pending": 10, "approved": 5, "rejected": 2}
        """
        counts: dict[str, int] = {"pending": 0, "approved": 0, "rejected": 0}

        for draft_file in self.drafts_dir.glob("*.json"):
            try:
                data = json.loads(draft_file.read_text(encoding="utf-8"))
                status = data.get("status", "pending")
                if status in counts:
                    counts[status] += 1
            except (json.JSONDecodeError, ValueError):
                continue

        return counts
