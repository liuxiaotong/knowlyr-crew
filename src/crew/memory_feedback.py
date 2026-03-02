"""记忆使用分析和反馈系统."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MemoryFeedback(BaseModel):
    """记忆反馈."""

    feedback_id: str = Field(description="反馈 ID")
    memory_id: str = Field(description="记忆 ID")
    employee: str = Field(description="员工名")
    feedback_type: Literal["helpful", "not_helpful", "outdated", "incorrect"] = Field(
        description="反馈类型"
    )
    context: str = Field(default="", description="使用场景")
    comment: str = Field(default="", description="反馈评论")
    submitted_by: str = Field(description="提交人")
    submitted_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="提交时间",
    )


class MemoryUsageStats(BaseModel):
    """记忆使用统计."""

    memory_id: str = Field(description="记忆 ID")
    employee: str = Field(description="员工名")
    total_uses: int = Field(default=0, description="总使用次数")
    helpful_count: int = Field(default=0, description="有帮助次数")
    not_helpful_count: int = Field(default=0, description="无帮助次数")
    outdated_count: int = Field(default=0, description="过时次数")
    incorrect_count: int = Field(default=0, description="错误次数")
    last_used: str = Field(default="", description="最后使用时间")
    avg_relevance_score: float = Field(default=0.0, description="平均相关性分数")


class MemoryFeedbackManager:
    """记忆反馈管理器."""

    def __init__(
        self,
        feedback_dir: Path | None = None,
        stats_dir: Path | None = None,
    ):
        """初始化反馈管理器.

        Args:
            feedback_dir: 反馈存储目录，默认 /data/memory_feedback
            stats_dir: 统计存储目录，默认 /data/memory_usage_stats
        """
        self.feedback_dir = feedback_dir or Path("/data/memory_feedback")
        self.stats_dir = stats_dir or Path("/data/memory_usage_stats")
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir.mkdir(parents=True, exist_ok=True)

    def submit_feedback(
        self,
        memory_id: str,
        employee: str,
        feedback_type: Literal["helpful", "not_helpful", "outdated", "incorrect"],
        submitted_by: str,
        context: str = "",
        comment: str = "",
    ) -> MemoryFeedback:
        """提交反馈.

        Args:
            memory_id: 记忆 ID
            employee: 员工名
            feedback_type: 反馈类型
            submitted_by: 提交人
            context: 使用场景
            comment: 反馈评论

        Returns:
            反馈对象
        """
        import uuid

        feedback_id = f"fb-{uuid.uuid4().hex[:8]}"
        feedback = MemoryFeedback(
            feedback_id=feedback_id,
            memory_id=memory_id,
            employee=employee,
            feedback_type=feedback_type,
            context=context,
            comment=comment,
            submitted_by=submitted_by,
        )

        # 保存反馈到 JSONL
        feedback_file = self.feedback_dir / f"{memory_id}.jsonl"
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(feedback.model_dump_json() + "\n")

        # 更新统计
        self._update_stats(memory_id, employee, feedback_type)

        logger.info(
            "记忆反馈已提交: memory_id=%s type=%s by=%s",
            memory_id,
            feedback_type,
            submitted_by,
        )

        return feedback

    def get_feedback(self, memory_id: str) -> list[MemoryFeedback]:
        """获取记忆的所有反馈.

        Args:
            memory_id: 记忆 ID

        Returns:
            反馈列表（按时间倒序）
        """
        feedback_file = self.feedback_dir / f"{memory_id}.jsonl"
        if not feedback_file.exists():
            return []

        feedbacks = []
        try:
            for line in feedback_file.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                data = json.loads(stripped)
                feedbacks.append(MemoryFeedback(**data))
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("读取反馈文件失败: %s, error=%s", feedback_file, e)

        # 按时间倒序
        feedbacks.sort(key=lambda f: f.submitted_at, reverse=True)
        return feedbacks

    def get_stats(self, memory_id: str) -> MemoryUsageStats | None:
        """获取记忆使用统计.

        Args:
            memory_id: 记忆 ID

        Returns:
            统计对象，不存在返回 None
        """
        stats_file = self.stats_dir / f"{memory_id}.json"
        if not stats_file.exists():
            return None

        try:
            data = json.loads(stats_file.read_text(encoding="utf-8"))
            return MemoryUsageStats(**data)
        except (json.JSONDecodeError, ValueError):
            return None

    def get_low_quality_memories(
        self,
        employee: str | None = None,
        min_uses: int = 5,
        max_helpful_ratio: float = 0.3,
    ) -> list[MemoryUsageStats]:
        """获取低质量记忆（使用多但反馈差）.

        Args:
            employee: 按员工过滤
            min_uses: 最少使用次数
            max_helpful_ratio: 最大有帮助比例

        Returns:
            低质量记忆列表（按有帮助比例升序）
        """
        low_quality = []

        for stats_file in self.stats_dir.glob("*.json"):
            try:
                data = json.loads(stats_file.read_text(encoding="utf-8"))
                stats = MemoryUsageStats(**data)

                # 过滤
                if employee and stats.employee != employee:
                    continue
                if stats.total_uses < min_uses:
                    continue

                # 计算有帮助比例
                helpful_ratio = (
                    stats.helpful_count / stats.total_uses if stats.total_uses > 0 else 0.0
                )
                if helpful_ratio <= max_helpful_ratio:
                    low_quality.append(stats)

            except (json.JSONDecodeError, ValueError) as e:
                logger.debug("跳过损坏的统计文件: %s, error=%s", stats_file, e)
                continue

        # 按有帮助比例升序排序
        low_quality.sort(
            key=lambda s: s.helpful_count / s.total_uses if s.total_uses > 0 else 0.0
        )
        return low_quality

    def get_popular_memories(
        self,
        employee: str | None = None,
        limit: int = 10,
    ) -> list[MemoryUsageStats]:
        """获取热门记忆（使用多且反馈好）.

        Args:
            employee: 按员工过滤
            limit: 最大返回数量

        Returns:
            热门记忆列表（按使用次数降序）
        """
        popular = []

        for stats_file in self.stats_dir.glob("*.json"):
            try:
                data = json.loads(stats_file.read_text(encoding="utf-8"))
                stats = MemoryUsageStats(**data)

                # 过滤
                if employee and stats.employee != employee:
                    continue

                # 计算有帮助比例
                helpful_ratio = (
                    stats.helpful_count / stats.total_uses if stats.total_uses > 0 else 0.0
                )

                # 只包含有帮助比例 > 0.5 的记忆
                if helpful_ratio > 0.5:
                    popular.append(stats)

            except (json.JSONDecodeError, ValueError) as e:
                logger.debug("跳过损坏的统计文件: %s, error=%s", stats_file, e)
                continue

        # 按使用次数降序排序
        popular.sort(key=lambda s: s.total_uses, reverse=True)
        return popular[:limit]

    def record_usage(
        self,
        memory_id: str,
        employee: str,
        relevance_score: float = 0.0,
    ) -> None:
        """记录记忆使用.

        Args:
            memory_id: 记忆 ID
            employee: 员工名
            relevance_score: 相关性分数
        """
        stats_file = self.stats_dir / f"{memory_id}.json"

        # 加载现有统计
        if stats_file.exists():
            try:
                data = json.loads(stats_file.read_text(encoding="utf-8"))
                stats = MemoryUsageStats(**data)
            except (json.JSONDecodeError, ValueError):
                stats = MemoryUsageStats(memory_id=memory_id, employee=employee)
        else:
            stats = MemoryUsageStats(memory_id=memory_id, employee=employee)

        # 更新统计
        stats.total_uses += 1
        stats.last_used = datetime.now().isoformat()

        # 更新平均相关性分数
        if relevance_score > 0:
            old_avg = stats.avg_relevance_score
            old_count = stats.total_uses - 1
            stats.avg_relevance_score = (
                (old_avg * old_count + relevance_score) / stats.total_uses
                if old_count > 0
                else relevance_score
            )

        # 保存统计
        stats_file.write_text(stats.model_dump_json(indent=2), encoding="utf-8")

    def _update_stats(
        self,
        memory_id: str,
        employee: str,
        feedback_type: str,
    ) -> None:
        """更新统计（根据反馈类型）."""
        stats_file = self.stats_dir / f"{memory_id}.json"

        # 加载现有统计
        if stats_file.exists():
            try:
                data = json.loads(stats_file.read_text(encoding="utf-8"))
                stats = MemoryUsageStats(**data)
            except (json.JSONDecodeError, ValueError):
                stats = MemoryUsageStats(memory_id=memory_id, employee=employee)
        else:
            stats = MemoryUsageStats(memory_id=memory_id, employee=employee)

        # 更新总使用次数（反馈也算一次使用）
        stats.total_uses += 1

        # 更新计数
        if feedback_type == "helpful":
            stats.helpful_count += 1
        elif feedback_type == "not_helpful":
            stats.not_helpful_count += 1
        elif feedback_type == "outdated":
            stats.outdated_count += 1
        elif feedback_type == "incorrect":
            stats.incorrect_count += 1

        # 保存统计
        stats_file.write_text(stats.model_dump_json(indent=2), encoding="utf-8")

    def get_feedback_summary(
        self,
        employee: str | None = None,
    ) -> dict[str, Any]:
        """获取反馈汇总.

        Args:
            employee: 按员工过滤

        Returns:
            汇总统计
        """
        total_memories = 0
        total_feedback = 0
        feedback_by_type = {
            "helpful": 0,
            "not_helpful": 0,
            "outdated": 0,
            "incorrect": 0,
        }

        for stats_file in self.stats_dir.glob("*.json"):
            try:
                data = json.loads(stats_file.read_text(encoding="utf-8"))
                stats = MemoryUsageStats(**data)

                # 过滤
                if employee and stats.employee != employee:
                    continue

                total_memories += 1
                total_feedback += (
                    stats.helpful_count
                    + stats.not_helpful_count
                    + stats.outdated_count
                    + stats.incorrect_count
                )
                feedback_by_type["helpful"] += stats.helpful_count
                feedback_by_type["not_helpful"] += stats.not_helpful_count
                feedback_by_type["outdated"] += stats.outdated_count
                feedback_by_type["incorrect"] += stats.incorrect_count

            except (json.JSONDecodeError, ValueError) as e:
                logger.debug("跳过损坏的统计文件: %s, error=%s", stats_file, e)
                continue

        return {
            "total_memories": total_memories,
            "total_feedback": total_feedback,
            "feedback_by_type": feedback_by_type,
            "avg_feedback_per_memory": (
                total_feedback / total_memories if total_memories > 0 else 0.0
            ),
        }
