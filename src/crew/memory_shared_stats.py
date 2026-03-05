"""共享记忆使用统计."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SharedMemoryUsage(BaseModel):
    """共享记忆使用记录."""

    memory_id: str = Field(description="记忆 ID")
    memory_owner: str = Field(description="记忆所有者")
    used_by: str = Field(description="使用者")
    used_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="使用时间",
    )
    context: str = Field(default="", description="使用场景")


class SharedMemoryStats:
    """共享记忆统计管理器."""

    def __init__(self, stats_dir: Path | None = None):
        """初始化统计管理器.

        Args:
            stats_dir: 统计数据目录，默认 /data/memory_shared_stats
        """
        self.stats_dir = stats_dir or Path("/data/memory_shared_stats")
        self.stats_dir.mkdir(parents=True, exist_ok=True)

    def _get_usage_file(self, memory_id: str) -> Path:
        """获取记忆使用记录文件路径.

        Args:
            memory_id: 记忆 ID

        Returns:
            使用记录文件路径
        """
        return self.stats_dir / f"{memory_id}.jsonl"

    def record_usage(
        self,
        memory_id: str,
        memory_owner: str,
        used_by: str,
        context: str = "",
    ) -> None:
        """记录共享记忆使用.

        Args:
            memory_id: 记忆 ID
            memory_owner: 记忆所有者
            used_by: 使用者
            context: 使用场景
        """
        usage = SharedMemoryUsage(
            memory_id=memory_id,
            memory_owner=memory_owner,
            used_by=used_by,
            context=context,
        )

        usage_file = self._get_usage_file(memory_id)
        with open(usage_file, "a", encoding="utf-8") as f:
            f.write(usage.model_dump_json() + "\n")

        logger.debug(
            "记录共享记忆使用: memory_id=%s owner=%s used_by=%s",
            memory_id,
            memory_owner,
            used_by,
        )

    def get_usage_stats(self, memory_id: str) -> dict[str, int | list[str]]:
        """获取记忆使用统计.

        Args:
            memory_id: 记忆 ID

        Returns:
            统计信息: {
                "total_uses": 总使用次数,
                "unique_users": 唯一使用者数量,
                "users": 使用者列表
            }
        """
        usage_file = self._get_usage_file(memory_id)
        if not usage_file.exists():
            return {"total_uses": 0, "unique_users": 0, "users": []}

        users: set[str] = set()
        total = 0

        try:
            for line in usage_file.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue

                usage = SharedMemoryUsage(**json.loads(stripped))
                users.add(usage.used_by)
                total += 1

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("解析使用记录失败: %s", e)

        return {
            "total_uses": total,
            "unique_users": len(users),
            "users": sorted(users),
        }

    def get_popular_memories(
        self,
        min_uses: int = 2,
        limit: int = 20,
    ) -> list[dict[str, int | str]]:
        """获取热门共享记忆.

        Args:
            min_uses: 最小使用次数
            limit: 最大返回数量

        Returns:
            热门记忆列表，按使用次数降序
        """
        memories: list[dict[str, int | str]] = []

        for usage_file in self.stats_dir.glob("*.jsonl"):
            memory_id = usage_file.stem
            stats = self.get_usage_stats(memory_id)

            if stats["total_uses"] >= min_uses:
                memories.append(
                    {
                        "memory_id": memory_id,
                        "total_uses": stats["total_uses"],
                        "unique_users": stats["unique_users"],
                    }
                )

        # 按使用次数降序
        memories.sort(key=lambda m: m["total_uses"], reverse=True)
        return memories[:limit]

    def get_user_shared_usage(
        self,
        user: str,
        limit: int = 50,
    ) -> list[SharedMemoryUsage]:
        """获取用户使用的共享记忆列表.

        Args:
            user: 用户名
            limit: 最大返回数量

        Returns:
            使用记录列表（按时间倒序）
        """
        usages: list[SharedMemoryUsage] = []

        for usage_file in self.stats_dir.glob("*.jsonl"):
            try:
                for line in usage_file.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue

                    usage = SharedMemoryUsage(**json.loads(stripped))
                    if usage.used_by == user:
                        usages.append(usage)

            except (json.JSONDecodeError, ValueError):
                continue

        # 按时间倒序
        usages.sort(key=lambda u: u.used_at, reverse=True)
        return usages[:limit]

    def get_memory_owner_stats(self, owner: str) -> dict[str, int]:
        """获取记忆所有者的共享统计.

        Args:
            owner: 记忆所有者

        Returns:
            统计信息: {
                "total_memories_shared": 被共享的记忆数量,
                "total_uses": 总使用次数,
                "unique_users": 唯一使用者数量
            }
        """
        total_memories = 0
        total_uses = 0
        all_users: set[str] = set()

        for usage_file in self.stats_dir.glob("*.jsonl"):
            try:
                is_owner_memory = False
                memory_uses = 0

                for line in usage_file.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue

                    usage = SharedMemoryUsage(**json.loads(stripped))

                    if usage.memory_owner == owner:
                        is_owner_memory = True
                        memory_uses += 1
                        all_users.add(usage.used_by)

                if is_owner_memory:
                    total_memories += 1
                    total_uses += memory_uses

            except (json.JSONDecodeError, ValueError):
                continue

        return {
            "total_memories_shared": total_memories,
            "total_uses": total_uses,
            "unique_users": len(all_users),
        }
