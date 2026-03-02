"""记忆查询性能优化 — 索引和缓存."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from crew.memory import MemoryEntry, MemoryStore

logger = logging.getLogger(__name__)


class MemoryIndex(BaseModel):
    """记忆索引."""

    employee: str = Field(description="员工名")
    category_index: dict[str, list[str]] = Field(
        default_factory=dict,
        description="类别索引: {category: [memory_ids]}",
    )
    tag_index: dict[str, list[str]] = Field(
        default_factory=dict,
        description="标签索引: {tag: [memory_ids]}",
    )
    last_updated: float = Field(
        default_factory=time.time,
        description="最后更新时间戳",
    )


class MemoryCache:
    """记忆缓存（内存中）."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        """初始化缓存.

        Args:
            max_size: 最大缓存条目数
            ttl_seconds: 缓存过期时间（秒）
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[Any, float]] = {}  # {key: (value, timestamp)}

    def get(self, key: str) -> Any | None:
        """获取缓存值.

        Args:
            key: 缓存键

        Returns:
            缓存值，不存在或过期返回 None
        """
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]
        if time.time() - timestamp > self.ttl_seconds:
            # 过期，删除
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any) -> None:
        """设置缓存值.

        Args:
            key: 缓存键
            value: 缓存值
        """
        # 如果缓存满了，删除最旧的条目
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        self._cache[key] = (value, time.time())

    def invalidate(self, key: str) -> None:
        """使缓存失效.

        Args:
            key: 缓存键
        """
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        """清空缓存."""
        self._cache.clear()

    def stats(self) -> dict[str, Any]:
        """获取缓存统计.

        Returns:
            统计信息
        """
        now = time.time()
        valid_count = sum(
            1 for _, timestamp in self._cache.values() if now - timestamp <= self.ttl_seconds
        )

        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_count,
            "expired_entries": len(self._cache) - valid_count,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }


class OptimizedMemoryStore:
    """优化的记忆存储（带索引和缓存）."""

    def __init__(
        self,
        memory_store: MemoryStore | None = None,
        index_dir: Path | None = None,
        enable_cache: bool = True,
        cache_size: int = 1000,
        cache_ttl: float = 300.0,
    ):
        """初始化优化存储.

        Args:
            memory_store: 底层记忆存储
            index_dir: 索引存储目录，默认 /data/memory_indexes
            enable_cache: 是否启用缓存
            cache_size: 缓存大小
            cache_ttl: 缓存过期时间（秒）
        """
        self.memory_store = memory_store or MemoryStore()
        self.index_dir = index_dir or Path("/data/memory_indexes")
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.enable_cache = enable_cache
        self.cache = MemoryCache(max_size=cache_size, ttl_seconds=cache_ttl) if enable_cache else None

        # 索引缓存（内存中）
        self._index_cache: dict[str, MemoryIndex] = {}

    def query(
        self,
        employee: str,
        category: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """优化的查询（使用索引和缓存）.

        Args:
            employee: 员工名
            category: 类别过滤
            tags: 标签过滤
            limit: 最大返回数量

        Returns:
            记忆列表
        """
        # 构建缓存键
        cache_key = f"query:{employee}:{category}:{','.join(tags or [])}:{limit}"

        # 尝试从缓存获取
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug("缓存命中: %s", cache_key)
                return cached

        # 加载索引
        index = self._load_or_build_index(employee)

        # 使用索引过滤
        candidate_ids: set[str] | None = None

        if category:
            # 按类别过滤
            category_ids = set(index.category_index.get(category, []))
            candidate_ids = category_ids if candidate_ids is None else candidate_ids & category_ids

        if tags:
            # 按标签过滤（交集）
            for tag in tags:
                tag_ids = set(index.tag_index.get(tag, []))
                candidate_ids = tag_ids if candidate_ids is None else candidate_ids & tag_ids

        # 如果没有过滤条件，加载所有记忆
        if candidate_ids is None:
            memories = self.memory_store.query(employee=employee, limit=limit)
        else:
            # 根据索引加载记忆
            memories = self._load_memories_by_ids(employee, list(candidate_ids), limit)

        # 缓存结果
        if self.cache:
            self.cache.set(cache_key, memories)

        return memories

    def rebuild_index(self, employee: str) -> MemoryIndex:
        """重建索引.

        Args:
            employee: 员工名

        Returns:
            索引对象
        """
        logger.info("重建索引: employee=%s", employee)

        # 加载所有记忆
        memories = self.memory_store.query(employee=employee, limit=10000)

        # 构建索引
        category_index: dict[str, list[str]] = {}
        tag_index: dict[str, list[str]] = {}

        for memory in memories:
            # 类别索引
            if memory.category not in category_index:
                category_index[memory.category] = []
            category_index[memory.category].append(memory.id)

            # 标签索引
            for tag in memory.tags:
                if tag not in tag_index:
                    tag_index[tag] = []
                tag_index[tag].append(memory.id)

        index = MemoryIndex(
            employee=employee,
            category_index=category_index,
            tag_index=tag_index,
        )

        # 保存索引
        index_file = self.index_dir / f"{employee}.json"
        index_file.write_text(index.model_dump_json(indent=2), encoding="utf-8")

        # 更新内存缓存
        self._index_cache[employee] = index

        logger.info(
            "索引已重建: employee=%s categories=%d tags=%d",
            employee,
            len(category_index),
            len(tag_index),
        )

        return index

    def invalidate_cache(self, employee: str | None = None) -> None:
        """使缓存失效.

        Args:
            employee: 员工名，None 表示清空所有缓存
        """
        if not self.cache:
            return

        if employee is None:
            self.cache.clear()
            logger.info("已清空所有缓存")
        else:
            # 清除该员工相关的缓存
            keys_to_remove = [k for k in self.cache._cache.keys() if k.startswith(f"query:{employee}:")]
            for key in keys_to_remove:
                self.cache.invalidate(key)
            logger.info("已清除缓存: employee=%s count=%d", employee, len(keys_to_remove))

    def get_cache_stats(self) -> dict[str, Any]:
        """获取缓存统计.

        Returns:
            统计信息
        """
        if not self.cache:
            return {"enabled": False}

        stats = self.cache.stats()
        stats["enabled"] = True
        return stats

    def get_index_stats(self, employee: str) -> dict[str, Any]:
        """获取索引统计.

        Args:
            employee: 员工名

        Returns:
            统计信息
        """
        index = self._load_or_build_index(employee)

        total_memories = sum(len(ids) for ids in index.category_index.values())

        return {
            "employee": employee,
            "total_memories": total_memories,
            "categories": len(index.category_index),
            "tags": len(index.tag_index),
            "last_updated": index.last_updated,
        }

    def _load_or_build_index(self, employee: str) -> MemoryIndex:
        """加载或构建索引.

        Args:
            employee: 员工名

        Returns:
            索引对象
        """
        # 检查内存缓存
        if employee in self._index_cache:
            return self._index_cache[employee]

        # 尝试从磁盘加载
        index_file = self.index_dir / f"{employee}.json"
        if index_file.exists():
            try:
                data = json.loads(index_file.read_text(encoding="utf-8"))
                index = MemoryIndex(**data)

                # 检查索引是否过期（超过 1 小时）
                if time.time() - index.last_updated < 3600:
                    self._index_cache[employee] = index
                    return index
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("加载索引失败: %s, error=%s", index_file, e)

        # 构建新索引
        return self.rebuild_index(employee)

    def _load_memories_by_ids(
        self,
        employee: str,
        memory_ids: list[str],
        limit: int,
    ) -> list[MemoryEntry]:
        """根据 ID 列表加载记忆.

        Args:
            employee: 员工名
            memory_ids: 记忆 ID 列表
            limit: 最大返回数量

        Returns:
            记忆列表
        """
        # 加载所有记忆并过滤
        all_memories = self.memory_store.query(employee=employee, limit=10000)
        memory_map = {m.id: m for m in all_memories}

        memories = []
        for memory_id in memory_ids:
            if memory_id in memory_map:
                memories.append(memory_map[memory_id])
            if len(memories) >= limit:
                break

        return memories
