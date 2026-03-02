"""共享记忆统计测试."""

from pathlib import Path

import pytest

from crew.memory import MemoryStore
from crew.memory_shared_stats import SharedMemoryStats


@pytest.fixture
def temp_dirs(tmp_path):
    """创建临时目录."""
    memory_dir = tmp_path / "memory"
    stats_dir = tmp_path / "stats"
    memory_dir.mkdir()
    stats_dir.mkdir()
    return memory_dir, stats_dir


@pytest.fixture
def memory_store(temp_dirs):
    """创建测试用的记忆存储."""
    memory_dir, _ = temp_dirs
    return MemoryStore(memory_dir=memory_dir)


@pytest.fixture
def stats(temp_dirs):
    """创建测试用的统计管理器."""
    _, stats_dir = temp_dirs
    return SharedMemoryStats(stats_dir=stats_dir)


class TestSharedMemoryStats:
    """测试共享记忆统计."""

    def test_record_usage(self, stats):
        """测试记录使用."""
        stats.record_usage(
            memory_id="mem123",
            memory_owner="赵云帆",
            used_by="卫子昂",
            context="实现前端组件",
        )

        # 验证记录文件已创建
        usage_file = stats._get_usage_file("mem123")
        assert usage_file.exists()

        # 验证内容
        lines = usage_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1

    def test_get_usage_stats(self, stats):
        """测试获取使用统计."""
        # 记录多次使用
        stats.record_usage("mem123", "赵云帆", "卫子昂", "")
        stats.record_usage("mem123", "赵云帆", "林锐", "")
        stats.record_usage("mem123", "赵云帆", "卫子昂", "")  # 重复使用

        # 获取统计
        usage_stats = stats.get_usage_stats("mem123")

        assert usage_stats["total_uses"] == 3
        assert usage_stats["unique_users"] == 2
        assert set(usage_stats["users"]) == {"卫子昂", "林锐"}

    def test_get_usage_stats_nonexistent(self, stats):
        """测试获取不存在记忆的统计."""
        usage_stats = stats.get_usage_stats("nonexistent")

        assert usage_stats["total_uses"] == 0
        assert usage_stats["unique_users"] == 0
        assert usage_stats["users"] == []

    def test_get_popular_memories(self, stats):
        """测试获取热门记忆."""
        # 记录多个记忆的使用
        for _ in range(5):
            stats.record_usage("mem1", "赵云帆", "卫子昂", "")

        for _ in range(3):
            stats.record_usage("mem2", "赵云帆", "林锐", "")

        stats.record_usage("mem3", "赵云帆", "程薇", "")  # 只用一次

        # 获取热门记忆（最少 2 次使用）
        popular = stats.get_popular_memories(min_uses=2, limit=10)

        assert len(popular) == 2
        assert popular[0]["memory_id"] == "mem1"
        assert popular[0]["total_uses"] == 5
        assert popular[1]["memory_id"] == "mem2"
        assert popular[1]["total_uses"] == 3

    def test_get_user_shared_usage(self, stats):
        """测试获取用户使用记录."""
        # 记录使用
        stats.record_usage("mem1", "赵云帆", "卫子昂", "场景1")
        stats.record_usage("mem2", "林锐", "卫子昂", "场景2")
        stats.record_usage("mem3", "赵云帆", "程薇", "场景3")

        # 获取卫子昂的使用记录
        usages = stats.get_user_shared_usage("卫子昂", limit=50)

        assert len(usages) == 2
        assert usages[0].memory_owner in ["赵云帆", "林锐"]
        assert all(u.used_by == "卫子昂" for u in usages)

    def test_get_memory_owner_stats(self, stats):
        """测试获取所有者统计."""
        # 记录赵云帆的记忆被使用
        stats.record_usage("mem1", "赵云帆", "卫子昂", "")
        stats.record_usage("mem1", "赵云帆", "林锐", "")
        stats.record_usage("mem2", "赵云帆", "程薇", "")

        # 记录其他人的记忆
        stats.record_usage("mem3", "林锐", "卫子昂", "")

        # 获取赵云帆的统计
        owner_stats = stats.get_memory_owner_stats("赵云帆")

        assert owner_stats["total_memories_shared"] == 2  # mem1 和 mem2
        assert owner_stats["total_uses"] == 3
        assert owner_stats["unique_users"] == 3  # 卫子昂、林锐、程薇

    def test_get_memory_owner_stats_no_usage(self, stats):
        """测试没有使用记录的所有者."""
        owner_stats = stats.get_memory_owner_stats("不存在的员工")

        assert owner_stats["total_memories_shared"] == 0
        assert owner_stats["total_uses"] == 0
        assert owner_stats["unique_users"] == 0


class TestSharedMemoryIntegration:
    """测试共享记忆集成."""

    def test_query_shared_memories(self, memory_store):
        """测试查询共享记忆."""
        # 创建共享记忆
        memory_store.add(
            employee="赵云帆",
            category="pattern",
            content="API 设计模式",
            tags=["api", "design"],
            shared=True,
        )

        memory_store.add(
            employee="林锐",
            category="finding",
            content="代码审查发现",
            tags=["review"],
            shared=True,
        )

        # 创建非共享记忆
        memory_store.add(
            employee="赵云帆",
            category="finding",
            content="私有发现",
            shared=False,
        )

        # 查询共享记忆
        shared = memory_store.query_shared(limit=10)
        assert len(shared) == 2

        # 按标签过滤
        api_shared = memory_store.query_shared(tags=["api"], limit=10)
        assert len(api_shared) == 1
        assert api_shared[0].content == "API 设计模式"

    def test_exclude_employee_from_shared(self, memory_store):
        """测试排除指定员工的共享记忆."""
        memory_store.add(
            employee="赵云帆",
            category="pattern",
            content="赵云帆的模式",
            shared=True,
        )

        memory_store.add(
            employee="林锐",
            category="pattern",
            content="林锐的模式",
            shared=True,
        )

        # 排除赵云帆
        shared = memory_store.query_shared(exclude_employee="赵云帆", limit=10)
        assert len(shared) == 1
        assert shared[0].employee == "林锐"
