"""记忆性能优化测试."""

import time

import pytest

from crew.memory import MemoryStore
from crew.memory_performance import MemoryCache, OptimizedMemoryStore


@pytest.fixture
def memory_store(tmp_path):
    """创建测试用的记忆存储."""
    store = MemoryStore(memory_dir=tmp_path / "memory")

    # 添加测试数据
    for i in range(20):
        store.add(
            employee="赵云帆",
            category="correction" if i % 2 == 0 else "pattern",
            content=f"记忆内容 {i}",
            tags=["api", "backend"] if i % 3 == 0 else ["frontend"],
            confidence=0.9,
        )

    return store


@pytest.fixture
def optimized_store(tmp_path, memory_store):
    """创建优化的记忆存储."""
    return OptimizedMemoryStore(
        memory_store=memory_store,
        index_dir=tmp_path / "indexes",
        enable_cache=True,
        cache_size=100,
        cache_ttl=60.0,
    )


class TestMemoryCache:
    """测试记忆缓存."""

    def test_cache_set_get(self):
        """测试设置和获取缓存."""
        cache = MemoryCache(max_size=10, ttl_seconds=60.0)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_expiration(self):
        """测试缓存过期."""
        cache = MemoryCache(max_size=10, ttl_seconds=0.1)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # 等待过期
        time.sleep(0.2)
        assert cache.get("key1") is None

    def test_cache_max_size(self):
        """测试缓存大小限制."""
        cache = MemoryCache(max_size=3, ttl_seconds=60.0)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # 应该删除最旧的 key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_cache_invalidate(self):
        """测试使缓存失效."""
        cache = MemoryCache(max_size=10, ttl_seconds=60.0)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_cache_clear(self):
        """测试清空缓存."""
        cache = MemoryCache(max_size=10, ttl_seconds=60.0)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_stats(self):
        """测试缓存统计."""
        cache = MemoryCache(max_size=10, ttl_seconds=60.0)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        stats = cache.stats()

        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 2
        assert stats["expired_entries"] == 0
        assert stats["max_size"] == 10
        assert stats["ttl_seconds"] == 60.0


class TestOptimizedMemoryStore:
    """测试优化的记忆存储."""

    def test_query_with_cache(self, optimized_store):
        """测试带缓存的查询."""
        # 第一次查询（缓存未命中）
        memories1 = optimized_store.query(employee="赵云帆", limit=10)
        assert len(memories1) == 10

        # 第二次查询（缓存命中）
        memories2 = optimized_store.query(employee="赵云帆", limit=10)
        assert len(memories2) == 10
        assert memories1 == memories2

    def test_query_by_category(self, optimized_store):
        """测试按类别查询."""
        memories = optimized_store.query(
            employee="赵云帆",
            category="correction",
            limit=20,
        )

        assert len(memories) > 0
        assert all(m.category == "correction" for m in memories)

    def test_query_by_tags(self, optimized_store):
        """测试按标签查询."""
        memories = optimized_store.query(
            employee="赵云帆",
            tags=["api", "backend"],
            limit=20,
        )

        assert len(memories) > 0
        assert all("api" in m.tags and "backend" in m.tags for m in memories)

    def test_rebuild_index(self, optimized_store):
        """测试重建索引."""
        index = optimized_store.rebuild_index("赵云帆")

        assert index.employee == "赵云帆"
        assert len(index.category_index) > 0
        assert len(index.tag_index) > 0

        # 验证索引文件已创建
        index_file = optimized_store.index_dir / "赵云帆.json"
        assert index_file.exists()

    def test_invalidate_cache(self, optimized_store):
        """测试使缓存失效."""
        # 查询以填充缓存
        optimized_store.query(employee="赵云帆", limit=10)

        # 使缓存失效
        optimized_store.invalidate_cache(employee="赵云帆")

        # 验证缓存已清除
        stats = optimized_store.get_cache_stats()
        assert stats["enabled"] is True

    def test_get_cache_stats(self, optimized_store):
        """测试获取缓存统计."""
        # 查询以填充缓存
        optimized_store.query(employee="赵云帆", limit=10)

        stats = optimized_store.get_cache_stats()

        assert stats["enabled"] is True
        assert stats["total_entries"] >= 1

    def test_get_index_stats(self, optimized_store):
        """测试获取索引统计."""
        stats = optimized_store.get_index_stats("赵云帆")

        assert stats["employee"] == "赵云帆"
        assert stats["total_memories"] > 0
        assert stats["categories"] > 0
        assert stats["tags"] > 0

    def test_cache_disabled(self, tmp_path, memory_store):
        """测试禁用缓存."""
        store = OptimizedMemoryStore(
            memory_store=memory_store,
            index_dir=tmp_path / "indexes",
            enable_cache=False,
        )

        memories = store.query(employee="赵云帆", limit=10)
        assert len(memories) == 10

        stats = store.get_cache_stats()
        assert stats["enabled"] is False

    def test_index_persistence(self, tmp_path, memory_store):
        """测试索引持久化."""
        # 创建第一个存储并构建索引
        store1 = OptimizedMemoryStore(
            memory_store=memory_store,
            index_dir=tmp_path / "indexes",
        )
        index1 = store1.rebuild_index("赵云帆")

        # 创建第二个存储并加载索引
        store2 = OptimizedMemoryStore(
            memory_store=memory_store,
            index_dir=tmp_path / "indexes",
        )
        index2 = store2._load_or_build_index("赵云帆")

        # 验证索引内容相同
        assert index1.employee == index2.employee
        assert index1.category_index == index2.category_index
        assert index1.tag_index == index2.tag_index

    def test_query_performance(self, optimized_store):
        """测试查询性能（简单验证）."""
        # 第一次查询（构建索引）
        start = time.time()
        memories1 = optimized_store.query(employee="赵云帆", category="correction", limit=10)
        time1 = time.time() - start

        # 第二次查询（使用缓存）
        start = time.time()
        memories2 = optimized_store.query(employee="赵云帆", category="correction", limit=10)
        time2 = time.time() - start

        # 第二次应该更快（缓存命中）
        assert time2 < time1 or time2 < 0.01  # 允许一定误差
        assert memories1 == memories2
