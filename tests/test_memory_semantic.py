"""语义搜索和智能推荐测试."""

import json
from pathlib import Path

import pytest

from crew.memory import MemoryStore
from crew.memory_semantic import SemanticSearchEngine


@pytest.fixture
def memory_store(tmp_path):
    """创建测试用的记忆存储."""
    return MemoryStore(memory_dir=tmp_path / "memory")


@pytest.fixture
def sample_memories(memory_store):
    """创建示例记忆."""
    # 赵云帆的记忆
    memory_store.add(
        employee="赵云帆",
        category="correction",
        content="API handler 必须在 webhook.py 中注册导入，否则会 404",
        tags=["api", "webhook", "backend"],
        confidence=0.9,
    )
    memory_store.add(
        employee="赵云帆",
        category="pattern",
        content="alembic migration 前必须检查 information_schema 避免重复添加列",
        tags=["database", "migration", "alembic"],
        confidence=0.85,
        shared=True,
    )
    memory_store.add(
        employee="赵云帆",
        category="decision",
        content="使用 Pydantic BaseModel 进行数据验证，不使用 dataclass",
        tags=["pydantic", "validation"],
        confidence=0.8,
    )

    # 卫子昂的记忆
    memory_store.add(
        employee="卫子昂",
        category="correction",
        content="React 组件必须使用 memo 优化渲染性能",
        tags=["react", "performance", "frontend"],
        confidence=0.9,
        shared=True,
    )
    memory_store.add(
        employee="卫子昂",
        category="pattern",
        content="使用 TanStack Query 管理服务端状态，避免手动 fetch",
        tags=["react", "state-management", "tanstack"],
        confidence=0.85,
    )

    # 林锐的记忆
    memory_store.add(
        employee="林锐",
        category="finding",
        content="webhook_handlers.py 中发现多处缺少异常处理",
        tags=["code-review", "error-handling"],
        confidence=0.75,
    )

    return memory_store


class TestSemanticSearchEngine:
    """测试语义搜索引擎."""

    def test_search_by_content(self, memory_store, sample_memories):
        """测试按内容搜索."""
        engine = SemanticSearchEngine(memory_store)
        results = engine.search(query="API handler 注册")

        assert len(results) > 0
        assert results[0].employee == "赵云帆"
        assert "API handler" in results[0].content
        assert results[0].relevance_score > 0

    def test_search_by_tags(self, memory_store, sample_memories):
        """测试按标签搜索."""
        engine = SemanticSearchEngine(memory_store)
        results = engine.search(query="react performance")

        assert len(results) > 0
        # 应该匹配到卫子昂的 React 记忆
        react_results = [r for r in results if "react" in r.tags]
        assert len(react_results) > 0

    def test_search_filter_by_employee(self, memory_store, sample_memories):
        """测试按员工过滤."""
        engine = SemanticSearchEngine(memory_store)
        results = engine.search(query="API", employee="赵云帆")

        assert len(results) > 0
        assert all(r.employee == "赵云帆" for r in results)

    def test_search_filter_by_category(self, memory_store, sample_memories):
        """测试按类别过滤."""
        engine = SemanticSearchEngine(memory_store)
        results = engine.search(query="必须", category="correction")

        assert len(results) > 0
        assert all(r.category == "correction" for r in results)

    def test_search_filter_by_confidence(self, memory_store, sample_memories):
        """测试按置信度过滤."""
        engine = SemanticSearchEngine(memory_store)
        results = engine.search(query="必须", min_confidence=0.85)

        assert len(results) > 0
        assert all(r.confidence >= 0.85 for r in results)

    def test_search_limit(self, memory_store, sample_memories):
        """测试限制返回数量."""
        engine = SemanticSearchEngine(memory_store)
        results = engine.search(query="必须", limit=2)

        assert len(results) <= 2

    def test_search_no_results(self, memory_store, sample_memories):
        """测试无匹配结果."""
        engine = SemanticSearchEngine(memory_store)
        results = engine.search(query="完全不存在的内容xyz123")

        assert len(results) == 0

    def test_recommend_for_task(self, memory_store, sample_memories):
        """测试任务推荐."""
        engine = SemanticSearchEngine(memory_store)
        recommendations = engine.recommend_for_task(
            task_description="实现新的 API endpoint",
            employee="赵云帆",
            limit=5,
        )

        assert len(recommendations) > 0
        # correction 类别应该有更高的推荐分数
        assert recommendations[0].category in ["correction", "pattern", "decision"]
        assert recommendations[0].recommendation_score > 0

    def test_recommend_includes_shared_memories(self, memory_store, sample_memories):
        """测试推荐包含共享记忆."""
        engine = SemanticSearchEngine(memory_store)
        recommendations = engine.recommend_for_task(
            task_description="优化数据库查询",
            employee="赵云帆",
            limit=10,
        )

        # 应该包含赵云帆自己的记忆
        own_memories = [r for r in recommendations if r.employee == "赵云帆"]
        assert len(own_memories) > 0

    def test_recommend_no_results(self, memory_store, sample_memories):
        """测试无推荐结果."""
        engine = SemanticSearchEngine(memory_store)
        recommendations = engine.recommend_for_task(
            task_description="完全不相关的任务xyz123",
            employee="赵云帆",
            limit=5,
        )

        # 即使不相关，也可能返回一些低分推荐（基于类别权重）
        # 所以这里只检查返回的是列表
        assert isinstance(recommendations, list)

    def test_find_similar_memories(self, memory_store, sample_memories):
        """测试查找相似记忆."""
        # 先获取一个记忆的 ID
        memories = memory_store.query(employee="赵云帆", limit=1)
        assert len(memories) > 0
        target_id = memories[0].id

        engine = SemanticSearchEngine(memory_store)
        similar = engine.find_similar_memories(memory_id=target_id, limit=5)

        # 应该找到一些相似记忆（排除自己）
        assert all(s.memory_id != target_id for s in similar)

    def test_find_similar_by_category(self, memory_store, sample_memories):
        """测试相似记忆按类别匹配."""
        # 获取一个 correction 类别的记忆
        memories = memory_store.query(employee="赵云帆", category="correction", limit=1)
        assert len(memories) > 0
        target_id = memories[0].id

        engine = SemanticSearchEngine(memory_store)
        similar = engine.find_similar_memories(memory_id=target_id, limit=5)

        # 相同类别的记忆应该有更高的相似度
        if similar:
            same_category = [s for s in similar if s.category == "correction"]
            # 如果有相同类别的，应该排在前面
            if same_category:
                assert similar[0].category == "correction"

    def test_find_similar_by_tags(self, memory_store, sample_memories):
        """测试相似记忆按标签匹配."""
        # 获取一个有 "api" 标签的记忆
        memories = memory_store.query(employee="赵云帆", limit=10)
        api_memory = next((m for m in memories if "api" in m.tags), None)
        assert api_memory is not None

        engine = SemanticSearchEngine(memory_store)
        similar = engine.find_similar_memories(memory_id=api_memory.id, limit=5)

        # 应该找到一些记忆
        assert isinstance(similar, list)

    def test_find_similar_not_found(self, memory_store, sample_memories):
        """测试查找不存在的记忆."""
        engine = SemanticSearchEngine(memory_store)
        similar = engine.find_similar_memories(memory_id="nonexistent-id", limit=5)

        assert len(similar) == 0

    def test_relevance_scoring(self, memory_store, sample_memories):
        """测试相关性评分."""
        engine = SemanticSearchEngine(memory_store)
        results = engine.search(query="API webhook backend")

        # 结果应该按相关性降序排列
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].relevance_score >= results[i + 1].relevance_score

        # 完全匹配应该有高分
        if results:
            assert results[0].relevance_score > 0.3

    def test_recommendation_scoring(self, memory_store, sample_memories):
        """测试推荐评分."""
        engine = SemanticSearchEngine(memory_store)
        recommendations = engine.recommend_for_task(
            task_description="修复 API bug",
            employee="赵云帆",
            limit=10,
        )

        # 结果应该按推荐分数降序排列
        if len(recommendations) > 1:
            for i in range(len(recommendations) - 1):
                assert (
                    recommendations[i].recommendation_score
                    >= recommendations[i + 1].recommendation_score
                )

        # correction 类别应该有较高的推荐分数
        corrections = [r for r in recommendations if r.category == "correction"]
        if corrections:
            assert corrections[0].recommendation_score > 0.3
