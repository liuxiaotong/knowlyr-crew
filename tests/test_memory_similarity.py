"""测试记忆相似度检测."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def temp_memory_dir(tmp_path: Path) -> Path:
    """创建临时记忆目录."""
    memory_dir = tmp_path / ".crew" / "memory"
    memory_dir.mkdir(parents=True)
    return memory_dir


def test_cosine_similarity():
    """测试余弦相似度计算."""
    from crew.memory_similarity import cosine_similarity

    # 完全相同的向量
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    assert cosine_similarity(vec1, vec2) == pytest.approx(1.0)

    # 完全相反的向量
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [-1.0, 0.0, 0.0]
    assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    # 正交向量
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)

    # 部分相似
    vec1 = [1.0, 1.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    similarity = cosine_similarity(vec1, vec2)
    assert 0.5 < similarity < 1.0


def test_jaccard_similarity():
    """测试 Jaccard 相似度计算."""
    from crew.memory_similarity import jaccard_similarity

    # 完全相同
    set1 = {"a", "b", "c"}
    set2 = {"a", "b", "c"}
    assert jaccard_similarity(set1, set2) == pytest.approx(1.0)

    # 完全不同
    set1 = {"a", "b", "c"}
    set2 = {"d", "e", "f"}
    assert jaccard_similarity(set1, set2) == pytest.approx(0.0)

    # 部分重叠
    set1 = {"a", "b", "c"}
    set2 = {"b", "c", "d"}
    # 交集 2，并集 4
    assert jaccard_similarity(set1, set2) == pytest.approx(0.5)

    # 空集
    set1 = set()
    set2 = {"a", "b"}
    assert jaccard_similarity(set1, set2) == pytest.approx(0.0)


def test_extract_keywords():
    """测试关键词提取."""
    from crew.memory_similarity import _extract_keywords

    text = "修复了数据库连接池的内存泄漏问题，原因是连接未正确关闭"
    keywords = _extract_keywords(text, top_n=5)

    assert isinstance(keywords, set)
    assert len(keywords) <= 5
    # 应该包含一些高频字
    assert any(char in keywords for char in ["修", "复", "数", "据", "库"])


def test_keyword_similarity():
    """测试基于关键词的相似度."""
    from crew.memory_similarity import _keyword_similarity

    # 相似的文本
    text1 = "修复了数据库连接池的内存泄漏问题"
    text2 = "解决了数据库连接池内存泄漏的bug"
    similarity = _keyword_similarity(text1, text2)
    assert similarity > 0.3  # 应该有一定相似度

    # 完全不同的文本
    text1 = "修复了数据库连接池的内存泄漏问题"
    text2 = "今天天气很好，适合出去玩"
    similarity = _keyword_similarity(text1, text2)
    assert similarity < 0.2  # 相似度应该很低


def test_embedding_cache(temp_memory_dir: Path):
    """测试 embedding 缓存."""
    from crew.memory_similarity import (
        _get_embedding_cache_path,
        _load_embedding_cache,
        _save_embedding_cache,
    )

    employee = "测试员工"

    # 初始缓存为空
    cache = _load_embedding_cache(temp_memory_dir, employee)
    assert cache == {}

    # 保存缓存
    test_cache = {
        "entry1": [0.1, 0.2, 0.3],
        "entry2": [0.4, 0.5, 0.6],
    }
    _save_embedding_cache(temp_memory_dir, employee, test_cache)

    # 加载缓存
    loaded_cache = _load_embedding_cache(temp_memory_dir, employee)
    assert loaded_cache == test_cache

    # 检查文件存在
    cache_path = _get_embedding_cache_path(temp_memory_dir, employee)
    assert cache_path.exists()


@pytest.mark.asyncio
async def test_find_similar_memories_no_memories(tmp_path: Path):
    """测试没有记忆时的相似度检测."""
    from crew.memory_similarity import find_similar_memories

    # 创建空的记忆存储
    project_dir = tmp_path
    memory_dir = project_dir / ".crew" / "memory"
    memory_dir.mkdir(parents=True)

    similar = await find_similar_memories(
        employee="测试员工",
        content="这是一条新记忆",
        category="finding",
        project_dir=project_dir,
    )

    assert similar == []


@pytest.mark.asyncio
async def test_find_similar_memories_with_keyword_fallback(tmp_path: Path):
    """测试关键词匹配降级方案."""
    from crew.memory import MemoryStore
    from crew.memory_similarity import find_similar_memories

    # 创建记忆存储
    project_dir = tmp_path
    store = MemoryStore(project_dir=project_dir)

    # 添加一些记忆
    store.add(
        employee="测试员工",
        category="finding",
        content="修复了数据库连接池的内存泄漏问题，原因是连接未正确关闭",
    )

    store.add(
        employee="测试员工",
        category="finding",
        content="今天天气很好，适合出去玩",
    )

    # 查找相似记忆（使用关键词匹配，因为没有 OpenAI API key）
    similar = await find_similar_memories(
        employee="测试员工",
        content="解决了数据库连接池内存泄漏的bug",
        category="finding",
        threshold=0.15,  # 降低阈值以便测试（关键词匹配相似度较低）
        project_dir=project_dir,
        use_keyword_fallback=True,
    )

    # 应该找到相似的记忆
    assert len(similar) > 0
    mem, score = similar[0]
    assert "数据库" in mem["content"] or "连接池" in mem["content"] or "内存" in mem["content"]
    assert score > 0.0


@pytest.mark.asyncio
async def test_find_similar_memories_different_category(tmp_path: Path):
    """测试不同类别的记忆不会被检测为相似."""
    from crew.memory import MemoryStore
    from crew.memory_similarity import find_similar_memories

    # 创建记忆存储
    project_dir = tmp_path
    store = MemoryStore(project_dir=project_dir)

    # 添加不同类别的记忆
    store.add(
        employee="测试员工",
        category="decision",
        content="决定使用 PostgreSQL 作为主数据库",
    )

    # 查找 finding 类别的相似记忆
    similar = await find_similar_memories(
        employee="测试员工",
        content="决定使用 PostgreSQL 作为主数据库",
        category="finding",  # 不同类别
        project_dir=project_dir,
    )

    # 不应该找到相似记忆（因为类别不同）
    assert similar == []


def test_get_openai_api_key_not_configured(monkeypatch):
    """测试 OpenAI API key 未配置的情况."""
    from crew.memory_similarity import _get_openai_api_key

    # 清除环境变量
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    api_key = _get_openai_api_key()
    assert api_key is None


def test_get_openai_api_key_configured(monkeypatch):
    """测试 OpenAI API key 已配置的情况."""
    from crew.memory_similarity import _get_openai_api_key

    # 设置环境变量
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    api_key = _get_openai_api_key()
    assert api_key == "sk-test-key"


@pytest.mark.asyncio
async def test_get_embedding_no_api_key(monkeypatch):
    """测试没有 API key 时的 embedding 计算."""
    from crew.memory_similarity import get_embedding

    # 清除环境变量
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    embedding = await get_embedding("测试文本")
    assert embedding is None


@pytest.mark.asyncio
async def test_find_similar_memories_threshold(tmp_path: Path):
    """测试相似度阈值过滤."""
    from crew.memory import MemoryStore
    from crew.memory_similarity import find_similar_memories

    # 创建记忆存储
    project_dir = tmp_path
    store = MemoryStore(project_dir=project_dir)

    # 添加记忆
    store.add(
        employee="测试员工",
        category="finding",
        content="修复了数据库连接池的内存泄漏问题",
    )

    # 使用很高的阈值（0.95），应该找不到相似记忆
    similar = await find_similar_memories(
        employee="测试员工",
        content="解决了数据库连接池内存泄漏的bug",
        category="finding",
        threshold=0.95,  # 很高的阈值
        project_dir=project_dir,
    )

    # 因为阈值太高，应该找不到
    assert len(similar) == 0
