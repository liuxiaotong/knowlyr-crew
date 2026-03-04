"""记忆相似度检测 — 基于 Embedding 的去重机制.

支持两种检测方式：
1. OpenAI Embedding（主要方式，语义相似度更准确）
2. 关键词匹配（降级方案，API 失败时使用）
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# OpenAI API 配置
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def _get_openai_api_key() -> str | None:
    """获取 OpenAI API Key（从环境变量）."""
    return os.getenv("OPENAI_API_KEY")


async def get_embedding(text: str) -> List[float] | None:
    """获取文本的 embedding 向量.

    Args:
        text: 要计算 embedding 的文本

    Returns:
        embedding 向量（1536 维），失败返回 None
    """
    api_key = _get_openai_api_key()
    if not api_key or api_key == "sk-xxx":
        logger.debug("OpenAI API key 未配置，跳过 embedding 计算")
        return None

    try:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": EMBEDDING_MODEL,
                    "input": text,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
    except Exception as e:
        logger.warning(f"OpenAI embedding 计算失败: {e}")
        return None


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算余弦相似度.

    Args:
        vec1: 向量 1
        vec2: 向量 2

    Returns:
        余弦相似度（0-1）
    """
    try:
        import numpy as np

        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
    except Exception as e:
        logger.warning(f"余弦相似度计算失败: {e}")
        return 0.0


def _extract_keywords(text: str, top_n: int = 10) -> set[str]:
    """提取关键词（简单实现：高频词 + 去停用词）.

    Args:
        text: 文本内容
        top_n: 返回前 N 个关键词

    Returns:
        关键词集合
    """
    # 简单的中文停用词
    stopwords = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
        "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
        "自己", "这", "那", "为", "与", "及", "等", "但", "或", "而", "因为", "所以",
    }

    # 分词（简单按字符分割）
    words = []
    for char in text:
        if "\u4e00" <= char <= "\u9fff":  # 中文字符
            words.append(char)

    # 统计词频
    word_freq: Dict[str, int] = {}
    for word in words:
        if word not in stopwords and len(word) > 0:
            word_freq[word] = word_freq.get(word, 0) + 1

    # 返回高频词
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return {word for word, _ in sorted_words[:top_n]}


def jaccard_similarity(set1: set, set2: set) -> float:
    """计算 Jaccard 相似度.

    Args:
        set1: 集合 1
        set2: 集合 2

    Returns:
        Jaccard 相似度（0-1）
    """
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    return intersection / union


def _keyword_similarity(content1: str, content2: str) -> float:
    """基于关键词的相似度（降级方案）.

    Args:
        content1: 文本 1
        content2: 文本 2

    Returns:
        相似度（0-1）
    """
    keywords1 = _extract_keywords(content1)
    keywords2 = _extract_keywords(content2)
    return jaccard_similarity(keywords1, keywords2)


def _get_embedding_cache_path(memory_dir: Path, employee: str) -> Path:
    """获取 embedding 缓存文件路径."""
    cache_dir = memory_dir / ".embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 文件名安全化
    import re
    safe_name = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]", "_", employee)
    return cache_dir / f"{safe_name}.json"


def _load_embedding_cache(memory_dir: Path, employee: str) -> Dict[str, List[float]]:
    """加载 embedding 缓存.

    Returns:
        {entry_id: embedding} 字典
    """
    cache_path = _get_embedding_cache_path(memory_dir, employee)
    if not cache_path.exists():
        return {}

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data
    except Exception as e:
        logger.debug(f"加载 embedding 缓存失败: {e}")
        return {}


def _save_embedding_cache(
    memory_dir: Path,
    employee: str,
    cache: Dict[str, List[float]],
) -> None:
    """保存 embedding 缓存."""
    cache_path = _get_embedding_cache_path(memory_dir, employee)
    try:
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.debug(f"保存 embedding 缓存失败: {e}")


async def find_similar_memories(
    employee: str,
    content: str,
    category: str,
    threshold: float = 0.85,
    project_dir: Path | None = None,
    use_keyword_fallback: bool = True,
) -> List[Tuple[Dict[str, Any], float]]:
    """查找相似记忆.

    Args:
        employee: 员工名称
        content: 新记忆内容
        category: 记忆类别
        threshold: 相似度阈值（0-1）
        project_dir: 项目目录
        use_keyword_fallback: API 失败时是否使用关键词匹配降级

    Returns:
        List of (memory_dict, similarity_score) tuples，按相似度降序
    """
    from crew.memory import get_memory_store
    from crew.paths import resolve_project_dir

    # 初始化存储
    if project_dir is None:
        project_dir = resolve_project_dir(None)

    store = get_memory_store(project_dir=project_dir)
    memory_dir = store.memory_dir

    # 解析员工名
    employee = store._resolve_to_character_name(employee)

    # 查询该员工最近 100 条同类记忆
    recent_memories = store.query(
        employee=employee,
        category=category,
        limit=100,
        include_expired=False,
    )

    if not recent_memories:
        return []

    # 尝试使用 embedding 计算相似度
    new_embedding = await get_embedding(content)

    if new_embedding is not None:
        # 使用 embedding 方式
        similar = await _find_similar_with_embedding(
            new_embedding=new_embedding,
            recent_memories=recent_memories,
            threshold=threshold,
            memory_dir=memory_dir,
            employee=employee,
        )
        logger.info(f"使用 embedding 检测到 {len(similar)} 条相似记忆")
        return similar

    # 降级到关键词匹配
    if use_keyword_fallback:
        # 关键词匹配的阈值：如果用户指定了较低的阈值，则使用用户指定的；否则降低 0.15
        keyword_threshold = min(threshold, max(0.5, threshold - 0.15))
        similar = _find_similar_with_keywords(
            content=content,
            recent_memories=recent_memories,
            threshold=keyword_threshold,
        )
        logger.info(f"使用关键词匹配检测到 {len(similar)} 条相似记忆（阈值={keyword_threshold}）")
        return similar

    return []


async def _find_similar_with_embedding(
    new_embedding: List[float],
    recent_memories: List[Any],
    threshold: float,
    memory_dir: Path,
    employee: str,
) -> List[Tuple[Dict[str, Any], float]]:
    """使用 embedding 查找相似记忆."""
    # 加载缓存
    cache = _load_embedding_cache(memory_dir, employee)
    updated_cache = False

    similar: List[Tuple[Dict[str, Any], float]] = []

    for mem in recent_memories:
        mem_id = mem.id

        # 尝试从缓存读取
        mem_embedding = cache.get(mem_id)

        # 缓存未命中，计算并缓存
        if mem_embedding is None:
            mem_embedding = await get_embedding(mem.content)
            if mem_embedding is not None:
                cache[mem_id] = mem_embedding
                updated_cache = True

        # 计算相似度
        if mem_embedding is not None:
            similarity = cosine_similarity(new_embedding, mem_embedding)

            if similarity >= threshold:
                similar.append((mem.model_dump(), similarity))

    # 保存更新的缓存
    if updated_cache:
        _save_embedding_cache(memory_dir, employee, cache)

    # 按相似度降序排序，返回前 3 条
    similar.sort(key=lambda x: x[1], reverse=True)
    return similar[:3]


def _find_similar_with_keywords(
    content: str,
    recent_memories: List[Any],
    threshold: float,
) -> List[Tuple[Dict[str, Any], float]]:
    """使用关键词匹配查找相似记忆（降级方案）."""
    similar: List[Tuple[Dict[str, Any], float]] = []

    for mem in recent_memories:
        similarity = _keyword_similarity(content, mem.content)

        if similarity >= threshold:
            similar.append((mem.model_dump(), similarity))

    # 按相似度降序排序，返回前 3 条
    similar.sort(key=lambda x: x[1], reverse=True)
    return similar[:3]
