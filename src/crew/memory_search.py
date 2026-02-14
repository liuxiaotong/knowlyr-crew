"""语义记忆搜索 — 基于 embedding 的相关记忆检索（混合搜索 + 多后端）."""

from __future__ import annotations

import logging
import math
import sqlite3
import struct
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crew.memory import MemoryEntry

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算余弦相似度."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _pack_embedding(embedding: list[float]) -> bytes:
    """将 float 列表打包为 bytes."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _unpack_embedding(data: bytes) -> list[float]:
    """将 bytes 解包为 float 列表."""
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


def _keyword_score(query_tokens: set[str], content_tokens: set[str]) -> float:
    """关键词匹配分数：交集 / query token 数."""
    if not query_tokens:
        return 0.0
    overlap = query_tokens & content_tokens
    return len(overlap) / len(query_tokens)


class SemanticMemoryIndex:
    """基于 embedding 的语义记忆索引（混合搜索）.

    搜索策略：向量余弦相似度 + 关键词匹配，加权合并。
    存储: SQLite 文件（{memory_dir}/embeddings.db）
    Embedding: OpenAI > Gemini > TF-IDF 降级
    """

    # 混合搜索权重
    VEC_WEIGHT = 0.7
    KW_WEIGHT = 0.3

    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self._db_path = memory_dir / "embeddings.db"
        self._conn: sqlite3.Connection | None = None
        self._embedder: _Embedder | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    id TEXT PRIMARY KEY,
                    employee TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    content TEXT NOT NULL
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_employee ON memory_vectors(employee)
            """)
            self._conn.commit()
        return self._conn

    def _get_embedder(self) -> "_Embedder":
        if self._embedder is None:
            self._embedder = _create_embedder()
        return self._embedder

    def index(self, entry: "MemoryEntry") -> bool:
        """为一条记忆计算 embedding 并存储.

        Returns:
            True 如果成功索引，False 如果 embedding 失败.
        """
        embedder = self._get_embedder()
        embedding = embedder.embed(entry.content)
        if embedding is None:
            return False

        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO memory_vectors (id, employee, embedding, content) VALUES (?, ?, ?, ?)",
            (entry.id, entry.employee, _pack_embedding(embedding), entry.content),
        )
        conn.commit()
        return True

    def search(self, employee: str, query: str, limit: int = 10) -> list[tuple[str, str, float]]:
        """混合搜索：向量相似度 + 关键词匹配，加权合并.

        Returns:
            [(id, content, score), ...] 按综合分数降序排列.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, embedding, content FROM memory_vectors WHERE employee = ?",
            (employee,),
        ).fetchall()

        if not rows:
            return []

        # 向量搜索
        embedder = self._get_embedder()
        query_embedding = embedder.embed(query)

        # 关键词搜索
        query_tokens = set(_tokenize(query))

        scored: list[tuple[str, str, float]] = []
        for row_id, emb_bytes, content in rows:
            # 向量分数
            vec_score = 0.0
            if query_embedding is not None:
                emb = _unpack_embedding(emb_bytes)
                vec_score = _cosine_similarity(query_embedding, emb)

            # 关键词分数
            content_tokens = set(_tokenize(content))
            kw_score = _keyword_score(query_tokens, content_tokens)

            # 加权合并
            if query_embedding is not None:
                final_score = self.VEC_WEIGHT * vec_score + self.KW_WEIGHT * kw_score
            else:
                # embedding 失败时完全依赖关键词
                final_score = kw_score

            scored.append((row_id, content, final_score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:limit]

    def reindex(self, employee: str, entries: list["MemoryEntry"]) -> int:
        """全量重建某员工的索引.

        Returns:
            成功索引的条目数.
        """
        conn = self._get_conn()
        conn.execute("DELETE FROM memory_vectors WHERE employee = ?", (employee,))
        conn.commit()

        count = 0
        for entry in entries:
            if self.index(entry):
                count += 1
        return count

    def has_index(self, employee: str) -> bool:
        """检查某员工是否有索引数据."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM memory_vectors WHERE employee = ?",
            (employee,),
        ).fetchone()
        return row[0] > 0 if row else False

    def close(self) -> None:
        """关闭数据库连接."""
        if self._conn:
            self._conn.close()
            self._conn = None


# ── Embedding 抽象 ──


class _Embedder:
    """Embedding 抽象接口."""

    def embed(self, text: str) -> list[float] | None:
        raise NotImplementedError


class _OpenAIEmbedder(_Embedder):
    """OpenAI text-embedding-3-small."""

    def __init__(self):
        import openai
        self._client = openai.OpenAI()

    def embed(self, text: str) -> list[float] | None:
        try:
            resp = self._client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                timeout=5.0,
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.warning("OpenAI embedding 失败: %s", e)
            return None


class _GeminiEmbedder(_Embedder):
    """Google Gemini text-embedding-004."""

    def __init__(self):
        import google.generativeai as genai
        genai.configure()

    def embed(self, text: str) -> list[float] | None:
        try:
            import google.generativeai as genai
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    genai.embed_content,
                    model="models/text-embedding-004",
                    content=text,
                )
                result = future.result(timeout=5.0)
            return result["embedding"]
        except Exception as e:
            logger.warning("Gemini embedding 失败: %s", e)
            return None


class _TfIdfEmbedder(_Embedder):
    """TF-IDF 关键词匹配降级方案（无外部依赖）.

    使用简单的词频向量 + 余弦相似度，维度固定为哈希桶数。
    """

    DIM = 256

    def embed(self, text: str) -> list[float] | None:
        tokens = _tokenize(text)
        if not tokens:
            return [0.0] * self.DIM

        vec = [0.0] * self.DIM
        for token in tokens:
            idx = hash(token) % self.DIM
            vec[idx] += 1.0

        # L2 归一化
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


def _tokenize(text: str) -> list[str]:
    """简单分词：中文按字，英文按空格 + 小写."""
    import re

    tokens: list[str] = []
    # 中文字符
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            tokens.append(ch)
    # 英文单词
    words = re.findall(r"[a-zA-Z]+", text)
    tokens.extend(w.lower() for w in words)
    return tokens


def _create_embedder() -> _Embedder:
    """创建 embedder：优先 OpenAI → Gemini → TF-IDF."""
    import os

    if os.environ.get("OPENAI_API_KEY"):
        try:
            embedder = _OpenAIEmbedder()
            logger.info("使用 OpenAI embedding")
            return embedder
        except Exception:
            pass

    if os.environ.get("GOOGLE_API_KEY"):
        try:
            embedder = _GeminiEmbedder()
            logger.info("使用 Gemini embedding")
            return embedder
        except Exception:
            pass

    logger.info("使用 TF-IDF 降级 embedding")
    return _TfIdfEmbedder()
