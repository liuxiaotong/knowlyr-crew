"""语义记忆搜索 — 基于 embedding 的相关记忆检索（混合搜索 + 多后端）."""

from __future__ import annotations

import logging
import math
import sqlite3
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crew.memory import MemoryEntry

logger = logging.getLogger(__name__)

_EMBED_POOL = ThreadPoolExecutor(max_workers=1)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算余弦相似度."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
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
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_vectors (
                        id TEXT PRIMARY KEY,
                        employee TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        content TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_employee ON memory_vectors(employee)
                """)
                # Schema 升级：tags 列
                try:
                    conn.execute("ALTER TABLE memory_vectors ADD COLUMN tags TEXT DEFAULT ''")
                except sqlite3.OperationalError:
                    pass  # 列已存在
                conn.commit()
            except Exception:
                conn.close()
                raise
            self._conn = conn
        return self._conn

    def _get_embedder(self) -> _Embedder:
        if self._embedder is None:
            self._embedder = _create_embedder()
        return self._embedder

    def index(self, entry: MemoryEntry) -> bool:
        """为一条记忆计算 embedding 并存储.

        Returns:
            True 如果成功索引，False 如果 embedding 失败.
        """
        embedder = self._get_embedder()
        embedding = embedder.embed(entry.content)
        if embedding is None:
            return False

        tags_str = ",".join(entry.tags) if hasattr(entry, "tags") and entry.tags else ""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO memory_vectors (id, employee, embedding, content, tags) VALUES (?, ?, ?, ?, ?)",
            (entry.id, entry.employee, _pack_embedding(embedding), entry.content, tags_str),
        )
        conn.commit()
        return True

    def remove(self, entry_id: str) -> bool:
        """从索引中删除一条记忆.

        Returns:
            True 如果条目存在并被删除.
        """
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM memory_vectors WHERE id = ?", (entry_id,))
        conn.commit()
        return cursor.rowcount > 0

    def search(
        self, employee: str, query: str, limit: int = 10, timeout: float = 10.0
    ) -> list[tuple[str, str, float]]:
        """混合搜索：向量相似度 + 关键词匹配，加权合并.

        Args:
            timeout: embedding 超时秒数，超时后降级为纯关键词搜索.

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

        # 向量搜索（带超时保护）
        embedder = self._get_embedder()
        try:
            from concurrent.futures import TimeoutError as FuturesTimeout

            future = _EMBED_POOL.submit(embedder.embed, query)
            query_embedding = future.result(timeout=timeout)
        except (FuturesTimeout, Exception) as e:
            logger.warning("Embedding 超时或失败，降级为关键词搜索: %s", e)
            query_embedding = None

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

    def search_cross_employee(
        self,
        query: str,
        exclude_employee: str = "",
        limit: int = 10,
        timeout: float = 10.0,
    ) -> list[tuple[str, str, str, float]]:
        """跨员工搜索.

        Returns:
            [(id, employee, content, score), ...] 按分数降序.
        """
        conn = self._get_conn()
        if exclude_employee:
            rows = conn.execute(
                "SELECT id, employee, embedding, content FROM memory_vectors WHERE employee != ?",
                (exclude_employee,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, employee, embedding, content FROM memory_vectors",
            ).fetchall()

        if not rows:
            return []

        embedder = self._get_embedder()
        try:
            future = _EMBED_POOL.submit(embedder.embed, query)
            query_embedding = future.result(timeout=timeout)
        except Exception as e:
            logger.warning("跨员工搜索 embedding 失败: %s", e)
            query_embedding = None

        query_tokens = set(_tokenize(query))
        scored: list[tuple[str, str, str, float]] = []
        for row_id, emp, emb_bytes, content in rows:
            vec_score = 0.0
            if query_embedding is not None:
                emb = _unpack_embedding(emb_bytes)
                vec_score = _cosine_similarity(query_embedding, emb)

            content_tokens = set(_tokenize(content))
            kw_score = _keyword_score(query_tokens, content_tokens)

            if query_embedding is not None:
                final_score = self.VEC_WEIGHT * vec_score + self.KW_WEIGHT * kw_score
            else:
                final_score = kw_score

            scored.append((row_id, emp, content, final_score))

        scored.sort(key=lambda x: x[3], reverse=True)
        return scored[:limit]

    def reindex(self, employee: str, entries: list[MemoryEntry]) -> int:
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

    def __enter__(self) -> SemanticMemoryIndex:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_conn", None) is not None:
            logger.debug("SemanticMemoryIndex 未显式关闭，自动清理连接")
            self.close()

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
            if not resp.data:
                return None
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
            from concurrent.futures import ThreadPoolExecutor

            import google.generativeai as genai

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    genai.embed_content,
                    model="models/text-embedding-004",
                    content=text,
                )
                result = future.result(timeout=5.0)
            return result.get("embedding") if isinstance(result, dict) else None
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
        except Exception as e:
            logger.debug("OpenAI embedding 不可用: %s", e)

    if os.environ.get("GOOGLE_API_KEY"):
        try:
            embedder = _GeminiEmbedder()
            logger.info("使用 Gemini embedding")
            return embedder
        except Exception as e:
            logger.debug("Gemini embedding 不可用: %s", e)

    logger.info("使用 TF-IDF 降级 embedding")
    return _TfIdfEmbedder()
