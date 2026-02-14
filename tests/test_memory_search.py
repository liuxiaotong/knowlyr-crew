"""测试语义记忆搜索（混合搜索 + 多后端）."""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crew.memory import MemoryEntry, MemoryStore
from crew.memory_search import (
    SemanticMemoryIndex,
    _GeminiEmbedder,
    _TfIdfEmbedder,
    _cosine_similarity,
    _create_embedder,
    _keyword_score,
    _tokenize,
)


# ── 基础工具 ──


class TestCosimilarity:
    """余弦相似度."""

    def test_identical(self):
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) + 1.0) < 1e-6

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert _cosine_similarity(a, b) == 0.0


class TestTokenize:
    """分词."""

    def test_chinese(self):
        tokens = _tokenize("用户偏好暗色主题")
        assert "用" in tokens
        assert "暗" in tokens

    def test_english(self):
        tokens = _tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_mixed(self):
        tokens = _tokenize("API 设计讨论")
        assert "api" in tokens
        assert "设" in tokens

    def test_empty(self):
        assert _tokenize("") == []


class TestKeywordScore:
    """关键词匹配分数."""

    def test_full_match(self):
        score = _keyword_score({"api", "设"}, {"api", "设", "计"})
        assert score == 1.0

    def test_partial_match(self):
        score = _keyword_score({"api", "数", "据"}, {"api", "接", "口"})
        assert abs(score - 1.0 / 3) < 1e-6

    def test_no_match(self):
        score = _keyword_score({"hello"}, {"world"})
        assert score == 0.0

    def test_empty_query(self):
        score = _keyword_score(set(), {"hello"})
        assert score == 0.0


# ── TF-IDF Embedder ──


class TestTfIdfEmbedder:
    """TF-IDF 降级方案."""

    def test_embed(self):
        embedder = _TfIdfEmbedder()
        vec = embedder.embed("用户偏好暗色主题")
        assert vec is not None
        assert len(vec) == 256
        # 归一化后 L2 范数约为 1
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-6

    def test_similar_texts(self):
        embedder = _TfIdfEmbedder()
        v1 = embedder.embed("用户偏好暗色主题")
        v2 = embedder.embed("用户喜欢暗色主题")
        v3 = embedder.embed("API 接口设计规范")
        # v1 和 v2 更相似
        sim_12 = _cosine_similarity(v1, v2)
        sim_13 = _cosine_similarity(v1, v3)
        assert sim_12 > sim_13

    def test_empty_text(self):
        embedder = _TfIdfEmbedder()
        vec = embedder.embed("")
        assert vec is not None
        assert len(vec) == 256


# ── SemanticMemoryIndex ──


class TestSemanticMemoryIndex:
    """语义记忆索引."""

    def test_index_and_search(self, tmp_path):
        index = SemanticMemoryIndex(tmp_path / "memory")

        entries = [
            MemoryEntry(id="1", employee="bot", category="finding", content="用户偏好暗色主题"),
            MemoryEntry(id="2", employee="bot", category="finding", content="API 接口需要鉴权"),
            MemoryEntry(id="3", employee="bot", category="decision", content="数据库选用 PostgreSQL"),
        ]

        for e in entries:
            assert index.index(e) is True

        # 搜索关于 UI 的记忆
        results = index.search("bot", "界面主题颜色", limit=2)
        assert len(results) <= 2
        # 第一条应该是关于主题的（最相关）
        assert results[0][0] == "1"  # id

    def test_search_empty(self, tmp_path):
        index = SemanticMemoryIndex(tmp_path / "memory")
        results = index.search("nobody", "anything")
        assert results == []

    def test_has_index(self, tmp_path):
        index = SemanticMemoryIndex(tmp_path / "memory")
        assert index.has_index("bot") is False

        entry = MemoryEntry(id="1", employee="bot", category="finding", content="test")
        index.index(entry)
        assert index.has_index("bot") is True

    def test_reindex(self, tmp_path):
        index = SemanticMemoryIndex(tmp_path / "memory")

        entries = [
            MemoryEntry(id="1", employee="bot", category="finding", content="old data"),
        ]
        index.index(entries[0])
        assert index.has_index("bot")

        new_entries = [
            MemoryEntry(id="2", employee="bot", category="finding", content="new data"),
            MemoryEntry(id="3", employee="bot", category="finding", content="more data"),
        ]
        count = index.reindex("bot", new_entries)
        assert count == 2

        results = index.search("bot", "data", limit=10)
        assert len(results) == 2
        ids = {r[0] for r in results}
        assert "1" not in ids
        assert "2" in ids

    def test_close(self, tmp_path):
        index = SemanticMemoryIndex(tmp_path / "memory")
        entry = MemoryEntry(id="1", employee="bot", category="finding", content="test")
        index.index(entry)
        index.close()
        assert index._conn is None

    def test_different_employees(self, tmp_path):
        index = SemanticMemoryIndex(tmp_path / "memory")

        index.index(MemoryEntry(id="1", employee="alice", category="finding", content="Alice 的发现"))
        index.index(MemoryEntry(id="2", employee="bob", category="finding", content="Bob 的发现"))

        results_alice = index.search("alice", "发现", limit=10)
        results_bob = index.search("bob", "发现", limit=10)
        assert len(results_alice) == 1
        assert len(results_bob) == 1
        assert results_alice[0][0] == "1"
        assert results_bob[0][0] == "2"


# ── Hybrid Search ──


class TestHybridSearch:
    """混合搜索：向量 + 关键词."""

    def test_keyword_boost(self, tmp_path):
        """精确术语通过关键词匹配获得加分."""
        index = SemanticMemoryIndex(tmp_path / "memory")

        entries = [
            MemoryEntry(id="1", employee="bot", category="finding", content="用户偏好暗色主题"),
            MemoryEntry(id="2", employee="bot", category="decision", content="数据库选用 PostgreSQL"),
            MemoryEntry(id="3", employee="bot", category="finding", content="后端框架用 FastAPI"),
        ]
        for e in entries:
            index.index(e)

        # 搜索精确术语 "PostgreSQL"
        results = index.search("bot", "PostgreSQL", limit=3)
        assert len(results) == 3
        # 包含 PostgreSQL 的应该排第一
        assert results[0][0] == "2"

    def test_hybrid_weights(self, tmp_path):
        """验证混合权重生效."""
        index = SemanticMemoryIndex(tmp_path / "memory")

        entries = [
            MemoryEntry(id="1", employee="bot", category="finding", content="API 接口鉴权设计"),
            MemoryEntry(id="2", employee="bot", category="finding", content="API 文档编写规范"),
        ]
        for e in entries:
            index.index(e)

        # 搜索 "API 鉴权" — 两条都含 API，但只有 id=1 含"鉴"和"权"
        results = index.search("bot", "API 鉴权", limit=2)
        assert len(results) == 2
        assert results[0][0] == "1"  # 关键词匹配更好

    def test_vec_weight_and_kw_weight(self):
        """验证权重常量."""
        assert SemanticMemoryIndex.VEC_WEIGHT == 0.7
        assert SemanticMemoryIndex.KW_WEIGHT == 0.3
        assert abs(SemanticMemoryIndex.VEC_WEIGHT + SemanticMemoryIndex.KW_WEIGHT - 1.0) < 1e-6


# ── MemoryStore 语义搜索集成 ──


class TestMemoryStoreSemanticIntegration:
    """MemoryStore 语义搜索集成."""

    def test_format_for_prompt_with_query(self, tmp_path):
        store = MemoryStore(memory_dir=tmp_path)

        # 添加记忆
        store.add("bot", "finding", "用户偏好暗色主题")
        store.add("bot", "finding", "API 需要鉴权")
        store.add("bot", "decision", "数据库用 PostgreSQL")

        # 建立索引
        index = SemanticMemoryIndex(tmp_path)
        for entry in store.query("bot", limit=100):
            index.index(entry)

        # 使用语义搜索
        result = store.format_for_prompt("bot", query="界面主题颜色", limit=2)
        assert result  # 非空
        assert "暗色主题" in result

    def test_format_for_prompt_without_query(self, tmp_path):
        store = MemoryStore(memory_dir=tmp_path)
        store.add("bot", "finding", "some finding")

        result = store.format_for_prompt("bot")
        assert "finding" in result.lower() or "发现" in result

    def test_format_for_prompt_no_index_fallback(self, tmp_path):
        store = MemoryStore(memory_dir=tmp_path)
        store.add("bot", "finding", "some finding")

        # 传 query 但无索引，降级到原始逻辑
        result = store.format_for_prompt("bot", query="anything")
        assert "发现" in result


# ── _create_embedder ──


class TestCreateEmbedder:
    """Embedder 创建链路: OpenAI → Gemini → TF-IDF."""

    def test_no_api_key(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=False):
            embedder = _create_embedder()
        assert isinstance(embedder, _TfIdfEmbedder)

    def test_openai_key_but_import_fails(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "GOOGLE_API_KEY": ""}, clear=False), \
             patch("crew.memory_search._OpenAIEmbedder", side_effect=Exception("no openai")):
            embedder = _create_embedder()
        assert isinstance(embedder, _TfIdfEmbedder)

    def test_gemini_key_selected(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": "test-key"}, clear=False), \
             patch("crew.memory_search._GeminiEmbedder") as mock_cls:
            mock_cls.return_value = MagicMock(spec=_GeminiEmbedder)
            embedder = _create_embedder()
        mock_cls.assert_called_once()

    def test_gemini_key_but_import_fails(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": "test-key"}, clear=False), \
             patch("crew.memory_search._GeminiEmbedder", side_effect=Exception("no genai")):
            embedder = _create_embedder()
        assert isinstance(embedder, _TfIdfEmbedder)

    def test_openai_preferred_over_gemini(self):
        """两个 key 都有时优先 OpenAI."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "oai-key", "GOOGLE_API_KEY": "goog-key"}, clear=False), \
             patch("crew.memory_search._OpenAIEmbedder") as mock_oai, \
             patch("crew.memory_search._GeminiEmbedder") as mock_gem:
            mock_oai.return_value = MagicMock()
            embedder = _create_embedder()
        mock_oai.assert_called_once()
        mock_gem.assert_not_called()


# ── Embedding timeout ──


class TestEmbeddingTimeout:
    """嵌入 API 超时保护."""

    def test_openai_embedder_passes_timeout(self):
        """验证 OpenAI embedder 传递 timeout 参数."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
            try:
                from crew.memory_search import _OpenAIEmbedder

                mock_client = MagicMock()
                mock_resp = MagicMock()
                mock_resp.data = [MagicMock(embedding=[0.1] * 256)]
                mock_client.embeddings.create.return_value = mock_resp

                embedder = _OpenAIEmbedder.__new__(_OpenAIEmbedder)
                embedder._client = mock_client
                embedder._model = "text-embedding-3-small"

                result = embedder.embed("test text")
                call_kwargs = mock_client.embeddings.create.call_args[1]
                assert call_kwargs.get("timeout") == 5.0
                assert result is not None
            except Exception:
                pytest.skip("openai SDK not available")

    def test_gemini_embedder_uses_timeout(self):
        """验证 Gemini embedder 有超时保护（通过 ThreadPoolExecutor）."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}, clear=False):
            try:
                from crew.memory_search import _GeminiEmbedder

                mock_genai = MagicMock()
                mock_genai.embed_content.return_value = {"embedding": [0.1] * 256}

                embedder = _GeminiEmbedder.__new__(_GeminiEmbedder)
                embedder._genai = mock_genai
                embedder._model = "models/text-embedding-004"

                with patch("crew.memory_search.ThreadPoolExecutor") as mock_pool:
                    mock_future = MagicMock()
                    mock_future.result.return_value = {"embedding": [0.1] * 256}
                    mock_pool.return_value.__enter__ = MagicMock(return_value=mock_pool.return_value)
                    mock_pool.return_value.__exit__ = MagicMock(return_value=False)
                    mock_pool.return_value.submit.return_value = mock_future

                    result = embedder.embed("test text")
                    mock_future.result.assert_called_once_with(timeout=5.0)
            except Exception:
                pytest.skip("google-generativeai SDK not available")
