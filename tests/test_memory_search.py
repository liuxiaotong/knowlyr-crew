"""测试语义记忆搜索."""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crew.memory import MemoryEntry, MemoryStore
from crew.memory_search import (
    SemanticMemoryIndex,
    _TfIdfEmbedder,
    _cosine_similarity,
    _create_embedder,
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
        # 关闭后 _conn 为 None
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
        # 应该包含关于主题的记忆
        assert "暗色主题" in result

    def test_format_for_prompt_without_query(self, tmp_path):
        store = MemoryStore(memory_dir=tmp_path)
        store.add("bot", "finding", "some finding")

        # 不传 query，使用原始逻辑
        result = store.format_for_prompt("bot")
        assert "finding" in result.lower() or "发现" in result

    def test_format_for_prompt_no_index_fallback(self, tmp_path):
        store = MemoryStore(memory_dir=tmp_path)
        store.add("bot", "finding", "some finding")

        # 传 query 但无索引，降级到原始逻辑
        result = store.format_for_prompt("bot", query="anything")
        assert "发现" in result  # 原始格式包含类别标签


# ── _create_embedder ──


class TestCreateEmbedder:
    """Embedder 创建."""

    def test_no_api_key(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False):
            embedder = _create_embedder()
        assert isinstance(embedder, _TfIdfEmbedder)

    def test_with_api_key_but_import_fails(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False), \
             patch("crew.memory_search._OpenAIEmbedder", side_effect=Exception("no openai")):
            embedder = _create_embedder()
        assert isinstance(embedder, _TfIdfEmbedder)
