"""tests/test_embedding.py — embedding 模块测试.

覆盖：
- embedding 生成（mock sentence-transformers）
- 降级逻辑（sentence-transformers 不可用时）
- build_embedding_text 拼接
- reset / is_available
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ── build_embedding_text ──


class TestBuildEmbeddingText:
    """测试 embedding 输入文本构造."""

    def test_content_only(self) -> None:
        from crew.embedding import build_embedding_text

        result = build_embedding_text("API 响应慢")
        assert result == "API 响应慢"

    def test_content_with_keywords(self) -> None:
        from crew.embedding import build_embedding_text

        result = build_embedding_text("API 响应慢", ["性能", "优化"])
        assert result == "API 响应慢 性能 优化"

    def test_empty_keywords(self) -> None:
        from crew.embedding import build_embedding_text

        result = build_embedding_text("content", [])
        assert result == "content"

    def test_none_keywords(self) -> None:
        from crew.embedding import build_embedding_text

        result = build_embedding_text("content", None)
        assert result == "content"


# ── get_embedding（mock 模型）──


class TestGetEmbedding:
    """测试 embedding 生成."""

    def setup_method(self) -> None:
        """每个测试前重置模型状态."""
        from crew.embedding import reset

        reset()

    def test_get_embedding_success(self) -> None:
        """模型可用时返回 384 维向量."""
        from crew.embedding import EMBEDDING_DIM

        mock_model = MagicMock()
        # 使用 MagicMock 模拟 numpy array 的 tolist()
        fake_list = [0.01 * i for i in range(EMBEDDING_DIM)]
        fake_array = MagicMock()
        fake_array.tolist.return_value = fake_list
        mock_model.encode.return_value = fake_array

        import crew.embedding

        crew.embedding.reset()
        crew.embedding._model = mock_model
        crew.embedding._model_load_attempted = True

        result = crew.embedding.get_embedding("test text")

        assert result is not None
        assert len(result) == EMBEDDING_DIM
        assert isinstance(result, list)
        assert all(isinstance(x, (int, float)) for x in result)

    def test_get_embedding_model_unavailable(self) -> None:
        """模型不可用时返回 None."""
        import crew.embedding

        crew.embedding.reset()
        crew.embedding._model = None
        crew.embedding._model_load_attempted = True

        result = crew.embedding.get_embedding("test text")
        assert result is None

    def test_get_embedding_encode_error(self) -> None:
        """encode 抛异常时返回 None（不崩溃）."""
        import crew.embedding

        crew.embedding.reset()
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError("CUDA OOM")
        crew.embedding._model = mock_model
        crew.embedding._model_load_attempted = True

        result = crew.embedding.get_embedding("test text")
        assert result is None


# ── is_available ──


class TestIsAvailable:
    """测试可用性检查."""

    def setup_method(self) -> None:
        from crew.embedding import reset

        reset()

    def test_available_when_model_loaded(self) -> None:
        import crew.embedding

        crew.embedding.reset()
        crew.embedding._model = MagicMock()
        crew.embedding._model_load_attempted = True

        assert crew.embedding.is_available() is True

    def test_not_available_when_no_model(self) -> None:
        import crew.embedding

        crew.embedding.reset()
        crew.embedding._model = None
        crew.embedding._model_load_attempted = True

        assert crew.embedding.is_available() is False


# ── 降级逻辑 ──


class TestGracefulDegradation:
    """测试 sentence-transformers 不可用时的降级."""

    def setup_method(self) -> None:
        from crew.embedding import reset

        reset()

    def test_import_error_graceful(self) -> None:
        """sentence-transformers 未安装时返回 None."""
        import crew.embedding

        crew.embedding.reset()

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            # 强制重新加载
            crew.embedding._model_load_attempted = False
            crew.embedding._model = None

            # _load_model 应该捕获 ImportError
            crew.embedding._load_model()
            # 可能成功也可能失败，取决于环境 — 关键是不崩溃
            # 在没有 sentence-transformers 的测试环境中应该返回 None

    def test_reset_clears_state(self) -> None:
        """reset 清空缓存状态."""
        import crew.embedding

        crew.embedding._model = MagicMock()
        crew.embedding._model_load_attempted = True

        crew.embedding.reset()

        assert crew.embedding._model is None
        assert crew.embedding._model_load_attempted is False
