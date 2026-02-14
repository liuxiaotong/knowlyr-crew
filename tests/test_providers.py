"""测试 LLM Provider 检测与 API Key 解析."""

import os
from unittest.mock import patch

import pytest

from crew.providers import (
    API_KEY_ENV_VARS,
    DEFAULT_MODELS,
    DEEPSEEK_BASE_URL,
    MOONSHOT_BASE_URL,
    ZHIPU_BASE_URL,
    QWEN_BASE_URL,
    Provider,
    detect_provider,
    resolve_api_key,
)


class TestDetectProvider:
    """测试 detect_provider."""

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("claude-sonnet-4-20250514", Provider.ANTHROPIC),
            ("claude-opus-4-6", Provider.ANTHROPIC),
            ("claude-haiku-4-5-20251001", Provider.ANTHROPIC),
            ("Claude-Sonnet-4-20250514", Provider.ANTHROPIC),  # 大小写不敏感
        ],
    )
    def test_anthropic(self, model, expected):
        assert detect_provider(model) == expected

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gpt-4o", Provider.OPENAI),
            ("gpt-4-turbo", Provider.OPENAI),
            ("gpt-3.5-turbo", Provider.OPENAI),
            ("o1-preview", Provider.OPENAI),
            ("o1-mini", Provider.OPENAI),
            ("o3-mini", Provider.OPENAI),
            ("o4-mini", Provider.OPENAI),
            ("chatgpt-4o-latest", Provider.OPENAI),
        ],
    )
    def test_openai(self, model, expected):
        assert detect_provider(model) == expected

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("deepseek-chat", Provider.DEEPSEEK),
            ("deepseek-coder", Provider.DEEPSEEK),
            ("deepseek-reasoner", Provider.DEEPSEEK),
        ],
    )
    def test_deepseek(self, model, expected):
        assert detect_provider(model) == expected

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("moonshot-v1-8k", Provider.MOONSHOT),
            ("moonshot-v1-32k", Provider.MOONSHOT),
            ("moonshot-v1-128k", Provider.MOONSHOT),
        ],
    )
    def test_moonshot(self, model, expected):
        assert detect_provider(model) == expected

    def test_unknown_model(self):
        with pytest.raises(ValueError, match="无法识别模型"):
            detect_provider("llama-3-70b")

    def test_empty_model(self):
        with pytest.raises(ValueError, match="无效的模型名"):
            detect_provider("")

    def test_none_model(self):
        with pytest.raises(ValueError, match="无效的模型名"):
            detect_provider(None)

    def test_non_string_model(self):
        with pytest.raises(ValueError, match="无效的模型名"):
            detect_provider(123)


class TestResolveApiKey:
    """测试 resolve_api_key."""

    def test_explicit_key_takes_precedence(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            assert resolve_api_key(Provider.ANTHROPIC, "explicit-key") == "explicit-key"

    def test_from_env_anthropic(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-xxx"}):
            assert resolve_api_key(Provider.ANTHROPIC) == "sk-ant-xxx"

    def test_from_env_openai(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai-xxx"}):
            assert resolve_api_key(Provider.OPENAI) == "sk-openai-xxx"

    def test_from_env_deepseek(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-ds-xxx"}):
            assert resolve_api_key(Provider.DEEPSEEK) == "sk-ds-xxx"

    def test_from_env_moonshot(self):
        with patch.dict(os.environ, {"MOONSHOT_API_KEY": "sk-moon-xxx"}):
            assert resolve_api_key(Provider.MOONSHOT) == "sk-moon-xxx"

    def test_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key 未设置"):
                resolve_api_key(Provider.OPENAI)

    def test_missing_key_message_includes_env_var(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
                resolve_api_key(Provider.DEEPSEEK)


class TestConstants:
    """测试常量定义."""

    def test_all_providers_have_default_model(self):
        for provider in Provider:
            assert provider in DEFAULT_MODELS

    def test_all_providers_have_env_var(self):
        for provider in Provider:
            assert provider in API_KEY_ENV_VARS

    def test_deepseek_base_url(self):
        assert "deepseek" in DEEPSEEK_BASE_URL

    def test_moonshot_base_url(self):
        assert "moonshot" in MOONSHOT_BASE_URL

    def test_zhipu_base_url(self):
        assert "bigmodel" in ZHIPU_BASE_URL

    def test_qwen_base_url(self):
        assert "dashscope" in QWEN_BASE_URL


class TestGeminiProvider:
    @pytest.mark.parametrize("model,expected", [
        ("gemini-2.0-flash", Provider.GEMINI),
        ("gemini-1.5-pro", Provider.GEMINI),
    ])
    def test_gemini(self, model, expected):
        assert detect_provider(model) == expected


class TestZhipuProvider:
    @pytest.mark.parametrize("model,expected", [
        ("glm-4-flash", Provider.ZHIPU),
        ("glm-4", Provider.ZHIPU),
    ])
    def test_zhipu(self, model, expected):
        assert detect_provider(model) == expected

    def test_zhipu_api_key(self):
        with patch.dict(os.environ, {"ZHIPUAI_API_KEY": "sk-zhipu"}):
            assert resolve_api_key(Provider.ZHIPU) == "sk-zhipu"


class TestQwenProvider:
    @pytest.mark.parametrize("model,expected", [
        ("qwen-turbo", Provider.QWEN),
        ("qwen-max", Provider.QWEN),
        ("qwen-plus", Provider.QWEN),
    ])
    def test_qwen(self, model, expected):
        assert detect_provider(model) == expected

    def test_qwen_api_key(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-dash"}):
            assert resolve_api_key(Provider.QWEN) == "sk-dash"
