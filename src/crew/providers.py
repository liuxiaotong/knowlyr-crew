"""LLM Provider 检测与 API Key 解析 — 支持 Anthropic / OpenAI / DeepSeek / Moonshot."""

from __future__ import annotations

import os
from enum import Enum


class Provider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    MOONSHOT = "moonshot"


# ── 模型前缀 → Provider 映射 ──

_MODEL_PREFIXES: list[tuple[str, Provider]] = [
    ("claude-", Provider.ANTHROPIC),
    ("gpt-", Provider.OPENAI),
    ("o1-", Provider.OPENAI),
    ("o3-", Provider.OPENAI),
    ("o4-", Provider.OPENAI),
    ("chatgpt-", Provider.OPENAI),
    ("deepseek-", Provider.DEEPSEEK),
    ("moonshot-", Provider.MOONSHOT),
]

DEFAULT_MODELS: dict[Provider, str] = {
    Provider.ANTHROPIC: "claude-sonnet-4-20250514",
    Provider.OPENAI: "gpt-4o",
    Provider.DEEPSEEK: "deepseek-chat",
    Provider.MOONSHOT: "moonshot-v1-128k",
}

API_KEY_ENV_VARS: dict[Provider, str] = {
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.DEEPSEEK: "DEEPSEEK_API_KEY",
    Provider.MOONSHOT: "MOONSHOT_API_KEY",
}

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"


def detect_provider(model: str) -> Provider:
    """根据模型名前缀检测 LLM 提供商.

    Raises:
        ValueError: 无法识别的模型前缀.
    """
    model_lower = model.lower()
    for prefix, provider in _MODEL_PREFIXES:
        if model_lower.startswith(prefix):
            return provider
    supported = ", ".join(p for p, _ in _MODEL_PREFIXES)
    raise ValueError(f"无法识别模型 '{model}' 的提供商。支持的前缀: {supported}")


def resolve_api_key(provider: Provider, api_key: str | None = None) -> str:
    """解析 API key: 显式传参 > 环境变量.

    Raises:
        ValueError: 未找到 API key.
    """
    if api_key:
        return api_key
    env_var = API_KEY_ENV_VARS[provider]
    key = os.environ.get(env_var, "")
    if not key:
        raise ValueError(
            f"{provider.value} API key 未设置。"
            f"请设置环境变量 {env_var} 或通过参数传递 api_key。"
        )
    return key
