"""LLM 执行器 — 调用 Anthropic Claude API 执行 prompt."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


def _get_anthropic():
    """Lazy import anthropic SDK."""
    try:
        import anthropic

        return anthropic
    except ImportError:
        return None


@dataclass
class ExecutionResult:
    """LLM 执行结果."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str


def execute_prompt(
    *,
    system_prompt: str,
    user_message: str = "请开始执行上述任务。",
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    temperature: float | None = None,
    max_tokens: int | None = None,
    stream: bool = True,
    on_chunk: Callable[[str], None] | None = None,
) -> ExecutionResult:
    """调用 Anthropic Claude API 执行 prompt.

    Args:
        system_prompt: 完整 system prompt（engine.prompt() 输出）.
        user_message: 用户消息.
        api_key: Anthropic API key.
        model: 模型标识符.
        temperature: 采样温度（None 使用 API 默认值）.
        max_tokens: 最大输出 token 数.
        stream: 是否流式输出.
        on_chunk: 流式模式下的文本块回调.

    Returns:
        ExecutionResult.

    Raises:
        ImportError: anthropic SDK 未安装.
    """
    anthropic = _get_anthropic()
    if anthropic is None:
        raise ImportError("anthropic SDK 未安装。请运行: pip install knowlyr-crew[execute]")

    client = anthropic.Anthropic(api_key=api_key)
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens or 4096,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    if stream and on_chunk:
        collected: list[str] = []
        with client.messages.stream(**kwargs) as resp:
            for text in resp.text_stream:
                collected.append(text)
                on_chunk(text)
            final = resp.get_final_message()
        return ExecutionResult(
            content="".join(collected),
            model=final.model,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
            stop_reason=final.stop_reason,
        )

    resp = client.messages.create(**kwargs)
    content = resp.content[0].text if resp.content else ""
    return ExecutionResult(
        content=content,
        model=resp.model,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        stop_reason=resp.stop_reason,
    )
