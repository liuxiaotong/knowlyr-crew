"""LLM 执行器 — 支持 Anthropic / OpenAI / DeepSeek."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from collections.abc import AsyncIterator
from typing import Callable

from crew.providers import Provider, detect_provider, resolve_api_key, DEEPSEEK_BASE_URL

logger = logging.getLogger(__name__)


# ── Lazy SDK imports ──


def _get_anthropic():
    """Lazy import anthropic SDK."""
    try:
        import anthropic

        return anthropic
    except ImportError:
        return None


def _get_openai():
    """Lazy import openai SDK."""
    try:
        import openai

        return openai
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


# ── Anthropic 实现 ──


def _anthropic_execute(
    system_prompt: str,
    user_message: str,
    api_key: str,
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    stream: bool,
    on_chunk: Callable[[str], None] | None,
) -> ExecutionResult:
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


async def _anthropic_aexecute(
    system_prompt: str,
    user_message: str,
    api_key: str,
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    stream: bool,
) -> ExecutionResult | AsyncIterator[str]:
    anthropic = _get_anthropic()
    if anthropic is None:
        raise ImportError("anthropic SDK 未安装。请运行: pip install knowlyr-crew[execute]")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens or 4096,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    if stream:
        async def _stream() -> AsyncIterator[str]:
            collected: list[str] = []
            async with client.messages.stream(**kwargs) as resp:
                async for text in resp.text_stream:
                    collected.append(text)
                    yield text
                final = resp.get_final_message()
            _stream.result = ExecutionResult(  # type: ignore[attr-defined]
                content="".join(collected),
                model=final.model,
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
                stop_reason=final.stop_reason,
            )
        return _stream()

    resp = await client.messages.create(**kwargs)
    content = resp.content[0].text if resp.content else ""
    return ExecutionResult(
        content=content,
        model=resp.model,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        stop_reason=resp.stop_reason,
    )


# ── OpenAI 兼容实现（OpenAI + DeepSeek）──


def _openai_execute(
    system_prompt: str,
    user_message: str,
    api_key: str,
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    stream: bool,
    on_chunk: Callable[[str], None] | None,
    base_url: str | None = None,
) -> ExecutionResult:
    openai = _get_openai()
    if openai is None:
        raise ImportError("openai SDK 未安装。请运行: pip install knowlyr-crew[openai]")

    client_kwargs: dict = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    kwargs: dict = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if stream and on_chunk:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        collected: list[str] = []
        input_tokens = 0
        output_tokens = 0
        resp = client.chat.completions.create(**kwargs)
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                collected.append(text)
                on_chunk(text)
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0
        return ExecutionResult(
            content="".join(collected),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason="stop",
        )

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0] if resp.choices else None
    content = choice.message.content if choice and choice.message else ""
    return ExecutionResult(
        content=content or "",
        model=resp.model or model,
        input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
        output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        stop_reason=choice.finish_reason if choice else "unknown",
    )


async def _openai_aexecute(
    system_prompt: str,
    user_message: str,
    api_key: str,
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    stream: bool,
    base_url: str | None = None,
) -> ExecutionResult | AsyncIterator[str]:
    openai = _get_openai()
    if openai is None:
        raise ImportError("openai SDK 未安装。请运行: pip install knowlyr-crew[openai]")

    client_kwargs: dict = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.AsyncOpenAI(**client_kwargs)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    kwargs: dict = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if stream:
        async def _stream() -> AsyncIterator[str]:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
            collected: list[str] = []
            input_tokens = 0
            output_tokens = 0
            resp = await client.chat.completions.create(**kwargs)
            async for chunk in resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    collected.append(text)
                    yield text
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0
            _stream.result = ExecutionResult(  # type: ignore[attr-defined]
                content="".join(collected),
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                stop_reason="stop",
            )
        return _stream()

    resp = await client.chat.completions.create(**kwargs)
    choice = resp.choices[0] if resp.choices else None
    content = choice.message.content if choice and choice.message else ""
    return ExecutionResult(
        content=content or "",
        model=resp.model or model,
        input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
        output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        stop_reason=choice.finish_reason if choice else "unknown",
    )


# ── 公开 API ──


def execute_prompt(
    *,
    system_prompt: str,
    user_message: str = "请开始执行上述任务。",
    api_key: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    temperature: float | None = None,
    max_tokens: int | None = None,
    stream: bool = True,
    on_chunk: Callable[[str], None] | None = None,
) -> ExecutionResult:
    """调用 LLM API 执行 prompt（支持 Anthropic / OpenAI / DeepSeek）.

    Args:
        system_prompt: 完整 system prompt（engine.prompt() 输出）.
        user_message: 用户消息.
        api_key: API key（None 时自动从环境变量解析）.
        model: 模型标识符（根据前缀自动检测提供商）.
        temperature: 采样温度（None 使用 API 默认值）.
        max_tokens: 最大输出 token 数.
        stream: 是否流式输出.
        on_chunk: 流式模式下的文本块回调.

    Returns:
        ExecutionResult.
    """
    provider = detect_provider(model)
    resolved_key = resolve_api_key(provider, api_key)

    if provider == Provider.ANTHROPIC:
        return _anthropic_execute(
            system_prompt, user_message, resolved_key, model,
            temperature, max_tokens, stream, on_chunk,
        )
    elif provider == Provider.DEEPSEEK:
        return _openai_execute(
            system_prompt, user_message, resolved_key, model,
            temperature, max_tokens, stream, on_chunk,
            base_url=DEEPSEEK_BASE_URL,
        )
    else:
        return _openai_execute(
            system_prompt, user_message, resolved_key, model,
            temperature, max_tokens, stream, on_chunk,
        )


async def aexecute_prompt(
    *,
    system_prompt: str,
    user_message: str = "请开始执行上述任务。",
    api_key: str | None = None,
    model: str = "claude-sonnet-4-20250514",
    temperature: float | None = None,
    max_tokens: int | None = None,
    stream: bool = True,
) -> ExecutionResult | AsyncIterator[str]:
    """execute_prompt 的异步版本.

    当 stream=True 时返回 AsyncIterator[str]（调用者需用 async for 消费），
    完成后可通过 .result 属性获取 ExecutionResult。
    当 stream=False 时直接返回 ExecutionResult。
    """
    provider = detect_provider(model)
    resolved_key = resolve_api_key(provider, api_key)

    if provider == Provider.ANTHROPIC:
        return await _anthropic_aexecute(
            system_prompt, user_message, resolved_key, model,
            temperature, max_tokens, stream,
        )
    elif provider == Provider.DEEPSEEK:
        return await _openai_aexecute(
            system_prompt, user_message, resolved_key, model,
            temperature, max_tokens, stream,
            base_url=DEEPSEEK_BASE_URL,
        )
    else:
        return await _openai_aexecute(
            system_prompt, user_message, resolved_key, model,
            temperature, max_tokens, stream,
        )
