"""SG API Bridge — 通过 Anthropic API 调用 Claude，本地执行工具.

替代 SSH Bridge，直接用 Claude API：
- 调用 Anthropic API（使用 SG 节点的 API key）
- 解析流式响应（thinking, tool_use）
- 本地拦截工具调用，请求权限确认
- 本地执行工具
- 继续推理直到完成

优势：
- 完全控制工具执行和权限确认
- 支持流式输出（thinking blocks, tool use blocks）
- 蚁聚和飞书体验一致
- 不依赖 Claude Code CLI
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from crew.paths import resolve_project_dir

logger = logging.getLogger(__name__)


@dataclass
class SGAPIConfig:
    """SG API Bridge 配置."""

    # Anthropic API
    api_key: str = ""  # 必须通过配置文件提供
    api_base_url: str = "https://api.anthropic.com"  # 可选自定义

    # 模型配置
    default_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 8192
    temperature: float = 1.0

    # 是否启用
    enabled: bool = True


def load_sg_api_config(project_dir: Path | None = None) -> SGAPIConfig:
    """从 .crew/sg_api.yaml 加载配置."""
    base = resolve_project_dir(project_dir)
    config_path = base / ".crew" / "sg_api.yaml"

    if not config_path.exists():
        return SGAPIConfig(enabled=False)

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return SGAPIConfig(enabled=False)

        config = SGAPIConfig(**data)

        # 验证必需的 API key
        if not config.api_key:
            logger.warning("SG API Bridge 配置缺少 api_key，已禁用")
            return SGAPIConfig(enabled=False)

        return config
    except Exception as e:
        logger.warning("SG API Bridge 配置加载失败: %s", e)
        return SGAPIConfig(enabled=False)


class SGAPIBridgeError(Exception):
    """SG API Bridge 通用异常."""


async def sg_api_dispatch(
    message: str,
    *,
    ctx: Any = None,
    project_dir: Path | None = None,
    employee_name: str | None = None,
    message_history: list[dict] | None = None,
    permission_callback: Callable[[str, dict], bool] | None = None,
    push_event_fn: Callable[[dict], None] | None = None,
    channel: str = "",
    sender_type: str = "",
) -> str:
    """SG API 转发主入口 — 使用 Anthropic API + 本地工具执行.

    Args:
        message: 用户消息
        ctx: AppContext（用于工具执行）
        project_dir: 项目目录
        employee_name: 员工名称（用于加载 soul）
        message_history: 对话历史
        permission_callback: 权限回调 async (tool_name, tool_params) -> bool
        push_event_fn: 事件推送函数（用于流式输出）
        channel: 对话渠道标识（空字符串时按内部对话处理）

    Returns:
        最终回复文本

    Raises:
        SGAPIBridgeError: 执行失败
    """
    config = load_sg_api_config(project_dir)
    if not config.enabled:
        raise SGAPIBridgeError("SG API Bridge 未启用")

    # 加载员工 soul
    soul = ""
    if employee_name:
        soul = _get_employee_soul(employee_name, project_dir)

    # 构建 system prompt
    system_parts = []
    if soul:
        system_parts.append(soul)

    system_parts.extend(
        [
            "",
            "【聊天规则】",
            "1. 不要加【墨言】或任何方括号前缀",
            "2. 不要输出 Sources/来源 引用块",
            "3. 直接用自然中文回复，像微信聊天一样简洁",
        ]
    )

    # Phase 3：外部对话输出控制（综合 channel + sender_type）
    from crew.classification import CHANNEL_SOURCE_TYPE, EXTERNAL_OUTPUT_CONTROL_PROMPT

    _source_type = CHANNEL_SOURCE_TYPE.get(channel, "external" if channel else "internal")
    # 内部员工（sender_type=internal/agent）在蚁聚聊天时不注入外部限制
    _is_internal_sender = sender_type in ("internal", "agent")
    if _source_type == "external" and not _is_internal_sender:
        system_parts.append("")
        system_parts.append(EXTERNAL_OUTPUT_CONTROL_PROMPT)

    system_prompt = "\n".join(system_parts)

    # 构建消息历史
    messages = []
    if message_history:
        for msg in message_history[-6:]:  # 最近 6 条
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    # 当前消息
    messages.append({"role": "user", "content": message})

    # 调用 Anthropic API（带工具执行循环）
    try:
        import anthropic
    except ImportError:
        raise SGAPIBridgeError("anthropic SDK 未安装")

    client = anthropic.AsyncAnthropic(api_key=config.api_key)

    # 加载可用工具
    tools = _load_available_tools(project_dir)

    max_turns = 10  # 最多 10 轮工具调用
    collected_text = []

    for turn in range(max_turns):
        logger.info("SG API turn %d: messages=%d", turn + 1, len(messages))

        # 调用 API（流式）
        try:
            kwargs = {
                "model": config.default_model,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "system": system_prompt,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools

            event_stream = client.messages.stream(**kwargs)

            event_stream = client.messages.stream(**kwargs)

            # 处理事件流
            tool_uses = []
            current_tool = None
            turn_text = []

            async with event_stream as stream:
                async for event in stream:
                    event_type = event.type

                    # 推送事件到前端（转换为统一格式）
                    if push_event_fn:
                        frontend_event = _convert_event_to_frontend(event)
                        if frontend_event:
                            push_event_fn(frontend_event)

                    # 收集文本
                    if event_type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            text = delta.text
                            turn_text.append(text)

                    # 收集工具调用
                    elif event_type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            current_tool = {
                                "id": block.id,
                                "name": block.name,
                                "input": "",
                            }

                    elif event_type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "input_json_delta" and current_tool:
                            current_tool["input"] += delta.partial_json

                    elif event_type == "content_block_stop":
                        if current_tool:
                            # 解析完整的 tool input
                            try:
                                current_tool["input"] = json.loads(current_tool["input"])
                            except json.JSONDecodeError:
                                logger.warning("工具输入 JSON 解析失败: %s", current_tool["input"])
                            tool_uses.append(current_tool)
                            current_tool = None

                # 获取最终消息
                final_message = await stream.get_final_message()
                stop_reason = final_message.stop_reason

            # 收集文本
            turn_text_str = "".join(turn_text)
            if turn_text_str:
                collected_text.append(turn_text_str)

            # 如果没有工具调用，结束
            if not tool_uses or stop_reason == "end_turn":
                break

            # 执行工具
            tool_results = []
            for tool_use in tool_uses:
                tool_name = tool_use["name"]
                tool_input = tool_use["input"]
                tool_id = tool_use["id"]

                # 执行工具（复用 webhook_executor 的逻辑）
                tool_output = await _execute_tool(
                    ctx=ctx,
                    employee_name=employee_name or "assistant",
                    tool_name=tool_name,
                    tool_input=tool_input,
                    project_dir=project_dir,
                    push_event_fn=push_event_fn,
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": tool_output or "[工具执行完成]",
                    }
                )

            # 构建下一轮消息
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": turn_text_str},
                        *[
                            {
                                "type": "tool_use",
                                "id": t["id"],
                                "name": t["name"],
                                "input": t["input"],
                            }
                            for t in tool_uses
                        ],
                    ],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": tool_results,
                }
            )

        except Exception as e:
            logger.error("SG API 调用失败: %s", e, exc_info=True)
            raise SGAPIBridgeError(f"API 调用失败: {e}") from e

    return "".join(collected_text)


def _get_employee_soul(employee_name: str, project_dir: Path | None = None) -> str:
    """加载员工 soul（复用 sg_bridge 的实现）."""
    try:
        from crew.discovery import discover_employees

        result = discover_employees(project_dir=project_dir)
        employee = result.get(employee_name)
        if employee is None:
            return ""

        # 优先用 chat-profile.md
        if employee.source_path:
            chat_profile = employee.source_path / "chat-profile.md"
            if chat_profile.exists():
                return chat_profile.read_text(encoding="utf-8").strip()

        # fallback: 完整 body
        return employee.body or ""
    except Exception as e:
        logger.warning("加载员工 soul 失败: %s", e)
        return ""


async def _execute_tool(
    ctx: Any,
    employee_name: str,
    tool_name: str,
    tool_input: dict,
    project_dir: Path | None,
    push_event_fn: Callable[[dict], None] | None = None,
) -> str:
    """执行工具（复用 crew webhook_executor 的逻辑）."""
    try:
        from crew.webhook_executor import _handle_tool_call

        # 调用现有的工具执行逻辑（包含权限确认）
        # TODO: sg_api_dispatch 尚未传入 sender_id，待认证体系完善后补充 target_user_id
        result = await _handle_tool_call(
            ctx=ctx,
            employee_name=employee_name,
            tool_name=tool_name,
            arguments=tool_input,
            agent_id=None,
            guard=None,  # 不使用 guard，权限确认在 _handle_tool_call 内部处理
            max_visibility="open",
            push_event_fn=push_event_fn,
        )

        return result or "[工具执行完成]"

    except Exception as e:
        logger.error("工具执行失败: %s", e, exc_info=True)
        return f"[错误] {e}"


def _load_available_tools(project_dir: Path | None) -> list[dict]:
    """加载可用工具定义（Anthropic API 格式）."""
    # TODO: 这里应该根据员工配置加载工具
    # 暂时返回空列表，让 Claude 自己决定需要什么工具
    return []


def _convert_event_to_frontend(event: Any) -> dict | None:
    """转换 Anthropic API 事件为前端格式."""
    event_type = event.type

    if event_type == "content_block_start":
        block = event.content_block
        evt = {
            "type": "content_block_start",
            "index": event.index,
            "content_type": block.type,
        }
        if block.type == "tool_use":
            evt["tool_name"] = block.name
            evt["tool_use_id"] = block.id
        return evt

    elif event_type == "content_block_delta":
        delta = event.delta
        evt = {
            "type": "content_block_delta",
            "index": event.index,
        }
        if delta.type == "text_delta":
            evt["content_type"] = "text"
            evt["text"] = delta.text
        elif delta.type == "thinking_delta":
            evt["content_type"] = "thinking"
            evt["thinking"] = delta.thinking
        elif delta.type == "input_json_delta":
            evt["content_type"] = "tool_use"
            evt["tool_input"] = delta.partial_json
        return evt

    elif event_type == "content_block_stop":
        return {
            "type": "content_block_stop",
            "index": event.index,
        }

    elif event_type == "message_delta":
        evt = {"type": "message_delta"}
        if hasattr(event.delta, "stop_reason") and event.delta.stop_reason:
            evt["stop_reason"] = event.delta.stop_reason
        return evt

    return None
