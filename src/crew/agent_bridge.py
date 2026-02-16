"""Agent 桥接 — 将 crew 员工 + LLM tool_use 封装为 collect() 兼容的 agent 函数."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from crew.engine import CrewEngine
from crew.executor import execute_with_tools
from crew.models import ToolCall, ToolExecutionResult
from crew.tool_schema import (
    employee_tools_to_schemas,
    is_finish_tool,
    map_tool_call,
)

logger = logging.getLogger(__name__)


def create_crew_agent(
    employee_name: str,
    task_description: str,
    *,
    model: str = "claude-sonnet-4-5-20250929",
    api_key: str | None = None,
    max_tokens: int = 4096,
    project_dir: Path | None = None,
    smart_context: bool = True,
    on_step: Callable[[int, str, dict[str, Any]], None] | None = None,
) -> Callable[[str], dict[str, Any]]:
    """创建基于 crew 员工的 agent 函数.

    返回兼容 knowlyr-agent collect() 签名的 agent:
        agent(observation: str) -> {"tool": "...", "params": {...}}

    Args:
        employee_name: 员工名称（如 "code-reviewer"）
        task_description: 任务描述（如 "审查 src/auth.py 的安全性"）
        model: LLM 模型 ID
        api_key: API key（None 时自动解析）
        max_tokens: 最大输出 token
        project_dir: 项目目录
        smart_context: 是否使用智能上下文检测
        on_step: 每步回调 (step_num, tool_name, params)

    Returns:
        agent 函数: (observation: str) -> {"tool": ..., "params": ...}
    """
    from crew.discovery import discover_employees

    # 1. 加载员工
    discovery = discover_employees(project_dir=project_dir)
    employee = discovery.get(employee_name)
    if employee is None:
        raise ValueError(f"未找到员工: {employee_name}")

    # 2. 生成 system prompt
    engine = CrewEngine(project_dir=project_dir)
    project_info = None
    if smart_context:
        try:
            from crew.context_detector import detect_project
            project_info = detect_project(project_dir)
        except Exception:
            pass

    system_prompt = engine.prompt(employee, project_info=project_info)

    # 3. 权限守卫
    from crew.permission import PermissionGuard
    guard = PermissionGuard(employee)

    # 4. 生成 tool schemas
    tool_schemas, _ = employee_tools_to_schemas(employee.tools, defer=False)

    # 5. 闭包状态
    messages: list[dict[str, Any]] = []
    step_count = 0
    last_tool_calls: list[ToolCall] = []
    _deny_retries = 0  # 连续权限拒绝计数，防止无限递归

    def agent(observation: str) -> dict[str, Any]:
        nonlocal step_count, last_tool_calls, _deny_retries

        step_count += 1

        # 轨迹录制：用上一步的 tool_result 补全 pending step
        try:
            from crew.trajectory import TrajectoryCollector

            _collector = TrajectoryCollector.current()
            if _collector is not None and step_count > 1 and last_tool_calls:
                _collector.complete_tool_step(
                    tool_output=observation[:5000],
                )
        except Exception:
            _collector = None

        # 首次调用：发送任务描述 + 初始 observation
        if step_count == 1:
            user_content = f"## 任务\n\n{task_description}"
            if observation and observation != "沙箱就绪":
                user_content += f"\n\n## 环境状态\n\n{observation}"
            messages.append({"role": "user", "content": user_content})
        else:
            # 后续调用：追加 tool_result（Anthropic 格式）
            if last_tool_calls:
                # 构建 assistant message (tool_use blocks)
                assistant_content: list[dict[str, Any]] = []
                for tc in last_tool_calls:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                messages.append({"role": "assistant", "content": assistant_content})

                # 构建 tool result message（每个 tool_use 都需要对应的 tool_result）
                tool_results = []
                for tc in last_tool_calls:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": observation[:10000] if tc is last_tool_calls[0] else "",
                    })
                messages.append({"role": "user", "content": tool_results})
            else:
                messages.append({"role": "user", "content": observation[:10000]})

        # 5. 调用 LLM (with tool_use)
        try:
            result: ToolExecutionResult = execute_with_tools(
                system_prompt=system_prompt,
                messages=messages,
                tools=tool_schemas,
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error("LLM 调用失败 (step %d): %s", step_count, e)
            return {"tool": "submit", "params": {"result": f"LLM 调用失败: {e}"}}

        # 6. 解析工具调用
        if result.has_tool_calls:
            tc = result.tool_calls[0]  # 取第一个工具调用
            last_tool_calls = result.tool_calls

            # 轨迹录制：记录 thought + tool_call，等待下次 observation 补全
            if _collector is not None:
                _collector.begin_tool_step(
                    thought=result.content,
                    tool_name=tc.name,
                    tool_params=tc.arguments,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    model=result.model,
                )

            if on_step:
                on_step(step_count, tc.name, tc.arguments)

            # 终止工具
            if is_finish_tool(tc.name):
                return {"tool": "submit", "params": tc.arguments}

            # 权限检查 — 拒绝时反馈给 LLM 让其自我纠正（最多重试 3 次）
            denied_msg = guard.check_soft(tc.name)
            if denied_msg:
                logger.warning("Agent bridge 权限拒绝: %s.%s", employee_name, tc.name)
                _deny_retries += 1
                if _deny_retries > 3:
                    return {"tool": "submit", "params": {"result": f"[权限拒绝] {denied_msg}"}}
                return agent(f"[权限拒绝] {denied_msg}")

            _deny_retries = 0  # 成功通过，重置计数

            # 映射为 sandbox 格式
            return map_tool_call(tc.name, tc.arguments)

        # 无工具调用 — LLM 给出了最终文本回复
        last_tool_calls = []

        # 轨迹录制：最终回复记为完整步骤
        if _collector is not None:
            _collector.add_tool_step(
                thought=result.content,
                tool_name="submit",
                tool_params={},
                tool_output=result.content,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )

        if on_step:
            on_step(step_count, "submit", {"result": result.content[:200]})
        return {"tool": "submit", "params": {"result": result.content}}

    return agent
