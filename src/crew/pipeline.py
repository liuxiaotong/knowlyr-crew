"""流水线引擎 — 多员工顺序/并行执行，支持输出传递和 LLM 自动执行."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

# 异步并行步骤的超时（秒）
_ASYNC_STEP_TIMEOUT = 600

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from crew.context_detector import detect_project
from crew.discovery import discover_employees
from crew.engine import CrewEngine
from crew.models import (
    ConditionalStep,
    LoopStep,
    ParallelGroup,
    PipelineResult,
    PipelineStep,
    StepResult,
)

if TYPE_CHECKING:
    from crew.context_detector import ProjectInfo
    from crew.id_client import AgentIdentity
    from crew.models import DiscoveryResult, DiscussionActionPlan


# ── Pipeline 定义 ──


class Pipeline(BaseModel):
    """流水线定义."""

    name: str = Field(description="流水线名称")
    description: str = Field(default="", description="描述")
    steps: list[PipelineStep | ParallelGroup | ConditionalStep | LoopStep] = Field(
        description="步骤列表"
    )


# ── 加载 / 校验 ──


def load_pipeline(path: Path) -> Pipeline:
    """从 YAML 文件加载流水线定义.

    支持两种步骤格式：
    - 普通步骤: {employee, args, id?}
    - 并行组: {parallel: [{employee, args, id?}, ...]}
    """
    content = path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)

    # 标准化 steps：区分并行组、条件分支、循环与普通步骤
    raw_steps = data.get("steps", [])
    normalized: list[dict] = []
    for item in raw_steps:
        if "parallel" in item:
            normalized.append({"parallel": item["parallel"]})
        elif "condition" in item:
            normalized.append({"condition": item["condition"]})
        elif "loop" in item:
            normalized.append({"loop": item["loop"]})
        else:
            normalized.append(item)
    data["steps"] = normalized

    return Pipeline(**data)


def validate_pipeline(
    pipeline: Pipeline, project_dir: Path | None = None,
) -> list[str]:
    """校验流水线定义，返回错误列表."""
    errors: list[str] = []
    if not pipeline.steps:
        errors.append("流水线至少需要一个步骤")
        return errors

    result = discover_employees(project_dir=project_dir)
    seen_ids: set[str] = set()

    def _check_step(step: PipelineStep, position: str) -> None:
        if step.id:
            if step.id in seen_ids:
                errors.append(f"{position}: 重复的步骤 ID '{step.id}'")
            seen_ids.add(step.id)
        emp = result.get(step.employee)
        if emp is None:
            errors.append(f"{position}: 未找到员工 '{step.employee}'")

    for i, item in enumerate(pipeline.steps):
        if isinstance(item, ParallelGroup):
            for j, sub in enumerate(item.parallel):
                _check_step(sub, f"步骤 {i + 1} 并行子步骤 {j + 1}")
        elif isinstance(item, ConditionalStep):
            body = item.condition
            if not body.then:
                errors.append(f"步骤 {i + 1}: condition.then 不能为空")
            for j, sub in enumerate(body.then):
                _check_step(sub, f"步骤 {i + 1} condition.then[{j + 1}]")
            for j, sub in enumerate(body.else_):
                _check_step(sub, f"步骤 {i + 1} condition.else[{j + 1}]")
        elif isinstance(item, LoopStep):
            body = item.loop
            if not body.steps:
                errors.append(f"步骤 {i + 1}: loop.steps 不能为空")
            for j, sub in enumerate(body.steps):
                _check_step(sub, f"步骤 {i + 1} loop.steps[{j + 1}]")
        else:
            _check_step(item, f"步骤 {i + 1}")

    return errors


def pipeline_to_mermaid(pipeline: Pipeline) -> str:
    """将流水线定义转换为 Mermaid 流程图.

    Returns:
        Mermaid markdown 文本.
    """
    lines = ["graph LR"]
    lines.append('  S(["开始"])')

    prev_nodes: list[str] = ["S"]
    parallel_count = 0
    cond_count = 0
    loop_count = 0

    for i, item in enumerate(pipeline.steps):
        if isinstance(item, ParallelGroup):
            parallel_count += 1
            fork = f"F{parallel_count}"
            join = f"J{parallel_count}"
            lines.append(f'  {fork}{{"并行"}}')
            lines.append(f'  {join}{{"合并"}}')

            for prev in prev_nodes:
                lines.append(f"  {prev} --> {fork}")

            sub_nodes = []
            for j, sub in enumerate(item.parallel):
                node_id = sub.id or f"s{i}_{j}"
                lines.append(f'  {node_id}["{sub.employee}"]')
                lines.append(f"  {fork} --> {node_id}")
                lines.append(f"  {node_id} --> {join}")
                sub_nodes.append(node_id)

            prev_nodes = [join]
        elif isinstance(item, ConditionalStep):
            cond_count += 1
            body = item.condition
            cond_id = f"C{cond_count}"
            merge_id = f"CM{cond_count}"
            label = f"contains '{body.contains}'" if body.contains else f"matches '{body.matches}'"
            lines.append(f'  {cond_id}{{"{label}"}}')
            for prev in prev_nodes:
                lines.append(f"  {prev} --> {cond_id}")

            # then 分支
            then_prev = cond_id
            for j, sub in enumerate(body.then):
                nid = sub.id or f"ct{cond_count}_{j}"
                lines.append(f'  {nid}["{sub.employee}"]')
                edge = " -->|then| " if j == 0 else " --> "
                lines.append(f"  {then_prev}{edge}{nid}")
                then_prev = nid
            then_last = then_prev

            # else 分支
            if body.else_:
                else_prev = cond_id
                for j, sub in enumerate(body.else_):
                    nid = sub.id or f"ce{cond_count}_{j}"
                    lines.append(f'  {nid}["{sub.employee}"]')
                    edge = " -->|else| " if j == 0 else " --> "
                    lines.append(f"  {else_prev}{edge}{nid}")
                    else_prev = nid
                else_last = else_prev
            else:
                else_last = cond_id

            lines.append(f'  {merge_id}(["合并"])')
            lines.append(f"  {then_last} --> {merge_id}")
            if else_last != cond_id or body.else_:
                lines.append(f"  {else_last} --> {merge_id}")
            prev_nodes = [merge_id]
        elif isinstance(item, LoopStep):
            loop_count += 1
            body = item.loop
            loop_id = f"L{loop_count}"
            check_id = f"LC{loop_count}"
            lines.append(f'  {loop_id}[["循环 (max {body.max_iterations})"]]')
            for prev in prev_nodes:
                lines.append(f"  {prev} --> {loop_id}")

            loop_prev = loop_id
            for j, sub in enumerate(body.steps):
                nid = sub.id or f"ls{loop_count}_{j}"
                lines.append(f'  {nid}["{sub.employee}"]')
                lines.append(f"  {loop_prev} --> {nid}")
                loop_prev = nid

            label = f"contains '{body.until.contains}'" if body.until.contains else f"matches '{body.until.matches}'"
            lines.append(f'  {check_id}{{"{label}"}}')
            lines.append(f"  {loop_prev} --> {check_id}")
            lines.append(f'  {check_id} -->|"继续"| {loop_id}')
            prev_nodes = [check_id]
        else:
            node_id = item.id or f"s{i}"
            lines.append(f'  {node_id}["{item.employee}"]')
            for prev in prev_nodes:
                lines.append(f"  {prev} --> {node_id}")
            prev_nodes = [node_id]

    # 终止节点
    lines.append('  E(["结束"])')
    for prev in prev_nodes:
        lines.append(f"  {prev} --> E")

    return "\n".join(lines)


# ── 输出引用解析 ──


def _resolve_initial_args(value: str, initial_args: dict[str, str]) -> str:
    """解析 $variable 引用."""
    if value.startswith("$") and value[1:] in initial_args:
        return initial_args[value[1:]]
    return value


def _resolve_output_refs(
    value: str,
    outputs_by_id: dict[str, str],
    outputs_by_index: dict[int, str],
    prev_output: str,
    execute: bool,
) -> str:
    """解析输出引用占位符.

    - {prev}: 上一步输出
    - {steps.<id>.output}: 按 ID 引用
    - {steps.<N>.output}: 按 flat index 引用

    prompt-only 模式下保留占位符原样，execute 模式下替换为实际输出。
    """
    if not execute:
        return value

    value = value.replace("{prev}", prev_output)

    def _replace_ref(m: re.Match) -> str:
        ref = m.group(1)
        if ref in outputs_by_id:
            return outputs_by_id[ref]
        try:
            idx = int(ref)
            result = outputs_by_index.get(idx, m.group(0))
            if result == m.group(0):
                logger.warning("未解析输出引用: %s", m.group(0))
            return result
        except ValueError:
            logger.warning("未解析输出引用: %s", m.group(0))
            return m.group(0)

    value = re.sub(r"\{steps\.([^.]+)\.output\}", _replace_ref, value)
    return value


# ── 条件评估 ──


_REGEX_TIMEOUT = 2.0  # 正则匹配最大秒数


def _evaluate_check(
    check: str,
    contains: str,
    matches: str,
    outputs_by_id: dict[str, str],
    outputs_by_index: dict[int, str],
    prev_output: str,
    execute: bool,
) -> bool:
    """解析条件中的输出引用并评估.

    prompt-only 模式总是返回 True（走 then 分支 / 继续循环一次）。
    正则匹配有 2 秒超时防护，超时视为不匹配。
    """
    if not execute:
        return True
    resolved = _resolve_output_refs(check, outputs_by_id, outputs_by_index, prev_output, execute)
    if contains:
        return contains in resolved
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            return bool(pool.submit(re.search, matches, resolved).result(timeout=_REGEX_TIMEOUT))
        except FuturesTimeout:
            logger.warning("正则匹配超时 (%.1fs)，视为不匹配: %s", _REGEX_TIMEOUT, matches[:80])
            return False


# ── 单步执行 ──


def _resolve_agent_identity(agent_id: int | None) -> "AgentIdentity | None":
    """获取 agent 身份信息（可选）."""
    if agent_id is None:
        return None
    try:
        from crew.id_client import fetch_agent_identity
        return fetch_agent_identity(agent_id)
    except ImportError:
        return None


class AgentDisabledError(RuntimeError):
    """Agent 已停用，拒绝执行."""


def _check_agent_active(agent_identity: "AgentIdentity | None", agent_id: int | None) -> None:
    """如果 agent 已停用则抛出异常."""
    if agent_identity is None:
        return
    status = agent_identity.agent_status
    if status and status != "active":
        raise AgentDisabledError(
            f"Agent {agent_id} 状态为 '{status}'，拒绝执行"
        )


def _resolve_exemplars(agent_id: int | None, employee_name: str) -> str:
    """获取 agent 的 few-shot 范例提示文本（可选）."""
    if agent_id is None:
        return ""
    try:
        from crew.id_client import fetch_exemplars
        return fetch_exemplars(agent_id, employee_name)
    except Exception:
        return ""


def _execute_single_step(
    step: PipelineStep,
    index: int,
    engine: CrewEngine,
    employees: "DiscoveryResult",
    initial_args: dict[str, str],
    outputs_by_id: dict[str, str],
    outputs_by_index: dict[int, str],
    prev_output: str,
    agent_identity: "AgentIdentity | None",
    project_info: "ProjectInfo | None",
    execute: bool,
    api_key: str | None,
    model: str | None,
    exemplar_prompt: str = "",
) -> StepResult:
    """执行单个步骤：解析参数 → 生成 prompt → 可选 LLM 调用."""
    emp = employees.get(step.employee)
    if emp is None:
        return StepResult(
            employee=step.employee,
            step_id=step.id,
            step_index=index,
            args=step.args,
            prompt=f"[错误] 未找到员工: {step.employee}",
            error=True,
            error_message=f"未找到员工: {step.employee}",
        )

    # 1. 解析 $variable + 输出引用
    resolved_args: dict[str, str] = {}
    for k, v in step.args.items():
        v = _resolve_initial_args(v, initial_args)
        v = _resolve_output_refs(v, outputs_by_id, outputs_by_index, prev_output, execute)
        resolved_args[k] = v

    # 2. 生成 prompt
    prompt = engine.prompt(
        emp,
        args=resolved_args,
        agent_identity=agent_identity,
        project_info=project_info,
        exemplar_prompt=exemplar_prompt,
    )

    # 3. 可选 LLM 执行
    output = ""
    result_model = ""
    input_tokens = 0
    output_tokens = 0
    duration_ms = 0
    error = False
    error_message = ""

    if execute:
        try:
            from crew.executor import execute_prompt

            use_model = emp.model or model or "claude-sonnet-4-20250514"
            t0 = time.monotonic()
            exec_result = execute_prompt(
                system_prompt=prompt,
                api_key=api_key,
                model=use_model,
                stream=False,
            )
            duration_ms = int((time.monotonic() - t0) * 1000)
            output = exec_result.content
            result_model = exec_result.model
            input_tokens = exec_result.input_tokens
            output_tokens = exec_result.output_tokens
        except Exception as exc:
            error = True
            error_message = str(exc)[:500]

    return StepResult(
        employee=step.employee,
        step_id=step.id,
        step_index=index,
        args=resolved_args,
        prompt=prompt,
        output=output,
        error=error,
        error_message=error_message,
        model=result_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=duration_ms,
    )


# ── 主执行入口 ──


def run_pipeline(
    pipeline: Pipeline,
    initial_args: dict[str, str] | None = None,
    project_dir: Path | None = None,
    agent_id: int | None = None,
    smart_context: bool = True,
    *,
    execute: bool = False,
    api_key: str | None = None,
    model: str | None = None,
    on_step_complete: Callable[[StepResult], None] | None = None,
) -> PipelineResult:
    """执行流水线.

    Args:
        pipeline: 流水线定义.
        initial_args: 初始参数（替换 $variable）.
        project_dir: 项目目录.
        agent_id: Agent ID.
        smart_context: 自动检测项目类型.
        execute: 启用 execute 模式（调用 LLM）.
        api_key: Anthropic API key（execute 模式需要）.
        model: LLM 模型标识符.
        on_step_complete: 每步完成后的回调.

    Returns:
        PipelineResult.
    """
    initial_args = initial_args or {}
    employees = discover_employees(project_dir=project_dir)
    engine = CrewEngine(project_dir=project_dir)
    project_info = detect_project(project_dir) if smart_context else None
    agent_identity = _resolve_agent_identity(agent_id)
    _check_agent_active(agent_identity, agent_id)

    # 启用轨迹录制（仅 execute 模式）
    _traj_collector = None
    if execute:
        try:
            from crew.trajectory import TrajectoryCollector

            _traj_collector = TrajectoryCollector(
                f"pipeline/{pipeline.name}",
                pipeline.description or pipeline.name,
                model=model or "",
            )
            _traj_collector.__enter__()
        except Exception:
            _traj_collector = None

    # 输出注册表
    outputs_by_id: dict[str, str] = {}
    outputs_by_index: dict[int, str] = {}
    prev_output: str = ""

    step_results: list[StepResult | list[StepResult]] = []
    flat_index = 0
    t0 = time.monotonic()

    for item in pipeline.steps:
        if isinstance(item, ParallelGroup):
            group_results: list[StepResult] = []

            if execute:
                # 并行 LLM 调用
                with ThreadPoolExecutor(max_workers=len(item.parallel)) as pool:
                    futures = {}
                    for sub in item.parallel:
                        idx = flat_index
                        ex_prompt = _resolve_exemplars(agent_id, sub.employee)
                        future = pool.submit(
                            _execute_single_step,
                            sub, idx, engine, employees, initial_args,
                            outputs_by_id, outputs_by_index, prev_output,
                            agent_identity, project_info,
                            execute, api_key, model, ex_prompt,
                        )
                        futures[future] = (sub, idx)
                        flat_index += 1

                    for future in as_completed(futures):
                        r = future.result()
                        if r.error:
                            logger.warning(
                                "并行步骤 %s (#%d) 失败: %s",
                                r.employee, r.step_index, r.error_message,
                            )
                        group_results.append(r)
                        if r.step_id:
                            outputs_by_id[r.step_id] = r.output
                        outputs_by_index[r.step_index] = r.output
                        if on_step_complete:
                            on_step_complete(r)
            else:
                # prompt-only 模式顺序生成
                for sub in item.parallel:
                    ex_prompt = _resolve_exemplars(agent_id, sub.employee)
                    r = _execute_single_step(
                        sub, flat_index, engine, employees, initial_args,
                        outputs_by_id, outputs_by_index, prev_output,
                        agent_identity, project_info,
                        execute, api_key, model, ex_prompt,
                    )
                    if r.error:
                        logger.warning(
                            "并行步骤 %s (#%d) 失败: %s",
                            r.employee, r.step_index, r.error_message,
                        )
                    group_results.append(r)
                    if r.step_id:
                        outputs_by_id[r.step_id] = r.output
                    outputs_by_index[flat_index] = r.output
                    flat_index += 1
                    if on_step_complete:
                        on_step_complete(r)

            # 按 step_index 排序确保结果顺序稳定
            group_results.sort(key=lambda r: r.step_index)
            step_results.append(group_results)
            prev_output = "\n\n---\n\n".join(r.output for r in group_results)
        elif isinstance(item, ConditionalStep):
            body = item.condition
            took_then = _evaluate_check(
                body.check, body.contains, body.matches,
                outputs_by_id, outputs_by_index, prev_output, execute,
            )
            branch_steps = body.then if took_then else body.else_
            branch_label = "then" if took_then else "else"

            branch_results: list[StepResult] = []
            for sub in branch_steps:
                ex_prompt = _resolve_exemplars(agent_id, sub.employee)
                r = _execute_single_step(
                    sub, flat_index, engine, employees, initial_args,
                    outputs_by_id, outputs_by_index, prev_output,
                    agent_identity, project_info,
                    execute, api_key, model, ex_prompt,
                )
                r.branch = branch_label
                if r.step_id:
                    outputs_by_id[r.step_id] = r.output
                outputs_by_index[flat_index] = r.output
                flat_index += 1
                branch_results.append(r)
                prev_output = r.output
                if on_step_complete:
                    on_step_complete(r)

            if branch_results:
                step_results.append(
                    branch_results if len(branch_results) > 1 else branch_results[0],
                )
        elif isinstance(item, LoopStep):
            body = item.loop
            loop_all_results: list[StepResult] = []

            for iteration in range(body.max_iterations):
                for sub in body.steps:
                    ex_prompt = _resolve_exemplars(agent_id, sub.employee)
                    r = _execute_single_step(
                        sub, flat_index, engine, employees, initial_args,
                        outputs_by_id, outputs_by_index, prev_output,
                        agent_identity, project_info,
                        execute, api_key, model, ex_prompt,
                    )
                    r.branch = f"loop-{iteration}"
                    if r.step_id:
                        outputs_by_id[r.step_id] = r.output
                    outputs_by_index[flat_index] = r.output
                    flat_index += 1
                    loop_all_results.append(r)
                    prev_output = r.output
                    if on_step_complete:
                        on_step_complete(r)

                # 检查终止条件
                should_stop = _evaluate_check(
                    body.until.check, body.until.contains, body.until.matches,
                    outputs_by_id, outputs_by_index, prev_output, execute,
                )
                if should_stop:
                    break
                if not execute:
                    break  # prompt-only 只执行一次

            if loop_all_results:
                step_results.append(
                    loop_all_results if len(loop_all_results) > 1 else loop_all_results[0],
                )
        else:
            ex_prompt = _resolve_exemplars(agent_id, item.employee)
            r = _execute_single_step(
                item, flat_index, engine, employees, initial_args,
                outputs_by_id, outputs_by_index, prev_output,
                agent_identity, project_info,
                execute, api_key, model, ex_prompt,
            )
            if r.step_id:
                outputs_by_id[r.step_id] = r.output
            outputs_by_index[flat_index] = r.output
            flat_index += 1
            step_results.append(r)
            prev_output = r.output
            if on_step_complete:
                on_step_complete(r)

    total_ms = int((time.monotonic() - t0) * 1000)

    # 汇总 token 统计
    all_results = _flatten_results(step_results)

    # 完成轨迹录制
    if _traj_collector is not None:
        try:
            has_error = any(r.error for r in all_results)
            _traj_collector.finish(success=not has_error)
        except Exception as e:
            logger.debug("流水线轨迹录制失败: %s", e)
        finally:
            _traj_collector.__exit__(None, None, None)

    return PipelineResult(
        pipeline_name=pipeline.name,
        mode="execute" if execute else "prompt",
        steps=step_results,
        total_duration_ms=total_ms,
        total_input_tokens=sum(r.input_tokens for r in all_results),
        total_output_tokens=sum(r.output_tokens for r in all_results),
    )


# ── 异步版本（供 MCP Server 使用）──


async def arun_pipeline(
    pipeline: Pipeline,
    initial_args: dict[str, str] | None = None,
    project_dir: Path | None = None,
    agent_id: int | None = None,
    smart_context: bool = True,
    *,
    execute: bool = False,
    api_key: str | None = None,
    model: str | None = None,
    on_step_complete: Callable[[StepResult, dict], None] | None = None,
) -> PipelineResult:
    """异步执行流水线 — 并行组使用 asyncio.gather.

    Args:
        on_step_complete: 每步完成后的回调 (step_result, checkpoint_data).
            checkpoint_data 包含恢复所需的完整状态。
    """
    initial_args = initial_args or {}
    employees = discover_employees(project_dir=project_dir)
    engine = CrewEngine(project_dir=project_dir)
    project_info = detect_project(project_dir) if smart_context else None
    agent_identity = _resolve_agent_identity(agent_id)
    _check_agent_active(agent_identity, agent_id)

    outputs_by_id: dict[str, str] = {}
    outputs_by_index: dict[int, str] = {}
    prev_output: str = ""

    step_results: list[StepResult | list[StepResult]] = []
    flat_index = 0
    t0 = time.monotonic()

    for step_i, item in enumerate(pipeline.steps):
        if isinstance(item, ParallelGroup):
            if execute:
                tasks = []
                indices = []
                for sub in item.parallel:
                    idx = flat_index
                    indices.append(idx)
                    ex_prompt = _resolve_exemplars(agent_id, sub.employee)
                    tasks.append(
                        _aexecute_single_step(
                            sub, idx, engine, employees, initial_args,
                            outputs_by_id, outputs_by_index, prev_output,
                            agent_identity, project_info,
                            api_key, model, ex_prompt,
                        )
                    )
                    flat_index += 1
                group_results = list(
                    await asyncio.wait_for(
                        asyncio.gather(*tasks),
                        timeout=_ASYNC_STEP_TIMEOUT,
                    )
                )
                for r in group_results:
                    if r.error:
                        logger.warning(
                            "并行步骤 %s (#%d) 失败: %s",
                            r.employee, r.step_index, r.error_message,
                        )
            else:
                group_results = []
                for sub in item.parallel:
                    ex_prompt = _resolve_exemplars(agent_id, sub.employee)
                    r = _execute_single_step(
                        sub, flat_index, engine, employees, initial_args,
                        outputs_by_id, outputs_by_index, prev_output,
                        agent_identity, project_info,
                        False, None, None, ex_prompt,
                    )
                    group_results.append(r)
                    flat_index += 1

            group_results.sort(key=lambda r: r.step_index)
            for r in group_results:
                if r.step_id:
                    outputs_by_id[r.step_id] = r.output
                outputs_by_index[r.step_index] = r.output
            step_results.append(group_results)
            prev_output = "\n\n---\n\n".join(r.output for r in group_results)

            if on_step_complete:
                checkpoint = _build_checkpoint(
                    pipeline.name, step_results, outputs_by_id, outputs_by_index, flat_index, step_i + 1,
                )
                for r in group_results:
                    on_step_complete(r, checkpoint)
        elif isinstance(item, ConditionalStep):
            body = item.condition
            took_then = _evaluate_check(
                body.check, body.contains, body.matches,
                outputs_by_id, outputs_by_index, prev_output, execute,
            )
            branch_steps = body.then if took_then else body.else_
            branch_label = "then" if took_then else "else"

            branch_results: list[StepResult] = []
            for sub in branch_steps:
                ex_prompt = _resolve_exemplars(agent_id, sub.employee)
                if execute:
                    r = await _aexecute_single_step(
                        sub, flat_index, engine, employees, initial_args,
                        outputs_by_id, outputs_by_index, prev_output,
                        agent_identity, project_info,
                        api_key, model, ex_prompt,
                    )
                else:
                    r = _execute_single_step(
                        sub, flat_index, engine, employees, initial_args,
                        outputs_by_id, outputs_by_index, prev_output,
                        agent_identity, project_info,
                        False, None, None, ex_prompt,
                    )
                r.branch = branch_label
                if r.step_id:
                    outputs_by_id[r.step_id] = r.output
                outputs_by_index[flat_index] = r.output
                flat_index += 1
                branch_results.append(r)
                prev_output = r.output
                if on_step_complete:
                    checkpoint = _build_checkpoint(
                        pipeline.name, step_results, outputs_by_id, outputs_by_index, flat_index, step_i + 1,
                    )
                    on_step_complete(r, checkpoint)

            if branch_results:
                step_results.append(
                    branch_results if len(branch_results) > 1 else branch_results[0],
                )
        elif isinstance(item, LoopStep):
            body = item.loop
            loop_all_results: list[StepResult] = []

            for iteration in range(body.max_iterations):
                for sub in body.steps:
                    ex_prompt = _resolve_exemplars(agent_id, sub.employee)
                    if execute:
                        r = await _aexecute_single_step(
                            sub, flat_index, engine, employees, initial_args,
                            outputs_by_id, outputs_by_index, prev_output,
                            agent_identity, project_info,
                            api_key, model, ex_prompt,
                        )
                    else:
                        r = _execute_single_step(
                            sub, flat_index, engine, employees, initial_args,
                            outputs_by_id, outputs_by_index, prev_output,
                            agent_identity, project_info,
                            False, None, None, ex_prompt,
                        )
                    r.branch = f"loop-{iteration}"
                    if r.step_id:
                        outputs_by_id[r.step_id] = r.output
                    outputs_by_index[flat_index] = r.output
                    flat_index += 1
                    loop_all_results.append(r)
                    prev_output = r.output
                    if on_step_complete:
                        checkpoint = _build_checkpoint(
                            pipeline.name, step_results, outputs_by_id, outputs_by_index, flat_index, step_i + 1,
                        )
                        on_step_complete(r, checkpoint)

                should_stop = _evaluate_check(
                    body.until.check, body.until.contains, body.until.matches,
                    outputs_by_id, outputs_by_index, prev_output, execute,
                )
                if should_stop:
                    break
                if not execute:
                    break

            if loop_all_results:
                step_results.append(
                    loop_all_results if len(loop_all_results) > 1 else loop_all_results[0],
                )
        else:
            ex_prompt = _resolve_exemplars(agent_id, item.employee)
            if execute:
                r = await _aexecute_single_step(
                    item, flat_index, engine, employees, initial_args,
                    outputs_by_id, outputs_by_index, prev_output,
                    agent_identity, project_info,
                    api_key, model, ex_prompt,
                )
            else:
                r = _execute_single_step(
                    item, flat_index, engine, employees, initial_args,
                    outputs_by_id, outputs_by_index, prev_output,
                    agent_identity, project_info,
                    False, None, None, ex_prompt,
                )
            if r.step_id:
                outputs_by_id[r.step_id] = r.output
            outputs_by_index[flat_index] = r.output
            flat_index += 1
            step_results.append(r)
            prev_output = r.output

            if on_step_complete:
                checkpoint = _build_checkpoint(
                    pipeline.name, step_results, outputs_by_id, outputs_by_index, flat_index, step_i + 1,
                )
                on_step_complete(r, checkpoint)

    total_ms = int((time.monotonic() - t0) * 1000)
    all_results = _flatten_results(step_results)
    return PipelineResult(
        pipeline_name=pipeline.name,
        mode="execute" if execute else "prompt",
        steps=step_results,
        total_duration_ms=total_ms,
        total_input_tokens=sum(r.input_tokens for r in all_results),
        total_output_tokens=sum(r.output_tokens for r in all_results),
    )


async def aresume_pipeline(
    pipeline: Pipeline,
    checkpoint: dict,
    initial_args: dict[str, str] | None = None,
    project_dir: Path | None = None,
    agent_id: int | None = None,
    smart_context: bool = True,
    *,
    api_key: str | None = None,
    model: str | None = None,
    on_step_complete: Callable[[StepResult, dict], None] | None = None,
) -> PipelineResult:
    """从断点恢复执行流水线 — 跳过已完成步骤.

    Args:
        pipeline: 流水线定义.
        checkpoint: 上次保存的断点数据.
        on_step_complete: 每步完成后的回调.
    """
    initial_args = initial_args or {}
    employees = discover_employees(project_dir=project_dir)
    engine = CrewEngine(project_dir=project_dir)
    project_info = detect_project(project_dir) if smart_context else None
    agent_identity = _resolve_agent_identity(agent_id)
    _check_agent_active(agent_identity, agent_id)

    # 从 checkpoint 恢复状态
    completed_steps_data = checkpoint.get("completed_steps", [])
    outputs_by_id: dict[str, str] = {k: v for k, v in checkpoint.get("outputs_by_id", {}).items()}
    outputs_by_index: dict[int, str] = {int(k): v for k, v in checkpoint.get("outputs_by_index", {}).items()}
    next_flat_index: int = checkpoint.get("next_flat_index", 0)
    next_step_i: int = checkpoint.get("next_step_i", 0)

    # 重建已完成的 step_results
    step_results: list[StepResult | list[StepResult]] = []
    for item in completed_steps_data:
        if isinstance(item, list):
            step_results.append([StepResult(**s) for s in item])
        else:
            step_results.append(StepResult(**item))

    # 计算 prev_output
    flat_all = _flatten_results(step_results)
    prev_output = flat_all[-1].output if flat_all else ""

    flat_index = next_flat_index
    t0 = time.monotonic()

    # 从 next_step_i 开始继续执行
    for step_i in range(next_step_i, len(pipeline.steps)):
        item = pipeline.steps[step_i]

        if isinstance(item, ParallelGroup):
            tasks = []
            for sub in item.parallel:
                idx = flat_index
                ex_prompt = _resolve_exemplars(agent_id, sub.employee)
                tasks.append(
                    _aexecute_single_step(
                        sub, idx, engine, employees, initial_args,
                        outputs_by_id, outputs_by_index, prev_output,
                        agent_identity, project_info,
                        api_key, model, ex_prompt,
                    )
                )
                flat_index += 1
            group_results = list(
                await asyncio.wait_for(
                    asyncio.gather(*tasks),
                    timeout=_ASYNC_STEP_TIMEOUT,
                )
            )
            group_results.sort(key=lambda r: r.step_index)

            for r in group_results:
                if r.step_id:
                    outputs_by_id[r.step_id] = r.output
                outputs_by_index[r.step_index] = r.output
            step_results.append(group_results)
            prev_output = "\n\n---\n\n".join(r.output for r in group_results)

            if on_step_complete:
                cp = _build_checkpoint(
                    pipeline.name, step_results, outputs_by_id, outputs_by_index, flat_index, step_i + 1,
                )
                for r in group_results:
                    on_step_complete(r, cp)
        elif isinstance(item, ConditionalStep):
            body = item.condition
            took_then = _evaluate_check(
                body.check, body.contains, body.matches,
                outputs_by_id, outputs_by_index, prev_output, True,
            )
            branch_steps = body.then if took_then else body.else_
            branch_label = "then" if took_then else "else"

            branch_results: list[StepResult] = []
            for sub in branch_steps:
                ex_prompt = _resolve_exemplars(agent_id, sub.employee)
                r = await _aexecute_single_step(
                    sub, flat_index, engine, employees, initial_args,
                    outputs_by_id, outputs_by_index, prev_output,
                    agent_identity, project_info,
                    api_key, model, ex_prompt,
                )
                r.branch = branch_label
                if r.step_id:
                    outputs_by_id[r.step_id] = r.output
                outputs_by_index[flat_index] = r.output
                flat_index += 1
                branch_results.append(r)
                prev_output = r.output
                if on_step_complete:
                    cp = _build_checkpoint(
                        pipeline.name, step_results, outputs_by_id, outputs_by_index, flat_index, step_i + 1,
                    )
                    on_step_complete(r, cp)

            if branch_results:
                step_results.append(
                    branch_results if len(branch_results) > 1 else branch_results[0],
                )
        elif isinstance(item, LoopStep):
            body = item.loop
            loop_all_results: list[StepResult] = []

            for iteration in range(body.max_iterations):
                for sub in body.steps:
                    ex_prompt = _resolve_exemplars(agent_id, sub.employee)
                    r = await _aexecute_single_step(
                        sub, flat_index, engine, employees, initial_args,
                        outputs_by_id, outputs_by_index, prev_output,
                        agent_identity, project_info,
                        api_key, model, ex_prompt,
                    )
                    r.branch = f"loop-{iteration}"
                    if r.step_id:
                        outputs_by_id[r.step_id] = r.output
                    outputs_by_index[flat_index] = r.output
                    flat_index += 1
                    loop_all_results.append(r)
                    prev_output = r.output
                    if on_step_complete:
                        cp = _build_checkpoint(
                            pipeline.name, step_results, outputs_by_id, outputs_by_index, flat_index, step_i + 1,
                        )
                        on_step_complete(r, cp)

                should_stop = _evaluate_check(
                    body.until.check, body.until.contains, body.until.matches,
                    outputs_by_id, outputs_by_index, prev_output, True,
                )
                if should_stop:
                    break

            if loop_all_results:
                step_results.append(
                    loop_all_results if len(loop_all_results) > 1 else loop_all_results[0],
                )
        else:
            ex_prompt = _resolve_exemplars(agent_id, item.employee)
            r = await _aexecute_single_step(
                item, flat_index, engine, employees, initial_args,
                outputs_by_id, outputs_by_index, prev_output,
                agent_identity, project_info,
                api_key, model, ex_prompt,
            )
            if r.step_id:
                outputs_by_id[r.step_id] = r.output
            outputs_by_index[flat_index] = r.output
            flat_index += 1
            step_results.append(r)
            prev_output = r.output

            if on_step_complete:
                cp = _build_checkpoint(
                    pipeline.name, step_results, outputs_by_id, outputs_by_index, flat_index, step_i + 1,
                )
                on_step_complete(r, cp)

    total_ms = int((time.monotonic() - t0) * 1000)
    all_results = _flatten_results(step_results)
    return PipelineResult(
        pipeline_name=pipeline.name,
        mode="execute",
        steps=step_results,
        total_duration_ms=total_ms,
        total_input_tokens=sum(r.input_tokens for r in all_results),
        total_output_tokens=sum(r.output_tokens for r in all_results),
    )


async def _aexecute_single_step(
    step: PipelineStep,
    index: int,
    engine: CrewEngine,
    employees: "DiscoveryResult",
    initial_args: dict[str, str],
    outputs_by_id: dict[str, str],
    outputs_by_index: dict[int, str],
    prev_output: str,
    agent_identity: "AgentIdentity | None",
    project_info: "ProjectInfo | None",
    api_key: str | None,
    model: str | None,
    exemplar_prompt: str = "",
) -> StepResult:
    """异步执行单个步骤."""
    emp = employees.get(step.employee)
    if emp is None:
        return StepResult(
            employee=step.employee,
            step_id=step.id,
            step_index=index,
            args=step.args,
            prompt=f"[错误] 未找到员工: {step.employee}",
            error=True,
            error_message=f"未找到员工: {step.employee}",
        )

    resolved_args: dict[str, str] = {}
    for k, v in step.args.items():
        v = _resolve_initial_args(v, initial_args)
        v = _resolve_output_refs(v, outputs_by_id, outputs_by_index, prev_output, True)
        resolved_args[k] = v

    prompt = engine.prompt(
        emp, args=resolved_args,
        agent_identity=agent_identity, project_info=project_info,
        exemplar_prompt=exemplar_prompt,
    )

    try:
        from crew.executor import aexecute_prompt

        use_model = emp.model or model or "claude-sonnet-4-20250514"
        t0 = time.monotonic()
        exec_result = await aexecute_prompt(
            system_prompt=prompt, api_key=api_key, model=use_model, stream=False,
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        return StepResult(
            employee=step.employee, step_id=step.id, step_index=index,
            args=resolved_args, prompt=prompt,
            output=exec_result.content,
            model=exec_result.model,
            input_tokens=exec_result.input_tokens,
            output_tokens=exec_result.output_tokens,
            duration_ms=duration_ms,
        )
    except Exception as exc:
        return StepResult(
            employee=step.employee, step_id=step.id, step_index=index,
            args=resolved_args, prompt=prompt,
            error=True, error_message=str(exc)[:500],
        )


# ── 工具函数 ──


def _flatten_results(
    steps: list[StepResult | list[StepResult]],
) -> list[StepResult]:
    """将嵌套结果展平为一维列表."""
    flat: list[StepResult] = []
    for item in steps:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _build_checkpoint(
    pipeline_name: str,
    step_results: list[StepResult | list[StepResult]],
    outputs_by_id: dict[str, str],
    outputs_by_index: dict[int, str],
    next_flat_index: int,
    next_step_i: int,
) -> dict:
    """构建断点数据 — 包含恢复所需的完整状态."""
    completed = []
    for item in step_results:
        if isinstance(item, list):
            completed.append([r.model_dump(mode="json") for r in item])
        else:
            completed.append(item.model_dump(mode="json"))
    return {
        "pipeline_name": pipeline_name,
        "completed_steps": completed,
        "outputs_by_id": dict(outputs_by_id),
        "outputs_by_index": {str(k): v for k, v in outputs_by_index.items()},
        "next_flat_index": next_flat_index,
        "next_step_i": next_step_i,
    }


# ── 内置流水线发现 ──

PIPELINES_DIR_NAME = "pipelines"


def discover_pipelines(project_dir: Path | None = None) -> dict[str, Path]:
    """发现所有可用流水线.

    搜索顺序：
    1. 内置（src/crew/employees/pipelines/）
    2. 项目（.crew/pipelines/）
    """
    pipelines: dict[str, Path] = {}

    # 内置流水线
    builtin_dir = Path(__file__).parent / "employees" / PIPELINES_DIR_NAME
    if builtin_dir.is_dir():
        for f in sorted(builtin_dir.glob("*.yaml")):
            pipelines[f.stem] = f

    # 项目流水线（覆盖同名内置）
    from crew.paths import resolve_project_dir
    root = resolve_project_dir(project_dir)
    project_pipeline_dir = root / ".crew" / PIPELINES_DIR_NAME
    if project_pipeline_dir.is_dir():
        for f in sorted(project_pipeline_dir.glob("*.yaml")):
            pipelines[f.stem] = f

    return pipelines


# ── ActionPlan → Pipeline 转换 ──


def pipeline_from_action_plan(
    action_plan: "DiscussionActionPlan",
    employee_mapping: dict[str, str] | None = None,
) -> Pipeline:
    """从讨论行动计划自动生成 Pipeline.

    策略:
    1. 按 depends_on 拓扑排序 actions
    2. 无依赖关系的 actions 合并为 ParallelGroup
    3. 自动在最后追加 review 步骤（如有 review_criteria）

    Args:
        action_plan: 讨论产出的行动计划
        employee_mapping: 角色到员工的映射（如 {"executor": "fullstack-engineer"}）
    """
    from crew.models import DiscussionActionPlan  # noqa: F811

    employee_mapping = employee_mapping or {}

    actions = {a.id: a for a in action_plan.actions}
    if not actions:
        return Pipeline(
            name=f"plan-{action_plan.discussion_name}",
            description=f"从讨论「{action_plan.topic}」自动生成的空 Pipeline",
            steps=[],
        )

    # 拓扑排序
    in_degree: dict[str, int] = {aid: 0 for aid in actions}
    dependents: dict[str, list[str]] = {aid: [] for aid in actions}
    for aid, action in actions.items():
        for dep in action.depends_on:
            if dep in actions:
                in_degree[aid] += 1
                dependents[dep].append(aid)

    # BFS 分层
    layers: list[list[str]] = []
    queue = [aid for aid, deg in in_degree.items() if deg == 0]
    while queue:
        layers.append(sorted(queue))
        next_queue: list[str] = []
        for aid in queue:
            for dep_id in dependents[aid]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    next_queue.append(dep_id)
        queue = next_queue

    # 生成 Pipeline steps
    steps: list[PipelineStep | ParallelGroup] = []
    for layer in layers:
        layer_steps: list[PipelineStep] = []
        for aid in layer:
            action = actions[aid]
            employee = (
                action.assignee_employee
                or employee_mapping.get(action.assignee_role, "")
                or action.assignee_role
            )
            step = PipelineStep(
                employee=employee,
                id=aid,
                args={
                    "task": action.description,
                    "priority": action.priority,
                    "verification": action.verification,
                },
            )
            layer_steps.append(step)

        if len(layer_steps) == 1:
            steps.append(layer_steps[0])
        else:
            steps.append(ParallelGroup(parallel=layer_steps))

    # 追加 review 步骤
    if action_plan.review_criteria:
        criteria_text = "; ".join(action_plan.review_criteria)
        review_employee = employee_mapping.get("reviewer", "code-reviewer")
        steps.append(PipelineStep(
            employee=review_employee,
            id="review",
            args={
                "target": "{prev}",
                "criteria": criteria_text,
            },
        ))

    return Pipeline(
        name=f"plan-{action_plan.discussion_name}",
        description=f"从讨论「{action_plan.topic}」自动生成",
        steps=steps,
    )
