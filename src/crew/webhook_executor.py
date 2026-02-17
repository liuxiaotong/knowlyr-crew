"""执行引擎 — 任务调度、员工执行、工具路由、会议编排."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from crew.exceptions import EmployeeNotFoundError, PipelineNotFoundError
from crew.webhook_context import _AppContext, _MAX_TOOL_ROUNDS
from crew.webhook_tools import get_all_tool_handlers

_TOOL_HANDLERS: dict[str, Any] = get_all_tool_handlers()


async def _dispatch_task(
    ctx: _AppContext,
    trigger: str,
    target_type: str,
    target_name: str,
    args: dict[str, str],
    sync: bool = False,
    agent_id: int | None = None,
    model: str | None = None,
) -> Any:
    """创建任务并调度执行."""
    from starlette.responses import JSONResponse

    trace_id = uuid.uuid4().hex[:12]
    record = ctx.registry.create(
        trigger=trigger,
        target_type=target_type,
        target_name=target_name,
        args=args,
    )
    logger.info(
        "任务开始 [trace=%s] %s → %s/%s (task=%s)",
        trace_id, trigger, target_type, target_name, record.task_id,
    )

    # 通过 webhook 模块查找，确保 mock patch 生效
    import crew.webhook as _wh

    if sync:
        await _wh._execute_task(ctx, record.task_id, agent_id=agent_id, model=model, trace_id=trace_id)
        record = ctx.registry.get(record.task_id)
        return JSONResponse(record.model_dump(mode="json"))

    asyncio.create_task(_wh._execute_task(ctx, record.task_id, agent_id=agent_id, model=model, trace_id=trace_id))
    return JSONResponse(
        {"task_id": record.task_id, "status": "pending"},
        status_code=202,
    )


async def _execute_task(
    ctx: _AppContext,
    task_id: str,
    agent_id: int | None = None,
    model: str | None = None,
    trace_id: str = "",
) -> None:
    """执行任务."""
    record = ctx.registry.get(task_id)
    if record is None:
        return

    ctx.registry.update(task_id, "running")

    # 通过 webhook 模块查找，确保 mock patch 生效
    import crew.webhook as _wh

    try:
        if record.target_type == "pipeline":
            logger.info("执行 pipeline [trace=%s] %s", trace_id, record.target_name)
            result = await _wh._execute_pipeline(
                ctx, record.target_name, record.args, agent_id=agent_id, task_id=task_id,
            )
        elif record.target_type == "employee":
            logger.info("执行 employee [trace=%s] %s", trace_id, record.target_name)
            result = await _wh._execute_employee(
                ctx, record.target_name, record.args, agent_id=agent_id, model=model,
            )
        elif record.target_type == "meeting":
            logger.info("执行 meeting [trace=%s] %s", trace_id, record.target_name)
            employees = [e.strip() for e in record.args.get("employees", "").split(",") if e.strip()]
            result = await _wh._execute_meeting(
                ctx,
                task_id=task_id,
                employees=employees,
                topic=record.args.get("topic", ""),
                goal=record.args.get("goal", ""),
                rounds=int(record.args.get("rounds", "2")),
            )
        elif record.target_type == "chain":
            logger.info("执行 chain [trace=%s] %s", trace_id, record.target_name)
            import json as _json

            steps = _json.loads(record.args.get("steps_json", "[]"))
            result = await _wh._execute_chain(ctx, task_id, steps)
        else:
            ctx.registry.update(task_id, "failed", error=f"未知目标类型: {record.target_type}")
            return

        # 成本追踪：向结果追加 cost_usd
        if isinstance(result, dict):
            try:
                from crew.cost import enrich_result_with_cost
                enrich_result_with_cost(result)
            except Exception as e:
                logger.debug("成本追踪失败: %s", e)

        logger.info("任务完成 [trace=%s] task=%s", trace_id, task_id)
        ctx.registry.update(task_id, "completed", result=result)

        # B 类权限标记 + 质量评分 + 自动降级检查
        if record.target_type == "employee" and isinstance(result, dict):
            try:
                from crew.organization import load_organization, record_task_outcome

                org = load_organization(project_dir=ctx.project_dir)
                auth = org.get_authority(record.target_name)
                if auth == "B":
                    result["needs_kai_approval"] = True
                    result["authority_note"] = (
                        "此员工为 B 类（需 Kai 确认），"
                        "结果仅供参考，请 Kai 过目后再决定下一步。"
                    )

                # 质量评分解析
                output_text = result.get("output", "")
                if output_text:
                    from crew.cost import parse_quality_score
                    qscore = parse_quality_score(output_text)
                    if qscore:
                        result["quality_score"] = qscore

                # 自动降级检查（记录成功）
                record_task_outcome(
                    record.target_name, success=True,
                    project_dir=ctx.project_dir,
                )
            except Exception as e:
                logger.warning("任务后处理失败 (employee=%s): %s", record.target_name, e)
    except Exception as e:
        logger.exception("任务执行失败 [trace=%s]: %s", trace_id, task_id)
        ctx.registry.update(task_id, "failed", error=str(e))
        # 自动降级检查（记录失败）
        if record.target_type == "employee":
            try:
                from crew.organization import record_task_outcome
                record_task_outcome(
                    record.target_name, success=False,
                    project_dir=ctx.project_dir,
                )
            except Exception:
                pass


async def _execute_pipeline(
    ctx: _AppContext,
    name: str,
    args: dict[str, str],
    agent_id: int | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """执行 pipeline."""
    from crew.pipeline import arun_pipeline, discover_pipelines, load_pipeline

    pipelines = discover_pipelines(project_dir=ctx.project_dir)
    if name not in pipelines:
        raise PipelineNotFoundError(name)

    pipeline = load_pipeline(pipelines[name])

    # 构建 checkpoint 回调
    on_step_complete = None
    if task_id:
        def on_step_complete(step_result, checkpoint_data):
            ctx.registry.update_checkpoint(task_id, checkpoint_data)

    result = await arun_pipeline(
        pipeline,
        initial_args=args,
        project_dir=ctx.project_dir,
        execute=True,
        api_key=None,
        agent_id=agent_id,
        on_step_complete=on_step_complete,
    )
    return result.model_dump(mode="json")


async def _delegate_employee(
    ctx: _AppContext,
    employee_name: str,
    task: str,
    *,
    model: str | None = None,
) -> str:
    """执行被委派的员工（纯文本输入/输出，不支持递归委派）."""
    from crew.discovery import discover_employees
    from crew.engine import CrewEngine

    discovery = discover_employees(project_dir=ctx.project_dir)
    target = discovery.get(employee_name)
    if target is None:
        available = ", ".join(sorted(discovery.employees.keys()))
        return f"错误：未找到员工 '{employee_name}'。可用员工：{available}"

    engine = CrewEngine(project_dir=ctx.project_dir)
    prompt = engine.prompt(target, args={"task": task})

    try:
        from crew.executor import aexecute_prompt

        use_model = target.model or model or "claude-sonnet-4-20250514"
        result = await aexecute_prompt(
            system_prompt=prompt,
            user_message=task,
            api_key=None,
            model=use_model,
            stream=False,
        )
        return result.content
    except Exception as e:
        return f"委派执行失败: {e}"


async def _execute_meeting(
    ctx: _AppContext,
    task_id: str,
    employees: list[str],
    topic: str,
    goal: str = "",
    rounds: int = 2,
) -> dict[str, Any]:
    """执行多员工会议（编排式讨论）— 每轮参会者并行."""
    from crew.discussion import create_adhoc_discussion, render_discussion_plan
    from crew.executor import aexecute_prompt

    discussion = create_adhoc_discussion(
        employees=employees, topic=topic, goal=goal, rounds=rounds,
    )
    plan = render_discussion_plan(
        discussion, initial_args={}, project_dir=ctx.project_dir, smart_context=True,
    )

    all_rounds: list[dict[str, Any]] = []
    previous_rounds_text = ""

    for rp in plan.rounds:
        logger.info(
            "会议 %s 第 %d 轮 '%s' (%d 人)",
            task_id, rp.round_number, rp.name, len(rp.participant_prompts),
        )

        # 替换 {previous_rounds} 并并行执行
        coros = []
        names = []
        for pp in rp.participant_prompts:
            prompt_text = pp.prompt.replace("{previous_rounds}", previous_rounds_text)
            coros.append(aexecute_prompt(
                system_prompt=prompt_text,
                user_message="请开始。",
                api_key=None,
                model="claude-sonnet-4-20250514",
                stream=False,
            ))
            names.append(pp.employee_name)

        results = await asyncio.gather(*coros, return_exceptions=True)

        round_outputs = []
        for i, out in enumerate(results):
            content = f"[执行失败: {out}]" if isinstance(out, Exception) else out.content
            round_outputs.append({"employee": names[i], "content": content})

        all_rounds.append({
            "round_num": rp.round_number,
            "name": rp.name,
            "outputs": round_outputs,
        })

        # 积累上下文
        parts = [f"**{o['employee']}**: {o['content']}" for o in round_outputs]
        previous_rounds_text += f"\n\n## 第 {rp.round_number} 轮: {rp.name}\n" + "\n\n".join(parts)

    # 综合结论
    synthesis_prompt = plan.synthesis_prompt.replace("{previous_rounds}", previous_rounds_text)
    synthesis = await aexecute_prompt(
        system_prompt=synthesis_prompt,
        user_message="请综合以上讨论，给出最终结论。",
        api_key=None,
        model="claude-sonnet-4-20250514",
        stream=False,
    )

    return {
        "rounds": all_rounds,
        "synthesis": synthesis.content,
    }


async def _execute_chain(
    ctx: _AppContext,
    task_id: str,
    steps: list[dict[str, str]],
) -> dict[str, Any]:
    """按顺序执行委派链，前一步结果传给下一步."""
    from crew.discovery import discover_employees
    from crew.engine import CrewEngine
    from crew.executor import aexecute_prompt

    discovery = discover_employees(project_dir=ctx.project_dir)
    engine = CrewEngine(project_dir=ctx.project_dir)

    prev_output = ""
    step_results: list[dict[str, str]] = []

    for i, step in enumerate(steps):
        emp_name = step["employee_name"]
        task_desc = step["task"].replace("{prev}", prev_output)

        target = discovery.get(emp_name)
        if target is None:
            step_results.append({"employee": emp_name, "error": f"未找到员工 '{emp_name}'"})
            break

        logger.info("Chain %s 步骤 %d/%d: %s", task_id, i + 1, len(steps), emp_name)
        prompt = engine.prompt(target, args={"task": task_desc})
        use_model = target.model or "claude-sonnet-4-20250514"

        try:
            result = await aexecute_prompt(
                system_prompt=prompt,
                user_message=task_desc,
                api_key=None,
                model=use_model,
                stream=False,
            )
            prev_output = result.content
            step_results.append({"employee": emp_name, "content": result.content})
        except Exception as e:
            step_results.append({"employee": emp_name, "error": str(e)})
            break

    return {"steps": step_results, "final_output": prev_output}


async def _execute_employee_with_tools(
    ctx: _AppContext,
    name: str,
    args: dict[str, str],
    *,
    agent_id: int | None = None,
    model: str | None = None,
    user_message: "str | list[dict[str, Any]] | None" = None,
    message_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """执行带工具的员工（agent loop with tools）."""
    from crew.discovery import discover_employees
    from crew.engine import CrewEngine
    from crew.executor import aexecute_with_tools
    from crew.providers import Provider, detect_provider
    from crew.tool_schema import (
        AGENT_TOOLS, employee_tools_to_schemas,
        get_tool_schema, is_finish_tool, _make_load_tools_schema,
    )

    discovery = discover_employees(project_dir=ctx.project_dir)
    match = discovery.get(name)
    if match is None:
        raise EmployeeNotFoundError(name)

    # agent 身份
    agent_identity = None
    if agent_id:
        try:
            from crew.id_client import afetch_agent_identity
            agent_identity = await afetch_agent_identity(agent_id)
        except Exception as e:
            logger.debug("agent 身份获取失败 (agent_id=%s): %s", agent_id, e)

    engine = CrewEngine(project_dir=ctx.project_dir)
    prompt = engine.prompt(match, args=args, agent_identity=agent_identity)

    # 如果有 delegate 工具，追加同事名单（按组织架构分组）
    if "delegate" in (match.tools or []):
        from crew.organization import load_organization, get_effective_authority

        org = load_organization(project_dir=ctx.project_dir)

        team_rosters: dict[str, list[str]] = {}
        ungrouped: list[str] = []

        for emp_name, emp in discovery.employees.items():
            if emp_name == name:
                continue
            label = emp.character_name or emp.effective_display_name
            auth = get_effective_authority(org, emp_name, project_dir=ctx.project_dir) or "?"
            line = f"- {emp_name}（{label}，{auth}类）：{emp.description}"
            team_id = org.get_team(emp_name)
            if team_id:
                team_rosters.setdefault(team_id, []).append(line)
            else:
                ungrouped.append(line)

        sections: list[str] = []
        for tid, lines in team_rosters.items():
            team_def = org.teams.get(tid)
            team_label = team_def.label if team_def else tid
            sections.append(f"### {team_label}\n" + "\n".join(lines))
        if ungrouped:
            sections.append("### 其他\n" + "\n".join(ungrouped))

        authority_note = (
            "\n\n权限级别说明：A类=自主执行可直接交付，"
            "B类=需 Kai 确认后才能决定下一步，"
            "C类=简单任务直接做/复杂任务需确认。"
        )

        if sections:
            prompt += (
                "\n\n---\n\n## 可委派的同事\n\n"
                "使用 delegate/delegate_async/delegate_chain/route 工具调用他们。\n"
                + authority_note
                + "\n\n"
                + "\n\n".join(sections)
            )

    from crew.permission import PermissionGuard

    guard = PermissionGuard(match)

    # 从 employee 的 tools 列表中筛选 agent tools
    agent_tool_names = [t for t in (match.tools or []) if t in AGENT_TOOLS]
    tool_schemas, deferred_names = employee_tools_to_schemas(agent_tool_names)
    loaded_deferred: set[str] = set()  # 已加载的延迟工具

    use_model = match.model or model or "claude-sonnet-4-20250514"
    provider = detect_provider(use_model)
    # base_url 强制走 OpenAI 兼容路径，消息格式也要对应
    is_anthropic = provider == Provider.ANTHROPIC and not match.base_url

    # 构建消息列表（含历史对话）
    messages: list[dict[str, Any]] = []
    if message_history:
        for h in message_history:
            messages.append({"role": h["role"], "content": h["content"]})

    task_text = user_message or args.get("task", "请开始执行上述任务。")
    messages.append({"role": "user", "content": task_text})

    total_input = 0
    total_output = 0
    final_content = ""
    rounds = 0

    # 解析 agent_id（从 match 的 agent_id 属性，或参数）
    effective_agent_id = agent_id or getattr(match, "agent_id", None)

    import crew.webhook as _wh
    _max_rounds = getattr(_wh, "_MAX_TOOL_ROUNDS", _MAX_TOOL_ROUNDS)
    for rounds in range(_max_rounds):  # noqa: B007
        result = await aexecute_with_tools(
            system_prompt=prompt,
            messages=messages,
            tools=tool_schemas,
            api_key=match.api_key or None,
            model=use_model,
            max_tokens=4096,
            base_url=match.base_url or None,
            fallback_model=match.fallback_model or None,
            fallback_api_key=match.fallback_api_key or None,
            fallback_base_url=match.fallback_base_url or None,
        )
        total_input += result.input_tokens
        total_output += result.output_tokens

        if not result.has_tool_calls:
            final_content = result.content
            break

        # ── 处理 tool calls ──
        if is_anthropic:
            assistant_content: list[dict[str, Any]] = []
            if result.content:
                assistant_content.append({"type": "text", "text": result.content})
            for tc in result.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            finished = False
            for tc in result.tool_calls:
                if tc.name == "load_tools":
                    # ── 延迟加载工具 ──
                    requested = {n.strip() for n in tc.arguments.get("names", "").split(",") if n.strip()}
                    newly = []
                    for tn in sorted(requested):
                        if tn in deferred_names and tn not in loaded_deferred:
                            schema = get_tool_schema(tn)
                            if schema:
                                tool_schemas.append(schema)
                                loaded_deferred.add(tn)
                                newly.append(tn)
                    remaining = deferred_names - loaded_deferred
                    if not remaining:
                        tool_schemas = [s for s in tool_schemas if s["name"] != "load_tools"]
                    else:
                        for s in tool_schemas:
                            if s["name"] == "load_tools":
                                s["description"] = f"加载额外工具后才能调用。可用: {', '.join(sorted(remaining))}"
                    load_msg = f"已加载: {', '.join(newly)}。现在可以直接调用这些工具。" if newly else "这些工具已加载。"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": load_msg,
                    })
                    continue
                tool_output = await _handle_tool_call(
                    ctx, name, tc.name, tc.arguments, effective_agent_id, guard=guard,
                )
                if tool_output is None:
                    # finish tool
                    final_content = tc.arguments.get("result", result.content)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": final_content,
                    })
                    finished = True
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": tool_output[:10000],
                    })
            messages.append({"role": "user", "content": tool_results})
            if finished:
                break
        else:
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": result.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": __import__("json").dumps(
                                tc.arguments, ensure_ascii=False
                            ),
                        },
                    }
                    for tc in result.tool_calls
                ],
            }
            messages.append(assistant_msg)

            finished = False
            for tc in result.tool_calls:
                if tc.name == "load_tools":
                    # ── 延迟加载工具 ──
                    requested = {n.strip() for n in tc.arguments.get("names", "").split(",") if n.strip()}
                    newly = []
                    for tn in sorted(requested):
                        if tn in deferred_names and tn not in loaded_deferred:
                            schema = get_tool_schema(tn)
                            if schema:
                                tool_schemas.append(schema)
                                loaded_deferred.add(tn)
                                newly.append(tn)
                    remaining = deferred_names - loaded_deferred
                    if not remaining:
                        tool_schemas = [s for s in tool_schemas if s["name"] != "load_tools"]
                    else:
                        for s in tool_schemas:
                            if s["name"] == "load_tools":
                                s["description"] = f"加载额外工具后才能调用。可用: {', '.join(sorted(remaining))}"
                    load_msg = f"已加载: {', '.join(newly)}。现在可以直接调用这些工具。" if newly else "这些工具已加载。"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": load_msg,
                    })
                    continue
                tool_output = await _handle_tool_call(
                    ctx, name, tc.name, tc.arguments, effective_agent_id, guard=guard,
                )
                if tool_output is None:
                    final_content = tc.arguments.get("result", result.content)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": final_content,
                    })
                    finished = True
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_output[:10000],
                    })
            if finished:
                break
    else:
        final_content = result.content or "达到最大工具调用轮次限制。"

    return {
        "employee": name,
        "prompt": prompt[:500],
        "output": final_content,
        "model": use_model,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "tool_rounds": rounds,
        "base_url": match.base_url or "",
    }


async def _handle_tool_call(
    ctx: _AppContext,
    employee_name: str,
    tool_name: str,
    arguments: dict[str, Any],
    agent_id: int | None,
    guard: Any | None = None,
) -> str | None:
    """处理单个 tool call，返回结果字符串。返回 None 表示 finish tool."""
    from crew.tool_schema import is_finish_tool

    if is_finish_tool(tool_name):
        return None

    # 权限检查
    if guard is not None:
        denied_msg = guard.check_soft(tool_name)
        if denied_msg:
            logger.warning("权限拒绝: %s.%s", employee_name, tool_name)
            return f"[权限拒绝] {denied_msg}"

    if tool_name == "add_memory":
        from crew.memory import MemoryStore
        project_dir = ctx.project_dir if ctx else Path(".")
        store = MemoryStore(project_dir=project_dir)
        entry = store.add(
            employee=employee_name,
            category=arguments.get("category", "finding"),
            content=arguments.get("content", ""),
            source_session="",
        )
        logger.info("记忆保存: %s → %s", employee_name, entry.content[:60])
        return "已记住。"

    if tool_name == "delegate":
        logger.info("委派: %s → %s", employee_name, arguments.get("employee_name"))
        import crew.webhook as _wh
        return await _wh._delegate_employee(
            ctx,
            arguments.get("employee_name", ""),
            arguments.get("task", ""),
        )

    handler = _TOOL_HANDLERS.get(tool_name)
    if handler:
        try:
            logger.info("工具调用: %s.%s(%s)", employee_name, tool_name, list(arguments.keys()))
            return await handler(arguments, agent_id=agent_id, ctx=ctx)
        except Exception as e:
            logger.warning("工具 %s 执行失败: %s", tool_name, e)
            return f"工具执行失败: {e}"

    return f"工具 '{tool_name}' 不可用。"


async def _execute_employee(
    ctx: _AppContext,
    name: str,
    args: dict[str, str],
    agent_id: int | None = None,
    model: str | None = None,
    user_message: "str | list[dict[str, Any]] | None" = None,
    message_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """执行单个员工."""
    from crew.discovery import discover_employees
    from crew.engine import CrewEngine

    discovery = discover_employees(project_dir=ctx.project_dir)
    match = discovery.get(name)

    if match is None:
        raise EmployeeNotFoundError(name)

    # 如果员工有 agent tools，使用带工具的 agent loop
    from crew.tool_schema import AGENT_TOOLS

    if any(t in AGENT_TOOLS for t in (match.tools or [])):
        import crew.webhook as _wh
        return await _wh._execute_employee_with_tools(
            ctx, name, args, agent_id=agent_id, model=model,
            user_message=user_message,
            message_history=message_history,
        )

    # 获取 agent 身份
    agent_identity = None
    if agent_id:
        try:
            from crew.id_client import afetch_agent_identity
            agent_identity = await afetch_agent_identity(agent_id)
        except Exception as e:
            logger.debug("agent 身份获取失败 (agent_id=%s): %s", agent_id, e)

    engine = CrewEngine(project_dir=ctx.project_dir)
    prompt = engine.prompt(match, args=args, agent_identity=agent_identity)

    # 尝试执行 LLM 调用（executor 自动从环境变量解析 API key）
    try:
        from crew.executor import aexecute_prompt

        use_model = match.model or model or "claude-sonnet-4-20250514"
        exec_kwargs = dict(
            system_prompt=prompt,
            api_key=match.api_key or None,
            model=use_model,
            stream=False,
            base_url=match.base_url or None,
            fallback_model=match.fallback_model or None,
            fallback_api_key=match.fallback_api_key or None,
            fallback_base_url=match.fallback_base_url or None,
        )
        if user_message:
            exec_kwargs["user_message"] = user_message
        result = await aexecute_prompt(**exec_kwargs)
    except (ValueError, ImportError):
        result = None

    if result is not None:
        return {
            "employee": name,
            "prompt": prompt,
            "output": result.content,
            "model": result.model,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "base_url": match.base_url or "",
        }

    return {"employee": name, "prompt": prompt, "output": ""}


async def _stream_employee(
    ctx: _AppContext,
    name: str,
    args: dict[str, str],
    agent_id: int | None = None,
    model: str | None = None,
) -> Any:
    """SSE 流式执行单个员工."""
    import json as _json

    from starlette.responses import StreamingResponse

    def _dumps(obj: Any) -> str:
        return _json.dumps(obj, ensure_ascii=False)

    from crew.discovery import discover_employees
    from crew.engine import CrewEngine

    result = discover_employees(project_dir=ctx.project_dir)
    match = result.get(name)

    if match is None:
        async def _error():
            yield f"event: error\ndata: {_dumps({'error': f'未找到员工: {name}'})}\n\n"
        return StreamingResponse(_error(), media_type="text/event-stream")

    # 获取 agent 身份
    agent_identity = None
    if agent_id:
        try:
            from crew.id_client import afetch_agent_identity

            agent_identity = await afetch_agent_identity(agent_id)
        except Exception as e:
            logger.debug("agent 身份获取失败 (agent_id=%s): %s", agent_id, e)

    engine = CrewEngine(project_dir=ctx.project_dir)
    prompt = engine.prompt(match, args=args, agent_identity=agent_identity)

    async def _generate():
        done_sent = False
        try:
            from crew.executor import aexecute_prompt

            use_model = match.model or model or "claude-sonnet-4-20250514"
            stream_iter = await asyncio.wait_for(
                aexecute_prompt(
                    system_prompt=prompt,
                    api_key=match.api_key or None,
                    model=use_model,
                    stream=True,
                    base_url=match.base_url or None,
                    fallback_model=match.fallback_model or None,
                    fallback_api_key=match.fallback_api_key or None,
                    fallback_base_url=match.fallback_base_url or None,
                ),
                timeout=300,
            )

            async for chunk in stream_iter:
                yield f"data: {_dumps({'token': chunk})}\n\n"

            # 流结束后发送完整的 result
            result = getattr(stream_iter, "result", None)
            if result:
                yield f"event: done\ndata: {_dumps({'employee': name, 'model': result.model, 'input_tokens': result.input_tokens, 'output_tokens': result.output_tokens})}\n\n"
            else:
                yield f"event: done\ndata: {_dumps({'employee': name})}\n\n"
            done_sent = True
        except asyncio.TimeoutError:
            yield f"event: error\ndata: {_dumps({'error': 'stream timeout (300s)'})}\n\n"
            done_sent = True
        except Exception as exc:
            yield f"event: error\ndata: {_dumps({'error': str(exc)[:500]})}\n\n"
            done_sent = True
        finally:
            if not done_sent:
                yield f"event: error\ndata: {_dumps({'error': 'stream interrupted'})}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


async def _resume_incomplete_pipelines(ctx: _AppContext) -> None:
    """恢复未完成的 pipeline 任务（服务重启后）."""
    for record in ctx.registry.list_recent(n=100):
        if record.status != "running" or record.target_type != "pipeline" or not record.checkpoint:
            continue

        checkpoint = record.checkpoint
        pipeline_name = checkpoint.get("pipeline_name", record.target_name)
        logger.info("恢复 pipeline 任务: %s (task=%s)", pipeline_name, record.task_id)

        try:
            from crew.pipeline import aresume_pipeline, discover_pipelines, load_pipeline

            pipelines = discover_pipelines(project_dir=ctx.project_dir)
            if pipeline_name not in pipelines:
                ctx.registry.update(record.task_id, "failed", error=f"恢复失败: 未找到 pipeline {pipeline_name}")
                continue

            pipeline = load_pipeline(pipelines[pipeline_name])

            def _make_callback(tid):
                def cb(step_result, checkpoint_data):
                    ctx.registry.update_checkpoint(tid, checkpoint_data)
                return cb

            result = await aresume_pipeline(
                pipeline,
                checkpoint=checkpoint,
                initial_args=record.args,
                project_dir=ctx.project_dir,
                api_key=None,
                on_step_complete=_make_callback(record.task_id),
            )
            ctx.registry.update(record.task_id, "completed", result=result.model_dump(mode="json"))
        except Exception as e:
            logger.exception("恢复任务失败: %s", record.task_id)
            ctx.registry.update(record.task_id, "failed", error=f"恢复失败: {e}")
