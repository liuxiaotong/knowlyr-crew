"""执行引擎 — 任务调度、员工执行、工具路由、会议编排."""

from __future__ import annotations

import asyncio
import json
import logging
import time as _time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from crew.exceptions import EmployeeNotFoundError, PipelineNotFoundError
from crew.webhook_context import _MAX_TOOL_ROUNDS, _AppContext
from crew.webhook_tools import get_all_tool_handlers

_TOOL_HANDLERS: dict[str, Any] = get_all_tool_handlers()

# 后台任务引用集合 — 防止 GC 提前回收 + 异常日志
_background_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]


def _task_done_callback(task: asyncio.Task) -> None:  # type: ignore[type-arg]
    """后台 task 完成回调：记录异常日志 + 从引用集合移除."""
    _background_tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("后台任务异常: %s", exc, exc_info=exc)


async def _dispatch_task(
    ctx: _AppContext,
    trigger: str,
    target_type: str,
    target_name: str,
    args: dict[str, str],
    sync: bool = False,
    agent_id: str | None = None,
    model: str | None = None,
    owner: str | None = None,
    tenant_id: str | None = None,
) -> Any:
    """创建任务并调度执行."""
    from starlette.responses import JSONResponse

    trace_id = uuid.uuid4().hex[:12]
    record = ctx.registry.create(
        trigger=trigger,
        target_type=target_type,
        target_name=target_name,
        args=args,
        owner=owner,
    )
    logger.info(
        "任务开始 [trace=%s] %s → %s/%s (task=%s)",
        trace_id,
        trigger,
        target_type,
        target_name,
        record.task_id,
    )

    # 通过 webhook 模块查找，确保 mock patch 生效
    import crew.webhook as _wh

    if sync:
        await _wh._execute_task(
            ctx,
            record.task_id,
            agent_id=agent_id,
            model=model,
            trace_id=trace_id,
            tenant_id=tenant_id,
        )
        record = ctx.registry.get(record.task_id)
        return JSONResponse(record.model_dump(mode="json"))

    task = asyncio.create_task(
        _wh._execute_task(
            ctx,
            record.task_id,
            agent_id=agent_id,
            model=model,
            trace_id=trace_id,
            tenant_id=tenant_id,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_task_done_callback)
    return JSONResponse(
        {"task_id": record.task_id, "status": "pending"},
        status_code=202,
    )


async def _execute_task(
    ctx: _AppContext,
    task_id: str,
    agent_id: str | None = None,
    model: str | None = None,
    trace_id: str = "",
    tenant_id: str | None = None,
) -> None:
    """执行任务."""
    record = ctx.registry.get(task_id)
    if record is None:
        return

    ctx.registry.update(task_id, "running")

    # ── 轨迹录制 ──
    from contextlib import ExitStack

    from crew.trajectory import TrajectoryCollector

    _exit_stack = ExitStack()
    _traj_collector = TrajectoryCollector.try_create_for_employee(
        record.target_name,
        record.args.get("task", "") or record.target_name,
        channel="delegate",
        project_dir=ctx.project_dir,
    )
    if _traj_collector is not None:
        _exit_stack.enter_context(_traj_collector)

    # 通过 webhook 模块查找，确保 mock patch 生效
    import crew.webhook as _wh

    _t0 = _time.monotonic()
    try:
        if record.target_type == "pipeline":
            logger.info("执行 pipeline [trace=%s] %s", trace_id, record.target_name)
            result = await _wh._execute_pipeline(
                ctx,
                record.target_name,
                record.args,
                agent_id=agent_id,
                task_id=task_id,
            )
        elif record.target_type == "employee":
            logger.info("执行 employee [trace=%s] %s", trace_id, record.target_name)
            result = await _wh._execute_employee(
                ctx,
                record.target_name,
                record.args,
                agent_id=agent_id,
                model=model,
                tenant_id=tenant_id,
            )
        elif record.target_type == "meeting":
            logger.info("执行 meeting [trace=%s] %s", trace_id, record.target_name)
            employees = [
                e.strip() for e in record.args.get("employees", "").split(",") if e.strip()
            ]
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
            result = await _wh._execute_chain(ctx, task_id, steps, tenant_id=tenant_id)
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

        # 完成轨迹录制（成功）
        if _traj_collector is not None:
            try:
                _traj_collector.finish(success=True)
            except Exception as _te:
                logger.debug("delegate 轨迹录制失败: %s", _te)
            finally:
                _exit_stack.close()

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
                        "此员工为 B 类（需 Kai 确认），结果仅供参考，请 Kai 过目后再决定下一步。"
                    )

                # 质量评分解析
                output_text = result.get("output", "")
                if output_text:
                    from crew.cost import parse_quality_score

                    qscore = parse_quality_score(output_text)
                    if qscore:
                        result["quality_score"] = qscore

                # 自动记忆保存（opt-in）+ 自检提取（独立于 auto_memory）
                if output_text and len(output_text) > 50:
                    try:
                        from crew.discovery import discover_employees

                        disc = discover_employees(project_dir=ctx.project_dir, tenant_id=tenant_id)
                        match = disc.get(record.target_name)
                        task_desc = record.args.get("task", "")[:100]
                        _mem_store = None

                        if match and getattr(match, "auto_memory", False):
                            from crew.memory import get_memory_store

                            _mem_store = get_memory_store(
                                project_dir=ctx.project_dir, tenant_id=tenant_id
                            )
                            summary = output_text[:300].strip()
                            if len(output_text) > 300:
                                summary += "..."
                            _mem_store.add(
                                employee=record.target_name,
                                category="finding",
                                content=f"[任务] {task_desc} → {summary}",
                                source_session=record.task_id
                                if hasattr(record, "task_id")
                                else task_id,
                                confidence=0.6,
                                ttl_days=30,
                            )
                            logger.info("自动记忆保存: %s", record.target_name)

                        # 自检摘要提取 — 独立于 auto_memory，有自检段就提取
                        import re as _re

                        check_match = _re.search(
                            r"##\s*完成后自检[^\n]*\n+((?:- \[.\].*\n?)+)",
                            output_text,
                        )
                        if check_match:
                            if _mem_store is None:
                                from crew.memory import get_memory_store

                                _mem_store = get_memory_store(
                                    project_dir=ctx.project_dir, tenant_id=tenant_id
                                )
                            check_lines = check_match.group(1).strip().split("\n")
                            passed = []
                            failed = []
                            for cl in check_lines:
                                cl = cl.strip()
                                if cl.startswith("- [x]") or cl.startswith("- [X]"):
                                    passed.append(cl[5:].strip())
                                elif cl.startswith("- [ ]"):
                                    failed.append(cl[5:].strip())
                            parts = [f"[自检] {task_desc}"]
                            if passed:
                                parts.append(f"通过: {'; '.join(passed)}")
                            if failed:
                                parts.append(f"待改进: {'; '.join(failed)}")
                            _mem_store.add(
                                employee=record.target_name,
                                category="correction",
                                content=" | ".join(parts),
                                source_session=record.task_id
                                if hasattr(record, "task_id")
                                else task_id,
                                confidence=0.7,
                                ttl_days=60,
                                shared=True,
                            )
                            logger.info("自检摘要保存: %s", record.target_name)

                        # 工作模式（pattern）自动提取
                        pattern_match = _re.search(
                            r"##\s*工作模式[^\n]*\n+((?:- .+\n?)+)",
                            output_text,
                        )
                        if pattern_match:
                            if _mem_store is None:
                                from crew.memory import get_memory_store

                                _mem_store = get_memory_store(
                                    project_dir=ctx.project_dir, tenant_id=tenant_id
                                )
                            pattern_block = pattern_match.group(1).strip()
                            # 解析结构化字段
                            p_name = ""
                            p_trigger = ""
                            p_steps = ""
                            p_roles: list[str] = []
                            for pl in pattern_block.split("\n"):
                                pl = pl.strip().lstrip("- ")
                                if pl.startswith("模式名称:"):
                                    p_name = pl.split(":", 1)[1].strip()
                                elif pl.startswith("触发条件:"):
                                    p_trigger = pl.split(":", 1)[1].strip()
                                elif pl.startswith("步骤:"):
                                    p_steps = pl.split(":", 1)[1].strip()
                                elif pl.startswith("适用角色:"):
                                    p_roles = [
                                        r.strip()
                                        for r in pl.split(":", 1)[1].split(",")
                                        if r.strip()
                                    ]
                            if p_name:
                                content = f"[模式] {p_name}"
                                if p_steps:
                                    content += f" — {p_steps}"
                                _mem_store.add(
                                    employee=record.target_name,
                                    category="pattern",
                                    content=content,
                                    source_session=record.task_id
                                    if hasattr(record, "task_id")
                                    else task_id,
                                    confidence=0.7,
                                    trigger_condition=p_trigger,
                                    applicability=p_roles,
                                    origin_employee=record.target_name,
                                )
                                logger.info("工作模式保存: %s → %s", record.target_name, p_name)
                    except Exception as e_mem:
                        logger.debug("自动记忆/自检/模式保存失败: %s", e_mem)

                # 自动降级检查（记录成功）
                record_task_outcome(
                    record.target_name,
                    success=True,
                    project_dir=ctx.project_dir,
                )
            except Exception as e:
                logger.warning("任务后处理失败 (employee=%s): %s", record.target_name, e)
    except Exception as e:
        # 完成轨迹录制（失败）
        if _traj_collector is not None:
            try:
                _traj_collector.finish(success=False)
            except Exception:
                pass
            finally:
                _exit_stack.close()

        logger.exception("任务执行失败 [trace=%s]: %s", trace_id, task_id)
        ctx.registry.update(task_id, "failed", error=str(e))
        # 自动降级检查（记录失败）
        if record.target_type == "employee":
            try:
                from crew.organization import record_task_outcome

                record_task_outcome(
                    record.target_name,
                    success=False,
                    project_dir=ctx.project_dir,
                )
            except Exception:
                pass


async def _execute_pipeline(
    ctx: _AppContext,
    name: str,
    args: dict[str, str],
    agent_id: str | None = None,
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
    tenant_id: str | None = None,
) -> str:
    """执行被委派的员工（纯文本输入/输出，不支持递归委派）."""
    from crew.discovery import discover_employees
    from crew.engine import CrewEngine

    discovery = discover_employees(project_dir=ctx.project_dir, tenant_id=tenant_id)
    target = discovery.get(employee_name)
    if target is None:
        available = ", ".join(sorted(discovery.employees.keys()))
        return f"错误：未找到员工 '{employee_name}'。可用员工：{available}"

    engine = CrewEngine(project_dir=ctx.project_dir, tenant_id=tenant_id)
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
        employees=employees,
        topic=topic,
        goal=goal,
        rounds=rounds,
    )
    plan = render_discussion_plan(
        discussion,
        initial_args={},
        project_dir=ctx.project_dir,
        smart_context=True,
    )

    all_rounds: list[dict[str, Any]] = []
    previous_rounds_text = ""

    for rp in plan.rounds:
        logger.info(
            "会议 %s 第 %d 轮 '%s' (%d 人)",
            task_id,
            rp.round_number,
            rp.name,
            len(rp.participant_prompts),
        )

        # 替换 {previous_rounds} 并并行执行
        coros = []
        names = []
        for pp in rp.participant_prompts:
            prompt_text = pp.prompt.replace("{previous_rounds}", previous_rounds_text)
            coros.append(
                aexecute_prompt(
                    system_prompt=prompt_text,
                    user_message="请开始。",
                    api_key=None,
                    model="claude-sonnet-4-20250514",
                    stream=False,
                )
            )
            names.append(pp.employee_name)

        results = await asyncio.gather(*coros, return_exceptions=True)

        round_outputs = []
        for i, out in enumerate(results):
            content = f"[执行失败: {out}]" if isinstance(out, Exception) else out.content
            round_outputs.append({"employee": names[i], "content": content})

        all_rounds.append(
            {
                "round_num": rp.round_number,
                "name": rp.name,
                "outputs": round_outputs,
            }
        )

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
    *,
    start_index: int = 0,
    prev_output: str = "",
    step_results: list[dict[str, str]] | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """按顺序执行委派链，前一步结果传给下一步.

    支持审批检查点: 遇到 approval 步骤时暂停链执行，保存断点，
    通知 Kai 审批。批准后通过 _resume_chain 从断点恢复。
    """
    from crew.discovery import discover_employees
    from crew.engine import CrewEngine
    from crew.executor import aexecute_prompt

    discovery = discover_employees(project_dir=ctx.project_dir, tenant_id=tenant_id)
    engine = CrewEngine(project_dir=ctx.project_dir, tenant_id=tenant_id)

    if step_results is None:
        step_results = []

    for i in range(start_index, len(steps)):
        step = steps[i]
        emp_name = step["employee_name"]

        # 审批检查点: 暂停链执行
        if step.get("approval") and i > start_index:
            checkpoint = {
                "chain_step": i,
                "steps_json": json.dumps(steps, ensure_ascii=False),
                "prev_output": prev_output,
                "step_results": step_results,
            }
            ctx.registry.update_checkpoint(task_id, checkpoint)
            ctx.registry.update(task_id, "awaiting_approval")
            await _notify_approval_needed(ctx, task_id, step, prev_output)
            return {
                "steps": step_results,
                "status": "awaiting_approval",
                "pending_step": emp_name,
                "message": f"步骤 {i + 1} ({emp_name}) 需要审批，已通知 Kai",
            }

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


async def _notify_approval_needed(
    ctx: _AppContext,
    task_id: str,
    step: dict,
    prev_output: str,
) -> None:
    """通过飞书私聊通知 Kai 有步骤等待审批."""
    if not (ctx.feishu_token_mgr and ctx.feishu_config):
        logger.info("审批通知跳过: 飞书未配置 (task=%s)", task_id)
        return
    owner_id = ctx.feishu_config.owner_open_id
    if not owner_id:
        logger.info("审批通知跳过: owner_open_id 未配置 (task=%s)", task_id)
        return

    emp_name = step.get("employee_name", "?")
    task_text = step.get("task", "")
    role = task_text.split("]")[0].lstrip("[") if "]" in task_text else emp_name
    summary = prev_output[:300] if prev_output else "（无前序输出）"

    text = (
        f"📋 任务 {task_id} 等待审批\n\n"
        f"下一步: {emp_name}（{role}）\n"
        f"前序结果摘要:\n{summary}\n\n"
        f"回复「approve {task_id}」批准\n"
        f"回复「reject {task_id}」拒绝"
    )

    try:
        import json as _json

        from crew.feishu import get_feishu_client

        token = await ctx.feishu_token_mgr.get_token()
        client = get_feishu_client()
        resp = await client.post(
            "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "receive_id": owner_id,
                "msg_type": "text",
                "content": _json.dumps({"text": text}),
            },
            timeout=15.0,
        )
        data = resp.json()
        if data.get("code") != 0:
            logger.warning("审批通知发送失败 (task=%s): %s", task_id, data.get("msg", ""))
    except Exception as e:
        logger.warning("审批通知发送异常 (task=%s): %s", task_id, e)


async def _resume_chain(ctx: _AppContext, task_id: str, *, tenant_id: str | None = None) -> None:
    """从审批检查点恢复链执行."""
    record = ctx.registry.get(task_id)
    if not record or record.status != "awaiting_approval":
        return
    if not record.checkpoint:
        ctx.registry.update(task_id, "failed", error="无断点数据")
        return

    cp = record.checkpoint
    steps = json.loads(cp["steps_json"])
    start_index = cp["chain_step"]
    prev_output = cp["prev_output"]
    step_results = cp.get("step_results", [])

    ctx.registry.update(task_id, "running")

    try:
        result = await _execute_chain(
            ctx,
            task_id,
            steps,
            start_index=start_index,
            prev_output=prev_output,
            step_results=step_results,
            tenant_id=tenant_id,
        )
        # 如果又遇到审批检查点，_execute_chain 已处理，不需要再 update
        if result.get("status") == "awaiting_approval":
            return
        ctx.registry.update(task_id, "completed", result=result)
    except Exception as e:
        logger.exception("恢复链执行失败 [task=%s]: %s", task_id, e)
        ctx.registry.update(task_id, "failed", error=str(e))


def _process_load_tools(
    arguments: dict[str, Any],
    deferred_names: set[str],
    loaded_deferred: set[str],
    tool_schemas: list[dict[str, Any]],
) -> str:
    """处理 load_tools 请求的公共逻辑（Anthropic/OpenAI 格式共用）.

    就地修改 loaded_deferred 和 tool_schemas，返回结果消息。
    """
    from crew.tool_schema import SKILL_PACKS, _make_load_tools_schema, get_tool_schema

    requested = {n.strip() for n in arguments.get("names", "").split(",") if n.strip()}
    # 展开技能包名为工具名
    expanded: set[str] = set()
    for rn in requested:
        if rn in SKILL_PACKS:
            expanded |= SKILL_PACKS[rn]["tools"]
        else:
            expanded.add(rn)
    newly = []
    for tn in sorted(expanded):
        if tn in deferred_names and tn not in loaded_deferred:
            schema = get_tool_schema(tn)
            if schema:
                tool_schemas.append(schema)
                loaded_deferred.add(tn)
                newly.append(tn)
    remaining = deferred_names - loaded_deferred
    if not remaining:
        tool_schemas[:] = [s for s in tool_schemas if s["name"] != "load_tools"]
    else:
        new_load_schema = _make_load_tools_schema(remaining)
        for s in tool_schemas:
            if s["name"] == "load_tools":
                s["description"] = new_load_schema["description"]
    return (
        f"已加载: {', '.join(newly)}。现在可以直接调用这些工具。" if newly else "这些工具已加载。"
    )


async def _execute_employee_with_tools(
    ctx: _AppContext,
    name: str,
    args: dict[str, str],
    *,
    agent_id: str | None = None,
    model: str | None = None,
    user_message: str | list[dict[str, Any]] | None = None,
    message_history: list[dict[str, Any]] | None = None,
    sender_id: str | None = None,
    attachments: list[dict[str, Any]] | None = None,
    sender_type: str = "",
    channel: str = "",
    tenant_id: str | None = None,
    is_admin: bool = False,
) -> dict[str, Any]:
    """执行带工具的员工（agent loop with tools）."""
    from crew.discovery import discover_employees
    from crew.engine import CrewEngine
    from crew.executor import aexecute_with_tools
    from crew.providers import Provider, detect_provider
    from crew.tool_schema import (
        AGENT_TOOLS,
        employee_tools_to_schemas,
    )

    discovery = discover_employees(project_dir=ctx.project_dir, tenant_id=tenant_id)
    match = discovery.get(name)
    if match is None:
        raise EmployeeNotFoundError(name)

    # 冻结检查
    if match.agent_status == "frozen":
        logger.info("员工 %s 已冻结，跳过执行", name)
        return {
            "output": f"员工 {match.character_name or name} 已冻结，无法执行任务",
            "skipped": True,
        }

    engine = CrewEngine(project_dir=ctx.project_dir, tenant_id=tenant_id)
    # 从 args 中提取 _max_visibility（飞书 dispatch 传入）
    max_visibility = args.pop("_max_visibility", "open") if isinstance(args, dict) else "open"

    # 信息分级：计算有效许可
    _cls_kwargs: dict = {}
    if channel:
        from crew.classification import get_effective_clearance

        _clearance = get_effective_clearance(match.name, channel, sender_type=sender_type)
        _cls_kwargs["classification_max"] = _clearance["classification_max"]
        _cls_kwargs["allowed_domains"] = _clearance["allowed_domains"]
        _cls_kwargs["include_confidential"] = _clearance["include_confidential"]

    prompt = engine.prompt(match, args=args, max_visibility=max_visibility, **_cls_kwargs)

    # 如果有 delegate 工具，追加同事名单（按组织架构分组）
    if "delegate" in (match.tools or []):
        from crew.organization import get_effective_authority, load_organization

        org = load_organization(project_dir=ctx.project_dir)

        # 紧凑名单格式 — 省掉描述（employee name 已自描述），每组一行
        team_members: dict[str, list[str]] = {}
        ungrouped: list[str] = []

        for emp_name, emp in discovery.employees.items():
            if emp_name == name:
                continue
            label = emp.character_name or emp.effective_display_name
            auth = get_effective_authority(org, emp_name, project_dir=ctx.project_dir) or "?"
            tag = f"{emp_name}({label},{auth})"
            team_id = org.get_team(emp_name)
            if team_id:
                team_members.setdefault(team_id, []).append(tag)
            else:
                ungrouped.append(tag)

        sections: list[str] = []
        for tid, members in team_members.items():
            team_def = org.teams.get(tid)
            team_label = team_def.label if team_def else tid
            sections.append(f"**{team_label}**: {' '.join(members)}")
        if ungrouped:
            sections.append(f"**其他**: {' '.join(ungrouped)}")

        if sections:
            prompt += (
                "\n\n---\n\n## 可委派的同事\n\n"
                "A=自主执行 B=需Kai确认 C=看场景。"
                "用 delegate/delegate_async/delegate_chain/route 调用。\n\n" + "\n".join(sections)
            )

    from crew.permission import PermissionGuard

    # 管理员判断：仅信任服务端 tenant 认证结果，不信任用户输入的 sender_id
    if is_admin:
        # 管理员模式：允许所有工具
        from crew.tool_schema import AGENT_TOOLS

        class AdminGuard:
            """管理员权限守卫 - 允许所有工具."""

            def __init__(self):
                self.employee_name = match.name
                self.allowed = AGENT_TOOLS | {"submit", "finish", "load_tools"}

            def check(self, tool_name: str) -> None:
                pass  # 管理员不检查权限

            def check_soft(self, tool_name: str) -> str | None:
                return None  # 管理员不检查权限

        guard = AdminGuard()
    else:
        guard = PermissionGuard(match)

    # 从 employee 的 tools 列表中筛选 agent tools
    agent_tool_names = [t for t in (match.tools or []) if t in AGENT_TOOLS]
    tool_schemas, deferred_names = employee_tools_to_schemas(agent_tool_names)
    loaded_deferred: set[str] = set()  # 已加载的延迟工具

    use_model = model or match.model or "claude-sonnet-4-20250514"
    provider = detect_provider(use_model)
    # base_url 强制走 OpenAI 兼容路径，消息格式也要对应
    is_anthropic = provider == Provider.ANTHROPIC and not match.base_url

    # 构建消息列表（含历史对话）
    messages: list[dict[str, Any]] = []
    if message_history:
        for h in message_history:
            # 历史消息可能包含附件
            hist_attachments = h.get("attachments")
            if hist_attachments and any(
                att.get("type", "").startswith("image/") for att in hist_attachments
            ):
                # 转换为 content blocks 格式
                content_blocks: list[dict[str, Any]] = [{"type": "text", "text": h["content"]}]
                for att in hist_attachments:
                    if att.get("type", "").startswith("image/"):
                        content_blocks.append(
                            {"type": "image", "source": {"type": "url", "url": att["url"]}}
                        )
                messages.append({"role": h["role"], "content": content_blocks})
            else:
                messages.append({"role": h["role"], "content": h["content"]})

    task_text = user_message or args.get("task", "请开始执行上述任务。")

    # 如果当前消息有附件，转换为 content blocks 格式
    if attachments and any(att.get("type", "").startswith("image/") for att in attachments):
        content_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": task_text if isinstance(task_text, str) else str(task_text)}
        ]
        for att in attachments:
            if att.get("type", "").startswith("image/"):
                content_blocks.append(
                    {"type": "image", "source": {"type": "url", "url": att["url"]}}
                )
        messages.append({"role": "user", "content": content_blocks})
    else:
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
            max_tokens=200000,
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
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            finished = False
            for tc in result.tool_calls:
                if tc.name == "load_tools":
                    load_msg = _process_load_tools(
                        tc.arguments,
                        deferred_names,
                        loaded_deferred,
                        tool_schemas,
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": load_msg,
                        }
                    )
                    continue
                tool_output = await _handle_tool_call(
                    ctx,
                    name,
                    tc.name,
                    tc.arguments,
                    effective_agent_id,
                    guard=guard,
                    max_visibility=max_visibility,
                    push_event_fn=None,
                    target_user_id=sender_id or "",
                    tenant_id=tenant_id,
                )
                if tool_output is None:
                    # finish tool
                    final_content = tc.arguments.get("result", result.content)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": final_content,
                        }
                    )
                    finished = True
                else:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": tool_output[:10000],
                        }
                    )
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
                            "arguments": __import__("json").dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in result.tool_calls
                ],
            }
            messages.append(assistant_msg)

            finished = False
            for tc in result.tool_calls:
                if tc.name == "load_tools":
                    load_msg = _process_load_tools(
                        tc.arguments,
                        deferred_names,
                        loaded_deferred,
                        tool_schemas,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": load_msg,
                        }
                    )
                    continue
                tool_output = await _handle_tool_call(
                    ctx,
                    name,
                    tc.name,
                    tc.arguments,
                    effective_agent_id,
                    guard=guard,
                    max_visibility=max_visibility,
                    push_event_fn=None,
                    target_user_id=sender_id or "",
                    tenant_id=tenant_id,
                )
                if tool_output is None:
                    final_content = tc.arguments.get("result", result.content)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": final_content,
                        }
                    )
                    finished = True
                else:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_output[:10000],
                        }
                    )
            if finished:
                break
    else:
        final_content = result.content or "达到最大工具调用轮次限制。"

    # ── 轨迹录制: 工具循环汇总步骤 ──
    try:
        from crew.trajectory import TrajectoryCollector

        _tc = TrajectoryCollector.current()
        if _tc is not None:
            _tc.add_tool_step(
                thought=f"[tool-loop] {rounds + 1} rounds",
                tool_name="agent_loop",
                tool_params={"employee": name, "rounds": rounds + 1},
                tool_output=final_content[:2000],
                tool_exit_code=0,
                input_tokens=total_input,
                output_tokens=total_output,
            )
    except Exception:
        pass

    # 清洗内部标签（<thinking>、工具调用 XML 等）
    from crew.output_sanitizer import strip_internal_tags

    final_content = strip_internal_tags(final_content)

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


async def _stream_employee_with_tools(
    ctx: _AppContext,
    name: str,
    args: dict[str, str],
    *,
    agent_id: str | None = None,
    model: str | None = None,
    user_message: str | list[dict[str, Any]] | None = None,
    message_history: list[dict[str, Any]] | None = None,
    sender_id: str | None = None,
    attachments: list[dict[str, Any]] | None = None,
    sender_type: str = "",
    channel: str = "",
    tenant_id: str | None = None,
    is_admin: bool = False,
) -> Any:
    """流式版 agent loop — 每一轮都用原生流式 API，逐 token 输出.

    Yields: {"delta": str, "done": False} 和最终 {"done": True, ...}
    """
    import json as _json

    from crew.discovery import discover_employees
    from crew.engine import CrewEngine
    from crew.executor import _resolve_key_for_context, aexecute_with_tools
    from crew.models import ToolCall
    from crew.providers import Provider, detect_provider
    from crew.tool_schema import (
        AGENT_TOOLS,
        employee_tools_to_schemas,
    )

    discovery = discover_employees(project_dir=ctx.project_dir, tenant_id=tenant_id)
    match = discovery.get(name)
    if match is None:
        raise EmployeeNotFoundError(name)

    if match.agent_status == "frozen":
        yield {"delta": f"员工 {match.character_name or name} 已冻结，无法执行任务", "done": False}
        yield {"done": True, "employee_id": name, "tokens_used": 0, "latency_ms": 0}
        return

    engine = CrewEngine(project_dir=ctx.project_dir, tenant_id=tenant_id)
    max_visibility = args.pop("_max_visibility", "open") if isinstance(args, dict) else "open"

    # 信息分级：计算有效许可
    _cls_kwargs: dict = {}
    if channel:
        from crew.classification import get_effective_clearance

        _clearance = get_effective_clearance(match.name, channel, sender_type=sender_type)
        _cls_kwargs["classification_max"] = _clearance["classification_max"]
        _cls_kwargs["allowed_domains"] = _clearance["allowed_domains"]
        _cls_kwargs["include_confidential"] = _clearance["include_confidential"]

    prompt = engine.prompt(match, args=args, max_visibility=max_visibility, **_cls_kwargs)

    # delegate 同事名单（同 _execute_employee_with_tools 逻辑）
    if "delegate" in (match.tools or []):
        from crew.organization import get_effective_authority, load_organization

        org = load_organization(project_dir=ctx.project_dir)
        team_members: dict[str, list[str]] = {}
        ungrouped: list[str] = []
        for emp_name, emp in discovery.employees.items():
            if emp_name == name:
                continue
            label = emp.character_name or emp.effective_display_name
            auth = get_effective_authority(org, emp_name, project_dir=ctx.project_dir) or "?"
            tag = f"{emp_name}({label},{auth})"
            team_id = org.get_team(emp_name)
            if team_id:
                team_members.setdefault(team_id, []).append(tag)
            else:
                ungrouped.append(tag)
        sections: list[str] = []
        for tid, members in team_members.items():
            team_def = org.teams.get(tid)
            team_label = team_def.label if team_def else tid
            sections.append(f"**{team_label}**: {' '.join(members)}")
        if ungrouped:
            sections.append(f"**其他**: {' '.join(ungrouped)}")
        if sections:
            prompt += (
                "\n\n---\n\n## 可委派的同事\n\n"
                "A=自主执行 B=需Kai确认 C=看场景。"
                "用 delegate/delegate_async/delegate_chain/route 调用。\n\n" + "\n".join(sections)
            )

    from crew.permission import PermissionGuard

    # 管理员判断：仅信任服务端 tenant 认证结果，不信任用户输入的 sender_id
    if is_admin:
        from crew.tool_schema import AGENT_TOOLS as _AT

        class AdminGuard:
            def __init__(self):
                self.employee_name = match.name
                self.allowed = _AT | {"submit", "finish", "load_tools"}

            def check(self, tool_name: str) -> None:
                pass

            def check_soft(self, tool_name: str) -> str | None:
                return None

        guard = AdminGuard()
    else:
        guard = PermissionGuard(match)

    agent_tool_names = [t for t in (match.tools or []) if t in AGENT_TOOLS]
    tool_schemas, deferred_names = employee_tools_to_schemas(agent_tool_names)
    loaded_deferred: set[str] = set()

    use_model = model or match.model or "claude-sonnet-4-20250514"
    provider = detect_provider(use_model)
    # base_url 指定时走 OpenAI 兼容路径，不走 Anthropic 原生 SDK
    is_anthropic = provider == Provider.ANTHROPIC and not match.base_url

    messages: list[dict[str, Any]] = []
    if message_history:
        for h in message_history:
            hist_attachments = h.get("attachments")
            if hist_attachments and any(
                att.get("type", "").startswith("image/") for att in hist_attachments
            ):
                content_blocks: list[dict[str, Any]] = [{"type": "text", "text": h["content"]}]
                for att in hist_attachments:
                    if att.get("type", "").startswith("image/"):
                        content_blocks.append(
                            {"type": "image", "source": {"type": "url", "url": att["url"]}}
                        )
                messages.append({"role": h["role"], "content": content_blocks})
            else:
                messages.append({"role": h["role"], "content": h["content"]})

    task_text = user_message or args.get("task", "请开始执行上述任务。")
    if attachments and any(att.get("type", "").startswith("image/") for att in attachments):
        content_blocks_user: list[dict[str, Any]] = [
            {"type": "text", "text": task_text if isinstance(task_text, str) else str(task_text)}
        ]
        for att in attachments:
            if att.get("type", "").startswith("image/"):
                content_blocks_user.append(
                    {"type": "image", "source": {"type": "url", "url": att["url"]}}
                )
        messages.append({"role": "user", "content": content_blocks_user})
    else:
        messages.append({"role": "user", "content": task_text})

    total_input = 0
    total_output = 0
    effective_agent_id = agent_id or getattr(match, "agent_id", None)

    import crew.webhook as _wh

    _max_rounds = getattr(_wh, "_MAX_TOOL_ROUNDS", _MAX_TOOL_ROUNDS)

    # ── 非 Anthropic provider: 降级到非流式 agent loop ──
    if not is_anthropic:
        async for chunk in _stream_employee_with_tools_fallback(
            ctx=ctx,
            name=name,
            prompt=prompt,
            messages=messages,
            tool_schemas=tool_schemas,
            deferred_names=deferred_names,
            loaded_deferred=loaded_deferred,
            match=match,
            use_model=use_model,
            effective_agent_id=effective_agent_id,
            guard=guard,
            max_visibility=max_visibility,
            max_rounds=_max_rounds,
            sender_id=sender_id,
            tenant_id=tenant_id,
        ):
            yield chunk
        return

    # ── Anthropic 原生流式 agent loop ──
    from crew.executor import _get_anthropic

    anthropic = _get_anthropic()
    if anthropic is None:
        raise ImportError("anthropic SDK 未安装。请运行: pip install knowlyr-crew[execute]")

    resolved_key = _resolve_key_for_context(provider, match.api_key or None, match.base_url)
    client_kwargs = {"api_key": resolved_key}
    if match.base_url:
        # Anthropic SDK 会自动加 /v1/messages，去掉 base_url 末尾的 /v1 避免重复
        base = match.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        client_kwargs["base_url"] = base
    client = anthropic.AsyncAnthropic(**client_kwargs)

    for round_idx in range(_max_rounds):  # noqa: B007
        # ── 每一轮都用流式 API ──
        text_parts: list[str] = []
        tool_calls_collected: list[dict[str, Any]] = []  # {id, name, input_json_parts}
        current_tool_idx = -1
        stop_reason = "end_turn"

        try:
            async with client.messages.stream(
                model=use_model,
                max_tokens=200000,
                system=prompt,
                messages=messages,
                tools=tool_schemas,
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "text":
                            pass  # 文本块开始，等 delta
                        elif event.content_block.type == "tool_use":
                            # 工具调用块开始 — 收集 id 和 name
                            tool_calls_collected.append(
                                {
                                    "id": event.content_block.id,
                                    "name": event.content_block.name,
                                    "input_json_parts": [],
                                }
                            )
                            current_tool_idx = len(tool_calls_collected) - 1

                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            # 文本增量 → 立即 yield 给用户
                            text_parts.append(event.delta.text)
                            yield {"delta": event.delta.text, "done": False}
                        elif event.delta.type == "input_json_delta":
                            # 工具参数增量 → 静默收集
                            if current_tool_idx >= 0:
                                tool_calls_collected[current_tool_idx]["input_json_parts"].append(
                                    event.delta.partial_json
                                )

                    elif event.type == "message_delta":
                        stop_reason = event.delta.stop_reason or "end_turn"

                # 流结束，获取 final message 以拿到准确的 usage
                final_msg = await stream.get_final_message()

            total_input += final_msg.usage.input_tokens
            total_output += final_msg.usage.output_tokens

        except Exception as _stream_err:
            logger.error("Anthropic 流式调用失败 (round %d): %s", round_idx, _stream_err)
            # 降级：用非流式执行
            try:
                result = await aexecute_with_tools(
                    system_prompt=prompt,
                    messages=messages,
                    tools=tool_schemas,
                    api_key=match.api_key or None,
                    model=use_model,
                    max_tokens=200000,
                    base_url=match.base_url or None,
                    fallback_model=match.fallback_model or None,
                    fallback_api_key=match.fallback_api_key or None,
                    fallback_base_url=match.fallback_base_url or None,
                )
                total_input += result.input_tokens
                total_output += result.output_tokens
                if result.content:
                    for chunk in _split_text_chunks(result.content, 20):
                        yield {"delta": chunk, "done": False}
                if not result.has_tool_calls:
                    from crew.output_sanitizer import strip_internal_tags

                    yield {
                        "done": True,
                        "employee_id": name,
                        "tokens_used": total_input + total_output,
                        "latency_ms": 0,
                    }
                    return
                # 有 tool_calls — 构造数据继续后续处理
                text_parts = [result.content] if result.content else []
                tool_calls_collected = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "input_json_parts": [_json.dumps(tc.arguments, ensure_ascii=False)],
                    }
                    for tc in result.tool_calls
                ]
                stop_reason = "tool_use"
            except Exception as _fallback_err:
                logger.error("非流式降级也失败: %s", _fallback_err)
                yield {"delta": f"执行出错: {_fallback_err}", "done": False}
                yield {
                    "done": True,
                    "employee_id": name,
                    "tokens_used": total_input + total_output,
                    "latency_ms": 0,
                }
                return

        # ── 这一轮结束，检查是否需要调工具 ──
        if stop_reason != "tool_use" or not tool_calls_collected:
            # 纯文本回复，结束
            from crew.output_sanitizer import strip_internal_tags

            # text_parts 已在流式中 yield 过了，这里只需要发 done
            # 但需要对完整内容做清洗检查（内部标签已在流式中输出，无法撤回，仅做记录）
            yield {
                "done": True,
                "employee_id": name,
                "tokens_used": total_input + total_output,
                "latency_ms": 0,
            }
            return

        # ── 有工具调用 — 解析收集到的 tool calls ──
        parsed_tool_calls: list[ToolCall] = []
        for tc_raw in tool_calls_collected:
            input_json_str = "".join(tc_raw["input_json_parts"])
            try:
                tc_args = _json.loads(input_json_str) if input_json_str else {}
            except _json.JSONDecodeError:
                tc_args = {}
            parsed_tool_calls.append(
                ToolCall(
                    id=tc_raw["id"],
                    name=tc_raw["name"],
                    arguments=tc_args if isinstance(tc_args, dict) else {},
                )
            )

        # 构建 assistant message（Anthropic 格式）
        assistant_content: list[dict[str, Any]] = []
        full_text = "".join(text_parts)
        if full_text:
            assistant_content.append({"type": "text", "text": full_text})
        for tc in parsed_tool_calls:
            assistant_content.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
            )
        messages.append({"role": "assistant", "content": assistant_content})

        # ── 执行工具 ──
        tool_results: list[dict[str, Any]] = []
        finished = False
        final_content = ""
        for tc in parsed_tool_calls:
            # yield 工具执行提示
            tool_display = tc.name
            yield {"delta": "", "done": False, "tool_call": True, "tool_name": tool_display}

            if tc.name == "load_tools":
                load_msg = _process_load_tools(
                    tc.arguments, deferred_names, loaded_deferred, tool_schemas
                )
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tc.id, "content": load_msg}
                )
                continue
            tool_output = await _handle_tool_call(
                ctx,
                name,
                tc.name,
                tc.arguments,
                effective_agent_id,
                guard=guard,
                max_visibility=max_visibility,
                push_event_fn=None,
                target_user_id=sender_id or "",
                tenant_id=tenant_id,
            )
            if tool_output is None:
                final_content = tc.arguments.get("result", full_text)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tc.id, "content": final_content}
                )
                finished = True
            else:
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tc.id, "content": tool_output[:10000]}
                )
        messages.append({"role": "user", "content": tool_results})
        if finished:
            from crew.output_sanitizer import strip_internal_tags

            final_content = strip_internal_tags(final_content)
            # finish tool 被调用 — yield 完整结果
            for chunk in _split_text_chunks(final_content, 20):
                yield {"delta": chunk, "done": False}
            yield {
                "done": True,
                "employee_id": name,
                "tokens_used": total_input + total_output,
                "latency_ms": 0,
            }
            return

    # 超过最大轮次
    yield {"delta": "达到最大工具调用轮次限制。", "done": False}
    yield {
        "done": True,
        "employee_id": name,
        "tokens_used": total_input + total_output,
        "latency_ms": 0,
    }


async def _stream_employee_with_tools_fallback(
    *,
    ctx: _AppContext,
    name: str,
    prompt: str,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    deferred_names: set[str],
    loaded_deferred: set[str],
    match: Any,
    use_model: str,
    effective_agent_id: str | None,
    guard: Any,
    max_visibility: str,
    max_rounds: int,
    sender_id: str | None = None,
    tenant_id: str | None = None,
) -> Any:
    """非 Anthropic provider 的降级 agent loop — 非流式执行，结果分块输出.

    Yields: {"delta": str, "done": False} 和最终 {"done": True, ...}
    """
    import json as _json

    from crew.executor import aexecute_with_tools

    total_input = 0
    total_output = 0

    yield {"delta": "", "done": False}  # 立即发一个 chunk 告诉前端已开始

    for rounds in range(max_rounds):  # noqa: B007
        result = await aexecute_with_tools(
            system_prompt=prompt,
            messages=messages,
            tools=tool_schemas,
            api_key=match.api_key or None,
            model=use_model,
            max_tokens=200000,
            base_url=match.base_url or None,
            fallback_model=match.fallback_model or None,
            fallback_api_key=match.fallback_api_key or None,
            fallback_base_url=match.fallback_base_url or None,
        )
        total_input += result.input_tokens
        total_output += result.output_tokens

        if not result.has_tool_calls:
            from crew.output_sanitizer import strip_internal_tags

            final_content = strip_internal_tags(result.content)
            for chunk in _split_text_chunks(final_content, 20):
                yield {"delta": chunk, "done": False}
            yield {
                "done": True,
                "employee_id": name,
                "tokens_used": total_input + total_output,
                "latency_ms": 0,
            }
            return

        # 中间轮：处理 tool calls（OpenAI 格式）
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": result.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": _json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in result.tool_calls
            ],
        }
        messages.append(assistant_msg)

        finished = False
        final_content = ""
        for tc in result.tool_calls:
            if tc.name == "load_tools":
                load_msg = _process_load_tools(
                    tc.arguments, deferred_names, loaded_deferred, tool_schemas
                )
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": load_msg})
                continue
            tool_output = await _handle_tool_call(
                ctx,
                name,
                tc.name,
                tc.arguments,
                effective_agent_id,
                guard=guard,
                max_visibility=max_visibility,
                push_event_fn=None,
                target_user_id=sender_id or "",
                tenant_id=tenant_id,
            )
            if tool_output is None:
                final_content = tc.arguments.get("result", result.content)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": final_content})
                finished = True
            else:
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": tool_output[:10000]}
                )
        if finished:
            from crew.output_sanitizer import strip_internal_tags

            final_content = strip_internal_tags(final_content)
            for chunk in _split_text_chunks(final_content, 20):
                yield {"delta": chunk, "done": False}
            yield {
                "done": True,
                "employee_id": name,
                "tokens_used": total_input + total_output,
                "latency_ms": 0,
            }
            return

    # 超过最大轮次
    yield {"delta": "达到最大工具调用轮次限制。", "done": False}
    yield {
        "done": True,
        "employee_id": name,
        "tokens_used": total_input + total_output,
        "latency_ms": 0,
    }


def _split_text_chunks(text: str, chunk_size: int = 20) -> list[str]:
    """将文本按固定长度拆分为块（用于非流式 fallback）."""
    if not text:
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


async def _handle_tool_call(
    ctx: _AppContext,
    employee_name: str,
    tool_name: str,
    arguments: dict[str, Any],
    agent_id: str | None,
    guard: Any | None = None,
    max_visibility: str = "open",
    push_event_fn: Any = None,
    target_user_id: str = "",
    tenant_id: str | None = None,
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

    # 敏感工具权限确认（仅在非管理员模式下）
    # 管理员模式下 guard 是 AdminGuard，跳过权限确认
    if guard is not None and guard.__class__.__name__ != "AdminGuard":
        from crew.permission_request import SENSITIVE_TOOLS, PermissionManager

        if tool_name in SENSITIVE_TOOLS:
            logger.info("检测到敏感工具: %s.%s，请求用户确认", employee_name, tool_name)
            manager = PermissionManager()
            approved = await manager.request_permission(
                tool_name=tool_name,
                tool_params=arguments,
                push_event_fn=push_event_fn,
                target_user_id=target_user_id,
            )
            if not approved:
                logger.warning("用户拒绝执行: %s.%s", employee_name, tool_name)
                return f"[用户拒绝] 您拒绝了执行 {tool_name} 操作"

    if tool_name == "query_memory":
        import json as _json

        from crew.memory import get_memory_store

        project_dir = ctx.project_dir if ctx else Path(".")
        store = get_memory_store(project_dir=project_dir)

        _keywords = arguments.get("keywords")
        _cross_employee = arguments.get("cross_employee", False)
        _cat = arguments.get("category")
        _limit = arguments.get("limit", 20)
        _emp = arguments.get("employee", employee_name)

        if _cross_employee and _keywords and hasattr(store, "query_cross_employee"):
            entries = store.query_cross_employee(
                keywords=_keywords,
                exclude_employee=_emp,
                limit=_limit,
                category=_cat,
            )
        elif _keywords and hasattr(store, "query_by_keywords"):
            entries = store.query_by_keywords(
                employee=_emp,
                keywords=_keywords,
                limit=_limit,
                category=_cat,
            )
        else:
            _query_kwargs: dict[str, Any] = {
                "employee": _emp,
                "limit": _limit,
            }
            if _cat:
                _query_kwargs["category"] = _cat
            _cls_max = arguments.get("classification_max")
            if _cls_max:
                _query_kwargs["classification_max"] = _cls_max
            entries = store.query(**_query_kwargs)

        data = [e.model_dump() if hasattr(e, "model_dump") else e for e in entries]
        logger.info("query_memory: %s → %d 条", _emp, len(data))
        return _json.dumps(data, ensure_ascii=False, default=str)

    if tool_name == "add_memory":
        # 统一走记忆管线，确保 Reflect/Connect 逻辑一致（skip_reflect 跳过 LLM 提炼）
        from crew.memory import get_memory_store
        from crew.memory_pipeline import process_memory

        project_dir = ctx.project_dir if ctx else Path(".")
        store = get_memory_store(project_dir=project_dir)
        default_vis = "private" if max_visibility == "private" else "open"
        entry = process_memory(
            raw_text=arguments.get("content", ""),
            employee=employee_name,
            store=store,
            skip_reflect=True,
            category=arguments.get("category", "finding"),
            tags=arguments.get("tags"),
            visibility=arguments.get("visibility", default_vis),
            trigger_condition=arguments.get("trigger_condition", ""),
            applicability=arguments.get("applicability"),
            origin_employee=arguments.get("origin_employee", ""),
            classification=arguments.get("classification", "internal"),
            domain=arguments.get("domain"),
        )
        if entry:
            logger.info(
                "记忆管线保存: %s → %s (visibility=%s, classification=%s)",
                employee_name,
                entry.content[:60],
                getattr(entry, 'visibility', 'open'),
                getattr(entry, 'classification', 'internal'),
            )
            return "已记住。"
        else:
            return "记忆写入被管线跳过（可能重复）。"

    if tool_name == "track_decision":
        from datetime import datetime, timedelta

        from crew.evaluation import EvaluationEngine

        project_dir = ctx.project_dir if ctx else Path(".")
        # 默认 deadline: 7 天后
        deadline = arguments.get("deadline", "")
        if not deadline:
            deadline = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

        engine = EvaluationEngine(project_dir=project_dir)
        decision = engine.track(
            employee=employee_name,
            category=arguments.get("category", "recommendation"),
            content=arguments.get("content", ""),
            expected_outcome=arguments.get("expected_outcome", ""),
            deadline=deadline,
        )
        logger.info(
            "决策记录: %s → %s (deadline=%s)", employee_name, decision.content[:60], deadline
        )
        return f"已记录决策（ID: {decision.id}，截止: {deadline}）。系统会在到期后自动评估。"

    if tool_name == "delegate":
        logger.info("委派: %s → %s", employee_name, arguments.get("employee_name"))
        import crew.webhook as _wh

        return await _wh._delegate_employee(
            ctx,
            arguments.get("employee_name", ""),
            arguments.get("task", ""),
            tenant_id=tenant_id,
        )

    # 注入 _max_visibility 到 read_notes / create_note 工具
    if tool_name in ("read_notes", "create_note"):
        arguments.setdefault("_max_visibility", max_visibility)
        if tool_name == "create_note" and max_visibility == "private":
            arguments.setdefault("visibility", "private")

    handler = _TOOL_HANDLERS.get(tool_name)
    if handler:
        try:
            logger.info("工具调用: %s.%s(%s)", employee_name, tool_name, list(arguments.keys()))
            # MCP 工具需要额外透传 tenant_id 和 user_id
            if tool_name.startswith("mcp__"):
                return await handler(
                    arguments,
                    agent_id=agent_id,
                    ctx=ctx,
                    tenant_id=tenant_id,
                    user_id=target_user_id or None,
                )
            return await handler(arguments, agent_id=agent_id, ctx=ctx)
        except Exception as e:
            logger.warning("工具 %s 执行失败: %s", tool_name, e)
            return f"工具执行失败: {e}"

    return f"工具 '{tool_name}' 不可用。"


async def _execute_employee(
    ctx: _AppContext,
    name: str,
    args: dict[str, str],
    agent_id: str | None = None,
    model: str | None = None,
    user_message: str | list[dict[str, Any]] | None = None,
    message_history: list[dict[str, Any]] | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """执行单个员工."""
    from crew.discovery import discover_employees
    from crew.engine import CrewEngine

    discovery = discover_employees(project_dir=ctx.project_dir, tenant_id=tenant_id)
    match = discovery.get(name)

    if match is None:
        raise EmployeeNotFoundError(name)

    # 冻结检查：冻结的员工不执行任务
    if match.agent_status == "frozen":
        logger.info("员工 %s 已冻结，跳过执行", name)
        return {
            "output": f"员工 {match.character_name or name} 已冻结，无法执行任务",
            "skipped": True,
        }

    # 如果员工有 agent tools，使用带工具的 agent loop
    from crew.tool_schema import AGENT_TOOLS

    if any(t in AGENT_TOOLS for t in (match.tools or [])):
        import crew.webhook as _wh

        return await _wh._execute_employee_with_tools(
            ctx,
            name,
            args,
            agent_id=agent_id,
            model=model,
            user_message=user_message,
            message_history=message_history,
            tenant_id=tenant_id,
        )

    engine = CrewEngine(project_dir=ctx.project_dir, tenant_id=tenant_id)
    # 从 args 中提取 _max_visibility（飞书 dispatch 传入）
    max_visibility = args.pop("_max_visibility", "open") if isinstance(args, dict) else "open"
    prompt = engine.prompt(match, args=args, max_visibility=max_visibility)

    # 尝试执行 LLM 调用（executor 自动从环境变量解析 API key）
    try:
        from crew.executor import aexecute_prompt

        use_model = model or match.model or "claude-sonnet-4-20250514"
        exec_kwargs = {
            "system_prompt": prompt,
            "api_key": match.api_key or None,
            "model": use_model,
            "stream": False,
            "base_url": match.base_url or None,
            "fallback_model": match.fallback_model or None,
            "fallback_api_key": match.fallback_api_key or None,
            "fallback_base_url": match.fallback_base_url or None,
        }
        if user_message:
            exec_kwargs["user_message"] = user_message
        result = await aexecute_prompt(**exec_kwargs)
    except (ValueError, ImportError):
        result = None

    if result is not None:
        # 清洗内部标签（<thinking>、工具调用 XML 等）
        from crew.output_sanitizer import strip_internal_tags

        cleaned_output = strip_internal_tags(result.content)
        return {
            "employee": name,
            "prompt": prompt,
            "output": cleaned_output,
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
    agent_id: str | None = None,
    model: str | None = None,
    tenant_id: str | None = None,
) -> Any:
    """SSE 流式执行单个员工."""
    import json as _json

    from starlette.responses import StreamingResponse

    def _dumps(obj: Any) -> str:
        return _json.dumps(obj, ensure_ascii=False)

    from crew.discovery import discover_employees
    from crew.engine import CrewEngine

    result = discover_employees(project_dir=ctx.project_dir, tenant_id=tenant_id)
    match = result.get(name)

    if match is None:

        async def _error():
            yield f"event: error\ndata: {_dumps({'error': f'未找到员工: {name}'})}\n\n"

        return StreamingResponse(_error(), media_type="text/event-stream")

    engine = CrewEngine(project_dir=ctx.project_dir, tenant_id=tenant_id)
    prompt = engine.prompt(match, args=args)

    async def _generate():
        done_sent = False
        try:
            from crew.executor import aexecute_prompt

            use_model = model or match.model or "claude-sonnet-4-20250514"
            stream_iter = await asyncio.wait_for(
                aexecute_prompt(
                    system_prompt=prompt,
                    api_key=match.api_key or None,
                    model=use_model,
                    stream=True,
                    full_events=True,  # 启用完整事件流
                    base_url=match.base_url or None,
                    fallback_model=match.fallback_model or None,
                    fallback_api_key=match.fallback_api_key or None,
                    fallback_base_url=match.fallback_base_url or None,
                ),
                timeout=300,
            )

            async for event in stream_iter:
                # 判断是字符串（旧格式）还是字典（新格式）
                if isinstance(event, str):
                    # 向后兼容：纯文本流
                    yield f"data: {_dumps({'token': event})}\n\n"
                elif isinstance(event, dict):
                    # 完整事件流
                    event_type = event.get("type")
                    if event_type == "content_block_start":
                        content_type = event.get("content_type")
                        if content_type == "thinking":
                            yield f"event: thinking_start\ndata: {_dumps({'index': event.get('index', 0)})}\n\n"
                        elif content_type == "tool_use":
                            yield f"event: tool_use_start\ndata: {_dumps({'index': event.get('index', 0), 'tool_name': event.get('tool_name'), 'tool_use_id': event.get('tool_use_id')})}\n\n"
                        elif content_type == "text":
                            yield f"event: text_start\ndata: {_dumps({'index': event.get('index', 0)})}\n\n"

                    elif event_type == "content_block_delta":
                        content_type = event.get("content_type")
                        if content_type == "thinking":
                            yield f"event: thinking_delta\ndata: {_dumps({'thinking': event.get('thinking', '')})}\n\n"
                        elif content_type == "tool_use":
                            yield f"event: tool_input_delta\ndata: {_dumps({'tool_input': event.get('tool_input', '')})}\n\n"
                        elif content_type == "text":
                            yield f"event: text_delta\ndata: {_dumps({'text': event.get('text', '')})}\n\n"

                    elif event_type == "content_block_stop":
                        yield f"event: content_block_stop\ndata: {_dumps({'index': event.get('index', 0)})}\n\n"

                    elif event_type == "message_delta":
                        if "stop_reason" in event:
                            yield f"event: message_delta\ndata: {_dumps({'stop_reason': event.get('stop_reason')})}\n\n"

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
                ctx.registry.update(
                    record.task_id, "failed", error=f"恢复失败: 未找到 pipeline {pipeline_name}"
                )
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
