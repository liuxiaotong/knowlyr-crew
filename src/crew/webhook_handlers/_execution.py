"""执行与运行处理器."""

from __future__ import annotations

import asyncio
from typing import Any

from crew.webhook_handlers._common import (
    JSONResponse,
    StreamingResponse,
    _AppContext,
    _background_tasks,
    _error_response,
    _extract_task_description,
    _parse_sender_name,
    _require_admin_token,
    _task_done_callback,
    _tenant_id_for_config,
    _tenant_id_for_store,
    get_current_tenant,
    get_memory_store,
    logger,
)


async def _handle_github(request: Any, ctx: _AppContext) -> Any:
    """处理 GitHub webhook."""

    from crew.webhook_config import (
        match_route,
        resolve_target_args,
        verify_github_signature,
    )

    body = await request.body()

    signature = request.headers.get("x-hub-signature-256")
    if not ctx.config.github_secret:
        logger.warning("GitHub webhook secret 未配置，拒绝请求")
        return JSONResponse({"error": "webhook secret not configured"}, status_code=403)
    if not verify_github_signature(body, signature, ctx.config.github_secret):
        return JSONResponse({"error": "invalid signature"}, status_code=401)

    event_type = request.headers.get("x-github-event", "")
    if not event_type:
        return JSONResponse({"error": "missing X-GitHub-Event header"}, status_code=400)

    payload = await request.json()

    route = match_route(event_type, ctx.config)
    if route is None:
        return JSONResponse({"message": "no matching route", "event": event_type}, status_code=200)

    args = resolve_target_args(route.target, payload)
    import crew.webhook as _wh

    return await _wh._dispatch_task(
        ctx,
        trigger="github",
        target_type=route.target.type,
        target_name=route.target.name,
        args=args,
        sync=False,
        tenant_id=_tenant_id_for_store(request),
    )


_ALLOWED_TARGET_TYPES = {"employee", "pipeline", "discussion", "chain"}


async def _handle_openclaw(request: Any, ctx: _AppContext) -> Any:
    """处理 OpenClaw 消息事件."""

    payload = await request.json()

    target_type = payload.get("target_type", "employee")
    target_name = payload.get("target_name", "")
    args = payload.get("args", {})
    sync = payload.get("sync", False)

    if target_type not in _ALLOWED_TARGET_TYPES:
        return _error_response(
            f"不支持的 target_type: {target_type}，允许: {', '.join(sorted(_ALLOWED_TARGET_TYPES))}", 400
        )

    if not target_name:
        return JSONResponse({"error": "missing target_name"}, status_code=400)

    import crew.webhook as _wh

    return await _wh._dispatch_task(
        ctx,
        trigger="openclaw",
        target_type=target_type,
        target_name=target_name,
        args=args,
        sync=sync,
        tenant_id=_tenant_id_for_store(request),
    )


async def _handle_generic(request: Any, ctx: _AppContext) -> Any:
    """处理通用 JSON webhook."""

    payload = await request.json()

    target_type = payload.get("target_type", "pipeline")
    target_name = payload.get("target_name", "")
    args = payload.get("args", {})
    sync = payload.get("sync", False)

    if target_type not in _ALLOWED_TARGET_TYPES:
        return _error_response(
            f"不支持的 target_type: {target_type}，允许: {', '.join(sorted(_ALLOWED_TARGET_TYPES))}", 400
        )

    if not target_name:
        return JSONResponse({"error": "missing target_name"}, status_code=400)

    import crew.webhook as _wh

    return await _wh._dispatch_task(
        ctx,
        trigger="generic",
        target_type=target_type,
        target_name=target_name,
        args=args,
        sync=sync,
        tenant_id=_tenant_id_for_store(request),
    )


async def _handle_run_pipeline(request: Any, ctx: _AppContext) -> Any:
    """直接触发 pipeline."""

    # 安全加固: pipeline 配置是全局共享的，租户不应直接调用
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    name = request.path_params["name"]
    payload = (
        await request.json() if "application/json" in (request.headers.get("content-type") or "") else {}
    )
    args = payload.get("args", {})
    sync = payload.get("sync", False)
    agent_id = payload.get("agent_id")

    import crew.webhook as _wh

    return await _wh._dispatch_task(
        ctx,
        trigger="direct",
        target_type="pipeline",
        target_name=name,
        args=args,
        sync=sync,
        agent_id=agent_id,
        tenant_id=_tenant_id_for_store(request),
    )


async def _run_and_callback(
    *,
    ctx: _AppContext,
    name: str,
    args: dict,
    agent_id: str | None,
    model: str | None,
    user_message: str,
    message_history: list | None,
    extra_context: str | None,
    sender_name: str,
    channel: str,
    callback_channel_id: int,
    callback_sender_id: str | None,
    callback_parent_id: int | None,
    tenant_id: str | None = None,
) -> None:
    """后台执行员工 + 回调蚁聚发频道消息（异步回调模式）."""
    import time as _time

    import httpx

    from crew.webhook_context import _ANTGATHER_API_TOKEN, _ANTGATHER_API_URL

    logger.info(
        "异步回调开始: emp=%s channel=%d sender=%s",
        name,
        callback_channel_id,
        callback_sender_id,
    )

    # 注入额外上下文
    _args = dict(args)
    if extra_context:
        task = _args.get("task", "")
        _args["task"] = (extra_context + "\n\n" + task) if task else extra_context

    # 轨迹录制
    from contextlib import ExitStack

    from crew.trajectory import TrajectoryCollector

    _exit_stack = ExitStack()
    _task_desc = _extract_task_description(
        user_message if isinstance(user_message, str) else str(user_message)
    )
    _traj_collector = TrajectoryCollector.try_create_for_employee(
        name,
        _task_desc,
        channel=channel,
        project_dir=ctx.project_dir,
    )
    if _traj_collector is not None:
        _exit_stack.enter_context(_traj_collector)

    # 执行员工（fast/full path 路由）
    try:
        from crew.discovery import discover_employees
        from crew.tool_schema import AGENT_TOOLS
        from crew.webhook_feishu import _needs_tools

        discovery = discover_employees(project_dir=ctx.project_dir, tenant_id=tenant_id)
        emp = discovery.get(name)
        has_tools = any(t in AGENT_TOOLS for t in (emp.tools or [])) if emp else False
        use_fast_path = (
            has_tools
            and emp is not None
            and emp.fallback_model
            and isinstance(user_message, str)
            and not _needs_tools(user_message)
        )

        _t0 = _time.monotonic()
        if use_fast_path:
            from crew.webhook_feishu import _feishu_fast_reply

            result = await _feishu_fast_reply(
                ctx,
                emp,
                user_message,
                message_history=message_history,
                max_visibility="private",
                extra_context=extra_context,
                sender_name=sender_name,
            )
        else:
            import crew.webhook as _wh

            result = await _wh._execute_employee(
                ctx,
                name,
                _args,
                agent_id=agent_id,
                model=model,
                user_message=user_message,
                message_history=message_history,
                tenant_id=tenant_id,
            )

        _elapsed = _time.monotonic() - _t0
        _path = "fast" if use_fast_path else "full"
        _m = result.get("model", "?") if isinstance(result, dict) else "?"
        _in = result.get("input_tokens", 0) if isinstance(result, dict) else 0
        _out = result.get("output_tokens", 0) if isinstance(result, dict) else 0
        logger.info(
            "异步回调执行完成 [%s] %.1fs model=%s in=%d out=%d emp=%s msg=%s",
            _path,
            _elapsed,
            _m,
            _in,
            _out,
            name,
            user_message[:40],
        )
    except Exception:
        logger.exception("异步回调执行失败: emp=%s channel=%d", name, callback_channel_id)
        result = None

    # 轨迹录制完成
    if _traj_collector is not None:
        try:
            _traj_collector.finish(success=result is not None)
        except Exception as _te:
            logger.debug("异步轨迹录制失败: %s", _te)
        finally:
            _exit_stack.close()

    # 记录任务
    try:
        record = ctx.registry.create(
            trigger=channel,
            target_type="employee",
            target_name=name,
            args=_args,
            owner=callback_sender_id or None,
        )
        ctx.registry.update(record.task_id, "completed", result=result)
    except Exception:
        logger.debug("任务注册表更新失败", exc_info=True)

    # 回复后记忆写回（fire-and-forget）
    if isinstance(result, dict) and result.get("output"):
        try:
            from crew.reply_postprocess import push_if_needed

            _reply_text = result["output"].strip()
            _turn_count = len(message_history) if message_history else 1
            push_if_needed(
                employee=name,
                reply=_reply_text,
                turn_count=_turn_count,
                session_id=f"antgather-callback-{callback_channel_id}",
            )
        except Exception as _mem_err:
            logger.debug("回复记忆写回失败（不影响回调）: %s", _mem_err)

    # 回调蚁聚：发频道消息
    from crew.output_sanitizer import strip_internal_tags

    output = ""
    if isinstance(result, dict):
        output = strip_internal_tags((result.get("output") or "").strip())
    if not output:
        logger.warning(
            "异步回调: 员工返回空内容，跳过回调 emp=%s channel=%d", name, callback_channel_id
        )
        return

    if not _ANTGATHER_API_URL or not _ANTGATHER_API_TOKEN:
        logger.error("异步回调: 蚁聚 API 未配置，无法发送频道消息")
        return

    # 引用回复时，默认 @原消息发送者（触发通知）
    if callback_parent_id and sender_name and not output.startswith(f"@{sender_name}"):
        output = f"@{sender_name} {output}"

    callback_url = f"{_ANTGATHER_API_URL}/api/internal/channels/{callback_channel_id}/messages"
    callback_payload: dict[str, Any] = {
        "sender_id": callback_sender_id,
        "content": output,
    }
    if callback_parent_id:
        callback_payload["parent_id"] = callback_parent_id

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                callback_url,
                json=callback_payload,
                headers={"Authorization": f"Bearer {_ANTGATHER_API_TOKEN}"},
            )
        if resp.is_success:
            logger.info(
                "异步回调成功: emp=%s channel=%d len=%d",
                name,
                callback_channel_id,
                len(output),
            )
        else:
            logger.error(
                "异步回调失败 (HTTP %d): %s, emp=%s channel=%d",
                resp.status_code,
                resp.text[:200],
                name,
                callback_channel_id,
            )
    except Exception:
        logger.exception("异步回调请求异常: emp=%s channel=%d", name, callback_channel_id)


def _execute_skills(
    *,
    project_dir: Any,
    tenant_id: str | None,
    employee_id: str,
    employee: Any,
    message: str,
    trigger_context: dict[str, Any],
    log_prefix: str = "",
) -> str | None:
    """Skills 自动触发 — 返回注入用的记忆文本（或 None）.

    共享逻辑：同时被 _handle_chat 和 _handle_run_employee 调用。
    """
    from crew.skills import SkillStore
    from crew.skills_engine import SkillsEngine

    skill_store = SkillStore(project_dir=project_dir)
    memory_store = get_memory_store(project_dir=project_dir, tenant_id=tenant_id)
    skills_engine = SkillsEngine(skill_store, memory_store)

    employee_name = employee.character_name or employee_id

    triggered = skills_engine.check_triggers(employee_name, message, trigger_context)
    if not triggered:
        return None

    logger.info(
        "Skills 触发%s: employee=%s task=%s triggered=%d",
        log_prefix,
        employee_name,
        message[:50],
        len(triggered),
    )

    enhanced_context: dict[str, Any] = {}
    for skill, score in triggered[:3]:
        try:
            result = skills_engine.execute_skill(
                skill,
                employee_name,
                {"task": message, **trigger_context},
            )
            if result.get("enhanced_context"):
                for key, value in result["enhanced_context"].items():
                    if key in enhanced_context:
                        if isinstance(enhanced_context[key], list) and isinstance(
                            value, list
                        ):
                            enhanced_context[key].extend(value)
                        else:
                            enhanced_context[key] = value
                    else:
                        enhanced_context[key] = value
            skills_engine.record_trigger(
                skill=skill,
                employee=employee_name,
                task=message,
                match_score=score,
                execution_result=result,
            )
        except Exception as skill_exec_error:
            logger.warning(
                "Skill 执行失败%s: skill=%s error=%s",
                log_prefix,
                skill.name,
                skill_exec_error,
            )

    # 将 enhanced_context 中的 memories 格式化为文本
    memories = enhanced_context.get("memories", [])
    if not memories:
        return None

    memory_text = "【相关历史记忆】\n" + "\n".join(
        f"- [{m.get('category', '?')}] {m.get('content', '')[:200]}"
        for m in memories[:5]
    )
    logger.info(
        "Skills 记忆注入%s: employee=%s memories=%d text_len=%d",
        log_prefix,
        employee_name,
        len(memories),
        len(memory_text),
    )
    return memory_text


async def _handle_run_employee(request: Any, ctx: _AppContext) -> Any:
    """直接触发员工（支持 SSE 流式输出 + 对话模式）."""

    name = request.path_params["name"]
    payload = (
        await request.json() if "application/json" in (request.headers.get("content-type") or "") else {}
    )
    args = payload.get("args", {})
    # 兼容：顶层 task 自动塞入 args（旧版调用方式）
    if not args and "task" in payload:
        args = {"task": payload["task"]}
    sync = payload.get("sync", False)
    stream = payload.get("stream", False)
    agent_id = payload.get("agent_id")
    model = payload.get("model")

    # ── 对话模式：站内消息等渠道通过此接口统一执行 ──
    user_message = payload.get("user_message")
    if user_message is not None:
        message_history = payload.get("message_history")
        extra_context = payload.get("extra_context")
        channel = payload.get("channel", "api")
        sender_name = payload.get("sender_name") or _parse_sender_name(extra_context) or "Kai"

        # ── Skills 自动触发（必须在 callback 分支之前执行）──
        from crew.discovery import discover_employees

        discovery = discover_employees(project_dir=ctx.project_dir, tenant_id=_tenant_id_for_config(request))
        emp = discovery.get(name)

        if emp is not None and isinstance(user_message, str):
            try:
                _skills_memory_text = _execute_skills(
                    project_dir=ctx.project_dir,
                    tenant_id=_tenant_id_for_store(request),
                    employee_id=name,
                    employee=emp,
                    message=user_message,
                    trigger_context={"task": user_message, "channel": channel, **args},
                )
                if _skills_memory_text:
                    if extra_context:
                        extra_context = _skills_memory_text + "\n\n" + extra_context
                    else:
                        extra_context = _skills_memory_text
            except Exception as skills_error:
                logger.warning("Skills 检查失败: %s", skills_error)

        # Phase 3：外部对话输出控制
        from crew.classification import CHANNEL_SOURCE_TYPE, EXTERNAL_OUTPUT_CONTROL_PROMPT

        _source_type = CHANNEL_SOURCE_TYPE.get(channel, "external")
        if _source_type == "external":
            if extra_context:
                extra_context = EXTERNAL_OUTPUT_CONTROL_PROMPT + "\n\n" + extra_context
            else:
                extra_context = EXTERNAL_OUTPUT_CONTROL_PROMPT

        # ── 异步回调模式（频道 @mention）──
        # callback_sender_id 强制绑定调用者身份（X-User-Id header），防止伪造 sender
        # 安全：只有 admin 租户才能使用 callback 功能，防止任意租户通过回调在蚁聚频道发帖
        callback_channel_id = payload.get("callback_channel_id")
        if callback_channel_id is not None:
            try:
                callback_channel_id = int(callback_channel_id)
            except (TypeError, ValueError):
                callback_channel_id = None
        callback_sender_id = payload.get("callback_sender_id")
        if callback_sender_id is not None:
            callback_sender_id = str(callback_sender_id)
        callback_parent_id = payload.get("callback_parent_id")
        if callback_parent_id is not None:
            try:
                callback_parent_id = int(callback_parent_id)
            except (TypeError, ValueError):
                callback_parent_id = None

        # 安全校验：非 admin 租户禁止使用 callback 功能
        if callback_channel_id:
            tenant = get_current_tenant(request)
            if not tenant.is_admin:
                return JSONResponse(
                    {"error": "callback is only allowed for admin tenant"},
                    status_code=403,
                )

        # 绑定调用者身份：如果有 X-User-Id，强制覆盖 callback_sender_id 防止伪造
        if callback_channel_id:
            caller_id = request.headers.get("x-user-id")
            if caller_id:
                callback_sender_id = str(caller_id)
            else:
                logger.warning(
                    "callback_channel_id=%s 提供但缺少 X-User-Id header，无法绑定调用者身份（可能是内部调用）",
                    callback_channel_id,
                )

        if callback_channel_id:
            # 立即返回 202，后台处理 + 回调蚁聚
            task = asyncio.create_task(
                _run_and_callback(
                    ctx=ctx,
                    name=name,
                    args=args,
                    agent_id=agent_id,
                    model=model,
                    user_message=user_message,
                    message_history=message_history,
                    extra_context=extra_context,
                    sender_name=sender_name,
                    channel=channel,
                    callback_channel_id=callback_channel_id,
                    callback_sender_id=callback_sender_id,
                    callback_parent_id=callback_parent_id,
                    tenant_id=_tenant_id_for_config(request),
                )
            )
            _background_tasks.add(task)
            task.add_done_callback(_task_done_callback)
            return JSONResponse({"status": "accepted"}, status_code=202)

        # 注入额外上下文到 args（和飞书 handler 相同模式）
        if extra_context:
            task = args.get("task", "")
            args["task"] = (extra_context + "\n\n" + task) if task else extra_context

        # ── 轨迹录制 ──
        from contextlib import ExitStack

        from crew.trajectory import TrajectoryCollector

        _exit_stack = ExitStack()
        _task_desc = _extract_task_description(
            user_message if isinstance(user_message, str) else str(user_message)
        )
        _traj_collector = TrajectoryCollector.try_create_for_employee(
            name,
            _task_desc,
            channel=channel,
            project_dir=ctx.project_dir,
        )
        if _traj_collector is not None:
            _exit_stack.enter_context(_traj_collector)

        # 和飞书相同逻辑：闲聊走 fast path，工作消息走 full path
        import time as _time

        from crew.tool_schema import AGENT_TOOLS
        from crew.webhook_feishu import _needs_tools

        # emp 已经在 Skills 代码块中获取
        has_tools = any(t in AGENT_TOOLS for t in (emp.tools or [])) if emp else False
        use_fast_path = (
            has_tools
            and emp is not None
            and emp.fallback_model
            and isinstance(user_message, str)
            and not _needs_tools(user_message)
        )

        # 添加路径选择日志
        logger.info(
            "执行路径: employee=%s use_fast_path=%s has_tools=%s extra_context_len=%d",
            name,
            use_fast_path,
            has_tools,
            len(extra_context) if extra_context else 0,
        )

        _t0 = _time.monotonic()
        if use_fast_path:
            from crew.webhook_feishu import _feishu_fast_reply

            result = await _feishu_fast_reply(
                ctx,
                emp,
                user_message,
                message_history=message_history,
                max_visibility="private",
                extra_context=extra_context,
                sender_name=sender_name,
            )
        else:
            import crew.webhook as _wh

            result = await _wh._execute_employee(
                ctx,
                name,
                args,
                agent_id=agent_id,
                model=model,
                user_message=user_message,
                message_history=message_history,
                tenant_id=_tenant_id_for_config(request),
            )

        _elapsed = _time.monotonic() - _t0
        _path = "fast" if use_fast_path else "full"
        _m = result.get("model", "?") if isinstance(result, dict) else "?"
        _in = result.get("input_tokens", 0) if isinstance(result, dict) else 0
        _out = result.get("output_tokens", 0) if isinstance(result, dict) else 0
        logger.info(
            "站内回复 [%s] %.1fs model=%s in=%d out=%d emp=%s msg=%s",
            _path,
            _elapsed,
            _m,
            _in,
            _out,
            name,
            user_message[:40],
        )

        # 完成轨迹录制
        if _traj_collector is not None:
            try:
                _traj_collector.finish(success=True)
            except Exception as _te:
                logger.debug("站内轨迹录制失败: %s", _te)
            finally:
                _exit_stack.close()

        # 记录任务用于成本追踪
        _owner_id = payload.get("callback_sender_id") or payload.get("user_id") or ""
        record = ctx.registry.create(
            trigger=channel,
            target_type="employee",
            target_name=name,
            args=args,
            owner=_owner_id or None,
        )
        ctx.registry.update(record.task_id, "completed", result=result)

        # 回复后记忆写回（fire-and-forget）
        if isinstance(result, dict) and result.get("output"):
            try:
                from crew.reply_postprocess import push_if_needed

                _reply_text = result["output"].strip()
                _turn_count = len(message_history) if message_history else 1
                push_if_needed(
                    employee=name,
                    reply=_reply_text,
                    turn_count=_turn_count,
                    session_id=f"{channel}-sync-{name}",
                )
            except Exception as _mem_err:
                logger.debug("回复记忆写回失败（不影响响应）: %s", _mem_err)

        return JSONResponse(result)

    # ── 流式模式 ──
    if stream:
        import crew.webhook as _wh

        return await _wh._stream_employee(ctx, name, args, agent_id=agent_id, model=model, tenant_id=_tenant_id_for_config(request))

    # ── 原有同步/异步模式 ──
    import crew.webhook as _wh

    # 从 payload 提取 owner（callback_sender_id 或 user_id）
    _owner = payload.get("callback_sender_id") or payload.get("user_id") or ""

    return await _wh._dispatch_task(
        ctx,
        trigger="direct",
        target_type="employee",
        target_name=name,
        args=args,
        sync=sync,
        agent_id=agent_id,
        model=model,
        owner=_owner or None,
        tenant_id=_tenant_id_for_store(request),
    )


async def _handle_run_route(request: Any, ctx: _AppContext) -> Any:
    """直接触发路由模板 — 展开为 delegate_chain 执行."""
    import json as _json


    # 安全加固: 路由模板是全局共享的，租户不应直接调用
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.organization import load_organization

    name = request.path_params["name"]
    payload = (
        await request.json() if "application/json" in (request.headers.get("content-type") or "") else {}
    )
    task = payload.get("args", {}).get("task", "") or payload.get("task", "")
    overrides = payload.get("overrides", {})
    sync = payload.get("sync", False)

    if not task:
        return JSONResponse({"error": "缺少 task 参数"}, status_code=400)

    org = load_organization(project_dir=ctx.project_dir)
    tmpl = org.routing_templates.get(name)
    if not tmpl:
        available = list(org.routing_templates.keys())
        return JSONResponse(
            {"error": f"未找到路由模板: {name}", "available": available},
            status_code=404,
        )

    # 展开模板为 chain steps（同 _tool_route 逻辑）
    steps: list[dict[str, Any]] = []
    for step in tmpl.steps:
        if step.optional and step.role not in overrides:
            continue
        if step.human:
            continue  # 人工步骤跳过
        emp_name = overrides.get(step.role)
        if not emp_name:
            if step.employee:
                emp_name = step.employee
            elif step.employees:
                emp_name = step.employees[0]
            elif step.team:
                members = org.get_team_members(step.team)
                emp_name = members[0] if members else None
        if not emp_name:
            continue
        step_task = f"[{step.role}] {task}"
        if steps:
            step_task += "\n\n上一步结果: {prev}"
        step_dict: dict[str, Any] = {"employee_name": emp_name, "task": step_task}
        if step.approval:
            step_dict["approval"] = True
        steps.append(step_dict)

    if not steps:
        return JSONResponse({"error": "模板展开后无可执行步骤"}, status_code=400)

    chain_name = " → ".join(s["employee_name"] for s in steps)

    import crew.webhook as _wh

    return await _wh._dispatch_task(
        ctx,
        trigger="direct",
        target_type="chain",
        target_name=chain_name,
        args={"steps_json": _json.dumps(steps, ensure_ascii=False)},
        sync=sync,
        tenant_id=_tenant_id_for_store(request),
    )


async def _handle_agent_run(request: Any, ctx: _AppContext) -> Any:
    """Agent 模式执行员工 — 在 Docker 沙箱中自主完成任务 (SSE 流式)."""
    import json as _json


    # 安全加固: agent run 涉及沙箱远程执行，仅 admin 可用
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    name = request.path_params["name"]
    payload = (
        await request.json() if "application/json" in (request.headers.get("content-type") or "") else {}
    )
    task_desc = payload.get("task", "")
    if not task_desc:
        return JSONResponse({"error": "缺少 task 参数"}, status_code=400)

    model = payload.get("model", "claude-sonnet-4-5-20250929")
    max_steps = min(int(payload.get("max_steps", 30)), 100)  # 上限 100 步
    repo = payload.get("repo", "")
    base_commit = payload.get("base_commit", "")
    # 安全加固: 忽略用户传入的 sandbox 参数，硬编码安全配置

    try:
        from agentsandbox import Sandbox, SandboxEnv  # noqa: F401
        from agentsandbox.config import SandboxConfig, TaskConfig
        from knowlyrcore.wrappers import MaxStepsWrapper, RecorderWrapper  # noqa: F401
    except ImportError:
        return JSONResponse(
            {"error": "knowlyr-gym 未安装。请运行: pip install knowlyr-crew[agent]"},
            status_code=501,
        )

    from crew.agent_bridge import create_crew_agent

    async def _event_stream():
        steps_log: list[dict] = []

        def on_step(step_num: int, tool_name: str, params: dict):
            entry = {"type": "step", "step": step_num, "tool": tool_name, "params": params}
            steps_log.append(entry)

        try:
            agent = create_crew_agent(
                name,
                task_desc,
                model=model,
                project_dir=ctx.project_dir,
                on_step=on_step,
            )

            # 安全加固: 沙箱参数硬编码，不接受用户输入
            s_config = SandboxConfig(
                image="python:3.11-slim",
                memory_limit="512m",
                cpu_limit=1.0,
                network_enabled=False,
            )
            t_config = TaskConfig(
                repo_url=repo,
                base_commit=base_commit,
                description=task_desc,
            )
            env = SandboxEnv(config=s_config, task_config=t_config, max_steps=max_steps)

            def _run_loop():
                ts = env.reset()
                while not ts.done:
                    action = agent(ts.observation)
                    ts = env.step(action)
                return ts

            loop = asyncio.get_running_loop()
            final_ts = await loop.run_in_executor(None, _run_loop)

            trajectory = None
            if hasattr(env, "get_trajectory"):
                trajectory = env.get_trajectory()

            env.close()

            for entry in steps_log:
                yield f"data: {_json.dumps(entry, ensure_ascii=False)}\n\n"

            result_data = {
                "type": "result",
                "output": final_ts.observation,
                "terminated": final_ts.terminated,
                "truncated": final_ts.truncated,
                "total_steps": len(steps_log),
            }
            if trajectory:
                result_data["trajectory"] = trajectory
            yield f"data: {_json.dumps(result_data, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.exception("Agent 执行失败: %s", e)
            yield f"data: {_json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")



async def _chat_via_sg_bridge(
    *,
    ctx: _AppContext,
    effective_message: str,
    raw_message: str,
    employee_id: str,
    message_history: list[dict[str, Any]] | None,
    channel: str,
    sender_type: str,
) -> str | None:
    """SG Bridge 通道尝试 — 返回回复文本或 None（表示 fallback）."""
    # 优先尝试 SG API Bridge（直接用 Claude API + 本地工具执行）
    _sg_reply: str | None = None
    try:
        from crew.sg_api_bridge import SGAPIBridgeError, sg_api_dispatch

        _sg_reply = await sg_api_dispatch(
            effective_message,
            ctx=ctx,
            project_dir=ctx.project_dir,
            employee_name=employee_id,
            message_history=message_history,
            push_event_fn=None,  # TODO: 支持流式输出
            channel=channel,
            sender_type=sender_type,
        )
        logger.info("SG API Bridge 成功: reply_len=%d", len(_sg_reply))
    except SGAPIBridgeError as _sg_api_err:
        logger.info("SG API Bridge fallback: %s → 尝试 SSH Bridge", _sg_api_err)
        _sg_reply = None
    except Exception as _sg_api_exc:
        logger.warning("SG API Bridge 意外异常: %s → 尝试 SSH Bridge", _sg_api_exc)
        _sg_reply = None

    # Fallback: SSH Bridge（两阶段权限确认）
    if _sg_reply is None:
        try:
            from crew.sg_bridge import sg_dispatch

            # 定义权限回调函数
            async def permission_callback(operations: list[dict]) -> bool:
                """请求用户权限确认."""
                from crew.permission_request import PermissionManager

                manager = PermissionManager()

                # 构建权限请求参数
                tool_names = [op["tool"] for op in operations]
                tool_params = {
                    "operations": operations,
                    "message": raw_message[:200],
                }

                # 请求权限（会推送事件到前端）
                approved = await manager.request_permission(
                    tool_name=f"SG执行: {', '.join(tool_names)}",
                    tool_params=tool_params,
                    timeout=60.0,
                )

                return approved

            _sg_reply = await sg_dispatch(
                effective_message,
                project_dir=ctx.project_dir,
                employee_name=employee_id,
                message_history=message_history,
                permission_callback=permission_callback,
            )
        except Exception as _sg_exc:
            logger.info("SG Bridge fallback (/api/chat): %s → 走 crew 引擎", _sg_exc)
            _sg_reply = None

    return _sg_reply


async def _chat_via_agent_tools(
    *,
    ctx: _AppContext,
    request: Any,
    employee_id: str,
    effective_message: str,
    model: str | None,
    message_history: list[dict[str, Any]] | None,
    sender_id: str,
    attachments: list[dict[str, Any]] | None,
    sender_type: str,
    channel: str,
    stream: bool,
) -> Any:
    """Agent tools 执行路径 — 返回 Response（流式）或 dict（非流式）."""
    import json as _json

    import crew.webhook_executor as _wh_exec

    _chat_tenant = get_current_tenant(request)
    _tenant = _tenant_id_for_config(request)

    if stream:
        # 流式 agent tools 路径 — 真流式逐 token 推送
        agent_stream = _wh_exec._stream_employee_with_tools(
            ctx,
            employee_id,
            {},  # args
            agent_id=None,
            model=model,
            user_message=effective_message,
            message_history=message_history,
            sender_id=sender_id,
            attachments=attachments,
            sender_type=sender_type,
            channel=channel,
            tenant_id=_tenant,
            is_admin=_chat_tenant.is_admin,
        )

        async def _agent_sse_generator():
            async for chunk in agent_stream:
                chunk_data = _json.dumps(chunk, ensure_ascii=False)
                yield f"data: {chunk_data}\n\n"

        return StreamingResponse(
            _agent_sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # 非流式 agent tools 路径
        exec_result = await _wh_exec._execute_employee_with_tools(
            ctx,
            employee_id,
            {},  # args
            agent_id=None,
            model=model,
            user_message=effective_message,
            message_history=message_history,
            sender_id=sender_id,
            attachments=attachments,
            sender_type=sender_type,
            channel=channel,
            tenant_id=_tenant,
            is_admin=_chat_tenant.is_admin,
        )
        return {
            "reply": exec_result.get("output", ""),
            "employee_id": employee_id,
            "memory_updated": False,
            "tokens_used": exec_result.get("input_tokens", 0)
            + exec_result.get("output_tokens", 0),
            "latency_ms": 0,
        }


async def _handle_chat(request: Any, ctx: _AppContext) -> Any:
    """统一对话接口 — 供蚁聚、飞书、外部渠道统一调用.

    POST /api/chat
    {
        "employee_id": "moya",
        "message": "用户输入文本",
        "channel": "antgather_dm",
        "sender_id": "user_123",
        "max_visibility": "internal",   // 可选，默认 internal
        "stream": false,                // 可选，默认 false
        "context_only": false           // 可选，仅返回 prompt+记忆
    }
    """
    import json as _json


    # ── 解析请求体 ──
    try:
        payload: dict[str, Any] = (
            await request.json()
            if request.headers.get("content-type", "").startswith("application/json")
            else {}
        )
    except Exception:
        return JSONResponse({"ok": False, "error": "请求体 JSON 解析失败"}, status_code=400)

    # ── 必填字段校验 ──
    employee_id = payload.get("employee_id", "")
    message = payload.get("message", "")
    channel = payload.get("channel", "")
    sender_id = payload.get("sender_id", "")

    _required_fields = [
        ("employee_id", employee_id),
        ("message", message),
        ("channel", channel),
        ("sender_id", sender_id),
    ]
    missing = [f for f, v in _required_fields if not v]
    if missing:
        return JSONResponse(
            {"ok": False, "error": f"缺少必填字段: {', '.join(missing)}"},
            status_code=400,
        )

    # ── 可选字段 ──
    max_visibility: str = payload.get("max_visibility", "internal")
    stream: bool = bool(payload.get("stream", False))
    context_only: bool = bool(payload.get("context_only", False))
    message_history: list[dict[str, Any]] | None = payload.get("message_history")
    model: str | None = payload.get("model")
    attachments: list[dict[str, Any]] | None = payload.get("attachments")

    # 发送者身份（"internal"/"external"/"agent"/""，用于信息分级）
    sender_type: str = payload.get("sender_type", "")

    # 来自蚁聚的额外上下文（用户名、上下文预算等）
    extra_context: str | None = payload.get("extra_context")

    # 异步回调参数（用于长任务）
    callback_channel_id: int | None = payload.get("callback_channel_id")
    callback_sender_id: str | None = payload.get("callback_sender_id")
    async_mode: bool = bool(payload.get("async", False))

    # ── Skills 自动触发（在所有执行路径之前统一执行）──
    _skills_memory_text: str | None = None
    try:
        from crew.discovery import discover_employees as _chat_discover

        _chat_discovery = _chat_discover(project_dir=ctx.project_dir)
        _chat_emp = _chat_discovery.get(employee_id)

        if _chat_emp is not None and isinstance(message, str):
            _skills_memory_text = _execute_skills(
                project_dir=ctx.project_dir,
                tenant_id=_tenant_id_for_store(request),
                employee_id=employee_id,
                employee=_chat_emp,
                message=message,
                trigger_context={"task": message, "channel": channel, "sender_type": sender_type},
                log_prefix=" (/api/chat)",
            )
    except Exception as _skills_err:
        logger.warning("Skills 检查失败 (/api/chat): %s", _skills_err)

    # 合并 Skills 记忆与 extra_context
    if _skills_memory_text:
        if extra_context:
            extra_context = _skills_memory_text + "\n\n" + extra_context
        else:
            extra_context = _skills_memory_text

    # 如果指定了异步模式，立即返回并在后台执行
    # 安全校验：非 admin 租户禁止使用 callback 功能
    if async_mode and callback_channel_id:
        _chat_tenant = get_current_tenant(request)
        if not _chat_tenant.is_admin:
            return JSONResponse(
                {"ok": False, "error": "callback is only allowed for admin tenant"},
                status_code=403,
            )

        import asyncio

        # 启动后台任务
        task = asyncio.create_task(
            _run_and_callback(
                ctx=ctx,
                name=employee_id,
                args={},
                agent_id=None,
                model=model,
                user_message=message,
                message_history=message_history,
                extra_context=extra_context,
                sender_name=sender_id,
                channel=channel,
                callback_channel_id=callback_channel_id,
                callback_sender_id=callback_sender_id,
                callback_parent_id=None,
                tenant_id=_tenant_id_for_config(request),
            )
        )
        _background_tasks.add(task)
        task.add_done_callback(_task_done_callback)

        return JSONResponse(
            {
                "ok": True,
                "reply": "正在处理您的请求，完成后会通知您...",
                "async": True,
                "employee_id": employee_id,
            }
        )

    # ── 构建带上下文的消息（将 extra_context 注入到所有路径） ──
    _effective_message = message
    if extra_context:
        _effective_message = f"[上下文信息]\n{extra_context}\n\n[用户消息]\n{message}"

    # ── SG Bridge 主通道尝试（非 stream / 非 context_only 时） ──
    _sg_reply: str | None = None
    if not stream and not context_only:
        _sg_reply = await _chat_via_sg_bridge(
            ctx=ctx,
            effective_message=_effective_message,
            raw_message=message,
            employee_id=employee_id,
            message_history=message_history,
            channel=channel,
            sender_type=sender_type,
        )

    if _sg_reply is not None:
        result: dict[str, Any] = {
            "ok": True,
            "reply": _sg_reply,
            "output": _sg_reply,
            "tokens_used": 0,
            "employee_id": employee_id,
            "path": "sg",
        }
    else:
        # ── Fallback: 检查员工是否有工具，决定调用路径 ──
        from crew.discovery import discover_employees
        from crew.engine import CrewEngine
        from crew.exceptions import EmployeeNotFoundError
        from crew.tool_schema import AGENT_TOOLS

        discovery = discover_employees(project_dir=ctx.project_dir, tenant_id=_tenant_id_for_config(request))
        emp = discovery.get(employee_id)
        if emp is None:
            return JSONResponse(
                {"ok": False, "error": f"员工不存在: {employee_id}"},
                status_code=404,
            )

        # 如果员工有 agent tools，使用带工具的执行路径
        has_agent_tools = any(t in AGENT_TOOLS for t in (emp.tools or []))

        try:
            if has_agent_tools and not context_only:
                # agent tools 路径（流式或非流式）
                agent_result = await _chat_via_agent_tools(
                    ctx=ctx,
                    request=request,
                    employee_id=employee_id,
                    effective_message=_effective_message,
                    model=model,
                    message_history=message_history,
                    sender_id=sender_id,
                    attachments=attachments,
                    sender_type=sender_type,
                    channel=channel,
                    stream=stream,
                )
                # 流式时 _chat_via_agent_tools 返回 StreamingResponse，直接返回
                if stream:
                    return agent_result
                result = agent_result
            else:
                # 使用简单的 chat 路径（无工具）
                engine = CrewEngine(project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request))
                chat_result = await engine.chat(
                    employee_id=employee_id,
                    message=_effective_message,
                    channel=channel,
                    sender_id=sender_id,
                    max_visibility=max_visibility,
                    stream=stream,
                    context_only=context_only,
                    message_history=message_history,
                    model=model,
                    sender_type=sender_type,
                )

                # stream=True 时 engine.chat 返回 AsyncIterator
                if stream and not context_only:
                    # 真流式：直接消费 AsyncIterator
                    async def _real_sse_generator():
                        async for chunk in chat_result:
                            chunk_data = _json.dumps(chunk, ensure_ascii=False)
                            yield f"data: {chunk_data}\n\n"

                    return StreamingResponse(
                        _real_sse_generator(),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "X-Accel-Buffering": "no",
                        },
                    )
                else:
                    result = chat_result

        except EmployeeNotFoundError:
            return JSONResponse(
                {"ok": False, "error": f"员工不存在: {employee_id}"},
                status_code=404,
            )
        except ValueError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
        except Exception:
            logger.exception("chat() 异常: emp=%s channel=%s", employee_id, channel)
            return _error_response("内部错误", 500)

    # ── 流式响应（SSE）— SG Bridge 或 agent tools 路径的兼容模式 ──
    if stream and not context_only:
        reply_text: str = result.get("reply", "")

        async def _sse_generator():
            # 兼容模式：将完整回复拆成小块推送（模拟流式体验）
            # 每次推送 3-8 个字符，间隔 15-30ms，模拟真实 token 生成速度
            import random as _rng

            i = 0
            while i < len(reply_text):
                chunk_size = _rng.randint(3, 8)
                chunk = reply_text[i : i + chunk_size]
                i += chunk_size
                chunk_data = _json.dumps({"delta": chunk, "done": False}, ensure_ascii=False)
                yield f"data: {chunk_data}\n\n"
                await asyncio.sleep(_rng.uniform(0.015, 0.030))
            done_data = _json.dumps(
                {
                    "done": True,
                    "employee_id": employee_id,
                    "memory_updated": result.get("memory_updated", False),
                    "tokens_used": result.get("tokens_used", 0),
                    "latency_ms": result.get("latency_ms", 0),
                },
                ensure_ascii=False,
            )
            yield f"data: {done_data}\n\n"

        return StreamingResponse(
            _sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ── 非流式响应 ──
    return JSONResponse(result)


# ── KV 存储端点 ──────────────────────────────────────────────────

