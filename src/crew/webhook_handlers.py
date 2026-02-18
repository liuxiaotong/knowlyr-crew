"""HTTP 请求处理器 — 各 API 端点的业务逻辑."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from crew.webhook_context import _AppContext, _EMPLOYEE_UPDATABLE_FIELDS


async def _health(request: Any) -> Any:
    """健康检查."""
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "ok", "service": "crew-webhook"})


async def _metrics(request: Any) -> Any:
    """运行时指标."""
    from starlette.responses import JSONResponse
    from crew.metrics import get_collector
    return JSONResponse(get_collector().snapshot())


async def _handle_employee_prompt(request: Any, ctx: _AppContext) -> Any:
    """返回员工配置和渲染后的 system_prompt（供 knowlyr-id 调用）."""
    from starlette.responses import JSONResponse
    from crew.discovery import discover_employees
    from crew.engine import CrewEngine
    from crew.tool_schema import employee_tools_to_schemas

    identifier = request.path_params["identifier"]
    result = discover_employees(ctx.project_dir)

    # 按 agent_id（数字）或 name（字符串）查找
    employee = None
    try:
        agent_id = int(identifier)
        for emp in result.employees.values():
            if emp.agent_id == agent_id:
                employee = emp
                break
    except ValueError:
        employee = result.employees.get(identifier)

    if not employee:
        return JSONResponse({"error": "Employee not found"}, status_code=404)

    # 渲染 prompt（不传 agent_identity → 不含 DB 记忆）
    engine = CrewEngine(ctx.project_dir)
    system_prompt = engine.prompt(employee)
    tool_schemas, _ = employee_tools_to_schemas(employee.tools, defer=False)

    # 从 YAML 读取 Employee model 之外的字段（bio, temperature 等）
    bio = ""
    temperature = None
    max_tokens = None
    if employee.source_path:
        yaml_path = employee.source_path / "employee.yaml"
        if yaml_path.exists():
            import yaml
            with open(yaml_path) as f:
                yaml_config = yaml.safe_load(f) or {}
            bio = yaml_config.get("bio", "")
            temperature = yaml_config.get("temperature")
            max_tokens = yaml_config.get("max_tokens")

    # 组织架构信息（团队、权限、成本）
    from crew.organization import load_organization, get_effective_authority
    org = load_organization(project_dir=ctx.project_dir)
    team = org.get_team(employee.name)
    authority = get_effective_authority(org, employee.name, project_dir=ctx.project_dir)

    # 7 天成本
    from crew.cost import query_cost_summary
    cost_summary = query_cost_summary(ctx.registry, employee=employee.name, days=7)

    return JSONResponse({
        "name": employee.name,
        "character_name": employee.character_name,
        "display_name": employee.display_name,
        "description": employee.description,
        "bio": bio,
        "version": employee.version,
        "model": employee.model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": employee.tools,
        "tool_schemas": tool_schemas,
        "system_prompt": system_prompt,
        "agent_id": employee.agent_id,
        "team": team,
        "authority": authority,
        "cost_7d": cost_summary,
    })


async def _handle_employee_state(request: Any, ctx: _AppContext) -> Any:
    """返回员工完整运行时状态：角色设定 + 最近记忆 + 最近笔记."""
    from starlette.responses import JSONResponse
    from crew.discovery import discover_employees
    from crew.memory import MemoryStore

    identifier = request.path_params["identifier"]
    result = discover_employees(ctx.project_dir)

    employee = None
    try:
        agent_id = int(identifier)
        for emp in result.employees.values():
            if emp.agent_id == agent_id:
                employee = emp
                break
    except ValueError:
        employee = result.employees.get(identifier)

    if not employee:
        return JSONResponse({"error": "Employee not found"}, status_code=404)

    limit = int(request.query_params.get("memory_limit", "10"))

    # 读取 soul.md
    soul = ""
    if employee.source_path:
        soul_path = employee.source_path / "soul.md"
        if soul_path.exists():
            soul = soul_path.read_text(encoding="utf-8")

    # 读取最近记忆
    store = MemoryStore(project_dir=ctx.project_dir)
    memories = store.query(employee.name, limit=limit)
    memory_list = [
        {"category": m.category, "content": m.content, "created_at": m.created_at, "tags": m.tags}
        for m in memories
    ]

    # 读取最近笔记
    notes_dir = (ctx.project_dir or Path.cwd()) / ".crew" / "notes"
    recent_notes: list[dict] = []
    if notes_dir.is_dir():
        note_files = sorted(notes_dir.glob("*.md"), reverse=True)
        for nf in note_files[:5]:
            text = nf.read_text(encoding="utf-8")
            if employee.character_name in text or employee.name in text:
                recent_notes.append({"filename": nf.name, "content": text[:500]})

    return JSONResponse({
        "name": employee.name,
        "character_name": employee.character_name,
        "display_name": employee.display_name,
        "soul": soul,
        "memories": memory_list,
        "notes": recent_notes,
    })


async def _handle_employee_update(request: Any, ctx: _AppContext) -> Any:
    """更新员工配置（model 等）— employee.yaml 是唯一真相源."""
    from starlette.responses import JSONResponse
    from crew.discovery import discover_employees

    identifier = request.path_params["identifier"]
    result = discover_employees(ctx.project_dir)

    employee = None
    try:
        agent_id = int(identifier)
        for emp in result.employees.values():
            if emp.agent_id == agent_id:
                employee = emp
                break
    except ValueError:
        employee = result.employees.get(identifier)

    if not employee:
        return JSONResponse({"error": "Employee not found"}, status_code=404)

    if not employee.source_path:
        return JSONResponse({"error": "Employee source path unknown"}, status_code=400)

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    # 只允许白名单字段
    updates = {}
    for key in _EMPLOYEE_UPDATABLE_FIELDS:
        if key in payload:
            updates[key] = payload[key]

    if not updates:
        return JSONResponse(
            {"error": f"No updatable fields. Allowed: {', '.join(sorted(_EMPLOYEE_UPDATABLE_FIELDS))}"},
            status_code=400,
        )

    # 写回 employee.yaml
    from crew.sync import _write_yaml_field

    try:
        _write_yaml_field(employee.source_path, updates)
    except Exception as e:
        logger.exception("更新 employee.yaml 失败: %s", identifier)
        return JSONResponse({"error": f"Write failed: {e}"}, status_code=500)

    # 同步到 knowlyr-id
    synced = False
    if employee.agent_id:
        try:
            from crew.id_client import aupdate_agent

            sync_kwargs: dict = {}
            if "model" in updates:
                sync_kwargs["model"] = updates["model"]
            if "temperature" in updates:
                sync_kwargs["temperature"] = updates["temperature"]
            if "max_tokens" in updates:
                sync_kwargs["max_tokens"] = updates["max_tokens"]
            if sync_kwargs:
                synced = await aupdate_agent(agent_id=employee.agent_id, **sync_kwargs)
        except Exception:
            logger.warning("同步到 knowlyr-id 失败（更新已写入本地）: %s", identifier)

    return JSONResponse({
        "ok": True,
        "updated": updates,
        "synced_to_id": synced,
        "employee": employee.name,
    })


async def _handle_employee_delete(request: Any, ctx: _AppContext) -> Any:
    """删除员工（本地文件 + 远端标记为 inactive）."""
    import shutil
    from starlette.responses import JSONResponse
    from crew.discovery import discover_employees

    identifier = request.path_params["identifier"]
    result = discover_employees(ctx.project_dir, cache_ttl=0)

    employee = None
    try:
        agent_id = int(identifier)
        for emp in result.employees.values():
            if emp.agent_id == agent_id:
                employee = emp
                break
    except ValueError:
        employee = result.employees.get(identifier)

    if not employee:
        return JSONResponse({"error": "Employee not found"}, status_code=404)

    if not employee.source_path:
        return JSONResponse({"error": "Employee source path unknown"}, status_code=400)

    # 删除本地文件
    source = employee.source_path
    try:
        if source.is_dir():
            shutil.rmtree(source)
        elif source.is_file():
            source.unlink()
    except Exception as e:
        logger.exception("删除员工文件失败: %s", identifier)
        return JSONResponse({"error": f"Delete failed: {e}"}, status_code=500)

    # 远端标 inactive
    remote_disabled = False
    also_remote = request.query_params.get("also_remote", "true").lower() != "false"
    if employee.agent_id and also_remote:
        try:
            from crew.id_client import aupdate_agent

            remote_disabled = await aupdate_agent(
                agent_id=employee.agent_id, agent_status="inactive",
            )
        except Exception:
            logger.warning("远端禁用失败: Agent #%s", employee.agent_id)

    return JSONResponse({
        "ok": True,
        "deleted": employee.name,
        "remote_disabled": remote_disabled,
    })


async def _handle_memory_ingest(request: Any, ctx: _AppContext) -> Any:
    """接收外部讨论数据，写入参与者记忆和会议记录."""
    from starlette.responses import JSONResponse

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    try:
        from crew.discussion_ingest import DiscussionIngestor, DiscussionInput

        data = DiscussionInput(**payload)
        ingestor = DiscussionIngestor(project_dir=ctx.project_dir)
        results = ingestor.ingest(data)
        return JSONResponse({
            "ok": True,
            "topic": data.topic,
            "memories_written": results["memories_written"],
            "meeting_id": results.get("meeting_id"),
            "participants": results["participants"],
        })
    except Exception as e:
        logger.exception("记忆导入失败")
        return JSONResponse({"error": str(e)}, status_code=500)


async def _handle_github(request: Any, ctx: _AppContext) -> Any:
    """处理 GitHub webhook."""
    from starlette.responses import JSONResponse
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
    )


async def _handle_openclaw(request: Any, ctx: _AppContext) -> Any:
    """处理 OpenClaw 消息事件."""
    from starlette.responses import JSONResponse

    payload = await request.json()

    target_type = payload.get("target_type", "employee")
    target_name = payload.get("target_name", "")
    args = payload.get("args", {})
    sync = payload.get("sync", False)

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
    )


async def _handle_generic(request: Any, ctx: _AppContext) -> Any:
    """处理通用 JSON webhook."""
    from starlette.responses import JSONResponse

    payload = await request.json()

    target_type = payload.get("target_type", "pipeline")
    target_name = payload.get("target_name", "")
    args = payload.get("args", {})
    sync = payload.get("sync", False)

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
    )


async def _handle_run_pipeline(request: Any, ctx: _AppContext) -> Any:
    """直接触发 pipeline."""
    from starlette.responses import JSONResponse

    name = request.path_params["name"]
    payload = await request.json() if request.headers.get("content-type") == "application/json" else {}
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
    )


async def _handle_run_employee(request: Any, ctx: _AppContext) -> Any:
    """直接触发员工（支持 SSE 流式输出）."""
    name = request.path_params["name"]
    payload = await request.json() if request.headers.get("content-type") == "application/json" else {}
    args = payload.get("args", {})
    sync = payload.get("sync", False)
    stream = payload.get("stream", False)
    agent_id = payload.get("agent_id")
    model = payload.get("model")

    if stream:
        import crew.webhook as _wh
        return await _wh._stream_employee(ctx, name, args, agent_id=agent_id, model=model)

    import crew.webhook as _wh
    return await _wh._dispatch_task(
        ctx,
        trigger="direct",
        target_type="employee",
        target_name=name,
        args=args,
        sync=sync,
        agent_id=agent_id,
        model=model,
    )


async def _handle_agent_run(request: Any, ctx: _AppContext) -> Any:
    """Agent 模式执行员工 — 在 Docker 沙箱中自主完成任务 (SSE 流式)."""
    import json as _json

    from starlette.responses import JSONResponse, StreamingResponse

    name = request.path_params["name"]
    payload = await request.json() if request.headers.get("content-type") == "application/json" else {}
    task_desc = payload.get("task", "")
    if not task_desc:
        return JSONResponse({"error": "缺少 task 参数"}, status_code=400)

    model = payload.get("model", "claude-sonnet-4-5-20250929")
    max_steps = payload.get("max_steps", 30)
    repo = payload.get("repo", "")
    base_commit = payload.get("base_commit", "")
    sandbox_cfg = payload.get("sandbox", {})

    try:
        from agentsandbox import Sandbox, SandboxEnv
        from agentsandbox.config import SandboxConfig, TaskConfig
        from knowlyrcore.wrappers import MaxStepsWrapper, RecorderWrapper
    except ImportError:
        return JSONResponse(
            {"error": "knowlyr-agent 未安装。请运行: pip install knowlyr-crew[agent]"},
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

            s_config = SandboxConfig(
                image=sandbox_cfg.get("image", "python:3.11-slim"),
                memory_limit=sandbox_cfg.get("memory_limit", "512m"),
                cpu_limit=sandbox_cfg.get("cpu_limit", 1.0),
                network_enabled=sandbox_cfg.get("network_enabled", False),
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


async def _handle_task_status(request: Any, ctx: _AppContext) -> Any:
    """查询任务状态."""
    from starlette.responses import JSONResponse

    task_id = request.path_params["task_id"]
    record = ctx.registry.get(task_id)
    if record is None:
        return JSONResponse({"error": "task not found"}, status_code=404)
    return JSONResponse(record.model_dump(mode="json"))


async def _handle_task_replay(request: Any, ctx: _AppContext) -> Any:
    """重放已完成/失败的任务."""
    from starlette.responses import JSONResponse

    task_id = request.path_params["task_id"]
    record = ctx.registry.get(task_id)
    if record is None:
        return JSONResponse({"error": "task not found"}, status_code=404)

    if record.status not in ("completed", "failed"):
        return JSONResponse({"error": "只能重放已完成或失败的任务"}, status_code=400)

    import crew.webhook as _wh
    return await _wh._dispatch_task(
        ctx,
        trigger="replay",
        target_type=record.target_type,
        target_name=record.target_name,
        args=record.args,
        sync=False,
    )


async def _handle_task_approve(request: Any, ctx: _AppContext) -> Any:
    """POST /api/tasks/{task_id}/approve — 批准或拒绝等待审批的任务."""
    import asyncio

    from starlette.responses import JSONResponse

    task_id = request.path_params["task_id"]
    body = await request.json()
    action = body.get("action", "approve")

    record = ctx.registry.get(task_id)
    if record is None:
        return JSONResponse({"error": "task not found"}, status_code=404)
    if record.status != "awaiting_approval":
        return JSONResponse(
            {"error": f"任务状态为 {record.status}，无法审批"}, status_code=400,
        )

    if action == "reject":
        reason = body.get("reason", "人工拒绝")
        ctx.registry.update(task_id, "failed", error=reason)
        return JSONResponse({"status": "rejected", "task_id": task_id})

    from crew.webhook_executor import _resume_chain

    asyncio.create_task(_resume_chain(ctx, task_id))
    return JSONResponse({"status": "approved", "task_id": task_id})


async def _handle_cron_status(request: Any, ctx: _AppContext) -> Any:
    """查询 cron 调度器状态."""
    from starlette.responses import JSONResponse

    if ctx.scheduler is None:
        return JSONResponse({"enabled": False, "schedules": []})
    return JSONResponse({
        "enabled": True,
        "running": ctx.scheduler.running,
        "schedules": ctx.scheduler.get_next_runs(),
    })


async def _handle_cost_summary(request: Any, ctx: _AppContext) -> Any:
    """成本汇总 — GET /api/cost/summary?days=7&employee=xxx&source=work."""
    from starlette.responses import JSONResponse
    from crew.cost import query_cost_summary

    days = int(request.query_params.get("days", "7"))
    employee = request.query_params.get("employee")
    source = request.query_params.get("source")

    summary = query_cost_summary(ctx.registry, employee=employee, days=days, source=source)
    return JSONResponse(summary)


async def _handle_authority_restore(request: Any, ctx: _AppContext) -> Any:
    """恢复员工权限 — POST /api/employees/{identifier}/authority/restore."""
    from starlette.responses import JSONResponse
    from crew.discovery import discover_employees
    from crew.organization import (
        load_organization,
        get_effective_authority,
        _load_overrides,
        _save_overrides,
        _authority_overrides,
    )

    identifier = request.path_params["identifier"]
    result = discover_employees(ctx.project_dir)

    employee = None
    try:
        agent_id = int(identifier)
        for emp in result.employees.values():
            if emp.agent_id == agent_id:
                employee = emp
                break
    except ValueError:
        employee = result.employees.get(identifier)

    if not employee:
        return JSONResponse({"error": "Employee not found"}, status_code=404)

    _load_overrides(ctx.project_dir)
    override = _authority_overrides.pop(employee.name, None)
    if override is None:
        org = load_organization(project_dir=ctx.project_dir)
        current = org.get_authority(employee.name)
        return JSONResponse({
            "ok": True,
            "employee": employee.name,
            "authority": current,
            "message": "无覆盖记录，权限未变更",
        })

    _save_overrides(ctx.project_dir)
    org = load_organization(project_dir=ctx.project_dir)
    restored = org.get_authority(employee.name)

    return JSONResponse({
        "ok": True,
        "employee": employee.name,
        "authority": restored,
        "previous_override": override,
        "message": f"权限已恢复: {override['level']} → {restored}",
    })


async def _handle_project_status(request: Any, ctx: _AppContext) -> Any:
    """项目状态概览 — 组织架构 + 成本 + 员工列表."""
    from starlette.responses import JSONResponse
    from crew.discovery import discover_employees
    from crew.organization import load_organization, get_effective_authority
    from crew.cost import query_cost_summary

    result = discover_employees(ctx.project_dir)
    org = load_organization(project_dir=ctx.project_dir)
    cost = query_cost_summary(ctx.registry, days=7)

    employees_info = []
    for name, emp in sorted(result.employees.items()):
        team = org.get_team(name)
        authority = get_effective_authority(org, name, project_dir=ctx.project_dir)
        employees_info.append({
            "name": name,
            "character_name": emp.character_name,
            "display_name": emp.display_name,
            "description": emp.description,
            "model": emp.model,
            "agent_id": emp.agent_id,
            "team": team,
            "authority": authority,
        })

    # 团队汇总
    teams_info = {}
    for tid, team in org.teams.items():
        teams_info[tid] = {
            "label": team.label,
            "member_count": len(team.members),
            "members": team.members,
        }

    return JSONResponse({
        "total_employees": len(result.employees),
        "teams": teams_info,
        "authority_levels": {
            level: {"label": auth.label, "count": len(auth.members)}
            for level, auth in org.authority.items()
        },
        "cost_7d": cost,
        "employees": employees_info,
        "routing_templates": {
            name: {
                "label": tmpl.label,
                "steps": [
                    {
                        "role": step.role,
                        "employee": step.employee,
                        "employees": step.employees or [],
                        "team": step.team,
                        "description": step.description,
                        "optional": step.optional,
                    }
                    for step in tmpl.steps
                ],
            }
            for name, tmpl in org.routing_templates.items()
        },
    })
