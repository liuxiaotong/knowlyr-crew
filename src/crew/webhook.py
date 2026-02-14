"""Webhook 服务器 — 接收外部事件，触发 crew pipeline / 员工执行."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    HAS_STARLETTE = True
except ImportError:
    HAS_STARLETTE = False

from crew.task_registry import TaskRegistry


def create_webhook_app(
    project_dir: Path | None = None,
    token: str | None = None,
    config: "WebhookConfig | None" = None,
    cron_config: "CronConfig | None" = None,
    cors_origins: list[str] | None = None,
) -> "Starlette":
    """创建 webhook Starlette 应用.

    Args:
        project_dir: 项目目录.
        token: Bearer token（可选，为空则不启用认证）.
        config: Webhook 配置.
        cron_config: Cron 调度配置（可选）.
        cors_origins: 允许的 CORS 来源列表（可选，为空则不启用 CORS）.
    """
    if not HAS_STARLETTE:
        raise ImportError("starlette 未安装。请运行: pip install knowlyr-crew[webhook]")

    from crew.webhook_config import WebhookConfig

    if config is None:
        config = WebhookConfig()

    # 任务持久化：project_dir/.crew/tasks.jsonl
    persist_path = None
    if project_dir:
        persist_path = project_dir / ".crew" / "tasks.jsonl"
    registry = TaskRegistry(persist_path=persist_path)
    ctx = _AppContext(
        project_dir=project_dir,
        config=config,
        registry=registry,
    )

    # 初始化 cron 调度器
    scheduler = None
    if cron_config and cron_config.schedules:
        from crew.cron_scheduler import CronScheduler

        async def _cron_execute(schedule):
            record = ctx.registry.create(
                trigger="cron",
                target_type=schedule.target_type,
                target_name=schedule.target_name,
                args=schedule.args,
            )
            await _execute_task(ctx, record.task_id)

        scheduler = CronScheduler(config=cron_config, execute_fn=_cron_execute)
        ctx.scheduler = scheduler

    routes = [
        Route("/health", endpoint=_health, methods=["GET"]),
        Route("/webhook/github", endpoint=_make_handler(ctx, _handle_github), methods=["POST"]),
        Route("/webhook/openclaw", endpoint=_make_handler(ctx, _handle_openclaw), methods=["POST"]),
        Route("/webhook", endpoint=_make_handler(ctx, _handle_generic), methods=["POST"]),
        Route("/run/pipeline/{name}", endpoint=_make_handler(ctx, _handle_run_pipeline), methods=["POST"]),
        Route("/run/employee/{name}", endpoint=_make_handler(ctx, _handle_run_employee), methods=["POST"]),
        Route("/tasks/{task_id}", endpoint=_make_handler(ctx, _handle_task_status), methods=["GET"]),
        Route("/cron/status", endpoint=_make_handler(ctx, _handle_cron_status), methods=["GET"]),
    ]

    async def on_startup():
        if scheduler:
            await scheduler.start()

    async def on_shutdown():
        if scheduler:
            await scheduler.stop()

    app = Starlette(routes=routes, on_startup=[on_startup], on_shutdown=[on_shutdown])

    # 添加认证中间件
    if token:
        from crew.auth import BearerTokenMiddleware

        skip_paths = ["/health", "/webhook/github"]
        app.add_middleware(BearerTokenMiddleware, token=token, skip_paths=skip_paths)

    # 添加 CORS 中间件（后添加 = 先执行，确保 OPTIONS 预检不被认证拦截）
    if cors_origins:
        from starlette.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization"],
        )

    return app


class _AppContext:
    """应用上下文，共享于所有 handler."""

    def __init__(
        self,
        project_dir: Path | None,
        config: "WebhookConfig",
        registry: TaskRegistry,
    ):
        self.project_dir = project_dir
        self.config = config
        self.registry = registry
        self.scheduler = None  # CronScheduler, set by create_webhook_app


def _make_handler(ctx: _AppContext, handler):
    """包装 handler，注入 context."""
    async def wrapper(request: Request):
        return await handler(request, ctx)
    return wrapper


# ── Handlers ──


async def _health(request: Request) -> JSONResponse:
    """健康检查."""
    return JSONResponse({"status": "ok", "service": "crew-webhook"})


async def _handle_github(request: Request, ctx: _AppContext) -> JSONResponse:
    """处理 GitHub webhook."""
    from crew.webhook_config import (
        match_route,
        resolve_target_args,
        verify_github_signature,
    )

    body = await request.body()

    # 签名验证
    signature = request.headers.get("x-hub-signature-256")
    if ctx.config.github_secret:
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
    return await _dispatch_task(
        ctx,
        trigger="github",
        target_type=route.target.type,
        target_name=route.target.name,
        args=args,
        sync=False,
    )


async def _handle_openclaw(request: Request, ctx: _AppContext) -> JSONResponse:
    """处理 OpenClaw 消息事件."""
    payload = await request.json()

    target_type = payload.get("target_type", "employee")
    target_name = payload.get("target_name", "")
    args = payload.get("args", {})
    sync = payload.get("sync", False)

    if not target_name:
        return JSONResponse({"error": "missing target_name"}, status_code=400)

    return await _dispatch_task(
        ctx,
        trigger="openclaw",
        target_type=target_type,
        target_name=target_name,
        args=args,
        sync=sync,
    )


async def _handle_generic(request: Request, ctx: _AppContext) -> JSONResponse:
    """处理通用 JSON webhook."""
    payload = await request.json()

    target_type = payload.get("target_type", "pipeline")
    target_name = payload.get("target_name", "")
    args = payload.get("args", {})
    sync = payload.get("sync", False)

    if not target_name:
        return JSONResponse({"error": "missing target_name"}, status_code=400)

    return await _dispatch_task(
        ctx,
        trigger="generic",
        target_type=target_type,
        target_name=target_name,
        args=args,
        sync=sync,
    )


async def _handle_run_pipeline(request: Request, ctx: _AppContext) -> JSONResponse:
    """直接触发 pipeline."""
    name = request.path_params["name"]
    payload = await request.json() if request.headers.get("content-type") == "application/json" else {}
    args = payload.get("args", {})
    sync = payload.get("sync", False)
    agent_id = payload.get("agent_id")

    return await _dispatch_task(
        ctx,
        trigger="direct",
        target_type="pipeline",
        target_name=name,
        args=args,
        sync=sync,
        agent_id=agent_id,
    )


async def _handle_run_employee(request: Request, ctx: _AppContext):
    """直接触发员工（支持 SSE 流式输出）."""
    name = request.path_params["name"]
    payload = await request.json() if request.headers.get("content-type") == "application/json" else {}
    args = payload.get("args", {})
    sync = payload.get("sync", False)
    stream = payload.get("stream", False)
    agent_id = payload.get("agent_id")

    if stream:
        return await _stream_employee(ctx, name, args, agent_id=agent_id)

    return await _dispatch_task(
        ctx,
        trigger="direct",
        target_type="employee",
        target_name=name,
        args=args,
        sync=sync,
        agent_id=agent_id,
    )


async def _handle_task_status(request: Request, ctx: _AppContext) -> JSONResponse:
    """查询任务状态."""
    task_id = request.path_params["task_id"]
    record = ctx.registry.get(task_id)
    if record is None:
        return JSONResponse({"error": "task not found"}, status_code=404)
    return JSONResponse(record.model_dump(mode="json"))


async def _handle_cron_status(request: Request, ctx: _AppContext) -> JSONResponse:
    """查询 cron 调度器状态."""
    if ctx.scheduler is None:
        return JSONResponse({"enabled": False, "schedules": []})
    return JSONResponse({
        "enabled": True,
        "running": ctx.scheduler.running,
        "schedules": ctx.scheduler.get_next_runs(),
    })


# ── 任务调度 ──


async def _dispatch_task(
    ctx: _AppContext,
    trigger: str,
    target_type: str,
    target_name: str,
    args: dict[str, str],
    sync: bool = False,
    agent_id: int | None = None,
) -> JSONResponse:
    """创建任务并调度执行."""
    record = ctx.registry.create(
        trigger=trigger,
        target_type=target_type,
        target_name=target_name,
        args=args,
    )

    if sync:
        await _execute_task(ctx, record.task_id, agent_id=agent_id)
        record = ctx.registry.get(record.task_id)
        return JSONResponse(record.model_dump(mode="json"))

    asyncio.create_task(_execute_task(ctx, record.task_id, agent_id=agent_id))
    return JSONResponse(
        {"task_id": record.task_id, "status": "pending"},
        status_code=202,
    )


async def _execute_task(
    ctx: _AppContext, task_id: str, agent_id: int | None = None,
) -> None:
    """执行任务."""
    record = ctx.registry.get(task_id)
    if record is None:
        return

    ctx.registry.update(task_id, "running")

    try:
        if record.target_type == "pipeline":
            result = await _execute_pipeline(
                ctx, record.target_name, record.args, agent_id=agent_id,
            )
        elif record.target_type == "employee":
            result = await _execute_employee(
                ctx, record.target_name, record.args, agent_id=agent_id,
            )
        else:
            ctx.registry.update(task_id, "failed", error=f"未知目标类型: {record.target_type}")
            return

        ctx.registry.update(task_id, "completed", result=result)
    except Exception as e:
        logger.exception("任务执行失败: %s", task_id)
        ctx.registry.update(task_id, "failed", error=str(e))


async def _execute_pipeline(
    ctx: _AppContext,
    name: str,
    args: dict[str, str],
    agent_id: int | None = None,
) -> dict[str, Any]:
    """执行 pipeline."""
    from crew.pipeline import arun_pipeline, discover_pipelines, load_pipeline

    pipelines = discover_pipelines(project_dir=ctx.project_dir)
    if name not in pipelines:
        raise ValueError(f"未找到 pipeline: {name}")

    pipeline = load_pipeline(pipelines[name])

    result = await arun_pipeline(
        pipeline,
        initial_args=args,
        project_dir=ctx.project_dir,
        execute=True,
        api_key=None,
        agent_id=agent_id,
    )
    return result.model_dump(mode="json")


async def _execute_employee(
    ctx: _AppContext,
    name: str,
    args: dict[str, str],
    agent_id: int | None = None,
) -> dict[str, Any]:
    """执行单个员工."""
    from crew.discovery import discover_employees
    from crew.engine import CrewEngine

    employees = discover_employees(project_dir=ctx.project_dir)
    match = None
    for emp in employees:
        if emp.name == name:
            match = emp
            break

    if match is None:
        raise ValueError(f"未找到员工: {name}")

    # 获取 agent 身份
    agent_identity = None
    if agent_id:
        try:
            from crew.id_client import afetch_agent_identity
            agent_identity = await afetch_agent_identity(agent_id)
        except Exception:
            pass

    engine = CrewEngine(project_dir=ctx.project_dir)
    prompt = engine.prompt(match, args=args, agent_identity=agent_identity)

    # 尝试执行 LLM 调用（executor 自动从环境变量解析 API key）
    try:
        from crew.executor import aexecute_prompt

        use_model = match.model or "claude-sonnet-4-20250514"
        result = await aexecute_prompt(
            system_prompt=prompt,
            api_key=None,
            model=use_model,
            stream=False,
        )
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
        }

    return {"employee": name, "prompt": prompt, "output": ""}


async def _stream_employee(
    ctx: _AppContext,
    name: str,
    args: dict[str, str],
    agent_id: int | None = None,
) -> StreamingResponse:
    """SSE 流式执行单个员工."""
    import json as _json

    from crew.discovery import discover_employees
    from crew.engine import CrewEngine

    employees = discover_employees(project_dir=ctx.project_dir)
    match = None
    for emp in employees:
        if emp.name == name:
            match = emp
            break

    if match is None:
        async def _error():
            yield f"event: error\ndata: {_json.dumps({'error': f'未找到员工: {name}'})}\n\n"
        return StreamingResponse(_error(), media_type="text/event-stream")

    # 获取 agent 身份
    agent_identity = None
    if agent_id:
        try:
            from crew.id_client import afetch_agent_identity

            agent_identity = await afetch_agent_identity(agent_id)
        except Exception:
            pass

    engine = CrewEngine(project_dir=ctx.project_dir)
    prompt = engine.prompt(match, args=args, agent_identity=agent_identity)

    async def _generate():
        try:
            from crew.executor import aexecute_prompt

            use_model = match.model or "claude-sonnet-4-20250514"
            stream_iter = await aexecute_prompt(
                system_prompt=prompt,
                api_key=None,
                model=use_model,
                stream=True,
            )

            async for chunk in stream_iter:
                yield f"data: {_json.dumps({'token': chunk})}\n\n"

            # 流结束后发送完整的 result
            result = getattr(stream_iter, "result", None)
            if result:
                yield f"event: done\ndata: {_json.dumps({'employee': name, 'model': result.model, 'input_tokens': result.input_tokens, 'output_tokens': result.output_tokens})}\n\n"
            else:
                yield f"event: done\ndata: {_json.dumps({'employee': name})}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {_json.dumps({'error': str(exc)[:500]})}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


def serve_webhook(
    host: str = "0.0.0.0",
    port: int = 8765,
    project_dir: Path | None = None,
    token: str | None = None,
    enable_cron: bool = True,
    cors_origins: list[str] | None = None,
) -> None:
    """启动 webhook 服务器."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn 未安装。请运行: pip install knowlyr-crew[webhook]")

    from crew.webhook_config import load_webhook_config

    config = load_webhook_config(project_dir)

    cron_config = None
    if enable_cron:
        from crew.cron_config import load_cron_config
        cron_config = load_cron_config(project_dir)

    app = create_webhook_app(
        project_dir=project_dir,
        token=token,
        config=config,
        cron_config=cron_config,
        cors_origins=cors_origins,
    )

    logger.info("启动 Webhook 服务器: %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
