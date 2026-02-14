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
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    HAS_STARLETTE = True
except ImportError:
    HAS_STARLETTE = False

from crew.task_registry import TaskRegistry


def create_webhook_app(
    project_dir: Path | None = None,
    token: str | None = None,
    config: "WebhookConfig | None" = None,
) -> "Starlette":
    """创建 webhook Starlette 应用.

    Args:
        project_dir: 项目目录.
        token: Bearer token（可选，为空则不启用认证）.
        config: Webhook 配置.
    """
    if not HAS_STARLETTE:
        raise ImportError("starlette 未安装。请运行: pip install knowlyr-crew[webhook]")

    from crew.webhook_config import WebhookConfig

    if config is None:
        config = WebhookConfig()

    registry = TaskRegistry()
    ctx = _AppContext(
        project_dir=project_dir,
        config=config,
        registry=registry,
    )

    routes = [
        Route("/health", endpoint=_health, methods=["GET"]),
        Route("/webhook/github", endpoint=_make_handler(ctx, _handle_github), methods=["POST"]),
        Route("/webhook/openclaw", endpoint=_make_handler(ctx, _handle_openclaw), methods=["POST"]),
        Route("/webhook", endpoint=_make_handler(ctx, _handle_generic), methods=["POST"]),
        Route("/run/pipeline/{name}", endpoint=_make_handler(ctx, _handle_run_pipeline), methods=["POST"]),
        Route("/run/employee/{name}", endpoint=_make_handler(ctx, _handle_run_employee), methods=["POST"]),
        Route("/tasks/{task_id}", endpoint=_make_handler(ctx, _handle_task_status), methods=["GET"]),
    ]

    app = Starlette(routes=routes)

    # 添加认证中间件
    if token:
        from crew.auth import BearerTokenMiddleware

        skip_paths = ["/health", "/webhook/github"]
        app.add_middleware(BearerTokenMiddleware, token=token, skip_paths=skip_paths)

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


def _make_handler(ctx: _AppContext, handler):
    """包装 handler，注入 context."""
    async def wrapper(request: Request) -> JSONResponse:
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

    return await _dispatch_task(
        ctx,
        trigger="direct",
        target_type="pipeline",
        target_name=name,
        args=args,
        sync=sync,
    )


async def _handle_run_employee(request: Request, ctx: _AppContext) -> JSONResponse:
    """直接触发员工."""
    name = request.path_params["name"]
    payload = await request.json() if request.headers.get("content-type") == "application/json" else {}
    args = payload.get("args", {})
    sync = payload.get("sync", False)

    return await _dispatch_task(
        ctx,
        trigger="direct",
        target_type="employee",
        target_name=name,
        args=args,
        sync=sync,
    )


async def _handle_task_status(request: Request, ctx: _AppContext) -> JSONResponse:
    """查询任务状态."""
    task_id = request.path_params["task_id"]
    record = ctx.registry.get(task_id)
    if record is None:
        return JSONResponse({"error": "task not found"}, status_code=404)
    return JSONResponse(record.model_dump(mode="json"))


# ── 任务调度 ──


async def _dispatch_task(
    ctx: _AppContext,
    trigger: str,
    target_type: str,
    target_name: str,
    args: dict[str, str],
    sync: bool = False,
) -> JSONResponse:
    """创建任务并调度执行."""
    record = ctx.registry.create(
        trigger=trigger,
        target_type=target_type,
        target_name=target_name,
        args=args,
    )

    if sync:
        # 同步模式：等待执行完成
        await _execute_task(ctx, record.task_id)
        record = ctx.registry.get(record.task_id)
        return JSONResponse(record.model_dump(mode="json"))

    # 异步模式：后台执行，立即返回 task_id
    asyncio.create_task(_execute_task(ctx, record.task_id))
    return JSONResponse(
        {"task_id": record.task_id, "status": "pending"},
        status_code=202,
    )


async def _execute_task(ctx: _AppContext, task_id: str) -> None:
    """执行任务."""
    record = ctx.registry.get(task_id)
    if record is None:
        return

    ctx.registry.update(task_id, "running")

    try:
        if record.target_type == "pipeline":
            result = await _execute_pipeline(ctx, record.target_name, record.args)
        elif record.target_type == "employee":
            result = await _execute_employee(ctx, record.target_name, record.args)
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
) -> dict[str, Any]:
    """执行 pipeline."""
    from crew.pipeline import arun_pipeline, discover_pipelines, load_pipeline

    pipelines = discover_pipelines(project_dir=ctx.project_dir)
    if name not in pipelines:
        raise ValueError(f"未找到 pipeline: {name}")

    pipeline = load_pipeline(pipelines[name])
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    result = await arun_pipeline(
        pipeline,
        initial_args=args,
        project_dir=ctx.project_dir,
        execute=bool(api_key),
        api_key=api_key or None,
    )
    return result.model_dump(mode="json")


async def _execute_employee(
    ctx: _AppContext,
    name: str,
    args: dict[str, str],
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

    engine = CrewEngine(project_dir=ctx.project_dir)
    prompt = engine.prompt(match, args=args)

    # 如果有 API key，执行 LLM 调用
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        from crew.executor import aexecute_prompt

        result = await aexecute_prompt(
            system_prompt=prompt,
            api_key=api_key,
            stream=False,
        )
        return {
            "employee": name,
            "prompt": prompt,
            "output": result.content,
            "model": result.model,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }

    return {"employee": name, "prompt": prompt, "output": ""}


def serve_webhook(
    host: str = "0.0.0.0",
    port: int = 8765,
    project_dir: Path | None = None,
    token: str | None = None,
) -> None:
    """启动 webhook 服务器."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn 未安装。请运行: pip install knowlyr-crew[webhook]")

    from crew.webhook_config import load_webhook_config

    config = load_webhook_config(project_dir)
    app = create_webhook_app(project_dir=project_dir, token=token, config=config)

    logger.info("启动 Webhook 服务器: %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
