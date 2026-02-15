"""Webhook 服务器 — 接收外部事件，触发 crew pipeline / 员工执行."""

from __future__ import annotations

import asyncio
import logging
import uuid
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

from crew.exceptions import EmployeeNotFoundError, PipelineNotFoundError
from crew.task_registry import TaskRegistry


def create_webhook_app(
    project_dir: Path | None = None,
    token: str | None = None,
    config: "WebhookConfig | None" = None,
    cron_config: "CronConfig | None" = None,
    cors_origins: list[str] | None = None,
    feishu_config: "FeishuConfig | None" = None,
) -> "Starlette":
    """创建 webhook Starlette 应用.

    Args:
        project_dir: 项目目录.
        token: Bearer token（可选，为空则不启用认证）.
        config: Webhook 配置.
        cron_config: Cron 调度配置（可选）.
        cors_origins: 允许的 CORS 来源列表（可选，为空则不启用 CORS）.
        feishu_config: 飞书 Bot 配置（可选）.
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

    # 初始化飞书 Bot
    if feishu_config and feishu_config.app_id and feishu_config.app_secret:
        from crew.feishu import EventDeduplicator, FeishuTokenManager

        ctx.feishu_config = feishu_config
        ctx.feishu_token_mgr = FeishuTokenManager(
            feishu_config.app_id, feishu_config.app_secret,
        )
        ctx.feishu_dedup = EventDeduplicator()
        logger.info("飞书 Bot 已启用 (app_id=%s)", feishu_config.app_id)

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

            # 投递结果到配置的渠道
            if schedule.delivery:
                record = ctx.registry.get(record.task_id)
                try:
                    from crew.delivery import deliver, DeliveryTarget as DT

                    targets = [DT(**t.model_dump()) for t in schedule.delivery]
                    results = await deliver(
                        targets,
                        task_name=schedule.name,
                        task_result=record.result if record else None,
                        task_error=record.error if record else None,
                    )
                    for r in results:
                        if not r.success:
                            logger.warning("投递失败 [%s → %s]: %s", schedule.name, r.target_type, r.detail)
                except Exception as e:
                    logger.warning("投递异常 [%s]: %s", schedule.name, e)

        scheduler = CronScheduler(config=cron_config, execute_fn=_cron_execute)
        ctx.scheduler = scheduler

    routes = [
        Route("/health", endpoint=_health, methods=["GET"]),
        Route("/metrics", endpoint=_metrics, methods=["GET"]),
        Route("/webhook/github", endpoint=_make_handler(ctx, _handle_github), methods=["POST"]),
        Route("/webhook/openclaw", endpoint=_make_handler(ctx, _handle_openclaw), methods=["POST"]),
        Route("/webhook", endpoint=_make_handler(ctx, _handle_generic), methods=["POST"]),
        Route("/run/pipeline/{name}", endpoint=_make_handler(ctx, _handle_run_pipeline), methods=["POST"]),
        Route("/run/employee/{name}", endpoint=_make_handler(ctx, _handle_run_employee), methods=["POST"]),
        Route("/agent/run/{name}", endpoint=_make_handler(ctx, _handle_agent_run), methods=["POST"]),
        Route("/tasks/{task_id}", endpoint=_make_handler(ctx, _handle_task_status), methods=["GET"]),
        Route("/tasks/{task_id}/replay", endpoint=_make_handler(ctx, _handle_task_replay), methods=["POST"]),
        Route("/cron/status", endpoint=_make_handler(ctx, _handle_cron_status), methods=["GET"]),
        Route("/feishu/event", endpoint=_make_handler(ctx, _handle_feishu_event), methods=["POST"]),
    ]

    async def on_startup():
        if scheduler:
            await scheduler.start()
        # 启动周期心跳
        try:
            from crew.id_client import HeartbeatManager

            ctx.heartbeat_mgr = HeartbeatManager(interval=60.0)
            await ctx.heartbeat_mgr.start()
        except ImportError:
            pass
        # 恢复未完成的 pipeline 任务
        asyncio.create_task(_resume_incomplete_pipelines(ctx))

    async def on_shutdown():
        if scheduler:
            await scheduler.stop()
        if ctx.heartbeat_mgr:
            await ctx.heartbeat_mgr.stop()

    app = Starlette(routes=routes, on_startup=[on_startup], on_shutdown=[on_shutdown])

    # 添加请求大小限制（始终生效）
    from crew.auth import RequestSizeLimitMiddleware

    app.add_middleware(RequestSizeLimitMiddleware)

    # 添加认证中间件
    if token:
        from crew.auth import BearerTokenMiddleware, RateLimitMiddleware

        skip_paths = ["/health", "/webhook/github", "/feishu/event"]
        app.add_middleware(BearerTokenMiddleware, token=token, skip_paths=skip_paths)
        app.add_middleware(
            RateLimitMiddleware,
            skip_paths=["/health", "/metrics", "/webhook/github", "/feishu/event"],
        )

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
        self.heartbeat_mgr = None  # HeartbeatManager, set by create_webhook_app
        self.feishu_config = None  # FeishuConfig, set by create_webhook_app
        self.feishu_token_mgr = None  # FeishuTokenManager, set by create_webhook_app
        self.feishu_dedup = None  # EventDeduplicator, set by create_webhook_app


def _make_handler(ctx: _AppContext, handler):
    """包装 handler，注入 context."""
    async def wrapper(request: Request):
        return await handler(request, ctx)
    return wrapper


# ── Handlers ──


async def _health(request: Request) -> JSONResponse:
    """健康检查."""
    return JSONResponse({"status": "ok", "service": "crew-webhook"})


async def _metrics(request: Request) -> JSONResponse:
    """运行时指标."""
    from crew.metrics import get_collector
    return JSONResponse(get_collector().snapshot())


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
    model = payload.get("model")

    if stream:
        return await _stream_employee(ctx, name, args, agent_id=agent_id, model=model)

    return await _dispatch_task(
        ctx,
        trigger="direct",
        target_type="employee",
        target_name=name,
        args=args,
        sync=sync,
        agent_id=agent_id,
        model=model,
    )


async def _handle_agent_run(request: Request, ctx: _AppContext):
    """Agent 模式执行员工 — 在 Docker 沙箱中自主完成任务 (SSE 流式).

    Body:
        task: 任务描述 (必填)
        model: 模型 ID (可选)
        max_steps: 最大步数 (可选, 默认 30)
        repo: Git 仓库 (可选, 如 "owner/repo")
        base_commit: 基准 commit (可选)
        sandbox: 沙箱配置 (可选)
    """
    import json as _json

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
            # 创建 agent 函数
            agent = create_crew_agent(
                name,
                task_desc,
                model=model,
                project_dir=ctx.project_dir,
                on_step=on_step,
            )

            # 创建沙箱环境
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

            # 运行 agent loop (同步，在线程中执行)
            def _run_loop():
                ts = env.reset()
                while not ts.done:
                    action = agent(ts.observation)
                    ts = env.step(action)
                return ts

            loop = asyncio.get_event_loop()
            final_ts = await loop.run_in_executor(None, _run_loop)

            # 尝试获取 trajectory
            trajectory = None
            if hasattr(env, "get_trajectory"):
                trajectory = env.get_trajectory()

            env.close()

            # 发送每步 SSE
            for entry in steps_log:
                yield f"data: {_json.dumps(entry, ensure_ascii=False)}\n\n"

            # 发送最终结果
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


async def _handle_task_status(request: Request, ctx: _AppContext) -> JSONResponse:
    """查询任务状态."""
    task_id = request.path_params["task_id"]
    record = ctx.registry.get(task_id)
    if record is None:
        return JSONResponse({"error": "task not found"}, status_code=404)
    return JSONResponse(record.model_dump(mode="json"))


async def _handle_task_replay(request: Request, ctx: _AppContext) -> JSONResponse:
    """重放已完成/失败的任务."""
    task_id = request.path_params["task_id"]
    record = ctx.registry.get(task_id)
    if record is None:
        return JSONResponse({"error": "task not found"}, status_code=404)

    if record.status not in ("completed", "failed"):
        return JSONResponse({"error": "只能重放已完成或失败的任务"}, status_code=400)

    return await _dispatch_task(
        ctx,
        trigger="replay",
        target_type=record.target_type,
        target_name=record.target_name,
        args=record.args,
        sync=False,
    )


async def _handle_feishu_event(request: Request, ctx: _AppContext) -> JSONResponse:
    """处理飞书事件回调.

    支持:
    1. URL verification challenge
    2. im.message.receive_v1 消息事件
    """
    payload = await request.json()

    # 1. URL 验证（飞书注册回调地址时发送）
    if payload.get("type") == "url_verification":
        challenge = payload.get("challenge", "")
        return JSONResponse({"challenge": challenge})

    # 2. 检查飞书是否配置
    if ctx.feishu_config is None or ctx.feishu_token_mgr is None:
        return JSONResponse({"error": "飞书 Bot 未配置"}, status_code=501)

    # 3. 验证 event token
    header = payload.get("header", {})
    event_token = header.get("token", payload.get("token", ""))
    if ctx.feishu_config.verification_token:
        from crew.feishu import verify_feishu_event

        if not verify_feishu_event(ctx.feishu_config.verification_token, event_token):
            return JSONResponse({"error": "invalid token"}, status_code=401)

    # 4. 只处理消息事件
    event_type = header.get("event_type", "")
    if event_type != "im.message.receive_v1":
        return JSONResponse({"message": "ignored", "event_type": event_type})

    # 5. 解析消息
    from crew.feishu import parse_message_event

    msg_event = parse_message_event(payload)
    if msg_event is None:
        return JSONResponse({"message": "unsupported message type"})

    # 6. 去重
    if ctx.feishu_dedup and ctx.feishu_dedup.is_duplicate(msg_event.message_id):
        return JSONResponse({"message": "duplicate"})

    # 7. 后台处理（飞书要求 3s 内响应）
    asyncio.create_task(_feishu_dispatch(ctx, msg_event))

    return JSONResponse({"message": "ok"})


async def _feishu_dispatch(ctx: _AppContext, msg_event: Any) -> None:
    """后台处理飞书消息：路由到员工 → 执行 → 回复卡片."""
    from crew.discovery import discover_employees
    from crew.feishu import (
        resolve_employee_from_mention,
        send_feishu_card,
        send_feishu_text,
    )

    assert ctx.feishu_config is not None
    assert ctx.feishu_token_mgr is not None

    try:
        discovery = discover_employees(project_dir=ctx.project_dir)

        employee_name, task_text = resolve_employee_from_mention(
            msg_event.mentions,
            msg_event.text,
            discovery,
            default_employee=ctx.feishu_config.default_employee,
        )

        if employee_name is None:
            await send_feishu_text(
                ctx.feishu_token_mgr,
                msg_event.chat_id,
                "未能识别目标员工，请 @员工名 + 任务描述。",
            )
            return

        # 发送"处理中"提示
        emp = discovery.get(employee_name)
        emp_display = ""
        if emp:
            emp_display = emp.character_name or emp.effective_display_name
        await send_feishu_text(
            ctx.feishu_token_mgr,
            msg_event.chat_id,
            f"\U0001f4dd {emp_display} 已收到任务，正在处理...",
        )

        # 执行员工 — 飞书对话明确标注为实时聊天，防止幻觉
        feishu_task = (
            f"[实时对话] Kai 在飞书上对你说：{task_text}\n\n"
            f"这是实时对话，不是定时任务。你手上没有任何业务数据。"
            f"只回答 Kai 说的内容，不要主动汇报工作、不要编造任何具体信息。"
            f"如果他问你不知道的事，就说你不知道。"
        )
        args = {"task": feishu_task}
        if emp and emp.args:
            first_required = next((a for a in emp.args if a.required), None)
            if first_required:
                args[first_required.name] = feishu_task

        result = await _execute_employee(ctx, employee_name, args, model=None)

        # 发送结果卡片
        task_name = f"{emp_display} — 飞书任务"
        await send_feishu_card(
            ctx.feishu_token_mgr,
            msg_event.chat_id,
            task_name=task_name,
            task_result=result,
            task_error=None,
        )

        # 记录任务
        record = ctx.registry.create(
            trigger="feishu",
            target_type="employee",
            target_name=employee_name,
            args=args,
        )
        ctx.registry.update(record.task_id, "completed", result=result)

    except Exception as e:
        logger.exception("飞书消息处理失败: %s", e)
        try:
            await send_feishu_card(
                ctx.feishu_token_mgr,
                msg_event.chat_id,
                task_name="飞书任务",
                task_result=None,
                task_error=str(e),
            )
        except Exception:
            logger.exception("飞书错误回复发送失败")


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
    model: str | None = None,
) -> JSONResponse:
    """创建任务并调度执行."""
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

    if sync:
        await _execute_task(ctx, record.task_id, agent_id=agent_id, model=model, trace_id=trace_id)
        record = ctx.registry.get(record.task_id)
        return JSONResponse(record.model_dump(mode="json"))

    asyncio.create_task(_execute_task(ctx, record.task_id, agent_id=agent_id, model=model, trace_id=trace_id))
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

    try:
        if record.target_type == "pipeline":
            logger.info("执行 pipeline [trace=%s] %s", trace_id, record.target_name)
            result = await _execute_pipeline(
                ctx, record.target_name, record.args, agent_id=agent_id, task_id=task_id,
            )
        elif record.target_type == "employee":
            logger.info("执行 employee [trace=%s] %s", trace_id, record.target_name)
            result = await _execute_employee(
                ctx, record.target_name, record.args, agent_id=agent_id, model=model,
            )
        else:
            ctx.registry.update(task_id, "failed", error=f"未知目标类型: {record.target_type}")
            return

        logger.info("任务完成 [trace=%s] task=%s", trace_id, task_id)
        ctx.registry.update(task_id, "completed", result=result)
    except Exception as e:
        logger.exception("任务执行失败 [trace=%s]: %s", trace_id, task_id)
        ctx.registry.update(task_id, "failed", error=str(e))


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

        use_model = model or target.model or "claude-sonnet-4-20250514"
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


_MAX_DELEGATION_ROUNDS = 10


async def _execute_employee_with_delegation(
    ctx: _AppContext,
    name: str,
    args: dict[str, str],
    *,
    agent_id: int | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """执行带委派能力的员工（agent loop with delegate tool）."""
    from crew.discovery import discover_employees
    from crew.engine import CrewEngine
    from crew.executor import aexecute_with_tools
    from crew.providers import Provider, detect_provider
    from crew.tool_schema import employee_tools_to_schemas, is_finish_tool

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
        except Exception:
            pass

    engine = CrewEngine(project_dir=ctx.project_dir)
    prompt = engine.prompt(match, args=args, agent_identity=agent_identity)

    # 追加可委派的同事名单
    roster_lines: list[str] = []
    for emp_name, emp in discovery.employees.items():
        if emp_name != name:
            label = emp.character_name or emp.effective_display_name
            roster_lines.append(f"- {emp_name}（{label}）：{emp.description}")
    if roster_lines:
        prompt += "\n\n---\n\n## 可委派的同事\n\n使用 delegate 工具调用他们。\n\n" + "\n".join(
            roster_lines
        )

    # 仅暴露 delegate + submit
    tool_schemas = employee_tools_to_schemas(["delegate"])

    use_model = model or match.model or "claude-sonnet-4-20250514"
    provider = detect_provider(use_model)
    is_anthropic = provider == Provider.ANTHROPIC

    task_text = args.get("task", "请开始执行上述任务。")
    messages: list[dict[str, Any]] = [{"role": "user", "content": task_text}]

    total_input = 0
    total_output = 0
    final_content = ""
    rounds = 0

    for rounds in range(_MAX_DELEGATION_ROUNDS):  # noqa: B007
        result = await aexecute_with_tools(
            system_prompt=prompt,
            messages=messages,
            tools=tool_schemas,
            api_key=None,
            model=use_model,
            max_tokens=4096,
        )
        total_input += result.input_tokens
        total_output += result.output_tokens

        if not result.has_tool_calls:
            final_content = result.content
            break

        # ── 构建 assistant 消息 + 处理 tool calls ──
        if is_anthropic:
            # Anthropic: content 数组包含 text + tool_use blocks
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

            # 处理每个 tool call → tool_result blocks
            tool_results: list[dict[str, Any]] = []
            finished = False
            for tc in result.tool_calls:
                if is_finish_tool(tc.name):
                    final_content = tc.arguments.get("result", result.content)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": final_content,
                    })
                    finished = True
                elif tc.name == "delegate":
                    logger.info("委派: %s → %s", name, tc.arguments.get("employee_name"))
                    delegate_result = await _delegate_employee(
                        ctx,
                        tc.arguments.get("employee_name", ""),
                        tc.arguments.get("task", ""),
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": delegate_result[:10000],
                    })
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": f"工具 '{tc.name}' 不可用，请使用 delegate 或 submit。",
                        "is_error": True,
                    })
            messages.append({"role": "user", "content": tool_results})
            if finished:
                break
        else:
            # OpenAI-compatible: assistant 有 tool_calls 数组，回复用 role=tool
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
                if is_finish_tool(tc.name):
                    final_content = tc.arguments.get("result", result.content)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": final_content,
                    })
                    finished = True
                elif tc.name == "delegate":
                    logger.info("委派: %s → %s", name, tc.arguments.get("employee_name"))
                    delegate_result = await _delegate_employee(
                        ctx,
                        tc.arguments.get("employee_name", ""),
                        tc.arguments.get("task", ""),
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": delegate_result[:10000],
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"工具 '{tc.name}' 不可用，请使用 delegate 或 submit。",
                    })
            if finished:
                break
    else:
        final_content = result.content or "达到最大委派轮次限制。"

    return {
        "employee": name,
        "prompt": prompt[:500],
        "output": final_content,
        "model": use_model,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "delegations": rounds,
    }


async def _execute_employee(
    ctx: _AppContext,
    name: str,
    args: dict[str, str],
    agent_id: int | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """执行单个员工."""
    from crew.discovery import discover_employees
    from crew.engine import CrewEngine

    discovery = discover_employees(project_dir=ctx.project_dir)
    match = discovery.get(name)

    if match is None:
        raise EmployeeNotFoundError(name)

    # 如果员工有 delegate 工具，使用带委派的 agent loop
    if "delegate" in (match.tools or []):
        return await _execute_employee_with_delegation(
            ctx, name, args, agent_id=agent_id, model=model,
        )

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

        use_model = model or match.model or "claude-sonnet-4-20250514"
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
    model: str | None = None,
) -> StreamingResponse:
    """SSE 流式执行单个员工."""
    import json as _json

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
        except Exception:
            pass

    engine = CrewEngine(project_dir=ctx.project_dir)
    prompt = engine.prompt(match, args=args, agent_identity=agent_identity)

    async def _generate():
        done_sent = False
        try:
            from crew.executor import aexecute_prompt

            use_model = model or match.model or "claude-sonnet-4-20250514"
            stream_iter = await asyncio.wait_for(
                aexecute_prompt(
                    system_prompt=prompt,
                    api_key=None,
                    model=use_model,
                    stream=True,
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

    # 加载飞书配置
    feishu_cfg = None
    try:
        from crew.feishu import load_feishu_config

        feishu_cfg = load_feishu_config(project_dir)
        if not (feishu_cfg.app_id and feishu_cfg.app_secret):
            feishu_cfg = None
    except Exception:
        pass

    app = create_webhook_app(
        project_dir=project_dir,
        token=token,
        config=config,
        cron_config=cron_config,
        cors_origins=cors_origins,
        feishu_config=feishu_cfg,
    )

    logger.info("启动 Webhook 服务器: %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
