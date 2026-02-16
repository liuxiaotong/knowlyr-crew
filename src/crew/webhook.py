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

        from crew.feishu_memory import FeishuChatStore

        chat_store_dir = (project_dir or Path(".")) / ".crew" / "feishu-chats"
        ctx.feishu_chat_store = FeishuChatStore(chat_store_dir)
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
        Route("/api/employees/{identifier}/prompt", endpoint=_make_handler(ctx, _handle_employee_prompt), methods=["GET"]),
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
            skip_paths=["/health", "/metrics", "/webhook/github"],
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
        self.feishu_chat_store = None  # FeishuChatStore, set by create_webhook_app


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


async def _handle_employee_prompt(request: Request, ctx: _AppContext) -> JSONResponse:
    """返回员工配置和渲染后的 system_prompt（供 knowlyr-id 调用）."""
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
    })


async def _handle_github(request: Request, ctx: _AppContext) -> JSONResponse:
    """处理 GitHub webhook."""
    from crew.webhook_config import (
        match_route,
        resolve_target_args,
        verify_github_signature,
    )

    body = await request.body()

    # 签名验证（未配置 secret 时记录警告并拒绝）
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

            loop = asyncio.get_running_loop()
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

    # 3. 验证 event token（未配置时跳过但记录警告）
    header = payload.get("header", {})
    event_token = header.get("token", payload.get("token", ""))
    if ctx.feishu_config.verification_token:
        from crew.feishu import verify_feishu_event

        if not verify_feishu_event(ctx.feishu_config.verification_token, event_token):
            return JSONResponse({"error": "invalid token"}, status_code=401)
    else:
        logger.warning("飞书 verification_token 未配置，跳过事件验证（建议设置 FEISHU_VERIFICATION_TOKEN）")

    # 4. 只处理消息事件
    event_type = header.get("event_type", "")
    if event_type != "im.message.receive_v1":
        logger.warning("飞书事件忽略: event_type=%s", event_type)
        return JSONResponse({"message": "ignored", "event_type": event_type})

    # 5. 解析消息
    from crew.feishu import parse_message_event

    msg_event = parse_message_event(payload)
    if msg_event is None:
        msg_type = payload.get("event", {}).get("message", {}).get("message_type", "?")
        logger.warning("飞书消息类型不支持: msg_type=%s", msg_type)
        return JSONResponse({"message": "unsupported message type"})

    logger.warning(
        "飞书消息: type=%s chat=%s text=%s image_key=%s mentions=%d",
        msg_event.msg_type, msg_event.chat_type,
        msg_event.text[:50] if msg_event.text else "(empty)",
        msg_event.image_key or "-",
        len(msg_event.mentions),
    )

    # 6. 去重
    if ctx.feishu_dedup and ctx.feishu_dedup.is_duplicate(msg_event.message_id):
        logger.warning("飞书消息去重: %s", msg_event.message_id)
        return JSONResponse({"message": "duplicate"})

    # 7. 后台处理（飞书要求 3s 内响应）
    asyncio.create_task(_feishu_dispatch(ctx, msg_event))

    return JSONResponse({"message": "ok"})


async def _find_recent_image_in_chat(
    token_mgr: Any,
    chat_id: str,
    sender_id: str,
    max_messages: int = 5,
) -> tuple[str, str] | None:
    """在群聊历史中查找同一发送者最近发的图片，返回 (image_key, message_id)."""
    import httpx
    import json as _json

    token = await token_mgr.get_token()
    url = "https://open.feishu.cn/open-apis/im/v1/messages"
    params = {
        "container_id_type": "chat",
        "container_id": chat_id,
        "page_size": max_messages,
        "sort_type": "ByCreateTimeDesc",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            params=params,
        )
        data = resp.json()

    if data.get("code", -1) != 0:
        logger.warning("查群消息失败: %s", data.get("msg", ""))
        return None

    for item in data.get("data", {}).get("items", []):
        if item.get("msg_type") != "image":
            continue
        msg_sender = item.get("sender", {}).get("id", "")
        if msg_sender != sender_id:
            continue
        try:
            content = _json.loads(item.get("body", {}).get("content", "{}"))
        except _json.JSONDecodeError:
            continue
        img_key = content.get("image_key", "")
        msg_id = item.get("message_id", "")
        if img_key and msg_id:
            return img_key, msg_id

    return None


async def _feishu_dispatch(ctx: _AppContext, msg_event: Any) -> None:
    """后台处理飞书消息：路由到员工 → 执行 → 回复卡片."""
    from crew.discovery import discover_employees
    from crew.feishu import (
        download_feishu_image,
        resolve_employee_from_mention,
        send_feishu_card,
        send_feishu_reply,
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

        emp = discovery.get(employee_name)

        # 图片 → 下载图片 + vision
        # 来源：(a) image/post 消息自带 image_key
        #       (b) 群聊文本消息 → 往前找同一发送者的最近图片
        image_key = msg_event.image_key
        image_message_id = msg_event.message_id

        if (
            not image_key
            and msg_event.chat_type == "group"
            and msg_event.msg_type == "text"
        ):
            # 群聊纯文本 @bot：尝试往前查最近一条同人图片
            try:
                found = await _find_recent_image_in_chat(
                    ctx.feishu_token_mgr,
                    msg_event.chat_id,
                    msg_event.sender_id,
                )
                if found:
                    image_key, image_message_id = found
            except Exception as exc:
                logger.warning("查找群聊近期图片失败: %s", exc)

        image_data: tuple[bytes, str] | None = None
        if image_key:
            try:
                image_data = await download_feishu_image(
                    ctx.feishu_token_mgr, image_key,
                    message_id=image_message_id,
                )
            except Exception as exc:
                logger.warning("飞书图片下载失败: %s", exc)
                await send_feishu_text(
                    ctx.feishu_token_mgr,
                    msg_event.chat_id,
                    "图片没下载下来，你描述一下？",
                )
                return
            # 图片消息的 text 为空，给个默认提示
            if not task_text:
                task_text = "Kai 发了一张图片，请看图片内容并回应。"

        # 执行员工 — 飞书实时聊天
        from crew.tool_schema import AGENT_TOOLS

        has_tools = any(
            t in AGENT_TOOLS
            for t in (emp.tools or [])
        ) if emp else False

        if has_tools:
            chat_context = (
                "你正在飞书上和 Kai 聊天。像平时一样自然回复。"
                "需要数据就调工具查，拿到数据用自己的话说，别搬 JSON。"
                "如果 Kai 在追问上一个话题，直接接着聊，不用重新查。"
                "只陈述工具返回的事实，没查过的事不要说，不要主动编造任何信息。"
            )
        else:
            chat_context = (
                "这是飞书实时聊天。你现在没有任何业务数据。"
                "你不知道：今天的会议安排、项目进度、客户状态、合同情况、同事的工作产出、任何具体数字。"
                "Kai 问这些就说「这个我得看看今天的数据才能说」。"
                "你可以聊：你的职能介绍、帮他想问题、帮他起草文字。"
            )

        # 加载对话历史
        message_history = None
        if ctx.feishu_chat_store:
            history_entries = ctx.feishu_chat_store.get_recent(
                msg_event.chat_id, limit=40,
            )
            if history_entries:
                message_history = [
                    {"role": e["role"], "content": e["content"]}
                    for e in history_entries
                ]

        # 如果没有工具且没有 message_history 格式的支持，回退到 prompt 注入
        if not has_tools and message_history:
            history_text = ctx.feishu_chat_store.format_for_prompt(
                msg_event.chat_id, limit=40,
            )
            if history_text:
                chat_context += (
                    "\n\n## 最近对话记录\n\n"
                    "以下是你和 Kai 最近的对话，用来保持上下文连贯。\n\n"
                    + history_text
                )
            message_history = None  # 无工具时不走 message_history 参数

        args = {"task": chat_context}
        if emp and emp.args:
            first_required = next((a for a in emp.args if a.required), None)
            if first_required:
                args[first_required.name] = chat_context

        # 把 Kai 的原话作为 user_message，让模型直接回复
        # 如果有图片，构建 multimodal content（OpenAI 格式）
        user_msg: str | list[dict[str, Any]] = task_text
        if image_data is not None:
            import base64 as _b64
            img_bytes, media_type = image_data
            b64 = _b64.b64encode(img_bytes).decode()
            user_msg = [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                {"type": "text", "text": task_text},
            ]

        result = await _execute_employee(
            ctx, employee_name, args, model=None,
            user_message=user_msg,
            message_history=message_history,
        )

        # 记录对话历史
        output_text = result.get("output", "") if isinstance(result, dict) else str(result)
        if ctx.feishu_chat_store:
            ctx.feishu_chat_store.append(msg_event.chat_id, "user", task_text)
            ctx.feishu_chat_store.append(msg_event.chat_id, "assistant", output_text)

        # 沉淀长期记忆（后台，不阻塞回复）
        if ctx.project_dir:
            try:
                from crew.feishu_memory import capture_feishu_memory

                capture_feishu_memory(
                    project_dir=ctx.project_dir,
                    employee_name=employee_name,
                    chat_id=msg_event.chat_id,
                    user_text=task_text,
                    assistant_text=output_text,
                )
            except Exception as e:
                logger.debug("飞书记忆沉淀失败: %s", e)

        # 发送回复（群聊用 reply thread，私聊用顶层消息）
        if msg_event.chat_type == "group":
            await send_feishu_reply(
                ctx.feishu_token_mgr,
                msg_event.message_id,
                output_text,
            )
        else:
            await send_feishu_text(
                ctx.feishu_token_mgr,
                msg_event.chat_id,
                output_text,
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
            error_text = "处理时出了点问题，请稍后再试。"
            if msg_event.chat_type == "group":
                await send_feishu_reply(
                    ctx.feishu_token_mgr,
                    msg_event.message_id,
                    error_text,
                )
            else:
                await send_feishu_text(
                    ctx.feishu_token_mgr,
                    msg_event.chat_id,
                    error_text,
                )
        except Exception:
            logger.warning("飞书错误回复发送失败", exc_info=True)


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
        elif record.target_type == "meeting":
            logger.info("执行 meeting [trace=%s] %s", trace_id, record.target_name)
            employees = [e.strip() for e in record.args.get("employees", "").split(",") if e.strip()]
            result = await _execute_meeting(
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
            result = await _execute_chain(ctx, task_id, steps)
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


# ── 异步委派 & 会议编排 ──


async def _tool_delegate_async(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """异步委派 — 立即返回 task_id，后台执行."""
    if ctx is None:
        return "错误: 上下文不可用"

    employee_name = args.get("employee_name", "")
    task_desc = args.get("task", "")

    from crew.discovery import discover_employees

    discovery = discover_employees(project_dir=ctx.project_dir)
    if discovery.get(employee_name) is None:
        available = ", ".join(sorted(discovery.employees.keys()))
        return f"错误：未找到员工 '{employee_name}'。可用员工：{available}"

    record = ctx.registry.create(
        trigger="delegate_async",
        target_type="employee",
        target_name=employee_name,
        args={"task": task_desc},
    )
    asyncio.create_task(_execute_task(ctx, record.task_id, agent_id=agent_id))
    return f"已异步委派给 {employee_name}。任务 ID: {record.task_id}"


async def _tool_check_task(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查询任务状态和结果."""
    if ctx is None:
        return "错误: 上下文不可用"

    task_id = args.get("task_id", "")
    record = ctx.registry.get(task_id)
    if record is None:
        return f"未找到任务: {task_id}"

    lines = [
        f"任务 ID: {record.task_id}",
        f"状态: {record.status}",
        f"类型: {record.target_type}",
        f"目标: {record.target_name}",
        f"创建: {record.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if record.completed_at:
        lines.append(f"完成: {record.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if record.status == "completed" and record.result:
        if record.target_type == "meeting":
            synthesis = record.result.get("synthesis", "")
            lines.append(f"\n会议综合结论:\n{synthesis[:1000]}")
        else:
            content = record.result.get("content", "")
            lines.append(f"\n执行结果:\n{content[:1000]}")
    elif record.status == "failed":
        lines.append(f"错误: {record.error}")

    return "\n".join(lines)


async def _tool_list_tasks(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """列出最近的任务."""
    if ctx is None:
        return "错误: 上下文不可用"

    status_filter = args.get("status")
    type_filter = args.get("type")
    limit = int(args.get("limit", 10))

    if status_filter:
        tasks = ctx.registry.list_by_status(status_filter, limit=limit)
    elif type_filter:
        tasks = ctx.registry.list_by_type(type_filter, limit=limit)
    else:
        tasks = ctx.registry.list_recent(n=limit)

    if not tasks:
        return "暂无任务记录。"

    _icons = {"pending": "⏳", "running": "▶️", "completed": "✅", "failed": "❌"}
    lines = [f"最近任务（共 {len(tasks)} 条）:"]
    for r in tasks:
        icon = _icons.get(r.status, "•")
        t = r.created_at.strftime("%m-%d %H:%M")
        lines.append(f"{icon} [{r.task_id}] {r.target_name} ({r.status}) - {t}")
    return "\n".join(lines)


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


async def _tool_organize_meeting(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """组织多员工会议（异步）."""
    if ctx is None:
        return "错误: 上下文不可用"

    employees_raw = args.get("employees", [])
    if isinstance(employees_raw, str):
        employees_list = [e.strip() for e in employees_raw.split(",") if e.strip()]
    else:
        employees_list = list(employees_raw)
    topic = args.get("topic", "")
    goal = args.get("goal", "")
    rounds = int(args.get("rounds", 2))

    if not employees_list or not topic:
        return "错误: 必须提供 employees 和 topic"

    from crew.discovery import discover_employees

    discovery = discover_employees(project_dir=ctx.project_dir)
    missing = [e for e in employees_list if discovery.get(e) is None]
    if missing:
        available = ", ".join(sorted(discovery.employees.keys()))
        return f"错误：未找到员工 {', '.join(missing)}。可用员工：{available}"

    meeting_name = f"{'、'.join(employees_list[:3])}{'等' if len(employees_list) > 3 else ''}会议"
    record = ctx.registry.create(
        trigger="organize_meeting",
        target_type="meeting",
        target_name=meeting_name,
        args={
            "employees": ",".join(employees_list),
            "topic": topic,
            "goal": goal,
            "rounds": str(rounds),
        },
    )
    asyncio.create_task(_execute_task(ctx, record.task_id, agent_id=agent_id))
    return (
        f"已组织会议。会议 ID: {record.task_id}\n"
        f"参会者: {', '.join(employees_list)}\n"
        f"议题: {topic}\n"
        f"轮次: {rounds}（异步执行中）"
    )


async def _tool_check_meeting(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查询会议进展（check_task 别名）."""
    return await _tool_check_task(args, agent_id=agent_id, ctx=ctx)


# ── Pipeline / Chain / 定时 / 文件 / 数据 / 日程 ──


async def _tool_run_pipeline(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """执行预定义流水线（异步）."""
    if ctx is None:
        return "错误: 上下文不可用"

    name = args.get("name", "")
    pipeline_args = args.get("args", {})
    if isinstance(pipeline_args, str):
        import json as _json
        try:
            pipeline_args = _json.loads(pipeline_args)
        except Exception:
            pipeline_args = {}

    from crew.pipeline import discover_pipelines

    pipelines = discover_pipelines(project_dir=ctx.project_dir)
    if name not in pipelines:
        available = ", ".join(sorted(pipelines.keys()))
        return f"错误：未找到流水线 '{name}'。可用：{available}"

    record = ctx.registry.create(
        trigger="tool",
        target_type="pipeline",
        target_name=name,
        args={k: str(v) for k, v in pipeline_args.items()},
    )
    asyncio.create_task(_execute_task(ctx, record.task_id, agent_id=agent_id))
    return f"已启动流水线 {name}。任务 ID: {record.task_id}（异步执行中）"


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


async def _tool_delegate_chain(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """顺序委派链（异步）."""
    if ctx is None:
        return "错误: 上下文不可用"

    steps = args.get("steps", [])
    if not steps or not isinstance(steps, list):
        return "错误: 必须提供 steps 列表"

    from crew.discovery import discover_employees

    discovery = discover_employees(project_dir=ctx.project_dir)
    missing = [s["employee_name"] for s in steps if discovery.get(s.get("employee_name", "")) is None]
    if missing:
        return f"错误：未找到员工 {', '.join(missing)}"

    names = [s["employee_name"] for s in steps]
    chain_name = " → ".join(names)
    record = ctx.registry.create(
        trigger="delegate_chain",
        target_type="chain",
        target_name=chain_name,
        args={"steps_json": __import__("json").dumps(steps, ensure_ascii=False)},
    )
    asyncio.create_task(_execute_task(ctx, record.task_id, agent_id=agent_id))
    return f"已启动委派链: {chain_name}。任务 ID: {record.task_id}（异步执行中）"


async def _tool_schedule_task(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """创建定时任务."""
    if ctx is None or ctx.scheduler is None:
        return "错误: 定时任务调度器未启用"

    name = args.get("name", "")
    cron_expr = args.get("cron", "")
    emp_name = args.get("employee_name", "")
    task_desc = args.get("task", "")

    if not all([name, cron_expr, emp_name, task_desc]):
        return "错误: 必须提供 name、cron、employee_name、task"

    # 检查 cron 表达式
    try:
        from croniter import croniter
        if not croniter.is_valid(cron_expr):
            return f"错误: 无效的 cron 表达式 '{cron_expr}'"
    except ImportError:
        return "错误: croniter 未安装"

    # 检查名称冲突
    existing = [s.name for s in ctx.scheduler.schedules]
    if name in existing:
        return f"错误: 已存在同名定时任务 '{name}'"

    from crew.cron_config import CronSchedule

    schedule = CronSchedule(
        name=name,
        cron=cron_expr,
        target_type="employee",
        target_name=emp_name,
        args={"task": task_desc, "format": "memo"},
    )
    await ctx.scheduler.add_schedule(schedule)
    return f"已创建定时任务 '{name}'（{cron_expr} → {emp_name}）"


async def _tool_list_schedules(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """列出定时任务."""
    if ctx is None or ctx.scheduler is None:
        return "定时任务调度器未启用，暂无定时任务。"

    runs = ctx.scheduler.get_next_runs()
    if not runs:
        return "暂无定时任务。"

    lines = [f"定时任务（共 {len(runs)} 个）:"]
    for r in runs:
        if "error" in r:
            lines.append(f"  ❌ {r['name']}: {r['error']}")
        else:
            lines.append(
                f"  ⏰ {r['name']} ({r['cron']}) → {r['target_type']}/{r['target_name']}"
                f"  下次: {r['next_run']}"
            )
    return "\n".join(lines)


async def _tool_cancel_schedule(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """取消定时任务."""
    if ctx is None or ctx.scheduler is None:
        return "错误: 定时任务调度器未启用"

    name = args.get("name", "")
    if not name:
        return "错误: 必须提供任务名称"

    removed = await ctx.scheduler.remove_schedule(name)
    if removed:
        return f"已取消定时任务 '{name}'"
    return f"未找到定时任务 '{name}'"


async def _tool_agent_file_read(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """读取项目目录内的文件."""
    if ctx is None:
        return "错误: 上下文不可用"

    rel_path = args.get("path", "")
    if not rel_path:
        return "错误: 必须提供 path"

    project_dir = ctx.project_dir or Path(".")
    full_path = (project_dir / rel_path).resolve()

    # 安全检查
    if not full_path.is_relative_to(project_dir.resolve()):
        return "错误: 路径不在项目目录内"

    if not full_path.is_file():
        return f"错误: 文件不存在 — {rel_path}"

    start = args.get("start_line")
    end = args.get("end_line")

    try:
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)

        if start is not None:
            start = max(1, int(start))
        else:
            start = 1
        if end is not None:
            end = min(total, int(end))
        else:
            end = min(total, start + 4999)  # 最多 5000 行

        selected = lines[start - 1 : end]
        numbered = [f"{start + i:5d}│{line}" for i, line in enumerate(selected)]
        header = f"文件: {rel_path} ({total} 行, 显示 {start}-{end})\n"
        return header + "\n".join(numbered)
    except Exception as e:
        return f"读取失败: {e}"


async def _tool_agent_file_grep(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """在项目目录内搜索文件内容."""
    if ctx is None:
        return "错误: 上下文不可用"

    pattern = args.get("pattern", "")
    if not pattern:
        return "错误: 必须提供 pattern"

    project_dir = ctx.project_dir or Path(".")
    search_path = project_dir
    rel = args.get("path", "")
    if rel:
        search_path = (project_dir / rel).resolve()
        if not search_path.is_relative_to(project_dir.resolve()):
            return "错误: 路径不在项目目录内"

    file_pattern = args.get("file_pattern", "")

    import subprocess

    cmd = ["grep", "-rn", "--max-count=200", "--include", file_pattern or "*", pattern, str(search_path)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10,
            cwd=str(project_dir),
        )
        output = result.stdout.strip()
        if not output:
            return f"未找到匹配: {pattern}"

        # 将绝对路径替换为相对路径
        output = output.replace(str(project_dir) + "/", "")
        lines = output.splitlines()
        if len(lines) > 100:
            return "\n".join(lines[:100]) + f"\n\n... 共 {len(lines)} 条匹配（已截断前 100 条）"
        return output
    except subprocess.TimeoutExpired:
        return "搜索超时（10s），请缩小搜索范围"
    except Exception as e:
        return f"搜索失败: {e}"


async def _tool_query_data(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查询细粒度业务数据."""
    metric = args.get("metric", "")
    period = args.get("period", "week")
    group_by = args.get("group_by", "")

    import httpx

    params = {"metric": metric, "period": period}
    if group_by:
        params["group_by"] = group_by

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{_ID_API_BASE}/api/stats/query",
                params=params,
                headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
            )
            if resp.status_code == 200:
                return resp.text
            return f"查询失败 (HTTP {resp.status_code}): {resp.text[:200]}"
    except Exception as e:
        return f"查询失败: {e}"


async def _tool_find_free_time(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查询飞书用户共同空闲时间."""
    if ctx is None or ctx.feishu_token_mgr is None:
        return "错误: 飞书未配置"

    user_ids = args.get("user_ids", [])
    if isinstance(user_ids, str):
        user_ids = [u.strip() for u in user_ids.split(",") if u.strip()]
    days = int(args.get("days", 7))
    duration = int(args.get("duration_minutes", 60))

    if not user_ids:
        return "错误: 必须提供 user_ids"

    from crew.feishu import get_freebusy

    import time as _time

    now = int(_time.time())
    end = now + days * 86400

    try:
        busy_data = await get_freebusy(ctx.feishu_token_mgr, user_ids, now, end)
        if "error" in busy_data:
            return f"查询忙闲失败: {busy_data['error']}"

        # 计算空闲交集
        from datetime import datetime, timedelta

        # busy_data["freebusy_list"] = [{user_id, busy: [{start_time, end_time}]}]
        all_busy = []
        for user_info in busy_data.get("freebusy_list", []):
            for slot in user_info.get("busy", []):
                all_busy.append((int(slot["start_time"]), int(slot["end_time"])))

        # 合并重叠时段
        all_busy.sort()
        merged = []
        for s, e in all_busy:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # 在工作时间（9-18点）找空闲
        free_slots = []
        check_time = now
        for day_offset in range(days):
            day_start = datetime.fromtimestamp(now) + timedelta(days=day_offset)
            work_start = day_start.replace(hour=9, minute=0, second=0).timestamp()
            work_end = day_start.replace(hour=18, minute=0, second=0).timestamp()

            cursor = int(work_start)
            for bs, be in merged:
                if be <= cursor or bs >= work_end:
                    continue
                if bs > cursor:
                    gap = bs - cursor
                    if gap >= duration * 60:
                        free_slots.append((cursor, bs))
                cursor = max(cursor, int(be))
            if cursor < work_end and (work_end - cursor) >= duration * 60:
                free_slots.append((cursor, int(work_end)))

        if not free_slots:
            return f"未来 {days} 天内没有找到 {duration} 分钟的共同空闲时段。"

        lines = [f"共同空闲时段（{duration} 分钟以上）:"]
        for s, e in free_slots[:10]:
            st = datetime.fromtimestamp(s)
            et = datetime.fromtimestamp(e)
            lines.append(f"  {st.strftime('%m-%d %H:%M')} ~ {et.strftime('%H:%M')}")
        if len(free_slots) > 10:
            lines.append(f"  ... 共 {len(free_slots)} 个时段")
        return "\n".join(lines)
    except Exception as e:
        return f"查询空闲时间失败: {e}"


_MAX_TOOL_ROUNDS = 10


# ── Tool handlers（调用 knowlyr-id API）──

import os as _os

_ID_API_BASE = _os.environ.get("KNOWLYR_ID_API", "https://id.knowlyr.com")
_ID_API_TOKEN = _os.environ.get("AGENT_API_TOKEN", "")
_GITHUB_TOKEN = _os.environ.get("GITHUB_TOKEN", "")
_GITHUB_API_BASE = "https://api.github.com"
_GITHUB_REPO_RE = __import__("re").compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def _is_valid_github_repo(repo: str) -> bool:
    """验证 GitHub repo 格式，防止路径穿越."""
    return bool(_GITHUB_REPO_RE.match(repo))
_NOTION_API_KEY = _os.environ.get("NOTION_API_KEY", "")
_NOTION_API_BASE = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"


async def _tool_query_stats(args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None) -> str:
    """调用 knowlyr-id /api/stats/briefing."""
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_ID_API_BASE}/api/stats/briefing",
            headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
        )
        return resp.text


async def _tool_send_message(args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None) -> str:
    """调用 knowlyr-id /api/messages/agent-send."""
    import httpx

    sender = agent_id or args.get("sender_id", 0)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{_ID_API_BASE}/api/messages/agent-send",
            json={
                "sender_id": sender,
                "recipient_id": args.get("recipient_id"),
                "content": args.get("content", ""),
            },
            headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
        )
        return resp.text


async def _tool_list_agents(args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None) -> str:
    """调用 knowlyr-id /api/agents."""
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_ID_API_BASE}/api/agents",
            headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
        )
        return resp.text


async def _tool_web_search(args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None) -> str:
    """搜索互联网（Bing cn）."""
    import re

    import httpx

    query = args.get("query", "")
    max_results = min(args.get("max_results", 5), 10)
    if not query:
        return "错误：query 不能为空"

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://cn.bing.com/search",
                params={"q": query, "count": max_results},
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                    "Accept-Language": "zh-CN,zh;q=0.9",
                },
            )

        results: list[str] = []
        for block in re.finditer(r'<li class="b_algo".*?</li>', resp.text, re.DOTALL):
            if len(results) >= max_results:
                break
            title_m = re.search(
                r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', block.group(), re.DOTALL,
            )
            snippet_m = re.search(r"<p[^>]*>(.*?)</p>", block.group(), re.DOTALL)
            if title_m:
                href = title_m.group(1)
                title = re.sub(r"<[^>]+>", "", title_m.group(2)).strip()
                snippet = re.sub(r"<[^>]+>", "", snippet_m.group(1)).strip() if snippet_m else ""
                if title or snippet:
                    results.append(f"{title}\n{snippet}\n{href}")

        if not results:
            return f"未找到关于「{query}」的搜索结果"
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"搜索失败: {e}"


async def _tool_create_note(args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None) -> str:
    """保存备忘/笔记到 .crew/notes/."""
    import re
    from datetime import datetime

    title = args.get("title", "untitled")
    content = args.get("content", "")
    tags = args.get("tags", "")

    if not content:
        return "错误：content 不能为空"

    # 确定项目目录
    project_dir = ctx.project_dir if ctx and ctx.project_dir else Path(".")

    # sanitize filename
    safe_title = re.sub(r"[^\w\u4e00-\u9fff-]", "-", title)[:60].strip("-")
    date_str = datetime.now().strftime("%Y%m%d-%H%M")
    filename = f"{date_str}-{safe_title}.md"

    notes_dir = project_dir / ".crew" / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    # frontmatter + content
    lines = [
        "---",
        f"title: {title}",
        f"date: {datetime.now().isoformat()}",
    ]
    if tags:
        lines.append(f"tags: [{tags}]")
    lines.extend(["---", "", content])

    note_path = notes_dir / filename
    note_path.write_text("\n".join(lines), encoding="utf-8")
    return f"笔记已保存: {filename}"


async def _tool_lookup_user(args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None) -> str:
    """按昵称查用户详情."""
    import httpx
    name = args.get("name", "")
    if not name:
        return "错误：需要 name 参数"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_ID_API_BASE}/api/stats/user",
            params={"q": name},
            headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
        )
        return resp.text


async def _tool_query_agent_work(args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None) -> str:
    """查 AI 同事最近工作记录."""
    import httpx
    name = args.get("name", "")
    days = args.get("days", 3)
    if not name:
        return "错误：需要 name 参数"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_ID_API_BASE}/api/stats/agent-work",
            params={"name": name, "days": days},
            headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
        )
        return resp.text


async def _tool_read_notes(args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None) -> str:
    """列出最近笔记，可选按关键词过滤."""
    keyword = args.get("keyword", "")
    limit = min(args.get("limit", 10), 20)

    notes_dir = (ctx.project_dir if ctx and ctx.project_dir else Path(".")) / ".crew" / "notes"
    if not notes_dir.exists():
        return "暂无笔记"

    files = sorted(notes_dir.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
    results = []
    for f in files:
        if len(results) >= limit:
            break
        content = f.read_text(encoding="utf-8")
        if keyword and keyword.lower() not in content.lower():
            continue
        results.append(f"【{f.stem}】\n{content[:200]}")

    return "\n---\n".join(results) if results else "没有匹配的笔记"


async def _tool_read_messages(args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None) -> str:
    """查 Kai 的未读消息概要."""
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_ID_API_BASE}/api/stats/unread",
            params={"user_id": 1},
            headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
        )
        return resp.text


async def _tool_get_system_health(args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None) -> str:
    """查服务器健康状态."""
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_ID_API_BASE}/api/stats/system-health",
            headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
        )
        return resp.text


async def _tool_mark_read(args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None) -> str:
    """标消息已读."""
    import httpx
    mark_all = args.get("all", False)
    sender_name = args.get("sender_name", "")

    payload: dict[str, Any] = {"user_id": 1}

    if mark_all:
        payload["all"] = True
    elif sender_name:
        # 先查 sender_id
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{_ID_API_BASE}/api/stats/user",
                params={"q": sender_name},
                headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
            )
            try:
                users = resp.json()
                if not users:
                    return f"找不到用户「{sender_name}」"
                payload["sender_id"] = users[0]["id"]
            except Exception:
                return f"查询用户失败: {resp.text[:200]}"
    else:
        return "需要指定 sender_name 或 all=true"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{_ID_API_BASE}/api/messages/mark-read-batch",
            json=payload,
            headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
        )
        return resp.text


async def _tool_update_agent(args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None) -> str:
    """管理 AI 同事."""
    import httpx
    target_id = args.get("agent_id")
    if not target_id:
        return "需要 agent_id 参数"

    update_data: dict[str, Any] = {}
    if args.get("status"):
        update_data["agent_status"] = args["status"]
    if args.get("memory"):
        update_data["memory"] = args["memory"]

    if not update_data:
        return "没有要更新的内容（需要 status 或 memory）"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.put(
            f"{_ID_API_BASE}/api/agents/{target_id}",
            json=update_data,
            headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
        )
        return resp.text


async def _tool_create_feishu_event(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """在飞书日历创建日程."""
    from datetime import datetime, timedelta, timezone as _tz

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置，无法创建日程。"

    summary = (args.get("summary") or "").strip()
    date_str = (args.get("date") or "").strip()
    start_hour = args.get("start_hour", 9)
    start_minute = args.get("start_minute", 0)
    duration = args.get("duration_minutes", 60)
    description = args.get("description", "")

    if not summary:
        return "缺少日程标题。"
    if not date_str:
        return "缺少日期。"

    try:
        start_hour = int(start_hour)
        start_minute = int(start_minute)
        duration = int(duration)
    except (TypeError, ValueError):
        return "时间参数格式不对。"

    tz_cn = _tz(timedelta(hours=8))
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        start_time = d.replace(hour=start_hour, minute=start_minute, tzinfo=tz_cn)
    except ValueError:
        return f"日期格式不对: {date_str}，需要 YYYY-MM-DD。"

    end_time = start_time + timedelta(minutes=max(duration, 15))

    from crew.feishu import add_attendees_to_event, create_calendar_event

    cal_id = (ctx.feishu_config.calendar_id if ctx.feishu_config else "") or ""
    result = await create_calendar_event(
        token_mgr=ctx.feishu_token_mgr,
        summary=summary,
        start_timestamp=int(start_time.timestamp()),
        end_timestamp=int(end_time.timestamp()),
        description=description,
        calendar_id=cal_id,
    )

    if result.get("ok"):
        event_id = result.get("event_id", "")
        # 自动邀请日历所有者（让日程出现在他/她的日历上）
        owner_id = (ctx.feishu_config.owner_open_id if ctx.feishu_config else "") or ""
        if owner_id and event_id and cal_id:
            att_result = await add_attendees_to_event(
                token_mgr=ctx.feishu_token_mgr,
                calendar_id=cal_id,
                event_id=event_id,
                attendee_open_ids=[owner_id],
            )
            if not att_result.get("ok"):
                logger.warning("日程创建成功但邀请参与者失败: %s", att_result.get("error"))
        end_str = end_time.strftime("%H:%M")
        start_str = start_time.strftime("%H:%M")
        return f"日程已创建：{date_str} {start_str}-{end_str}《{summary}》"
    else:
        return f"创建失败: {result.get('error', '未知错误')}"


# ── 飞书日程查询/删除 ──


async def _tool_read_feishu_calendar(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查看飞书日历日程."""
    import os
    from datetime import datetime, timedelta, timezone as _tz

    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    tz_cn = _tz(timedelta(hours=8))
    date_str = (args.get("date") or "").strip()
    days = max(int(args.get("days", 1)), 1)

    if date_str:
        try:
            start_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=tz_cn)
        except ValueError:
            return f"日期格式不对: {date_str}，需要 YYYY-MM-DD。"
    else:
        start_dt = datetime.now(tz_cn).replace(hour=0, minute=0, second=0, microsecond=0)

    end_dt = start_dt + timedelta(days=days)

    cal_id = (ctx.feishu_config.calendar_id if ctx.feishu_config else "") or os.environ.get("FEISHU_CALENDAR_ID", "")
    if not cal_id:
        return "未配置日历 ID。"

    token = await ctx.feishu_token_mgr.get_token()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"https://open.feishu.cn/open-apis/calendar/v4/calendars/{cal_id}/events",
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "start_time": str(int(start_dt.timestamp())),
                    "end_time": str(int(end_dt.timestamp())),
                    "page_size": 50,
                },
            )
            data = resp.json()

        if data.get("code") != 0:
            return f"查询失败: {data.get('msg', '未知错误')}"

        items = data.get("data", {}).get("items", [])
        if not items:
            date_range = date_str or start_dt.strftime("%Y-%m-%d")
            if days > 1:
                date_range += f" ~ {end_dt.strftime('%Y-%m-%d')}"
            return f"{date_range} 没有日程。"

        lines = []
        for ev in items:
            summary = ev.get("summary", "无标题")
            event_id = ev.get("event_id", "")
            st = ev.get("start_time", {})
            et = ev.get("end_time", {})
            # timestamp or date
            if st.get("timestamp"):
                s = datetime.fromtimestamp(int(st["timestamp"]), tz=tz_cn)
                e = datetime.fromtimestamp(int(et.get("timestamp", st["timestamp"])), tz=tz_cn)
                time_str = f"{s.strftime('%m-%d %H:%M')}-{e.strftime('%H:%M')}"
            elif st.get("date"):
                time_str = f"{st['date']} 全天"
            else:
                time_str = "时间未知"
            lines.append(f"{time_str} {summary} [event_id={event_id}]")

        return "\n".join(lines)
    except Exception as e:
        return f"查询失败: {e}"


async def _tool_delete_feishu_event(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """删除飞书日历日程."""
    import os

    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    event_id = (args.get("event_id") or "").strip()
    if not event_id:
        return "需要 event_id 参数。先用 read_feishu_calendar 查到 event_id。"

    cal_id = (ctx.feishu_config.calendar_id if ctx.feishu_config else "") or os.environ.get("FEISHU_CALENDAR_ID", "")
    if not cal_id:
        return "未配置日历 ID。"

    token = await ctx.feishu_token_mgr.get_token()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.delete(
                f"https://open.feishu.cn/open-apis/calendar/v4/calendars/{cal_id}/events/{event_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            data = resp.json()

        if data.get("code") == 0:
            return f"日程已删除 (event_id={event_id})。"
        return f"删除失败: {data.get('msg', '未知错误')}"
    except Exception as e:
        return f"删除失败: {e}"


# ── 飞书待办任务 ──


async def _tool_create_feishu_task(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """在飞书创建待办任务."""
    from datetime import datetime, timedelta, timezone as _tz

    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    summary = (args.get("summary") or "").strip()
    if not summary:
        return "需要任务标题。"

    due_str = (args.get("due") or "").strip()
    description = args.get("description", "")

    body: dict[str, Any] = {"summary": summary}
    if description:
        body["description"] = description
    if due_str:
        tz_cn = _tz(timedelta(hours=8))
        try:
            due_dt = datetime.strptime(due_str, "%Y-%m-%d").replace(hour=23, minute=59, tzinfo=tz_cn)
            body["due"] = {"timestamp": str(int(due_dt.timestamp())), "is_all_day": True}
        except ValueError:
            return f"截止日期格式不对: {due_str}，需要 YYYY-MM-DD。"

    token = await ctx.feishu_token_mgr.get_token()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://open.feishu.cn/open-apis/task/v2/tasks",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=body,
            )
            data = resp.json()

        if data.get("code") == 0:
            task = data.get("data", {}).get("task", {})
            task_id = task.get("guid", "")
            due_info = f"，截止 {due_str}" if due_str else ""
            return f"待办已创建：{summary}{due_info} [task_id={task_id}]"
        return f"创建失败: {data.get('msg', '未知错误')}"
    except Exception as e:
        return f"创建失败: {e}"


async def _tool_list_feishu_tasks(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查看飞书待办任务列表."""
    from datetime import datetime, timedelta, timezone as _tz

    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    limit = min(int(args.get("limit", 20)), 50)

    token = await ctx.feishu_token_mgr.get_token()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://open.feishu.cn/open-apis/task/v1/tasks",
                headers={"Authorization": f"Bearer {token}"},
                params={"page_size": limit},
            )
            data = resp.json()

        if data.get("code") != 0:
            return f"查询失败: {data.get('msg', '未知错误')}"

        items = data.get("data", {}).get("items", [])
        if not items:
            return "没有待办任务。"

        tz_cn = _tz(timedelta(hours=8))
        lines = []
        for task in items:
            summary = task.get("summary", "无标题")
            complete_time = task.get("complete_time", "0")
            status = "✅" if complete_time and complete_time != "0" else "⬜"
            due = task.get("due", {})
            due_str = ""
            if due and due.get("time") and due["time"] != "0":
                due_dt = datetime.fromtimestamp(int(due["time"]), tz=tz_cn)
                due_str = f" 截止{due_dt.strftime('%m-%d')}"
            task_id = task.get("id", "")
            lines.append(f"{status} {summary}{due_str} [task_id={task_id}]")

        return "\n".join(lines)
    except Exception as e:
        return f"查询失败: {e}"


async def _tool_complete_feishu_task(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """完成飞书待办任务."""
    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    task_id = (args.get("task_id") or "").strip()
    if not task_id:
        return "需要 task_id。先用 list_feishu_tasks 查看任务列表。"

    token = await ctx.feishu_token_mgr.get_token()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"https://open.feishu.cn/open-apis/task/v1/tasks/{task_id}/complete",
                headers={"Authorization": f"Bearer {token}"},
            )
            data = resp.json()

        if data.get("code") == 0:
            return f"任务已完成 ✅ [task_id={task_id}]"
        return f"操作失败: {data.get('msg', '未知错误')}"
    except Exception as e:
        return f"操作失败: {e}"


async def _tool_delete_feishu_task(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """删除飞书待办任务."""
    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    task_id = (args.get("task_id") or "").strip()
    if not task_id:
        return "需要 task_id。先用 list_feishu_tasks 查看任务列表。"

    token = await ctx.feishu_token_mgr.get_token()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.delete(
                f"https://open.feishu.cn/open-apis/task/v1/tasks/{task_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            data = resp.json()

        if data.get("code") == 0:
            return f"任务已删除 [task_id={task_id}]"
        return f"删除失败: {data.get('msg', '未知错误')}"
    except Exception as e:
        return f"删除失败: {e}"


async def _tool_update_feishu_task(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """更新飞书待办任务."""
    from datetime import datetime, timedelta, timezone as _tz

    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    task_id = (args.get("task_id") or "").strip()
    if not task_id:
        return "需要 task_id。先用 list_feishu_tasks 查看任务列表。"

    summary = (args.get("summary") or "").strip()
    due_str = (args.get("due") or "").strip()
    description = args.get("description")

    body: dict[str, Any] = {}
    update_fields: list[str] = []
    if summary:
        body["summary"] = summary
        update_fields.append("summary")
    if description is not None and description != "":
        body["description"] = description
        update_fields.append("description")
    if due_str:
        tz_cn = _tz(timedelta(hours=8))
        try:
            due_dt = datetime.strptime(due_str, "%Y-%m-%d").replace(hour=23, minute=59, tzinfo=tz_cn)
            body["due"] = {"time": str(int(due_dt.timestamp()))}
            update_fields.append("due")
        except ValueError:
            return f"截止日期格式不对: {due_str}，需要 YYYY-MM-DD。"

    if not body:
        return "需要至少一个要更新的字段（summary/due/description）。"

    token = await ctx.feishu_token_mgr.get_token()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.patch(
                f"https://open.feishu.cn/open-apis/task/v1/tasks/{task_id}",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={"task": body, "update_fields": update_fields},
            )
            data = resp.json()

        if data.get("code") == 0:
            parts = []
            if summary:
                parts.append(f"标题→{summary}")
            if due_str:
                parts.append(f"截止→{due_str}")
            if description is not None and description != "":
                parts.append("描述已更新")
            return f"任务已更新: {', '.join(parts)} [task_id={task_id}]"
        return f"更新失败: {data.get('msg', '未知错误')}"
    except Exception as e:
        return f"更新失败: {e}"


async def _tool_feishu_chat_history(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """读取飞书群/会话最近消息."""
    import json as _json
    from datetime import datetime, timedelta, timezone as _tz

    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    chat_id = (args.get("chat_id") or "").strip()
    if not chat_id:
        return "需要 chat_id。先用 list_feishu_groups 查群列表。"

    limit = min(int(args.get("limit", 10)), 50)
    token = await ctx.feishu_token_mgr.get_token()
    tz_cn = _tz(timedelta(hours=8))
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://open.feishu.cn/open-apis/im/v1/messages",
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "container_id_type": "chat",
                    "container_id": chat_id,
                    "page_size": limit,
                    "sort_type": "ByCreateTimeDesc",
                },
            )
            data = resp.json()

        if data.get("code") != 0:
            return f"查询失败: {data.get('msg', '未知错误')}"

        items = data.get("data", {}).get("items", [])
        if not items:
            return "没有消息记录。"

        lines = []
        for msg in items:
            sender = msg.get("sender", {}).get("id", "?")
            msg_type = msg.get("msg_type", "text")
            create_time = msg.get("create_time", "")
            time_str = ""
            if create_time:
                try:
                    dt = datetime.fromtimestamp(int(create_time) / 1000, tz=tz_cn)
                    time_str = dt.strftime("%m-%d %H:%M")
                except (ValueError, OSError):
                    pass

            body = msg.get("body", {}).get("content", "")
            try:
                content_obj = _json.loads(body) if body else {}
                text = content_obj.get("text", "")
            except _json.JSONDecodeError:
                text = body

            if msg_type == "image":
                text = "[图片]"
            elif msg_type == "file":
                text = "[文件]"
            elif msg_type == "sticker":
                text = "[表情]"
            elif not text:
                text = f"[{msg_type}]"

            # 截断长消息
            if len(text) > 100:
                text = text[:100] + "…"
            lines.append(f"[{time_str}] {sender}: {text}")

        return "\n".join(lines)
    except Exception as e:
        return f"查询失败: {e}"


# ── 天气工具 ──

# 中国主要城市代码映射（中国气象局编码）
_CITY_CODES: dict[str, str] = {
    "北京": "101010100", "上海": "101020100", "广州": "101280101",
    "深圳": "101280601", "杭州": "101210101", "南京": "101190101",
    "成都": "101270101", "重庆": "101040100", "武汉": "101200101",
    "西安": "101110101", "苏州": "101190401", "天津": "101030100",
    "长沙": "101250101", "郑州": "101180101", "青岛": "101120201",
    "大连": "101070201", "宁波": "101210401", "厦门": "101230201",
    "合肥": "101220101", "昆明": "101290101", "哈尔滨": "101050101",
    "沈阳": "101070101", "济南": "101120101", "福州": "101230101",
    "南昌": "101240101", "长春": "101060101", "贵阳": "101260101",
    "石家庄": "101090101", "太原": "101100101", "南宁": "101300101",
    "海口": "101310101", "兰州": "101160101", "银川": "101170101",
    "西宁": "101150101", "拉萨": "101140101", "乌鲁木齐": "101130101",
    "呼和浩特": "101080101", "珠海": "101280701", "无锡": "101190201",
    "东莞": "101281601", "佛山": "101280800", "温州": "101210701",
    "常州": "101191101", "泉州": "101230501", "烟台": "101120501",
    "惠州": "101280301", "嘉兴": "101210301", "中山": "101281701",
    "台州": "101210601", "绍兴": "101210501", "潍坊": "101120601",
    "金华": "101210901", "保定": "101090201", "芜湖": "101220301",
    "三亚": "101310201", "洛阳": "101180901", "桂林": "101300501",
    "襄阳": "101200201", "徐州": "101190801", "扬州": "101190601",
}


async def _tool_weather(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查天气（国内城市）."""
    import httpx

    city = (args.get("city") or "").strip().replace("市", "").replace("省", "")
    if not city:
        return "需要城市名，如：上海、北京、杭州。"

    code = _CITY_CODES.get(city)
    if not code:
        avail = "、".join(list(_CITY_CODES.keys())[:20]) + "…"
        return f"暂不支持「{city}」，支持的城市：{avail}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"http://t.weather.itboy.net/api/weather/city/{code}")
            data = resp.json()

        if data.get("status") != 200:
            return f"查询失败: {data.get('message', '未知错误')}"

        info = data.get("data", {})
        city_name = data.get("cityInfo", {}).get("city", city)
        now_temp = info.get("wendu", "?")
        humidity = info.get("shidu", "?")
        quality = info.get("quality", "")
        pm25 = info.get("pm25", "")

        forecast = info.get("forecast", [])
        lines = [f"{city_name} 当前 {now_temp}℃，湿度 {humidity}"]
        if quality:
            lines[0] += f"，空气{quality}"
            if pm25:
                lines[0] += f"(PM2.5:{pm25})"

        days = min(int(args.get("days", 3)), 7)
        for day in forecast[:days]:
            high = day.get("high", "").replace("高温 ", "")
            low = day.get("low", "").replace("低温 ", "")
            weather_type = day.get("type", "")
            wind = day.get("fx", "")
            wind_level = day.get("fl", "")
            date = day.get("ymd", "")
            week = day.get("week", "")
            lines.append(f"{date}({week}) {weather_type} {low}~{high} {wind}{wind_level}")

        return "\n".join(lines)
    except Exception as e:
        return f"天气查询失败: {e}"


async def _tool_exchange_rate(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查汇率."""
    import httpx

    base = (args.get("from") or args.get("base") or "USD").upper().strip()
    targets = (args.get("to") or "CNY").upper().strip()
    target_list = [t.strip() for t in targets.split(",") if t.strip()]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"https://api.exchangerate-api.com/v4/latest/{base}")
            data = resp.json()

        if "rates" not in data:
            return f"查询失败: 不支持货币 {base}"

        rates = data["rates"]
        lines = [f"基准: 1 {base}"]
        for t in target_list:
            rate = rates.get(t)
            if rate is not None:
                lines.append(f"= {rate} {t}")
            else:
                lines.append(f"{t}: 不支持")
        return "\n".join(lines)
    except Exception as e:
        return f"汇率查询失败: {e}"


async def _tool_stock_price(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查股价（A股/美股）."""
    import re

    import httpx

    symbol = (args.get("symbol") or "").strip()
    if not symbol:
        return "需要股票代码，如：sh600519（茅台）、gb_aapl（苹果）。"

    # 规范化：纯数字默认为沪市A股
    sym = symbol.lower()
    if re.match(r"^\d{6}$", sym):
        sym = f"sh{sym}" if sym.startswith("6") else f"sz{sym}"
    elif re.match(r"^[a-z]{1,5}$", sym):
        sym = f"gb_{sym}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"https://hq.sinajs.cn/list={sym}",
                headers={"Referer": "https://finance.sina.com.cn"},
            )
        text = resp.text.strip()
        if not text or '=""' in text:
            return f"未找到股票: {symbol}"

        # 解析 Sina 行情数据
        m = re.search(r'="(.+)"', text)
        if not m:
            return f"数据解析失败: {symbol}"

        fields = m.group(1).split(",")
        if sym.startswith("gb_"):
            # 美股格式: 名称,现价,涨跌幅%,时间,涨跌额,开盘,最高,最低,...
            if len(fields) < 4:
                return f"数据不完整: {symbol}"
            name, price, change_pct = fields[0], fields[1], fields[2]
            return f"{name} ({symbol.upper()})\n现价: ${price}  涨跌: {change_pct}%"
        else:
            # A股格式: 名称,今开,昨收,现价,最高,最低,...
            if len(fields) < 6:
                return f"数据不完整: {symbol}"
            name, today_open, prev_close, price, high, low = fields[:6]
            try:
                change = float(price) - float(prev_close)
                change_pct = change / float(prev_close) * 100
                sign = "+" if change >= 0 else ""
                return f"{name} ({symbol.upper()})\n现价: ¥{price}  涨跌: {sign}{change:.2f} ({sign}{change_pct:.2f}%)\n今开: {today_open}  最高: {high}  最低: {low}"
            except (ValueError, ZeroDivisionError):
                return f"{name} ({symbol.upper()})\n现价: ¥{price}"
    except Exception as e:
        return f"股价查询失败: {e}"


async def _tool_get_datetime(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """获取当前准确日期时间."""
    from datetime import datetime, timedelta, timezone as _tz

    tz_cn = _tz(timedelta(hours=8))
    now = datetime.now(tz_cn)
    weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()]
    return f"{now.strftime('%Y-%m-%d %H:%M')} {weekday}"


async def _tool_calculate(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """安全计算数学表达式."""
    import ast
    import math
    import operator

    expr = (args.get("expression") or "").strip()
    if not expr:
        return "需要一个数学表达式，如 1+2*3 或 (100*1.15**12)。"

    # 安全求值：只允许数学运算
    _OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    _FUNCS = {
        "abs": abs, "round": round, "int": int, "float": float,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e,
        "max": max, "min": min, "sum": sum,
        "pow": pow,
    }

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"不允许的常量: {node.value}")
        if isinstance(node, ast.BinOp):
            op = _OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的运算: {type(node.op).__name__}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的运算: {type(node.op).__name__}")
            return op(_eval(node.operand))
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
                fn = _FUNCS[node.func.id]
                args_vals = [_eval(a) for a in node.args]
                return fn(*args_vals)
            raise ValueError(f"不允许的函数: {ast.dump(node.func)}")
        if isinstance(node, ast.Name):
            if node.id in _FUNCS:
                return _FUNCS[node.id]
            raise ValueError(f"未知变量: {node.id}")
        if isinstance(node, ast.Tuple):
            return tuple(_eval(e) for e in node.elts)
        if isinstance(node, ast.List):
            return [_eval(e) for e in node.elts]
        raise ValueError(f"不支持的语法: {type(node).__name__}")

    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval(tree)
        # 格式化结果
        if isinstance(result, float):
            if result == int(result) and abs(result) < 1e15:
                return str(int(result))
            return f"{result:,.6g}"
        return str(result)
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError) as e:
        return f"计算错误: {e}"


# ── 飞书文档工具 ──


async def _tool_search_feishu_docs(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """搜索飞书云文档."""
    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置，无法搜索文档。"

    query = (args.get("query") or "").strip()
    count = min(args.get("count", 10), 20)
    if not query:
        return "搜索关键词不能为空。"

    token = await ctx.feishu_token_mgr.get_token()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://open.feishu.cn/open-apis/suite/docs-api/search/object",
            json={"query": query, "count": count, "offset": 0},
            headers={"Authorization": f"Bearer {token}"},
        )
    data = resp.json()
    if not data.get("data", {}).get("docs_entities"):
        return "没有找到匹配的文档。"

    lines = []
    for doc in data["data"]["docs_entities"][:count]:
        title = doc.get("title", "无标题")
        url = doc.get("url", "")
        doc_type = doc.get("docs_type", "")
        lines.append(f"[{doc_type}] {title}\n{url}")
    return "\n---\n".join(lines)


async def _tool_read_feishu_doc(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """读取飞书文档内容."""
    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置，无法读取文档。"

    doc_id = (args.get("document_id") or "").strip()
    if not doc_id:
        return "缺少 document_id。"

    token = await ctx.feishu_token_mgr.get_token()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_id}/raw_content",
            headers={"Authorization": f"Bearer {token}"},
        )
    data = resp.json()
    content = data.get("data", {}).get("content", "")
    if not content:
        return f"文档 {doc_id} 内容为空或无权限访问。"
    if len(content) > 9500:
        return content[:9500] + f"\n\n[内容已截断，共 {len(content)} 字符]"
    return content


async def _tool_create_feishu_doc(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """在飞书创建新文档."""
    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置，无法创建文档。"

    title = (args.get("title") or "").strip()
    content = args.get("content", "")
    folder_token = args.get("folder_token", "")
    if not title:
        return "缺少文档标题。"

    token = await ctx.feishu_token_mgr.get_token()
    create_body: dict = {"title": title}
    if folder_token:
        create_body["folder_token"] = folder_token

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://open.feishu.cn/open-apis/docx/v1/documents",
            json=create_body,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        data = resp.json()
        doc_id = data.get("data", {}).get("document", {}).get("document_id", "")
        if not doc_id:
            return f"创建文档失败: {data.get('msg', '未知错误')}"

        if content:
            await client.post(
                f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_id}/blocks/{doc_id}/children",
                json={
                    "children": [{
                        "block_type": 2,
                        "text": {"elements": [{"text_run": {"content": content}}]},
                    }],
                },
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            )

    url = f"https://feishu.cn/docx/{doc_id}"
    return f"文档已创建：{title}\n{url}"


async def _tool_send_feishu_group(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """发消息到飞书群."""
    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置，无法发群消息。"

    chat_id = (args.get("chat_id") or "").strip()
    text = (args.get("text") or "").strip()
    if not chat_id or not text:
        return "需要 chat_id 和 text。"

    from crew.feishu import send_feishu_text
    result = await send_feishu_text(ctx.feishu_token_mgr, chat_id, text)
    if result.get("code") == 0 or result.get("ok"):
        return f"消息已发送到群 {chat_id}。"
    return f"发送失败: {result.get('msg') or result.get('error', '未知错误')}"


async def _tool_list_feishu_groups(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """列出机器人加入的所有飞书群."""
    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    token = await ctx.feishu_token_mgr.get_token()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://open.feishu.cn/open-apis/im/v1/chats",
                headers={"Authorization": f"Bearer {token}"},
                params={"page_size": 50},
            )
            data = resp.json()
            if data.get("code") != 0:
                return f"查询失败: {data.get('msg', '未知错误')}"
            items = data.get("data", {}).get("items", [])
            if not items:
                return "机器人没有加入任何群。"
            lines = []
            for item in items:
                name = item.get("name", "未命名")
                chat_id = item.get("chat_id", "")
                lines.append(f"{name} — {chat_id}")
            return "\n".join(lines)
    except Exception as e:
        return f"查询失败: {e}"


async def _tool_send_feishu_dm(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """给飞书用户发私聊消息."""
    import json as _json

    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    open_id = (args.get("open_id") or "").strip()
    text = (args.get("text") or "").strip()
    if not open_id:
        return "需要 open_id。先用 feishu_group_members 查成员列表获取 open_id。"
    if not text:
        return "消息内容不能为空。"

    token = await ctx.feishu_token_mgr.get_token()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={
                    "receive_id": open_id,
                    "msg_type": "text",
                    "content": _json.dumps({"text": text}),
                },
            )
            data = resp.json()

        if data.get("code") == 0:
            return f"私聊消息已发送给 {open_id}。"
        return f"发送失败: {data.get('msg', '未知错误')}"
    except Exception as e:
        return f"发送失败: {e}"


async def _tool_feishu_group_members(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查看飞书群成员列表."""
    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    chat_id = (args.get("chat_id") or "").strip()
    if not chat_id:
        return "需要 chat_id。先用 list_feishu_groups 查群列表。"

    token = await ctx.feishu_token_mgr.get_token()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"https://open.feishu.cn/open-apis/im/v1/chats/{chat_id}/members",
                headers={"Authorization": f"Bearer {token}"},
                params={"page_size": 50},
            )
            data = resp.json()

        if data.get("code") != 0:
            return f"查询失败: {data.get('msg', '未知错误')}"

        items = data.get("data", {}).get("items", [])
        if not items:
            return "群里没有成员（或无权限查看）。"

        lines = []
        for m in items:
            name = m.get("name", "未知")
            mid = m.get("member_id", "")
            lines.append(f"{name} [open_id={mid}]")
        return "\n".join(lines)
    except Exception as e:
        return f"查询失败: {e}"


# ── GitHub 工具 ──


async def _tool_github_prs(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查看 GitHub 仓库 PR 列表."""
    import httpx

    if not _GITHUB_TOKEN:
        return "GitHub 未配置。请设置 GITHUB_TOKEN 环境变量。"

    repo = (args.get("repo") or "").strip()
    state = args.get("state", "open")
    limit = min(args.get("limit", 10), 30)
    if not repo or "/" not in repo:
        return "repo 格式错误，应为 owner/repo。"
    if not _is_valid_github_repo(repo):
        return "repo 格式错误，应为 owner/repo（仅允许字母数字、-、_、.）。"

    headers = {"Authorization": f"token {_GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_GITHUB_API_BASE}/repos/{repo}/pulls",
            params={"state": state, "per_page": limit},
            headers=headers,
        )
    if resp.status_code != 200:
        return f"GitHub API 错误 {resp.status_code}: {resp.text[:500]}"

    prs = resp.json()
    if not prs:
        return f"{repo} 没有 {state} 状态的 PR。"

    lines = []
    for pr in prs[:limit]:
        labels = ", ".join(l["name"] for l in pr.get("labels", []))
        label_str = f" [{labels}]" if labels else ""
        lines.append(f"#{pr['number']} {pr['title']}{label_str} — {pr['user']['login']} ({pr['state']})\n{pr['html_url']}")
    return "\n---\n".join(lines)


async def _tool_github_issues(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查看 GitHub 仓库 Issue 列表."""
    import httpx

    if not _GITHUB_TOKEN:
        return "GitHub 未配置。请设置 GITHUB_TOKEN 环境变量。"

    repo = (args.get("repo") or "").strip()
    state = args.get("state", "open")
    labels = args.get("labels", "")
    limit = min(args.get("limit", 10), 30)
    if not repo or "/" not in repo:
        return "repo 格式错误，应为 owner/repo。"
    if not _is_valid_github_repo(repo):
        return "repo 格式错误，应为 owner/repo（仅允许字母数字、-、_、.）。"

    params: dict = {"state": state, "per_page": limit}
    if labels:
        params["labels"] = labels

    headers = {"Authorization": f"token {_GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_GITHUB_API_BASE}/repos/{repo}/issues",
            params=params,
            headers=headers,
        )
    if resp.status_code != 200:
        return f"GitHub API 错误 {resp.status_code}: {resp.text[:500]}"

    # GitHub Issues API 也返回 PR，需要过滤
    issues = [i for i in resp.json() if "pull_request" not in i]
    if not issues:
        return f"{repo} 没有 {state} 状态的 Issue。"

    lines = []
    for issue in issues[:limit]:
        labels_str = ", ".join(l["name"] for l in issue.get("labels", []))
        label_part = f" [{labels_str}]" if labels_str else ""
        assignee = issue.get("assignee", {})
        assignee_str = f" → {assignee['login']}" if assignee else ""
        lines.append(f"#{issue['number']} {issue['title']}{label_part}{assignee_str}\n{issue['html_url']}")
    return "\n---\n".join(lines)


async def _tool_github_repo_activity(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查看 GitHub 仓库最近活动."""
    import httpx
    from datetime import datetime, timedelta, timezone

    if not _GITHUB_TOKEN:
        return "GitHub 未配置。请设置 GITHUB_TOKEN 环境变量。"

    repo = (args.get("repo") or "").strip()
    days = args.get("days", 7)
    if not repo or "/" not in repo:
        return "repo 格式错误，应为 owner/repo。"
    if not _is_valid_github_repo(repo):
        return "repo 格式错误，应为 owner/repo（仅允许字母数字、-、_、.）。"

    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    headers = {"Authorization": f"token {_GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_GITHUB_API_BASE}/repos/{repo}/commits",
            params={"per_page": 30, "since": since},
            headers=headers,
        )
    if resp.status_code != 200:
        return f"GitHub API 错误 {resp.status_code}: {resp.text[:500]}"

    commits = resp.json()
    if not commits:
        return f"{repo} 最近 {days} 天没有提交。"

    authors: dict[str, int] = {}
    lines = []
    for c in commits[:20]:
        sha = c["sha"][:7]
        msg = (c["commit"]["message"].split("\n")[0])[:80]
        author = c["commit"]["author"]["name"]
        date = c["commit"]["author"]["date"][:10]
        authors[author] = authors.get(author, 0) + 1
        lines.append(f"{sha} {msg} — {author} ({date})")

    summary = f"最近 {days} 天：{len(commits)} 次提交，{len(authors)} 位贡献者"
    top_authors = ", ".join(f"{k}({v})" for k, v in sorted(authors.items(), key=lambda x: -x[1])[:5])
    return f"{summary}\n贡献者: {top_authors}\n\n" + "\n".join(lines)


# ── Notion 工具 ──


def _notion_blocks_to_text(blocks: list[dict]) -> str:
    """将 Notion blocks 转为纯文本."""
    parts = []
    for block in blocks:
        btype = block.get("type", "")
        data = block.get(btype, {})
        rich_text = data.get("rich_text", [])
        text = "".join(rt.get("plain_text", "") for rt in rich_text)
        if btype.startswith("heading"):
            parts.append(f"\n{text}\n")
        elif btype == "to_do":
            checked = "x" if data.get("checked") else " "
            parts.append(f"[{checked}] {text}")
        elif btype == "code":
            parts.append(f"```\n{text}\n```")
        elif text:
            parts.append(text)
    return "\n".join(parts)


async def _tool_notion_search(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """搜索 Notion 页面."""
    import httpx

    if not _NOTION_API_KEY:
        return "Notion 未配置。请设置 NOTION_API_KEY 环境变量。"

    query = (args.get("query") or "").strip()
    limit = min(args.get("limit", 10), 20)
    if not query:
        return "搜索关键词不能为空。"

    headers = {
        "Authorization": f"Bearer {_NOTION_API_KEY}",
        "Notion-Version": _NOTION_VERSION,
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{_NOTION_API_BASE}/search",
            json={"query": query, "page_size": limit},
            headers=headers,
        )
    if resp.status_code != 200:
        return f"Notion API 错误 {resp.status_code}: {resp.text[:500]}"

    results = resp.json().get("results", [])
    if not results:
        return "没有找到匹配的页面。"

    lines = []
    for r in results[:limit]:
        obj_type = r.get("object", "page")
        url = r.get("url", "")
        edited = r.get("last_edited_time", "")[:10]
        # 提取标题
        props = r.get("properties", {})
        title_prop = props.get("title", props.get("Name", {}))
        title_arr = title_prop.get("title", []) if isinstance(title_prop, dict) else []
        title = "".join(t.get("plain_text", "") for t in title_arr) or "无标题"
        lines.append(f"[{obj_type}] {title} (edited: {edited})\n{url}")
    return "\n---\n".join(lines)


async def _tool_notion_read(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """读取 Notion 页面内容."""
    import httpx

    if not _NOTION_API_KEY:
        return "Notion 未配置。请设置 NOTION_API_KEY 环境变量。"

    page_id = (args.get("page_id") or "").strip().replace("-", "")
    if not page_id:
        return "缺少 page_id。"

    headers = {
        "Authorization": f"Bearer {_NOTION_API_KEY}",
        "Notion-Version": _NOTION_VERSION,
    }
    all_blocks: list[dict] = []
    next_cursor = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        for _ in range(5):  # 最多取 5 页
            params: dict = {"page_size": 100}
            if next_cursor:
                params["start_cursor"] = next_cursor
            resp = await client.get(
                f"{_NOTION_API_BASE}/blocks/{page_id}/children",
                params=params,
                headers=headers,
            )
            if resp.status_code != 200:
                return f"Notion API 错误 {resp.status_code}: {resp.text[:500]}"
            data = resp.json()
            all_blocks.extend(data.get("results", []))
            if not data.get("has_more"):
                break
            next_cursor = data.get("next_cursor")

    text = _notion_blocks_to_text(all_blocks)
    if not text.strip():
        return f"页面 {page_id} 内容为空。"
    if len(text) > 9500:
        return text[:9500] + f"\n\n[内容已截断，共 {len(text)} 字符]"
    return text


async def _tool_notion_create(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """在 Notion 创建新页面."""
    import httpx

    if not _NOTION_API_KEY:
        return "Notion 未配置。请设置 NOTION_API_KEY 环境变量。"

    parent_id = (args.get("parent_id") or "").strip().replace("-", "")
    title = (args.get("title") or "").strip()
    content = args.get("content", "")
    if not parent_id or not title:
        return "需要 parent_id 和 title。"

    children = []
    for para in content.split("\n\n")[:100]:
        if para.strip():
            children.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"type": "text", "text": {"content": para.strip()}}]},
            })

    headers = {
        "Authorization": f"Bearer {_NOTION_API_KEY}",
        "Notion-Version": _NOTION_VERSION,
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{_NOTION_API_BASE}/pages",
            json={
                "parent": {"page_id": parent_id},
                "properties": {"title": {"title": [{"text": {"content": title}}]}},
                "children": children,
            },
            headers=headers,
        )
    if resp.status_code not in (200, 201):
        return f"Notion 创建失败 {resp.status_code}: {resp.text[:500]}"

    url = resp.json().get("url", "")
    return f"页面已创建：{title}\n{url}"


# ── 信息采集工具 ──


async def _tool_read_url(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """读取网页正文."""
    import re

    import httpx

    import ipaddress
    from urllib.parse import urlparse

    url = (args.get("url") or "").strip()
    if not url:
        return "缺少 URL。"

    # SSRF 防护：仅允许 http/https，阻止私有/保留 IP
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "仅支持 http/https 协议。"
    hostname = parsed.hostname or ""
    if not hostname:
        return "无效 URL。"
    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            return "不允许访问内网地址。"
    except ValueError:
        # hostname 不是 IP（域名），检查常见内网域名
        if hostname in ("localhost", "metadata.google.internal"):
            return "不允许访问内网地址。"

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        html = resp.text
    except Exception as e:
        return f"请求失败: {e}"

    # 简单的 HTML → 纯文本提取
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<nav[^>]*>.*?</nav>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<footer[^>]*>.*?</footer>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<header[^>]*>.*?</header>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # 提取 <article> 或 <main>，否则取 <body>
    for tag in ("article", "main"):
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", html, re.DOTALL | re.IGNORECASE)
        if m:
            html = m.group(1)
            break

    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return "无法提取页面内容。"
    if len(text) > 9500:
        return text[:9500] + f"\n\n[内容已截断，共 {len(text)} 字符]"
    return text


async def _tool_rss_read(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """读取 RSS/Atom 订阅源."""
    import re
    import xml.etree.ElementTree as ET

    import httpx

    url = (args.get("url") or "").strip()
    limit = min(args.get("limit", 10), 30)
    if not url:
        return "缺少 RSS URL。"

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(resp.text)
    except Exception as e:
        return f"RSS 解析失败: {e}"

    entries = []
    # RSS 2.0: <channel><item>
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = (item.findtext("description") or "")[:200].strip()
        desc = re.sub(r"<[^>]+>", "", desc)  # strip HTML
        if title:
            entries.append(f"{title}\n{desc}\n{link}")

    # Atom: <entry>
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
        title = (entry.findtext("atom:title", "", ns) or entry.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
        link_el = entry.find("atom:link", ns) or entry.find("{http://www.w3.org/2005/Atom}link")
        link = link_el.get("href", "") if link_el is not None else ""
        summary = (entry.findtext("atom:summary", "", ns) or entry.findtext("{http://www.w3.org/2005/Atom}summary") or "")[:200].strip()
        summary = re.sub(r"<[^>]+>", "", summary)
        if title:
            entries.append(f"{title}\n{summary}\n{link}")

    if not entries:
        return "订阅源中没有找到条目。"

    return "\n---\n".join(entries[:limit])


# ── 生活助手工具 ──


async def _tool_translate(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """中英互译（MyMemory API）."""
    import httpx

    text = (args.get("text") or "").strip()
    if not text:
        return "需要翻译的文本。"
    if len(text) > 2000:
        return "文本过长，最多 2000 字符。"

    from_lang = (args.get("from_lang") or "auto").strip().lower()
    to_lang = (args.get("to_lang") or "").strip().lower()

    # 自动检测：CJK 占比 > 30% → 中→英，否则英→中
    if from_lang == "auto":
        cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        if cjk / max(len(text), 1) > 0.3:
            from_lang, to_lang = "zh-CN", to_lang or "en-GB"
        else:
            from_lang, to_lang = "en-GB", to_lang or "zh-CN"
    else:
        _lang_map = {
            "zh": "zh-CN", "en": "en-GB", "ja": "ja-JP",
            "ko": "ko-KR", "fr": "fr-FR", "de": "de-DE",
        }
        from_lang = _lang_map.get(from_lang, from_lang)
        to_lang = _lang_map.get(to_lang, to_lang) if to_lang else (
            "en-GB" if "zh" in from_lang else "zh-CN"
        )

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.mymemory.translated.net/get",
                params={"q": text, "langpair": f"{from_lang}|{to_lang}"},
            )
            data = resp.json()
        translated = data.get("responseData", {}).get("translatedText", "")
        if not translated:
            return "翻译失败，未获得结果。"
        return translated
    except Exception as e:
        return f"翻译失败: {e}"


async def _tool_countdown(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """计算距离目标日期的倒计时."""
    from datetime import datetime, timedelta, timezone as _tz

    tz_cn = _tz(timedelta(hours=8))
    date_str = (args.get("date") or "").strip()
    event = (args.get("event") or "").strip()

    if not date_str:
        return "需要目标日期，格式 YYYY-MM-DD。"

    try:
        target = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=tz_cn)
    except ValueError:
        return f"日期格式不对: {date_str}，需要 YYYY-MM-DD。"

    now = datetime.now(tz_cn)
    delta = target - now
    label = f"「{event}」" if event else date_str

    if delta.total_seconds() < 0:
        days = abs(delta.days)
        return f"{label} 已经过去了 {days} 天。"

    days = delta.days
    hours = delta.seconds // 3600
    if days == 0:
        return f"距离 {label} 还有 {hours} 小时。"
    return f"距离 {label} 还有 {days} 天 {hours} 小时。"


async def _tool_trending(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """热搜聚合（微博 / 知乎）."""
    import httpx

    platform = (args.get("platform") or "weibo").strip().lower()
    limit = min(args.get("limit", 15) or 15, 30)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            if platform == "zhihu":
                resp = await client.get(
                    "https://www.zhihu.com/api/v3/feed/topstory/hot-lists/total",
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                data = resp.json()
                items = data.get("data", [])[:limit]
                if not items:
                    return "知乎热榜暂无数据。"
                lines = []
                for i, item in enumerate(items, 1):
                    target = item.get("target", {})
                    title = target.get("title", "")
                    excerpt = target.get("excerpt", "")[:60]
                    lines.append(f"{i}. {title}\n   {excerpt}")
                return "📊 知乎热榜\n\n" + "\n".join(lines)
            else:
                # 微博热搜
                resp = await client.get(
                    "https://weibo.com/ajax/side/hotSearch",
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                data = resp.json()
                items = data.get("data", {}).get("realtime", [])[:limit]
                if not items:
                    return "微博热搜暂无数据。"
                lines = []
                for i, item in enumerate(items, 1):
                    word = item.get("word", "")
                    num = item.get("num", 0)
                    label_name = item.get("label_name", "")
                    tag = f" [{label_name}]" if label_name else ""
                    lines.append(f"{i}. {word}{tag}  ({num:,})")
                return "🔥 微博热搜\n\n" + "\n".join(lines)
    except Exception as e:
        return f"获取热搜失败: {e}"


# ── 飞书表格工具 ──


async def _tool_read_feishu_sheet(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """读取飞书表格数据."""
    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    ss_token = (args.get("spreadsheet_token") or "").strip()
    if not ss_token:
        return "缺少 spreadsheet_token。"

    sheet_id = (args.get("sheet_id") or "").strip()
    range_str = (args.get("range") or "A1:Z100").strip()

    token = await ctx.feishu_token_mgr.get_token()
    headers = {"Authorization": f"Bearer {token}"}
    base = "https://open.feishu.cn/open-apis"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if not sheet_id:
                meta_resp = await client.get(
                    f"{base}/sheets/v3/spreadsheets/{ss_token}/sheets/query",
                    headers=headers,
                )
                meta = meta_resp.json()
                sheets = meta.get("data", {}).get("sheets", [])
                if not sheets:
                    return "该表格没有工作表。"
                sheet_id = sheets[0].get("sheet_id", "")

            full_range = f"{sheet_id}!{range_str}"
            resp = await client.get(
                f"{base}/sheets/v2/spreadsheets/{ss_token}/values/{full_range}",
                headers=headers,
                params={"valueRenderOption": "ToString"},
            )
            data = resp.json()

        if data.get("code") != 0:
            return f"读取失败: {data.get('msg', '未知错误')}"

        values = data.get("data", {}).get("valueRange", {}).get("values", [])
        if not values:
            return "表格为空或指定范围无数据。"

        lines = []
        for i, row in enumerate(values[:100]):
            cells = [str(c) if c is not None else "" for c in row]
            lines.append(" | ".join(cells))
            if i == 0:
                lines.append("-" * min(len(lines[0]), 80))
        result = "\n".join(lines)
        if len(result) > 9500:
            result = result[:9500] + "\n\n[数据已截断]"
        return result
    except Exception as e:
        return f"读取表格失败: {e}"


async def _tool_update_feishu_sheet(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """写入飞书表格数据."""
    import httpx
    import json as _json

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    ss_token = (args.get("spreadsheet_token") or "").strip()
    range_str = (args.get("range") or "").strip()
    values_str = (args.get("values") or "").strip()
    sheet_id = (args.get("sheet_id") or "").strip()

    if not ss_token:
        return "缺少 spreadsheet_token。"
    if not range_str:
        return "缺少 range（如 A1:C3）。"
    if not values_str:
        return "缺少 values（JSON 二维数组）。"

    try:
        values = _json.loads(values_str)
        if not isinstance(values, list):
            return "values 必须是二维数组。"
    except _json.JSONDecodeError as e:
        return f"values JSON 解析失败: {e}"

    token = await ctx.feishu_token_mgr.get_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    base = "https://open.feishu.cn/open-apis"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if not sheet_id:
                meta_resp = await client.get(
                    f"{base}/sheets/v3/spreadsheets/{ss_token}/sheets/query",
                    headers={"Authorization": f"Bearer {token}"},
                )
                meta = meta_resp.json()
                sheets = meta.get("data", {}).get("sheets", [])
                if not sheets:
                    return "该表格没有工作表。"
                sheet_id = sheets[0].get("sheet_id", "")

            full_range = f"{sheet_id}!{range_str}"
            resp = await client.put(
                f"{base}/sheets/v2/spreadsheets/{ss_token}/values",
                headers=headers,
                json={
                    "valueRange": {
                        "range": full_range,
                        "values": values,
                    },
                },
            )
            data = resp.json()

        if data.get("code") != 0:
            return f"写入失败: {data.get('msg', '未知错误')}"

        updated = data.get("data", {}).get("updatedCells", 0)
        return f"写入成功，更新了 {updated} 个单元格。"
    except Exception as e:
        return f"写入表格失败: {e}"


# ── 飞书审批工具 ──


async def _tool_list_feishu_approvals(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查看飞书审批列表."""
    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    status = (args.get("status") or "PENDING").strip().upper()
    limit = min(args.get("limit", 10) or 10, 20)

    token = await ctx.feishu_token_mgr.get_token()
    headers = {"Authorization": f"Bearer {token}"}
    base = "https://open.feishu.cn/open-apis"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 先获取审批定义列表
            resp = await client.get(
                f"{base}/approval/v4/approvals",
                headers=headers,
                params={"page_size": 20},
            )
            data = resp.json()

        if data.get("code") != 0:
            return f"获取审批失败: {data.get('msg', '未知错误')}"

        approvals = data.get("data", {}).get("approval_list", [])
        if not approvals:
            return "没有找到审批流程。"

        # 遍历审批定义，查实例
        all_instances: list[str] = []
        async with httpx.AsyncClient(timeout=30.0) as client:
            for appr in approvals[:5]:  # 只查前 5 个审批定义
                code = appr.get("approval_code", "")
                name = appr.get("approval_name", "未命名")
                if not code:
                    continue
                params: dict[str, Any] = {
                    "approval_code": code,
                    "page_size": limit,
                }
                if status != "ALL":
                    params["status"] = status
                inst_resp = await client.get(
                    f"{base}/approval/v4/instances",
                    headers=headers,
                    params=params,
                )
                inst_data = inst_resp.json()
                instances = inst_data.get("data", {}).get("instance_list", [])
                for inst in instances:
                    inst_code = inst.get("instance_code", "")
                    inst_status = inst.get("status", "")
                    start_time = inst.get("start_time", "")
                    # 转换时间戳
                    ts_str = ""
                    if start_time:
                        try:
                            from datetime import datetime, timedelta, timezone as _tz
                            ts = int(start_time) // 1000 if len(start_time) > 10 else int(start_time)
                            dt = datetime.fromtimestamp(ts, _tz(timedelta(hours=8)))
                            ts_str = dt.strftime("%m-%d %H:%M")
                        except (ValueError, OSError):
                            ts_str = start_time
                    status_icon = {"PENDING": "⏳", "APPROVED": "✅", "REJECTED": "❌"}.get(
                        inst_status, "📋"
                    )
                    all_instances.append(
                        f"{status_icon} [{name}] {ts_str} (instance={inst_code})"
                    )
                    if len(all_instances) >= limit:
                        break
                if len(all_instances) >= limit:
                    break

        if not all_instances:
            label = {"PENDING": "待审批", "APPROVED": "已通过", "REJECTED": "已拒绝"}.get(
                status, ""
            )
            return f"没有{label}的审批。"

        return "\n".join(all_instances)
    except Exception as e:
        return f"获取审批失败: {e}"


# ── 实用工具 ──

# 单位换算表：(from, to) → multiplier  或  callable
_UNIT_CONVERSIONS: dict[tuple[str, str], float | Any] = {
    # 长度
    ("km", "mi"): 0.621371, ("mi", "km"): 1.60934,
    ("m", "ft"): 3.28084, ("ft", "m"): 0.3048,
    ("cm", "in"): 0.393701, ("in", "cm"): 2.54,
    ("km", "m"): 1000, ("m", "km"): 0.001,
    ("m", "cm"): 100, ("cm", "m"): 0.01,
    ("mi", "ft"): 5280, ("ft", "mi"): 1 / 5280,
    # 重量
    ("kg", "lb"): 2.20462, ("lb", "kg"): 0.453592,
    ("kg", "g"): 1000, ("g", "kg"): 0.001,
    ("kg", "oz"): 35.274, ("oz", "kg"): 0.0283495,
    ("lb", "oz"): 16, ("oz", "lb"): 0.0625,
    ("g", "mg"): 1000, ("mg", "g"): 0.001,
    # 面积
    ("sqm", "sqft"): 10.7639, ("sqft", "sqm"): 0.092903,
    ("mu", "sqm"): 666.667, ("sqm", "mu"): 0.0015,
    ("ha", "mu"): 15, ("mu", "ha"): 1 / 15,
    ("ha", "sqm"): 10000, ("sqm", "ha"): 0.0001,
    # 体积
    ("l", "gal"): 0.264172, ("gal", "l"): 3.78541,
    ("l", "ml"): 1000, ("ml", "l"): 0.001,
    # 数据
    ("gb", "mb"): 1024, ("mb", "gb"): 1 / 1024,
    ("tb", "gb"): 1024, ("gb", "tb"): 1 / 1024,
    ("mb", "kb"): 1024, ("kb", "mb"): 1 / 1024,
    # 速度
    ("kmh", "mph"): 0.621371, ("mph", "kmh"): 1.60934,
    ("ms", "kmh"): 3.6, ("kmh", "ms"): 1 / 3.6,
}


async def _tool_unit_convert(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """单位换算."""
    value = args.get("value")
    if value is None:
        return "缺少数值。"
    value = float(value)

    from_u = (args.get("from_unit") or "").strip().lower().replace("°", "").replace(" ", "")
    to_u = (args.get("to_unit") or "").strip().lower().replace("°", "").replace(" ", "")

    if not from_u or not to_u:
        return "需要原单位和目标单位。"

    # 温度特殊处理
    if from_u in ("c", "celsius") and to_u in ("f", "fahrenheit"):
        result = value * 9 / 5 + 32
        return f"{value}°C = {result:.2f}°F"
    if from_u in ("f", "fahrenheit") and to_u in ("c", "celsius"):
        result = (value - 32) * 5 / 9
        return f"{value}°F = {result:.2f}°C"

    key = (from_u, to_u)
    factor = _UNIT_CONVERSIONS.get(key)
    if factor is None:
        return f"不支持 {from_u} → {to_u} 的换算。支持：km/mi, m/ft, kg/lb, l/gal, gb/mb, c/f 等。"

    result = value * factor
    if result == int(result) and abs(result) < 1e15:
        return f"{value} {from_u} = {int(result)} {to_u}"
    return f"{value} {from_u} = {result:,.4g} {to_u}"


async def _tool_random_pick(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """随机选择 / 掷骰子."""
    import random

    options_str = (args.get("options") or "").strip()
    count = max(args.get("count", 1) or 1, 1)

    if not options_str:
        # 掷骰子
        result = random.randint(1, 6)
        return f"🎲 掷出了 {result} 点！"

    options = [o.strip() for o in options_str.replace("，", ",").split(",") if o.strip()]
    if len(options) < 2:
        return "至少需要两个选项。"

    count = min(count, len(options))
    picks = random.sample(options, count)
    if count == 1:
        return f"🎯 选中了：{picks[0]}"
    return f"🎯 选中了：{'、'.join(picks)}"


async def _tool_holidays(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查询中国法定节假日."""
    import httpx
    from datetime import datetime, timedelta, timezone as _tz

    tz_cn = _tz(timedelta(hours=8))
    now = datetime.now(tz_cn)
    year = args.get("year") or now.year
    month = args.get("month") or 0

    # 使用 timor.tech 免费节假日 API（国内可用）
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if month:
                resp = await client.get(
                    f"https://timor.tech/api/holiday/year/{year}/{month:02d}",
                )
            else:
                resp = await client.get(
                    f"https://timor.tech/api/holiday/year/{year}",
                )
            data = resp.json()

        if data.get("code") != 0:
            return f"查询失败: {data.get('msg', '未知错误')}"

        holidays_data = data.get("holiday", {})
        if not holidays_data:
            return f"{year}年{'%d月' % month if month else ''}没有节假日数据。"

        lines = []
        for date_str, info in sorted(holidays_data.items()):
            name = info.get("name", "")
            is_holiday = info.get("holiday", False)
            tag = "🟢 放假" if is_holiday else "🔴 补班"
            lines.append(f"{date_str} {tag} {name}")

        if not lines:
            return "没有找到节假日信息。"

        header = f"📅 {year}年{'%d月' % month if month else ''}节假日安排"
        return f"{header}\n\n" + "\n".join(lines)
    except Exception as e:
        return f"查询节假日失败: {e}"


async def _tool_timestamp_convert(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """Unix 时间戳 ↔ 可读时间互转."""
    from datetime import datetime, timedelta, timezone as _tz

    tz_cn = _tz(timedelta(hours=8))
    input_str = (args.get("input") or "").strip()
    if not input_str:
        return "需要时间戳或日期时间。"

    # 尝试解析为数字（时间戳）
    try:
        ts = int(input_str)
        # 毫秒级 → 秒级
        if ts > 1e12:
            ts = ts // 1000
        dt = datetime.fromtimestamp(ts, tz_cn)
        weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][dt.weekday()]
        return f"时间戳 {input_str} = {dt.strftime('%Y-%m-%d %H:%M:%S')} {weekday}（北京时间）"
    except (ValueError, OSError):
        pass

    # 尝试解析为日期时间
    fmts = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"]
    for fmt in fmts:
        try:
            dt = datetime.strptime(input_str, fmt).replace(tzinfo=tz_cn)
            ts = int(dt.timestamp())
            return f"{input_str} = 时间戳 {ts}（秒）/ {ts * 1000}（毫秒）"
        except ValueError:
            continue

    return f"无法解析: {input_str}。支持格式：Unix 时间戳 或 YYYY-MM-DD HH:MM:SS"


# ── 飞书表格创建 ──


async def _tool_create_feishu_spreadsheet(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """在飞书创建新表格."""
    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    title = (args.get("title") or "").strip()
    folder_token = (args.get("folder_token") or "").strip()
    if not title:
        return "需要表格标题。"

    token = await ctx.feishu_token_mgr.get_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        body: dict[str, Any] = {"title": title}
        if folder_token:
            body["folder_token"] = folder_token

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://open.feishu.cn/open-apis/sheets/v3/spreadsheets",
                headers=headers,
                json=body,
            )
            data = resp.json()

        if data.get("code") != 0:
            return f"创建失败: {data.get('msg', '未知错误')}"

        ss = data.get("data", {}).get("spreadsheet", {})
        ss_token = ss.get("spreadsheet_token", "")
        url = ss.get("url", "")
        return f"表格已创建: {title}\ntoken: {ss_token}\n{url}"
    except Exception as e:
        return f"创建表格失败: {e}"


# ── 飞书通讯录搜索 ──


async def _tool_feishu_contacts(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """飞书通讯录搜索."""
    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    query = (args.get("query") or "").strip()
    limit = min(args.get("limit", 5) or 5, 20)
    if not query:
        return "需要搜索关键词。"

    token = await ctx.feishu_token_mgr.get_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://open.feishu.cn/open-apis/search/v1/user",
                headers=headers,
                json={"query": query, "page_size": limit},
            )
            data = resp.json()

        if data.get("code") != 0:
            # 如果没有搜索权限，回退到群成员查找
            return f"通讯录搜索暂不可用({data.get('code')}): {data.get('msg', '')}。可以用 feishu_group_members 从群里查人。"

        users = data.get("data", {}).get("users", [])
        if not users:
            return f"未找到「{query}」。"

        lines = []
        for u in users:
            name = u.get("name", "未知")
            dept = u.get("department", {}).get("name", "")
            open_id = u.get("open_id", "")
            line = f"{name}"
            if dept:
                line += f" ({dept})"
            if open_id:
                line += f" [open_id={open_id}]"
            lines.append(line)
        return "\n".join(lines)
    except Exception as e:
        return f"搜索通讯录失败: {e}"


# ── 文本 & 开发工具 ──


async def _tool_text_extract(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """从文本中提取邮箱、手机号、URL、金额等."""
    import re

    text = args.get("text") or ""
    if not text:
        return "需要文本。"

    extract_type = (args.get("extract_type") or "all").strip().lower()

    results: dict[str, list[str]] = {}

    if extract_type in ("email", "all"):
        emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        if emails:
            results["邮箱"] = list(dict.fromkeys(emails))

    if extract_type in ("phone", "all"):
        phones = re.findall(r"1[3-9]\d{9}", text)
        # 也匹配带分隔的号码
        phones += re.findall(r"\+?\d{1,4}[-\s]?\d{3,4}[-\s]?\d{4}", text)
        if phones:
            results["手机号"] = list(dict.fromkeys(phones))

    if extract_type in ("url", "all"):
        urls = re.findall(r"https?://[^\s<>\"']+", text)
        if urls:
            results["URL"] = list(dict.fromkeys(urls))

    if extract_type in ("money", "all"):
        money = re.findall(r"[¥$￥]\s?[\d,]+\.?\d*|[\d,]+\.?\d*\s?(?:元|万|亿|美元|万元|亿元|USD|CNY|RMB)", text)
        if money:
            results["金额"] = list(dict.fromkeys(money))

    if not results:
        return "未提取到信息。"

    lines = []
    for category, items in results.items():
        lines.append(f"【{category}】")
        for item in items[:20]:
            lines.append(f"  {item}")
    return "\n".join(lines)


async def _tool_json_format(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """格式化 JSON."""
    import json as _json
    import re

    text = args.get("text") or ""
    if not text:
        return "需要 JSON 文本。"

    compact = args.get("compact", False)

    # 尝试直接解析
    try:
        obj = _json.loads(text)
    except _json.JSONDecodeError:
        # 尝试从文本中提取 JSON
        match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
        if not match:
            return "未找到有效的 JSON。"
        try:
            obj = _json.loads(match.group())
        except _json.JSONDecodeError as e:
            return f"JSON 解析失败: {e}"

    if compact:
        result = _json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    else:
        result = _json.dumps(obj, ensure_ascii=False, indent=2)

    if len(result) > 9500:
        result = result[:9500] + "\n\n[已截断]"
    return result


async def _tool_password_gen(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """生成安全随机密码."""
    import secrets
    import string

    length = max(min(args.get("length", 16) or 16, 128), 8)
    count = max(min(args.get("count", 3) or 3, 10), 1)
    no_symbols = args.get("no_symbols", False)

    chars = string.ascii_letters + string.digits
    if not no_symbols:
        chars += "!@#$%&*-_=+"

    passwords = []
    for _ in range(count):
        pw = "".join(secrets.choice(chars) for _ in range(length))
        passwords.append(pw)

    lines = [f"🔐 随机密码（{length}位）：", ""]
    for i, pw in enumerate(passwords, 1):
        lines.append(f"{i}. {pw}")
    return "\n".join(lines)


async def _tool_ip_lookup(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查询 IP 地址归属地."""
    import httpx

    ip = (args.get("ip") or "").strip()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if ip:
                resp = await client.get(f"http://ip-api.com/json/{ip}?lang=zh-CN")
            else:
                resp = await client.get("http://ip-api.com/json/?lang=zh-CN")
            data = resp.json()

        if data.get("status") != "success":
            return f"查询失败: {data.get('message', '未知错误')}"

        query_ip = data.get("query", ip or "本机")
        country = data.get("country", "")
        region = data.get("regionName", "")
        city = data.get("city", "")
        isp = data.get("isp", "")
        org = data.get("org", "")

        location = " ".join(filter(None, [country, region, city]))
        lines = [f"IP: {query_ip}", f"位置: {location}"]
        if isp:
            lines.append(f"运营商: {isp}")
        if org and org != isp:
            lines.append(f"组织: {org}")
        return "\n".join(lines)
    except Exception as e:
        return f"查询 IP 失败: {e}"


async def _tool_short_url(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """生成短链接（cleanuri.com 免费 API）."""
    import httpx

    url = (args.get("url") or "").strip()
    if not url:
        return "需要 URL。"
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://cleanuri.com/api/v1/shorten",
                data={"url": url},
            )
            data = resp.json()

        short = data.get("result_url", "")
        if short:
            return f"短链接: {short}\n原链接: {url}"
        return f"生成失败: {data.get('error', '未知错误')}"
    except Exception as e:
        return f"生成短链接失败: {e}"


async def _tool_word_count(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """统计文本字数."""
    text = args.get("text") or ""
    if not text:
        return "需要文本。"

    # 总字符数（含空格）
    total_chars = len(text)
    # 不含空格
    no_space = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
    # 中文字数
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    # 英文单词数
    import re
    words = len(re.findall(r"[a-zA-Z]+", text))
    # 数字个数
    numbers = len(re.findall(r"\d+", text))
    # 行数
    lines = text.count("\n") + 1
    # 段落数
    paragraphs = len([p for p in text.split("\n\n") if p.strip()])

    parts = [
        f"字符: {total_chars}（不含空格 {no_space}）",
        f"中文: {cjk} 字",
        f"英文: {words} 词",
    ]
    if numbers:
        parts.append(f"数字: {numbers} 个")
    parts.append(f"行: {lines}")
    parts.append(f"段落: {paragraphs}")

    return " | ".join(parts)


# ── 编码 & 开发辅助工具 ──


async def _tool_base64_codec(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """Base64 编解码."""
    import base64

    text = args.get("text") or ""
    if not text:
        return "需要文本。"

    decode = args.get("decode", False)
    try:
        if decode:
            result = base64.b64decode(text).decode("utf-8", errors="replace")
            return f"解码结果:\n{result}"
        else:
            result = base64.b64encode(text.encode("utf-8")).decode()
            return f"编码结果:\n{result}"
    except Exception as e:
        return f"Base64 {'解码' if decode else '编码'}失败: {e}"


async def _tool_color_convert(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """颜色格式转换."""
    import re

    color = (args.get("color") or "").strip()
    if not color:
        return "需要颜色值。"

    r = g = b = 0

    # HEX
    hex_match = re.match(r"^#?([0-9a-fA-F]{6})$", color)
    if hex_match:
        h = hex_match.group(1)
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    else:
        # RGB
        rgb_match = re.match(r"rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", color, re.I)
        if rgb_match:
            r, g, b = int(rgb_match.group(1)), int(rgb_match.group(2)), int(rgb_match.group(3))
        else:
            # 3位 HEX
            short_match = re.match(r"^#?([0-9a-fA-F]{3})$", color)
            if short_match:
                h = short_match.group(1)
                r, g, b = int(h[0]*2, 16), int(h[1]*2, 16), int(h[2]*2, 16)
            else:
                return f"无法解析颜色: {color}。支持 #FF5733、rgb(255,87,51) 格式。"

    # RGB → HSL
    r1, g1, b1 = r / 255, g / 255, b / 255
    mx, mn = max(r1, g1, b1), min(r1, g1, b1)
    l = (mx + mn) / 2
    if mx == mn:
        h_val = s = 0.0
    else:
        d = mx - mn
        s = d / (2 - mx - mn) if l > 0.5 else d / (mx + mn)
        if mx == r1:
            h_val = (g1 - b1) / d + (6 if g1 < b1 else 0)
        elif mx == g1:
            h_val = (b1 - r1) / d + 2
        else:
            h_val = (r1 - g1) / d + 4
        h_val /= 6

    hex_str = f"#{r:02X}{g:02X}{b:02X}"
    rgb_str = f"rgb({r}, {g}, {b})"
    hsl_str = f"hsl({int(h_val * 360)}, {int(s * 100)}%, {int(l * 100)}%)"

    return f"HEX: {hex_str}\nRGB: {rgb_str}\nHSL: {hsl_str}"


async def _tool_cron_explain(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """解释 cron 表达式."""
    expr = (args.get("expression") or "").strip()
    if not expr:
        return "需要 cron 表达式或自然语言描述。"

    # 简单自然语言 → cron 映射
    _NL_MAP = {
        "每分钟": "* * * * *",
        "每小时": "0 * * * *",
        "每天": "0 0 * * *",
        "每天早上9点": "0 9 * * *",
        "每天晚上10点": "0 22 * * *",
        "每周一": "0 0 * * 1",
        "工作日": "0 9 * * 1-5",
        "工作日早上9点": "0 9 * * 1-5",
        "每月1号": "0 0 1 * *",
        "每月15号": "0 0 15 * *",
    }

    for key, cron in _NL_MAP.items():
        if key in expr:
            return f"「{expr}」对应的 cron:\n{cron}"

    # 解析 cron 表达式
    parts = expr.split()
    if len(parts) not in (5, 6):
        return f"无法解析: {expr}。标准 cron 是 5 段（分 时 日 月 周），如 0 9 * * 1-5"

    fields = ["分钟", "小时", "日", "月", "星期"]
    if len(parts) == 6:
        fields = ["秒"] + fields

    _WEEKDAYS = {"0": "日", "1": "一", "2": "二", "3": "三", "4": "四", "5": "五", "6": "六", "7": "日"}

    lines = []
    for i, (p, name) in enumerate(zip(parts, fields)):
        if p == "*":
            lines.append(f"  {name}: 每{name}")
        elif p.startswith("*/"):
            lines.append(f"  {name}: 每 {p[2:]} {name}")
        elif name == "星期" and "-" in p:
            start, end = p.split("-", 1)
            lines.append(f"  {name}: 周{_WEEKDAYS.get(start, start)} 到 周{_WEEKDAYS.get(end, end)}")
        elif name == "星期":
            days = [f"周{_WEEKDAYS.get(d.strip(), d.strip())}" for d in p.split(",")]
            lines.append(f"  {name}: {','.join(days)}")
        else:
            lines.append(f"  {name}: {p}")

    return f"cron: {expr}\n\n" + "\n".join(lines)


async def _tool_regex_test(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """测试正则表达式."""
    import re

    pattern = args.get("pattern") or ""
    text = args.get("text") or ""
    replace = args.get("replace") or ""

    if not pattern:
        return "需要正则表达式。"
    if not text:
        return "需要测试文本。"

    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return f"正则语法错误: {e}"

    if replace:
        result = compiled.sub(replace, text)
        return f"替换结果:\n{result}"

    matches = list(compiled.finditer(text))
    if not matches:
        return "没有匹配。"

    lines = [f"找到 {len(matches)} 个匹配：", ""]
    for i, m in enumerate(matches[:20], 1):
        groups = m.groups()
        if groups:
            lines.append(f"{i}. 「{m.group()}」 groups={groups}")
        else:
            lines.append(f"{i}. 「{m.group()}」 位置 {m.start()}-{m.end()}")
    return "\n".join(lines)


async def _tool_hash_gen(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """计算文本哈希值."""
    import hashlib

    text = args.get("text") or ""
    if not text:
        return "需要文本。"

    algo = (args.get("algorithm") or "sha256").strip().lower()
    data = text.encode("utf-8")

    results = []
    if algo == "all" or algo == "md5":
        results.append(f"MD5:    {hashlib.md5(data).hexdigest()}")
    if algo == "all" or algo == "sha1":
        results.append(f"SHA1:   {hashlib.sha1(data).hexdigest()}")
    if algo == "all" or algo == "sha256":
        results.append(f"SHA256: {hashlib.sha256(data).hexdigest()}")

    if not results:
        # 默认 sha256
        results.append(f"SHA256: {hashlib.sha256(data).hexdigest()}")

    return "\n".join(results)


async def _tool_url_codec(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """URL 编解码."""
    from urllib.parse import quote, unquote

    text = args.get("text") or ""
    if not text:
        return "需要文本。"

    decode = args.get("decode", False)
    if decode:
        result = unquote(text)
        return f"解码结果:\n{result}"
    else:
        result = quote(text, safe="")
        return f"编码结果:\n{result}"


# ── 第 5 批工具 handler ──


async def _tool_feishu_bitable(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """读取飞书多维表格."""
    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    app_token = (args.get("app_token") or "").strip()
    table_id = (args.get("table_id") or "").strip()
    if not app_token or not table_id:
        return "需要 app_token 和 table_id。"

    limit = min(args.get("limit", 20) or 20, 100)
    filter_str = (args.get("filter") or "").strip()

    token = await ctx.feishu_token_mgr.get_token()
    headers = {"Authorization": f"Bearer {token}"}
    base = "https://open.feishu.cn/open-apis"

    try:
        params: dict[str, Any] = {"page_size": limit}
        if filter_str:
            params["filter"] = filter_str

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{base}/bitable/v1/apps/{app_token}/tables/{table_id}/records",
                headers=headers,
                params=params,
            )
            data = resp.json()

        if data.get("code") != 0:
            return f"读取多维表格失败: {data.get('msg', '未知错误')}"

        records = data.get("data", {}).get("items", [])
        if not records:
            return "表格中没有数据。"

        lines: list[str] = []
        for i, rec in enumerate(records, 1):
            fields = rec.get("fields", {})
            parts = [f"{k}: {v}" for k, v in fields.items()]
            lines.append(f"{i}. {' | '.join(parts)}")

        total = data.get("data", {}).get("total", len(records))
        lines.insert(0, f"共 {total} 条记录（显示前 {len(records)} 条）：\n")
        return "\n".join(lines)
    except Exception as e:
        return f"读取多维表格失败: {e}"


async def _tool_feishu_wiki(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """搜索飞书知识库."""
    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    query = (args.get("query") or "").strip()
    if not query:
        return "需要搜索关键词。"

    limit = min(args.get("limit", 10) or 10, 20)

    token = await ctx.feishu_token_mgr.get_token()
    headers = {"Authorization": f"Bearer {token}"}
    base = "https://open.feishu.cn/open-apis"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{base}/wiki/v2/spaces/search",
                headers=headers,
                json={"query": query, "page_size": limit},
            )
            data = resp.json()

        if data.get("code") != 0:
            return f"搜索知识库失败: {data.get('msg', '未知错误')}"

        items = data.get("data", {}).get("items", [])
        if not items:
            return f"知识库中没有找到「{query}」相关内容。"

        lines: list[str] = []
        for item in items:
            title = item.get("title", "无标题")
            url = item.get("url", "")
            space = item.get("space_name", "")
            line = f"- {title}"
            if space:
                line += f" [{space}]"
            if url:
                line += f"\n  {url}"
            lines.append(line)
        return "\n".join(lines)
    except Exception as e:
        return f"搜索知识库失败: {e}"


async def _tool_approve_feishu(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """操作飞书审批."""
    import httpx

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    instance_id = (args.get("instance_id") or "").strip()
    action = (args.get("action") or "").strip().lower()
    comment = (args.get("comment") or "").strip()

    if not instance_id:
        return "需要审批实例 ID。"
    if action not in ("approve", "reject"):
        return "action 必须是 approve 或 reject。"

    token = await ctx.feishu_token_mgr.get_token()
    headers = {"Authorization": f"Bearer {token}"}
    base = "https://open.feishu.cn/open-apis"

    try:
        body: dict[str, Any] = {"approval_code": "", "instance_code": instance_id}
        if action == "approve":
            body["status"] = "APPROVED"
        else:
            body["status"] = "REJECTED"
        if comment:
            body["comment"] = comment

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{base}/approval/v4/instances/{instance_id}/comments",
                headers=headers,
                json={"content": comment or ("同意" if action == "approve" else "拒绝")},
            )
            data = resp.json()

        action_cn = "通过" if action == "approve" else "拒绝"
        if data.get("code") == 0:
            return f"审批已{action_cn}。"
        return f"审批操作失败: {data.get('msg', '未知错误')}"
    except Exception as e:
        return f"审批操作失败: {e}"


async def _tool_summarize(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """长文摘要（由模型自身完成）."""
    text = (args.get("text") or "").strip()
    if not text:
        return "需要文本。"
    if len(text) > 50000:
        text = text[:50000] + "...(已截断)"

    style = (args.get("style") or "bullet").strip().lower()
    style_map = {
        "bullet": "用要点列表总结",
        "paragraph": "用一段话总结",
        "oneline": "用一句话总结",
    }
    instruction = style_map.get(style, style_map["bullet"])
    return f"[摘要任务] 请{instruction}以下内容：\n\n{text}"


async def _tool_sentiment(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """情感分析（由模型自身完成）."""
    text = (args.get("text") or "").strip()
    if not text:
        return "需要文本。"
    if len(text) > 10000:
        text = text[:10000] + "...(已截断)"

    return f"[情感分析任务] 请分析以下文本的情感倾向（正面/负面/中性）、语气和关键情绪词：\n\n{text}"


async def _tool_email_send(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """发送邮件（暂未对接 SMTP）."""
    to = (args.get("to") or "").strip()
    subject = (args.get("subject") or "").strip()
    if not to or not subject:
        return "需要收件人和主题。"
    return "邮件功能尚未配置 SMTP，暂时无法发送。请直接通过飞书或其他方式联系。"


async def _tool_qrcode(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """生成二维码."""
    from urllib.parse import quote

    data = (args.get("data") or "").strip()
    if not data:
        return "需要编码内容。"

    size = args.get("size", 300) or 300
    encoded = quote(data, safe="")
    url = f"https://api.qrserver.com/v1/create-qr-code/?size={size}x{size}&data={encoded}"
    return f"二维码已生成：\n{url}\n\n内容: {data}"


async def _tool_diff_text(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """文本对比."""
    import difflib

    text1 = args.get("text1") or ""
    text2 = args.get("text2") or ""
    if not text1 and not text2:
        return "需要两段文本。"

    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)
    diff = list(difflib.unified_diff(lines1, lines2, fromfile="原文", tofile="修改后", lineterm=""))

    if not diff:
        return "两段文本完全相同。"
    return "\n".join(diff[:200])


async def _tool_whois(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """WHOIS 域名查询."""
    import httpx

    domain = (args.get("domain") or "").strip().lower()
    if not domain:
        return "需要域名。"
    # 去掉 http:// 等前缀
    domain = domain.replace("https://", "").replace("http://", "").split("/")[0]

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"https://whois.freeaitools.xyz/api/{domain}")
            if resp.status_code != 200:
                return f"WHOIS 查询失败 (HTTP {resp.status_code})"
            data = resp.json()

        lines = [f"域名: {domain}"]
        for key in ("registrar", "creation_date", "expiration_date", "name_servers", "status"):
            val = data.get(key)
            if val:
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val)
                label = {
                    "registrar": "注册商",
                    "creation_date": "注册日期",
                    "expiration_date": "到期日期",
                    "name_servers": "DNS",
                    "status": "状态",
                }.get(key, key)
                lines.append(f"{label}: {val}")
        return "\n".join(lines) if len(lines) > 1 else f"未找到 {domain} 的 WHOIS 信息。"
    except Exception as e:
        return f"WHOIS 查询失败: {e}"


async def _tool_dns_lookup(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """DNS 解析."""
    import asyncio
    import socket

    domain = (args.get("domain") or "").strip().lower()
    if not domain:
        return "需要域名。"
    domain = domain.replace("https://", "").replace("http://", "").split("/")[0]

    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, lambda: socket.getaddrinfo(domain, None, socket.AF_UNSPEC, socket.SOCK_STREAM),
        )
        seen: set[str] = set()
        lines = [f"DNS 解析 {domain}："]
        for family, _type, _proto, _canon, addr in results:
            ip = addr[0]
            if ip in seen:
                continue
            seen.add(ip)
            record_type = "A" if family == socket.AF_INET else "AAAA"
            lines.append(f"  {record_type}: {ip}")
        return "\n".join(lines) if len(lines) > 1 else f"未找到 {domain} 的 DNS 记录。"
    except socket.gaierror:
        return f"无法解析域名: {domain}"
    except Exception as e:
        return f"DNS 查询失败: {e}"


async def _tool_http_check(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """网站可用性检查."""
    import httpx
    import time

    url = (args.get("url") or "").strip()
    if not url:
        return "需要 URL。"
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    try:
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.head(url)
        elapsed = (time.monotonic() - start) * 1000

        status = resp.status_code
        ok = "✅ 可用" if 200 <= status < 400 else "❌ 异常"
        lines = [
            f"{ok}",
            f"URL: {url}",
            f"状态码: {status}",
            f"响应时间: {elapsed:.0f}ms",
        ]
        if resp.headers.get("server"):
            lines.append(f"服务器: {resp.headers['server']}")
        return "\n".join(lines)
    except httpx.ConnectTimeout:
        return f"❌ 连接超时: {url}"
    except httpx.ConnectError:
        return f"❌ 无法连接: {url}"
    except Exception as e:
        return f"❌ 检查失败: {e}"


async def _tool_express_track(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """快递物流查询."""
    import httpx

    number = (args.get("number") or "").strip()
    if not number:
        return "需要快递单号。"

    company = (args.get("company") or "").strip()

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # 快递100 auto API
            url = f"https://www.kuaidi100.com/query"
            params = {"type": company or "auto", "postid": number}
            resp = await client.get(url, params=params)
            data = resp.json()

        if data.get("status") != "200" and not data.get("data"):
            # 尝试备用格式
            msg = data.get("message") or data.get("msg") or "未查到物流信息"
            return f"查询失败: {msg}"

        traces = data.get("data", [])
        if not traces:
            return f"快递单号 {number} 暂无物流信息。"

        com_name = data.get("com", company or "未知")
        state_map = {"0": "运输中", "1": "揽收", "2": "疑难", "3": "已签收", "4": "退签", "5": "派件中", "6": "退回"}
        state = state_map.get(str(data.get("state", "")), "未知")

        lines = [f"📦 {com_name} {number} [{state}]", ""]
        for t in traces[:10]:
            time_str = t.get("ftime") or t.get("time", "")
            context = t.get("context", "")
            lines.append(f"  {time_str}  {context}")
        return "\n".join(lines)
    except Exception as e:
        return f"快递查询失败: {e}"


async def _tool_flight_info(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """航班查询（暂用 web_search 代理）."""
    flight_no = (args.get("flight_no") or "").strip().upper()
    if not flight_no:
        return "需要航班号。"

    date = (args.get("date") or "").strip()
    return f"航班查询功能开发中。请使用 web_search 搜索「{flight_no} {date} 航班动态」获取信息。"


async def _tool_aqi(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """空气质量查询."""
    import httpx

    city = (args.get("city") or "").strip()
    if not city:
        return "需要城市名。"

    # 中文城市名映射
    city_map = {
        "上海": "shanghai", "北京": "beijing", "广州": "guangzhou",
        "深圳": "shenzhen", "杭州": "hangzhou", "成都": "chengdu",
        "重庆": "chongqing", "武汉": "wuhan", "南京": "nanjing",
        "西安": "xian", "苏州": "suzhou", "天津": "tianjin",
        "长沙": "changsha", "郑州": "zhengzhou", "青岛": "qingdao",
        "大连": "dalian", "厦门": "xiamen", "昆明": "kunming",
        "合肥": "hefei", "福州": "fuzhou",
    }
    query = city_map.get(city, city)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"https://api.waqi.info/feed/{query}/",
                params={"token": "demo"},
            )
            data = resp.json()

        if data.get("status") != "ok":
            return f"未找到 {city} 的空气质量数据。"

        d = data["data"]
        aqi_val = d.get("aqi", "N/A")
        station = d.get("city", {}).get("name", city)
        time_str = d.get("time", {}).get("s", "")

        # AQI 等级
        if isinstance(aqi_val, int):
            if aqi_val <= 50:
                level = "优 🟢"
            elif aqi_val <= 100:
                level = "良 🟡"
            elif aqi_val <= 150:
                level = "轻度污染 🟠"
            elif aqi_val <= 200:
                level = "中度污染 🔴"
            elif aqi_val <= 300:
                level = "重度污染 🟤"
            else:
                level = "严重污染 ⚫"
        else:
            level = ""

        iaqi = d.get("iaqi", {})
        lines = [f"🌍 {station}", f"AQI: {aqi_val} {level}"]
        if iaqi.get("pm25"):
            lines.append(f"PM2.5: {iaqi['pm25'].get('v', 'N/A')}")
        if iaqi.get("pm10"):
            lines.append(f"PM10: {iaqi['pm10'].get('v', 'N/A')}")
        if iaqi.get("o3"):
            lines.append(f"O₃: {iaqi['o3'].get('v', 'N/A')}")
        if iaqi.get("t"):
            lines.append(f"温度: {iaqi['t'].get('v', 'N/A')}℃")
        if iaqi.get("h"):
            lines.append(f"湿度: {iaqi['h'].get('v', 'N/A')}%")
        if time_str:
            lines.append(f"更新: {time_str}")
        return "\n".join(lines)
    except Exception as e:
        return f"空气质量查询失败: {e}"


_TOOL_HANDLERS: dict[str, Any] = {
    "query_stats": _tool_query_stats,
    "send_message": _tool_send_message,
    "list_agents": _tool_list_agents,
    "web_search": _tool_web_search,
    "create_note": _tool_create_note,
    "lookup_user": _tool_lookup_user,
    "query_agent_work": _tool_query_agent_work,
    "read_notes": _tool_read_notes,
    "read_messages": _tool_read_messages,
    "get_system_health": _tool_get_system_health,
    "mark_read": _tool_mark_read,
    "update_agent": _tool_update_agent,
    "read_feishu_calendar": _tool_read_feishu_calendar,
    "delete_feishu_event": _tool_delete_feishu_event,
    "create_feishu_event": _tool_create_feishu_event,
    "create_feishu_task": _tool_create_feishu_task,
    "list_feishu_tasks": _tool_list_feishu_tasks,
    "complete_feishu_task": _tool_complete_feishu_task,
    "delete_feishu_task": _tool_delete_feishu_task,
    "update_feishu_task": _tool_update_feishu_task,
    "feishu_chat_history": _tool_feishu_chat_history,
    "weather": _tool_weather,
    "get_datetime": _tool_get_datetime,
    "calculate": _tool_calculate,
    # 飞书文档
    "search_feishu_docs": _tool_search_feishu_docs,
    "read_feishu_doc": _tool_read_feishu_doc,
    "create_feishu_doc": _tool_create_feishu_doc,
    "send_feishu_group": _tool_send_feishu_group,
    "list_feishu_groups": _tool_list_feishu_groups,
    "send_feishu_dm": _tool_send_feishu_dm,
    "feishu_group_members": _tool_feishu_group_members,
    "exchange_rate": _tool_exchange_rate,
    "stock_price": _tool_stock_price,
    # GitHub
    "github_prs": _tool_github_prs,
    "github_issues": _tool_github_issues,
    "github_repo_activity": _tool_github_repo_activity,
    # Notion
    "notion_search": _tool_notion_search,
    "notion_read": _tool_notion_read,
    "notion_create": _tool_notion_create,
    # 信息采集
    "read_url": _tool_read_url,
    "rss_read": _tool_rss_read,
    # 生活助手
    "translate": _tool_translate,
    "countdown": _tool_countdown,
    "trending": _tool_trending,
    # 飞书表格 & 审批
    "read_feishu_sheet": _tool_read_feishu_sheet,
    "update_feishu_sheet": _tool_update_feishu_sheet,
    "list_feishu_approvals": _tool_list_feishu_approvals,
    # 实用工具
    "unit_convert": _tool_unit_convert,
    "random_pick": _tool_random_pick,
    "holidays": _tool_holidays,
    "timestamp_convert": _tool_timestamp_convert,
    "create_feishu_spreadsheet": _tool_create_feishu_spreadsheet,
    "feishu_contacts": _tool_feishu_contacts,
    # 文本 & 开发
    "text_extract": _tool_text_extract,
    "json_format": _tool_json_format,
    "password_gen": _tool_password_gen,
    "ip_lookup": _tool_ip_lookup,
    "short_url": _tool_short_url,
    "word_count": _tool_word_count,
    # 编码 & 开发辅助
    "base64_codec": _tool_base64_codec,
    "color_convert": _tool_color_convert,
    "cron_explain": _tool_cron_explain,
    "regex_test": _tool_regex_test,
    "hash_gen": _tool_hash_gen,
    "url_codec": _tool_url_codec,
    # 飞书增强
    "feishu_bitable": _tool_feishu_bitable,
    "feishu_wiki": _tool_feishu_wiki,
    "approve_feishu": _tool_approve_feishu,
    # AI 能力
    "summarize": _tool_summarize,
    "sentiment": _tool_sentiment,
    # 效率工具
    "email_send": _tool_email_send,
    "qrcode": _tool_qrcode,
    "diff_text": _tool_diff_text,
    # 数据查询
    "whois": _tool_whois,
    "dns_lookup": _tool_dns_lookup,
    "http_check": _tool_http_check,
    # 生活服务
    "express_track": _tool_express_track,
    "flight_info": _tool_flight_info,
    "aqi": _tool_aqi,
    # 异步委派 & 会议编排
    "delegate_async": _tool_delegate_async,
    "check_task": _tool_check_task,
    "list_tasks": _tool_list_tasks,
    "organize_meeting": _tool_organize_meeting,
    "check_meeting": _tool_check_meeting,
    # 流水线 & 委派链
    "run_pipeline": _tool_run_pipeline,
    "delegate_chain": _tool_delegate_chain,
    # 定时任务
    "schedule_task": _tool_schedule_task,
    "list_schedules": _tool_list_schedules,
    "cancel_schedule": _tool_cancel_schedule,
    # 文件 & 代码
    "agent_file_read": _tool_agent_file_read,
    "agent_file_grep": _tool_agent_file_grep,
    # 数据 & 日程
    "query_data": _tool_query_data,
    "find_free_time": _tool_find_free_time,
}


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
        AGENT_TOOLS, DEFERRED_TOOLS, employee_tools_to_schemas,
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
        except Exception:
            pass

    engine = CrewEngine(project_dir=ctx.project_dir)
    prompt = engine.prompt(match, args=args, agent_identity=agent_identity)

    # 如果有 delegate 工具，追加同事名单
    if "delegate" in (match.tools or []):
        roster_lines: list[str] = []
        for emp_name, emp in discovery.employees.items():
            if emp_name != name:
                label = emp.character_name or emp.effective_display_name
                roster_lines.append(f"- {emp_name}（{label}）：{emp.description}")
        if roster_lines:
            prompt += "\n\n---\n\n## 可委派的同事\n\n使用 delegate 工具调用他们。\n\n" + "\n".join(
                roster_lines
            )

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
            messages.append({"role": h["role"], "content": h["content"]})

    task_text = user_message or args.get("task", "请开始执行上述任务。")
    messages.append({"role": "user", "content": task_text})

    total_input = 0
    total_output = 0
    final_content = ""
    rounds = 0

    # 解析 agent_id（从 match 的 agent_id 属性，或参数）
    effective_agent_id = agent_id or getattr(match, "agent_id", None)

    for rounds in range(_MAX_TOOL_ROUNDS):  # noqa: B007
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
                    ctx, name, tc.name, tc.arguments, effective_agent_id,
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
                    ctx, name, tc.name, tc.arguments, effective_agent_id,
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
    }


async def _handle_tool_call(
    ctx: _AppContext,
    employee_name: str,
    tool_name: str,
    arguments: dict[str, Any],
    agent_id: int | None,
) -> str | None:
    """处理单个 tool call，返回结果字符串。返回 None 表示 finish tool."""
    from crew.tool_schema import is_finish_tool

    if is_finish_tool(tool_name):
        return None

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
        return await _delegate_employee(
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
        return await _execute_employee_with_tools(
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
        except Exception:
            pass

    engine = CrewEngine(project_dir=ctx.project_dir)
    prompt = engine.prompt(match, args=args, agent_identity=agent_identity)

    # 尝试执行 LLM 调用（executor 自动从环境变量解析 API key）
    try:
        from crew.executor import aexecute_prompt

        use_model = model or match.model or "claude-sonnet-4-20250514"
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
