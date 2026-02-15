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
    tool_schemas = employee_tools_to_schemas(employee.tools)

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
        "api_key": employee.api_key or "",
        "base_url": employee.base_url or "",
    })


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

        # 图片消息处理（当前不支持 vision）
        if msg_event.msg_type == "image":
            await send_feishu_text(
                ctx.feishu_token_mgr,
                msg_event.chat_id,
                "我暂时看不了图片，你描述一下？",
            )
            return

        # 发送"处理中"提示
        import random as _random

        _THINKING_MSGS = [
            "让我看看...",
            "稍等，我查一下",
            "好的，马上",
            "收到，稍等",
        ]
        await send_feishu_text(
            ctx.feishu_token_mgr,
            msg_event.chat_id,
            _random.choice(_THINKING_MSGS),
        )

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
        result = await _execute_employee(
            ctx, employee_name, args, model=None,
            user_message=task_text,
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
            error_text = f"出了点问题：{e}"
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


_MAX_TOOL_ROUNDS = 10


# ── Tool handlers（调用 knowlyr-id API）──

import os as _os

_ID_API_BASE = _os.environ.get("KNOWLYR_ID_API", "https://id.knowlyr.com")
_ID_API_TOKEN = _os.environ.get("AGENT_API_TOKEN", "")
_GITHUB_TOKEN = _os.environ.get("GITHUB_TOKEN", "")
_GITHUB_API_BASE = "https://api.github.com"
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

    from crew.feishu import create_calendar_event

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

    url = (args.get("url") or "").strip()
    if not url:
        return "缺少 URL。"

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
    "feishu_chat_history": _tool_feishu_chat_history,
    "weather": _tool_weather,
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
}


async def _execute_employee_with_tools(
    ctx: _AppContext,
    name: str,
    args: dict[str, str],
    *,
    agent_id: int | None = None,
    model: str | None = None,
    user_message: str | None = None,
    message_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """执行带工具的员工（agent loop with tools）."""
    from crew.discovery import discover_employees
    from crew.engine import CrewEngine
    from crew.executor import aexecute_with_tools
    from crew.providers import Provider, detect_provider
    from crew.tool_schema import AGENT_TOOLS, employee_tools_to_schemas, is_finish_tool

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
    tool_schemas = employee_tools_to_schemas(agent_tool_names)

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
    user_message: str | None = None,
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
