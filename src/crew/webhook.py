"""Webhook 服务器 — 接收外部事件，触发 crew pipeline / 员工执行."""

from __future__ import annotations

import asyncio
import logging
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
from crew.webhook_context import (  # noqa: F401
    _AppContext,
    _EMPLOYEE_UPDATABLE_FIELDS,
    _GITHUB_API_BASE,
    _GITHUB_REPO_RE,
    _GITHUB_TOKEN,
    _ID_API_BASE,
    _ID_API_TOKEN,
    _MAX_TOOL_ROUNDS,
    _NOTION_API_BASE,
    _NOTION_API_KEY,
    _NOTION_VERSION,
)

# ── re-export: 工具函数（测试和外部代码仍从 crew.webhook import）──
from crew.webhook_tools.orchestration import (  # noqa: F401
    _tool_agent_file_grep,
    _tool_agent_file_read,
    _tool_cancel_schedule,
    _tool_check_meeting,
    _tool_check_task,
    _tool_delegate_async,
    _tool_delegate_chain,
    _tool_find_free_time,
    _tool_list_schedules,
    _tool_list_tasks,
    _tool_organize_meeting,
    _tool_query_data,
    _tool_run_pipeline,
    _tool_schedule_task,
)
from crew.webhook_tools.data_query import (  # noqa: F401
    _tool_create_note,
    _tool_get_system_health,
    _tool_list_agents,
    _tool_lookup_user,
    _tool_mark_read,
    _tool_query_agent_work,
    _tool_query_stats,
    _tool_read_messages,
    _tool_read_notes,
    _tool_send_message,
    _tool_update_agent,
)
from crew.webhook_tools.external import (  # noqa: F401
    _tool_web_search,
)
from crew.webhook_tools.feishu import (  # noqa: F401
    _tool_read_feishu_calendar,
)

# ── re-export: 执行引擎 ──
from crew.webhook_executor import (  # noqa: F401
    _TOOL_HANDLERS,
    _delegate_employee,
    _dispatch_task,
    _execute_chain,
    _execute_employee,
    _execute_employee_with_tools,
    _execute_meeting,
    _execute_pipeline,
    _execute_task,
    _handle_tool_call,
    _resume_chain,
    _resume_incomplete_pipelines,
    _stream_employee,
)

# ── re-export: HTTP handler ──
from crew.webhook_handlers import (  # noqa: F401
    _handle_agent_run,
    _handle_authority_restore,
    _handle_cost_summary,
    _handle_cron_status,
    _handle_employee_delete,
    _handle_employee_prompt,
    _handle_employee_state,
    _handle_employee_update,
    _handle_generic,
    _handle_github,
    _handle_memory_ingest,
    _handle_openclaw,
    _handle_project_status,
    _handle_run_employee,
    _handle_run_pipeline,
    _handle_task_approve,
    _handle_task_replay,
    _handle_task_status,
    _health,
    _metrics,
)

# ── re-export: 飞书处理 ──
from crew.webhook_feishu import (  # noqa: F401
    _feishu_dispatch,
    _find_recent_image_in_chat,
    _handle_feishu_event,
)


def _make_handler(ctx: _AppContext, handler):
    """包装 handler，注入 context."""
    async def wrapper(request: Request):
        return await handler(request, ctx)
    return wrapper


def create_webhook_app(
    project_dir: Path | None = None,
    token: str | None = None,
    config: "WebhookConfig | None" = None,
    cron_config: "CronConfig | None" = None,
    cors_origins: list[str] | None = None,
    feishu_config: "FeishuConfig | None" = None,
) -> "Starlette":
    """创建 webhook Starlette 应用."""
    if not HAS_STARLETTE:
        raise ImportError("starlette 未安装。请运行: pip install knowlyr-crew[webhook]")

    from crew.webhook_config import WebhookConfig

    if config is None:
        config = WebhookConfig()

    # 任务持久化
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
        Route("/api/tasks/{task_id}/approve", endpoint=_make_handler(ctx, _handle_task_approve), methods=["POST"]),
        Route("/cron/status", endpoint=_make_handler(ctx, _handle_cron_status), methods=["GET"]),
        Route("/feishu/event", endpoint=_make_handler(ctx, _handle_feishu_event), methods=["POST"]),
        Route("/api/employees/{identifier}/prompt", endpoint=_make_handler(ctx, _handle_employee_prompt), methods=["GET"]),
        Route("/api/employees/{identifier}/state", endpoint=_make_handler(ctx, _handle_employee_state), methods=["GET"]),
        Route("/api/employees/{identifier}", endpoint=_make_handler(ctx, _handle_employee_update), methods=["PUT"]),
        Route("/api/employees/{identifier}", endpoint=_make_handler(ctx, _handle_employee_delete), methods=["DELETE"]),
        Route("/api/employees/{identifier}/authority/restore", endpoint=_make_handler(ctx, _handle_authority_restore), methods=["POST"]),
        Route("/api/memory/ingest", endpoint=_make_handler(ctx, _handle_memory_ingest), methods=["POST"]),
        Route("/api/cost/summary", endpoint=_make_handler(ctx, _handle_cost_summary), methods=["GET"]),
        Route("/api/project/status", endpoint=_make_handler(ctx, _handle_project_status), methods=["GET"]),
    ]

    async def on_startup():
        if scheduler:
            await scheduler.start()
        try:
            from crew.id_client import HeartbeatManager

            ctx.heartbeat_mgr = HeartbeatManager(interval=60.0)
            await ctx.heartbeat_mgr.start()
        except ImportError:
            pass
        asyncio.create_task(_resume_incomplete_pipelines(ctx))

    async def on_shutdown():
        if scheduler:
            await scheduler.stop()
        if ctx.heartbeat_mgr:
            await ctx.heartbeat_mgr.stop()

    app = Starlette(routes=routes, on_startup=[on_startup], on_shutdown=[on_shutdown])

    from crew.auth import RequestSizeLimitMiddleware

    app.add_middleware(RequestSizeLimitMiddleware)

    if token:
        from crew.auth import BearerTokenMiddleware, RateLimitMiddleware

        skip_paths = ["/health", "/webhook/github", "/feishu/event"]
        app.add_middleware(BearerTokenMiddleware, token=token, skip_paths=skip_paths)
        app.add_middleware(
            RateLimitMiddleware,
            skip_paths=["/health", "/metrics", "/webhook/github"],
        )

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
