"""Webhook 服务器 — 接收外部事件，触发 crew pipeline / 员工执行."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 后台启动任务引用集合 — 防止 GC 提前回收
_startup_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]

try:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, StreamingResponse  # noqa: F401
    from starlette.routing import Route

    HAS_STARLETTE = True
except ImportError:
    HAS_STARLETTE = False

from crew.task_registry import TaskRegistry
from crew.webhook_context import (  # noqa: F401
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
    _AppContext,
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

# ── re-export: 飞书处理 ──
try:
    from crew.webhook_feishu import (  # noqa: F401
        _feishu_dispatch,
        _find_recent_image_in_chat,
        _handle_feishu_event,
    )
except ImportError:
    pass

# ── re-export: 企微处理 ──
try:
    from crew.webhook_wecom import (  # noqa: F401
        handle_wecom_event,
    )
except ImportError:
    pass

# ── re-export: HTTP handler ──
from crew.webhook_handlers import (  # noqa: F401
    _handle_agent_run,
    _handle_audit_trends,
    _handle_authority_restore,
    _handle_chat,
    _handle_cost_summary,
    _handle_cron_status,
    _handle_decision_evaluate,
    _handle_decision_track,
    _handle_discussion_create,
    _handle_discussion_get,
    _handle_discussion_list,
    _handle_discussion_list_config,
    _handle_discussion_plan,
    _handle_discussion_prompt,
    _handle_discussion_update,
    _handle_employee_copy,
    _handle_employee_create,
    _handle_employee_delete,
    _handle_employee_get,
    _handle_employee_list,
    _handle_employee_prompt,
    _handle_employee_state,
    _handle_employee_update,
    _handle_evaluate_scan,
    _handle_evolution_candidates,
    _handle_evolution_review,
    _handle_generic,
    _handle_github,
    _handle_knowledge_dashboard,
    _handle_kv_get,
    _handle_kv_list,
    _handle_kv_put,
    _handle_meeting_detail,
    _handle_meeting_list,
    _handle_memory_add,
    _handle_memory_archive_query,
    _handle_memory_archive_restore,
    _handle_memory_archive_stats,
    _handle_memory_batch_delete,
    _handle_memory_batch_update,
    _handle_memory_consolidate,
    _handle_memory_dashboard,
    _handle_memory_delete,
    _handle_memory_drafts_approve,
    _handle_memory_drafts_get,
    _handle_memory_drafts_list,
    _handle_memory_drafts_reject,
    _handle_memory_feedback_get,
    _handle_memory_feedback_submit,
    _handle_memory_feedback_summary,
    _handle_memory_ingest,
    _handle_memory_low_quality,
    _handle_memory_popular,
    _handle_memory_query,
    _handle_memory_recommend,
    _handle_memory_search,
    _handle_memory_semantic_search,
    _handle_memory_shared_list,
    _handle_memory_shared_record_usage,
    _handle_memory_shared_stats,
    _handle_memory_similar,
    _handle_memory_tags_list,
    _handle_memory_tags_search,
    _handle_memory_tags_suggest,
    _handle_memory_update,
    _handle_memory_usage_record,
    _handle_memory_usage_stats,
    _handle_model_tiers,
    _handle_openclaw,
    _handle_org_memories,
    _handle_permission_list,
    _handle_permission_matrix,
    _handle_permission_respond,
    _handle_pipeline_create_config,
    _handle_pipeline_get_config,
    _handle_pipeline_list,
    _handle_pipeline_list_config,
    _handle_pipeline_update_config,
    _handle_project_status,
    _handle_recall_feedback,
    _handle_run_employee,
    _handle_run_pipeline,
    _handle_run_route,
    _handle_soul_get,
    _handle_soul_list,
    _handle_soul_update,
    _handle_task_approve,
    _handle_task_replay,
    _handle_task_status,
    _handle_team_agents,
    _handle_tenant_create,
    _handle_tenant_delete,
    _handle_tenant_get,
    _handle_tenant_list,
    _handle_tenant_update,
    _handle_trajectory_annotation_add,
    _handle_trajectory_annotation_list,
    _handle_trajectory_export,
    _handle_trajectory_report,
    _handle_wiki_file_delete,
    _handle_wiki_spaces_list,
    _handle_work_log,
    _health,
    _metrics,
)
from crew.webhook_tools.core import (  # noqa: F401  # noqa: F401
    _tool_create_note,
    _tool_list_agents,
    _tool_lookup_user,
    _tool_read_notes,
    _tool_send_message,
    _tool_web_search,
)

try:
    from crew.webhook_tools.feishu import (  # noqa: F401
        _tool_read_feishu_calendar,
    )
except ImportError:
    pass

# ── re-export: 工具函数（测试和外部代码仍从 crew.webhook import）──
# ── re-export: Skills handlers ──
from crew.webhook_skills import (  # noqa: F401
    _handle_skill_create,
    _handle_skill_delete,
    _handle_skill_get,
    _handle_skill_list,
    _handle_skill_update,
    _handle_skills_check_triggers,
    _handle_skills_execute,
    _handle_skills_stats,
    _handle_skills_trigger_history,
)
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
    _tool_run_pipeline,
    _tool_schedule_task,
)


def _make_handler(ctx: _AppContext, handler):
    """包装 handler，注入 context."""

    async def wrapper(request: Request):
        return await handler(request, ctx)

    return wrapper


def create_webhook_app(
    project_dir: Path | None = None,
    token: str | None = None,
    config: WebhookConfig | None = None,  # noqa: F821
    cron_config: CronConfig | None = None,  # noqa: F821
    cors_origins: list[str] | None = None,
    feishu_config: FeishuConfig | None = None,  # noqa: F821
    feishu_bots: list | None = None,
    wecom_config: WecomConfig | None = None,  # noqa: F821
) -> Starlette:
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

    # 初始化飞书 Bot（支持多 Bot）
    from crew.feishu import EventDeduplicator, FeishuBotConfig, FeishuTokenManager
    from crew.webhook_context import FeishuBotContext

    _bot_configs: list[FeishuBotConfig] = list(feishu_bots or [])

    # 兼容旧的单 feishu_config 参数
    if not _bot_configs and feishu_config and feishu_config.app_id and feishu_config.app_secret:
        _bot_configs = [
            FeishuBotConfig(bot_id="default", primary=True, **feishu_config.model_dump())
        ]

    for bot_cfg in _bot_configs:
        if not (bot_cfg.app_id and bot_cfg.app_secret):
            continue
        bot_ctx = FeishuBotContext(
            config=bot_cfg,
            token_mgr=FeishuTokenManager(bot_cfg.app_id, bot_cfg.app_secret),
            dedup=EventDeduplicator(),
        )
        ctx.feishu_bots[bot_cfg.bot_id] = bot_ctx
        # primary bot 同步到旧字段（工具调用向后兼容）
        if bot_cfg.primary:
            ctx.feishu_config = bot_cfg
            ctx.feishu_token_mgr = bot_ctx.token_mgr
            ctx.feishu_dedup = bot_ctx.dedup
        logger.info(
            "飞书 Bot 已启用: bot_id=%s app_id=%s default=%s",
            bot_cfg.bot_id,
            bot_cfg.app_id,
            bot_cfg.default_employee,
        )

    if ctx.feishu_bots:
        from crew.feishu_memory import FeishuChatStore

        chat_store_dir = (project_dir or Path(".")) / ".crew" / "feishu-chats"
        ctx.feishu_chat_store = FeishuChatStore(chat_store_dir)

    # 初始化企业微信
    if wecom_config and wecom_config.corp_id and wecom_config.secret:
        from crew.feishu import EventDeduplicator
        from crew.wecom import WecomCrypto, WecomTokenManager

        ctx.wecom_ctx = {
            "config": wecom_config,
            "crypto": WecomCrypto(wecom_config.encoding_aes_key, wecom_config.corp_id),
            "token_mgr": WecomTokenManager(wecom_config.corp_id, wecom_config.secret),
            "dedup": EventDeduplicator(),
        }
        # 通讯录同步独立密钥（如已配置）
        if wecom_config.contact_token and wecom_config.contact_encoding_aes_key:
            ctx.wecom_ctx["contact_crypto"] = WecomCrypto(
                wecom_config.contact_encoding_aes_key, wecom_config.corp_id
            )
            ctx.wecom_ctx["contact_token"] = wecom_config.contact_token
            logger.info("企微通讯录同步回调密钥已加载")
        if wecom_config.contact_secret:
            ctx.wecom_ctx["contact_token_mgr"] = WecomTokenManager(
                wecom_config.corp_id, wecom_config.contact_secret
            )
        logger.info(
            "企微 Bot 已启用: corp_id=%s agent_id=%d default=%s",
            wecom_config.corp_id,
            wecom_config.agent_id,
            wecom_config.default_employee,
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

            # 投递结果到配置的渠道
            if schedule.delivery:
                record = ctx.registry.get(record.task_id)
                try:
                    from crew.delivery import DeliveryTarget as DT
                    from crew.delivery import deliver

                    targets = [DT(**t.model_dump()) for t in schedule.delivery]
                    results = await deliver(
                        targets,
                        task_name=schedule.name,
                        task_result=record.result if record else None,
                        task_error=record.error if record else None,
                    )
                    for r in results:
                        if not r.success:
                            logger.warning(
                                "投递失败 [%s → %s]: %s", schedule.name, r.target_type, r.detail
                            )
                except Exception as e:
                    logger.warning("投递异常 [%s]: %s", schedule.name, e)

        scheduler = CronScheduler(config=cron_config, execute_fn=_cron_execute)
        ctx.scheduler = scheduler

    # ── 打卡通报 cron 循环 ──
    async def _run_checkin_cron() -> None:
        """每天 10:30（北京时间）执行打卡通报."""
        try:
            from croniter import croniter
        except ImportError:
            logger.error("croniter 未安装，打卡通报定时任务无法启动")
            return

        from datetime import datetime as _dt

        cron = croniter("30 10 * * *", _dt.now())
        logger.info("打卡通报定时任务已注册: 30 10 * * * (每天 10:30)")

        while True:
            try:
                import time as _time

                next_time = cron.get_next(float)
                delay = next_time - _time.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(0)

                logger.info("触发打卡通报定时任务")
                from crew.wecom_checkin import cron_checkin_report

                await cron_checkin_report()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("打卡通报定时任务异常，60s 后重试")
                await asyncio.sleep(60)

    # ── 决策扫描 cron 循环 ──
    async def _run_evaluate_scan_cron() -> None:
        """每天 10:00（北京时间）执行过期决策扫描."""
        try:
            from croniter import croniter
        except ImportError:
            logger.error("croniter 未安装，决策扫描定时任务无法启动")
            return

        from datetime import datetime as _dt

        cron = croniter("0 10 * * *", _dt.now())
        logger.info("决策扫描定时任务已注册: 0 10 * * * (每天 10:00)")

        while True:
            try:
                import time as _time

                next_time = cron.get_next(float)
                delay = next_time - _time.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(0)

                logger.info("触发决策扫描定时任务")
                from crew.cron_evaluate import format_scan_report, scan_overdue_decisions

                results = await scan_overdue_decisions()
                report = format_scan_report(results)

                if report:
                    logger.info(
                        "决策扫描结果: %d 自动评估, %d 待提醒, %d 超期关闭",
                        len(results.get("auto_evaluated", [])),
                        len(results.get("reminders", [])),
                        len(results.get("expired", [])),
                    )
                    # 私信发送给 Kai（使用 owner_open_id）
                    if (
                        ctx.feishu_token_mgr
                        and ctx.feishu_config
                        and ctx.feishu_config.owner_open_id
                    ):
                        try:
                            import json as _json

                            from crew.feishu import get_feishu_client

                            token = await ctx.feishu_token_mgr.get_token()
                            client = get_feishu_client()
                            resp = await client.post(
                                "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id",
                                headers={
                                    "Authorization": f"Bearer {token}",
                                    "Content-Type": "application/json",
                                },
                                json={
                                    "receive_id": ctx.feishu_config.owner_open_id,
                                    "msg_type": "text",
                                    "content": _json.dumps({"text": report}),
                                },
                                timeout=15.0,
                            )
                            data = resp.json()
                            if data.get("code") == 0:
                                logger.info("决策扫描报告已私信发送给 owner")
                            else:
                                logger.warning("决策扫描私信发送失败: %s", data.get("msg", ""))
                        except Exception as e:
                            logger.warning("决策扫描私信发送异常: %s", e)
                    else:
                        logger.info("决策扫描: 飞书未配置或 owner_open_id 未设置，跳过通知")
                else:
                    logger.info("决策扫描: 无过期决策")
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("决策扫描定时任务异常，60s 后重试")
                await asyncio.sleep(60)

    routes = [
        Route("/health", endpoint=_health, methods=["GET"]),
        Route("/metrics", endpoint=_metrics, methods=["GET"]),
        Route("/webhook/github", endpoint=_make_handler(ctx, _handle_github), methods=["POST"]),
        Route("/webhook/openclaw", endpoint=_make_handler(ctx, _handle_openclaw), methods=["POST"]),
        Route("/webhook", endpoint=_make_handler(ctx, _handle_generic), methods=["POST"]),
        Route(
            "/run/pipeline/{name}",
            endpoint=_make_handler(ctx, _handle_run_pipeline),
            methods=["POST"],
        ),
        Route(
            "/run/route/{name}", endpoint=_make_handler(ctx, _handle_run_route), methods=["POST"]
        ),
        Route(
            "/run/employee/{name}",
            endpoint=_make_handler(ctx, _handle_run_employee),
            methods=["POST"],
        ),
        Route(
            "/agent/run/{name}", endpoint=_make_handler(ctx, _handle_agent_run), methods=["POST"]
        ),
        Route(
            "/tasks/{task_id}", endpoint=_make_handler(ctx, _handle_task_status), methods=["GET"]
        ),
        Route(
            "/tasks/{task_id}/replay",
            endpoint=_make_handler(ctx, _handle_task_replay),
            methods=["POST"],
        ),
        Route(
            "/api/tasks/{task_id}/approve",
            endpoint=_make_handler(ctx, _handle_task_approve),
            methods=["POST"],
        ),
        Route("/cron/status", endpoint=_make_handler(ctx, _handle_cron_status), methods=["GET"]),
        # 飞书: 每个 bot 独立端点 + 旧路由兼容
        Route("/feishu/event", endpoint=_make_handler(ctx, _handle_feishu_event), methods=["POST"]),
        *[
            Route(
                f"/feishu/event/{bot_id}",
                endpoint=_make_handler(ctx, _handle_feishu_event),
                methods=["POST"],
            )
            for bot_id in ctx.feishu_bots
        ],
        # 企业微信: GET 验证 + POST 消息接收
        Route(
            "/wecom/event/{app_id}",
            endpoint=_make_handler(ctx, handle_wecom_event),
            methods=["GET", "POST"],
        ),
        Route(
            "/api/team/agents",
            endpoint=_make_handler(ctx, _handle_team_agents),
            methods=["GET"],
        ),
        Route(
            "/api/employees",
            endpoint=_make_handler(ctx, _handle_employee_list),
            methods=["GET"],
        ),
        Route(
            "/api/employees/copy",
            endpoint=_make_handler(ctx, _handle_employee_copy),
            methods=["POST"],
        ),
        Route(
            "/api/employees",
            endpoint=_make_handler(ctx, _handle_employee_create),
            methods=["POST"],
        ),
        Route(
            "/api/employees/{identifier}/prompt",
            endpoint=_make_handler(ctx, _handle_employee_prompt),
            methods=["GET"],
        ),
        Route(
            "/api/employees/{identifier}/state",
            endpoint=_make_handler(ctx, _handle_employee_state),
            methods=["GET"],
        ),
        Route(
            "/api/employees/{identifier}",
            endpoint=_make_handler(ctx, _handle_employee_get),
            methods=["GET"],
        ),
        Route(
            "/api/employees/{identifier}",
            endpoint=_make_handler(ctx, _handle_employee_update),
            methods=["PUT"],
        ),
        Route(
            "/api/employees/{identifier}",
            endpoint=_make_handler(ctx, _handle_employee_delete),
            methods=["DELETE"],
        ),
        Route(
            "/api/employees/{identifier}/authority/restore",
            endpoint=_make_handler(ctx, _handle_authority_restore),
            methods=["POST"],
        ),
        Route(
            "/api/memory/add",
            endpoint=_make_handler(ctx, _handle_memory_add),
            methods=["POST"],
        ),
        Route(
            "/api/memory/query",
            endpoint=_make_handler(ctx, _handle_memory_query),
            methods=["GET"],
        ),
        Route(
            "/api/memory/tags",
            endpoint=_make_handler(ctx, _handle_memory_tags_list),
            methods=["GET"],
        ),
        Route(
            "/api/memory/tags/suggest",
            endpoint=_make_handler(ctx, _handle_memory_tags_suggest),
            methods=["GET"],
        ),
        Route(
            "/api/memory/tags/search",
            endpoint=_make_handler(ctx, _handle_memory_tags_search),
            methods=["GET"],
        ),
        Route(
            "/api/permissions",
            endpoint=_make_handler(ctx, _handle_permission_list),
            methods=["GET"],
        ),
        Route(
            "/api/permissions/respond",
            endpoint=_make_handler(ctx, _handle_permission_respond),
            methods=["POST"],
        ),
        Route(
            "/api/memory/ingest",
            endpoint=_make_handler(ctx, _handle_memory_ingest),
            methods=["POST"],
        ),
        Route(
            "/api/memory/search",
            endpoint=_make_handler(ctx, _handle_memory_search),
            methods=["GET"],
        ),
        Route(
            "/api/memory/{entry_id}",
            endpoint=_make_handler(ctx, _handle_memory_delete),
            methods=["DELETE"],
        ),
        Route(
            "/api/memory/update",
            endpoint=_make_handler(ctx, _handle_memory_update),
            methods=["PUT"],
        ),
        Route(
            "/api/memory/drafts",
            endpoint=_make_handler(ctx, _handle_memory_drafts_list),
            methods=["GET"],
        ),
        Route(
            "/api/memory/drafts/{draft_id}",
            endpoint=_make_handler(ctx, _handle_memory_drafts_get),
            methods=["GET"],
        ),
        Route(
            "/api/memory/drafts/{draft_id}/approve",
            endpoint=_make_handler(ctx, _handle_memory_drafts_approve),
            methods=["POST"],
        ),
        Route(
            "/api/memory/drafts/{draft_id}/reject",
            endpoint=_make_handler(ctx, _handle_memory_drafts_reject),
            methods=["POST"],
        ),
        Route(
            "/api/memory/archive",
            endpoint=_make_handler(ctx, _handle_memory_archive_query),
            methods=["GET"],
        ),
        Route(
            "/api/memory/archive/restore",
            endpoint=_make_handler(ctx, _handle_memory_archive_restore),
            methods=["POST"],
        ),
        Route(
            "/api/memory/archive/stats",
            endpoint=_make_handler(ctx, _handle_memory_archive_stats),
            methods=["GET"],
        ),
        Route(
            "/api/memory/shared",
            endpoint=_make_handler(ctx, _handle_memory_shared_list),
            methods=["GET"],
        ),
        Route(
            "/api/memory/shared/usage",
            endpoint=_make_handler(ctx, _handle_memory_shared_record_usage),
            methods=["POST"],
        ),
        Route(
            "/api/memory/shared/stats",
            endpoint=_make_handler(ctx, _handle_memory_shared_stats),
            methods=["GET"],
        ),
        Route(
            "/api/memory/dashboard",
            endpoint=_make_handler(ctx, _handle_memory_dashboard),
            methods=["GET"],
        ),
        Route(
            "/api/knowledge/dashboard",
            endpoint=_make_handler(ctx, _handle_knowledge_dashboard),
            methods=["GET"],
        ),
        Route(
            "/api/memory/recall-feedback",
            endpoint=_make_handler(ctx, _handle_recall_feedback),
            methods=["POST"],
        ),
        Route(
            "/api/memory/batch/update",
            endpoint=_make_handler(ctx, _handle_memory_batch_update),
            methods=["POST"],
        ),
        Route(
            "/api/memory/batch/delete",
            endpoint=_make_handler(ctx, _handle_memory_batch_delete),
            methods=["POST"],
        ),
        Route(
            "/api/memory/consolidate",
            endpoint=_make_handler(ctx, _handle_memory_consolidate),
            methods=["POST"],
        ),
        Route(
            "/api/memory/org", endpoint=_make_handler(ctx, _handle_org_memories), methods=["GET"]
        ),
        Route(
            "/api/cost/summary", endpoint=_make_handler(ctx, _handle_cost_summary), methods=["GET"]
        ),
        Route(
            "/api/project/status",
            endpoint=_make_handler(ctx, _handle_project_status),
            methods=["GET"],
        ),
        Route(
            "/api/model-tiers",
            endpoint=_make_handler(ctx, _handle_model_tiers),
            methods=["GET"],
        ),
        Route(
            "/api/trajectory/report",
            endpoint=_make_handler(ctx, _handle_trajectory_report),
            methods=["POST"],
        ),
        Route(
            "/api/trajectory/export",
            endpoint=_make_handler(ctx, _handle_trajectory_export),
            methods=["POST"],
        ),
        Route(
            "/api/trajectory/annotations",
            endpoint=_make_handler(ctx, _handle_trajectory_annotation_list),
            methods=["GET"],
        ),
        Route(
            "/api/trajectory/annotations",
            endpoint=_make_handler(ctx, _handle_trajectory_annotation_add),
            methods=["POST"],
        ),
        Route(
            "/api/memory/semantic/search",
            endpoint=_make_handler(ctx, _handle_memory_semantic_search),
            methods=["POST"],
        ),
        Route(
            "/api/memory/semantic/recommend",
            endpoint=_make_handler(ctx, _handle_memory_recommend),
            methods=["POST"],
        ),
        Route(
            "/api/memory/semantic/similar/{memory_id}",
            endpoint=_make_handler(ctx, _handle_memory_similar),
            methods=["GET"],
        ),
        Route(
            "/api/memory/feedback",
            endpoint=_make_handler(ctx, _handle_memory_feedback_submit),
            methods=["POST"],
        ),
        Route(
            "/api/memory/feedback/{memory_id}",
            endpoint=_make_handler(ctx, _handle_memory_feedback_get),
            methods=["GET"],
        ),
        Route(
            "/api/memory/feedback/summary",
            endpoint=_make_handler(ctx, _handle_memory_feedback_summary),
            methods=["GET"],
        ),
        Route(
            "/api/memory/usage/stats/{memory_id}",
            endpoint=_make_handler(ctx, _handle_memory_usage_stats),
            methods=["GET"],
        ),
        Route(
            "/api/memory/usage/record",
            endpoint=_make_handler(ctx, _handle_memory_usage_record),
            methods=["POST"],
        ),
        Route(
            "/api/memory/usage/low-quality",
            endpoint=_make_handler(ctx, _handle_memory_low_quality),
            methods=["GET"],
        ),
        Route(
            "/api/memory/usage/popular",
            endpoint=_make_handler(ctx, _handle_memory_popular),
            methods=["GET"],
        ),
        Route(
            "/api/audit/trends",
            endpoint=_make_handler(ctx, _handle_audit_trends),
            methods=["GET"],
        ),
        Route(
            "/api/chat",
            endpoint=_make_handler(ctx, _handle_chat),
            methods=["POST"],
        ),
        # Skills API 端点
        Route(
            "/api/employees/{employee_name}/skills",
            endpoint=_make_handler(ctx, _handle_skill_create),
            methods=["POST"],
        ),
        Route(
            "/api/employees/{employee_name}/skills",
            endpoint=_make_handler(ctx, _handle_skill_list),
            methods=["GET"],
        ),
        Route(
            "/api/employees/{employee_name}/skills/{skill_name}",
            endpoint=_make_handler(ctx, _handle_skill_get),
            methods=["GET"],
        ),
        Route(
            "/api/employees/{employee_name}/skills/{skill_name}",
            endpoint=_make_handler(ctx, _handle_skill_update),
            methods=["PUT"],
        ),
        Route(
            "/api/employees/{employee_name}/skills/{skill_name}",
            endpoint=_make_handler(ctx, _handle_skill_delete),
            methods=["DELETE"],
        ),
        Route(
            "/api/skills/check-triggers",
            endpoint=_make_handler(ctx, _handle_skills_check_triggers),
            methods=["POST"],
        ),
        Route(
            "/api/skills/execute",
            endpoint=_make_handler(ctx, _handle_skills_execute),
            methods=["POST"],
        ),
        Route(
            "/api/skills/stats",
            endpoint=_make_handler(ctx, _handle_skills_stats),
            methods=["GET"],
        ),
        Route(
            "/api/skills/trigger-history",
            endpoint=_make_handler(ctx, _handle_skills_trigger_history),
            methods=["GET"],
        ),
        # KV 存储端点
        Route(
            "/api/kv/",
            endpoint=_make_handler(ctx, _handle_kv_list),
            methods=["GET"],
        ),
        Route(
            "/api/kv/{key:path}",
            endpoint=_make_handler(ctx, _handle_kv_put),
            methods=["PUT"],
        ),
        Route(
            "/api/kv/{key:path}",
            endpoint=_make_handler(ctx, _handle_kv_get),
            methods=["GET"],
        ),
        # Pipeline / Discussion / Meeting / Decision / WorkLog / Permission 端点
        Route(
            "/api/pipelines",
            endpoint=_make_handler(ctx, _handle_pipeline_list),
            methods=["GET"],
        ),
        Route(
            "/api/discussions",
            endpoint=_make_handler(ctx, _handle_discussion_list),
            methods=["GET"],
        ),
        Route(
            "/api/discussions/{name}/plan",
            endpoint=_make_handler(ctx, _handle_discussion_plan),
            methods=["GET"],
        ),
        Route(
            "/api/discussions/{name}/prompt",
            endpoint=_make_handler(ctx, _handle_discussion_prompt),
            methods=["GET"],
        ),
        Route(
            "/api/meetings",
            endpoint=_make_handler(ctx, _handle_meeting_list),
            methods=["GET"],
        ),
        Route(
            "/api/meetings/{meeting_id}",
            endpoint=_make_handler(ctx, _handle_meeting_detail),
            methods=["GET"],
        ),
        Route(
            "/api/decisions/track",
            endpoint=_make_handler(ctx, _handle_decision_track),
            methods=["POST"],
        ),
        Route(
            "/api/decisions/{decision_id}/evaluate",
            endpoint=_make_handler(ctx, _handle_decision_evaluate),
            methods=["POST"],
        ),
        Route(
            "/api/evaluate/scan",
            endpoint=_make_handler(ctx, _handle_evaluate_scan),
            methods=["POST"],
        ),
        Route(
            "/api/work-log",
            endpoint=_make_handler(ctx, _handle_work_log),
            methods=["GET"],
        ),
        Route(
            "/api/permission-matrix",
            endpoint=_make_handler(ctx, _handle_permission_matrix),
            methods=["GET"],
        ),
        Route(
            "/api/permissions",
            endpoint=_make_handler(ctx, _handle_permission_list),
            methods=["GET"],
        ),
        Route(
            "/api/permissions/respond",
            endpoint=_make_handler(ctx, _handle_permission_respond),
            methods=["POST"],
        ),
        # 租户管理端点
        Route(
            "/api/tenants",
            endpoint=_make_handler(ctx, _handle_tenant_create),
            methods=["POST"],
        ),
        Route(
            "/api/tenants",
            endpoint=_make_handler(ctx, _handle_tenant_list),
            methods=["GET"],
        ),
        Route(
            "/api/tenants/{tenant_id}",
            endpoint=_make_handler(ctx, _handle_tenant_get),
            methods=["GET"],
        ),
        Route(
            "/api/tenants/{tenant_id}",
            endpoint=_make_handler(ctx, _handle_tenant_delete),
            methods=["DELETE"],
        ),
        Route(
            "/api/tenants/{tenant_id}",
            endpoint=_make_handler(ctx, _handle_tenant_update),
            methods=["PATCH"],
        ),
        # Wiki 文件管理端点
        Route(
            "/api/wiki/spaces",
            endpoint=_make_handler(ctx, _handle_wiki_spaces_list),
            methods=["GET"],
        ),
        Route(
            "/api/wiki/files/{file_id:int}",
            endpoint=_make_handler(ctx, _handle_wiki_file_delete),
            methods=["DELETE"],
        ),
        # 配置存储端点
        Route(
            "/api/souls",
            endpoint=_make_handler(ctx, _handle_soul_list),
            methods=["GET"],
        ),
        Route(
            "/api/souls/{employee_name}",
            endpoint=_make_handler(ctx, _handle_soul_get),
            methods=["GET"],
        ),
        Route(
            "/api/souls/{employee_name}",
            endpoint=_make_handler(ctx, _handle_soul_update),
            methods=["PUT"],
        ),
        # 灵魂进化端点
        Route(
            "/api/soul/review",
            endpoint=_make_handler(ctx, _handle_evolution_review),
            methods=["POST"],
        ),
        Route(
            "/api/soul/candidates",
            endpoint=_make_handler(ctx, _handle_evolution_candidates),
            methods=["GET"],
        ),
        Route(
            "/api/config/discussions",
            endpoint=_make_handler(ctx, _handle_discussion_list_config),
            methods=["GET"],
        ),
        Route(
            "/api/config/discussions",
            endpoint=_make_handler(ctx, _handle_discussion_create),
            methods=["POST"],
        ),
        Route(
            "/api/config/discussions/{name}",
            endpoint=_make_handler(ctx, _handle_discussion_get),
            methods=["GET"],
        ),
        Route(
            "/api/config/discussions/{name}",
            endpoint=_make_handler(ctx, _handle_discussion_update),
            methods=["PUT"],
        ),
        Route(
            "/api/config/pipelines",
            endpoint=_make_handler(ctx, _handle_pipeline_list_config),
            methods=["GET"],
        ),
        Route(
            "/api/config/pipelines",
            endpoint=_make_handler(ctx, _handle_pipeline_create_config),
            methods=["POST"],
        ),
        Route(
            "/api/config/pipelines/{name}",
            endpoint=_make_handler(ctx, _handle_pipeline_get_config),
            methods=["GET"],
        ),
        Route(
            "/api/config/pipelines/{name}",
            endpoint=_make_handler(ctx, _handle_pipeline_update_config),
            methods=["PUT"],
        ),
    ]

    async def on_startup():
        # ── 环境变量校验（不阻塞启动，仅记日志）──
        import os as _os

        if not _os.environ.get("ANTHROPIC_API_KEY") and not _os.environ.get("AIBERM_API_KEY"):
            logger.warning(
                "⚠ ANTHROPIC_API_KEY 未设置 — Agent 调用和定时任务将无法使用 Claude 模型"
            )
        if not _os.environ.get("FEISHU_CALENDAR_ID") and not (
            ctx.feishu_config and ctx.feishu_config.calendar_id
        ):
            logger.info("FEISHU_CALENDAR_ID 未设置且 feishu.yaml 无 calendar_id — 日历功能将不可用")

        # ── 初始化 memories 表（幂等 DDL）──
        try:
            from crew.memory_store_db import init_memory_tables

            init_memory_tables()
        except Exception as _init_err:
            logger.warning("memories 表初始化跳过: %s", _init_err)

        # ── 员工数据迁移到 employees 表（幂等）──
        try:
            from crew.config_store import migrate_employees_to_db

            _migrate_result = migrate_employees_to_db(project_dir=ctx.project_dir)
            if not _migrate_result.get("skipped"):
                logger.info(
                    "员工迁移: migrated=%d, skipped=%d, errors=%d",
                    _migrate_result.get("migrated", 0),
                    _migrate_result.get("skipped", 0),
                    len(_migrate_result.get("errors", [])),
                )
        except Exception as _migrate_err:
            logger.warning("员工迁移跳过: %s", _migrate_err)

        if scheduler:
            await scheduler.start()
        _task = asyncio.create_task(_resume_incomplete_pipelines(ctx))
        _startup_tasks.add(_task)
        _task.add_done_callback(_startup_tasks.discard)

        # ── 打卡通报定时任务（每天 10:30 北京时间）──
        if ctx.wecom_ctx:
            _checkin_task = asyncio.create_task(_run_checkin_cron(), name="cron-wecom-checkin")
            _startup_tasks.add(_checkin_task)
            _checkin_task.add_done_callback(_startup_tasks.discard)

        # ── 决策扫描定时任务（每天 10:00 北京时间）──
        _evaluate_task = asyncio.create_task(_run_evaluate_scan_cron(), name="cron-evaluate-scan")
        _startup_tasks.add(_evaluate_task)
        _evaluate_task.add_done_callback(_startup_tasks.discard)

        # ── MCP Gateway 启动 ──
        try:
            from crew.mcp_gateway import MCPGatewayManager
            from crew.mcp_gateway.audit import init_audit_table
            from crew.mcp_gateway.credentials import _init_credentials_table

            # 初始化审计和凭据表（幂等）
            try:
                init_audit_table()
                _init_credentials_table()
            except Exception as _tbl_err:
                logger.warning("MCP Gateway 表初始化跳过: %s", _tbl_err)

            _mcp_mgr = MCPGatewayManager(project_dir=project_dir)
            ctx.mcp_gateway = _mcp_mgr

            async def _start_mcp_gateway():
                try:
                    tool_count = await _mcp_mgr.start()
                    if tool_count > 0:
                        _mcp_mgr.inject_tools()
                        logger.info("MCP Gateway: %d 个外部工具已注入", tool_count)
                    else:
                        logger.info("MCP Gateway: 无外部工具需要注入")
                except Exception as _mcp_err:
                    logger.warning("MCP Gateway 启动异常（非致命）: %s", _mcp_err)

            _mcp_task = asyncio.create_task(_start_mcp_gateway(), name="mcp-gateway-startup")
            _startup_tasks.add(_mcp_task)
            _mcp_task.add_done_callback(_startup_tasks.discard)
        except ImportError:
            logger.debug("MCP Gateway 模块未安装，跳过")

    async def on_shutdown():
        if scheduler:
            await scheduler.stop()
        # 关闭 MCP Gateway
        if ctx.mcp_gateway:
            try:
                await ctx.mcp_gateway.stop()
            except Exception as _mcp_err:
                logger.warning("MCP Gateway 关闭异常: %s", _mcp_err)
        # 关闭企微 httpx 连接池
        from crew.wecom import close_wecom_client

        await close_wecom_client()

    app = Starlette(routes=routes, on_startup=[on_startup], on_shutdown=[on_shutdown])

    # 静态文件挂载（头像等）
    avatars_dir = (project_dir or Path(".")) / "static" / "avatars"
    if avatars_dir.is_dir():
        from starlette.staticfiles import StaticFiles

        app.mount("/static/avatars", StaticFiles(directory=str(avatars_dir)), name="avatars")

    from crew.auth import RequestSizeLimitMiddleware

    app.add_middleware(RequestSizeLimitMiddleware)

    if token:
        from crew.auth import RateLimitMiddleware
        from crew.tenant import MultiTenantAuthMiddleware, ensure_admin_tenant

        ensure_admin_tenant(token)

        skip_paths = [
            "/health",
            "/webhook/github",
            "/feishu/event",
            "/wecom/event",
            "/api/team/agents",  # 公开端点：官网+蚁聚匿名调用，不可删除（PR#35/36/41 三次教训）
            "/static",
        ] + [f"/feishu/event/{bot_id}" for bot_id in ctx.feishu_bots]
        # 共享缓存 dict：middleware 和 handlers 共用同一实例
        _shared_tenant_cache: dict = {}
        ctx.tenant_auth_cache = _shared_tenant_cache
        app.add_middleware(
            MultiTenantAuthMiddleware,
            admin_token=token,
            skip_paths=skip_paths,
            shared_cache=_shared_tenant_cache,
        )
        app.add_middleware(
            RateLimitMiddleware,
            skip_paths=["/health", "/metrics", "/webhook/github", "/api/permissions"],
        )

    if cors_origins:
        from starlette.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
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

    # 确保 crew.* 模块的日志输出到 stderr（被 systemd/journalctl 采集）
    _crew_logger = logging.getLogger("crew")
    if not _crew_logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(name)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
            )
        )
        _crew_logger.addHandler(_handler)
    _crew_logger.setLevel(logging.INFO)

    from crew.webhook_config import load_webhook_config

    config = load_webhook_config(project_dir)

    cron_config = None
    if enable_cron:
        from crew.cron_config import load_cron_config

        cron_config = load_cron_config(project_dir)

    feishu_bots = None
    try:
        from crew.feishu import load_feishu_configs

        feishu_bots = load_feishu_configs(project_dir) or None
    except Exception:
        pass

    _wecom_config = None
    try:
        from crew.wecom import load_wecom_config

        _wecom_config = load_wecom_config(project_dir)
        if not (_wecom_config.corp_id and _wecom_config.secret):
            _wecom_config = None
    except Exception:
        pass

    app = create_webhook_app(
        project_dir=project_dir,
        token=token,
        config=config,
        cron_config=cron_config,
        cors_origins=cors_origins,
        feishu_bots=feishu_bots,
        wecom_config=_wecom_config,
    )

    logger.info("启动 Webhook 服务器: %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
