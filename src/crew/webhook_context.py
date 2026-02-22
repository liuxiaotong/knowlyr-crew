"""Webhook 共享上下文 — _AppContext 和常量."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crew.task_registry import TaskRegistry
    from crew.webhook_config import WebhookConfig

# ── 共享常量 ──

_MAX_TOOL_ROUNDS = 10

# 过渡期：人类用户数据仍在 knowlyr-id（id 管人类，crew 管 AI）
_ID_API_BASE = os.environ.get("KNOWLYR_ID_API", "https://id.knowlyr.com")
_ID_API_TOKEN = os.environ.get("AGENT_API_TOKEN", "")
_GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
_GITHUB_API_BASE = "https://api.github.com"
_GITHUB_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

_ANTGATHER_API_URL = os.environ.get("ANTGATHER_API_URL", "")
_ANTGATHER_API_TOKEN = os.environ.get("ANTGATHER_API_TOKEN", "")

_NOTION_API_KEY = os.environ.get("NOTION_API_KEY", "")
_NOTION_API_BASE = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"

_EMPLOYEE_UPDATABLE_FIELDS = {"model", "model_tier", "temperature", "max_tokens", "agent_status"}


class FeishuBotContext:
    """单个飞书 Bot 的运行时上下文."""

    __slots__ = ("config", "token_mgr", "dedup")

    def __init__(self, config: Any, token_mgr: Any, dedup: Any):
        self.config = config  # FeishuBotConfig
        self.token_mgr = token_mgr  # FeishuTokenManager
        self.dedup = dedup  # EventDeduplicator


class _AppContext:
    """应用上下文，共享于所有 handler."""

    def __init__(
        self,
        project_dir: Path | None,
        config: WebhookConfig,
        registry: TaskRegistry,
    ):
        self.project_dir = project_dir
        self.config = config
        self.registry = registry
        self.scheduler = None  # CronScheduler, set by create_webhook_app
        # 飞书 — primary bot（工具调用用这个，向后兼容）
        self.feishu_config = None  # FeishuConfig, set by create_webhook_app
        self.feishu_token_mgr = None  # FeishuTokenManager, set by create_webhook_app
        self.feishu_dedup = None  # EventDeduplicator, set by create_webhook_app
        self.feishu_chat_store = None  # FeishuChatStore, set by create_webhook_app
        # 飞书 — 多 bot
        self.feishu_bots: dict[str, FeishuBotContext] = {}  # bot_id -> context
