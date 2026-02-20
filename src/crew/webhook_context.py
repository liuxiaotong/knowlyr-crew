"""Webhook 共享上下文 — _AppContext 和常量."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crew.task_registry import TaskRegistry
    from crew.webhook_config import WebhookConfig

# ── 共享常量 ──

_MAX_TOOL_ROUNDS = 10

_ID_API_BASE = os.environ.get("KNOWLYR_ID_API", "https://id.knowlyr.com")
_ID_API_TOKEN = os.environ.get("AGENT_API_TOKEN", "")
_GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
_GITHUB_API_BASE = "https://api.github.com"
_GITHUB_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

_NOTION_API_KEY = os.environ.get("NOTION_API_KEY", "")
_NOTION_API_BASE = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"

_EMPLOYEE_UPDATABLE_FIELDS = {"model", "temperature", "max_tokens"}


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
        self.heartbeat_mgr = None  # HeartbeatManager, set by create_webhook_app
        self.feishu_config = None  # FeishuConfig, set by create_webhook_app
        self.feishu_token_mgr = None  # FeishuTokenManager, set by create_webhook_app
        self.feishu_dedup = None  # EventDeduplicator, set by create_webhook_app
        self.feishu_chat_store = None  # FeishuChatStore, set by create_webhook_app
