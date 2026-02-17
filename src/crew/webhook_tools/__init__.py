"""Webhook 工具函数包 — 按领域分组的 93 个工具."""

from __future__ import annotations

from typing import Any


def get_all_tool_handlers() -> dict[str, Any]:
    """聚合所有子模块的 HANDLERS 字典."""
    from crew.webhook_tools import data_query, engineering, external, feishu, github, orchestration

    handlers: dict[str, Any] = {}
    handlers.update(feishu.HANDLERS)
    handlers.update(external.HANDLERS)
    handlers.update(engineering.HANDLERS)
    handlers.update(data_query.HANDLERS)
    handlers.update(github.HANDLERS)
    handlers.update(orchestration.HANDLERS)
    return handlers
