"""Webhook 工具函数包 — 按领域分组的 91 个工具."""

from __future__ import annotations

from typing import Any


def get_all_tool_handlers() -> dict[str, Any]:
    """聚合所有子模块的 HANDLERS 字典."""
    # 核心模块（必须加载）
    from crew.webhook_tools import core, orchestration

    handlers: dict[str, Any] = {}
    handlers.update(core.HANDLERS)
    handlers.update(orchestration.HANDLERS)

    # 可选插件模块
    for module_name in ("feishu", "external", "engineering", "github"):
        try:
            mod = __import__(f"crew.webhook_tools.{module_name}", fromlist=["HANDLERS"])
            handlers.update(mod.HANDLERS)
        except ImportError:
            pass

    return handlers
