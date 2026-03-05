"""MCP Gateway 注册表 — 工具命名空间管理 + Schema/Handler 注入.

负责:
1. 工具名校验（正则 + 碰撞检测）
2. 命名空间 ``mcp__{server}__{tool}`` 格式管理
3. 向 ``_TOOL_SCHEMAS`` 和 ``_TOOL_HANDLERS`` 注入外部工具
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)

# ── 命名规则 ──

# 合法的 MCP server 名: 仅 a-z 0-9 和 -（不超过 32 字符）
_SERVER_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,30}[a-z0-9]$")

# 合法的 MCP tool 名: 仅 a-z 0-9 _ -（不超过 64 字符）
_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}[a-z0-9]$")

# 命名空间格式
NAMESPACE_SEP = "__"
NAMESPACE_PREFIX = "mcp"


def make_namespaced_name(server_name: str, tool_name: str) -> str:
    """生成命名空间工具名: ``mcp__{server}__{tool}``."""
    return f"{NAMESPACE_PREFIX}{NAMESPACE_SEP}{server_name}{NAMESPACE_SEP}{tool_name}"


def parse_namespaced_name(namespaced: str) -> tuple[str, str] | None:
    """解析命名空间工具名，返回 (server_name, tool_name) 或 None."""
    prefix = f"{NAMESPACE_PREFIX}{NAMESPACE_SEP}"
    if not namespaced.startswith(prefix):
        return None
    rest = namespaced[len(prefix) :]
    sep_idx = rest.find(NAMESPACE_SEP)
    if sep_idx < 0:
        return None
    return rest[:sep_idx], rest[sep_idx + len(NAMESPACE_SEP) :]


def validate_server_name(name: str) -> bool:
    """校验 server 名是否合法."""
    return bool(_SERVER_NAME_RE.match(name))


def validate_tool_name(name: str) -> bool:
    """校验 tool 名是否合法."""
    if "__" in name:
        return False
    return bool(_TOOL_NAME_RE.match(name))


class MCPToolRegistry:
    """MCP 外部工具注册表.

    维护已注册的外部工具映射，并提供 schema/handler 注入功能。
    """

    def __init__(self) -> None:
        # namespaced_name -> {server, tool, schema, handler}
        self._tools: dict[str, dict[str, Any]] = {}
        # 已注册的 server 列表
        self._servers: set[str] = set()

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    @property
    def server_count(self) -> int:
        return len(self._servers)

    def registered_tools(self) -> list[str]:
        """返回所有已注册的命名空间工具名."""
        return list(self._tools.keys())

    def register_tool(
        self,
        server_name: str,
        tool_name: str,
        schema: dict[str, Any],
        handler: Callable[..., Coroutine[Any, Any, str]],
        *,
        existing_tools: set[str] | None = None,
    ) -> str | None:
        """注册一个外部 MCP 工具.

        Args:
            server_name: MCP server 标识符
            tool_name: 原始工具名
            schema: Anthropic 格式的工具 schema
            handler: 异步工具处理函数
            existing_tools: 现有内置工具名集合（碰撞检测用）

        Returns:
            namespaced_name 或 None（校验失败时）
        """
        # 1. 名称校验
        if not validate_server_name(server_name):
            logger.warning("MCP server 名不合法: '%s'", server_name)
            return None
        if not validate_tool_name(tool_name):
            logger.warning("MCP tool 名不合法: '%s' (server=%s)", tool_name, server_name)
            return None

        # 2. 生成命名空间名
        ns_name = make_namespaced_name(server_name, tool_name)

        # 3. 碰撞检测 — 不能与内置工具重名
        if existing_tools and ns_name in existing_tools:
            logger.warning("MCP 工具名碰撞（内置）: '%s'", ns_name)
            return None

        # 4. 重复注册检测
        if ns_name in self._tools:
            logger.info("MCP 工具重复注册（覆盖）: '%s'", ns_name)

        # 5. 修正 schema 名为命名空间名
        ns_schema = {**schema, "name": ns_name}

        self._tools[ns_name] = {
            "server": server_name,
            "tool": tool_name,
            "schema": ns_schema,
            "handler": handler,
        }
        self._servers.add(server_name)

        logger.debug("MCP 工具已注册: %s", ns_name)
        return ns_name

    def unregister_server(self, server_name: str) -> list[str]:
        """注销一个 server 的所有工具，返回被移除的工具名列表."""
        removed = []
        to_remove = [
            ns_name for ns_name, info in self._tools.items() if info["server"] == server_name
        ]
        for ns_name in to_remove:
            del self._tools[ns_name]
            removed.append(ns_name)

        if not any(info["server"] == server_name for info in self._tools.values()):
            self._servers.discard(server_name)

        logger.info("MCP server '%s' 已注销: %d 个工具", server_name, len(removed))
        return removed

    def get_handler(self, namespaced_name: str) -> Callable | None:
        """获取工具处理函数."""
        info = self._tools.get(namespaced_name)
        return info["handler"] if info else None

    def get_schema(self, namespaced_name: str) -> dict[str, Any] | None:
        """获取工具 schema."""
        info = self._tools.get(namespaced_name)
        return info["schema"] if info else None

    def get_all_schemas(self) -> dict[str, dict[str, Any]]:
        """返回所有已注册工具的 schema（name -> schema dict）."""
        return {ns_name: info["schema"] for ns_name, info in self._tools.items()}

    def get_all_handlers(self) -> dict[str, Callable]:
        """返回所有已注册工具的 handler（name -> handler）."""
        return {ns_name: info["handler"] for ns_name, info in self._tools.items()}

    def inject_into(
        self,
        tool_schemas: dict[str, dict[str, Any]],
        tool_handlers: dict[str, Any],
    ) -> int:
        """将所有已注册的 MCP 工具注入到现有的 schemas/handlers 字典中.

        Args:
            tool_schemas: ``tool_schema._TOOL_SCHEMAS``
            tool_handlers: ``webhook_executor._TOOL_HANDLERS``

        Returns:
            注入的工具数量
        """
        count = 0
        for ns_name, info in self._tools.items():
            tool_schemas[ns_name] = info["schema"]
            tool_handlers[ns_name] = info["handler"]
            count += 1

        if count > 0:
            logger.info("MCP 工具已注入: %d 个工具 (来自 %d 个 server)", count, len(self._servers))

        return count
