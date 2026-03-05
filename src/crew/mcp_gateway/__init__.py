"""MCP Gateway — 通用 MCP Client 网关模块.

将外部 MCP Server 的工具以 ``mcp__{server}__{tool}`` 命名空间
注入到现有的 _TOOL_HANDLERS / _TOOL_SCHEMAS 体系中，
实现主流程零改动接入外部工具。
"""

from __future__ import annotations

from crew.mcp_gateway.manager import MCPGatewayManager

__all__ = ["MCPGatewayManager"]
