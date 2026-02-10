"""Crew MCP Server — Model Context Protocol 服务."""

import json
from pathlib import Path

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from crew.discovery import discover_employees
from crew.engine import CrewEngine
from crew.log import WorkLogger


def create_server() -> "Server":
    """创建 MCP 服务器实例."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-crew[mcp]")

    server = Server("crew")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """列出可用的工具."""
        return [
            Tool(
                name="list_employees",
                description="列出所有可用的数字员工",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tag": {
                            "type": "string",
                            "description": "按标签过滤（可选）",
                        },
                    },
                },
            ),
            Tool(
                name="get_employee",
                description="获取数字员工的完整定义",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "员工名称或触发别名",
                        },
                    },
                    "required": ["name"],
                },
            ),
            Tool(
                name="run_employee",
                description="加载数字员工并生成可执行的 prompt",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "员工名称或触发别名",
                        },
                        "args": {
                            "type": "object",
                            "description": "传递给员工的参数（key-value）",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                    "required": ["name"],
                },
            ),
            Tool(
                name="get_work_log",
                description="查看数字员工的工作日志",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "employee_name": {
                            "type": "string",
                            "description": "按员工过滤（可选）",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "返回条数（默认 10）",
                            "default": 10,
                        },
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """调用工具."""
        if name == "list_employees":
            result = discover_employees()
            employees = list(result.employees.values())
            tag = arguments.get("tag")
            if tag:
                employees = [e for e in employees if tag in e.tags]
            data = [
                {
                    "name": e.name,
                    "display_name": e.effective_display_name,
                    "description": e.description,
                    "tags": e.tags,
                    "triggers": e.triggers,
                    "layer": e.source_layer,
                }
                for e in employees
            ]
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        elif name == "get_employee":
            emp_name = arguments["name"]
            result = discover_employees()
            emp = result.get(emp_name)
            if emp is None:
                return [TextContent(type="text", text=f"未找到员工: {emp_name}")]
            data = emp.model_dump(mode="json", exclude={"source_path"})
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        elif name == "run_employee":
            emp_name = arguments["name"]
            emp_args = arguments.get("args", {})
            result = discover_employees()
            emp = result.get(emp_name)
            if emp is None:
                return [TextContent(type="text", text=f"未找到员工: {emp_name}")]
            engine = CrewEngine()
            errors = engine.validate_args(emp, args=emp_args)
            if errors:
                return [TextContent(type="text", text=f"参数错误: {'; '.join(errors)}")]
            prompt = engine.prompt(emp, args=emp_args)
            return [TextContent(type="text", text=prompt)]

        elif name == "get_work_log":
            logger = WorkLogger()
            emp_name = arguments.get("employee_name")
            limit = arguments.get("limit", 10)
            sessions = logger.list_sessions(employee_name=emp_name, limit=limit)
            return [TextContent(
                type="text",
                text=json.dumps(sessions, ensure_ascii=False, indent=2),
            )]

        return [TextContent(type="text", text=f"未知工具: {name}")]

    return server


async def serve():
    """启动 MCP 服务器."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-crew[mcp]")

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


def main():
    """CLI 入口: knowlyr-crew mcp."""
    import asyncio
    asyncio.run(serve())
