"""工具定义桥接 — crew employee.tools → LLM JSON Schema + sandbox 映射."""

from __future__ import annotations

from typing import Any


# crew tool name → sandbox tool name
CREW_TO_SANDBOX: dict[str, str] = {
    "file_read": "file_read",
    "file_write": "file_write",
    "bash": "shell",
    "git": "git",
    "grep": "search",
    "glob": "search",
}

# 终止工具（agent loop 看到这些表示任务完成）
FINISH_TOOLS = {"submit", "finish"}

# LLM tool schemas（Anthropic 格式，OpenAI 自动转换）
_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "file_read": {
        "name": "file_read",
        "description": "读取文件内容。返回文件文本。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "start_line": {
                    "type": "integer",
                    "description": "起始行号 (0 表示从头)",
                    "default": 0,
                },
                "end_line": {
                    "type": "integer",
                    "description": "结束行号 (0 表示读到末尾)",
                    "default": 0,
                },
            },
            "required": ["path"],
        },
    },
    "file_write": {
        "name": "file_write",
        "description": "写入文件内容。自动创建父目录。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "文件内容"},
            },
            "required": ["path", "content"],
        },
    },
    "bash": {
        "name": "bash",
        "description": "在沙箱中执行 Shell 命令（bash -c）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell 命令"},
                "timeout": {
                    "type": "integer",
                    "description": "超时秒数",
                    "default": 300,
                },
            },
            "required": ["command"],
        },
    },
    "git": {
        "name": "git",
        "description": "执行 Git 操作（diff, log, status 等）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "subcommand": {
                    "type": "string",
                    "description": "Git 子命令 (diff, log, status, show 等)",
                },
                "args": {
                    "type": "string",
                    "description": "额外参数",
                    "default": "",
                },
            },
            "required": ["subcommand"],
        },
    },
    "grep": {
        "name": "grep",
        "description": "搜索代码内容（正则表达式匹配）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "搜索模式（正则）"},
                "path": {
                    "type": "string",
                    "description": "搜索路径",
                    "default": ".",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "文件过滤 (如 '*.py')",
                    "default": "",
                },
            },
            "required": ["pattern"],
        },
    },
    "glob": {
        "name": "glob",
        "description": "按文件名模式查找文件。",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "文件名模式 (如 '*.py')"},
                "path": {
                    "type": "string",
                    "description": "搜索路径",
                    "default": ".",
                },
            },
            "required": ["pattern"],
        },
    },
    "submit": {
        "name": "submit",
        "description": "提交最终结果，结束任务。用结论或最终输出作为 result 参数。",
        "input_schema": {
            "type": "object",
            "properties": {
                "result": {"type": "string", "description": "最终结果/结论"},
            },
            "required": ["result"],
        },
    },
    "delegate": {
        "name": "delegate",
        "description": "委派任务给另一位 AI 同事执行。同事将独立完成任务并返回结果。",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_name": {
                    "type": "string",
                    "description": "目标员工名称（如 code-reviewer、doc-writer）",
                },
                "task": {
                    "type": "string",
                    "description": "委派的具体任务描述",
                },
            },
            "required": ["employee_name", "task"],
        },
    },
}


def employee_tools_to_schemas(tools: list[str]) -> list[dict[str, Any]]:
    """将 employee 的 tools 列表转为 LLM tool schemas.

    始终包含 submit 工具（agent 需要它来结束任务）。

    Args:
        tools: employee.tools 列表，如 ["file_read", "bash", "git"]

    Returns:
        LLM tool schemas（Anthropic 格式，executor 会自动转为 OpenAI 格式）。
    """
    schemas = []
    seen = set()
    for tool_name in tools:
        if tool_name in _TOOL_SCHEMAS and tool_name not in seen:
            schemas.append(_TOOL_SCHEMAS[tool_name])
            seen.add(tool_name)
    # 始终包含 submit
    if "submit" not in seen:
        schemas.append(_TOOL_SCHEMAS["submit"])
    return schemas


def map_tool_call(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """将 LLM 返回的工具调用映射为 sandbox 格式.

    Args:
        tool_name: LLM 返回的工具名 (如 "bash")
        params: LLM 返回的参数

    Returns:
        {"tool": sandbox_tool_name, "params": mapped_params}
    """
    sandbox_name = CREW_TO_SANDBOX.get(tool_name, tool_name)

    # 特殊映射：bash → shell (参数名不同)
    if tool_name == "bash" and sandbox_name == "shell":
        return {"tool": "shell", "params": params}

    # glob → search (使用 file_pattern)
    if tool_name == "glob" and sandbox_name == "search":
        return {
            "tool": "search",
            "params": {
                "pattern": "",
                "path": params.get("path", "."),
                "file_pattern": params.get("pattern", ""),
            },
        }

    return {"tool": sandbox_name, "params": params}


def is_finish_tool(tool_name: str) -> bool:
    """判断是否为终止工具."""
    return tool_name in FINISH_TOOLS
