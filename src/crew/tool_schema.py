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
    "query_stats": {
        "name": "query_stats",
        "description": "查询公司实时业务数据。返回用户增长、消息活跃、AI团队状态、财务等数据。问业务问题前先调这个。",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "send_message": {
        "name": "send_message",
        "description": "给用户或AI同事发一条消息。",
        "input_schema": {
            "type": "object",
            "properties": {
                "recipient_id": {
                    "type": "integer",
                    "description": "收件人用户ID。Kai=1",
                },
                "content": {
                    "type": "string",
                    "description": "消息内容",
                },
            },
            "required": ["recipient_id", "content"],
        },
    },
    "list_agents": {
        "name": "list_agents",
        "description": "查看所有AI同事的列表和当前状态。",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "web_search": {
        "name": "web_search",
        "description": "搜索互联网获取实时信息。适用于查行业动态、竞品信息、技术概念等。返回搜索结果摘要。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                },
                "max_results": {
                    "type": "integer",
                    "description": "返回结果数量上限",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    "create_note": {
        "name": "create_note",
        "description": "保存备忘或笔记。存储在项目目录中，长期保留。适合记录决策、待办、会议要点。",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "笔记标题（也用作文件名）",
                },
                "content": {
                    "type": "string",
                    "description": "笔记内容（markdown 格式）",
                },
                "tags": {
                    "type": "string",
                    "description": "标签，逗号分隔（如 decision,urgent）",
                    "default": "",
                },
            },
            "required": ["title", "content"],
        },
    },
    "lookup_user": {
        "name": "lookup_user",
        "description": "按姓名查用户详情（活跃度、消息数、注册时间等）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "用户昵称（支持模糊搜索）",
                },
            },
            "required": ["name"],
        },
    },
    "query_agent_work": {
        "name": "query_agent_work",
        "description": "查某个AI同事最近的工作记录（任务数、成功率、评分、具体任务列表）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "AI同事昵称",
                },
                "days": {
                    "type": "integer",
                    "description": "查最近几天，默认3",
                    "default": 3,
                },
            },
            "required": ["name"],
        },
    },
    "read_notes": {
        "name": "read_notes",
        "description": "查看之前保存的备忘和笔记。可按关键词筛选。",
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "搜索关键词（可选）",
                    "default": "",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回数量上限",
                    "default": 10,
                },
            },
            "required": [],
        },
    },
    "read_messages": {
        "name": "read_messages",
        "description": "查看 Kai 的未读消息。返回未读数量和每个发件人的最新消息预览。",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "get_system_health": {
        "name": "get_system_health",
        "description": "查看服务器健康状态（磁盘、内存、CPU、服务状态、错误日志）。",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "mark_read": {
        "name": "mark_read",
        "description": "把 Kai 的未读消息标为已读。可以标某个人发的，也可以全部标已读。",
        "input_schema": {
            "type": "object",
            "properties": {
                "sender_name": {
                    "type": "string",
                    "description": "发件人姓名（标该人的消息已读）",
                },
                "all": {
                    "type": "boolean",
                    "description": "true=全部标已读",
                    "default": False,
                },
            },
            "required": [],
        },
    },
    "update_agent": {
        "name": "update_agent",
        "description": "管理AI同事：暂停/恢复运行、更新记忆。先用 list_agents 查到 agent_id。",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "integer",
                    "description": "AI同事的用户ID",
                },
                "status": {
                    "type": "string",
                    "description": "active=运行 / paused=暂停 / disabled=禁用",
                },
                "memory": {
                    "type": "string",
                    "description": "更新AI同事的记忆内容",
                },
            },
            "required": ["agent_id"],
        },
    },
    "read_feishu_calendar": {
        "name": "read_feishu_calendar",
        "description": "查看飞书日历日程。Kai 问今天有什么会、这周安排、日程时调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "起始日期 YYYY-MM-DD，默认今天",
                },
                "days": {
                    "type": "integer",
                    "description": "查几天（1=今天，7=本周），默认 1",
                    "default": 1,
                },
            },
            "required": [],
        },
    },
    "delete_feishu_event": {
        "name": "delete_feishu_event",
        "description": "取消/删除飞书日历日程。需要先用 read_feishu_calendar 查到 event_id。",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "要删除的日程 event_id",
                },
            },
            "required": ["event_id"],
        },
    },
    "create_feishu_task": {
        "name": "create_feishu_task",
        "description": "在飞书创建待办任务。Kai 说建个待办、记个任务时调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "任务标题",
                },
                "due": {
                    "type": "string",
                    "description": "截止日期 YYYY-MM-DD（可选）",
                },
                "description": {
                    "type": "string",
                    "description": "任务描述（可选）",
                    "default": "",
                },
            },
            "required": ["summary"],
        },
    },
    "list_feishu_tasks": {
        "name": "list_feishu_tasks",
        "description": "查看飞书待办任务列表。Kai 问还有什么没做的、看看待办时调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "返回条数，默认 20",
                    "default": 20,
                },
            },
            "required": [],
        },
    },
    "complete_feishu_task": {
        "name": "complete_feishu_task",
        "description": "完成飞书待办任务。需要先用 list_feishu_tasks 查到 task_id。",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "要完成的任务 task_id",
                },
            },
            "required": ["task_id"],
        },
    },
    "feishu_chat_history": {
        "name": "feishu_chat_history",
        "description": "读取飞书群/会话的最近消息。需要 chat_id，不知道就先用 list_feishu_groups 查。",
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "string",
                    "description": "飞书群聊 ID（形如 oc_xxxxx）",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回消息条数，默认 10",
                    "default": 10,
                },
            },
            "required": ["chat_id"],
        },
    },
    "weather": {
        "name": "weather",
        "description": "查国内城市天气。支持主要城市，返回当前温度和未来几天预报。",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名（如：上海、北京、杭州）",
                },
                "days": {
                    "type": "integer",
                    "description": "预报天数（1-7），默认 3",
                    "default": 3,
                },
            },
            "required": ["city"],
        },
    },
    "create_feishu_event": {
        "name": "create_feishu_event",
        "description": "在飞书日历创建日程。Kai 说安排日程、设提醒、记个时间时调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "日程标题（如：查返回上海的机票酒店）",
                },
                "date": {
                    "type": "string",
                    "description": "日期，YYYY-MM-DD 格式（如 2025-02-15）",
                },
                "start_hour": {
                    "type": "integer",
                    "description": "开始小时（0-23）",
                },
                "start_minute": {
                    "type": "integer",
                    "description": "开始分钟（0-59），默认 0",
                    "default": 0,
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "持续时长（分钟），默认 60",
                    "default": 60,
                },
                "description": {
                    "type": "string",
                    "description": "日程描述（可选）",
                    "default": "",
                },
            },
            "required": ["summary", "date", "start_hour"],
        },
    },
    # ── 飞书文档工具 ──
    "search_feishu_docs": {
        "name": "search_feishu_docs",
        "description": "搜索飞书云文档和知识库。输入关键词，返回匹配的文档列表（标题、链接、摘要）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "count": {"type": "integer", "description": "返回数量上限", "default": 10},
            },
            "required": ["query"],
        },
    },
    "read_feishu_doc": {
        "name": "read_feishu_doc",
        "description": "读取飞书文档内容。传入 document_id，返回文档纯文本。",
        "input_schema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "飞书文档 ID（从搜索结果或 URL 获取）"},
            },
            "required": ["document_id"],
        },
    },
    "create_feishu_doc": {
        "name": "create_feishu_doc",
        "description": "在飞书创建新文档。传入标题和正文，返回文档链接。",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "文档标题"},
                "content": {"type": "string", "description": "文档正文"},
                "folder_token": {
                    "type": "string",
                    "description": "目标文件夹 token（可选）",
                    "default": "",
                },
            },
            "required": ["title", "content"],
        },
    },
    "send_feishu_group": {
        "name": "send_feishu_group",
        "description": "发消息到飞书群。指定群聊 ID 和消息内容。不知道 chat_id 时先调 list_feishu_groups 查。",
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": {"type": "string", "description": "飞书群聊 ID（形如 oc_xxxxx）"},
                "text": {"type": "string", "description": "消息内容"},
            },
            "required": ["chat_id", "text"],
        },
    },
    "list_feishu_groups": {
        "name": "list_feishu_groups",
        "description": "列出机器人加入的所有飞书群，返回群名和 chat_id。发群消息前用这个查 chat_id。",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    # ── GitHub 工具 ──
    "github_prs": {
        "name": "github_prs",
        "description": "查看 GitHub 仓库的 Pull Requests 列表。",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "仓库（格式: owner/repo）"},
                "state": {"type": "string", "description": "状态: open/closed/all", "default": "open"},
                "limit": {"type": "integer", "description": "返回数量上限", "default": 10},
            },
            "required": ["repo"],
        },
    },
    "github_issues": {
        "name": "github_issues",
        "description": "查看 GitHub 仓库的 Issues 列表。",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "仓库（格式: owner/repo）"},
                "state": {"type": "string", "description": "状态: open/closed/all", "default": "open"},
                "labels": {"type": "string", "description": "标签过滤（逗号分隔）", "default": ""},
                "limit": {"type": "integer", "description": "返回数量上限", "default": 10},
            },
            "required": ["repo"],
        },
    },
    "github_repo_activity": {
        "name": "github_repo_activity",
        "description": "查看 GitHub 仓库最近活动：提交记录、贡献者。",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "仓库（格式: owner/repo）"},
                "days": {"type": "integer", "description": "查最近几天", "default": 7},
            },
            "required": ["repo"],
        },
    },
    # ── Notion 工具 ──
    "notion_search": {
        "name": "notion_search",
        "description": "搜索 Notion 页面和数据库。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "limit": {"type": "integer", "description": "返回数量上限", "default": 10},
            },
            "required": ["query"],
        },
    },
    "notion_read": {
        "name": "notion_read",
        "description": "读取 Notion 页面内容。传入页面 ID，返回正文文本。",
        "input_schema": {
            "type": "object",
            "properties": {
                "page_id": {"type": "string", "description": "Notion 页面 ID"},
            },
            "required": ["page_id"],
        },
    },
    "notion_create": {
        "name": "notion_create",
        "description": "在 Notion 创建新页面。",
        "input_schema": {
            "type": "object",
            "properties": {
                "parent_id": {"type": "string", "description": "父页面 ID"},
                "title": {"type": "string", "description": "页面标题"},
                "content": {"type": "string", "description": "页面内容"},
            },
            "required": ["parent_id", "title", "content"],
        },
    },
    "send_feishu_dm": {
        "name": "send_feishu_dm",
        "description": "给飞书用户发私聊消息。需要对方 open_id，不知道就先用 feishu_group_members 查。",
        "input_schema": {
            "type": "object",
            "properties": {
                "open_id": {
                    "type": "string",
                    "description": "接收人的 open_id（形如 ou_xxxxx）",
                },
                "text": {
                    "type": "string",
                    "description": "消息内容",
                },
            },
            "required": ["open_id", "text"],
        },
    },
    "feishu_group_members": {
        "name": "feishu_group_members",
        "description": "查看飞书群成员列表（姓名+open_id）。发私聊前用这个查 open_id。",
        "input_schema": {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "string",
                    "description": "飞书群聊 ID（形如 oc_xxxxx）",
                },
            },
            "required": ["chat_id"],
        },
    },
    # ── 金融工具 ──
    "exchange_rate": {
        "name": "exchange_rate",
        "description": "查汇率。默认美元兑人民币，支持所有主流货币。",
        "input_schema": {
            "type": "object",
            "properties": {
                "from": {
                    "type": "string",
                    "description": "基准货币代码（如 USD、EUR、CNY），默认 USD",
                    "default": "USD",
                },
                "to": {
                    "type": "string",
                    "description": "目标货币代码，逗号分隔（如 CNY,EUR,JPY），默认 CNY",
                    "default": "CNY",
                },
            },
            "required": [],
        },
    },
    "stock_price": {
        "name": "stock_price",
        "description": "查股价。支持A股（如 sh600519 茅台）和美股（如 gb_aapl 苹果）。纯数字默认A股，纯字母默认美股。",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "股票代码（如 sh600519、sz000001、gb_aapl、aapl、600519）",
                },
            },
            "required": ["symbol"],
        },
    },
    # ── 记忆工具 ──
    "add_memory": {
        "name": "add_memory",
        "description": "记录一条持久记忆。记忆会自动注入到未来的对话中。适合记录反思、发现、决策背景。",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "分类: finding(发现/反思) / decision(决策) / correction(纠正)",
                    "default": "finding",
                },
                "content": {
                    "type": "string",
                    "description": "记忆内容（用自己的话写）",
                },
            },
            "required": ["content"],
        },
    },
    # ── 信息采集工具 ──
    "read_url": {
        "name": "read_url",
        "description": "读取网页内容，自动提取正文。适合读文章、文档、公告。",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "网页 URL"},
            },
            "required": ["url"],
        },
    },
    "rss_read": {
        "name": "rss_read",
        "description": "读取 RSS/Atom 订阅源，返回最近条目的标题、链接和摘要。",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "RSS/Atom 订阅源 URL"},
                "limit": {"type": "integer", "description": "返回条目数量上限", "default": 10},
            },
            "required": ["url"],
        },
    },
}

# 需要 agent loop 的工具（区别于 sandbox 工具）
AGENT_TOOLS = {
    "query_stats", "send_message", "list_agents", "delegate",
    "web_search", "create_note", "lookup_user", "query_agent_work",
    "read_notes", "read_messages", "get_system_health",
    "mark_read", "update_agent",
    "read_feishu_calendar", "delete_feishu_event", "create_feishu_event",
    "create_feishu_task", "list_feishu_tasks", "complete_feishu_task",
    "feishu_chat_history", "weather",
    "send_feishu_dm", "feishu_group_members",
    "exchange_rate", "stock_price",
    # 记忆
    "add_memory",
    # 飞书文档
    "search_feishu_docs", "read_feishu_doc", "create_feishu_doc", "send_feishu_group", "list_feishu_groups",
    # GitHub
    "github_prs", "github_issues", "github_repo_activity",
    # Notion
    "notion_search", "notion_read", "notion_create",
    # 信息采集
    "read_url", "rss_read",
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
