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
    "delegate_async": {
        "name": "delegate_async",
        "description": "异步委派任务给同事，立即返回任务 ID，不等待完成。适合并行派多个人做事。",
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
    "check_task": {
        "name": "check_task",
        "description": "查询异步任务的状态和结果（包括委派任务和会议）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "任务 ID（由 delegate_async 或 organize_meeting 返回）",
                },
            },
            "required": ["task_id"],
        },
    },
    "list_tasks": {
        "name": "list_tasks",
        "description": "列出最近的异步任务，支持按状态或类型筛选。",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["pending", "running", "completed", "failed"],
                    "description": "按状态筛选（可选）",
                },
                "type": {
                    "type": "string",
                    "enum": ["employee", "meeting", "pipeline"],
                    "description": "按类型筛选（可选）",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回数量上限（默认 10）",
                    "default": 10,
                },
            },
        },
    },
    "organize_meeting": {
        "name": "organize_meeting",
        "description": (
            "组织多位同事的讨论会议，异步执行，立即返回会议 ID。"
            "各参会者并行讨论多轮后综合结论。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "employees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "参会员工名称列表（如 ['code-reviewer', 'test-engineer']）",
                },
                "topic": {
                    "type": "string",
                    "description": "会议议题",
                },
                "goal": {
                    "type": "string",
                    "description": "会议目标（可选）",
                },
                "rounds": {
                    "type": "integer",
                    "description": "讨论轮次（默认 2）",
                    "default": 2,
                },
            },
            "required": ["employees", "topic"],
        },
    },
    "check_meeting": {
        "name": "check_meeting",
        "description": "查询会议进展和结果（check_task 的会议专用别名）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "会议 ID（由 organize_meeting 返回）",
                },
            },
            "required": ["task_id"],
        },
    },
    "run_pipeline": {
        "name": "run_pipeline",
        "description": "执行预定义的多员工流水线（如安全审计、产品发布）。异步执行，返回任务 ID。",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "流水线名称（如 security-test、product-launch）",
                },
                "args": {
                    "type": "object",
                    "description": "流水线参数（如 {\"target\": \"auth.py\"}）",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["name"],
        },
    },
    "delegate_chain": {
        "name": "delegate_chain",
        "description": "按顺序委派多位同事，前一步的结果自动传给下一步。异步执行，返回任务 ID。",
        "input_schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "employee_name": {
                                "type": "string",
                                "description": "目标员工名称",
                            },
                            "task": {
                                "type": "string",
                                "description": "任务描述，可用 {prev} 引用上一步结果",
                            },
                        },
                        "required": ["employee_name", "task"],
                    },
                    "description": "按顺序执行的步骤列表",
                },
            },
            "required": ["steps"],
        },
    },
    "schedule_task": {
        "name": "schedule_task",
        "description": "创建定时任务（如每天早上发简报、每周五写周报）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "任务名称（唯一标识）",
                },
                "cron": {
                    "type": "string",
                    "description": "cron 表达式（如 '0 9 * * *' 表示每天 9 点）",
                },
                "employee_name": {
                    "type": "string",
                    "description": "执行的员工名称",
                },
                "task": {
                    "type": "string",
                    "description": "任务描述",
                },
            },
            "required": ["name", "cron", "employee_name", "task"],
        },
    },
    "list_schedules": {
        "name": "list_schedules",
        "description": "列出所有定时任务及下次执行时间。",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    "cancel_schedule": {
        "name": "cancel_schedule",
        "description": "取消一个定时任务。",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "要取消的定时任务名称",
                },
            },
            "required": ["name"],
        },
    },
    "agent_file_read": {
        "name": "agent_file_read",
        "description": "读取项目目录内的文件内容（只读，不能写入）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径（相对于项目根目录）",
                },
                "start_line": {
                    "type": "integer",
                    "description": "起始行号（可选，默认从头开始）",
                },
                "end_line": {
                    "type": "integer",
                    "description": "结束行号（可选，默认读到尾）",
                },
            },
            "required": ["path"],
        },
    },
    "agent_file_grep": {
        "name": "agent_file_grep",
        "description": "在项目目录内搜索文件内容（正则匹配）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "搜索模式（正则表达式）",
                },
                "path": {
                    "type": "string",
                    "description": "搜索目录（相对于项目根目录，默认整个项目）",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "文件名过滤（如 '*.py'）",
                },
            },
            "required": ["pattern"],
        },
    },
    "query_data": {
        "name": "query_data",
        "description": "查询细粒度业务数据（按指标、时间段、分组维度）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": ["users", "messages", "agents", "revenue", "tasks"],
                    "description": "查询指标",
                },
                "period": {
                    "type": "string",
                    "enum": ["today", "week", "month", "quarter"],
                    "description": "时间段（默认 week）",
                    "default": "week",
                },
                "group_by": {
                    "type": "string",
                    "enum": ["day", "week", "agent"],
                    "description": "分组维度（可选）",
                },
            },
            "required": ["metric"],
        },
    },
    "find_free_time": {
        "name": "find_free_time",
        "description": "查询多位飞书用户的共同空闲时间段。",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "飞书用户 open_id 列表",
                },
                "days": {
                    "type": "integer",
                    "description": "查询未来几天（默认 7）",
                    "default": 7,
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "需要的时长（分钟，默认 60）",
                    "default": 60,
                },
            },
            "required": ["user_ids"],
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
    "delete_feishu_task": {
        "name": "delete_feishu_task",
        "description": "删除飞书待办任务。需要先用 list_feishu_tasks 查到 task_id。",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "要删除的任务 task_id",
                },
            },
            "required": ["task_id"],
        },
    },
    "update_feishu_task": {
        "name": "update_feishu_task",
        "description": "更新飞书待办任务的标题、截止日期或描述。需要先用 list_feishu_tasks 查到 task_id。",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "要更新的任务 task_id",
                },
                "summary": {
                    "type": "string",
                    "description": "新标题（可选）",
                },
                "due": {
                    "type": "string",
                    "description": "新截止日期 YYYY-MM-DD（可选）",
                },
                "description": {
                    "type": "string",
                    "description": "新描述（可选）",
                },
            },
            "required": ["task_id"],
        },
    },
    "get_datetime": {
        "name": "get_datetime",
        "description": "获取当前准确日期时间（北京时间）。不确定现在几点、今天星期几时调用。",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    "calculate": {
        "name": "calculate",
        "description": "计算数学表达式。支持加减乘除、幂次、sqrt、log、三角函数等。适合精确计算场景。",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 100*1.15**12 或 sqrt(144)",
                },
            },
            "required": ["expression"],
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
    # ── 生活助手 ──
    "translate": {
        "name": "translate",
        "description": "中英互译。自动检测语言方向：中文→英文，英文→中文。也可手动指定。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要翻译的文本"},
                "from_lang": {
                    "type": "string",
                    "description": "源语言（auto=自动检测, zh=中文, en=英文, ja=日文）",
                    "default": "auto",
                },
                "to_lang": {
                    "type": "string",
                    "description": "目标语言（留空则自动选对向语言）",
                    "default": "",
                },
            },
            "required": ["text"],
        },
    },
    "countdown": {
        "name": "countdown",
        "description": "计算距离某个日期还有多少天。适合问倒计时、还有几天到XX。",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "目标日期，YYYY-MM-DD 格式"},
                "event": {
                    "type": "string",
                    "description": "事件名称（可选，如「融资截止」）",
                    "default": "",
                },
            },
            "required": ["date"],
        },
    },
    "trending": {
        "name": "trending",
        "description": "查看热搜榜。支持微博热搜和知乎热榜。",
        "input_schema": {
            "type": "object",
            "properties": {
                "platform": {
                    "type": "string",
                    "description": "平台：weibo（微博热搜）或 zhihu（知乎热榜）",
                    "default": "weibo",
                },
                "limit": {"type": "integer", "description": "返回条数", "default": 15},
            },
            "required": [],
        },
    },
    # ── 飞书表格 ──
    "read_feishu_sheet": {
        "name": "read_feishu_sheet",
        "description": "读取飞书表格数据。需要 spreadsheet_token（从 search_feishu_docs 或 URL 获取）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "spreadsheet_token": {
                    "type": "string",
                    "description": "飞书表格 token（从搜索结果或 URL 中获取）",
                },
                "sheet_id": {
                    "type": "string",
                    "description": "工作表 ID（可选，默认第一个 sheet）",
                    "default": "",
                },
                "range": {
                    "type": "string",
                    "description": "读取范围，如 A1:Z100（可选，默认前 100 行）",
                    "default": "",
                },
            },
            "required": ["spreadsheet_token"],
        },
    },
    "update_feishu_sheet": {
        "name": "update_feishu_sheet",
        "description": "写入飞书表格数据。需要 spreadsheet_token 和 range。",
        "input_schema": {
            "type": "object",
            "properties": {
                "spreadsheet_token": {
                    "type": "string",
                    "description": "飞书表格 token",
                },
                "sheet_id": {
                    "type": "string",
                    "description": "工作表 ID（可选，默认第一个 sheet）",
                    "default": "",
                },
                "range": {
                    "type": "string",
                    "description": "写入范围，如 A1:C3",
                },
                "values": {
                    "type": "string",
                    "description": "要写入的数据，JSON 二维数组，如 [[\"姓名\",\"部门\"],[\"张三\",\"产品\"]]",
                },
            },
            "required": ["spreadsheet_token", "range", "values"],
        },
    },
    # ── 飞书审批 ──
    "list_feishu_approvals": {
        "name": "list_feishu_approvals",
        "description": "查看飞书审批列表。可查待审、已批准、已拒绝的审批。",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "审批状态: PENDING（待审）/ APPROVED（已批准）/ REJECTED（已拒绝）/ ALL",
                    "default": "PENDING",
                },
                "limit": {"type": "integer", "description": "返回条数", "default": 10},
            },
            "required": [],
        },
    },
    # ── 编码 & 开发辅助 ──
    "base64_codec": {
        "name": "base64_codec",
        "description": "Base64 编码或解码文本。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要编码或解码的文本"},
                "decode": {
                    "type": "boolean",
                    "description": "true=解码，false=编码（默认编码）",
                    "default": False,
                },
            },
            "required": ["text"],
        },
    },
    "color_convert": {
        "name": "color_convert",
        "description": "颜色格式转换。支持 HEX、RGB、HSL 互转。",
        "input_schema": {
            "type": "object",
            "properties": {
                "color": {
                    "type": "string",
                    "description": "颜色值，如 #FF5733、rgb(255,87,51)、hsl(11,100%,60%)",
                },
            },
            "required": ["color"],
        },
    },
    "cron_explain": {
        "name": "cron_explain",
        "description": "解释 cron 表达式的含义，或根据描述生成 cron 表达式。",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "cron 表达式（如 0 9 * * 1-5）或自然语言描述（如「每天早上9点」）",
                },
            },
            "required": ["expression"],
        },
    },
    "regex_test": {
        "name": "regex_test",
        "description": "测试正则表达式是否匹配文本，返回匹配结果。",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "正则表达式"},
                "text": {"type": "string", "description": "要测试的文本"},
                "replace": {
                    "type": "string",
                    "description": "替换字符串（可选，不填则只匹配）",
                    "default": "",
                },
            },
            "required": ["pattern", "text"],
        },
    },
    "hash_gen": {
        "name": "hash_gen",
        "description": "计算文本的哈希值。支持 MD5、SHA1、SHA256。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要计算哈希的文本"},
                "algorithm": {
                    "type": "string",
                    "description": "算法：md5/sha1/sha256（默认 sha256）",
                    "default": "sha256",
                },
            },
            "required": ["text"],
        },
    },
    "url_codec": {
        "name": "url_codec",
        "description": "URL 编码或解码。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要编码或解码的文本/URL"},
                "decode": {
                    "type": "boolean",
                    "description": "true=解码，false=编码（默认编码）",
                    "default": False,
                },
            },
            "required": ["text"],
        },
    },
    # ── 文本 & 开发工具 ──
    "text_extract": {
        "name": "text_extract",
        "description": "从非结构化文本中提取关键信息：邮箱、手机号、URL、金额等。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要提取信息的文本"},
                "extract_type": {
                    "type": "string",
                    "description": "提取类型：email/phone/url/money/all（默认 all）",
                    "default": "all",
                },
            },
            "required": ["text"],
        },
    },
    "json_format": {
        "name": "json_format",
        "description": "格式化/压缩 JSON，或从文本中提取 JSON 片段。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "JSON 文本或包含 JSON 的文本"},
                "compact": {
                    "type": "boolean",
                    "description": "是否压缩（true=单行，false=美化缩进）",
                    "default": False,
                },
            },
            "required": ["text"],
        },
    },
    "password_gen": {
        "name": "password_gen",
        "description": "生成安全随机密码。",
        "input_schema": {
            "type": "object",
            "properties": {
                "length": {"type": "integer", "description": "密码长度（默认 16）", "default": 16},
                "count": {"type": "integer", "description": "生成几个（默认 3）", "default": 3},
                "no_symbols": {
                    "type": "boolean",
                    "description": "是否不含特殊字符（默认 false）",
                    "default": False,
                },
            },
            "required": [],
        },
    },
    "ip_lookup": {
        "name": "ip_lookup",
        "description": "查询 IP 地址的地理位置和运营商信息。",
        "input_schema": {
            "type": "object",
            "properties": {
                "ip": {"type": "string", "description": "IP 地址（留空查本机公网 IP）", "default": ""},
            },
            "required": [],
        },
    },
    "short_url": {
        "name": "short_url",
        "description": "生成短链接。",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要缩短的 URL"},
            },
            "required": ["url"],
        },
    },
    "word_count": {
        "name": "word_count",
        "description": "统计文本的字数、词数、行数等。适合检查文稿长度。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要统计的文本"},
            },
            "required": ["text"],
        },
    },
    # ── 实用工具 ──
    "unit_convert": {
        "name": "unit_convert",
        "description": "单位换算。支持长度、重量、温度、面积、体积、数据大小等。",
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "数值"},
                "from_unit": {"type": "string", "description": "原单位，如 km, lb, °F, GB"},
                "to_unit": {"type": "string", "description": "目标单位，如 mi, kg, °C, MB"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
    "random_pick": {
        "name": "random_pick",
        "description": "从多个选项中随机选一个，或掷骰子、生成随机数。帮 Kai 做选择。",
        "input_schema": {
            "type": "object",
            "properties": {
                "options": {
                    "type": "string",
                    "description": "用逗号分隔的选项列表，如「火锅,烤肉,日料」。留空则掷 1-6 骰子。",
                    "default": "",
                },
                "count": {"type": "integer", "description": "选几个（默认 1）", "default": 1},
            },
            "required": [],
        },
    },
    "holidays": {
        "name": "holidays",
        "description": "查询中国法定节假日和调休安排。适合问「下周放假吗」「五一怎么调休」。",
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "年份（默认今年）", "default": 0},
                "month": {"type": "integer", "description": "月份（可选，查某月安排）", "default": 0},
            },
            "required": [],
        },
    },
    "timestamp_convert": {
        "name": "timestamp_convert",
        "description": "Unix 时间戳和可读时间互转。支持秒级和毫秒级。",
        "input_schema": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "时间戳（如 1708012800）或日期时间（如 2026-02-16 10:00）",
                },
            },
            "required": ["input"],
        },
    },
    "create_feishu_spreadsheet": {
        "name": "create_feishu_spreadsheet",
        "description": "在飞书中创建一个新的表格。可指定标题和文件夹。",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "表格标题"},
                "folder_token": {
                    "type": "string",
                    "description": "文件夹 token（可选，默认根目录）",
                    "default": "",
                },
            },
            "required": ["title"],
        },
    },
    "feishu_contacts": {
        "name": "feishu_contacts",
        "description": "在飞书通讯录中搜索同事。可按姓名搜索。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词（姓名）"},
                "limit": {"type": "integer", "description": "返回条数", "default": 5},
            },
            "required": ["query"],
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
    # ── 飞书增强 ──
    "feishu_bitable": {
        "name": "feishu_bitable",
        "description": "读取飞书多维表格数据。需要 app_token 和 table_id。",
        "input_schema": {
            "type": "object",
            "properties": {
                "app_token": {"type": "string", "description": "多维表格 app_token"},
                "table_id": {"type": "string", "description": "数据表 table_id"},
                "filter": {
                    "type": "string",
                    "description": "过滤条件（可选，如字段名=值）",
                    "default": "",
                },
                "limit": {"type": "integer", "description": "返回条数", "default": 20},
            },
            "required": ["app_token", "table_id"],
        },
    },
    "feishu_wiki": {
        "name": "feishu_wiki",
        "description": "搜索飞书知识库。输入关键词，返回知识库中匹配的文档。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "limit": {"type": "integer", "description": "返回条数", "default": 10},
            },
            "required": ["query"],
        },
    },
    "approve_feishu": {
        "name": "approve_feishu",
        "description": "操作飞书审批：通过或拒绝。需要先用 list_feishu_approvals 查到 instance_id。",
        "input_schema": {
            "type": "object",
            "properties": {
                "instance_id": {"type": "string", "description": "审批实例 ID"},
                "action": {
                    "type": "string",
                    "description": "操作: approve（通过）/ reject（拒绝）",
                },
                "comment": {
                    "type": "string",
                    "description": "审批意见（可选）",
                    "default": "",
                },
            },
            "required": ["instance_id", "action"],
        },
    },
    # ── AI 能力 ──
    "summarize": {
        "name": "summarize",
        "description": "对长文本生成摘要。适合总结文章、会议记录、文档。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要摘要的文本"},
                "style": {
                    "type": "string",
                    "description": "摘要风格: bullet（要点）/ paragraph（段落）/ oneline（一句话）",
                    "default": "bullet",
                },
            },
            "required": ["text"],
        },
    },
    "sentiment": {
        "name": "sentiment",
        "description": "分析文本的情感倾向和语气。适合分析用户反馈、评论、消息。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要分析的文本"},
            },
            "required": ["text"],
        },
    },
    # ── 效率工具 ──
    "email_send": {
        "name": "email_send",
        "description": "发送电子邮件。指定收件人、主题和正文。",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "收件人邮箱地址"},
                "subject": {"type": "string", "description": "邮件主题"},
                "body": {"type": "string", "description": "邮件正文"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    "qrcode": {
        "name": "qrcode",
        "description": "生成二维码图片链接。输入文本或 URL，返回二维码图片地址。",
        "input_schema": {
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "要编码的文本或 URL"},
                "size": {"type": "integer", "description": "图片尺寸（像素）", "default": 300},
            },
            "required": ["data"],
        },
    },
    "diff_text": {
        "name": "diff_text",
        "description": "对比两段文本的差异。返回 diff 结果。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text1": {"type": "string", "description": "原始文本"},
                "text2": {"type": "string", "description": "修改后的文本"},
            },
            "required": ["text1", "text2"],
        },
    },
    # ── 数据查询 ──
    "whois": {
        "name": "whois",
        "description": "查询域名的 WHOIS 注册信息。",
        "input_schema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "域名（如 example.com）"},
            },
            "required": ["domain"],
        },
    },
    "dns_lookup": {
        "name": "dns_lookup",
        "description": "查询域名的 DNS 解析记录（A、AAAA）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "域名（如 example.com）"},
            },
            "required": ["domain"],
        },
    },
    "http_check": {
        "name": "http_check",
        "description": "检查网站是否可用。返回状态码和响应时间。",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要检查的 URL"},
            },
            "required": ["url"],
        },
    },
    # ── 生活服务 ──
    "express_track": {
        "name": "express_track",
        "description": "查快递物流信息。输入快递单号，自动识别快递公司。",
        "input_schema": {
            "type": "object",
            "properties": {
                "number": {"type": "string", "description": "快递单号"},
                "company": {
                    "type": "string",
                    "description": "快递公司代码（可选，自动识别）",
                    "default": "",
                },
            },
            "required": ["number"],
        },
    },
    "flight_info": {
        "name": "flight_info",
        "description": "查询航班信息。输入航班号，返回出发到达时间和状态。",
        "input_schema": {
            "type": "object",
            "properties": {
                "flight_no": {"type": "string", "description": "航班号（如 CA1234、MU5678）"},
                "date": {
                    "type": "string",
                    "description": "日期 YYYY-MM-DD（默认今天）",
                    "default": "",
                },
            },
            "required": ["flight_no"],
        },
    },
    "aqi": {
        "name": "aqi",
        "description": "查询城市空气质量指数（AQI）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名（如：上海、beijing）"},
            },
            "required": ["city"],
        },
    },
}

# 需要 agent loop 的工具（区别于 sandbox 工具）
AGENT_TOOLS = {
    "query_stats", "send_message", "list_agents", "delegate",
    "delegate_async", "check_task", "list_tasks", "organize_meeting", "check_meeting",
    "run_pipeline", "delegate_chain",
    "schedule_task", "list_schedules", "cancel_schedule",
    "agent_file_read", "agent_file_grep",
    "query_data", "find_free_time",
    "web_search", "create_note", "lookup_user", "query_agent_work",
    "read_notes", "read_messages", "get_system_health",
    "mark_read", "update_agent",
    "read_feishu_calendar", "delete_feishu_event", "create_feishu_event",
    "create_feishu_task", "list_feishu_tasks", "complete_feishu_task",
    "delete_feishu_task", "update_feishu_task",
    "feishu_chat_history", "weather",
    "get_datetime", "calculate",
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
    # 生活助手
    "translate", "countdown", "trending",
    # 飞书表格 & 审批
    "read_feishu_sheet", "update_feishu_sheet", "list_feishu_approvals",
    # 实用工具
    "unit_convert", "random_pick", "holidays", "timestamp_convert",
    "create_feishu_spreadsheet", "feishu_contacts",
    # 文本 & 开发
    "text_extract", "json_format", "password_gen", "ip_lookup", "short_url", "word_count",
    # 编码 & 开发辅助
    "base64_codec", "color_convert", "cron_explain", "regex_test", "hash_gen", "url_codec",
    # 飞书增强
    "feishu_bitable", "feishu_wiki", "approve_feishu",
    # AI 能力
    "summarize", "sentiment",
    # 效率工具
    "email_send", "qrcode", "diff_text",
    # 数据查询
    "whois", "dns_lookup", "http_check",
    # 生活服务
    "express_track", "flight_info", "aqi",
}

# 延迟加载工具 — 首轮只传名字列表，模型需要时调 load_tools 加载完整 schema
# 所有 agent 工具都延迟加载，首轮只有 load_tools + submit
DEFERRED_TOOLS = AGENT_TOOLS


def _make_load_tools_schema(available: set[str]) -> dict[str, Any]:
    """构建 load_tools 元工具 schema."""
    names_str = ", ".join(sorted(available))
    return {
        "name": "load_tools",
        "description": f"加载额外工具后才能调用。可用: {names_str}",
        "input_schema": {
            "type": "object",
            "properties": {
                "names": {
                    "type": "string",
                    "description": "要加载的工具名，逗号分隔",
                },
            },
            "required": ["names"],
        },
    }


def employee_tools_to_schemas(
    tools: list[str], *, defer: bool = True,
) -> tuple[list[dict[str, Any]], set[str]]:
    """将 employee 的 tools 列表转为 LLM tool schemas.

    Args:
        tools: employee.tools 列表，如 ["file_read", "bash", "git"]
        defer: 是否启用延迟加载（默认 True）

    Returns:
        (schemas, deferred_names) — schemas 是核心工具 + load_tools，
        deferred_names 是被延迟的工具名集合（空集表示无延迟）。
    """
    schemas: list[dict[str, Any]] = []
    seen: set[str] = set()
    deferred: set[str] = set()
    for tool_name in tools:
        if tool_name not in _TOOL_SCHEMAS or tool_name in seen:
            continue
        if defer and tool_name in DEFERRED_TOOLS:
            deferred.add(tool_name)
        else:
            schemas.append(_TOOL_SCHEMAS[tool_name])
            seen.add(tool_name)
    # 有延迟工具时加 load_tools 元工具
    if deferred:
        schemas.append(_make_load_tools_schema(deferred))
    # 始终包含 submit
    if "submit" not in seen:
        schemas.append(_TOOL_SCHEMAS["submit"])
    return schemas, deferred


def get_tool_schema(name: str) -> dict[str, Any] | None:
    """按名称获取单个工具 schema."""
    return _TOOL_SCHEMAS.get(name)


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
