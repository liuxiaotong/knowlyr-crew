"""Crew MCP Server — Model Context Protocol 服务."""

import json
from pathlib import Path

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.server.lowlevel.helper_types import ReadResourceContents
    from mcp.types import (
        GetPromptResult,
        Prompt,
        PromptArgument,
        PromptMessage,
        Resource,
        TextContent,
        Tool,
    )

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from crew.context_detector import detect_project
from crew.discovery import discover_employees
from crew.engine import CrewEngine
from crew.log import WorkLogger
from crew.pipeline import discover_pipelines, load_pipeline, run_pipeline, validate_pipeline


def create_server(project_dir: Path | None = None) -> "Server":
    """创建 MCP 服务器实例."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-crew[mcp]")

    _project_dir = project_dir  # captured in closure

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
                        "agent_id": {
                            "type": "integer",
                            "description": "绑定的 knowlyr-id Agent ID（可选）",
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
            Tool(
                name="detect_project",
                description="检测当前项目类型、框架、包管理器等信息",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_dir": {
                            "type": "string",
                            "description": "项目目录路径（默认当前目录）",
                        },
                    },
                },
            ),
            Tool(
                name="list_pipelines",
                description="列出所有可用的流水线",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="run_pipeline",
                description="执行流水线，按顺序运行多个员工并生成 prompt 序列",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "流水线名称或 YAML 文件路径",
                        },
                        "args": {
                            "type": "object",
                            "description": "传递给流水线的参数（key-value）",
                            "additionalProperties": {"type": "string"},
                        },
                        "agent_id": {
                            "type": "integer",
                            "description": "绑定的 knowlyr-id Agent ID（可选）",
                        },
                        "smart_context": {
                            "type": "boolean",
                            "description": "自动检测项目类型（默认 true）",
                            "default": True,
                        },
                    },
                    "required": ["name"],
                },
            ),
            Tool(
                name="list_discussions",
                description="列出所有可用的讨论会",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="run_discussion",
                description="生成讨论会 prompt — 支持预定义 YAML 或即席讨论（employees+topic）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "讨论会名称或 YAML 文件路径（与 employees+topic 二选一）",
                        },
                        "employees": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "即席讨论的员工列表（与 name 二选一）",
                        },
                        "topic": {
                            "type": "string",
                            "description": "即席讨论的议题（与 employees 搭配使用）",
                        },
                        "goal": {
                            "type": "string",
                            "description": "讨论目标（可选）",
                        },
                        "rounds": {
                            "type": "integer",
                            "description": "讨论轮次（默认 2，即席讨论时使用）",
                        },
                        "round_template": {
                            "type": "string",
                            "description": "轮次模板 (standard, brainstorm-to-decision, adversarial)",
                        },
                        "args": {
                            "type": "object",
                            "description": "传递给讨论会的参数（key-value）",
                            "additionalProperties": {"type": "string"},
                        },
                        "agent_id": {
                            "type": "integer",
                            "description": "绑定的 knowlyr-id Agent ID（可选）",
                        },
                        "smart_context": {
                            "type": "boolean",
                            "description": "自动检测项目类型（默认 true）",
                            "default": True,
                        },
                        "orchestrated": {
                            "type": "boolean",
                            "description": "编排模式：每个参会者独立推理（默认 false）",
                            "default": False,
                        },
                    },
                },
            ),
            Tool(
                name="add_memory",
                description="为员工添加一条持久化记忆",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "employee": {
                            "type": "string",
                            "description": "员工名称",
                        },
                        "category": {
                            "type": "string",
                            "enum": ["decision", "estimate", "finding", "correction"],
                            "description": "记忆类别",
                        },
                        "content": {
                            "type": "string",
                            "description": "记忆内容",
                        },
                        "source_session": {
                            "type": "string",
                            "description": "来源 session ID（可选）",
                        },
                    },
                    "required": ["employee", "category", "content"],
                },
            ),
            Tool(
                name="query_memory",
                description="查询员工的持久化记忆",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "employee": {
                            "type": "string",
                            "description": "员工名称",
                        },
                        "category": {
                            "type": "string",
                            "enum": ["decision", "estimate", "finding", "correction"],
                            "description": "按类别过滤（可选）",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "最大返回条数（默认 20）",
                            "default": 20,
                        },
                    },
                    "required": ["employee"],
                },
            ),
            Tool(
                name="track_decision",
                description="记录一个待评估的决策（来自会议或日常工作）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "employee": {
                            "type": "string",
                            "description": "提出决策的员工名称",
                        },
                        "category": {
                            "type": "string",
                            "enum": ["estimate", "recommendation", "commitment"],
                            "description": "决策类别",
                        },
                        "content": {
                            "type": "string",
                            "description": "决策内容",
                        },
                        "expected_outcome": {
                            "type": "string",
                            "description": "预期结果（可选）",
                        },
                        "meeting_id": {
                            "type": "string",
                            "description": "来源会议 ID（可选）",
                        },
                    },
                    "required": ["employee", "category", "content"],
                },
            ),
            Tool(
                name="evaluate_decision",
                description="评估一个决策 — 记录实际结果并将经验写入员工记忆",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "decision_id": {
                            "type": "string",
                            "description": "决策 ID",
                        },
                        "actual_outcome": {
                            "type": "string",
                            "description": "实际结果",
                        },
                        "evaluation": {
                            "type": "string",
                            "description": "评估结论（可选，为空则自动生成）",
                        },
                    },
                    "required": ["decision_id", "actual_outcome"],
                },
            ),
            Tool(
                name="list_meeting_history",
                description="查看讨论会历史记录",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "返回条数（默认 20）",
                            "default": 20,
                        },
                        "keyword": {
                            "type": "string",
                            "description": "按关键词过滤",
                        },
                    },
                },
            ),
            Tool(
                name="get_meeting_detail",
                description="获取某次讨论会的完整记录",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "meeting_id": {
                            "type": "string",
                            "description": "会议 ID",
                        },
                    },
                    "required": ["meeting_id"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """调用工具."""
        if name == "list_employees":
            result = discover_employees(project_dir=_project_dir)
            employees = list(result.employees.values())
            tag = arguments.get("tag")
            if tag:
                employees = [e for e in employees if tag in e.tags]
            data = [
                {
                    "name": e.name,
                    "display_name": e.effective_display_name,
                    "character_name": e.character_name,
                    "description": e.description,
                    "tags": e.tags,
                    "triggers": e.triggers,
                    "model": e.model,
                    "layer": e.source_layer,
                }
                for e in employees
            ]
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        elif name == "get_employee":
            emp_name = arguments["name"]
            result = discover_employees(project_dir=_project_dir)
            emp = result.get(emp_name)
            if emp is None:
                return [TextContent(type="text", text=f"未找到员工: {emp_name}")]
            data = emp.model_dump(mode="json", exclude={"source_path"})
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        elif name == "run_employee":
            emp_name = arguments["name"]
            emp_args = arguments.get("args", {})
            agent_id = arguments.get("agent_id")
            result = discover_employees(project_dir=_project_dir)
            emp = result.get(emp_name)
            if emp is None:
                return [TextContent(type="text", text=f"未找到员工: {emp_name}")]
            engine = CrewEngine(project_dir=_project_dir)
            errors = engine.validate_args(emp, args=emp_args)
            if errors:
                return [TextContent(type="text", text=f"参数错误: {'; '.join(errors)}")]

            # 获取 Agent 身份（可选）
            agent_identity = None
            if agent_id is not None:
                try:
                    from crew.id_client import afetch_agent_identity
                    agent_identity = await afetch_agent_identity(agent_id)
                except Exception:
                    pass

            # 智能上下文检测
            project_info = detect_project(_project_dir)

            prompt = engine.prompt(emp, args=emp_args, agent_identity=agent_identity, project_info=project_info)

            # 记录工作日志
            try:
                log = WorkLogger(project_dir=_project_dir)
                sid = log.create_session(emp.name, args=emp_args, agent_id=agent_id)
                log.add_entry(sid, "prompt_generated", f"{len(prompt)} chars")
            except Exception:
                pass  # 日志失败不影响主流程

            # 发送心跳（可选）
            if agent_id is not None:
                try:
                    from crew.id_client import asend_heartbeat
                    await asend_heartbeat(agent_id, detail=f"employee={emp.name}")
                except Exception:
                    pass

            return [TextContent(type="text", text=prompt)]

        elif name == "get_work_log":
            logger = WorkLogger(project_dir=_project_dir)
            emp_name = arguments.get("employee_name")
            limit = arguments.get("limit", 10)
            sessions = logger.list_sessions(employee_name=emp_name, limit=limit)
            return [TextContent(
                type="text",
                text=json.dumps(sessions, ensure_ascii=False, indent=2),
            )]

        elif name == "detect_project":
            arg_project_dir = arguments.get("project_dir")
            info = detect_project(Path(arg_project_dir) if arg_project_dir else _project_dir)
            data = info.model_dump(mode="json")
            data["display_label"] = info.display_label
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        elif name == "list_pipelines":
            pipelines = discover_pipelines(project_dir=_project_dir)
            data = []
            for pname, ppath in pipelines.items():
                pl = load_pipeline(ppath)
                data.append({
                    "name": pname,
                    "description": pl.description,
                    "steps": [s.employee for s in pl.steps],
                    "path": str(ppath),
                })
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        elif name == "run_pipeline":
            pl_name = arguments["name"]
            pl_args = arguments.get("args", {})
            agent_id = arguments.get("agent_id")
            smart_context = arguments.get("smart_context", True)

            # 查找流水线
            pl_path = Path(pl_name)
            if not pl_path.exists():
                pipelines = discover_pipelines(project_dir=_project_dir)
                if pl_name in pipelines:
                    pl_path = pipelines[pl_name]
                else:
                    return [TextContent(type="text", text=f"未找到流水线: {pl_name}")]

            pipeline = load_pipeline(pl_path)
            errors = validate_pipeline(pipeline, project_dir=_project_dir)
            if errors:
                return [TextContent(type="text", text=f"流水线校验失败: {'; '.join(errors)}")]

            outputs = run_pipeline(
                pipeline,
                initial_args=pl_args,
                agent_id=agent_id,
                smart_context=smart_context,
                project_dir=_project_dir,
            )
            return [TextContent(type="text", text=json.dumps(outputs, ensure_ascii=False, indent=2))]

        elif name == "list_discussions":
            from crew.discussion import discover_discussions, load_discussion

            discussions = discover_discussions(project_dir=_project_dir)
            data = []
            for dname, dpath in discussions.items():
                try:
                    d = load_discussion(dpath)
                    rounds_count = d.rounds if isinstance(d.rounds, int) else len(d.rounds)
                    data.append({
                        "name": dname,
                        "description": d.description,
                        "participants": [p.employee for p in d.participants],
                        "rounds": rounds_count,
                        "path": str(dpath),
                    })
                except Exception:
                    data.append({"name": dname, "error": "解析失败", "path": str(dpath)})
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        elif name == "run_discussion":
            from crew.discussion import (
                create_adhoc_discussion,
                discover_discussions,
                load_discussion,
                render_discussion,
                render_discussion_plan,
                validate_discussion,
            )

            d_args = arguments.get("args", {})
            agent_id = arguments.get("agent_id")
            smart_context = arguments.get("smart_context", True)
            is_orchestrated = arguments.get("orchestrated", False)

            employees_list = arguments.get("employees")
            adhoc_topic = arguments.get("topic")

            if employees_list and adhoc_topic:
                # 即席讨论模式
                discussion = create_adhoc_discussion(
                    employees=employees_list,
                    topic=adhoc_topic,
                    goal=arguments.get("goal", ""),
                    rounds=arguments.get("rounds", 2),
                    round_template=arguments.get("round_template"),
                )
            elif "name" in arguments:
                d_name = arguments["name"]
                # 查找讨论会
                d_path = Path(d_name)
                if not d_path.exists():
                    discussions = discover_discussions(project_dir=_project_dir)
                    if d_name in discussions:
                        d_path = discussions[d_name]
                    else:
                        return [TextContent(type="text", text=f"未找到讨论会: {d_name}")]
                discussion = load_discussion(d_path)
            else:
                return [TextContent(type="text", text="请提供 name 或 employees+topic")]

            errors = validate_discussion(discussion, project_dir=_project_dir)
            if errors:
                return [TextContent(type="text", text=f"讨论会校验失败: {'; '.join(errors)}")]

            if is_orchestrated:
                plan = render_discussion_plan(
                    discussion,
                    initial_args=d_args,
                    agent_id=agent_id,
                    smart_context=smart_context,
                    project_dir=_project_dir,
                )
                return [TextContent(
                    type="text",
                    text=plan.model_dump_json(indent=2),
                )]
            else:
                prompt = render_discussion(
                    discussion,
                    initial_args=d_args,
                    agent_id=agent_id,
                    smart_context=smart_context,
                    project_dir=_project_dir,
                )
                return [TextContent(type="text", text=prompt)]

        elif name == "add_memory":
            from crew.memory import MemoryStore
            store = MemoryStore(project_dir=_project_dir)
            entry = store.add(
                employee=arguments["employee"],
                category=arguments["category"],
                content=arguments["content"],
                source_session=arguments.get("source_session", ""),
            )
            return [TextContent(
                type="text",
                text=json.dumps(entry.model_dump(), ensure_ascii=False, indent=2),
            )]

        elif name == "query_memory":
            from crew.memory import MemoryStore
            store = MemoryStore(project_dir=_project_dir)
            entries = store.query(
                employee=arguments["employee"],
                category=arguments.get("category"),
                limit=arguments.get("limit", 20),
            )
            data = [e.model_dump() for e in entries]
            return [TextContent(
                type="text",
                text=json.dumps(data, ensure_ascii=False, indent=2),
            )]

        elif name == "track_decision":
            from crew.evaluation import EvaluationEngine
            engine = EvaluationEngine(project_dir=_project_dir)
            decision = engine.track(
                employee=arguments["employee"],
                category=arguments["category"],
                content=arguments["content"],
                expected_outcome=arguments.get("expected_outcome", ""),
                meeting_id=arguments.get("meeting_id", ""),
            )
            return [TextContent(
                type="text",
                text=json.dumps(decision.model_dump(), ensure_ascii=False, indent=2),
            )]

        elif name == "evaluate_decision":
            from crew.evaluation import EvaluationEngine
            engine = EvaluationEngine(project_dir=_project_dir)
            decision = engine.evaluate(
                decision_id=arguments["decision_id"],
                actual_outcome=arguments["actual_outcome"],
                evaluation=arguments.get("evaluation", ""),
            )
            if decision is None:
                return [TextContent(type="text", text=f"未找到决策: {arguments['decision_id']}")]
            return [TextContent(
                type="text",
                text=json.dumps(decision.model_dump(), ensure_ascii=False, indent=2),
            )]

        elif name == "list_meeting_history":
            from crew.meeting_log import MeetingLogger

            logger = MeetingLogger(project_dir=_project_dir)
            records = logger.list(
                limit=arguments.get("limit", 20),
                keyword=arguments.get("keyword"),
            )
            data = [r.model_dump() for r in records]
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        elif name == "get_meeting_detail":
            from crew.meeting_log import MeetingLogger

            logger = MeetingLogger(project_dir=_project_dir)
            result = logger.get(arguments["meeting_id"])
            if result is None:
                return [TextContent(type="text", text=f"未找到会议: {arguments['meeting_id']}")]
            record, content = result
            data = {**record.model_dump(), "content": content}
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        return [TextContent(type="text", text=f"未知工具: {name}")]

    # ── MCP Prompts: 每个员工 = 一个可调用的 prompt ──

    @server.list_prompts()
    async def list_prompts() -> list[Prompt]:
        """列出所有员工作为 MCP Prompts."""
        result = discover_employees(project_dir=_project_dir)
        prompts = []
        for emp in result.employees.values():
            arguments = [
                PromptArgument(
                    name=a.name,
                    description=a.description,
                    required=a.required,
                )
                for a in emp.args
            ]
            prompts.append(Prompt(
                name=emp.name,
                title=emp.effective_display_name,
                description=emp.description,
                arguments=arguments or None,
            ))
        return prompts

    @server.get_prompt()
    async def get_prompt(
        name: str, arguments: dict[str, str] | None,
    ) -> GetPromptResult:
        """获取渲染后的 prompt."""
        result = discover_employees(project_dir=_project_dir)
        emp = result.get(name)
        if emp is None:
            raise ValueError(f"未找到: {name}")

        engine = CrewEngine(project_dir=_project_dir)
        args = arguments or {}
        errors = engine.validate_args(emp, args=args)
        if errors:
            raise ValueError(f"参数错误: {'; '.join(errors)}")

        rendered = engine.prompt(emp, args=args)
        return GetPromptResult(
            description=emp.description,
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=rendered),
                ),
            ],
        )

    # ── MCP Resources: 员工定义的原始 Markdown ──

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        """列出所有员工定义作为可读资源."""
        result = discover_employees(project_dir=_project_dir)
        return [
            Resource(
                uri=f"crew://employee/{emp.name}",
                name=emp.name,
                title=emp.effective_display_name,
                description=emp.description,
                mimeType="text/markdown",
            )
            for emp in result.employees.values()
        ]

    @server.read_resource()
    async def read_resource(uri) -> list[ReadResourceContents]:
        """读取员工定义的原始 Markdown 内容."""
        uri_str = str(uri)
        prefix = "crew://employee/"
        if not uri_str.startswith(prefix):
            raise ValueError(f"未知资源: {uri_str}")

        emp_name = uri_str[len(prefix):]
        result = discover_employees(project_dir=_project_dir)
        emp = result.get(emp_name)
        if emp is None:
            raise ValueError(f"未找到: {emp_name}")

        if emp.source_path and emp.source_path.exists():
            if emp.source_path.is_dir():
                # 目录格式：返回拼接后的完整内容（即 body）
                content = emp.body
            else:
                content = emp.source_path.read_text(encoding="utf-8")
        else:
            content = emp.body

        return [ReadResourceContents(content=content, mime_type="text/markdown")]

    return server


async def serve(project_dir: Path | None = None):
    """启动 MCP 服务器."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-crew[mcp]")

    server = create_server(project_dir=project_dir)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


def main():
    """CLI 入口: knowlyr-crew mcp."""
    import asyncio
    import os
    project_dir = os.environ.get("KNOWLYR_CREW_PROJECT_DIR")
    asyncio.run(serve(Path(project_dir) if project_dir else None))
