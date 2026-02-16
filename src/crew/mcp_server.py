"""Crew MCP Server — Model Context Protocol 服务."""

import json
import logging
import time as _time
from pathlib import Path

logger = logging.getLogger(__name__)

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
from crew.exceptions import EmployeeNotFoundError
from crew.log import WorkLogger
from crew.pipeline import arun_pipeline, discover_pipelines, load_pipeline, run_pipeline, validate_pipeline


def _get_version() -> str:
    """读取包版本."""
    try:
        from importlib.metadata import version
        return version("knowlyr-crew")
    except Exception:
        return "unknown"


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
                description="执行流水线 — 支持 prompt-only 模式和 execute 模式（自动调用 LLM 串联执行）",
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
                        "execute": {
                            "type": "boolean",
                            "description": "执行模式 — 自动调用 LLM 串联执行（默认 false）",
                            "default": False,
                        },
                        "model": {
                            "type": "string",
                            "description": "LLM 模型标识符（execute 模式使用）",
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
                        "ttl_days": {
                            "type": "integer",
                            "description": "生存期天数 (0=永不过期，默认 0)",
                            "default": 0,
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "语义标签列表（可选）",
                        },
                        "shared": {
                            "type": "boolean",
                            "description": "是否加入共享记忆池（默认 false）",
                            "default": False,
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
            Tool(
                name="crew_feedback",
                description="向 knowlyr-id 提交员工工作反馈（人工评分+评语），用于 RLHF 闭环",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "integer",
                            "description": "Agent ID（knowlyr-id 用户 ID）",
                        },
                        "task_type": {
                            "type": "string",
                            "description": "任务类型（如 daily_check, code_review）",
                        },
                        "task_output": {
                            "type": "string",
                            "description": "员工的输出内容",
                        },
                        "human_score": {
                            "type": "number",
                            "description": "人工评分（0-100）",
                        },
                        "human_feedback": {
                            "type": "string",
                            "description": "人工评语（可选）",
                        },
                    },
                    "required": ["agent_id", "task_type", "task_output", "human_score"],
                },
            ),
            Tool(
                name="crew_status",
                description="查询 AI 员工在 knowlyr-id 的在线状态和基本信息",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "integer",
                            "description": "查询指定 Agent（可选，不传则列出所有）",
                        },
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """调用工具."""
        logger.info("tool_call: %s", name)
        try:
            return await _handle_tool(name, arguments)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logger.exception("tool_call_error: %s", name)
            return [TextContent(type="text", text=f"内部错误: {name}")]

    async def _handle_tool(name: str, arguments: dict) -> list[TextContent]:
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
            data = emp.model_dump(mode="json", exclude={
                "source_path", "api_key", "fallback_api_key", "fallback_base_url",
            })
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
            if arg_project_dir:
                p = Path(arg_project_dir).resolve()
                if not p.is_relative_to(_project_dir):
                    return [TextContent(type="text", text="路径不在项目目录范围内")]
            info = detect_project(Path(arg_project_dir) if arg_project_dir else _project_dir)
            data = info.model_dump(mode="json")
            data["display_label"] = info.display_label
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        elif name == "list_pipelines":
            pipelines = discover_pipelines(project_dir=_project_dir)
            def _step_summary(s):
                if hasattr(s, "employee"):
                    return s.employee
                if hasattr(s, "parallel"):
                    return [sub.employee for sub in s.parallel]
                if hasattr(s, "condition"):
                    return {"condition": [sub.employee for sub in s.condition.then]}
                if hasattr(s, "loop"):
                    return {"loop": [sub.employee for sub in s.loop.steps]}
                return "unknown"

            data = []
            for pname, ppath in pipelines.items():
                pl = load_pipeline(ppath)
                data.append({
                    "name": pname,
                    "description": pl.description,
                    "steps": [_step_summary(s) for s in pl.steps],
                    "path": str(ppath),
                })
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        elif name == "run_pipeline":
            pl_name = arguments["name"]
            pl_args = arguments.get("args", {})
            agent_id = arguments.get("agent_id")
            smart_context = arguments.get("smart_context", True)
            execute = arguments.get("execute", False)
            pl_model = arguments.get("model")

            # 查找流水线
            pl_path = Path(pl_name)
            if pl_path.is_absolute() and not pl_path.resolve().is_relative_to(_project_dir):
                return [TextContent(type="text", text="路径不在项目目录范围内")]
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

            # execute 模式需要 API key（自动从环境变量解析）
            api_key = None
            if execute:
                from crew.providers import detect_provider, resolve_api_key
                eff_model = pl_model or "claude-sonnet-4-20250514"
                try:
                    _prov = detect_provider(eff_model)
                    api_key = resolve_api_key(_prov)
                except ValueError as e:
                    return [TextContent(type="text", text=f"错误: {e}")]

            result = await arun_pipeline(
                pipeline,
                initial_args=pl_args,
                agent_id=agent_id,
                smart_context=smart_context,
                project_dir=_project_dir,
                execute=execute,
                api_key=api_key,
                model=pl_model,
            )
            return [TextContent(type="text", text=result.model_dump_json(indent=2))]

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
                if d_path.is_absolute() and not d_path.resolve().is_relative_to(_project_dir):
                    return [TextContent(type="text", text="路径不在项目目录范围内")]
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
                ttl_days=arguments.get("ttl_days", 0),
                tags=arguments.get("tags"),
                shared=arguments.get("shared", False),
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

        elif name == "crew_feedback":
            from crew.id_client import alog_work

            result = await alog_work(
                agent_id=arguments["agent_id"],
                task_type=arguments["task_type"],
                task_output=arguments["task_output"],
                human_score=arguments["human_score"],
                human_feedback=arguments.get("human_feedback", ""),
            )
            if result:
                return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
            return [TextContent(type="text", text="提交失败 — 请检查 knowlyr-id 连接和 AGENT_API_TOKEN 配置")]

        elif name == "crew_status":
            agent_id = arguments.get("agent_id")
            if agent_id is not None:
                from crew.id_client import afetch_agent_identity
                identity = await afetch_agent_identity(agent_id)
                if identity is None:
                    return [TextContent(type="text", text=f"未找到 Agent: {agent_id}")]
                data = {
                    "agent_id": identity.agent_id,
                    "display_name": identity.display_name,
                    "status": identity.status,
                    "model": identity.model,
                    "memory_length": len(identity.memory) if identity.memory else 0,
                }
                return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]
            else:
                from crew.id_client import alist_agents
                agents = await alist_agents()
                if agents is None:
                    return [TextContent(type="text", text="查询失败 — 请检查 knowlyr-id 连接")]
                return [TextContent(type="text", text=json.dumps(agents, ensure_ascii=False, indent=2))]

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
            raise EmployeeNotFoundError(name)

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
            raise EmployeeNotFoundError(emp_name)

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
    """启动 MCP 服务器（stdio 传输）."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-crew[mcp]")

    server = create_server(project_dir=project_dir)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


async def serve_sse(
    project_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    api_token: str | None = None,
):
    """启动 MCP 服务器（SSE 传输 — 兼容 Claude Desktop / Cursor）."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-crew[mcp]")

    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route
    import uvicorn

    server = create_server(project_dir=project_dir)
    sse = SseServerTransport("/messages/")
    _start = _time.monotonic()

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send,
        ) as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options(),
            )

    async def health(request):
        emp_count = len(discover_employees(project_dir=project_dir).employees)
        return JSONResponse({
            "status": "ok",
            "version": _get_version(),
            "employees": emp_count,
            "uptime_seconds": round(_time.monotonic() - _start),
        })

    async def metrics(request):
        from crew.metrics import get_collector
        return JSONResponse(get_collector().snapshot())

    _heartbeat_mgr = None

    async def sse_lifespan(app):
        nonlocal _heartbeat_mgr
        try:
            from crew.id_client import HeartbeatManager

            _heartbeat_mgr = HeartbeatManager(interval=60.0)
            await _heartbeat_mgr.start()
        except ImportError:
            pass
        yield
        if _heartbeat_mgr:
            await _heartbeat_mgr.stop()

    app = Starlette(
        routes=[
            Route("/health", endpoint=health),
            Route("/metrics", endpoint=metrics),
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
        lifespan=sse_lifespan,
    )
    if api_token:
        from crew.auth import BearerTokenMiddleware
        app.add_middleware(BearerTokenMiddleware, token=api_token)

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()


async def serve_http(
    project_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    api_token: str | None = None,
):
    """启动 MCP 服务器（Streamable HTTP 传输 — MCP 最新规范）."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-crew[mcp]")

    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route
    import uvicorn

    server = create_server(project_dir=project_dir)
    session_manager = StreamableHTTPSessionManager(app=server)
    _start = _time.monotonic()

    async def handle_mcp(scope, receive, send):
        await session_manager.handle_request(scope, receive, send)

    async def health(request):
        emp_count = len(discover_employees(project_dir=project_dir).employees)
        return JSONResponse({
            "status": "ok",
            "version": _get_version(),
            "employees": emp_count,
            "uptime_seconds": round(_time.monotonic() - _start),
        })

    async def metrics(request):
        from crew.metrics import get_collector
        return JSONResponse(get_collector().snapshot())

    async def lifespan(app):
        _hb_mgr = None
        try:
            from crew.id_client import HeartbeatManager

            _hb_mgr = HeartbeatManager(interval=60.0)
            await _hb_mgr.start()
        except ImportError:
            pass
        async with session_manager.run():
            yield
        if _hb_mgr:
            await _hb_mgr.stop()

    app = Starlette(
        routes=[
            Route("/health", endpoint=health),
            Route("/metrics", endpoint=metrics),
            Mount("/mcp", app=handle_mcp),
        ],
        lifespan=lifespan,
    )
    if api_token:
        from crew.auth import BearerTokenMiddleware
        app.add_middleware(BearerTokenMiddleware, token=api_token)

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()


def main():
    """CLI 入口: knowlyr-crew mcp."""
    import asyncio
    import os

    transport = os.environ.get("KNOWLYR_CREW_TRANSPORT", "stdio")
    project_dir = os.environ.get("KNOWLYR_CREW_PROJECT_DIR")
    host = os.environ.get("KNOWLYR_CREW_HOST", "127.0.0.1")
    port = int(os.environ.get("KNOWLYR_CREW_PORT", "8000"))
    api_token = os.environ.get("KNOWLYR_CREW_API_TOKEN")
    pd = Path(project_dir) if project_dir else None

    if transport == "sse":
        asyncio.run(serve_sse(pd, host, port, api_token))
    elif transport == "http":
        asyncio.run(serve_http(pd, host, port, api_token))
    else:
        asyncio.run(serve(pd))
