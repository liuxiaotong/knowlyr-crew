"""HTTP 请求处理器 — 各 API 端点的业务逻辑."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 后台任务引用集合 — 防止 GC 提前回收 + 异常日志
_background_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]


def _task_done_callback(task: asyncio.Task) -> None:  # type: ignore[type-arg]
    """后台 task 完成回调：记录异常日志 + 从引用集合移除."""
    _background_tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("后台任务异常: %s", exc, exc_info=exc)


import re as _re

from crew.webhook_context import _EMPLOYEE_UPDATABLE_FIELDS, _AppContext


def _safe_int(value: str | None, default: int = 0) -> int:
    """安全转换为 int，转换失败时返回默认值."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _find_employee(result: Any, identifier: str) -> Any:
    """按 agent_id 或 name（字符串）查找员工."""
    # 先按 name 查找
    emp = result.get(identifier)
    if emp is not None:
        return emp
    # 再按 agent_id 查找（agent_id 现在是 "AI3050" 格式的字符串）
    for emp in result.employees.values():
        if emp.agent_id == identifier:
            return emp
    return None


def _ok_response(data: dict | None = None, status_code: int = 200) -> Any:
    """统一成功响应格式 — 后续逐步迁移各端点使用."""
    from starlette.responses import JSONResponse

    body: dict[str, Any] = {"ok": True}
    if data:
        body.update(data)
    return JSONResponse(body, status_code=status_code)


def _error_response(message: str, status_code: int = 400) -> Any:
    """统一错误响应格式 — 后续逐步迁移各端点使用."""
    from starlette.responses import JSONResponse

    return JSONResponse({"ok": False, "error": message}, status_code=status_code)


def _write_yaml_field(emp_dir: Path, updates: dict) -> None:
    """更新 employee.yaml 中的指定字段."""
    import tempfile

    import yaml

    config_path = emp_dir / "employee.yaml"
    if not config_path.exists():
        return
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        return
    config.update(updates)
    content = yaml.dump(config, allow_unicode=True, sort_keys=False, default_flow_style=False)
    fd, tmp = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
    fd_closed = False
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        fd_closed = True
        os.replace(tmp, config_path)
    except OSError:
        if not fd_closed:
            os.close(fd)
        Path(tmp).unlink(missing_ok=True)
        raise


def _parse_sender_name(extra_context: str | None) -> str | None:
    """从 extra_context 解析发送者名（格式: '当前对话用户: XXX'）."""
    if not extra_context:
        return None
    m = _re.search(r"当前对话用户[:：]\s*(\S+?)(?:（|$|\n)", extra_context)
    return m.group(1) if m else None


def _extract_task_description(text: str) -> str:
    """从 user_message 提取简洁的 task_description，过滤 soul prompt 污染.

    规则：
    - 以"你是"开头或超过 500 字 → 疑似 soul prompt，尝试提取 ## 任务 内容
    - 否则取前 200 字
    """
    if not text:
        return ""
    is_soul_prompt = text.startswith("你是") or len(text) > 500
    if is_soul_prompt:
        # 尝试提取 ## 任务 后的内容
        m = _re.search(r"##\s*(?:本次)?任务\s*\n+(.+)", text, _re.DOTALL)
        if m:
            task_text = m.group(1).strip()
            # 截取到下一个 ## 标题或文本结尾
            next_section = _re.search(r"\n##\s", task_text)
            if next_section:
                task_text = task_text[: next_section.start()].strip()
            if task_text:
                return task_text[:200]
        # 提取不到则用前 200 字，但不用 soul prompt 的"你是XXX"开头
        return text[:200]
    return text[:200]


async def _health(request: Any) -> Any:
    """健康检查."""
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "ok", "service": "crew-webhook"})


async def _metrics(request: Any) -> Any:
    """运行时指标."""
    from starlette.responses import JSONResponse

    from crew.metrics import get_collector

    return JSONResponse(get_collector().snapshot())


async def _handle_employee_prompt(request: Any, ctx: _AppContext) -> Any:
    """返回员工配置和渲染后的 system_prompt."""
    from starlette.responses import JSONResponse

    from crew.discovery import discover_employees
    from crew.engine import CrewEngine
    from crew.tool_schema import employee_tools_to_schemas

    identifier = request.path_params["identifier"]
    result = discover_employees(ctx.project_dir)

    employee = _find_employee(result, identifier)
    if not employee:
        return JSONResponse({"error": "Employee not found"}, status_code=404)

    engine = CrewEngine(ctx.project_dir)
    system_prompt = engine.prompt(employee)
    tool_schemas, _ = employee_tools_to_schemas(employee.tools, defer=False)

    # 从 YAML 读取 Employee model 之外的字段（bio, temperature 等）
    bio = ""
    temperature = None
    max_tokens = None
    if employee.source_path:
        yaml_path = employee.source_path / "employee.yaml"
        if yaml_path.exists():
            import yaml

            with open(yaml_path) as f:
                yaml_config = yaml.safe_load(f) or {}
            bio = yaml_config.get("bio", "")
            temperature = yaml_config.get("temperature")
            max_tokens = yaml_config.get("max_tokens")

    # 组织架构信息（团队、权限、成本）
    from crew.organization import get_effective_authority, load_organization

    org = load_organization(project_dir=ctx.project_dir)
    team = org.get_team(employee.name)
    authority = get_effective_authority(org, employee.name, project_dir=ctx.project_dir)

    # 7 天成本
    from crew.cost import query_cost_summary

    cost_summary = query_cost_summary(ctx.registry, employee=employee.name, days=7)

    # 推理模型不支持自定义 temperature（kimi-k2.5, o1, o3, deepseek-r 等）
    _model = employee.model or ""
    _REASONING_PREFIXES = ("kimi-k2", "o1-", "o3-", "o4-", "deepseek-r")
    if any(_model.startswith(p) for p in _REASONING_PREFIXES):
        temperature = 1

    return JSONResponse(
        {
            "name": employee.name,
            "character_name": employee.character_name,
            "display_name": employee.display_name,
            "description": employee.description,
            "bio": bio,
            "version": employee.version,
            "model": employee.model,
            "model_tier": employee.model_tier,
            "base_url": employee.base_url,
            "fallback_model": employee.fallback_model,
            "fallback_base_url": employee.fallback_base_url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": employee.tools,
            "tool_schemas": tool_schemas,
            "system_prompt": system_prompt,
            "agent_id": employee.agent_id,
            "team": team,
            "authority": authority,
            "cost_7d": cost_summary,
            "kpi": employee.kpi,
            "auto_memory": employee.auto_memory,
        }
    )


async def _handle_model_tiers(request: Any, ctx: _AppContext) -> Any:
    """返回可用的模型档位列表（不含密钥和内部 URL）."""
    from starlette.responses import JSONResponse

    from crew.organization import load_organization

    org = load_organization(project_dir=ctx.project_dir)
    tiers: dict[str, dict[str, str]] = {}
    for tier_name, tier_config in org.model_defaults.items():
        tiers[tier_name] = {
            "model": tier_config.model,
            "fallback_model": tier_config.fallback_model,
        }

    return JSONResponse({"tiers": tiers})


async def _handle_employee_list(request: Any, ctx: _AppContext) -> Any:
    """返回所有员工基本信息列表（供外部服务获取员工花名册）."""
    from starlette.responses import JSONResponse

    from crew.discovery import discover_employees

    result = discover_employees(ctx.project_dir)
    items = []
    for emp in result.employees.values():
        avatar_url = f"/static/avatars/{emp.agent_id}.webp" if emp.agent_id else None
        items.append(
            {
                "name": emp.name,
                "character_name": emp.character_name,
                "display_name": emp.display_name,
                "description": emp.description,
                "agent_id": emp.agent_id,
                "agent_status": emp.agent_status,
                "model": emp.model,
                "model_tier": emp.model_tier,
                "tags": emp.tags,
                "avatar_url": avatar_url,
            }
        )

    return JSONResponse({"items": items})


async def _generate_avatar_background(
    agent_id: str,
    character_name: str,
    description: str,
    avatar_prompt: str,
) -> None:
    """后台任务：生成并压缩头像."""
    from pathlib import Path

    from crew.avatar import compress_avatar, generate_avatar

    try:
        # 确定输出目录
        static_dir = Path(__file__).parent.parent.parent / "static" / "avatars"
        static_dir.mkdir(parents=True, exist_ok=True)

        # 生成原图（同步操作，但在后台任务中执行）
        raw_path = generate_avatar(
            character_name=character_name,
            description=description,
            avatar_prompt=avatar_prompt,
            output_dir=static_dir,
        )

        if not raw_path:
            logger.error("头像生成失败: %s", agent_id)
            return

        # 压缩为 webp
        final_path = static_dir / f"{agent_id}.webp"
        result = compress_avatar(raw_path, final_path)

        if result:
            logger.info("头像生成成功: %s", final_path)
        else:
            logger.error("头像压缩失败: %s", agent_id)

    except Exception:
        logger.exception("头像生成异常: %s", agent_id)


async def _handle_employee_create(request: Any, ctx: _AppContext) -> Any:
    """创建新员工 — POST /api/employees."""

    from crew.config_store import create_employee

    # 1. 解析请求
    try:
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body", 400)

    # 2. 验证必填字段
    name = body.get("name", "").strip()
    character_name = body.get("character_name", "").strip()
    soul_content = body.get("soul_content", "").strip()

    if not name or not character_name or not soul_content:
        return _error_response("name, character_name, soul_content are required", 400)

    # 3. 验证 name 格式（只允许 [a-z0-9-]）
    if not _re.match(r"^[a-z0-9-]+$", name):
        return _error_response("name must match [a-z0-9-]", 400)

    # 4. 提取其他参数
    display_name = body.get("display_name", "")
    description = body.get("description", "")
    model = body.get("model", "claude-sonnet-4-6")
    model_tier = body.get("model_tier", "claude")
    tags = body.get("tags", [])
    avatar_prompt = body.get("avatar_prompt", "")
    agent_status = body.get("agent_status", "active")

    # 5. 创建员工（数据库 + 文件系统）
    try:
        result = create_employee(
            name=name,
            character_name=character_name,
            display_name=display_name,
            description=description,
            model=model,
            model_tier=model_tier,
            tags=tags,
            soul_content=soul_content,
            agent_status=agent_status,
            avatar_prompt=avatar_prompt,
        )
    except ValueError as e:
        return _error_response(str(e), 409)
    except Exception as e:
        logger.exception("创建员工失败")
        return _error_response(str(e), 500)

    # 6. 启动后台任务生成头像
    agent_id = result.get("agent_id")
    if agent_id:
        task = asyncio.create_task(
            _generate_avatar_background(
                agent_id=agent_id,
                character_name=character_name,
                description=description,
                avatar_prompt=avatar_prompt,
            )
        )
        _background_tasks.add(task)
        task.add_done_callback(_task_done_callback)

    # 7. 返回成功响应
    return _ok_response(
        {
            "agent_id": agent_id,
            "employee_name": character_name,
            "name": name,
            "version": result.get("version"),
            "created_at": result.get("updated_at"),
            "avatar_status": "generating",
        },
        status_code=201,
    )


async def _handle_team_agents(request: Any, ctx: _AppContext) -> Any:
    """返回 active 状态的 AI 员工展示数据（供官网 about 页面使用）.

    返回格式兼容官网模板，每个元素包含:
    id, nickname, title, avatar_url, is_agent, staff_badge, bio, expertise, domains
    """
    import yaml as _yaml
    from starlette.responses import JSONResponse

    from crew.discovery import discover_employees

    result = discover_employees(ctx.project_dir)
    agents = []
    for emp in result.employees.values():
        if emp.agent_status != "active":
            continue

        # 从 employee.yaml 读取 bio 和 domains（Employee 模型未收录这两个字段）
        bio = ""
        domains: list[str] = []
        if emp.source_path and (emp.source_path / "employee.yaml").exists():
            try:
                raw = _yaml.safe_load(
                    (emp.source_path / "employee.yaml").read_text(encoding="utf-8")
                )
                if isinstance(raw, dict):
                    bio = raw.get("bio", "")
                    raw_domains = raw.get("domains", [])
                    domains = raw_domains if isinstance(raw_domains, list) else []
            except Exception:
                pass

        # agent_id 已经是 "AI3050" 格式的字符串
        public_id = emp.agent_id if emp.agent_id else None

        # 头像 URL：检查 static/avatars/{agent_id}.webp 是否存在
        avatar_url = ""
        if public_id:
            avatar_path = (
                (ctx.project_dir or Path(".")) / "static" / "avatars" / f"{public_id}.webp"
            )
            if avatar_path.exists():
                avatar_url = f"/static/avatars/{public_id}.webp"

        agents.append(
            {
                "id": public_id,
                "nickname": emp.character_name,
                "title": emp.display_name,
                "avatar_url": avatar_url,
                "is_agent": True,
                "staff_badge": "集识光年",
                "bio": bio,
                "expertise": emp.tags,
                "domains": domains,
            }
        )

    # 按 id 升序排列，None 排最后
    agents.sort(key=lambda a: (a["id"] is None, a["id"] or ""))

    return JSONResponse(agents)


async def _handle_employee_get(request: Any, ctx: _AppContext) -> Any:
    """返回单个员工的完整定义（对应 MCP get_employee）."""
    from starlette.responses import JSONResponse

    from crew.discovery import discover_employees

    identifier = request.path_params["identifier"]
    result = discover_employees(ctx.project_dir)

    employee = _find_employee(result, identifier)
    if not employee:
        return JSONResponse({"error": "Employee not found"}, status_code=404)

    data = employee.model_dump(
        mode="json",
        exclude={
            "source_path",
            "api_key",
            "fallback_api_key",
            "fallback_base_url",
        },
    )
    return JSONResponse(data)


async def _handle_employee_state(request: Any, ctx: _AppContext) -> Any:
    """返回员工完整运行时状态：角色设定 + 最近记忆 + 最近笔记."""
    from starlette.responses import JSONResponse

    from crew.discovery import discover_employees
    from crew.memory import MemoryStore

    identifier = request.path_params["identifier"]
    result = discover_employees(ctx.project_dir)

    employee = _find_employee(result, identifier)
    if not employee:
        return JSONResponse({"error": "Employee not found"}, status_code=404)

    limit = _safe_int(request.query_params.get("memory_limit", "10"), 10)
    sort_by = request.query_params.get("sort_by", "created_at")
    min_importance = _safe_int(request.query_params.get("min_importance", "0"), 0)
    max_tokens = _safe_int(request.query_params.get("max_tokens", "0"), 0)  # 0=不限

    # 读取 soul.md
    soul = ""
    if employee.source_path:
        soul_path = employee.source_path / "soul.md"
        if soul_path.exists():
            soul = soul_path.read_text(encoding="utf-8")

    # 读取最近记忆（API 只返回公开记忆，过滤 private）
    store = MemoryStore(project_dir=ctx.project_dir)
    memories = store.query(
        employee.name,
        limit=limit,
        max_visibility="open",
        sort_by=sort_by,
        min_importance=min_importance,
        update_access=True,
    )
    memory_list = [
        {"category": m.category, "content": m.content, "created_at": m.created_at, "tags": m.tags}
        for m in memories
    ]

    # 读取最近笔记（API 只返回公开笔记，过滤 private）
    notes_dir = (ctx.project_dir or Path.cwd()) / ".crew" / "notes"
    recent_notes: list[dict] = []
    if notes_dir.is_dir():
        note_files = sorted(notes_dir.glob("*.md"), reverse=True)
        for nf in note_files[:5]:
            text = nf.read_text(encoding="utf-8")
            # 跳过 private 笔记
            if "visibility: private" in text:
                continue
            if employee.character_name in text or employee.name in text:
                recent_notes.append({"filename": nf.name, "content": text[:500]})

    response_data = {
        "name": employee.name,
        "character_name": employee.character_name,
        "display_name": employee.display_name,
        "agent_status": employee.agent_status,
        "soul": soul,
        "memories": memory_list,
        "notes": recent_notes,
    }

    # Token 预算截断：粗估 1 token ≈ 3 中文字符 / 4 英文字符
    if max_tokens > 0:
        import json as _json

        budget_chars = max_tokens * 3  # 保守估计
        # soul 优先保留，记忆按顺序截断
        soul_chars = len(soul)
        remaining = budget_chars - soul_chars
        if remaining < 200:
            # soul 已经超预算，截断 soul
            response_data["soul"] = soul[:budget_chars] + "\n...(truncated)"
            response_data["memories"] = []
            response_data["notes"] = []
        else:
            # 从后往前裁记忆，保留高优先级的
            truncated_memories = []
            used = 0
            for m in memory_list:
                m_size = len(_json.dumps(m, ensure_ascii=False))
                if used + m_size > remaining:
                    break
                truncated_memories.append(m)
                used += m_size
            response_data["memories"] = truncated_memories
            # 剩余空间给 notes
            remaining -= used
            if remaining < 100:
                response_data["notes"] = []
            else:
                truncated_notes = []
                for n in recent_notes:
                    n_size = len(_json.dumps(n, ensure_ascii=False))
                    if remaining - n_size < 0:
                        break
                    truncated_notes.append(n)
                    remaining -= n_size
                response_data["notes"] = truncated_notes

    return JSONResponse(response_data)


async def _handle_employee_update(request: Any, ctx: _AppContext) -> Any:
    """更新员工配置（model 等）— employee.yaml 是唯一真相源."""
    from starlette.responses import JSONResponse

    from crew.discovery import discover_employees

    identifier = request.path_params["identifier"]
    result = discover_employees(ctx.project_dir)

    employee = _find_employee(result, identifier)
    if not employee:
        return JSONResponse({"error": "Employee not found"}, status_code=404)

    if not employee.source_path:
        return JSONResponse({"error": "Employee source path unknown"}, status_code=400)

    try:
        payload = await request.json()
    except (ValueError, TypeError):
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    # 只允许白名单字段
    updates = {}
    for key in _EMPLOYEE_UPDATABLE_FIELDS:
        if key in payload:
            updates[key] = payload[key]

    if not updates:
        return JSONResponse(
            {
                "error": f"No updatable fields. Allowed: {', '.join(sorted(_EMPLOYEE_UPDATABLE_FIELDS))}"
            },
            status_code=400,
        )

    # 写回 employee.yaml
    try:
        _write_yaml_field(employee.source_path, updates)
    except OSError as e:
        logger.exception("更新 employee.yaml 失败: %s", identifier)
        return JSONResponse({"error": f"Write failed: {e}"}, status_code=500)

    return JSONResponse(
        {
            "ok": True,
            "updated": updates,
            "employee": employee.name,
        }
    )


async def _handle_employee_delete(request: Any, ctx: _AppContext) -> Any:
    """删除员工（本地文件 + 数据库 soul + 头像）."""
    import shutil
    from pathlib import Path

    from starlette.responses import JSONResponse

    from crew.config_store import get_soul
    from crew.database import get_connection, is_pg
    from crew.discovery import discover_employees

    identifier = request.path_params["identifier"]
    result = discover_employees(ctx.project_dir, cache_ttl=0)

    employee = _find_employee(result, identifier)

    # 如果员工发现机制找不到，尝试直接从数据库查询
    character_name = None
    if employee:
        character_name = employee.character_name
    else:
        # 尝试用 identifier 作为 character_name 查询数据库
        if get_soul(identifier):
            character_name = identifier
        else:
            return JSONResponse({"error": "Employee not found"}, status_code=404)

    deleted_items = []

    # 1. 删除本地文件（如果存在）
    if employee and employee.source_path:
        source = employee.source_path
        try:
            if source.is_dir():
                shutil.rmtree(source)
                deleted_items.append(f"directory: {source}")
            elif source.is_file():
                source.unlink()
                deleted_items.append(f"file: {source}")
        except OSError:
            logger.exception("删除员工文件失败: %s", identifier)
            # 继续删除其他资源，不中断

    # 2. 删除数据库 soul 记录
    if character_name and is_pg():
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "DELETE FROM employee_souls WHERE employee_name = %s",
                    (character_name,),
                )
                if cur.rowcount > 0:
                    deleted_items.append(f"soul: {character_name}")
        except Exception:
            logger.exception("删除员工 soul 失败: %s", character_name)
            # 继续删除其他资源

    # 3. 删除头像文件（如果存在）
    if employee and employee.agent_id:
        try:
            static_dir = Path(__file__).parent.parent.parent / "static" / "avatars"
            avatar_path = static_dir / f"{employee.agent_id}.webp"
            if avatar_path.exists():
                avatar_path.unlink()
                deleted_items.append(f"avatar: {employee.agent_id}.webp")
        except OSError:
            logger.exception("删除头像失败: %s", employee.agent_id)
            # 继续

    if not deleted_items:
        return JSONResponse(
            {"error": "No resources deleted (employee may not exist)"},
            status_code=404,
        )

    return JSONResponse(
        {
            "ok": True,
            "deleted": employee.name if employee else identifier,
            "character_name": character_name,
            "deleted_items": deleted_items,
        }
    )


async def _handle_memory_add(request: Any, ctx: _AppContext) -> Any:
    """记忆写入 — POST /api/memory/add.

    接受（与 MCP add_memory 工具入参对齐）:
        {
            "employee": "backend-engineer",
            "category": "decision" | "estimate" | "finding" | "correction" | "pattern",
            "content": "记忆内容",
            "source_session": "claude-sess-xxx",
            "tags": ["auto-push", "claude-code"],
            "ttl_days": 0,
            "shared": false,
            "trigger_condition": "",
            "applicability": [],
            "origin_employee": ""
        }

    幂等：同 employee + source_session + category 不重复写入。
    """
    from starlette.responses import JSONResponse

    from crew.memory import get_memory_store

    try:
        payload = await request.json()
    except (ValueError, TypeError):
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    employee = payload.get("employee", "")
    category = payload.get("category", "")
    content = payload.get("content", "")
    source_session = payload.get("source_session", "")
    tags = payload.get("tags", [])
    ttl_days = payload.get("ttl_days", 0)
    shared = payload.get("shared", False)
    trigger_condition = payload.get("trigger_condition", "")
    applicability = payload.get("applicability", [])
    origin_employee = payload.get("origin_employee", "")
    classification = payload.get("classification", "internal")
    domain = payload.get("domain", [])

    if not employee or not category or not content:
        return JSONResponse({"error": "employee, category, content are required"}, status_code=400)

    valid_categories = {"decision", "estimate", "finding", "correction", "pattern"}
    if category not in valid_categories:
        return JSONResponse(
            {"error": f"category must be one of {valid_categories}"}, status_code=400
        )

    # 信息分级校验（写端）
    valid_classifications = {"public", "internal", "restricted", "confidential"}
    if classification not in valid_classifications:
        classification = "internal"  # 无效值降级为 internal

    # public 写入审计（外部用户可见，需记录）
    if classification == "public":
        logger.info(
            "public_memory_write: employee=%s category=%s content_preview=%s tags=%s",
            employee,
            category,
            content[:80],
            tags,
        )

    # restricted/confidential 写入限制
    if classification in ("restricted", "confidential"):
        from crew.classification import EMPLOYEE_CLEARANCE, CLASSIFICATION_LEVELS
        emp_clearance = EMPLOYEE_CLEARANCE.get(employee, {})
        emp_level = emp_clearance.get("clearance", "internal")
        if CLASSIFICATION_LEVELS.get(emp_level, 1) < CLASSIFICATION_LEVELS.get(classification, 2):
            return JSONResponse(
                {"error": f"员工 {employee} 的许可等级 ({emp_level}) 不足以写入 {classification} 级别记忆"},
                status_code=403,
            )

    # 质量检查（2026-03-02 记忆质量控制）
    from crew.memory_quality import check_memory_quality

    quality_result = check_memory_quality(category, content)
    if quality_result["score"] < 0.6:
        return JSONResponse(
            {
                "ok": False,
                "error": "Memory quality too low",
                "score": quality_result["score"],
                "issues": quality_result["issues"],
                "suggestions": quality_result["suggestions"],
            },
            status_code=400,
        )

    # 标签规范化和建议（2026-03-02 标签系统）
    from crew.memory_tags import normalize_tags, suggest_tags

    # 规范化用户提供的标签
    if tags and isinstance(tags, list):
        tags = normalize_tags(tags)

    # 自动建议标签（不强制添加，仅返回给用户）
    suggested_tags = suggest_tags(category, content, tags or [])

    # 拦截 trajectory 标签写入（2026-03-02 记忆系统优化）
    if isinstance(tags, list) and "trajectory" in tags:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Intercepted trajectory memory: employee={employee}, session={source_session}")
        return JSONResponse(
            {
                "ok": True,
                "skipped": True,
                "reason": "trajectory tag intercepted",
            }
        )

    # 相似度检测（2026-03-02 记忆去重）
    from crew.memory_similarity import find_similar_memories

    # 检查是否强制写入
    force = request.query_params.get("force") == "true"

    if not force:
        similar_memories = await find_similar_memories(
            employee=employee,
            content=content,
            category=category,
            threshold=0.85,
            project_dir=ctx.project_dir,
        )

        if similar_memories:
            return JSONResponse(
                {
                    "ok": False,
                    "warning": "similar_memories_found",
                    "similar_memories": [
                        {
                            "id": mem["id"],
                            "content": mem["content"][:200],
                            "similarity": round(score, 2),
                            "created_at": mem["created_at"],
                            "category": mem["category"],
                        }
                        for mem, score in similar_memories
                    ],
                    "suggestions": [
                        "如果是相同内容，考虑更新已有记忆而非新增",
                        "如果是补充信息，可以在原记忆基础上扩展",
                        "如果确实是新的独立经验，添加 ?force=true 参数重新提交",
                    ],
                }
            )

    store = get_memory_store(project_dir=ctx.project_dir)

    # 幂等检查：同 employee + source_session + category 不重复写入
    if source_session:
        existing = store.query(employee, limit=50)
        for entry in existing:
            # 兼容 MemoryEntry（属性访问）和 dict（键访问）
            _src = entry.source_session if hasattr(entry, "source_session") else entry.get("source_session", "")
            _cat = entry.category if hasattr(entry, "category") else entry.get("category", "")
            _eid = entry.id if hasattr(entry, "id") else entry.get("id", "")
            if _src == source_session and _cat == category:
                return JSONResponse(
                    {
                        "ok": True,
                        "skipped": True,
                        "reason": "duplicate source_session + category",
                        "existing_id": _eid,
                    }
                )

    result = store.add(
        employee=employee,
        category=category,
        content=content,
        source_session=source_session,
        tags=tags if isinstance(tags, list) else [],
        ttl_days=int(ttl_days) if ttl_days else 0,
        shared=bool(shared),
        trigger_condition=str(trigger_condition),
        applicability=applicability if isinstance(applicability, list) else [],
        origin_employee=str(origin_employee),
        classification=str(classification) if classification in ("public", "internal", "restricted", "confidential") else "internal",
        domain=domain if isinstance(domain, list) else [],
    )

    # 兼容 MemoryEntry（属性访问）和 dict（键访问）
    result_employee = result.employee if hasattr(result, "employee") else result.get("employee", employee)
    result_id = result.id if hasattr(result, "id") else result.get("id", "")

    # 写入后失效缓存（用解析后的花名作为 cache key）
    try:
        from crew.memory_cache import invalidate

        invalidate(result_employee)
    except Exception:
        pass

    return JSONResponse(
        {
            "ok": True,
            "skipped": False,
            "entry_id": result_id,
            "employee": result_employee,
            "category": category,
            "suggested_tags": suggested_tags,  # 2026-03-02 标签建议
        }
    )


async def _handle_memory_query(request: Any, ctx: _AppContext) -> Any:
    """记忆查询 — GET /api/memory/query.

    查询参数（与 MCP query_memory 工具入参对齐）:
        employee (required): 员工名称
        category (optional): 按类别过滤
        limit (optional): 最大返回条数，默认 20
    """
    from starlette.responses import JSONResponse

    from crew.memory import MemoryStore

    employee = request.query_params.get("employee", "")
    category = request.query_params.get("category") or None
    try:
        limit = int(request.query_params.get("limit", "20"))
    except (ValueError, TypeError):
        limit = 20

    if not employee:
        return JSONResponse({"error": "employee is required"}, status_code=400)

    store = MemoryStore(project_dir=ctx.project_dir)
    entries = store.query(
        employee=employee,
        category=category,
        limit=limit,
    )
    data = [e.model_dump() for e in entries]
    return JSONResponse({"ok": True, "entries": data, "total": len(data)})


async def _handle_memory_update(request: Any, ctx: _AppContext) -> Any:
    """更新已有记忆 — PUT /api/memory/{entry_id}.

    接受（JSON body）:
        {
            "entry_id": "abc123",
            "employee": "backend-engineer",
            "content": "更新后的内容",
            "tags": ["tag1", "tag2"],  # 可选
            "updated_by": "姜墨言",  # 可选，记录谁更新的
        }

    返回:
        {
            "ok": true,
            "entry_id": "abc123",
            "updated": true
        }
    """
    import json
    from datetime import datetime

    from starlette.responses import JSONResponse

    from crew.memory import MemoryStore

    try:
        payload = await request.json()
    except (ValueError, TypeError):
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    entry_id = payload.get("entry_id", "")
    employee = payload.get("employee", "")
    content = payload.get("content", "")
    tags = payload.get("tags")
    updated_by = payload.get("updated_by", "")

    if not entry_id or not employee or not content:
        return JSONResponse(
            {"error": "entry_id, employee, content are required"}, status_code=400
        )

    store = MemoryStore(project_dir=ctx.project_dir)

    # 查找原记忆
    employee = store._resolve_to_character_name(employee)
    path = store._employee_file(employee)

    if not path.exists():
        return JSONResponse({"error": "Employee not found"}, status_code=404)

    from crew.paths import file_lock

    found = False
    with file_lock(path):
        lines = path.read_text(encoding="utf-8").splitlines()
        new_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            try:
                from crew.memory import MemoryEntry

                entry = MemoryEntry(**json.loads(stripped))

                if entry.id == entry_id:
                    found = True

                    # 更新内容
                    entry.content = content

                    # 更新标签（如果提供）
                    if tags is not None and isinstance(tags, list):
                        from crew.memory_tags import normalize_tags

                        entry.tags = normalize_tags(tags)

                    # 记录更新历史（添加到 tags）
                    update_tag = f"updated-by:{updated_by or 'unknown'}"
                    update_time_tag = f"updated-at:{datetime.now().strftime('%Y-%m-%d')}"

                    if update_tag not in entry.tags:
                        entry.tags.append(update_tag)
                    if update_time_tag not in entry.tags:
                        entry.tags.append(update_time_tag)

                    new_lines.append(entry.model_dump_json())
                else:
                    new_lines.append(stripped)

            except (json.JSONDecodeError, ValueError):
                new_lines.append(stripped)

        if not found:
            return JSONResponse({"error": "Memory entry not found"}, status_code=404)

        # 重写文件
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    # 失效缓存
    try:
        from crew.memory_cache import invalidate

        invalidate(employee)
    except Exception:
        pass

    # 更新 embedding 缓存
    try:
        from crew.memory_similarity import get_embedding, _get_embedding_cache_path, _load_embedding_cache, _save_embedding_cache

        new_embedding = await get_embedding(content)
        if new_embedding is not None:
            cache = _load_embedding_cache(store.memory_dir, employee)
            cache[entry_id] = new_embedding
            _save_embedding_cache(store.memory_dir, employee, cache)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"更新 embedding 缓存失败: {e}")

    return JSONResponse(
        {
            "ok": True,
            "entry_id": entry_id,
            "updated": True,
        }
    )


async def _handle_memory_tags_list(request: Any, ctx: _AppContext) -> Any:
    """列出所有预定义标签 — GET /api/memory/tags.

    返回所有标签词典，用于前端展示和自动补全。
    """
    from starlette.responses import JSONResponse

    from crew.memory_tags import get_all_predefined_tags

    tags = get_all_predefined_tags()
    return JSONResponse({"ok": True, "tags": tags})


async def _handle_memory_tags_suggest(request: Any, ctx: _AppContext) -> Any:
    """根据内容建议标签 — GET /api/memory/tags/suggest.

    查询参数:
        category (required): 记忆类别
        content (required): 记忆内容
        existing_tags (optional): 已有标签（逗号分隔）
    """
    from starlette.responses import JSONResponse

    from crew.memory_tags import suggest_tags

    category = request.query_params.get("category", "")
    content = request.query_params.get("content", "")
    existing_tags_str = request.query_params.get("existing_tags", "")

    if not category or not content:
        return JSONResponse({"error": "category and content are required"}, status_code=400)

    valid_categories = {"decision", "estimate", "finding", "correction", "pattern"}
    if category not in valid_categories:
        return JSONResponse(
            {"error": f"category must be one of {valid_categories}"}, status_code=400
        )

    existing_tags = [tag.strip() for tag in existing_tags_str.split(",") if tag.strip()]
    suggestions = suggest_tags(category, content, existing_tags)

    return JSONResponse({"ok": True, "suggestions": suggestions})


async def _handle_memory_tags_search(request: Any, ctx: _AppContext) -> Any:
    """搜索标签 — GET /api/memory/tags/search.

    查询参数:
        query (required): 搜索关键词
        limit (optional): 最多返回数量，默认 10
    """
    from starlette.responses import JSONResponse

    from crew.memory_tags import search_tags

    query = request.query_params.get("query", "")
    try:
        limit = int(request.query_params.get("limit", "10"))
    except (ValueError, TypeError):
        limit = 10

    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)

    matches = search_tags(query, limit)
    return JSONResponse({"ok": True, "matches": matches})


async def _handle_memory_ingest(request: Any, ctx: _AppContext) -> Any:
    """接收外部讨论数据，写入参与者记忆和会议记录."""
    from starlette.responses import JSONResponse

    try:
        payload = await request.json()
    except (ValueError, TypeError):
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    try:
        from crew.discussion_ingest import DiscussionIngestor, DiscussionInput

        data = DiscussionInput(**payload)
        ingestor = DiscussionIngestor(project_dir=ctx.project_dir)
        results = ingestor.ingest(data)
        return JSONResponse(
            {
                "ok": True,
                "topic": data.topic,
                "memories_written": results["memories_written"],
                "meeting_id": results.get("meeting_id"),
                "participants": results["participants"],
            }
        )
    except (ValueError, TypeError, OSError) as e:
        logger.exception("记忆导入失败")
        return JSONResponse({"error": str(e)}, status_code=500)


async def _handle_memory_delete(request: Any, ctx: _AppContext) -> Any:
    """记忆删除 — DELETE /api/memory/{entry_id}.

    路径参数:
        entry_id: 记忆条目 ID

    查询参数（可选）:
        employee: 员工名（提供后只在该员工文件中查找，提高效率）

    返回:
        {"ok": true, "deleted": true} 或 {"ok": false, "error": "..."}
    """
    from starlette.responses import JSONResponse

    from crew.memory import MemoryStore

    # 从路径参数获取 entry_id
    entry_id = request.path_params.get("entry_id", "")
    if not entry_id:
        return JSONResponse({"ok": False, "error": "entry_id is required"}, status_code=400)

    # 从查询参数获取可选的 employee
    employee = request.query_params.get("employee")

    store = MemoryStore(project_dir=ctx.project_dir)

    try:
        deleted = store.delete(entry_id, employee=employee)
        if deleted:
            return JSONResponse({"ok": True, "deleted": True, "entry_id": entry_id})
        else:
            return JSONResponse(
                {"ok": False, "error": "Entry not found", "entry_id": entry_id}, status_code=404
            )
    except Exception as e:
        logger.exception("记忆删除失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_drafts_list(request: Any, ctx: _AppContext) -> Any:
    """列出记忆草稿 — GET /api/memory/drafts.

    查询参数（可选）:
        status: 按状态过滤（pending/approved/rejected）
        employee: 按员工过滤
        limit: 最大返回数量，默认 100

    返回:
        {
            "ok": true,
            "drafts": [...],
            "total": 10,
            "counts": {"pending": 5, "approved": 3, "rejected": 2}
        }
    """
    from starlette.responses import JSONResponse

    from crew.memory_drafts import MemoryDraftStore

    status = request.query_params.get("status")
    employee = request.query_params.get("employee")
    try:
        limit = int(request.query_params.get("limit", "100"))
    except (ValueError, TypeError):
        limit = 100

    try:
        store = MemoryDraftStore()
        drafts = store.list_drafts(status=status, employee=employee, limit=limit)
        counts = store.count_by_status()

        return JSONResponse(
            {
                "ok": True,
                "drafts": [d.model_dump() for d in drafts],
                "total": len(drafts),
                "counts": counts,
            }
        )
    except Exception as e:
        logger.exception("列出草稿失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_drafts_get(request: Any, ctx: _AppContext) -> Any:
    """查看草稿详情 — GET /api/memory/drafts/{draft_id}.

    路径参数:
        draft_id: 草稿 ID

    返回:
        {"ok": true, "draft": {...}} 或 {"ok": false, "error": "..."}
    """
    from starlette.responses import JSONResponse

    from crew.memory_drafts import MemoryDraftStore

    draft_id = request.path_params.get("draft_id", "")
    if not draft_id:
        return JSONResponse({"ok": False, "error": "draft_id is required"}, status_code=400)

    try:
        store = MemoryDraftStore()
        draft = store.get_draft(draft_id)

        if draft is None:
            return JSONResponse({"ok": False, "error": "Draft not found"}, status_code=404)

        return JSONResponse({"ok": True, "draft": draft.model_dump()})
    except Exception as e:
        logger.exception("获取草稿失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_drafts_approve(request: Any, ctx: _AppContext) -> Any:
    """批准草稿 — POST /api/memory/drafts/{draft_id}/approve.

    路径参数:
        draft_id: 草稿 ID

    JSON body（可选）:
        {
            "reviewed_by": "姜墨言"
        }

    返回:
        {"ok": true, "draft": {...}, "memory_id": "..."} 或 {"ok": false, "error": "..."}
    """
    from starlette.responses import JSONResponse

    from crew.memory import MemoryStore
    from crew.memory_drafts import MemoryDraftStore

    draft_id = request.path_params.get("draft_id", "")
    if not draft_id:
        return JSONResponse({"ok": False, "error": "draft_id is required"}, status_code=400)

    payload = (
        await request.json() if request.headers.get("content-type") == "application/json" else {}
    )
    reviewed_by = payload.get("reviewed_by", "system")

    try:
        draft_store = MemoryDraftStore()
        draft = draft_store.approve_draft(draft_id, reviewed_by=reviewed_by)

        if draft is None:
            return JSONResponse({"ok": False, "error": "Draft not found"}, status_code=404)

        # 写入正式记忆
        memory_store = MemoryStore(project_dir=ctx.project_dir)
        entry = memory_store.add(
            employee=draft.employee,
            category=draft.category,
            content=draft.content,
            tags=draft.tags,
            confidence=draft.confidence,
            source_session=draft.source_trajectory_id,
        )

        logger.info(
            "批准草稿并写入记忆: draft_id=%s memory_id=%s employee=%s",
            draft_id,
            entry.id,
            draft.employee,
        )

        return JSONResponse(
            {
                "ok": True,
                "draft": draft.model_dump(),
                "memory_id": entry.id,
            }
        )
    except Exception as e:
        logger.exception("批准草稿失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_drafts_reject(request: Any, ctx: _AppContext) -> Any:
    """拒绝草稿 — POST /api/memory/drafts/{draft_id}/reject.

    路径参数:
        draft_id: 草稿 ID

    JSON body（可选）:
        {
            "reason": "内容不够具体",
            "reviewed_by": "姜墨言"
        }

    返回:
        {"ok": true, "draft": {...}} 或 {"ok": false, "error": "..."}
    """
    from starlette.responses import JSONResponse

    from crew.memory_drafts import MemoryDraftStore

    draft_id = request.path_params.get("draft_id", "")
    if not draft_id:
        return JSONResponse({"ok": False, "error": "draft_id is required"}, status_code=400)

    payload = (
        await request.json() if request.headers.get("content-type") == "application/json" else {}
    )
    reason = payload.get("reason", "")
    reviewed_by = payload.get("reviewed_by", "system")

    try:
        store = MemoryDraftStore()
        draft = store.reject_draft(draft_id, reason=reason, reviewed_by=reviewed_by)

        if draft is None:
            return JSONResponse({"ok": False, "error": "Draft not found"}, status_code=404)

        logger.info("拒绝草稿: draft_id=%s reason=%s", draft_id, reason)

        return JSONResponse({"ok": True, "draft": draft.model_dump()})
    except Exception as e:
        logger.exception("拒绝草稿失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_archive_query(request: Any, ctx: _AppContext) -> Any:
    """查询归档记忆 — GET /api/memory/archive.

    查询参数:
        employee (required): 员工名称
        start_date (optional): 起始日期 ISO 8601
        end_date (optional): 结束日期 ISO 8601
        category (optional): 按类别过滤
        limit (optional): 最大返回数量，默认 100

    返回:
        {"ok": true, "entries": [...], "total": 10}
    """
    from starlette.responses import JSONResponse

    from crew.memory import MemoryStore
    from crew.memory_archive import MemoryArchive

    employee = request.query_params.get("employee", "")
    if not employee:
        return JSONResponse({"ok": False, "error": "employee is required"}, status_code=400)

    start_date_str = request.query_params.get("start_date")
    end_date_str = request.query_params.get("end_date")
    category = request.query_params.get("category")

    try:
        limit = int(request.query_params.get("limit", "100"))
    except (ValueError, TypeError):
        limit = 100

    try:
        from datetime import datetime

        start_date = datetime.fromisoformat(start_date_str) if start_date_str else None
        end_date = datetime.fromisoformat(end_date_str) if end_date_str else None

        memory_store = MemoryStore(project_dir=ctx.project_dir)
        archive = MemoryArchive(memory_store=memory_store)

        entries = archive.query_archive(
            employee=employee,
            start_date=start_date,
            end_date=end_date,
            category=category,
            limit=limit,
        )

        return JSONResponse(
            {
                "ok": True,
                "entries": [e.model_dump() for e in entries],
                "total": len(entries),
            }
        )
    except Exception as e:
        logger.exception("查询归档记忆失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_archive_restore(request: Any, ctx: _AppContext) -> Any:
    """恢复归档记忆 — POST /api/memory/archive/restore.

    JSON body:
        {
            "employee": "赵云帆",
            "entry_ids": ["id1", "id2", ...]
        }

    返回:
        {"ok": true, "restored": 2, "not_found": 0}
    """
    from starlette.responses import JSONResponse

    from crew.memory import MemoryStore
    from crew.memory_archive import MemoryArchive

    payload = (
        await request.json() if request.headers.get("content-type") == "application/json" else {}
    )

    employee = payload.get("employee", "")
    entry_ids = payload.get("entry_ids", [])

    if not employee:
        return JSONResponse({"ok": False, "error": "employee is required"}, status_code=400)

    if not entry_ids or not isinstance(entry_ids, list):
        return JSONResponse({"ok": False, "error": "entry_ids is required"}, status_code=400)

    try:
        memory_store = MemoryStore(project_dir=ctx.project_dir)
        archive = MemoryArchive(memory_store=memory_store)

        stats = archive.restore_from_archive(employee, entry_ids)

        logger.info(
            "恢复归档记忆: employee=%s restored=%d not_found=%d",
            employee,
            stats["restored"],
            stats["not_found"],
        )

        return JSONResponse({"ok": True, **stats})
    except Exception as e:
        logger.exception("恢复归档记忆失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_archive_stats(request: Any, ctx: _AppContext) -> Any:
    """获取归档统计 — GET /api/memory/archive/stats.

    查询参数:
        employee (required): 员工名称

    返回:
        {"ok": true, "total": 100, "by_year": {"2026": 50, "2025": 50}}
    """
    from starlette.responses import JSONResponse

    from crew.memory import MemoryStore
    from crew.memory_archive import MemoryArchive

    employee = request.query_params.get("employee", "")
    if not employee:
        return JSONResponse({"ok": False, "error": "employee is required"}, status_code=400)

    try:
        memory_store = MemoryStore(project_dir=ctx.project_dir)
        archive = MemoryArchive(memory_store=memory_store)

        stats = archive.get_archive_stats(employee)

        return JSONResponse({"ok": True, **stats})
    except Exception as e:
        logger.exception("获取归档统计失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_shared_list(request: Any, ctx: _AppContext) -> Any:
    """列出共享记忆 — GET /api/memory/shared.

    查询参数:
        tags (optional): 按标签过滤（逗号分隔）
        category (optional): 按类别过滤
        exclude_employee (optional): 排除指定员工
        limit (optional): 最大返回数量，默认 20

    返回:
        {"ok": true, "entries": [...], "total": 10}
    """
    from starlette.responses import JSONResponse

    from crew.memory import MemoryStore

    tags_str = request.query_params.get("tags", "")
    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else None
    category = request.query_params.get("category")
    exclude_employee = request.query_params.get("exclude_employee", "")

    try:
        limit = int(request.query_params.get("limit", "20"))
    except (ValueError, TypeError):
        limit = 20

    try:
        memory_store = MemoryStore(project_dir=ctx.project_dir)
        entries = memory_store.query_shared(
            tags=tags,
            exclude_employee=exclude_employee,
            limit=limit,
        )

        # 按类别过滤（query_shared 不支持 category 参数）
        if category:
            entries = [e for e in entries if e.category == category]

        return JSONResponse(
            {
                "ok": True,
                "entries": [e.model_dump() for e in entries],
                "total": len(entries),
            }
        )
    except Exception as e:
        logger.exception("列出共享记忆失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_shared_record_usage(request: Any, ctx: _AppContext) -> Any:
    """记录共享记忆使用 — POST /api/memory/shared/usage.

    JSON body:
        {
            "memory_id": "abc123",
            "memory_owner": "赵云帆",
            "used_by": "卫子昂",
            "context": "实现前端组件时参考"
        }

    返回:
        {"ok": true}
    """
    from starlette.responses import JSONResponse

    from crew.memory_shared_stats import SharedMemoryStats

    payload = (
        await request.json() if request.headers.get("content-type") == "application/json" else {}
    )

    memory_id = payload.get("memory_id", "")
    memory_owner = payload.get("memory_owner", "")
    used_by = payload.get("used_by", "")
    context = payload.get("context", "")

    if not memory_id or not memory_owner or not used_by:
        return JSONResponse(
            {"ok": False, "error": "memory_id, memory_owner, used_by are required"},
            status_code=400,
        )

    try:
        stats = SharedMemoryStats()
        stats.record_usage(memory_id, memory_owner, used_by, context)

        return JSONResponse({"ok": True})
    except Exception as e:
        logger.exception("记录共享记忆使用失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_shared_stats(request: Any, ctx: _AppContext) -> Any:
    """获取共享记忆统计 — GET /api/memory/shared/stats.

    查询参数:
        memory_id (optional): 获取指定记忆的使用统计
        owner (optional): 获取指定所有者的共享统计
        user (optional): 获取指定用户的使用记录
        popular (optional): 获取热门记忆（值为 true）

    返回:
        根据查询参数返回不同的统计信息
    """
    from starlette.responses import JSONResponse

    from crew.memory_shared_stats import SharedMemoryStats

    memory_id = request.query_params.get("memory_id")
    owner = request.query_params.get("owner")
    user = request.query_params.get("user")
    popular = request.query_params.get("popular") == "true"

    try:
        stats_manager = SharedMemoryStats()

        if memory_id:
            # 获取指定记忆的使用统计
            stats = stats_manager.get_usage_stats(memory_id)
            return JSONResponse({"ok": True, "memory_id": memory_id, **stats})

        elif owner:
            # 获取所有者的共享统计
            stats = stats_manager.get_memory_owner_stats(owner)
            return JSONResponse({"ok": True, "owner": owner, **stats})

        elif user:
            # 获取用户的使用记录
            usages = stats_manager.get_user_shared_usage(user, limit=50)
            return JSONResponse(
                {
                    "ok": True,
                    "user": user,
                    "usages": [u.model_dump() for u in usages],
                    "total": len(usages),
                }
            )

        elif popular:
            # 获取热门记忆
            memories = stats_manager.get_popular_memories(min_uses=2, limit=20)
            return JSONResponse({"ok": True, "popular_memories": memories})

        else:
            return JSONResponse(
                {"ok": False, "error": "请提供 memory_id、owner、user 或 popular 参数"},
                status_code=400,
            )

    except Exception as e:
        logger.exception("获取共享记忆统计失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_dashboard(request: Any, ctx: _AppContext) -> Any:
    """记忆管理仪表板数据 — GET /api/memory/dashboard.

    查询参数:
        employee (optional): 指定员工，不指定则返回全局统计

    返回:
        {
            "ok": true,
            "total_memories": 总记忆数,
            "by_category": {"finding": 10, "correction": 5, ...},
            "by_employee": {"赵云帆": 20, "卫子昂": 15, ...},
            "quality_distribution": {"high": 50, "medium": 30, "low": 20},
            "top_tags": [{"tag": "api", "count": 15}, ...]
        }
    """
    from starlette.responses import JSONResponse

    from crew.memory import MemoryStore

    employee = request.query_params.get("employee")

    try:
        memory_store = MemoryStore(project_dir=ctx.project_dir)

        if employee:
            # 单个员工的统计
            entries = memory_store.query(employee, limit=1000)
            total = len(entries)

            by_category = {}
            tag_counts = {}
            quality_dist = {"high": 0, "medium": 0, "low": 0}

            for entry in entries:
                # 类别统计
                by_category[entry.category] = by_category.get(entry.category, 0) + 1

                # 标签统计
                for tag in entry.tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

                # 质量分布（基于置信度）
                if entry.confidence >= 0.8:
                    quality_dist["high"] += 1
                elif entry.confidence >= 0.5:
                    quality_dist["medium"] += 1
                else:
                    quality_dist["low"] += 1

            top_tags = sorted(
                [{"tag": tag, "count": count} for tag, count in tag_counts.items()],
                key=lambda x: x["count"],
                reverse=True,
            )[:20]

            return JSONResponse(
                {
                    "ok": True,
                    "employee": employee,
                    "total_memories": total,
                    "by_category": by_category,
                    "quality_distribution": quality_dist,
                    "top_tags": top_tags,
                }
            )

        else:
            # 全局统计
            employees = memory_store.list_employees()
            total = 0
            by_category = {}
            by_employee = {}
            tag_counts = {}
            quality_dist = {"high": 0, "medium": 0, "low": 0}

            for emp in employees:
                entries = memory_store.query(emp, limit=1000)
                emp_count = len(entries)
                by_employee[emp] = emp_count
                total += emp_count

                for entry in entries:
                    by_category[entry.category] = by_category.get(entry.category, 0) + 1

                    for tag in entry.tags:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1

                    if entry.confidence >= 0.8:
                        quality_dist["high"] += 1
                    elif entry.confidence >= 0.5:
                        quality_dist["medium"] += 1
                    else:
                        quality_dist["low"] += 1

            top_tags = sorted(
                [{"tag": tag, "count": count} for tag, count in tag_counts.items()],
                key=lambda x: x["count"],
                reverse=True,
            )[:20]

            return JSONResponse(
                {
                    "ok": True,
                    "total_memories": total,
                    "by_category": by_category,
                    "by_employee": by_employee,
                    "quality_distribution": quality_dist,
                    "top_tags": top_tags,
                }
            )

    except Exception as e:
        logger.exception("获取仪表板数据失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_batch_update(request: Any, ctx: _AppContext) -> Any:
    """批量更新记忆 — POST /api/memory/batch/update.

    JSON body:
        {
            "employee": "赵云帆",
            "entry_ids": ["id1", "id2", ...],
            "updates": {
                "tags": ["new-tag"],  // 添加标签
                "remove_tags": ["old-tag"],  // 移除标签
                "confidence": 0.9  // 更新置信度
            }
        }

    返回:
        {"ok": true, "updated": 2, "failed": 0}
    """
    from starlette.responses import JSONResponse

    from crew.memory import MemoryStore

    payload = (
        await request.json() if request.headers.get("content-type") == "application/json" else {}
    )

    employee = payload.get("employee", "")
    entry_ids = payload.get("entry_ids", [])
    updates = payload.get("updates", {})

    if not employee or not entry_ids:
        return JSONResponse(
            {"ok": False, "error": "employee and entry_ids are required"},
            status_code=400,
        )

    try:
        memory_store = MemoryStore(project_dir=ctx.project_dir)
        path = memory_store._employee_file(employee)

        if not path.exists():
            return JSONResponse({"ok": False, "error": "Employee not found"}, status_code=404)

        from crew.memory import MemoryEntry
        from crew.paths import file_lock

        id_set = set(entry_ids)
        updated_count = 0
        failed_count = 0

        with file_lock(path):
            lines = path.read_text(encoding="utf-8").splitlines()
            new_lines = []

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    entry = MemoryEntry(**json.loads(stripped))

                    if entry.id in id_set:
                        # 应用更新
                        if "tags" in updates:
                            entry.tags = list(set(entry.tags + updates["tags"]))

                        if "remove_tags" in updates:
                            for tag in updates["remove_tags"]:
                                if tag in entry.tags:
                                    entry.tags.remove(tag)

                        if "confidence" in updates:
                            entry.confidence = float(updates["confidence"])

                        new_lines.append(entry.model_dump_json())
                        updated_count += 1
                    else:
                        new_lines.append(stripped)

                except (json.JSONDecodeError, ValueError):
                    new_lines.append(stripped)
                    failed_count += 1

            path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        logger.info(
            "批量更新记忆: employee=%s updated=%d failed=%d",
            employee,
            updated_count,
            failed_count,
        )

        return JSONResponse({"ok": True, "updated": updated_count, "failed": failed_count})

    except Exception as e:
        logger.exception("批量更新记忆失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_batch_delete(request: Any, ctx: _AppContext) -> Any:
    """批量删除记忆 — POST /api/memory/batch/delete.

    JSON body:
        {
            "employee": "赵云帆",
            "entry_ids": ["id1", "id2", ...]
        }

    返回:
        {"ok": true, "deleted": 2}
    """
    from starlette.responses import JSONResponse

    from crew.memory import MemoryStore

    payload = (
        await request.json() if request.headers.get("content-type") == "application/json" else {}
    )

    employee = payload.get("employee", "")
    entry_ids = payload.get("entry_ids", [])

    if not employee or not entry_ids:
        return JSONResponse(
            {"ok": False, "error": "employee and entry_ids are required"},
            status_code=400,
        )

    try:
        memory_store = MemoryStore(project_dir=ctx.project_dir)
        deleted_count = 0

        for entry_id in entry_ids:
            if memory_store.delete(entry_id, employee=employee):
                deleted_count += 1

        logger.info("批量删除记忆: employee=%s deleted=%d", employee, deleted_count)

        return JSONResponse({"ok": True, "deleted": deleted_count})

    except Exception as e:
        logger.exception("批量删除记忆失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_trajectory_export(request: Any, ctx: _AppContext) -> Any:
    """导出轨迹数据集 — POST /api/trajectory/export.

    JSON body:
        {
            "employee": "赵云帆",  // 可选
            "start_date": "2026-03-01T00:00:00",  // 可选
            "end_date": "2026-03-02T23:59:59",  // 可选
            "min_quality": 0.7,  // 可选，最低质量分数
            "max_samples": 1000,  // 可选，最大样本数
            "output_file": "/tmp/dataset.jsonl"  // 可选，默认自动生成
        }

    返回:
        {"ok": true, "total": 100, "exported": 95, "output_file": "..."}
    """
    from starlette.responses import JSONResponse

    from crew.trajectory_export import TrajectoryExporter

    payload = (
        await request.json() if request.headers.get("content-type") == "application/json" else {}
    )

    employee = payload.get("employee")
    start_date_str = payload.get("start_date")
    end_date_str = payload.get("end_date")
    min_quality = payload.get("min_quality", 0.0)
    max_samples = payload.get("max_samples", 0)
    output_file_str = payload.get("output_file")

    try:
        from datetime import datetime
        from pathlib import Path

        start_date = datetime.fromisoformat(start_date_str) if start_date_str else None
        end_date = datetime.fromisoformat(end_date_str) if end_date_str else None

        # 生成输出文件名
        if output_file_str:
            output_file = Path(output_file_str)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"/tmp/trajectory_dataset_{timestamp}.jsonl")

        exporter = TrajectoryExporter()
        stats = exporter.export_dataset(
            output_file=output_file,
            employee=employee,
            start_date=start_date,
            end_date=end_date,
            min_quality=min_quality,
            max_samples=max_samples,
        )

        logger.info(
            "导出轨迹数据集: file=%s total=%d exported=%d",
            output_file,
            stats["total"],
            stats["exported"],
        )

        return JSONResponse(
            {
                "ok": True,
                "output_file": str(output_file),
                **stats,
            }
        )

    except Exception as e:
        logger.exception("导出轨迹数据集失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_trajectory_annotation_add(request: Any, ctx: _AppContext) -> Any:
    """添加轨迹标注 — POST /api/trajectory/annotations.

    JSON body:
        {
            "trajectory_id": "traj-123",
            "quality_score": 0.85,
            "annotator": "姜墨言",
            "notes": "高质量轨迹，推理清晰"
        }

    返回:
        {"ok": true, "annotation": {...}}
    """
    from starlette.responses import JSONResponse

    from crew.trajectory_export import TrajectoryExporter

    payload = (
        await request.json() if request.headers.get("content-type") == "application/json" else {}
    )

    trajectory_id = payload.get("trajectory_id", "")
    quality_score = payload.get("quality_score")
    annotator = payload.get("annotator", "")
    notes = payload.get("notes", "")

    if not trajectory_id or quality_score is None or not annotator:
        return JSONResponse(
            {"ok": False, "error": "trajectory_id, quality_score, annotator are required"},
            status_code=400,
        )

    try:
        quality_score = float(quality_score)
        if not 0 <= quality_score <= 1:
            return JSONResponse(
                {"ok": False, "error": "quality_score must be between 0 and 1"},
                status_code=400,
            )

        exporter = TrajectoryExporter()
        annotation = exporter.add_annotation(
            trajectory_id=trajectory_id,
            quality_score=quality_score,
            annotator=annotator,
            notes=notes,
        )

        return JSONResponse({"ok": True, "annotation": annotation.model_dump()})

    except Exception as e:
        logger.exception("添加轨迹标注失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_trajectory_annotation_list(request: Any, ctx: _AppContext) -> Any:
    """列出轨迹标注 — GET /api/trajectory/annotations.

    查询参数:
        min_quality (optional): 最低质量分数
        annotator (optional): 按标注人过滤

    返回:
        {"ok": true, "annotations": [...], "total": 10}
    """
    from starlette.responses import JSONResponse

    from crew.trajectory_export import TrajectoryExporter

    min_quality_str = request.query_params.get("min_quality", "0.0")
    annotator = request.query_params.get("annotator")

    try:
        min_quality = float(min_quality_str)

        exporter = TrajectoryExporter()
        annotations = exporter.list_annotations(
            min_quality=min_quality,
            annotator=annotator,
        )

        return JSONResponse(
            {
                "ok": True,
                "annotations": [a.model_dump() for a in annotations],
                "total": len(annotations),
            }
        )

    except Exception as e:
        logger.exception("列出轨迹标注失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_semantic_search(request: Any, ctx: _AppContext) -> Any:
    """语义搜索记忆 — POST /api/memory/semantic/search.

    请求体:
        {
            "query": "搜索查询",
            "employee": "员工名（可选）",
            "category": "类别（可选）",
            "min_confidence": 0.0,
            "limit": 10
        }

    返回:
        {"ok": true, "results": [...], "total": 5}
    """
    from starlette.responses import JSONResponse

    from crew.memory_semantic import SemanticSearchEngine

    try:
        body = await request.json()
        query = body.get("query", "")
        employee = body.get("employee")
        category = body.get("category")
        min_confidence = body.get("min_confidence", 0.0)
        limit = body.get("limit", 10)

        if not query:
            return JSONResponse({"ok": False, "error": "query 参数必填"}, status_code=400)

        engine = SemanticSearchEngine()
        results = engine.search(
            query=query,
            employee=employee,
            category=category,
            min_confidence=min_confidence,
            limit=limit,
        )

        return JSONResponse(
            {
                "ok": True,
                "results": [r.model_dump() for r in results],
                "total": len(results),
            }
        )

    except Exception as e:
        logger.exception("语义搜索失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_recommend(request: Any, ctx: _AppContext) -> Any:
    """为任务推荐记忆 — POST /api/memory/semantic/recommend.

    请求体:
        {
            "task_description": "任务描述",
            "employee": "员工名",
            "limit": 5
        }

    返回:
        {"ok": true, "recommendations": [...], "total": 3}
    """
    from starlette.responses import JSONResponse

    from crew.memory_semantic import SemanticSearchEngine

    try:
        body = await request.json()
        task_description = body.get("task_description", "")
        employee = body.get("employee", "")
        limit = body.get("limit", 5)

        if not task_description or not employee:
            return JSONResponse(
                {"ok": False, "error": "task_description 和 employee 参数必填"},
                status_code=400,
            )

        engine = SemanticSearchEngine()
        recommendations = engine.recommend_for_task(
            task_description=task_description,
            employee=employee,
            limit=limit,
        )

        return JSONResponse(
            {
                "ok": True,
                "recommendations": [r.model_dump() for r in recommendations],
                "total": len(recommendations),
            }
        )

    except Exception as e:
        logger.exception("推荐记忆失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_similar(request: Any, ctx: _AppContext) -> Any:
    """查找相似记忆 — GET /api/memory/semantic/similar/{memory_id}.

    路径参数:
        memory_id: 记忆 ID

    查询参数:
        limit (optional): 最大返回数量，默认 5

    返回:
        {"ok": true, "similar": [...], "total": 3}
    """
    from starlette.responses import JSONResponse

    from crew.memory_semantic import SemanticSearchEngine

    try:
        # 从路径中提取 memory_id
        path = request.url.path
        memory_id = path.split("/")[-1]

        limit_str = request.query_params.get("limit", "5")
        limit = int(limit_str)

        engine = SemanticSearchEngine()
        similar = engine.find_similar_memories(
            memory_id=memory_id,
            limit=limit,
        )

        return JSONResponse(
            {
                "ok": True,
                "similar": [s.model_dump() for s in similar],
                "total": len(similar),
            }
        )

    except Exception as e:
        logger.exception("查找相似记忆失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_feedback_submit(request: Any, ctx: _AppContext) -> Any:
    """提交记忆反馈 — POST /api/memory/feedback.

    请求体:
        {
            "memory_id": "记忆 ID",
            "employee": "员工名",
            "feedback_type": "helpful|not_helpful|outdated|incorrect",
            "submitted_by": "提交人",
            "context": "使用场景（可选）",
            "comment": "反馈评论（可选）"
        }

    返回:
        {"ok": true, "feedback": {...}}
    """
    from starlette.responses import JSONResponse

    from crew.memory_feedback import MemoryFeedbackManager

    try:
        body = await request.json()
        memory_id = body.get("memory_id", "")
        employee = body.get("employee", "")
        feedback_type = body.get("feedback_type", "")
        submitted_by = body.get("submitted_by", "")
        context = body.get("context", "")
        comment = body.get("comment", "")

        if not memory_id or not employee or not feedback_type or not submitted_by:
            return JSONResponse(
                {"ok": False, "error": "memory_id, employee, feedback_type, submitted_by 参数必填"},
                status_code=400,
            )

        if feedback_type not in ["helpful", "not_helpful", "outdated", "incorrect"]:
            return JSONResponse(
                {"ok": False, "error": "feedback_type 必须是 helpful/not_helpful/outdated/incorrect"},
                status_code=400,
            )

        manager = MemoryFeedbackManager()
        feedback = manager.submit_feedback(
            memory_id=memory_id,
            employee=employee,
            feedback_type=feedback_type,
            submitted_by=submitted_by,
            context=context,
            comment=comment,
        )

        return JSONResponse(
            {
                "ok": True,
                "feedback": feedback.model_dump(),
            }
        )

    except Exception as e:
        logger.exception("提交反馈失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_feedback_get(request: Any, ctx: _AppContext) -> Any:
    """获取记忆反馈 — GET /api/memory/feedback/{memory_id}.

    路径参数:
        memory_id: 记忆 ID

    返回:
        {"ok": true, "feedback": [...], "total": 5}
    """
    from starlette.responses import JSONResponse

    from crew.memory_feedback import MemoryFeedbackManager

    try:
        # 从路径中提取 memory_id
        path = request.url.path
        memory_id = path.split("/")[-1]

        manager = MemoryFeedbackManager()
        feedback = manager.get_feedback(memory_id)

        return JSONResponse(
            {
                "ok": True,
                "feedback": [f.model_dump() for f in feedback],
                "total": len(feedback),
            }
        )

    except Exception as e:
        logger.exception("获取反馈失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_usage_stats(request: Any, ctx: _AppContext) -> Any:
    """获取记忆使用统计 — GET /api/memory/usage/stats/{memory_id}.

    路径参数:
        memory_id: 记忆 ID

    返回:
        {"ok": true, "stats": {...}}
    """
    from starlette.responses import JSONResponse

    from crew.memory_feedback import MemoryFeedbackManager

    try:
        # 从路径中提取 memory_id
        path = request.url.path
        memory_id = path.split("/")[-1]

        manager = MemoryFeedbackManager()
        stats = manager.get_stats(memory_id)

        if stats is None:
            return JSONResponse(
                {"ok": False, "error": "统计不存在"},
                status_code=404,
            )

        return JSONResponse(
            {
                "ok": True,
                "stats": stats.model_dump(),
            }
        )

    except Exception as e:
        logger.exception("获取统计失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_usage_record(request: Any, ctx: _AppContext) -> Any:
    """记录记忆使用 — POST /api/memory/usage/record.

    请求体:
        {
            "memory_id": "记忆 ID",
            "employee": "员工名",
            "relevance_score": 0.85
        }

    返回:
        {"ok": true}
    """
    from starlette.responses import JSONResponse

    from crew.memory_feedback import MemoryFeedbackManager

    try:
        body = await request.json()
        memory_id = body.get("memory_id", "")
        employee = body.get("employee", "")
        relevance_score = body.get("relevance_score", 0.0)

        if not memory_id or not employee:
            return JSONResponse(
                {"ok": False, "error": "memory_id 和 employee 参数必填"},
                status_code=400,
            )

        manager = MemoryFeedbackManager()
        manager.record_usage(
            memory_id=memory_id,
            employee=employee,
            relevance_score=relevance_score,
        )

        return JSONResponse({"ok": True})

    except Exception as e:
        logger.exception("记录使用失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_low_quality(request: Any, ctx: _AppContext) -> Any:
    """获取低质量记忆 — GET /api/memory/usage/low-quality.

    查询参数:
        employee (optional): 按员工过滤
        min_uses (optional): 最少使用次数，默认 5
        max_helpful_ratio (optional): 最大有帮助比例，默认 0.3

    返回:
        {"ok": true, "memories": [...], "total": 3}
    """
    from starlette.responses import JSONResponse

    from crew.memory_feedback import MemoryFeedbackManager

    try:
        employee = request.query_params.get("employee")
        min_uses = int(request.query_params.get("min_uses", "5"))
        max_helpful_ratio = float(request.query_params.get("max_helpful_ratio", "0.3"))

        manager = MemoryFeedbackManager()
        memories = manager.get_low_quality_memories(
            employee=employee,
            min_uses=min_uses,
            max_helpful_ratio=max_helpful_ratio,
        )

        return JSONResponse(
            {
                "ok": True,
                "memories": [m.model_dump() for m in memories],
                "total": len(memories),
            }
        )

    except Exception as e:
        logger.exception("获取低质量记忆失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_popular(request: Any, ctx: _AppContext) -> Any:
    """获取热门记忆 — GET /api/memory/usage/popular.

    查询参数:
        employee (optional): 按员工过滤
        limit (optional): 最大返回数量，默认 10

    返回:
        {"ok": true, "memories": [...], "total": 10}
    """
    from starlette.responses import JSONResponse

    from crew.memory_feedback import MemoryFeedbackManager

    try:
        employee = request.query_params.get("employee")
        limit = int(request.query_params.get("limit", "10"))

        manager = MemoryFeedbackManager()
        memories = manager.get_popular_memories(
            employee=employee,
            limit=limit,
        )

        return JSONResponse(
            {
                "ok": True,
                "memories": [m.model_dump() for m in memories],
                "total": len(memories),
            }
        )

    except Exception as e:
        logger.exception("获取热门记忆失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_memory_feedback_summary(request: Any, ctx: _AppContext) -> Any:
    """获取反馈汇总 — GET /api/memory/feedback/summary.

    查询参数:
        employee (optional): 按员工过滤

    返回:
        {"ok": true, "summary": {...}}
    """
    from starlette.responses import JSONResponse

    from crew.memory_feedback import MemoryFeedbackManager

    try:
        employee = request.query_params.get("employee")

        manager = MemoryFeedbackManager()
        summary = manager.get_feedback_summary(employee=employee)

        return JSONResponse(
            {
                "ok": True,
                "summary": summary,
            }
        )

    except Exception as e:
        logger.exception("获取反馈汇总失败")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


async def _handle_github(request: Any, ctx: _AppContext) -> Any:
    """处理 GitHub webhook."""
    from starlette.responses import JSONResponse

    from crew.webhook_config import (
        match_route,
        resolve_target_args,
        verify_github_signature,
    )

    body = await request.body()

    signature = request.headers.get("x-hub-signature-256")
    if not ctx.config.github_secret:
        logger.warning("GitHub webhook secret 未配置，拒绝请求")
        return JSONResponse({"error": "webhook secret not configured"}, status_code=403)
    if not verify_github_signature(body, signature, ctx.config.github_secret):
        return JSONResponse({"error": "invalid signature"}, status_code=401)

    event_type = request.headers.get("x-github-event", "")
    if not event_type:
        return JSONResponse({"error": "missing X-GitHub-Event header"}, status_code=400)

    payload = await request.json()

    route = match_route(event_type, ctx.config)
    if route is None:
        return JSONResponse({"message": "no matching route", "event": event_type}, status_code=200)

    args = resolve_target_args(route.target, payload)
    import crew.webhook as _wh

    return await _wh._dispatch_task(
        ctx,
        trigger="github",
        target_type=route.target.type,
        target_name=route.target.name,
        args=args,
        sync=False,
    )


async def _handle_openclaw(request: Any, ctx: _AppContext) -> Any:
    """处理 OpenClaw 消息事件."""
    from starlette.responses import JSONResponse

    payload = await request.json()

    target_type = payload.get("target_type", "employee")
    target_name = payload.get("target_name", "")
    args = payload.get("args", {})
    sync = payload.get("sync", False)

    if not target_name:
        return JSONResponse({"error": "missing target_name"}, status_code=400)

    import crew.webhook as _wh

    return await _wh._dispatch_task(
        ctx,
        trigger="openclaw",
        target_type=target_type,
        target_name=target_name,
        args=args,
        sync=sync,
    )


async def _handle_generic(request: Any, ctx: _AppContext) -> Any:
    """处理通用 JSON webhook."""
    from starlette.responses import JSONResponse

    payload = await request.json()

    target_type = payload.get("target_type", "pipeline")
    target_name = payload.get("target_name", "")
    args = payload.get("args", {})
    sync = payload.get("sync", False)

    if not target_name:
        return JSONResponse({"error": "missing target_name"}, status_code=400)

    import crew.webhook as _wh

    return await _wh._dispatch_task(
        ctx,
        trigger="generic",
        target_type=target_type,
        target_name=target_name,
        args=args,
        sync=sync,
    )


async def _handle_run_pipeline(request: Any, ctx: _AppContext) -> Any:
    """直接触发 pipeline."""

    name = request.path_params["name"]
    payload = (
        await request.json() if request.headers.get("content-type") == "application/json" else {}
    )
    args = payload.get("args", {})
    sync = payload.get("sync", False)
    agent_id = payload.get("agent_id")

    import crew.webhook as _wh

    return await _wh._dispatch_task(
        ctx,
        trigger="direct",
        target_type="pipeline",
        target_name=name,
        args=args,
        sync=sync,
        agent_id=agent_id,
    )


async def _run_and_callback(
    *,
    ctx: _AppContext,
    name: str,
    args: dict,
    agent_id: str | None,
    model: str | None,
    user_message: str,
    message_history: list | None,
    extra_context: str | None,
    sender_name: str,
    channel: str,
    callback_channel_id: int,
    callback_sender_id: str | None,
    callback_parent_id: int | None,
) -> None:
    """后台执行员工 + 回调蚁聚发频道消息（异步回调模式）."""
    import time as _time

    import httpx

    from crew.webhook_context import _ANTGATHER_API_TOKEN, _ANTGATHER_API_URL

    logger.info(
        "异步回调开始: emp=%s channel=%d sender=%s",
        name,
        callback_channel_id,
        callback_sender_id,
    )

    # 注入额外上下文
    _args = dict(args)
    if extra_context:
        task = _args.get("task", "")
        _args["task"] = (extra_context + "\n\n" + task) if task else extra_context

    # 轨迹录制
    from contextlib import ExitStack

    from crew.trajectory import TrajectoryCollector

    _exit_stack = ExitStack()
    _task_desc = _extract_task_description(
        user_message if isinstance(user_message, str) else str(user_message)
    )
    _traj_collector = TrajectoryCollector.try_create_for_employee(
        name,
        _task_desc,
        channel=channel,
        project_dir=ctx.project_dir,
    )
    if _traj_collector is not None:
        _exit_stack.enter_context(_traj_collector)

    # 执行员工（fast/full path 路由）
    try:
        from crew.discovery import discover_employees
        from crew.tool_schema import AGENT_TOOLS
        from crew.webhook_feishu import _needs_tools

        discovery = discover_employees(project_dir=ctx.project_dir)
        emp = discovery.get(name)
        has_tools = any(t in AGENT_TOOLS for t in (emp.tools or [])) if emp else False
        use_fast_path = (
            has_tools
            and emp is not None
            and emp.fallback_model
            and isinstance(user_message, str)
            and not _needs_tools(user_message)
        )

        _t0 = _time.monotonic()
        if use_fast_path:
            from crew.webhook_feishu import _feishu_fast_reply

            result = await _feishu_fast_reply(
                ctx,
                emp,
                user_message,
                message_history=message_history,
                max_visibility="private",
                extra_context=extra_context,
                sender_name=sender_name,
            )
        else:
            import crew.webhook as _wh

            result = await _wh._execute_employee(
                ctx,
                name,
                _args,
                agent_id=agent_id,
                model=model,
                user_message=user_message,
                message_history=message_history,
            )

        _elapsed = _time.monotonic() - _t0
        _path = "fast" if use_fast_path else "full"
        _m = result.get("model", "?") if isinstance(result, dict) else "?"
        _in = result.get("input_tokens", 0) if isinstance(result, dict) else 0
        _out = result.get("output_tokens", 0) if isinstance(result, dict) else 0
        logger.info(
            "异步回调执行完成 [%s] %.1fs model=%s in=%d out=%d emp=%s msg=%s",
            _path,
            _elapsed,
            _m,
            _in,
            _out,
            name,
            user_message[:40],
        )
    except Exception:
        logger.exception("异步回调执行失败: emp=%s channel=%d", name, callback_channel_id)
        result = None

    # 轨迹录制完成
    if _traj_collector is not None:
        try:
            _traj_collector.finish(success=result is not None)
        except Exception as _te:
            logger.debug("异步轨迹录制失败: %s", _te)
        finally:
            _exit_stack.close()

    # 记录任务
    try:
        record = ctx.registry.create(
            trigger=channel,
            target_type="employee",
            target_name=name,
            args=_args,
        )
        ctx.registry.update(record.task_id, "completed", result=result)
    except Exception:
        pass

    # 回复后记忆写回（fire-and-forget）
    if isinstance(result, dict) and result.get("output"):
        try:
            from crew.reply_postprocess import push_if_needed

            _reply_text = result["output"].strip()
            _turn_count = len(message_history) if message_history else 1
            push_if_needed(
                employee=name,
                reply=_reply_text,
                turn_count=_turn_count,
                session_id=f"antgather-callback-{callback_channel_id}",
            )
        except Exception as _mem_err:
            logger.debug("回复记忆写回失败（不影响回调）: %s", _mem_err)

    # 回调蚁聚：发频道消息
    from crew.output_sanitizer import strip_internal_tags

    output = ""
    if isinstance(result, dict):
        output = strip_internal_tags((result.get("output") or "").strip())
    if not output:
        logger.warning(
            "异步回调: 员工返回空内容，跳过回调 emp=%s channel=%d", name, callback_channel_id
        )
        return

    if not _ANTGATHER_API_URL or not _ANTGATHER_API_TOKEN:
        logger.error("异步回调: 蚁聚 API 未配置，无法发送频道消息")
        return

    # 引用回复时，默认 @原消息发送者（触发通知）
    if callback_parent_id and sender_name and not output.startswith(f"@{sender_name}"):
        output = f"@{sender_name} {output}"

    callback_url = f"{_ANTGATHER_API_URL}/api/internal/channels/{callback_channel_id}/messages"
    callback_payload: dict[str, Any] = {
        "sender_id": callback_sender_id,
        "content": output,
    }
    if callback_parent_id:
        callback_payload["parent_id"] = callback_parent_id

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                callback_url,
                json=callback_payload,
                headers={"Authorization": f"Bearer {_ANTGATHER_API_TOKEN}"},
            )
        if resp.is_success:
            logger.info(
                "异步回调成功: emp=%s channel=%d len=%d",
                name,
                callback_channel_id,
                len(output),
            )
        else:
            logger.error(
                "异步回调失败 (HTTP %d): %s, emp=%s channel=%d",
                resp.status_code,
                resp.text[:200],
                name,
                callback_channel_id,
            )
    except Exception:
        logger.exception("异步回调请求异常: emp=%s channel=%d", name, callback_channel_id)


async def _handle_run_employee(request: Any, ctx: _AppContext) -> Any:
    """直接触发员工（支持 SSE 流式输出 + 对话模式）."""
    from starlette.responses import JSONResponse

    name = request.path_params["name"]
    payload = (
        await request.json() if request.headers.get("content-type") == "application/json" else {}
    )
    args = payload.get("args", {})
    sync = payload.get("sync", False)
    stream = payload.get("stream", False)
    agent_id = payload.get("agent_id")
    model = payload.get("model")

    # ── 对话模式：站内消息等渠道通过此接口统一执行 ──
    user_message = payload.get("user_message")
    if user_message is not None:
        message_history = payload.get("message_history")
        extra_context = payload.get("extra_context")
        channel = payload.get("channel", "api")
        sender_name = payload.get("sender_name") or _parse_sender_name(extra_context) or "Kai"

        # ── Skills 自动触发（必须在 callback 分支之前执行）──
        from crew.discovery import discover_employees

        discovery = discover_employees(project_dir=ctx.project_dir)
        emp = discovery.get(name)

        enhanced_context = {}
        employee_name = None
        if emp is not None and isinstance(user_message, str):
            try:
                from crew.memory import get_memory_store
                from crew.skills import SkillStore
                from crew.skills_engine import SkillsEngine

                skill_store = SkillStore(project_dir=ctx.project_dir)
                memory_store = get_memory_store(project_dir=ctx.project_dir)
                engine = SkillsEngine(skill_store, memory_store)

                employee_name = emp.character_name or name

                # 检查触发
                triggered = engine.check_triggers(employee_name, user_message, args)
                if triggered:
                    logger.info(
                        "Skills 触发: employee=%s task=%s triggered=%d",
                        employee_name,
                        user_message[:50],
                        len(triggered),
                    )
                    # 执行触发的 skills（按优先级排序）
                    for skill, score in triggered[:3]:
                        try:
                            result = engine.execute_skill(skill, employee_name, {"task": user_message, "channel": channel, **args})
                            if result.get("enhanced_context"):
                                for key, value in result["enhanced_context"].items():
                                    if key in enhanced_context:
                                        if isinstance(enhanced_context[key], list) and isinstance(value, list):
                                            enhanced_context[key].extend(value)
                                        else:
                                            enhanced_context[key] = value
                                    else:
                                        enhanced_context[key] = value
                            engine.record_trigger(
                                skill=skill,
                                employee=employee_name,
                                task=user_message,
                                match_score=score,
                                execution_result=result,
                            )
                        except Exception as skill_exec_error:
                            logger.warning("Skill 执行失败: skill=%s error=%s", skill.name, skill_exec_error)
            except Exception as skills_error:
                logger.warning("Skills 检查失败: %s", skills_error)

        # 将 enhanced_context 注入到 extra_context
        if enhanced_context:
            memories = enhanced_context.get("memories", [])
            if memories:
                memory_text = "【相关历史记忆】\n" + "\n".join(
                    f"- [{m.get('category', '?')}] {m.get('content', '')[:200]}" for m in memories[:5]
                )
                if extra_context:
                    extra_context = memory_text + "\n\n" + extra_context
                else:
                    extra_context = memory_text

                logger.info(
                    "Skills 记忆注入: employee=%s memories=%d extra_context_len=%d",
                    employee_name or name,
                    len(memories),
                    len(extra_context)
                )

        # Phase 3：外部对话输出控制
        from crew.classification import CHANNEL_SOURCE_TYPE, EXTERNAL_OUTPUT_CONTROL_PROMPT

        _source_type = CHANNEL_SOURCE_TYPE.get(channel, "external")
        if _source_type == "external":
            if extra_context:
                extra_context = EXTERNAL_OUTPUT_CONTROL_PROMPT + "\n\n" + extra_context
            else:
                extra_context = EXTERNAL_OUTPUT_CONTROL_PROMPT

        # ── 异步回调模式（频道 @mention）──
        callback_channel_id = payload.get("callback_channel_id")
        if callback_channel_id is not None:
            try:
                callback_channel_id = int(callback_channel_id)
            except (TypeError, ValueError):
                callback_channel_id = None
        callback_sender_id = payload.get("callback_sender_id")
        if callback_sender_id is not None:
            callback_sender_id = str(callback_sender_id)
        callback_parent_id = payload.get("callback_parent_id")
        if callback_parent_id is not None:
            try:
                callback_parent_id = int(callback_parent_id)
            except (TypeError, ValueError):
                callback_parent_id = None

        if callback_channel_id:
            # 立即返回 202，后台处理 + 回调蚁聚
            task = asyncio.create_task(
                _run_and_callback(
                    ctx=ctx,
                    name=name,
                    args=args,
                    agent_id=agent_id,
                    model=model,
                    user_message=user_message,
                    message_history=message_history,
                    extra_context=extra_context,
                    sender_name=sender_name,
                    channel=channel,
                    callback_channel_id=callback_channel_id,
                    callback_sender_id=callback_sender_id,
                    callback_parent_id=callback_parent_id,
                )
            )
            _background_tasks.add(task)
            task.add_done_callback(_task_done_callback)
            return JSONResponse({"status": "accepted"}, status_code=202)

        # 注入额外上下文到 args（和飞书 handler 相同模式）
        if extra_context:
            task = args.get("task", "")
            args["task"] = (extra_context + "\n\n" + task) if task else extra_context

        # ── 轨迹录制 ──
        from contextlib import ExitStack

        from crew.trajectory import TrajectoryCollector

        _exit_stack = ExitStack()
        _task_desc = _extract_task_description(
            user_message if isinstance(user_message, str) else str(user_message)
        )
        _traj_collector = TrajectoryCollector.try_create_for_employee(
            name,
            _task_desc,
            channel=channel,
            project_dir=ctx.project_dir,
        )
        if _traj_collector is not None:
            _exit_stack.enter_context(_traj_collector)

        # 和飞书相同逻辑：闲聊走 fast path，工作消息走 full path
        import time as _time

        from crew.tool_schema import AGENT_TOOLS
        from crew.webhook_feishu import _needs_tools

        # emp 已经在 Skills 代码块中获取
        has_tools = any(t in AGENT_TOOLS for t in (emp.tools or [])) if emp else False
        use_fast_path = (
            has_tools
            and emp is not None
            and emp.fallback_model
            and isinstance(user_message, str)
            and not _needs_tools(user_message)
        )

        # 添加路径选择日志
        logger.info(
            "执行路径: employee=%s use_fast_path=%s has_tools=%s extra_context_len=%d",
            name,
            use_fast_path,
            has_tools,
            len(extra_context) if extra_context else 0
        )

        _t0 = _time.monotonic()
        if use_fast_path:
            from crew.webhook_feishu import _feishu_fast_reply

            result = await _feishu_fast_reply(
                ctx,
                emp,
                user_message,
                message_history=message_history,
                max_visibility="private",
                extra_context=extra_context,
                sender_name=sender_name,
            )
        else:
            import crew.webhook as _wh

            result = await _wh._execute_employee(
                ctx,
                name,
                args,
                agent_id=agent_id,
                model=model,
                user_message=user_message,
                message_history=message_history,
            )

        _elapsed = _time.monotonic() - _t0
        _path = "fast" if use_fast_path else "full"
        _m = result.get("model", "?") if isinstance(result, dict) else "?"
        _in = result.get("input_tokens", 0) if isinstance(result, dict) else 0
        _out = result.get("output_tokens", 0) if isinstance(result, dict) else 0
        logger.info(
            "站内回复 [%s] %.1fs model=%s in=%d out=%d emp=%s msg=%s",
            _path,
            _elapsed,
            _m,
            _in,
            _out,
            name,
            user_message[:40],
        )

        # 完成轨迹录制
        if _traj_collector is not None:
            try:
                _traj_collector.finish(success=True)
            except Exception as _te:
                logger.debug("站内轨迹录制失败: %s", _te)
            finally:
                _exit_stack.close()

        # 记录任务用于成本追踪
        record = ctx.registry.create(
            trigger=channel,
            target_type="employee",
            target_name=name,
            args=args,
        )
        ctx.registry.update(record.task_id, "completed", result=result)

        # 回复后记忆写回（fire-and-forget）
        if isinstance(result, dict) and result.get("output"):
            try:
                from crew.reply_postprocess import push_if_needed

                _reply_text = result["output"].strip()
                _turn_count = len(message_history) if message_history else 1
                push_if_needed(
                    employee=name,
                    reply=_reply_text,
                    turn_count=_turn_count,
                    session_id=f"{channel}-sync-{name}",
                )
            except Exception as _mem_err:
                logger.debug("回复记忆写回失败（不影响响应）: %s", _mem_err)

        return JSONResponse(result)

    # ── 流式模式 ──
    if stream:
        import crew.webhook as _wh

        return await _wh._stream_employee(ctx, name, args, agent_id=agent_id, model=model)

    # ── 原有同步/异步模式 ──
    import crew.webhook as _wh

    return await _wh._dispatch_task(
        ctx,
        trigger="direct",
        target_type="employee",
        target_name=name,
        args=args,
        sync=sync,
        agent_id=agent_id,
        model=model,
    )


async def _handle_run_route(request: Any, ctx: _AppContext) -> Any:
    """直接触发路由模板 — 展开为 delegate_chain 执行."""
    import json as _json

    from starlette.responses import JSONResponse

    from crew.organization import load_organization

    name = request.path_params["name"]
    payload = (
        await request.json() if request.headers.get("content-type") == "application/json" else {}
    )
    task = payload.get("args", {}).get("task", "") or payload.get("task", "")
    overrides = payload.get("overrides", {})
    sync = payload.get("sync", False)

    if not task:
        return JSONResponse({"error": "缺少 task 参数"}, status_code=400)

    org = load_organization(project_dir=ctx.project_dir)
    tmpl = org.routing_templates.get(name)
    if not tmpl:
        available = list(org.routing_templates.keys())
        return JSONResponse(
            {"error": f"未找到路由模板: {name}", "available": available},
            status_code=404,
        )

    # 展开模板为 chain steps（同 _tool_route 逻辑）
    steps: list[dict[str, Any]] = []
    for step in tmpl.steps:
        if step.optional and step.role not in overrides:
            continue
        if step.human:
            continue  # 人工步骤跳过
        emp_name = overrides.get(step.role)
        if not emp_name:
            if step.employee:
                emp_name = step.employee
            elif step.employees:
                emp_name = step.employees[0]
            elif step.team:
                members = org.get_team_members(step.team)
                emp_name = members[0] if members else None
        if not emp_name:
            continue
        step_task = f"[{step.role}] {task}"
        if steps:
            step_task += "\n\n上一步结果: {prev}"
        step_dict: dict[str, Any] = {"employee_name": emp_name, "task": step_task}
        if step.approval:
            step_dict["approval"] = True
        steps.append(step_dict)

    if not steps:
        return JSONResponse({"error": "模板展开后无可执行步骤"}, status_code=400)

    chain_name = " → ".join(s["employee_name"] for s in steps)

    import crew.webhook as _wh

    return await _wh._dispatch_task(
        ctx,
        trigger="direct",
        target_type="chain",
        target_name=chain_name,
        args={"steps_json": _json.dumps(steps, ensure_ascii=False)},
        sync=sync,
    )


async def _handle_agent_run(request: Any, ctx: _AppContext) -> Any:
    """Agent 模式执行员工 — 在 Docker 沙箱中自主完成任务 (SSE 流式)."""
    import json as _json

    from starlette.responses import JSONResponse, StreamingResponse

    name = request.path_params["name"]
    payload = (
        await request.json() if request.headers.get("content-type") == "application/json" else {}
    )
    task_desc = payload.get("task", "")
    if not task_desc:
        return JSONResponse({"error": "缺少 task 参数"}, status_code=400)

    model = payload.get("model", "claude-sonnet-4-5-20250929")
    max_steps = payload.get("max_steps", 30)
    repo = payload.get("repo", "")
    base_commit = payload.get("base_commit", "")
    sandbox_cfg = payload.get("sandbox", {})

    try:
        from agentsandbox import Sandbox, SandboxEnv  # noqa: F401
        from agentsandbox.config import SandboxConfig, TaskConfig
        from knowlyrcore.wrappers import MaxStepsWrapper, RecorderWrapper  # noqa: F401
    except ImportError:
        return JSONResponse(
            {"error": "knowlyr-gym 未安装。请运行: pip install knowlyr-crew[agent]"},
            status_code=501,
        )

    from crew.agent_bridge import create_crew_agent

    async def _event_stream():
        steps_log: list[dict] = []

        def on_step(step_num: int, tool_name: str, params: dict):
            entry = {"type": "step", "step": step_num, "tool": tool_name, "params": params}
            steps_log.append(entry)

        try:
            agent = create_crew_agent(
                name,
                task_desc,
                model=model,
                project_dir=ctx.project_dir,
                on_step=on_step,
            )

            s_config = SandboxConfig(
                image=sandbox_cfg.get("image", "python:3.11-slim"),
                memory_limit=sandbox_cfg.get("memory_limit", "512m"),
                cpu_limit=sandbox_cfg.get("cpu_limit", 1.0),
                network_enabled=sandbox_cfg.get("network_enabled", False),
            )
            t_config = TaskConfig(
                repo_url=repo,
                base_commit=base_commit,
                description=task_desc,
            )
            env = SandboxEnv(config=s_config, task_config=t_config, max_steps=max_steps)

            def _run_loop():
                ts = env.reset()
                while not ts.done:
                    action = agent(ts.observation)
                    ts = env.step(action)
                return ts

            loop = asyncio.get_running_loop()
            final_ts = await loop.run_in_executor(None, _run_loop)

            trajectory = None
            if hasattr(env, "get_trajectory"):
                trajectory = env.get_trajectory()

            env.close()

            for entry in steps_log:
                yield f"data: {_json.dumps(entry, ensure_ascii=False)}\n\n"

            result_data = {
                "type": "result",
                "output": final_ts.observation,
                "terminated": final_ts.terminated,
                "truncated": final_ts.truncated,
                "total_steps": len(steps_log),
            }
            if trajectory:
                result_data["trajectory"] = trajectory
            yield f"data: {_json.dumps(result_data, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.exception("Agent 执行失败: %s", e)
            yield f"data: {_json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


async def _handle_task_status(request: Any, ctx: _AppContext) -> Any:
    """查询任务状态."""
    from starlette.responses import JSONResponse

    task_id = request.path_params["task_id"]
    record = ctx.registry.get(task_id)
    if record is None:
        return JSONResponse({"error": "task not found"}, status_code=404)
    return JSONResponse(record.model_dump(mode="json"))


async def _handle_task_replay(request: Any, ctx: _AppContext) -> Any:
    """重放已完成/失败的任务."""
    from starlette.responses import JSONResponse

    task_id = request.path_params["task_id"]
    record = ctx.registry.get(task_id)
    if record is None:
        return JSONResponse({"error": "task not found"}, status_code=404)

    if record.status not in ("completed", "failed"):
        return JSONResponse({"error": "只能重放已完成或失败的任务"}, status_code=400)

    import crew.webhook as _wh

    return await _wh._dispatch_task(
        ctx,
        trigger="replay",
        target_type=record.target_type,
        target_name=record.target_name,
        args=record.args,
        sync=False,
    )


async def _handle_task_approve(request: Any, ctx: _AppContext) -> Any:
    """POST /api/tasks/{task_id}/approve — 批准或拒绝等待审批的任务."""
    import asyncio

    from starlette.responses import JSONResponse

    task_id = request.path_params["task_id"]
    body = await request.json()
    action = body.get("action", "approve")

    record = ctx.registry.get(task_id)
    if record is None:
        return JSONResponse({"error": "task not found"}, status_code=404)
    if record.status != "awaiting_approval":
        return JSONResponse(
            {"error": f"任务状态为 {record.status}，无法审批"},
            status_code=400,
        )

    if action == "reject":
        reason = body.get("reason", "人工拒绝")
        ctx.registry.update(task_id, "failed", error=reason)
        return JSONResponse({"status": "rejected", "task_id": task_id})

    from crew.webhook_executor import _resume_chain

    task = asyncio.create_task(_resume_chain(ctx, task_id))
    _background_tasks.add(task)
    task.add_done_callback(_task_done_callback)
    return JSONResponse({"status": "approved", "task_id": task_id})


async def _handle_cron_status(request: Any, ctx: _AppContext) -> Any:
    """查询 cron 调度器状态."""
    from starlette.responses import JSONResponse

    if ctx.scheduler is None:
        return JSONResponse({"enabled": False, "schedules": []})
    return JSONResponse(
        {
            "enabled": True,
            "running": ctx.scheduler.running,
            "schedules": ctx.scheduler.get_next_runs(),
        }
    )


async def _handle_cost_summary(request: Any, ctx: _AppContext) -> Any:
    """成本汇总 — GET /api/cost/summary?days=7&employee=xxx&source=work."""
    from starlette.responses import JSONResponse

    from crew.cost import query_cost_summary

    days = _safe_int(request.query_params.get("days", "7"), 7)
    employee = request.query_params.get("employee")
    source = request.query_params.get("source")

    summary = query_cost_summary(ctx.registry, employee=employee, days=days, source=source)
    return JSONResponse(summary)


async def _handle_authority_restore(request: Any, ctx: _AppContext) -> Any:
    """恢复员工权限 — POST /api/employees/{identifier}/authority/restore."""
    from starlette.responses import JSONResponse

    from crew.discovery import discover_employees
    from crew.organization import (
        _authority_overrides,
        _load_overrides,
        _save_overrides,
        load_organization,
    )

    identifier = request.path_params["identifier"]
    result = discover_employees(ctx.project_dir)

    employee = _find_employee(result, identifier)
    if not employee:
        return JSONResponse({"error": "Employee not found"}, status_code=404)

    _load_overrides(ctx.project_dir)
    override = _authority_overrides.pop(employee.name, None)
    if override is None:
        org = load_organization(project_dir=ctx.project_dir)
        current = org.get_authority(employee.name)
        return JSONResponse(
            {
                "ok": True,
                "employee": employee.name,
                "authority": current,
                "message": "无覆盖记录，权限未变更",
            }
        )

    _save_overrides(ctx.project_dir)
    org = load_organization(project_dir=ctx.project_dir)
    restored = org.get_authority(employee.name)

    return JSONResponse(
        {
            "ok": True,
            "employee": employee.name,
            "authority": restored,
            "previous_override": override,
            "message": f"权限已恢复: {override['level']} → {restored}",
        }
    )


async def _handle_org_memories(request: Any, ctx: _AppContext) -> Any:
    """全组织记忆聚合 — GET /api/memory/org?days=7&category=pattern&limit=0.

    改进: 扫描 JSONL 文件发现所有员工（不依赖 discover_employees），
    去掉默认截断（limit=0 表示不限）。
    """
    from datetime import datetime, timedelta

    from starlette.responses import JSONResponse

    from crew.memory import MemoryStore

    days = _safe_int(request.query_params.get("days", "7"), 7)
    category = request.query_params.get("category") or None
    # limit=0 表示不限（向后兼容：客户端可传 limit=50 恢复旧行为）
    limit = _safe_int(request.query_params.get("limit", "0"), 0)

    store = MemoryStore(project_dir=ctx.project_dir)
    # 用 list_employees() 扫描实际 JSONL 文件，不遗漏任何员工
    employee_names = store.list_employees()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat() if days > 0 else ""

    all_memories: list[dict] = []
    stats: dict[str, int] = {
        "decision": 0,
        "estimate": 0,
        "finding": 0,
        "correction": 0,
        "pattern": 0,
    }

    for emp_name in employee_names:
        # limit=0 → 不截断每员工查询
        entries = store.query(emp_name, category=category, limit=0, max_visibility="open")
        for m in entries:
            stats[m.category] = stats.get(m.category, 0) + 1
            if not cutoff or m.created_at >= cutoff:
                all_memories.append(
                    {
                        "employee": m.employee,
                        "category": m.category,
                        "content": m.content,
                        "created_at": m.created_at,
                        "confidence": m.confidence,
                        "tags": m.tags,
                        "shared": m.shared,
                        "trigger_condition": getattr(m, "trigger_condition", ""),
                        "applicability": getattr(m, "applicability", []),
                        "origin_employee": getattr(m, "origin_employee", ""),
                        "verified_count": getattr(m, "verified_count", 0),
                    }
                )

    all_memories.sort(key=lambda x: x["created_at"], reverse=True)

    return JSONResponse(
        {
            "memories": all_memories if limit <= 0 else all_memories[:limit],
            "stats": stats,
            "total": sum(stats.values()),
        }
    )


async def _handle_permission_respond(request: Any, ctx: _AppContext) -> Any:
    """POST /api/permissions/respond — 响应权限请求."""
    from starlette.responses import JSONResponse

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "请求体 JSON 解析失败"}, status_code=400)

    request_id = payload.get("request_id", "")
    approved = payload.get("approved", False)

    if not request_id:
        return JSONResponse({"ok": False, "error": "缺少 request_id 参数"}, status_code=400)

    from crew.permission_request import PermissionManager

    manager = PermissionManager()
    success = manager.respond(request_id, approved)

    if success:
        logger.info("权限响应: request_id=%s, approved=%s", request_id, approved)
        return JSONResponse({"ok": True})
    else:
        return JSONResponse(
            {"ok": False, "error": "请求不存在或已过期"},
            status_code=404,
        )


async def _handle_permission_list(request: Any, ctx: _AppContext) -> Any:
    """GET /api/permissions — 获取待处理的权限请求列表."""
    from starlette.responses import JSONResponse

    from crew.permission_request import PermissionManager

    manager = PermissionManager()
    pending = manager.get_pending_requests()

    return JSONResponse({"ok": True, "requests": pending})


async def _handle_memory_search(request: Any, ctx: _AppContext) -> Any:
    """跨员工语义搜索 — GET /api/memory/search?q=关键词&limit=10&employee=xxx.

    查询参数:
        q (required): 搜索关键词
        limit (optional): 最大返回条数，默认 10
        employee (optional): 限定搜索某个员工（不传则跨全员工搜索）
    """
    from starlette.responses import JSONResponse

    from crew.memory import MemoryStore

    query = request.query_params.get("q", "").strip()
    if not query:
        return JSONResponse({"error": "q is required"}, status_code=400)

    limit = _safe_int(request.query_params.get("limit", "10"), 10)
    if limit <= 0:
        limit = 10
    employee = request.query_params.get("employee", "").strip()

    store = MemoryStore(project_dir=ctx.project_dir)

    try:
        idx = store._get_semantic_index()
        if idx is None:
            return JSONResponse({"error": "semantic index not available"}, status_code=503)

        if employee:
            # 指定员工搜索
            results = idx.search(employee=employee, query=query, limit=limit)
            entries = [
                {"id": r[0], "employee": employee, "content": r[1], "score": round(r[2], 4)}
                for r in results
            ]
        else:
            # 跨员工搜索
            results = idx.search_cross_employee(query=query, limit=limit)
            entries = [
                {"id": r[0], "employee": r[1], "content": r[2], "score": round(r[3], 4)}
                for r in results
            ]

        return JSONResponse({"ok": True, "entries": entries, "total": len(entries)})
    except Exception as e:
        logger.exception("记忆搜索失败")
        return JSONResponse({"error": str(e)}, status_code=500)


async def _handle_trajectory_report(request: Any, ctx: _AppContext) -> Any:
    """接收外部 agent 的轨迹数据 — POST /api/trajectory/report.

    轨迹数据存储到独立的文件系统，不写入永久记忆。
    存储路径：/data/trajectory_archive/{date}/{employee}-{uuid}.jsonl
    """
    from starlette.responses import JSONResponse

    # payload 大小限制 2MB（大轨迹可能有数百步）
    body = await request.body()
    if len(body) > 2 * 1024 * 1024:
        return JSONResponse({"error": "payload too large (max 2MB)"}, status_code=413)

    try:
        payload = await request.json()
    except (ValueError, TypeError):
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    # 兼容多种字段名: employee_name / name / employee
    employee_name = (
        payload.get("employee_name") or payload.get("name") or payload.get("employee") or ""
    )
    # slug → 中文名统一
    try:
        from crew.trajectory import resolve_character_name

        employee_name = resolve_character_name(employee_name, project_dir=ctx.project_dir)
    except Exception:
        pass
    steps = payload.get("steps")
    if not employee_name or not steps:
        return JSONResponse(
            {"error": "missing required fields: employee_name, steps"}, status_code=400
        )

    task_description = payload.get("task_description", "")
    model = payload.get("model", "")
    channel = payload.get("channel", "pull")
    success = payload.get("success", True)

    # ── 完整性校验 ──
    truncated_fields: list[str] = []  # 记录哪些字段被截断
    expected_steps = payload.get("expected_steps")
    if expected_steps is not None and expected_steps != len(steps):
        logger.warning(
            "轨迹步骤数不匹配: expected_steps=%s, actual=%s (employee=%s)",
            expected_steps,
            len(steps),
            employee_name,
        )

    try:
        import json as _json
        import uuid
        from datetime import date as dt_date, datetime

        # ── 独立轨迹存储（不使用 TrajectoryCollector，避免写入 .crew/trajectories） ──
        trajectory_id = f"traj_{uuid.uuid4().hex[:12]}"
        date_str = dt_date.today().isoformat()
        archive_base = Path("/data/trajectory_archive")
        date_dir = archive_base / date_str
        date_dir.mkdir(parents=True, exist_ok=True)

        # 文件名：{employee}-{short_uuid}.jsonl
        trajectory_file = date_dir / f"{employee_name}-{uuid.uuid4().hex[:8]}.jsonl"

        # ── 处理步骤数据 ──
        processed_steps = []
        for s in steps:
            # tool_name 必须存在，否则标记为 unknown
            tool_name = s.get("tool_name") or "unknown"

            # tool_params 规范化：尽量保留原始结构
            raw_params = s.get("tool_params", {})
            if not isinstance(raw_params, dict):
                if isinstance(raw_params, str):
                    # 尝试 JSON 解析
                    try:
                        parsed = _json.loads(raw_params)
                        if isinstance(parsed, dict):
                            raw_params = parsed
                        elif isinstance(parsed, list):
                            raw_params = {"_list": parsed, "_type": "list"}
                        else:
                            raw_params = {"_raw": str(raw_params)[:8000], "_type": "string"}
                    except (ValueError, TypeError):
                        raw_params = {"_raw": str(raw_params)[:8000], "_type": "string"}
                elif isinstance(raw_params, list):
                    raw_params = {"_list": raw_params, "_type": "list"}
                else:
                    raw_params = {
                        "_raw": str(raw_params)[:8000],
                        "_type": type(raw_params).__name__,
                    }

            # 截断 thought / tool_output（8000 字符上限）
            thought_raw = str(s.get("thought", ""))
            if len(thought_raw) > 8000:
                thought_raw = thought_raw[:8000]
                if "thought" not in truncated_fields:
                    truncated_fields.append("thought")
            tool_output_raw = str(s.get("tool_output", ""))
            if len(tool_output_raw) > 8000:
                tool_output_raw = tool_output_raw[:8000]
                if "tool_output" not in truncated_fields:
                    truncated_fields.append("tool_output")

            step_data = {
                "step": s.get("step_id", len(processed_steps) + 1),
                "observation": "",  # 可选，当前状态描述
                "thought": thought_raw,
                "action": {
                    "tool": tool_name,
                    "parameters": raw_params,
                },
                "result": tool_output_raw,
                "success": s.get("tool_exit_code", 0) == 0,
                "timestamp": s.get("timestamp", ""),
            }
            processed_steps.append(step_data)

        # ── 写入完整轨迹对象（行业标准格式）──
        total_steps = len(processed_steps)
        total_tokens = payload.get("total_tokens", 0)
        duration_ms = payload.get("duration_ms", 0)

        trajectory_data = {
            "trajectory_id": trajectory_id,
            "employee": employee_name,
            "task": task_description,
            "model": model,
            "channel": channel,
            "created_at": datetime.now().isoformat(),
            "success": success,
            "metadata": {
                "total_steps": total_steps,
                "total_tokens": total_tokens,
                "duration_ms": duration_ms,
                "truncated_fields": truncated_fields,
            },
            "trajectory": processed_steps,
        }

        with open(trajectory_file, "w", encoding="utf-8") as f:
            f.write(_json.dumps(trajectory_data, ensure_ascii=False) + "\n")

        total_steps = len(processed_steps)

        # ── 更新元数据索引 ──
        index_file = archive_base / "index.json"
        index_data = {}
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    index_data = _json.load(f)
            except Exception:
                pass

        from datetime import datetime

        index_data[trajectory_id] = {
            "trajectory_id": trajectory_id,
            "employee": employee_name,
            "task": task_description[:500],
            "model": model,
            "channel": channel,
            "success": success,
            "total_steps": total_steps,
            "created_at": datetime.now().isoformat(),
            "file_path": str(trajectory_file),
        }

        with open(index_file, "w", encoding="utf-8") as f:
            _json.dump(index_data, f, ensure_ascii=False, indent=2)

        logger.info(
            "轨迹已存储: %s (employee=%s, steps=%d, id=%s)",
            trajectory_file,
            employee_name,
            total_steps,
            trajectory_id,
        )

        resp_data: dict[str, Any] = {
            "ok": True,
            "trajectory_id": trajectory_id,
            "total_steps": total_steps,
            "file_path": str(trajectory_file),
        }
        if truncated_fields:
            resp_data["truncated_fields"] = truncated_fields
        return JSONResponse(resp_data)
    except Exception as e:
        logger.exception("轨迹上报处理失败")
        return JSONResponse({"error": str(e)}, status_code=500)


async def _handle_project_status(request: Any, ctx: _AppContext) -> Any:
    """项目状态概览 — 组织架构 + 成本 + 员工列表."""
    from starlette.responses import JSONResponse

    from crew.cost import (
        calibrate_employee_costs,
        fetch_aiberm_balance,
        fetch_aiberm_billing,
        fetch_moonshot_balance,
        query_cost_summary,
    )
    from crew.discovery import discover_employees
    from crew.organization import get_effective_authority, load_organization

    days = _safe_int(request.query_params.get("days", "7"), 7)
    if days not in (7, 30, 90):
        days = 7

    result = discover_employees(ctx.project_dir)
    org = load_organization(project_dir=ctx.project_dir)
    cost = query_cost_summary(ctx.registry, days=days)

    # 从任意员工配置提取 aiberm API key
    aiberm_billing = None
    for emp in result.employees.values():
        if emp.api_key and "aiberm.com" in (emp.base_url or ""):
            aiberm_billing = await fetch_aiberm_billing(
                api_key=emp.api_key,
                base_url=emp.base_url,
                days=days,
            )
            break

    # aiberm 余额（需要系统访问令牌 + 用户 ID）
    aiberm_balance = None
    aiberm_token = os.environ.get("AIBERM_ACCESS_TOKEN", "")
    aiberm_user_id = os.environ.get("AIBERM_USER_ID", "")
    if aiberm_token:
        aiberm_balance = await fetch_aiberm_balance(
            access_token=aiberm_token,
            user_id=aiberm_user_id,
        )
    if aiberm_billing is None:
        aiberm_billing = {}
    if aiberm_balance:
        aiberm_billing["balance_usd"] = aiberm_balance["balance_usd"]
        aiberm_billing["used_usd"] = aiberm_balance["used_usd"]

    # Moonshot/Kimi 余额 + 7日估算成本
    moonshot_billing = None
    moonshot_key = os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY", "")
    if moonshot_key:
        balance = await fetch_moonshot_balance(api_key=moonshot_key)
        # 从 cost_7d 提取 kimi 相关模型的估算成本
        kimi_cost_usd = 0.0
        kimi_tasks = 0
        for model_name, model_cost in cost.get("by_model", {}).items():
            if model_name.startswith("kimi") or model_name.startswith("moonshot"):
                kimi_cost_usd += model_cost.get("cost_usd", 0)
                kimi_tasks += model_cost.get("tasks", 0)
        moonshot_billing = {
            "balance": balance,
            "cost_7d_usd": round(kimi_cost_usd, 4),
            "cost_7d_tasks": kimi_tasks,
        }

    # 用真实账单校准员工估算成本
    aiberm_real = aiberm_billing.get("total_usd") if aiberm_billing else None
    kimi_real = moonshot_billing.get("cost_7d_usd") if moonshot_billing else None
    cost = calibrate_employee_costs(cost, aiberm_real_usd=aiberm_real, moonshot_real_usd=kimi_real)

    # 预加载所有员工的记忆数量
    from crew.memory import MemoryStore

    store = MemoryStore(project_dir=ctx.project_dir)
    memory_counts: dict[str, int] = {}
    for name in result.employees:
        memory_counts[name] = store.count(name)

    # 聚合员工运行时数据
    from datetime import datetime, timedelta

    cutoff = datetime.now() - timedelta(days=days)
    emp_runtime: dict[str, dict] = {}
    for rec in ctx.registry.snapshot():
        if rec.target_type != "employee":
            continue
        rname = rec.target_name
        if rname not in emp_runtime:
            emp_runtime[rname] = {
                "last_active_at": None,
                "recent_task_count": 0,
                "success_count": 0,
                "fail_count": 0,
                "last_error": None,
                "_last_fail_at": None,
            }
        rt = emp_runtime[rname]
        # 最近活跃时间
        if rec.status == "completed" and rec.completed_at:
            if rt["last_active_at"] is None or rec.completed_at > rt["last_active_at"]:
                rt["last_active_at"] = rec.completed_at
        # 近期任务计数（拆分 success/fail）
        if rec.created_at >= cutoff and rec.status in ("completed", "failed"):
            rt["recent_task_count"] += 1
            if rec.status == "completed":
                rt["success_count"] += 1
            elif rec.status == "failed":
                rt["fail_count"] += 1
        # 最近一次错误
        if rec.status == "failed" and rec.error and rec.completed_at:
            if rt["_last_fail_at"] is None or rec.completed_at > rt["_last_fail_at"]:
                rt["last_error"] = rec.error
                rt["_last_fail_at"] = rec.completed_at

    employees_info = []
    for name, emp in sorted(result.employees.items()):
        team = org.get_team(name)
        authority = get_effective_authority(org, name, project_dir=ctx.project_dir)
        employees_info.append(
            {
                "name": name,
                "character_name": emp.character_name,
                "display_name": emp.display_name,
                "description": emp.description,
                "model": emp.model,
                "agent_id": emp.agent_id,
                "agent_status": emp.agent_status,
                "team": team,
                "authority": authority,
                "memory_count": memory_counts.get(name, 0),
                "last_active_at": emp_runtime[name]["last_active_at"].isoformat()
                if emp_runtime.get(name, {}).get("last_active_at")
                else None,
                "recent_task_count": emp_runtime.get(name, {}).get("recent_task_count", 0),
                "success_count": emp_runtime.get(name, {}).get("success_count", 0),
                "fail_count": emp_runtime.get(name, {}).get("fail_count", 0),
                "last_error": emp_runtime.get(name, {}).get("last_error"),
            }
        )

    # 团队汇总
    teams_info = {}
    for tid, team in org.teams.items():
        teams_info[tid] = {
            "label": team.label,
            "member_count": len(team.members),
            "members": team.members,
        }

    return JSONResponse(
        {
            "total_employees": len(result.employees),
            "teams": teams_info,
            "authority_levels": {
                level: {"label": auth.label, "count": len(auth.members)}
                for level, auth in org.authority.items()
            },
            "cost_7d": cost,
            "aiberm_billing": aiberm_billing,
            "moonshot_billing": moonshot_billing,
            "employees": employees_info,
            "route_categories": {
                cat_id: {"label": cat.label, "icon": cat.icon}
                for cat_id, cat in org.route_categories.items()
            },
            "routing_templates": {
                name: {
                    "label": tmpl.label,
                    "category": tmpl.category,
                    "steps": [
                        {
                            "role": step.role,
                            "employee": step.employee,
                            "employees": step.employees or [],
                            "team": step.team,
                            "description": step.description,
                            "optional": step.optional,
                            "approval": step.approval,
                            "human": step.human,
                            "icon": step.icon,
                            "ci": step.ci,
                        }
                        for step in tmpl.steps
                    ],
                    "extra_flows": [
                        [
                            {
                                "role": s.role,
                                "description": s.description,
                                "icon": s.icon,
                                "ci": s.ci,
                                "employee": s.employee,
                                "employees": s.employees or [],
                                "team": s.team,
                                "optional": s.optional,
                                "approval": s.approval,
                                "human": s.human,
                            }
                            for s in flow
                        ]
                        for flow in tmpl.extra_flows
                    ],
                    "tags": tmpl.tags,
                    "tag_style": tmpl.tag_style,
                    "repo": tmpl.repo,
                    "notes": tmpl.notes,
                    "warnings": tmpl.warnings,
                }
                for name, tmpl in org.routing_templates.items()
            },
        }
    )


async def _handle_audit_trends(request: Any, ctx: _AppContext) -> Any:
    """审计趋势 — GET /api/audit/trends?days=7."""
    from collections import defaultdict
    from datetime import datetime, timedelta

    from starlette.responses import JSONResponse

    days = _safe_int(request.query_params.get("days", "7"), 7)
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_str = cutoff.isoformat()

    # 按日期聚合任务
    daily: dict[str, dict] = defaultdict(
        lambda: {"tasks": 0, "success": 0, "failed": 0, "cost_usd": 0.0}
    )
    total_tasks = 0
    total_success = 0
    total_failed = 0
    total_cost = 0.0

    for t in ctx.registry.snapshot():
        if not t.created_at or t.created_at.isoformat() < cutoff_str:
            continue
        if t.status not in ("completed", "failed"):
            continue

        date_key = t.created_at.strftime("%Y-%m-%d")
        daily[date_key]["tasks"] += 1
        total_tasks += 1

        if t.status == "completed":
            daily[date_key]["success"] += 1
            total_success += 1
        else:
            daily[date_key]["failed"] += 1
            total_failed += 1

        cost = 0.0
        if t.result and isinstance(t.result, dict):
            cost = t.result.get("cost_usd", 0.0) or 0.0
        daily[date_key]["cost_usd"] += cost
        total_cost += cost

    # 补齐没有数据的日期
    daily_list = []
    for i in range(days):
        d = (datetime.now() - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
        entry = daily.get(d, {"tasks": 0, "success": 0, "failed": 0, "cost_usd": 0.0})
        daily_list.append({"date": d, **entry})
        # round cost
        daily_list[-1]["cost_usd"] = round(daily_list[-1]["cost_usd"], 4)

    # 轨迹总数
    traj_count = 0
    if ctx.project_dir:
        traj_file = ctx.project_dir / ".crew" / "trajectories" / "trajectories.jsonl"
        if traj_file.exists():
            with open(traj_file, encoding="utf-8") as _f:
                traj_count = sum(1 for _ in _f)

    return JSONResponse(
        {
            "daily": daily_list,
            "totals": {
                "trajectories": traj_count,
                "tasks": total_tasks,
                "success": total_success,
                "failed": total_failed,
                "total_cost_usd": round(total_cost, 4),
            },
        }
    )


async def _handle_chat(request: Any, ctx: _AppContext) -> Any:
    """统一对话接口 — 供蚁聚、飞书、外部渠道统一调用.

    POST /api/chat
    {
        "employee_id": "moya",
        "message": "用户输入文本",
        "channel": "antgather_dm",
        "sender_id": "user_123",
        "max_visibility": "internal",   // 可选，默认 internal
        "stream": false,                // 可选，默认 false
        "context_only": false           // 可选，仅返回 prompt+记忆
    }
    """
    import json as _json

    from starlette.responses import JSONResponse, StreamingResponse

    # ── 解析请求体 ──
    try:
        payload: dict[str, Any] = (
            await request.json()
            if request.headers.get("content-type", "").startswith("application/json")
            else {}
        )
    except Exception:
        return JSONResponse({"ok": False, "error": "请求体 JSON 解析失败"}, status_code=400)

    # ── 必填字段校验 ──
    employee_id = payload.get("employee_id", "")
    message = payload.get("message", "")
    channel = payload.get("channel", "")
    sender_id = payload.get("sender_id", "")

    _required_fields = [
        ("employee_id", employee_id),
        ("message", message),
        ("channel", channel),
        ("sender_id", sender_id),
    ]
    missing = [f for f, v in _required_fields if not v]
    if missing:
        return JSONResponse(
            {"ok": False, "error": f"缺少必填字段: {', '.join(missing)}"},
            status_code=400,
        )

    # ── 可选字段 ──
    max_visibility: str = payload.get("max_visibility", "internal")
    stream: bool = bool(payload.get("stream", False))
    context_only: bool = bool(payload.get("context_only", False))
    message_history: list[dict[str, Any]] | None = payload.get("message_history")
    model: str | None = payload.get("model")
    attachments: list[dict[str, Any]] | None = payload.get("attachments")

    # 异步回调参数（用于长任务）
    callback_channel_id: int | None = payload.get("callback_channel_id")
    callback_sender_id: str | None = payload.get("callback_sender_id")
    async_mode: bool = bool(payload.get("async", False))

    # 如果指定了异步模式，立即返回并在后台执行
    if async_mode and callback_channel_id:
        import asyncio

        # 启动后台任务
        asyncio.create_task(
            _run_and_callback(
                ctx=ctx,
                name=employee_id,
                args={},
                agent_id=None,
                model=model,
                user_message=message,
                message_history=message_history,
                extra_context=None,
                sender_name=sender_id,
                channel=channel,
                callback_channel_id=callback_channel_id,
                callback_sender_id=callback_sender_id,
                callback_parent_id=None,
            )
        )

        return JSONResponse(
            {
                "ok": True,
                "reply": "正在处理您的请求，完成后会通知您...",
                "async": True,
                "employee_id": employee_id,
            }
        )

    # ── SG Bridge 主通道尝试（非 stream / 非 context_only 时） ──
    _sg_reply: str | None = None
    if not stream and not context_only:
        # 优先尝试 SG API Bridge（直接用 Claude API + 本地工具执行）
        try:
            from crew.sg_api_bridge import SGAPIBridgeError, sg_api_dispatch

            _sg_reply = await sg_api_dispatch(
                message,
                ctx=ctx,
                project_dir=ctx.project_dir,
                employee_name=employee_id,
                message_history=message_history,
                push_event_fn=None,  # TODO: 支持流式输出
                channel=channel,
            )
            logger.info("SG API Bridge 成功: reply_len=%d", len(_sg_reply))
        except SGAPIBridgeError as _sg_api_err:
            logger.info("SG API Bridge fallback: %s → 尝试 SSH Bridge", _sg_api_err)
            _sg_reply = None
        except Exception as _sg_api_exc:
            logger.warning("SG API Bridge 意外异常: %s → 尝试 SSH Bridge", _sg_api_exc)
            _sg_reply = None

        # Fallback: SSH Bridge（两阶段权限确认）
        if _sg_reply is None:
            try:
                from crew.sg_bridge import sg_dispatch

                # 定义权限回调函数
                async def permission_callback(operations: list[dict]) -> bool:
                    """请求用户权限确认."""
                    from crew.permission_request import PermissionManager

                    manager = PermissionManager()

                    # 构建权限请求参数
                    tool_names = [op["tool"] for op in operations]
                    tool_params = {
                        "operations": operations,
                        "message": message[:200],
                    }

                    # 请求权限（会推送事件到前端）
                    approved = await manager.request_permission(
                        tool_name=f"SG执行: {', '.join(tool_names)}",
                        tool_params=tool_params,
                        timeout=60.0,
                    )

                    return approved

                _sg_reply = await sg_dispatch(
                    message,
                    project_dir=ctx.project_dir,
                    employee_name=employee_id,
                    message_history=message_history,
                    permission_callback=permission_callback,
                )
            except Exception as _sg_exc:
                logger.info("SG Bridge fallback (/api/chat): %s → 走 crew 引擎", _sg_exc)
                _sg_reply = None

    if _sg_reply is not None:
        result: dict[str, Any] = {
            "ok": True,
            "reply": _sg_reply,
            "output": _sg_reply,
            "tokens_used": 0,
            "employee_id": employee_id,
            "path": "sg",
        }
    else:
        # ── Fallback: 检查员工是否有工具，决定调用路径 ──
        from crew.discovery import discover_employees
        from crew.engine import CrewEngine
        from crew.exceptions import EmployeeNotFoundError
        from crew.tool_schema import AGENT_TOOLS

        discovery = discover_employees(project_dir=ctx.project_dir)
        emp = discovery.get(employee_id)
        if emp is None:
            return JSONResponse(
                {"ok": False, "error": f"员工不存在: {employee_id}"},
                status_code=404,
            )

        # 如果员工有 agent tools，使用带工具的执行路径
        has_agent_tools = any(t in AGENT_TOOLS for t in (emp.tools or []))

        try:
            if has_agent_tools and not context_only:
                # 使用带工具的执行路径
                import crew.webhook_executor as _wh_exec

                exec_result = await _wh_exec._execute_employee_with_tools(
                    ctx,
                    employee_id,
                    {},  # args
                    agent_id=None,
                    model=model,
                    user_message=message,
                    message_history=message_history,
                    sender_id=sender_id,
                    attachments=attachments,
                )
                result = {
                    "reply": exec_result.get("output", ""),
                    "employee_id": employee_id,
                    "memory_updated": False,
                    "tokens_used": exec_result.get("input_tokens", 0)
                    + exec_result.get("output_tokens", 0),
                    "latency_ms": 0,
                }
            else:
                # 使用简单的 chat 路径（无工具）
                engine = CrewEngine(project_dir=ctx.project_dir)
                chat_result = await engine.chat(
                    employee_id=employee_id,
                    message=message,
                    channel=channel,
                    sender_id=sender_id,
                    max_visibility=max_visibility,
                    stream=stream,
                    context_only=context_only,
                    message_history=message_history,
                    model=model,
                )

                # stream=True 时 engine.chat 返回 AsyncIterator
                if stream and not context_only:
                    # 真流式：直接消费 AsyncIterator
                    async def _real_sse_generator():
                        async for chunk in chat_result:
                            chunk_data = _json.dumps(chunk, ensure_ascii=False)
                            yield f"data: {chunk_data}\n\n"

                    return StreamingResponse(
                        _real_sse_generator(),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "X-Accel-Buffering": "no",
                        },
                    )
                else:
                    result = chat_result

        except EmployeeNotFoundError:
            return JSONResponse(
                {"ok": False, "error": f"员工不存在: {employee_id}"},
                status_code=404,
            )
        except ValueError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
        except Exception as e:
            logger.exception("chat() 异常: emp=%s channel=%s", employee_id, channel)
            return JSONResponse(
                {"ok": False, "error": f"内部错误: {e}"},
                status_code=500,
            )

    # ── 流式响应（SSE）— SG Bridge 或 agent tools 路径的兼容模式 ──
    if stream and not context_only:
        reply_text: str = result.get("reply", "")

        async def _sse_generator():
            # 兼容模式：将完整回复拆成字符逐个推送（模拟流式体验）
            for ch in reply_text:
                chunk_data = _json.dumps({"delta": ch, "done": False}, ensure_ascii=False)
                yield f"data: {chunk_data}\n\n"
            done_data = _json.dumps(
                {
                    "done": True,
                    "employee_id": employee_id,
                    "memory_updated": result.get("memory_updated", False),
                    "tokens_used": result.get("tokens_used", 0),
                    "latency_ms": result.get("latency_ms", 0),
                },
                ensure_ascii=False,
            )
            yield f"data: {done_data}\n\n"

        return StreamingResponse(
            _sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ── 非流式响应 ──
    return JSONResponse(result)


# ── KV 存储端点 ──────────────────────────────────────────────────

# key 合法性正则：只允许字母数字、连字符、下划线、点、斜杠
_KV_KEY_RE = _re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_./-]*$")


def _validate_kv_key(key: str) -> str | None:
    """校验 KV key，返回错误消息或 None（合法）."""
    if not key:
        return "key is required"
    if ".." in key:
        return "key must not contain '..'"
    if key.startswith("/"):
        return "key must not start with '/'"
    if not _KV_KEY_RE.match(key):
        return "key contains invalid characters (allowed: a-z A-Z 0-9 _ - . /)"
    return None


def _kv_base_dir(ctx: _AppContext) -> Path:
    """返回 KV 存储的根目录."""
    return (ctx.project_dir or Path(".")) / ".crew" / "kv"


async def _handle_kv_put(request: Any, ctx: _AppContext) -> Any:
    """KV 写入 — PUT /api/kv/{key:path}.

    支持两种 Content-Type:
    - text/plain / application/octet-stream: body 就是文件内容
    - application/json: {"content": "文件内容"}
    """
    from starlette.responses import JSONResponse

    key = request.path_params.get("key", "")
    err = _validate_kv_key(key)
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)

    # 读取内容
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            payload = await request.json()
        except (ValueError, TypeError):
            return JSONResponse({"ok": False, "error": "Invalid JSON"}, status_code=400)
        content = payload.get("content")
        if content is None:
            return JSONResponse(
                {"ok": False, "error": "'content' field is required in JSON body"},
                status_code=400,
            )
        raw_bytes = content.encode("utf-8") if isinstance(content, str) else content
    else:
        raw_bytes = await request.body()
        if not raw_bytes:
            return JSONResponse({"ok": False, "error": "empty body"}, status_code=400)

    # 写入文件
    base_dir = _kv_base_dir(ctx)
    file_path = base_dir / key
    # 二次校验 resolved path 确实在 base_dir 内
    try:
        file_path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        return JSONResponse({"ok": False, "error": "path traversal detected"}, status_code=400)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(raw_bytes)

    return JSONResponse({"ok": True, "key": key, "size": len(raw_bytes)})


async def _handle_kv_get(request: Any, ctx: _AppContext) -> Any:
    """KV 读取 — GET /api/kv/{key:path}."""
    from starlette.responses import JSONResponse, Response

    key = request.path_params.get("key", "")
    err = _validate_kv_key(key)
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)

    base_dir = _kv_base_dir(ctx)
    file_path = base_dir / key
    try:
        file_path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        return JSONResponse({"ok": False, "error": "path traversal detected"}, status_code=400)

    if not file_path.is_file():
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)

    content = file_path.read_bytes()
    return Response(content=content, media_type="text/plain; charset=utf-8")


async def _handle_kv_list(request: Any, ctx: _AppContext) -> Any:
    """KV 列表 — GET /api/kv/ (可选 ?prefix=...)."""
    from starlette.responses import JSONResponse

    prefix = request.query_params.get("prefix", "")

    # prefix 也做安全校验（但允许空字符串）
    if prefix:
        if ".." in prefix:
            return JSONResponse(
                {"ok": False, "error": "prefix must not contain '..'"}, status_code=400
            )
        if prefix.startswith("/"):
            return JSONResponse(
                {"ok": False, "error": "prefix must not start with '/'"}, status_code=400
            )

    base_dir = _kv_base_dir(ctx)
    scan_dir = base_dir / prefix if prefix else base_dir

    # 安全检查
    try:
        scan_dir.resolve().relative_to(base_dir.resolve())
    except ValueError:
        return JSONResponse({"ok": False, "error": "path traversal detected"}, status_code=400)

    keys: list[str] = []
    if scan_dir.is_dir():
        for p in sorted(scan_dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(base_dir)
                keys.append(str(rel))

    return JSONResponse({"ok": True, "keys": keys})


# ── Pipeline / Discussion / Meeting / Decision / WorkLog / Permission API ──


async def _handle_pipeline_list(request: Any, ctx: _AppContext) -> Any:
    """列出所有流水线 — GET /api/pipelines."""
    from starlette.responses import JSONResponse

    from crew.pipeline import discover_pipelines, load_pipeline

    pipelines = discover_pipelines(project_dir=ctx.project_dir)
    data = []
    for pname, ppath in pipelines.items():
        try:
            pl = load_pipeline(ppath)
            data.append(
                {
                    "name": pname,
                    "description": pl.description,
                    "steps": len(pl.steps),
                }
            )
        except Exception:
            data.append({"name": pname, "error": "parse_failed"})
    return JSONResponse({"items": data})


async def _handle_discussion_list(request: Any, ctx: _AppContext) -> Any:
    """列出所有讨论会 — GET /api/discussions."""
    from starlette.responses import JSONResponse

    from crew.discussion import discover_discussions, load_discussion

    discussions = discover_discussions(project_dir=ctx.project_dir)
    data = []
    for dname, dpath in discussions.items():
        try:
            d = load_discussion(dpath)
            rounds_count = d.rounds if isinstance(d.rounds, int) else len(d.rounds)
            data.append(
                {
                    "name": dname,
                    "description": d.description,
                    "participants": [p.employee for p in d.participants],
                    "rounds": rounds_count,
                }
            )
        except Exception:
            data.append({"name": dname, "error": "parse_failed"})
    return JSONResponse({"items": data})


async def _handle_discussion_plan(request: Any, ctx: _AppContext) -> Any:
    """获取编排模式讨论计划 — GET /api/discussions/{name}/plan."""
    import json as _json

    from starlette.responses import JSONResponse

    from crew.discussion import (
        discover_discussions,
        load_discussion,
        render_discussion_plan,
        validate_discussion,
    )

    name = request.path_params["name"]
    discussions = discover_discussions(project_dir=ctx.project_dir)
    if name not in discussions:
        return JSONResponse({"error": f"not found: {name}"}, status_code=404)

    discussion = load_discussion(discussions[name])
    errors = validate_discussion(discussion, project_dir=ctx.project_dir)
    if errors:
        return JSONResponse({"error": "; ".join(errors)}, status_code=400)

    d_args: dict[str, str] = {}
    args_raw = request.query_params.get("args")
    if args_raw:
        try:
            d_args = _json.loads(args_raw)
        except (ValueError, TypeError):
            return JSONResponse({"error": "invalid args JSON"}, status_code=400)

    agent_id = request.query_params.get("agent_id")
    smart_context = request.query_params.get("smart_context", "true").lower() != "false"

    try:
        plan = render_discussion_plan(
            discussion,
            initial_args=d_args,
            agent_id=agent_id,
            smart_context=smart_context,
            project_dir=ctx.project_dir,
        )
        return JSONResponse(_json.loads(plan.model_dump_json()))
    except Exception as exc:
        logger.exception("render_discussion_plan failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _handle_discussion_prompt(request: Any, ctx: _AppContext) -> Any:
    """获取非编排模式讨论 prompt — GET /api/discussions/{name}/prompt."""
    import json as _json

    from starlette.responses import JSONResponse

    from crew.discussion import (
        discover_discussions,
        load_discussion,
        render_discussion,
        validate_discussion,
    )

    name = request.path_params["name"]
    discussions = discover_discussions(project_dir=ctx.project_dir)
    if name not in discussions:
        return JSONResponse({"error": f"not found: {name}"}, status_code=404)

    discussion = load_discussion(discussions[name])
    errors = validate_discussion(discussion, project_dir=ctx.project_dir)
    if errors:
        return JSONResponse({"error": "; ".join(errors)}, status_code=400)

    d_args: dict[str, str] = {}
    args_raw = request.query_params.get("args")
    if args_raw:
        try:
            d_args = _json.loads(args_raw)
        except (ValueError, TypeError):
            return JSONResponse({"error": "invalid args JSON"}, status_code=400)

    agent_id = request.query_params.get("agent_id")
    smart_context = request.query_params.get("smart_context", "true").lower() != "false"

    try:
        prompt = render_discussion(
            discussion,
            initial_args=d_args,
            agent_id=agent_id,
            smart_context=smart_context,
            project_dir=ctx.project_dir,
        )
        return JSONResponse({"prompt": prompt})
    except Exception as exc:
        logger.exception("render_discussion failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _handle_meeting_list(request: Any, ctx: _AppContext) -> Any:
    """列出会议历史 — GET /api/meetings."""
    from starlette.responses import JSONResponse

    from crew.meeting_log import MeetingLogger

    ml = MeetingLogger(project_dir=ctx.project_dir)
    limit = _safe_int(request.query_params.get("limit"), default=20)
    keyword = request.query_params.get("keyword") or None
    records = ml.list(limit=limit, keyword=keyword)
    return JSONResponse({"items": [r.model_dump() for r in records]})


async def _handle_meeting_detail(request: Any, ctx: _AppContext) -> Any:
    """获取会议详情 — GET /api/meetings/{meeting_id}."""
    from starlette.responses import JSONResponse

    from crew.meeting_log import MeetingLogger

    ml = MeetingLogger(project_dir=ctx.project_dir)
    meeting_id = request.path_params["meeting_id"]
    result = ml.get(meeting_id)
    if result is None:
        return JSONResponse({"error": f"not found: {meeting_id}"}, status_code=404)
    record, content = result
    data = {**record.model_dump(), "content": content}
    return JSONResponse(data)


async def _handle_decision_track(request: Any, ctx: _AppContext) -> Any:
    """追踪决策 — POST /api/decisions/track."""
    from starlette.responses import JSONResponse

    from crew.evaluation import EvaluationEngine

    try:
        payload = await request.json()
    except (ValueError, TypeError):
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    employee = payload.get("employee")
    category = payload.get("category")
    content = payload.get("content")
    if not employee or not category or not content:
        return JSONResponse({"error": "employee, category, content are required"}, status_code=400)

    engine = EvaluationEngine(project_dir=ctx.project_dir)
    try:
        decision = engine.track(
            employee=employee,
            category=category,
            content=content,
            expected_outcome=payload.get("expected_outcome", ""),
            meeting_id=payload.get("meeting_id", ""),
        )
        return JSONResponse(decision.model_dump())
    except Exception as exc:
        logger.exception("decision track failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _handle_decision_evaluate(request: Any, ctx: _AppContext) -> Any:
    """评估决策 — POST /api/decisions/{decision_id}/evaluate."""
    from starlette.responses import JSONResponse

    from crew.evaluation import EvaluationEngine

    decision_id = request.path_params["decision_id"]

    try:
        payload = await request.json()
    except (ValueError, TypeError):
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    actual_outcome = payload.get("actual_outcome")
    if not actual_outcome:
        return JSONResponse({"error": "actual_outcome is required"}, status_code=400)

    engine = EvaluationEngine(project_dir=ctx.project_dir)
    try:
        decision = engine.evaluate(
            decision_id=decision_id,
            actual_outcome=actual_outcome,
            evaluation=payload.get("evaluation", ""),
        )
        if decision is None:
            return JSONResponse({"error": f"not found: {decision_id}"}, status_code=404)
        return JSONResponse(decision.model_dump())
    except Exception as exc:
        logger.exception("decision evaluate failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _handle_work_log(request: Any, ctx: _AppContext) -> Any:
    """获取工作日志 — GET /api/work-log."""
    from starlette.responses import JSONResponse

    from crew.log import WorkLogger

    wl = WorkLogger(project_dir=ctx.project_dir)
    employee_name = request.query_params.get("employee_name") or None
    limit = _safe_int(request.query_params.get("limit"), default=10)
    sessions = wl.list_sessions(employee_name=employee_name, limit=limit)
    return JSONResponse({"items": sessions})


async def _handle_permission_matrix(request: Any, ctx: _AppContext) -> Any:
    """获取员工权限矩阵 — GET /api/permission-matrix."""
    from starlette.responses import JSONResponse

    from crew.discovery import discover_employees
    from crew.tool_schema import resolve_effective_tools

    result = discover_employees(project_dir=ctx.project_dir)
    emp_name = request.query_params.get("employee") or None

    employees = list(result.employees.values())
    if emp_name:
        emp = result.get(emp_name)
        if emp is None:
            return JSONResponse({"error": f"not found: {emp_name}"}, status_code=404)
        employees = [emp]

    matrix = []
    for emp in employees:
        effective = resolve_effective_tools(emp)
        entry = {
            "name": emp.name,
            "display_name": emp.effective_display_name,
            "tools_declared": len(emp.tools),
            "tools_effective": len(effective),
            "permissions": (emp.permissions.model_dump(mode="json") if emp.permissions else None),
            "effective_tools": sorted(effective),
        }
        matrix.append(entry)
    return JSONResponse({"items": matrix})


async def _handle_wiki_file_delete(request: Any, ctx: _AppContext) -> Any:
    """Wiki 文件删除 — DELETE /api/wiki/files/{file_id}.

    路径参数:
        file_id: Wiki 文件 ID

    返回:
        {"ok": true, "deleted_file_id": N} 或 404
    """
    from starlette.responses import JSONResponse

    file_id_str = request.path_params.get("file_id", "")
    if not file_id_str:
        return JSONResponse({"error": "file_id is required"}, status_code=400)

    try:
        file_id = int(file_id_str)
    except (ValueError, TypeError):
        return JSONResponse({"error": f"invalid file_id: {file_id_str}"}, status_code=400)

    # 调用 Wiki 后端删除文件
    wiki_api_url = os.environ.get("WIKI_API_URL", "").rstrip("/")
    wiki_api_token = os.environ.get("WIKI_API_TOKEN", "")
    if not wiki_api_url or not wiki_api_token:
        return JSONResponse(
            {"error": "Wiki API 未配置，请设置 WIKI_API_URL 和 WIKI_API_TOKEN 环境变量"},
            status_code=500,
        )

    import httpx

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.delete(
                f"{wiki_api_url}/api/wiki/files/{file_id}",
                headers={"X-Wiki-Token": wiki_api_token},
            )
            if resp.status_code == 404:
                return JSONResponse({"error": f"file not found: {file_id}"}, status_code=404)
            resp.raise_for_status()
            return JSONResponse({"ok": True, "deleted_file_id": file_id})
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return JSONResponse({"error": f"file not found: {file_id}"}, status_code=404)
        logger.exception("Wiki 文件删除失败: file_id=%s", file_id)
        return JSONResponse({"error": f"删除失败: {exc}"}, status_code=500)
    except Exception as exc:
        logger.exception("Wiki 文件删除失败: file_id=%s", file_id)
        return JSONResponse({"error": f"删除失败: {exc}"}, status_code=500)


async def _handle_wiki_spaces_list(request: Any, ctx: _AppContext) -> Any:
    """Wiki 空间列表 — GET /api/wiki/spaces.

    返回所有可用的 Wiki 空间，包含 slug、名称、ID 等信息。
    """
    from starlette.responses import JSONResponse

    wiki_api_url = os.environ.get("WIKI_API_URL", "").rstrip("/")
    wiki_admin_token = os.environ.get("WIKI_ADMIN_TOKEN", "") or os.environ.get(
        "ANTGATHER_API_TOKEN", ""
    )
    if not wiki_api_url or not wiki_admin_token:
        return JSONResponse(
            {"error": "Wiki Admin API 未配置，请设置 WIKI_API_URL 和 WIKI_ADMIN_TOKEN（或 ANTGATHER_API_TOKEN）环境变量"},
            status_code=500,
        )

    import httpx

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{wiki_api_url}/api/admin/wiki/spaces",
                headers={"Authorization": f"Bearer {wiki_admin_token}"},
            )
            resp.raise_for_status()
            data = resp.json()
            spaces = data.get("spaces", [])
            return JSONResponse({"spaces": spaces, "total": len(spaces)})
    except Exception as exc:
        logger.exception("Wiki 空间列表查询失败")
        return JSONResponse({"error": f"查询失败: {exc}"}, status_code=500)


# ── 配置存储 API ──


def _resolve_employee_name(identifier: str, ctx: _AppContext) -> str:
    """将 slug 或 agent_id 转换为中文名（用于配置查询）."""
    from crew.discovery import discover_employees

    # 如果已经是中文名，直接返回
    if any("\u4e00" <= c <= "\u9fff" for c in identifier):
        return identifier

    # 尝试从员工列表查找
    if ctx.project_dir:
        result = discover_employees(ctx.project_dir)
        emp = _find_employee(result, identifier)
        if emp:
            return emp.character_name

    # 找不到，返回原值
    return identifier


async def _handle_soul_get(request: Any, ctx: _AppContext) -> Any:
    """获取员工灵魂配置 — GET /api/souls/{employee_name}."""
    from starlette.responses import JSONResponse

    from crew.config_store import get_soul

    identifier = request.path_params.get("employee_name", "")
    if not identifier:
        return JSONResponse({"error": "employee_name is required"}, status_code=400)

    # 将 slug 转换为中文名
    employee_name = _resolve_employee_name(identifier, ctx)

    result = get_soul(employee_name)
    if not result:
        return JSONResponse({"error": f"soul not found: {identifier}"}, status_code=404)

    return JSONResponse(result)


async def _handle_soul_update(request: Any, ctx: _AppContext) -> Any:
    """更新员工灵魂配置 — PUT /api/souls/{employee_name}."""
    from starlette.responses import JSONResponse

    from crew.config_store import update_soul

    identifier = request.path_params.get("employee_name", "")
    if not identifier:
        return JSONResponse({"error": "employee_name is required"}, status_code=400)

    # 将 slug 转换为中文名
    employee_name = _resolve_employee_name(identifier, ctx)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    content = body.get("content", "")
    if not content:
        return JSONResponse({"error": "content is required"}, status_code=400)

    updated_by = body.get("updated_by", "")
    metadata = body.get("metadata")

    try:
        result = update_soul(employee_name, content, updated_by, metadata)
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("更新 soul 失败: employee=%s", employee_name)
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _handle_soul_list(request: Any, ctx: _AppContext) -> Any:
    """列出所有员工灵魂配置 — GET /api/souls."""
    from starlette.responses import JSONResponse

    from crew.config_store import list_souls

    try:
        items = list_souls()
        return JSONResponse({"items": items})
    except Exception as exc:
        logger.exception("列出 souls 失败")
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _handle_discussion_get(request: Any, ctx: _AppContext) -> Any:
    """获取讨论会配置 — GET /api/config/discussions/{name}."""
    from starlette.responses import JSONResponse

    from crew.config_store import get_discussion

    name = request.path_params.get("name", "")
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)

    result = get_discussion(name)
    if not result:
        return JSONResponse({"error": f"discussion not found: {name}"}, status_code=404)

    return JSONResponse(result)


async def _handle_discussion_create(request: Any, ctx: _AppContext) -> Any:
    """创建讨论会配置 — POST /api/config/discussions."""
    from starlette.responses import JSONResponse

    from crew.config_store import create_discussion

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    name = body.get("name", "")
    yaml_content = body.get("yaml_content", "")
    if not name or not yaml_content:
        return JSONResponse({"error": "name and yaml_content are required"}, status_code=400)

    description = body.get("description", "")
    metadata = body.get("metadata")

    try:
        result = create_discussion(name, yaml_content, description, metadata)
        return JSONResponse(result, status_code=201)
    except Exception as exc:
        logger.exception("创建 discussion 失败: name=%s", name)
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _handle_discussion_update(request: Any, ctx: _AppContext) -> Any:
    """更新讨论会配置 — PUT /api/config/discussions/{name}."""
    from starlette.responses import JSONResponse

    from crew.config_store import update_discussion

    name = request.path_params.get("name", "")
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    yaml_content = body.get("yaml_content", "")
    if not yaml_content:
        return JSONResponse({"error": "yaml_content is required"}, status_code=400)

    description = body.get("description")
    metadata = body.get("metadata")

    try:
        result = update_discussion(name, yaml_content, description, metadata)
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("更新 discussion 失败: name=%s", name)
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _handle_discussion_list_config(request: Any, ctx: _AppContext) -> Any:
    """列出所有讨论会配置 — GET /api/config/discussions."""
    from starlette.responses import JSONResponse

    from crew.config_store import get_discussion, list_discussions

    try:
        items = list_discussions()
        # 增强：添加 participants 和 rounds 信息
        for item in items:
            try:
                full_config = get_discussion(item["name"])
                if full_config and full_config.get("yaml_content"):
                    import yaml

                    parsed = yaml.safe_load(full_config["yaml_content"])
                    if parsed:
                        if "participants" in parsed:
                            item["participants"] = parsed["participants"]
                        if "rounds" in parsed:
                            item["rounds"] = parsed["rounds"]
            except Exception:
                pass  # 解析失败不影响列表
        return JSONResponse({"items": items})
    except Exception as exc:
        logger.exception("列出 discussions 失败")
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _handle_pipeline_get_config(request: Any, ctx: _AppContext) -> Any:
    """获取流水线配置 — GET /api/config/pipelines/{name}."""
    from starlette.responses import JSONResponse

    from crew.config_store import get_pipeline

    name = request.path_params.get("name", "")
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)

    result = get_pipeline(name)
    if not result:
        return JSONResponse({"error": f"pipeline not found: {name}"}, status_code=404)

    return JSONResponse(result)


async def _handle_pipeline_create_config(request: Any, ctx: _AppContext) -> Any:
    """创建流水线配置 — POST /api/config/pipelines."""
    from starlette.responses import JSONResponse

    from crew.config_store import create_pipeline

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    name = body.get("name", "")
    yaml_content = body.get("yaml_content", "")
    if not name or not yaml_content:
        return JSONResponse({"error": "name and yaml_content are required"}, status_code=400)

    description = body.get("description", "")
    metadata = body.get("metadata")

    try:
        result = create_pipeline(name, yaml_content, description, metadata)
        return JSONResponse(result, status_code=201)
    except Exception as exc:
        logger.exception("创建 pipeline 失败: name=%s", name)
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _handle_pipeline_update_config(request: Any, ctx: _AppContext) -> Any:
    """更新流水线配置 — PUT /api/config/pipelines/{name}."""
    from starlette.responses import JSONResponse

    from crew.config_store import update_pipeline

    name = request.path_params.get("name", "")
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    yaml_content = body.get("yaml_content", "")
    if not yaml_content:
        return JSONResponse({"error": "yaml_content is required"}, status_code=400)

    description = body.get("description")
    metadata = body.get("metadata")

    try:
        result = update_pipeline(name, yaml_content, description, metadata)
        return JSONResponse(result)
    except Exception as exc:
        logger.exception("更新 pipeline 失败: name=%s", name)
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _handle_pipeline_list_config(request: Any, ctx: _AppContext) -> Any:
    """列出所有流水线配置 — GET /api/config/pipelines."""
    from starlette.responses import JSONResponse

    from crew.config_store import get_pipeline, list_pipelines

    try:
        items = list_pipelines()
        # 增强：添加 steps 信息
        for item in items:
            try:
                full_config = get_pipeline(item["name"])
                if full_config and full_config.get("yaml_content"):
                    import yaml

                    parsed = yaml.safe_load(full_config["yaml_content"])
                    if parsed and "steps" in parsed:
                        item["steps"] = len(parsed["steps"])
            except Exception:
                pass  # 解析失败不影响列表
        return JSONResponse({"items": items})
    except Exception as exc:
        logger.exception("列出 pipelines 失败")
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _handle_evaluate_scan(request: Any, ctx: _AppContext) -> Any:
    """手动触发过期决策扫描 — POST /api/evaluate/scan."""
    from starlette.responses import JSONResponse

    from crew.cron_evaluate import format_scan_report, scan_overdue_decisions

    try:
        results = await scan_overdue_decisions()
        report = format_scan_report(results)

        return JSONResponse({
            "auto_evaluated": len(results.get("auto_evaluated", [])),
            "reminders": len(results.get("reminders", [])),
            "expired": len(results.get("expired", [])),
            "report": report,
            "details": results,
        })
    except Exception as exc:
        logger.exception("evaluate scan failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)
