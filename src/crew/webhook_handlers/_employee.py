"""员工管理处理器."""

from __future__ import annotations

import asyncio
import re as _re
from typing import Any

from crew.webhook_handlers._common import (
    _EMPLOYEE_UPDATABLE_FIELDS,
    JSONResponse,
    Path,
    _AppContext,
    _background_tasks,
    _error_response,
    _find_employee,
    _ok_response,
    _require_admin_token,
    _safe_int,
    _task_done_callback,
    _tenant_id_for_config,
    _tenant_id_for_store,
    _write_yaml_field,
    get_current_tenant,
    get_memory_store,
    logger,
)


async def _handle_employee_prompt(request: Any, ctx: _AppContext) -> Any:
    """返回员工配置和渲染后的 system_prompt."""

    # 安全加固: prompt 含完整指令/工具/成本数据，仅 admin 可读
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.discovery import discover_employees
    from crew.engine import CrewEngine
    from crew.tool_schema import employee_tools_to_schemas

    identifier = request.path_params["identifier"]
    result = discover_employees(ctx.project_dir, tenant_id=_tenant_id_for_config(request))

    employee = _find_employee(result, identifier)
    if not employee:
        return JSONResponse({"error": "Employee not found"}, status_code=404)

    engine = CrewEngine(ctx.project_dir, tenant_id=_tenant_id_for_store(request))
    system_prompt = engine.prompt(employee)
    tool_schemas, _ = employee_tools_to_schemas(employee.tools, defer=False)

    # 从 YAML 或 DB 读取 Employee model 之外的字段（bio, temperature 等）
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
    else:
        # DB 模式：source_path 为空，从 employees 表读取额外字段
        try:
            from crew.config_store import get_employee_from_db

            db_row = get_employee_from_db(employee.name, tenant_id=_tenant_id_for_config(request))
            if db_row:
                bio = db_row.get("bio", "")
                temperature = db_row.get("temperature")
                max_tokens = db_row.get("max_tokens")
        except Exception:
            logger.debug("读取员工 DB 额外字段失败", exc_info=True)

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

    # fields 过滤：?fields=system_prompt 可省略 tool_schemas 等大字段
    fields_param = request.query_params.get("fields", "")
    requested_fields = {f.strip() for f in fields_param.split(",") if f.strip()} if fields_param else set()

    resp: dict[str, Any] = {
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
        "system_prompt": system_prompt,
        "agent_id": employee.agent_id,
        "team": team,
        "authority": authority,
        "cost_7d": cost_summary,
        "kpi": employee.kpi,
        "auto_memory": employee.auto_memory,
    }
    # tool_schemas 可能很大，仅在未指定 fields 或显式请求时返回
    if not requested_fields or "tool_schemas" in requested_fields:
        resp["tool_schemas"] = tool_schemas

    return JSONResponse(resp)


async def _handle_model_tiers(request: Any, ctx: _AppContext) -> Any:
    """返回可用的模型档位列表（不含密钥和内部 URL）."""

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
    """返回所有员工基本信息列表（供外部服务获取员工花名册）.

    Bearer token: 返回展示安全字段（蚁聚社区等下游需要）
    Admin token: 返回全量字段（含 model/tags 等运营情报）
    """

    is_admin = _require_admin_token(request) is None

    from crew.discovery import discover_employees

    result = discover_employees(ctx.project_dir, tenant_id=_tenant_id_for_config(request))
    items = []
    for emp in result.employees.values():
        avatar_url = f"/static/avatars/{emp.agent_id}.webp" if emp.agent_id else None
        # 基础字段：所有 Bearer token 持有者可见
        item: dict[str, Any] = {
            "name": emp.name,
            "character_name": emp.character_name,
            "display_name": emp.display_name,
            "agent_id": emp.agent_id,
            "agent_status": emp.agent_status,
            "avatar_url": avatar_url,
        }
        # 敏感字段：仅 admin 可见（模型/标签/描述含内部运营情报）
        if is_admin:
            item["description"] = emp.description
            item["model"] = emp.model
            item["model_tier"] = emp.model_tier
            item["tags"] = emp.tags
        items.append(item)

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

    # P2: 员工创建需要管理员权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return _error_response(admin_err, 403)

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
            tenant_id=_tenant_id_for_config(request),
        )
    except ValueError as e:
        return _error_response(str(e), 409)
    except Exception:
        logger.exception("创建员工失败")
        return _error_response("创建员工失败", 500)

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


async def _handle_employee_copy(request: Any, ctx: _AppContext) -> Any:
    """复制员工到当前租户 — POST /api/employees/copy."""
    from crew.config_store import copy_employee_to_tenant
    from crew.tenant import DEFAULT_ADMIN_TENANT_ID

    # 1. 解析请求
    try:
        body = await request.json()
    except Exception:
        return _error_response("invalid JSON body", 400)

    source_name = (body.get("source_name") or "").strip()
    if not source_name:
        return _error_response("source_name is required", 400)

    source_tenant_id = (body.get("source_tenant_id") or DEFAULT_ADMIN_TENANT_ID).strip()
    new_name = (body.get("new_name") or "").strip() or None
    new_character_name = (body.get("new_character_name") or "").strip() or None
    customizations = body.get("customizations") or {}

    # 2. 验证 new_name 格式（如果提供）
    if new_name and not _re.match(r"^[a-z0-9-]+$", new_name):
        return _error_response("new_name must match [a-z0-9-]", 400)

    # 3. 权限检查：非 admin 只能从 admin 租户复制
    tenant = get_current_tenant(request)
    if not tenant.is_admin and source_tenant_id != DEFAULT_ADMIN_TENANT_ID:
        return _error_response("non-admin tenants can only copy from admin tenant", 403)

    # 4. 执行复制
    try:
        result = copy_employee_to_tenant(
            source_name=source_name,
            target_tenant_id=tenant.tenant_id,
            source_tenant_id=source_tenant_id,
            new_name=new_name,
            new_character_name=new_character_name,
            customizations=customizations,
        )
    except ValueError as e:
        err_msg = str(e)
        if "not found" in err_msg:
            return _error_response(err_msg, 404)
        if "already exists" in err_msg:
            return _error_response(err_msg, 409)
        return _error_response(err_msg, 400)
    except Exception:
        logger.exception("复制员工失败")
        return _error_response("复制员工失败", 500)

    # 5. 构建响应
    metadata = result.get("metadata")
    if isinstance(metadata, str):
        import json as _json
        try:
            metadata = _json.loads(metadata)
        except Exception:
            metadata = {}

    return _ok_response(
        {
            "employee": {
                "name": result.get("name"),
                "tenant_id": result.get("tenant_id"),
                "character_name": result.get("character_name"),
                "agent_id": result.get("agent_id"),
                "source_copied_from": (metadata or {}).get("source_copied_from", ""),
            },
        },
        status_code=201,
    )


async def _handle_team_agents(request: Any, ctx: _AppContext) -> Any:
    """返回 active 状态的 AI 员工展示数据（供官网 about 页面 + 蚁聚社区使用）.

    此端点在 middleware skip_paths 中，允许匿名访问。
    安全策略：匿名只返回展示安全字段，admin 返回全量（含 domains/expertise）。
    """
    import yaml as _yaml

    from crew.discovery import discover_employees

    # 判断是否 admin（不阻断，仅决定返回字段范围）
    # 注意：匿名访问（通过 skip_paths）时 tenant middleware 会分配默认租户，
    # _tenant_id_for_config 返回该默认值，这是预期行为（展示所有公开员工）。
    is_admin = _require_admin_token(request) is None

    result = discover_employees(ctx.project_dir, tenant_id=_tenant_id_for_config(request))
    agents = []
    for emp in result.employees.values():
        if emp.agent_status != "active":
            continue

        # 从 employee.yaml 或 DB 读取 bio 和 domains（Employee 模型未收录这两个字段）
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
                logger.debug("读取 employee.yaml bio/domains 失败", exc_info=True)
        elif not emp.source_path:
            # DB 模式：从 employees 表读取
            try:
                from crew.config_store import get_employee_from_db

                db_row = get_employee_from_db(emp.name, tenant_id=_tenant_id_for_config(request))
                if db_row:
                    bio = db_row.get("bio", "")
                    raw_domains = db_row.get("domains") or []
                    domains = raw_domains if isinstance(raw_domains, list) else []
            except Exception:
                logger.debug("读取员工 DB bio/domains 失败", exc_info=True)

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

        # 公开安全字段（官网 + 蚁聚社区展示用）
        agent_data: dict[str, Any] = {
            "id": public_id,
            "nickname": emp.character_name,
            "title": emp.display_name,
            "avatar_url": avatar_url,
            "is_agent": True,
            "staff_badge": "集识光年",
            "bio": bio,
        }

        # 敏感字段仅 admin 可见（内部能力标签、领域覆盖）
        if is_admin:
            agent_data["expertise"] = emp.tags
            agent_data["domains"] = domains

        agents.append(agent_data)

    # 按 id 升序排列，None 排最后
    agents.sort(key=lambda a: (a["id"] is None, a["id"] or ""))

    return JSONResponse(agents)


async def _handle_employee_get(request: Any, ctx: _AppContext) -> Any:
    """返回单个员工的完整定义（对应 MCP get_employee）."""

    from crew.discovery import discover_employees

    identifier = request.path_params["identifier"]
    result = discover_employees(ctx.project_dir, tenant_id=_tenant_id_for_config(request))

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

    from crew.discovery import discover_employees

    identifier = request.path_params["identifier"]
    result = discover_employees(ctx.project_dir, tenant_id=_tenant_id_for_config(request))

    employee = _find_employee(result, identifier)
    if not employee:
        return JSONResponse({"error": "Employee not found"}, status_code=404)

    limit = _safe_int(request.query_params.get("memory_limit", "10"), 10)
    _ALLOWED_SORT_FIELDS = {"created_at", "importance", "updated_at"}
    sort_by = request.query_params.get("sort_by", "created_at")
    if sort_by not in _ALLOWED_SORT_FIELDS:
        sort_by = "created_at"
    min_importance = _safe_int(request.query_params.get("min_importance", "0"), 0)
    max_tokens = _safe_int(request.query_params.get("max_tokens", "0"), 0)  # 0=不限

    # 读取 soul.md（文件系统或 DB）
    soul = ""
    if employee.source_path:
        soul_path = employee.source_path / "soul.md"
        if soul_path.exists():
            soul = soul_path.read_text(encoding="utf-8")
    else:
        # DB 模式：从 employees 表读取 soul_content
        try:
            from crew.config_store import get_employee_from_db

            db_row = get_employee_from_db(employee.name, tenant_id=_tenant_id_for_config(request))
            if db_row:
                soul = db_row.get("soul_content", "")
        except Exception:
            logger.debug("读取员工 DB soul_content 失败", exc_info=True)

    # 读取最近记忆（API 只返回公开记忆，过滤 private）
    store = get_memory_store(project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request))
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

    # 租户上下文（一次获取，后续复用）
    _state_tenant = get_current_tenant(request)

    # 读取最近笔记 — 仅 admin 租户可见，防止内部 notes 泄露
    # 注意：notes_dir 路径硬编码为 .crew/notes，这是设计选择——笔记是项目级资源，
    # 不做租户隔离（与记忆系统不同），仅通过 admin 权限控制可见性。
    recent_notes: list[dict] = []
    if _state_tenant.is_admin:
        notes_dir = (ctx.project_dir or Path.cwd()) / ".crew" / "notes"
        if notes_dir.is_dir():
            note_files = sorted(notes_dir.glob("*.md"), reverse=True)
            for nf in note_files[:5]:
                text = nf.read_text(encoding="utf-8")
                # 跳过 private 笔记
                if "visibility: private" in text:
                    continue
                if employee.character_name in text or employee.name in text:
                    recent_notes.append({"filename": nf.name, "content": text[:500]})

    # soul 完整内容仅对 admin 租户返回，非 admin 只返回摘要
    if _state_tenant.is_admin:
        soul_field = soul
    else:
        # 只返回基本信息，不暴露完整 prompt/内部 API 说明
        soul_field = f"{employee.character_name} ({employee.display_name})" if soul else ""

    response_data = {
        "name": employee.name,
        "character_name": employee.character_name,
        "display_name": employee.display_name,
        "agent_status": employee.agent_status,
        "soul": soul_field,
        "memories": memory_list,
        "notes": recent_notes,
    }

    # Token 预算截断：粗估 1 token ≈ 3 中文字符 / 4 英文字符
    if max_tokens > 0:
        import json as _json

        budget_chars = max_tokens * 3  # 保守估计
        # soul 优先保留，记忆按顺序截断（必须用 soul_field，不能用原始 soul，否则非 admin 会泄露完整 soul）
        soul_chars = len(soul_field)
        remaining = budget_chars - soul_chars
        if remaining < 200:
            # soul 已经超预算，截断 soul
            response_data["soul"] = soul_field[:budget_chars] + "\n...(truncated)"
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

    # P2: 员工修改需要管理员权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.discovery import discover_employees

    identifier = request.path_params["identifier"]
    result = discover_employees(ctx.project_dir, tenant_id=_tenant_id_for_config(request))

    employee = _find_employee(result, identifier)
    if not employee:
        return JSONResponse({"error": "Employee not found"}, status_code=404)

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

    if employee.source_path:
        # 文件系统模式：写回 employee.yaml
        try:
            _write_yaml_field(employee.source_path, updates)
        except OSError:
            logger.exception("更新 employee.yaml 失败: %s", identifier)
            return _error_response("文件写入失败", 500)
    else:
        # DB 模式：更新 employees 表
        try:
            from crew.config_store import get_employee_from_db, upsert_employee_to_db

            tenant = _tenant_id_for_config(request)
            db_row = get_employee_from_db(employee.name, tenant_id=tenant)
            if db_row:
                db_row.update(updates)
                upsert_employee_to_db(db_row, tenant_id=tenant)
            else:
                return JSONResponse({"error": "Employee not found in DB"}, status_code=404)
        except Exception:
            logger.exception("更新 employees 表失败: %s", identifier)
            return _error_response("数据库更新失败", 500)

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


    # 管理员权限校验
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.config_store import get_soul
    from crew.database import get_connection, is_pg
    from crew.discovery import discover_employees

    identifier = request.path_params["identifier"]
    result = discover_employees(ctx.project_dir, cache_ttl=0, tenant_id=_tenant_id_for_config(request))

    employee = _find_employee(result, identifier)

    # 如果员工发现机制找不到，尝试直接从数据库查询
    character_name = None
    if employee:
        character_name = employee.character_name
    else:
        # 尝试用 identifier 作为 character_name 查询数据库
        if get_soul(identifier, tenant_id=_tenant_id_for_config(request)):
            character_name = identifier
        else:
            return JSONResponse({"error": "Employee not found"}, status_code=404)

    deleted_items = []

    # 1. 删除本地文件（如果存在）
    if employee and employee.source_path:
        source = employee.source_path
        try:
            if source.is_symlink():
                # 符号链接安全：不用 rmtree，只移除链接本身
                source.unlink()
                deleted_items.append(f"symlink: {source}")
            elif source.is_dir():
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
                    "DELETE FROM employee_souls WHERE employee_name = %s AND tenant_id = %s",
                    (character_name, _tenant_id_for_config(request)),
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

