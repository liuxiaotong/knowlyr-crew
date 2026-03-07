"""配置管理处理器（soul、讨论会、流水线配置等）."""

from __future__ import annotations

from typing import Any

from crew.webhook_handlers._common import (
    JSONResponse,
    _AppContext,
    _error_response,
    _find_employee,
    _require_admin_token,
    _safe_limit,
    _tenant_data_dir,
    _tenant_id_for_config,
    _tenant_id_for_store,
    logger,
    os,
)


async def _handle_pipeline_list(request: Any, ctx: _AppContext) -> Any:
    """列出所有流水线 — GET /api/pipelines."""

    # 安全加固: 流水线列表含执行逻辑，仅 admin 可读
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

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

    # 安全加固: 讨论配置含流程/角色/策略，仅 admin 可读
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

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


    # 安全加固: 讨论计划含完整编排流程，仅 admin 可读
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

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
        return _error_response("内部错误", 500)


async def _handle_discussion_prompt(request: Any, ctx: _AppContext) -> Any:
    """获取非编排模式讨论 prompt — GET /api/discussions/{name}/prompt."""
    import json as _json


    # 安全加固: 讨论 prompt 含完整指令，仅 admin 可读
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

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
        return _error_response("内部错误", 500)


async def _handle_meeting_list(request: Any, ctx: _AppContext) -> Any:
    """列出会议历史 — GET /api/meetings."""

    # 安全加固: 会议日志含内部讨论/路线规划，仅 admin 可读
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.meeting_log import MeetingLogger

    _mtg_dir = _tenant_data_dir(request, "meetings")
    ml = MeetingLogger(meetings_dir=_mtg_dir, project_dir=ctx.project_dir)
    limit = _safe_limit(request.query_params.get("limit"), default=20)
    keyword = request.query_params.get("keyword") or None
    records = ml.list(limit=limit, keyword=keyword)
    return JSONResponse({"items": [r.model_dump() for r in records]})


async def _handle_meeting_detail(request: Any, ctx: _AppContext) -> Any:
    """获取会议详情 — GET /api/meetings/{meeting_id}."""

    # 安全加固: 会议详情含完整 prompt/参数，仅 admin 可读
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.meeting_log import MeetingLogger

    _mtg_dir = _tenant_data_dir(request, "meetings")
    ml = MeetingLogger(meetings_dir=_mtg_dir, project_dir=ctx.project_dir)
    meeting_id = request.path_params["meeting_id"]
    result = ml.get(meeting_id)
    if result is None:
        return JSONResponse({"error": f"not found: {meeting_id}"}, status_code=404)
    record, content = result
    data = {**record.model_dump(), "content": content}
    return JSONResponse(data)


async def _handle_decision_track(request: Any, ctx: _AppContext) -> Any:
    """追踪决策 — POST /api/decisions/track."""

    # 安全加固: 决策追踪涉及写入记忆，需要 admin 权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

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

    engine = EvaluationEngine(project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request))
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
        return _error_response("内部错误", 500)


async def _handle_decision_evaluate(request: Any, ctx: _AppContext) -> Any:
    """评估决策 — POST /api/decisions/{decision_id}/evaluate."""

    # 安全加固: 决策评估写入纠正记忆，需要 admin 权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.evaluation import EvaluationEngine

    decision_id = request.path_params["decision_id"]

    try:
        payload = await request.json()
    except (ValueError, TypeError):
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    actual_outcome = payload.get("actual_outcome")
    if not actual_outcome:
        return JSONResponse({"error": "actual_outcome is required"}, status_code=400)

    engine = EvaluationEngine(project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request))
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
        return _error_response("内部错误", 500)


async def _handle_work_log(request: Any, ctx: _AppContext) -> Any:
    """获取工作日志 — GET /api/work-log."""

    # 权限校验：工作日志包含所有 session，需要 admin 权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.log import WorkLogger

    wl = WorkLogger(project_dir=ctx.project_dir)
    employee_name = request.query_params.get("employee_name") or None
    limit = _safe_limit(request.query_params.get("limit"), default=10)
    sessions = wl.list_sessions(employee_name=employee_name, limit=limit)
    return JSONResponse({"items": sessions})


async def _handle_permission_matrix(request: Any, ctx: _AppContext) -> Any:
    """获取员工权限矩阵 — GET /api/permission-matrix."""

    # 安全加固: 权限矩阵暴露所有 agent 工具清单和权限策略，仅 admin 可读
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.discovery import discover_employees
    from crew.tool_schema import resolve_effective_tools

    result = discover_employees(project_dir=ctx.project_dir, tenant_id=_tenant_id_for_config(request))
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

    # P2: Wiki 文件删除需要管理员权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

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

    wiki_api_url = os.environ.get("WIKI_API_URL", "").rstrip("/")
    wiki_admin_token = os.environ.get("WIKI_ADMIN_TOKEN", "") or os.environ.get(
        "ANTGATHER_API_TOKEN", ""
    )
    if not wiki_api_url or not wiki_admin_token:
        return JSONResponse(
            {
                "error": "Wiki Admin API 未配置，请设置 WIKI_API_URL 和 WIKI_ADMIN_TOKEN（或 ANTGATHER_API_TOKEN）环境变量"
            },
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


def _resolve_employee_name(identifier: str, ctx: _AppContext, *, tenant_id: str | None = None) -> str:
    """将 slug 或 agent_id 转换为中文名（用于配置查询）."""
    from crew.discovery import discover_employees

    # 如果已经是中文名，直接返回
    if any("\u4e00" <= c <= "\u9fff" for c in identifier):
        return identifier

    # 尝试从员工列表查找
    if ctx.project_dir:
        result = discover_employees(ctx.project_dir, tenant_id=tenant_id)
        emp = _find_employee(result, identifier)
        if emp:
            return emp.character_name

    # 找不到，返回原值
    return identifier


async def _handle_soul_get(request: Any, ctx: _AppContext) -> Any:
    """获取员工灵魂配置 — GET /api/souls/{employee_name}."""

    # 安全加固: soul 包含完整 prompt/指令，仅 admin 可读
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.config_store import get_soul

    identifier = request.path_params.get("employee_name", "")
    if not identifier:
        return JSONResponse({"error": "employee_name is required"}, status_code=400)

    # 将 slug 转换为中文名
    employee_name = _resolve_employee_name(identifier, ctx, tenant_id=_tenant_id_for_config(request))

    result = get_soul(employee_name, tenant_id=_tenant_id_for_config(request))
    if not result:
        return JSONResponse({"error": f"soul not found: {identifier}"}, status_code=404)

    return JSONResponse(result)


async def _handle_soul_update(request: Any, ctx: _AppContext) -> Any:
    """更新员工灵魂配置 — PUT /api/souls/{employee_name}."""

    # 管理员权限校验
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.config_store import update_soul

    identifier = request.path_params.get("employee_name", "")
    if not identifier:
        return JSONResponse({"error": "employee_name is required"}, status_code=400)

    # 将 slug 转换为中文名
    employee_name = _resolve_employee_name(identifier, ctx, tenant_id=_tenant_id_for_config(request))

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
        result = update_soul(employee_name, content, updated_by, metadata, tenant_id=_tenant_id_for_config(request))
        return JSONResponse(result)
    except Exception:
        logger.exception("更新 soul 失败: employee=%s", employee_name)
        return _error_response("内部错误", 500)


async def _handle_soul_list(request: Any, ctx: _AppContext) -> Any:
    """列出所有员工灵魂配置 — GET /api/souls."""

    # 安全加固: soul 列表包含所有员工 prompt，仅 admin 可读
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.config_store import list_souls

    try:
        items = list_souls(tenant_id=_tenant_id_for_config(request))
        return JSONResponse({"items": items})
    except Exception:
        logger.exception("列出 souls 失败")
        return _error_response("内部错误", 500)


async def _handle_discussion_get(request: Any, ctx: _AppContext) -> Any:
    """获取讨论会配置 — GET /api/config/discussions/{name}."""

    from crew.config_store import get_discussion

    name = request.path_params.get("name", "")
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)

    result = get_discussion(name, tenant_id=_tenant_id_for_config(request))
    if not result:
        return JSONResponse({"error": f"discussion not found: {name}"}, status_code=404)

    return JSONResponse(result)


async def _handle_discussion_create(request: Any, ctx: _AppContext) -> Any:
    """创建讨论会配置 — POST /api/config/discussions."""

    # P2: 讨论会创建需要管理员权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

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
        result = create_discussion(name, yaml_content, description, metadata, tenant_id=_tenant_id_for_config(request))
        return JSONResponse(result, status_code=201)
    except Exception:
        logger.exception("创建 discussion 失败: name=%s", name)
        return _error_response("内部错误", 500)


async def _handle_discussion_update(request: Any, ctx: _AppContext) -> Any:
    """更新讨论会配置 — PUT /api/config/discussions/{name}."""

    # P2: 讨论会更新需要管理员权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

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
        result = update_discussion(name, yaml_content, description, metadata, tenant_id=_tenant_id_for_config(request))
        return JSONResponse(result)
    except Exception:
        logger.exception("更新 discussion 失败: name=%s", name)
        return _error_response("内部错误", 500)


async def _handle_discussion_list_config(request: Any, ctx: _AppContext) -> Any:
    """列出所有讨论会配置 — GET /api/config/discussions."""

    from crew.config_store import get_discussion, list_discussions

    try:
        _cfg_tid = _tenant_id_for_config(request)
        items = list_discussions(tenant_id=_cfg_tid)
        # 增强：添加 participants 和 rounds 信息
        for item in items:
            try:
                full_config = get_discussion(item["name"], tenant_id=_cfg_tid)
                if full_config and full_config.get("yaml_content"):
                    import yaml

                    parsed = yaml.safe_load(full_config["yaml_content"])
                    if parsed:
                        if "participants" in parsed:
                            item["participants"] = parsed["participants"]
                        if "rounds" in parsed:
                            item["rounds"] = parsed["rounds"]
            except Exception:
                logger.debug("讨论会配置解析失败: %s", item.get("name"), exc_info=True)
        return JSONResponse({"items": items})
    except Exception:
        logger.exception("列出 discussions 失败")
        return _error_response("内部错误", 500)


async def _handle_pipeline_get_config(request: Any, ctx: _AppContext) -> Any:
    """获取流水线配置 — GET /api/config/pipelines/{name}."""

    # 安全加固: 流水线含执行逻辑/工具/模型配置，仅 admin 可读
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.config_store import get_pipeline

    name = request.path_params.get("name", "")
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)

    result = get_pipeline(name, tenant_id=_tenant_id_for_config(request))
    if not result:
        return JSONResponse({"error": f"pipeline not found: {name}"}, status_code=404)

    return JSONResponse(result)


async def _handle_pipeline_create_config(request: Any, ctx: _AppContext) -> Any:
    """创建流水线配置 — POST /api/config/pipelines."""

    # 安全加固: 创建流水线是写操作，仅 admin 可执行
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

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
        result = create_pipeline(name, yaml_content, description, metadata, tenant_id=_tenant_id_for_config(request))
        return JSONResponse(result, status_code=201)
    except Exception:
        logger.exception("创建 pipeline 失败: name=%s", name)
        return _error_response("内部错误", 500)


async def _handle_pipeline_update_config(request: Any, ctx: _AppContext) -> Any:
    """更新流水线配置 — PUT /api/config/pipelines/{name}."""

    # 安全加固: 更新流水线是写操作，仅 admin 可执行
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

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
        result = update_pipeline(name, yaml_content, description, metadata, tenant_id=_tenant_id_for_config(request))
        return JSONResponse(result)
    except Exception:
        logger.exception("更新 pipeline 失败: name=%s", name)
        return _error_response("内部错误", 500)


async def _handle_pipeline_list_config(request: Any, ctx: _AppContext) -> Any:
    """列出所有流水线配置 — GET /api/config/pipelines."""

    # 安全加固: 流水线列表含执行逻辑，仅 admin 可读
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.config_store import get_pipeline, list_pipelines

    try:
        _cfg_tid = _tenant_id_for_config(request)
        items = list_pipelines(tenant_id=_cfg_tid)
        # 增强：添加 steps 信息
        for item in items:
            try:
                full_config = get_pipeline(item["name"], tenant_id=_cfg_tid)
                if full_config and full_config.get("yaml_content"):
                    import yaml

                    parsed = yaml.safe_load(full_config["yaml_content"])
                    if parsed and "steps" in parsed:
                        item["steps"] = len(parsed["steps"])
            except Exception:
                logger.debug("流水线配置解析失败: %s", item.get("name"), exc_info=True)
        return JSONResponse({"items": items})
    except Exception:
        logger.exception("列出 pipelines 失败")
        return _error_response("内部错误", 500)


async def _handle_evaluate_scan(request: Any, ctx: _AppContext) -> Any:
    """手动触发过期决策扫描 — POST /api/evaluate/scan."""

    # 安全加固: 扫描会批量写入记忆和评估结论，仅 admin 可触发
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.cron_evaluate import format_scan_report, scan_overdue_decisions

    try:
        results = await scan_overdue_decisions()
        report = format_scan_report(results)

        return JSONResponse(
            {
                "auto_evaluated": len(results.get("auto_evaluated", [])),
                "reminders": len(results.get("reminders", [])),
                "expired": len(results.get("expired", [])),
                "report": report,
                "details": results,
            }
        )
    except Exception as exc:
        logger.exception("evaluate scan failed: %s", exc)
        return _error_response("内部错误", 500)
