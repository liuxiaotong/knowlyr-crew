"""管理与运维处理器."""

from __future__ import annotations

import asyncio
import re as _re
from pathlib import Path
from typing import Any

from crew.webhook_handlers._common import (
    JSONResponse,
    _AppContext,
    _background_tasks,
    _error_response,
    _find_employee,
    _require_admin_token,
    _safe_int,
    _safe_limit,
    _task_done_callback,
    _tenant_base_dir,
    _tenant_id_for_config,
    _tenant_id_for_store,
    get_memory_store,
    logger,
    os,
)


async def _handle_task_status(request: Any, ctx: _AppContext) -> Any:
    """查询任务状态."""

    # 安全加固: 任务含 args/outputs/token 统计，owner 为空时默认需要 admin
    admin_err = _require_admin_token(request)

    task_id = request.path_params["task_id"]
    record = ctx.registry.get(task_id)
    if record is None:
        return JSONResponse({"error": "task not found"}, status_code=404)

    # 归属校验：有 owner 时检查 user_id，无 owner 时必须 admin
    user_id = request.query_params.get("user_id") or request.headers.get("x-user-id", "")
    if record.owner:
        if not user_id:
            return JSONResponse({"error": "user_id required to access owned task"}, status_code=401)
        if record.owner != user_id:
            return JSONResponse(
                {"error": "forbidden: task does not belong to this user"}, status_code=403
            )
    else:
        # 无 owner 的任务只有 admin 可查看
        if admin_err:
            return JSONResponse({"error": admin_err}, status_code=403)

    return JSONResponse(record.model_dump(mode="json"))


async def _handle_task_replay(request: Any, ctx: _AppContext) -> Any:
    """重放已完成/失败的任务."""

    # 安全加固: replay 重新执行任务，owner 为空时需要 admin
    admin_err = _require_admin_token(request)

    task_id = request.path_params["task_id"]
    record = ctx.registry.get(task_id)
    if record is None:
        return JSONResponse({"error": "task not found"}, status_code=404)

    # 归属校验：有 owner 时检查 user_id，无 owner 时必须 admin
    body = await request.json() if "application/json" in (request.headers.get("content-type") or "") else {}
    user_id = body.get("user_id") or request.headers.get("x-user-id", "")
    if record.owner:
        if not user_id:
            return JSONResponse({"error": "user_id required to replay owned task"}, status_code=401)
        if record.owner != user_id:
            return JSONResponse(
                {"error": "forbidden: task does not belong to this user"}, status_code=403
            )
    else:
        if admin_err:
            return JSONResponse({"error": admin_err}, status_code=403)

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
        owner=record.owner,
        tenant_id=_tenant_id_for_store(request),
    )


async def _handle_task_approve(request: Any, ctx: _AppContext) -> Any:
    """POST /api/tasks/{task_id}/approve — 批准或拒绝等待审批的任务."""
    import asyncio


    # 安全加固: owner 为空时默认需要 admin
    admin_err = _require_admin_token(request)
    task_id = request.path_params["task_id"]
    body = await request.json()
    action = body.get("action", "approve")

    record = ctx.registry.get(task_id)
    if record is None:
        return JSONResponse({"error": "task not found"}, status_code=404)

    # 归属校验
    user_id = body.get("user_id") or request.headers.get("x-user-id", "")
    if record.owner:
        if not user_id:
            return JSONResponse(
                {"error": "user_id required to approve owned task"}, status_code=401
            )
        if record.owner != user_id:
            return JSONResponse(
                {"error": "forbidden: task does not belong to this user"}, status_code=403
            )
    else:
        # 无 owner 的任务只有 admin 可审批
        if admin_err:
            return JSONResponse({"error": admin_err}, status_code=403)

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

    # 安全加固: 成本数据含全员工 token 用量和计费信息，仅 admin 可访问
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.cost import query_cost_summary

    days = _safe_int(request.query_params.get("days", "7"), 7)
    employee = request.query_params.get("employee")
    source = request.query_params.get("source")

    summary = query_cost_summary(ctx.registry, employee=employee, days=days, source=source)
    return JSONResponse(summary)


async def _handle_authority_restore(request: Any, ctx: _AppContext) -> Any:
    """恢复员工权限 — POST /api/employees/{identifier}/authority/restore."""

    # P2: 权限恢复需要管理员权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.discovery import discover_employees
    from crew.organization import (
        _authority_overrides,
        _load_overrides,
        _save_overrides,
        load_organization,
    )

    identifier = request.path_params["identifier"]
    result = discover_employees(ctx.project_dir, tenant_id=_tenant_id_for_config(request))

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


    # 安全加固: 组织级记忆聚合暴露全员工知识库，仅 admin 可访问
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)


    days = _safe_int(request.query_params.get("days", "7"), 7)
    category = request.query_params.get("category") or None
    # limit 上限 2000，默认 500（防止全量加载 OOM）
    limit = min(int(request.query_params.get("limit", "500")), 2000) if request.query_params.get("limit") else 500

    store = get_memory_store(project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request))
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
        # 每员工限制 limit 条，防止 OOM
        entries = store.query(emp_name, category=category, limit=limit, max_visibility="open")
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

    # 安全加固: 权限审批是高危操作（可授权工具调用），仅 admin 可操作
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "请求体 JSON 解析失败"}, status_code=400)

    request_id = payload.get("request_id", "")
    approved = payload.get("approved", False)
    user_id = payload.get("user_id", "")

    if not request_id:
        return JSONResponse({"ok": False, "error": "缺少 request_id 参数"}, status_code=400)

    from crew.permission_request import PermissionManager

    manager = PermissionManager()

    # 鉴权：校验发起响应的用户是否是 target_user_id
    perm_req = manager.get_request(request_id)
    if perm_req is None:
        return JSONResponse(
            {"ok": False, "error": "请求不存在或已过期"},
            status_code=404,
        )
    # 当权限请求指定了目标用户时，必须验证操作者身份
    if perm_req.target_user_id:
        if not user_id:
            return JSONResponse(
                {"ok": False, "error": "user_id required for targeted permission"},
                status_code=400,
            )
        if perm_req.target_user_id != user_id:
            logger.warning(
                "权限响应鉴权失败: request_id=%s, target=%s, actual=%s",
                request_id,
                perm_req.target_user_id,
                user_id,
            )
            return JSONResponse(
                {"ok": False, "error": "not authorized to respond"},
                status_code=403,
            )

    success = manager.respond(request_id, approved)

    if success:
        logger.info(
            "权限响应: request_id=%s, approved=%s, user_id=%s", request_id, approved, user_id
        )
        return JSONResponse({"ok": True})
    else:
        return JSONResponse(
            {"ok": False, "error": "请求不存在或已过期"},
            status_code=404,
        )


async def _handle_permission_list(request: Any, ctx: _AppContext) -> Any:
    """GET /api/permissions — 获取待处理的权限请求列表.

    支持 ?user_id=xxx 查询参数进行用户隔离过滤。
    """

    # 安全加固: 权限请求列表暴露工具参数和请求 ID，仅 admin 可查看
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.permission_request import PermissionManager

    # 优先从 X-User-Id header 获取调用者身份，query param 作为向后兼容 fallback
    # 长期方案：引入 caller identity 体系后改为从认证 token 中提取
    user_id = request.headers.get("x-user-id") or request.query_params.get("user_id") or None

    manager = PermissionManager()
    pending = manager.get_pending_requests(user_id=user_id)

    return JSONResponse({"ok": True, "requests": pending})


async def _handle_memory_search(request: Any, ctx: _AppContext) -> Any:
    """跨员工语义搜索 — GET /api/memory/search?q=关键词&limit=10&employee=xxx.

    查询参数:
        q (required): 搜索关键词
        limit (optional): 最大返回条数，默认 10
        employee (optional): 限定搜索某个员工（不传则跨全员工搜索）
    """

    # 安全加固: 跨员工搜索可泄露私有/受限记忆，仅 admin 可访问
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)


    query = request.query_params.get("q", "").strip()
    if not query:
        return JSONResponse({"error": "q is required"}, status_code=400)

    limit = _safe_limit(request.query_params.get("limit", "10"), default=10)
    if limit <= 0:
        limit = 10
    employee = request.query_params.get("employee", "").strip()

    store = get_memory_store(project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request))

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
    except Exception:
        logger.exception("记忆搜索失败")
        return _error_response("内部错误", 500)


async def _handle_trajectory_report(request: Any, ctx: _AppContext) -> Any:
    """接收外部 agent 的轨迹数据 — POST /api/trajectory/report.

    轨迹数据存储到独立的文件系统，不写入永久记忆。
    存储路径：/data/trajectory_archive/{date}/{employee}-{uuid}.jsonl
    """

    # 身份校验：必须经过多租户中间件认证（request.state.tenant 由中间件注入）
    # 不依赖 get_current_tenant 的 fallback（fallback 返回 admin，会绕过认证）
    _traj_tenant_obj = getattr(getattr(request, "state", None), "tenant", None)
    if _traj_tenant_obj is None:
        return _error_response("authentication required: valid Bearer token needed", 401)

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
        logger.debug("轨迹上报: 员工名解析失败, 使用原始值: %s", employee_name, exc_info=True)
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
        from datetime import date as dt_date
        from datetime import datetime

        # ── 独立轨迹存储（不使用 TrajectoryCollector，避免写入 .crew/trajectories） ──
        trajectory_id = f"traj_{uuid.uuid4().hex[:12]}"
        date_str = dt_date.today().isoformat()
        # 租户隔离：按 tenant 分目录存储轨迹
        _traj_base = _tenant_base_dir(request)
        archive_base = _traj_base / "trajectory_archive"
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

        # ── 更新元数据索引（加文件锁防并发写入冲突） ──
        from crew.paths import file_lock

        index_file = archive_base / "index.json"

        from datetime import datetime

        with file_lock(index_file):
            index_data = {}
            if index_file.exists():
                try:
                    with open(index_file, encoding="utf-8") as f:
                        index_data = _json.load(f)
                except Exception:
                    logger.debug("轨迹索引文件解析失败: %s", index_file, exc_info=True)

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
    except Exception:
        logger.exception("轨迹上报处理失败")
        return _error_response("内部错误", 500)


async def _handle_project_status(request: Any, ctx: _AppContext) -> Any:
    """项目状态概览 — 组织架构 + 成本 + 员工列表."""

    # 安全加固: 含组织架构/计费余额/API key 间接暴露，仅 admin 可读
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

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

    result = discover_employees(ctx.project_dir, tenant_id=_tenant_id_for_config(request))
    org = load_organization(project_dir=ctx.project_dir)
    cost = query_cost_summary(ctx.registry, days=days)

    # 并发获取外部余额/账单（P2-6: 避免串行 HTTP 请求）
    _billing_coro = None
    _billing_api_key = None
    _billing_base_url = None
    for emp in result.employees.values():
        if emp.api_key and "aiberm.com" in (emp.base_url or ""):
            _billing_api_key = emp.api_key
            _billing_base_url = emp.base_url
            _billing_coro = fetch_aiberm_billing(
                api_key=_billing_api_key,
                base_url=_billing_base_url,
                days=days,
            )
            break

    aiberm_token = os.environ.get("AIBERM_ACCESS_TOKEN", "")
    aiberm_user_id = os.environ.get("AIBERM_USER_ID", "")
    _balance_coro = (
        fetch_aiberm_balance(access_token=aiberm_token, user_id=aiberm_user_id)
        if aiberm_token
        else None
    )

    moonshot_key = os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY", "")
    _moonshot_coro = fetch_moonshot_balance(api_key=moonshot_key) if moonshot_key else None

    # 收集需要并发执行的协程
    _coros: list[Any] = []
    _coro_keys: list[str] = []
    if _billing_coro:
        _coros.append(_billing_coro)
        _coro_keys.append("billing")
    if _balance_coro:
        _coros.append(_balance_coro)
        _coro_keys.append("balance")
    if _moonshot_coro:
        _coros.append(_moonshot_coro)
        _coro_keys.append("moonshot")

    _results_map: dict[str, Any] = {}
    if _coros:
        _gathered = await asyncio.gather(*_coros, return_exceptions=True)
        for key, val in zip(_coro_keys, _gathered, strict=False):
            if isinstance(val, Exception):
                logger.warning("外部计费请求失败 (%s): %s", key, val)
                _results_map[key] = None
            else:
                _results_map[key] = val

    aiberm_billing = _results_map.get("billing")
    aiberm_balance = _results_map.get("balance")
    if aiberm_billing is None:
        aiberm_billing = {}
    if aiberm_balance:
        aiberm_billing["balance_usd"] = aiberm_balance["balance_usd"]
        aiberm_billing["used_usd"] = aiberm_balance["used_usd"]

    # Moonshot/Kimi 余额 + 7日估算成本
    moonshot_billing = None
    _moonshot_balance = _results_map.get("moonshot")
    if _moonshot_balance is not None:
        # 从 cost_7d 提取 kimi 相关模型的估算成本
        kimi_cost_usd = 0.0
        kimi_tasks = 0
        for model_name, model_cost in cost.get("by_model", {}).items():
            if model_name.startswith("kimi") or model_name.startswith("moonshot"):
                kimi_cost_usd += model_cost.get("cost_usd", 0)
                kimi_tasks += model_cost.get("tasks", 0)
        moonshot_billing = {
            "balance": _moonshot_balance,
            "cost_7d_usd": round(kimi_cost_usd, 4),
            "cost_7d_tasks": kimi_tasks,
        }

    # 用真实账单校准员工估算成本
    aiberm_real = aiberm_billing.get("total_usd") if aiberm_billing else None
    kimi_real = moonshot_billing.get("cost_7d_usd") if moonshot_billing else None
    cost = calibrate_employee_costs(cost, aiberm_real_usd=aiberm_real, moonshot_real_usd=kimi_real)

    # 预加载所有员工的记忆数量

    store = get_memory_store(project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request))
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

    # TODO: 给 registry.snapshot() 添加 since 参数，在存储层过滤，避免遍历全量任务快照
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

    # P2: KV 写入需要管理员权限 + 审计日志
    # 当前方案：admin_token 校验 + X-User-Id 审计日志记录
    # 长期方案：引入命名空间强隔离（需架构层面支持，当前不加 key 前缀以免破坏现有功能）
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    key = request.path_params.get("key", "")
    err = _validate_kv_key(key)
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)

    # 审计日志：记录 KV 写入者身份
    kv_caller_id = request.headers.get("x-user-id")
    if kv_caller_id:
        logger.info("KV write: key=%s caller=%s", key, kv_caller_id)
    else:
        logger.info("KV write: key=%s caller=unknown (no X-User-Id header)", key)

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
    from starlette.responses import Response

    # 安全加固: KV 读取需要管理员权限（与写接口对齐）
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

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

    # 安全加固: KV 列表需要管理员权限（与写接口对齐）
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

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

    limit = _safe_limit(request.query_params.get("limit", "500"), default=500)
    limit = min(limit, 2000)  # 硬上限

    keys: list[str] = []
    if scan_dir.is_dir():
        count = 0
        for p in sorted(scan_dir.rglob("*")):
            if count >= limit:
                break
            if p.is_file():
                rel = p.relative_to(base_dir)
                keys.append(str(rel))
                count += 1

    return JSONResponse({"ok": True, "keys": keys, "truncated": len(keys) >= limit})


# ── Pipeline / Discussion / Meeting / Decision / WorkLog / Permission API ──

