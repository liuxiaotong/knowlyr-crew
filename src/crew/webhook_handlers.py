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
    """按 agent_id（数字）或 name（字符串）查找员工."""
    try:
        agent_id = int(identifier)
        for emp in result.employees.values():
            if emp.agent_id == agent_id:
                return emp
        return None
    except ValueError:
        return result.get(identifier)


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
            }
        )

    return JSONResponse({"items": items})


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

        agents.append(
            {
                "id": emp.agent_id,
                "nickname": emp.character_name,
                "title": emp.display_name,
                "avatar_url": "",  # 头像 serving 端点待建，官网用 SVG 替代
                "is_agent": True,
                "staff_badge": "集识光年",
                "bio": bio,
                "expertise": emp.tags,
                "domains": domains,
            }
        )

    # 按 id (agent_id) 升序排列，None 排最后
    agents.sort(key=lambda a: (a["id"] is None, a["id"] or 0))

    return JSONResponse(agents)


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
    """删除员工（本地文件 + 远端标记为 inactive）."""
    import shutil

    from starlette.responses import JSONResponse

    from crew.discovery import discover_employees

    identifier = request.path_params["identifier"]
    result = discover_employees(ctx.project_dir, cache_ttl=0)

    employee = _find_employee(result, identifier)
    if not employee:
        return JSONResponse({"error": "Employee not found"}, status_code=404)

    if not employee.source_path:
        return JSONResponse({"error": "Employee source path unknown"}, status_code=400)

    # 删除本地文件
    source = employee.source_path
    try:
        if source.is_dir():
            shutil.rmtree(source)
        elif source.is_file():
            source.unlink()
    except OSError as e:
        logger.exception("删除员工文件失败: %s", identifier)
        return JSONResponse({"error": f"Delete failed: {e}"}, status_code=500)

    return JSONResponse(
        {
            "ok": True,
            "deleted": employee.name,
        }
    )


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
    agent_id: int | None,
    model: str | None,
    user_message: str,
    message_history: list | None,
    extra_context: str | None,
    sender_name: str,
    channel: str,
    callback_channel_id: int,
    callback_sender_id: int,
    callback_parent_id: int | None,
) -> None:
    """后台执行员工 + 回调蚁聚发频道消息（异步回调模式）."""
    import time as _time

    import httpx

    from crew.webhook_context import _ANTGATHER_API_TOKEN, _ANTGATHER_API_URL

    logger.info(
        "异步回调开始: emp=%s channel=%d sender=%d",
        name, callback_channel_id, callback_sender_id,
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
        name, _task_desc, channel=channel, project_dir=ctx.project_dir,
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
                ctx, emp, user_message,
                message_history=message_history,
                max_visibility="private",
                extra_context=extra_context,
                sender_name=sender_name,
            )
        else:
            import crew.webhook as _wh

            result = await _wh._execute_employee(
                ctx, name, _args,
                agent_id=agent_id, model=model,
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
            _path, _elapsed, _m, _in, _out, name, user_message[:40],
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
            trigger=channel, target_type="employee", target_name=name, args=_args,
        )
        ctx.registry.update(record.task_id, "completed", result=result)
    except Exception:
        pass

    # 回调蚁聚：发频道消息
    from crew.output_sanitizer import strip_internal_tags

    output = ""
    if isinstance(result, dict):
        output = strip_internal_tags((result.get("output") or "").strip())
    if not output:
        logger.warning("异步回调: 员工返回空内容，跳过回调 emp=%s channel=%d", name, callback_channel_id)
        return

    if not _ANTGATHER_API_URL or not _ANTGATHER_API_TOKEN:
        logger.error("异步回调: 蚁聚 API 未配置，无法发送频道消息")
        return

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
                name, callback_channel_id, len(output),
            )
        else:
            logger.error(
                "异步回调失败 (HTTP %d): %s, emp=%s channel=%d",
                resp.status_code, resp.text[:200], name, callback_channel_id,
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

        # ── 异步回调模式（频道 @mention）──
        callback_channel_id = payload.get("callback_channel_id")
        if callback_channel_id is not None:
            try:
                callback_channel_id = int(callback_channel_id)
            except (TypeError, ValueError):
                callback_channel_id = None
        callback_sender_id = payload.get("callback_sender_id")
        if callback_sender_id is not None:
            try:
                callback_sender_id = int(callback_sender_id)
            except (TypeError, ValueError):
                callback_sender_id = None
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
            name, _task_desc, channel=channel, project_dir=ctx.project_dir,
        )
        if _traj_collector is not None:
            _exit_stack.enter_context(_traj_collector)

        # 和飞书相同逻辑：闲聊走 fast path，工作消息走 full path
        import time as _time

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
    """全组织记忆聚合 — GET /api/memory/org?days=7&category=pattern&limit=50."""
    from datetime import datetime, timedelta

    from starlette.responses import JSONResponse

    from crew.discovery import discover_employees
    from crew.memory import MemoryStore

    days = _safe_int(request.query_params.get("days", "7"), 7)
    category = request.query_params.get("category") or None
    limit = _safe_int(request.query_params.get("limit", "50"), 50)

    result = discover_employees(ctx.project_dir)
    store = MemoryStore(project_dir=ctx.project_dir)
    cutoff = (datetime.now() - timedelta(days=days)).isoformat() if days > 0 else ""

    all_memories: list[dict] = []
    stats: dict[str, int] = {
        "decision": 0,
        "estimate": 0,
        "finding": 0,
        "correction": 0,
        "pattern": 0,
    }

    for emp_name in result.employees:
        entries = store.query(emp_name, category=category, limit=200, max_visibility="open")
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
            "memories": all_memories[:limit],
            "stats": stats,
            "total": sum(stats.values()),
        }
    )


async def _handle_trajectory_report(request: Any, ctx: _AppContext) -> Any:
    """接收外部 agent 的轨迹数据 — POST /api/trajectory/report."""
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
        payload.get("employee_name")
        or payload.get("name")
        or payload.get("employee")
        or ""
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
            expected_steps, len(steps), employee_name,
        )

    try:
        import json as _json

        from crew.trajectory import TrajectoryCollector

        output_dir = (ctx.project_dir / ".crew" / "trajectories") if ctx.project_dir else Path(".")
        tc = TrajectoryCollector(
            employee_name,
            task_description,
            model=model,
            channel=channel,
            output_dir=output_dir,
        )
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
                    raw_params = {"_raw": str(raw_params)[:8000], "_type": type(raw_params).__name__}

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

            tc.add_tool_step(
                thought=thought_raw,
                tool_name=tool_name,
                tool_params=raw_params,
                tool_output=tool_output_raw,
                tool_exit_code=s.get("tool_exit_code", 0),
            )
        result = tc.finish(success=success)
        total_steps = result.get("total_steps", len(steps)) if isinstance(result, dict) else len(steps)
        resp_data: dict[str, Any] = {"ok": True, "steps_received": total_steps}
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
                "last_active_at": emp_runtime[name]["last_active_at"].isoformat() if emp_runtime.get(name, {}).get("last_active_at") else None,
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
    daily: dict[str, dict] = defaultdict(lambda: {"tasks": 0, "success": 0, "failed": 0, "cost_usd": 0.0})
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
            with open(traj_file, "r", encoding="utf-8") as _f:
                traj_count = sum(1 for _ in _f)

    return JSONResponse({
        "daily": daily_list,
        "totals": {
            "trajectories": traj_count,
            "tasks": total_tasks,
            "success": total_success,
            "failed": total_failed,
            "total_cost_usd": round(total_cost, 4),
        },
    })
