"""记忆管理处理器."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crew.webhook_handlers._common import (
    JSONResponse,
    _AppContext,
    _error_response,
    _require_admin_token,
    _safe_limit,
    _tenant_base_dir,
    _tenant_data_dir,
    _tenant_id_for_store,
    get_current_tenant,
    get_memory_store,
    logger,
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

    # 内容长度限制（S4）
    MAX_CONTENT_LENGTH = 5000  # 字符
    if len(content) > MAX_CONTENT_LENGTH:
        return JSONResponse(
            {
                "ok": False,
                "error": f"Content too long ({len(content)} chars, max {MAX_CONTENT_LENGTH})",
                "suggestions": ["将内容拆分为多条记忆", "保留核心要点，删减重复信息"],
            },
            status_code=400,
        )

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
        from crew.classification import CLASSIFICATION_LEVELS, EMPLOYEE_CLEARANCE

        emp_clearance = EMPLOYEE_CLEARANCE.get(employee, {})
        emp_level = emp_clearance.get("clearance", "internal")
        if CLASSIFICATION_LEVELS.get(emp_level, 1) < CLASSIFICATION_LEVELS.get(classification, 2):
            return JSONResponse(
                {
                    "error": f"员工 {employee} 的许可等级 ({emp_level}) 不足以写入 {classification} 级别记忆"
                },
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
        logger.info(f"Intercepted trajectory memory: employee={employee}, session={source_session}")
        return JSONResponse(
            {
                "ok": True,
                "skipped": True,
                "reason": "trajectory tag intercepted",
            }
        )

    # 相似度检测（2026-03-02 记忆去重）
    # TODO: 考虑将相似度检测改为后台异步检查，避免热路径上的 embedding 计算阻塞写入
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

    store = get_memory_store(project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request))

    # 幂等检查：同 employee + source_session + category 不重复写入
    # TODO: 改用 DB 层 EXISTS 查询提升性能（当前遍历内存中最近 50 条）
    if source_session:
        existing = store.query(employee, limit=50)
        for entry in existing:
            # 兼容 MemoryEntry（属性访问）和 dict（键访问）
            _src = (
                entry.source_session
                if hasattr(entry, "source_session")
                else entry.get("source_session", "")
            )
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
        classification=str(classification)
        if classification in ("public", "internal", "restricted", "confidential")
        else "internal",
        domain=domain if isinstance(domain, list) else [],
    )

    # 兼容 MemoryEntry（属性访问）和 dict（键访问）
    result_employee = (
        result.employee if hasattr(result, "employee") else result.get("employee", employee)
    )
    result_id = result.id if hasattr(result, "id") else result.get("id", "")

    # 写入后失效缓存（用解析后的花名作为 cache key）
    try:
        from crew.memory_cache import invalidate

        invalidate(result_employee)
    except Exception:
        logger.debug("记忆缓存失效失败", exc_info=True)

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


def _semantic_memory_search(store, employee: str, query: str, category: str | None, limit: int):
    """语义混合搜索：优先 SemanticMemoryIndex（向量70%+关键词30%），降级到数据库 ILIKE 搜索."""
    try:
        from crew.memory_search import SemanticMemoryIndex

        memory_dir = getattr(store, "memory_dir", None)
        if memory_dir is not None:
            with SemanticMemoryIndex(memory_dir) as index:
                if index.has_index(employee):
                    results = index.search(employee, query, limit=limit * 2)
                    if results:
                        entries_map = {e.id: e for e in store._load_employee_entries(employee)}
                        filtered = []
                        for entry_id, _content, _score in results:
                            entry = entries_map.get(entry_id)
                            if entry is None or entry.superseded_by:
                                continue
                            if store._is_expired(entry):
                                continue
                            if category and entry.category != category:
                                continue
                            filtered.append(store._apply_decay(entry))
                        if filtered:
                            return filtered[:limit]
    except Exception as e:
        logger.debug("SemanticMemoryIndex unavailable, fallback to db search: %s", e)

    # 降级：如果 store.query 支持 search_text 参数（MemoryStoreDB），
    # 直接用数据库 ILIKE 做子串匹配（对中文友好）；
    # 否则回退到 Python 侧关键词匹配（仅适用于文件版 MemoryStore）。
    import inspect

    query_sig = inspect.signature(store.query)
    if "search_text" in query_sig.parameters:
        return store.query(employee=employee, category=category, limit=limit, search_text=query)

    # 文件版 MemoryStore 降级：Python 侧关键词匹配
    pool_size = min(limit * 10, 200)
    candidates = store.query(employee=employee, category=category, limit=pool_size)
    if not candidates:
        return []

    query_lower = query.lower()
    # 对中文：整个 query 作为子串匹配；对英文：按空格分词后匹配
    if any("\u4e00" <= c <= "\u9fff" for c in query_lower):
        scored = []
        for entry in candidates:
            if query_lower in entry.content.lower():
                scored.append(entry)
        return scored[:limit]

    query_tokens = set(query_lower.split())
    scored = []
    for entry in candidates:
        content_lower = entry.content.lower()
        hits = sum(1 for t in query_tokens if t in content_lower)
        if hits > 0:
            scored.append((entry, hits / len(query_tokens)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [e for e, _ in scored[:limit]]


async def _handle_memory_query(request: Any, ctx: _AppContext) -> Any:
    """记忆查询 — GET /api/memory/query.

    查询参数（与 MCP query_memory 工具入参对齐）:
        employee (required): 员工名称
        query (optional): 搜索关键词 — 有值时走语义混合搜索（向量 70% + 关键词 30%）
        category (optional): 按类别过滤
        limit (optional): 最大返回条数，默认 20
    """

    employee = request.query_params.get("employee", "")
    category = request.query_params.get("category") or None
    query = request.query_params.get("query", "").strip()
    limit = _safe_limit(request.query_params.get("limit", "20"), default=20)

    if not employee:
        return JSONResponse({"error": "employee is required"}, status_code=400)

    # P2: 记忆查询分级过滤 — 未指定 classification_max 时默认 "internal"
    classification_max = request.query_params.get("classification_max", "internal")
    _classification_levels = {"public": 0, "internal": 1, "restricted": 2, "confidential": 3}
    max_level = _classification_levels.get(classification_max, 1)

    store = get_memory_store(project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request))

    # 有搜索关键词时，走语义混合搜索
    if query:
        entries = _semantic_memory_search(store, employee, query, category, limit)
    else:
        entries = store.query(
            employee=employee,
            category=category,
            limit=limit,
        )
    # 过滤超出请求分级上限的记忆
    filtered = [
        e
        for e in entries
        if _classification_levels.get(getattr(e, "classification", "internal"), 1) <= max_level
    ]
    data = [e.model_dump() for e in filtered]
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
        return JSONResponse({"error": "entry_id, employee, content are required"}, status_code=400)

    store = get_memory_store(project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request))

    # DB 版：直接用 store.update()
    if hasattr(store, "update") and callable(getattr(store, "update", None)):
        employee = store.resolve_to_character_name(employee)

        # 构建更新标签
        update_tag = f"updated-by:{updated_by or 'unknown'}"
        update_time_tag = f"updated-at:{datetime.now().strftime('%Y-%m-%d')}"
        add_tags = [update_tag, update_time_tag]

        # 如果提供了 tags，先 normalize 再完全替换
        final_tags = None
        if tags is not None and isinstance(tags, list):
            try:
                from crew.memory_tags import normalize_tags

                final_tags = normalize_tags(tags)
            except ImportError:
                final_tags = tags

        updated = store.update(
            entry_id=entry_id,
            employee=employee,
            content=content,
            tags=final_tags,
            add_tags=add_tags,
        )
        if not updated:
            return JSONResponse({"error": "Memory entry not found"}, status_code=404)

        # 失效缓存
        try:
            from crew.memory_cache import invalidate

            invalidate(employee)
        except Exception:
            logger.debug("记忆缓存失效失败", exc_info=True)

        return JSONResponse(
            {
                "ok": True,
                "entry_id": entry_id,
                "updated": True,
            }
        )

    # 文件版：保留原有的 JSONL 操作逻辑
    employee = store.resolve_to_character_name(employee)
    path = store.employee_file(employee)

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
        logger.debug("记忆缓存失效失败", exc_info=True)

    # 更新 embedding 缓存
    try:
        from crew.memory_similarity import (
            _load_embedding_cache,
            _save_embedding_cache,
            get_embedding,
        )

        new_embedding = await get_embedding(content)
        if new_embedding is not None:
            cache = _load_embedding_cache(store.memory_dir, employee)
            cache[entry_id] = new_embedding
            _save_embedding_cache(store.memory_dir, employee, cache)
    except Exception as e:
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

    from crew.memory_tags import search_tags

    query = request.query_params.get("query", "")
    limit = _safe_limit(request.query_params.get("limit", "10"), default=10)

    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)

    matches = search_tags(query, limit)
    return JSONResponse({"ok": True, "matches": matches})


async def _handle_memory_ingest(request: Any, ctx: _AppContext) -> Any:
    """接收外部讨论数据，写入参与者记忆和会议记录."""

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
    except (ValueError, TypeError, OSError):
        logger.exception("记忆导入失败")
        return _error_response("内部错误", 500)


async def _handle_memory_delete(request: Any, ctx: _AppContext) -> Any:
    """记忆删除 — DELETE /api/memory/{entry_id}.

    路径参数:
        entry_id: 记忆条目 ID

    查询参数（可选）:
        employee: 员工名（提供后只在该员工文件中查找，提高效率）

    返回:
        {"ok": true, "deleted": true} 或 {"ok": false, "error": "..."}
    """

    # P2: 记忆删除需要管理员权限（所有权校验）
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    # 从路径参数获取 entry_id
    entry_id = request.path_params.get("entry_id", "")
    if not entry_id:
        return JSONResponse({"ok": False, "error": "entry_id is required"}, status_code=400)

    # 从查询参数获取可选的 employee
    employee = request.query_params.get("employee")

    store = get_memory_store(project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request))

    try:
        deleted = store.delete(entry_id, employee=employee)
        if deleted:
            return JSONResponse({"ok": True, "deleted": True, "entry_id": entry_id})
        else:
            return JSONResponse(
                {"ok": False, "error": "Entry not found", "entry_id": entry_id}, status_code=404
            )
    except Exception:
        logger.exception("记忆删除失败")
        return _error_response("内部错误", 500)


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

    # 安全加固: 草稿列表需要管理员权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.memory_drafts import MemoryDraftStore

    status = request.query_params.get("status")
    employee = request.query_params.get("employee")
    limit = _safe_limit(request.query_params.get("limit", "100"), default=100)

    drafts_dir = _tenant_data_dir(request, "memory_drafts") or Path("/data/memory_drafts")

    try:
        store = MemoryDraftStore(drafts_dir=drafts_dir)
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
    except Exception:
        logger.exception("列出草稿失败")
        return _error_response("内部错误", 500)


async def _handle_memory_drafts_get(request: Any, ctx: _AppContext) -> Any:
    """查看草稿详情 — GET /api/memory/drafts/{draft_id}.

    路径参数:
        draft_id: 草稿 ID

    返回:
        {"ok": true, "draft": {...}} 或 {"ok": false, "error": "..."}
    """

    # 安全加固: 草稿详情需要管理员权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.memory_drafts import MemoryDraftStore

    draft_id = request.path_params.get("draft_id", "")
    if not draft_id:
        return JSONResponse({"ok": False, "error": "draft_id is required"}, status_code=400)

    drafts_dir = _tenant_data_dir(request, "memory_drafts") or Path("/data/memory_drafts")

    try:
        store = MemoryDraftStore(drafts_dir=drafts_dir)
        draft = store.get_draft(draft_id)

        if draft is None:
            return JSONResponse({"ok": False, "error": "Draft not found"}, status_code=404)

        return JSONResponse({"ok": True, "draft": draft.model_dump()})
    except Exception:
        logger.exception("获取草稿失败")
        return _error_response("内部错误", 500)


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

    # 安全加固: 草稿审批需要管理员权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.memory_drafts import MemoryDraftStore

    draft_id = request.path_params.get("draft_id", "")
    if not draft_id:
        return JSONResponse({"ok": False, "error": "draft_id is required"}, status_code=400)

    # 安全加固: reviewed_by 从认证的租户上下文获取，不接受用户传入
    tenant = get_current_tenant(request)
    reviewed_by = tenant.tenant_id or "system"

    drafts_dir = _tenant_data_dir(request, "memory_drafts") or Path("/data/memory_drafts")

    try:
        draft_store = MemoryDraftStore(drafts_dir=drafts_dir)
        draft = draft_store.approve_draft(draft_id, reviewed_by=reviewed_by)

        if draft is None:
            return JSONResponse({"ok": False, "error": "Draft not found"}, status_code=404)

        # 写入正式记忆
        memory_store = get_memory_store(
            project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request)
        )
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
    except Exception:
        logger.exception("批准草稿失败")
        return _error_response("内部错误", 500)


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

    # 安全加固: 草稿拒绝需要管理员权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.memory_drafts import MemoryDraftStore

    draft_id = request.path_params.get("draft_id", "")
    if not draft_id:
        return JSONResponse({"ok": False, "error": "draft_id is required"}, status_code=400)

    payload = (
        await request.json()
        if "application/json" in (request.headers.get("content-type") or "")
        else {}
    )
    reason = payload.get("reason", "")
    # 安全加固: reviewed_by 从认证的租户上下文获取，不接受用户传入
    tenant = get_current_tenant(request)
    reviewed_by = tenant.tenant_id or "system"

    drafts_dir = _tenant_data_dir(request, "memory_drafts") or Path("/data/memory_drafts")

    try:
        store = MemoryDraftStore(drafts_dir=drafts_dir)
        draft = store.reject_draft(draft_id, reason=reason, reviewed_by=reviewed_by)

        if draft is None:
            return JSONResponse({"ok": False, "error": "Draft not found"}, status_code=404)

        logger.info("拒绝草稿: draft_id=%s reason=%s", draft_id, reason)

        return JSONResponse({"ok": True, "draft": draft.model_dump()})
    except Exception:
        logger.exception("拒绝草稿失败")
        return _error_response("内部错误", 500)


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

    from crew.memory_archive import MemoryArchive

    employee = request.query_params.get("employee", "")
    if not employee:
        return JSONResponse({"ok": False, "error": "employee is required"}, status_code=400)

    start_date_str = request.query_params.get("start_date")
    end_date_str = request.query_params.get("end_date")
    category = request.query_params.get("category")

    limit = _safe_limit(request.query_params.get("limit", "100"), default=100)

    try:
        from datetime import datetime

        start_date = datetime.fromisoformat(start_date_str) if start_date_str else None
        end_date = datetime.fromisoformat(end_date_str) if end_date_str else None

        _arc_tid = _tenant_id_for_store(request)
        _arc_dir = _tenant_data_dir(request, "memory_archive")
        memory_store = get_memory_store(project_dir=ctx.project_dir, tenant_id=_arc_tid)
        archive = MemoryArchive(archive_dir=_arc_dir, memory_store=memory_store)

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
    except Exception:
        logger.exception("查询归档记忆失败")
        return _error_response("内部错误", 500)


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

    # 安全加固: 恢复归档记忆需要管理员权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"ok": False, "error": admin_err}, status_code=403)

    from crew.memory_archive import MemoryArchive

    payload = (
        await request.json()
        if "application/json" in (request.headers.get("content-type") or "")
        else {}
    )

    employee = payload.get("employee", "")
    entry_ids = payload.get("entry_ids", [])

    if not employee:
        return JSONResponse({"ok": False, "error": "employee is required"}, status_code=400)

    if not entry_ids or not isinstance(entry_ids, list):
        return JSONResponse({"ok": False, "error": "entry_ids is required"}, status_code=400)

    try:
        _arc_tid = _tenant_id_for_store(request)
        _arc_dir = _tenant_data_dir(request, "memory_archive")
        memory_store = get_memory_store(project_dir=ctx.project_dir, tenant_id=_arc_tid)
        archive = MemoryArchive(archive_dir=_arc_dir, memory_store=memory_store)

        stats = archive.restore_from_archive(employee, entry_ids)

        logger.info(
            "恢复归档记忆: employee=%s restored=%d not_found=%d",
            employee,
            stats["restored"],
            stats["not_found"],
        )

        return JSONResponse({"ok": True, **stats})
    except Exception:
        logger.exception("恢复归档记忆失败")
        return _error_response("内部错误", 500)


async def _handle_memory_archive_stats(request: Any, ctx: _AppContext) -> Any:
    """获取归档统计 — GET /api/memory/archive/stats.

    查询参数:
        employee (required): 员工名称

    返回:
        {"ok": true, "total": 100, "by_year": {"2026": 50, "2025": 50}}
    """

    from crew.memory_archive import MemoryArchive

    employee = request.query_params.get("employee", "")
    if not employee:
        return JSONResponse({"ok": False, "error": "employee is required"}, status_code=400)

    try:
        _arc_tid = _tenant_id_for_store(request)
        _arc_dir = _tenant_data_dir(request, "memory_archive")
        memory_store = get_memory_store(project_dir=ctx.project_dir, tenant_id=_arc_tid)
        archive = MemoryArchive(archive_dir=_arc_dir, memory_store=memory_store)

        stats = archive.get_archive_stats(employee)

        return JSONResponse({"ok": True, **stats})
    except Exception:
        logger.exception("获取归档统计失败")
        return _error_response("内部错误", 500)


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

    tags_str = request.query_params.get("tags", "")
    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else None
    category = request.query_params.get("category")
    exclude_employee = request.query_params.get("exclude_employee", "")

    limit = _safe_limit(request.query_params.get("limit", "20"), default=20)

    try:
        memory_store = get_memory_store(
            project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request)
        )
        entries = memory_store.query_shared(
            tags=tags,
            exclude_employee=exclude_employee,
            limit=limit,
        )

        # 按类别过滤（query_shared 不支持 category 参数）
        # TODO: 给 query_shared 添加 category 参数支持，在存储层过滤而非查询后过滤
        if category:
            entries = [e for e in entries if e.category == category]

        return JSONResponse(
            {
                "ok": True,
                "entries": [e.model_dump() for e in entries],
                "total": len(entries),
            }
        )
    except Exception:
        logger.exception("列出共享记忆失败")
        return _error_response("内部错误", 500)


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

    from crew.memory_shared_stats import SharedMemoryStats

    payload = (
        await request.json()
        if "application/json" in (request.headers.get("content-type") or "")
        else {}
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

    # 权限校验：共享记忆统计需要 admin 权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"ok": False, "error": admin_err}, status_code=403)

    try:
        _stats_dir = _tenant_data_dir(request, "memory_shared_stats")
        stats = SharedMemoryStats(stats_dir=_stats_dir)
        stats.record_usage(memory_id, memory_owner, used_by, context)

        return JSONResponse({"ok": True})
    except Exception:
        logger.exception("记录共享记忆使用失败")
        return _error_response("内部错误", 500)


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

    from crew.memory_shared_stats import SharedMemoryStats

    # 权限校验：共享记忆统计需要 admin 权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"ok": False, "error": admin_err}, status_code=403)

    memory_id = request.query_params.get("memory_id")
    owner = request.query_params.get("owner")
    user = request.query_params.get("user")
    popular = request.query_params.get("popular") == "true"

    try:
        _stats_dir = _tenant_data_dir(request, "memory_shared_stats")
        stats_manager = SharedMemoryStats(stats_dir=_stats_dir)

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

    except Exception:
        logger.exception("获取共享记忆统计失败")
        return _error_response("内部错误", 500)


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

    # 仪表板包含所有员工隐私记忆统计，仅 admin 可访问
    admin_err = _require_admin_token(request)
    if admin_err:
        return _error_response(admin_err, 403)

    employee = request.query_params.get("employee")
    limit = _safe_limit(request.query_params.get("limit", "200"), default=200)

    try:
        memory_store = get_memory_store(
            project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request)
        )

        if employee:
            # 单个员工的统计
            entries = memory_store.query(employee, limit=limit)
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
                entries = memory_store.query(emp, limit=min(limit, 200))
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

    except Exception:
        logger.exception("获取仪表板数据失败")
        return _error_response("内部错误", 500)


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

    # P2: 批量记忆操作需要管理员权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    payload = (
        await request.json()
        if "application/json" in (request.headers.get("content-type") or "")
        else {}
    )

    employee = payload.get("employee", "")
    import json

    entry_ids = payload.get("entry_ids", [])
    updates = payload.get("updates", {})

    if not employee or not entry_ids:
        return JSONResponse(
            {"ok": False, "error": "employee and entry_ids are required"},
            status_code=400,
        )

    try:
        memory_store = get_memory_store(
            project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request)
        )

        # DB 版：直接用 store.update()
        if hasattr(memory_store, "update") and callable(getattr(memory_store, "update", None)):
            updated_count = 0
            failed_count = 0
            for eid in entry_ids:
                try:
                    ok = memory_store.update(
                        entry_id=eid,
                        employee=employee,
                        add_tags=updates.get("tags"),
                        remove_tags=updates.get("remove_tags"),
                        confidence=updates.get("confidence"),
                    )
                    if ok:
                        updated_count += 1
                    else:
                        failed_count += 1
                except Exception:
                    logger.debug("批量更新记忆单条失败: eid=%s", eid, exc_info=True)
                    failed_count += 1

            logger.info(
                "批量更新记忆(DB): employee=%s updated=%d failed=%d",
                employee,
                updated_count,
                failed_count,
            )

            return JSONResponse({"ok": True, "updated": updated_count, "failed": failed_count})

        # 文件版：保留原有的 JSONL 操作逻辑
        path = memory_store.employee_file(employee)

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

    except Exception:
        logger.exception("批量更新记忆失败")
        return _error_response("内部错误", 500)


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

    # P2: 批量删除需要管理员权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    payload = (
        await request.json()
        if "application/json" in (request.headers.get("content-type") or "")
        else {}
    )

    employee = payload.get("employee", "")
    entry_ids = payload.get("entry_ids", [])

    if not employee or not entry_ids:
        return JSONResponse(
            {"ok": False, "error": "employee and entry_ids are required"},
            status_code=400,
        )

    try:
        memory_store = get_memory_store(
            project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request)
        )
        deleted_count = 0

        for entry_id in entry_ids:
            if memory_store.delete(entry_id, employee=employee):
                deleted_count += 1

        logger.info("批量删除记忆: employee=%s deleted=%d", employee, deleted_count)

        return JSONResponse({"ok": True, "deleted": deleted_count})

    except Exception:
        logger.exception("批量删除记忆失败")
        return _error_response("内部错误", 500)


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

    # 安全加固: 轨迹导出需要管理员权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.trajectory_export import TrajectoryExporter

    payload = (
        await request.json()
        if "application/json" in (request.headers.get("content-type") or "")
        else {}
    )

    employee = payload.get("employee")
    start_date_str = payload.get("start_date")
    end_date_str = payload.get("end_date")
    min_quality = payload.get("min_quality", 0.0)
    max_samples = payload.get("max_samples", 0)

    try:
        from datetime import datetime
        from pathlib import Path

        start_date = datetime.fromisoformat(start_date_str) if start_date_str else None
        end_date = datetime.fromisoformat(end_date_str) if end_date_str else None

        # 安全加固: 强制输出到受控目录，不接受用户自定义路径
        exports_dir = Path("/data/exports")
        exports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(exports_dir / f"trajectory_dataset_{timestamp}.jsonl")

        # 租户隔离
        _exp_base = _tenant_base_dir(request)
        exporter = TrajectoryExporter(
            archive_dir=_exp_base / "trajectory_archive",
            annotations_dir=_exp_base / "trajectory_annotations",
        )
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

    except Exception:
        logger.exception("导出轨迹数据集失败")
        return _error_response("内部错误", 500)


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

    # 安全加固: 标注接口需要管理员权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.trajectory_export import TrajectoryExporter

    payload = (
        await request.json()
        if "application/json" in (request.headers.get("content-type") or "")
        else {}
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

        # 租户隔离
        _ann_base = _tenant_base_dir(request)
        exporter = TrajectoryExporter(
            archive_dir=_ann_base / "trajectory_archive",
            annotations_dir=_ann_base / "trajectory_annotations",
        )
        annotation = exporter.add_annotation(
            trajectory_id=trajectory_id,
            quality_score=quality_score,
            annotator=annotator,
            notes=notes,
        )

        return JSONResponse({"ok": True, "annotation": annotation.model_dump()})

    except Exception:
        logger.exception("添加轨迹标注失败")
        return _error_response("内部错误", 500)


async def _handle_trajectory_annotation_list(request: Any, ctx: _AppContext) -> Any:
    """列出轨迹标注 — GET /api/trajectory/annotations.

    查询参数:
        min_quality (optional): 最低质量分数
        annotator (optional): 按标注人过滤

    返回:
        {"ok": true, "annotations": [...], "total": 10}
    """

    # 安全加固: 列出标注需要管理员权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"error": admin_err}, status_code=403)

    from crew.trajectory_export import TrajectoryExporter

    min_quality_str = request.query_params.get("min_quality", "0.0")
    annotator = request.query_params.get("annotator")

    try:
        min_quality = float(min_quality_str)

        # 租户隔离
        _annl_base = _tenant_base_dir(request)
        exporter = TrajectoryExporter(
            archive_dir=_annl_base / "trajectory_archive",
            annotations_dir=_annl_base / "trajectory_annotations",
        )
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

    except Exception:
        logger.exception("列出轨迹标注失败")
        return _error_response("内部错误", 500)


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

        engine = SemanticSearchEngine(
            memory_store=get_memory_store(
                project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request)
            )
        )
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

    except Exception:
        logger.exception("语义搜索失败")
        return _error_response("内部错误", 500)


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

        engine = SemanticSearchEngine(
            memory_store=get_memory_store(
                project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request)
            )
        )
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

    except Exception:
        logger.exception("推荐记忆失败")
        return _error_response("内部错误", 500)


async def _handle_memory_similar(request: Any, ctx: _AppContext) -> Any:
    """查找相似记忆 — GET /api/memory/semantic/similar/{memory_id}.

    路径参数:
        memory_id: 记忆 ID

    查询参数:
        limit (optional): 最大返回数量，默认 5

    返回:
        {"ok": true, "similar": [...], "total": 3}
    """

    from crew.memory_semantic import SemanticSearchEngine

    try:
        # 从路径中提取 memory_id
        path = request.url.path
        parts = path.rstrip("/").split("/")
        memory_id = parts[-1] if parts else ""
        if not memory_id:
            return _error_response("missing memory_id", 400)

        limit = _safe_limit(request.query_params.get("limit", "5"), default=5)

        engine = SemanticSearchEngine(
            memory_store=get_memory_store(
                project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request)
            )
        )
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

    except Exception:
        logger.exception("查找相似记忆失败")
        return _error_response("内部错误", 500)


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

    # 安全加固: 提交反馈需要管理员权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"ok": False, "error": admin_err}, status_code=403)

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
                {
                    "ok": False,
                    "error": "feedback_type 必须是 helpful/not_helpful/outdated/incorrect",
                },
                status_code=400,
            )

        # 租户隔离: 按 tenant_id 分隔反馈数据目录
        _fb_base = _tenant_base_dir(request)
        manager = MemoryFeedbackManager(
            feedback_dir=_fb_base / "memory_feedback",
            stats_dir=_fb_base / "memory_usage_stats",
        )
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

    except Exception:
        logger.exception("提交反馈失败")
        return _error_response("内部错误", 500)


async def _handle_memory_feedback_get(request: Any, ctx: _AppContext) -> Any:
    """获取记忆反馈 — GET /api/memory/feedback/{memory_id}.

    路径参数:
        memory_id: 记忆 ID

    返回:
        {"ok": true, "feedback": [...], "total": 5}
    """

    from crew.memory_feedback import MemoryFeedbackManager

    try:
        # 安全加固: 读取反馈需要管理员权限
        admin_err = _require_admin_token(request)
        if admin_err:
            return JSONResponse({"error": admin_err}, status_code=403)

        # 从路径中提取 memory_id
        path = request.url.path
        parts = path.rstrip("/").split("/")
        memory_id = parts[-1] if parts else ""
        if not memory_id:
            return _error_response("missing memory_id", 400)

        # 租户隔离
        _fb_base = _tenant_base_dir(request)
        manager = MemoryFeedbackManager(
            feedback_dir=_fb_base / "memory_feedback",
            stats_dir=_fb_base / "memory_usage_stats",
        )
        feedback = manager.get_feedback(memory_id)

        return JSONResponse(
            {
                "ok": True,
                "feedback": [f.model_dump() for f in feedback],
                "total": len(feedback),
            }
        )

    except Exception:
        logger.exception("获取反馈失败")
        return _error_response("内部错误", 500)


async def _handle_memory_usage_stats(request: Any, ctx: _AppContext) -> Any:
    """获取记忆使用统计 — GET /api/memory/usage/stats/{memory_id}.

    路径参数:
        memory_id: 记忆 ID

    返回:
        {"ok": true, "stats": {...}}
    """

    from crew.memory_feedback import MemoryFeedbackManager

    try:
        # 安全加固: 读取统计需要管理员权限
        admin_err = _require_admin_token(request)
        if admin_err:
            return JSONResponse({"error": admin_err}, status_code=403)

        # 从路径中提取 memory_id
        path = request.url.path
        parts = path.rstrip("/").split("/")
        memory_id = parts[-1] if parts else ""
        if not memory_id:
            return _error_response("missing memory_id", 400)

        # 租户隔离
        _stats_base = _tenant_base_dir(request)
        manager = MemoryFeedbackManager(
            feedback_dir=_stats_base / "memory_feedback",
            stats_dir=_stats_base / "memory_usage_stats",
        )
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

    except Exception:
        logger.exception("获取统计失败")
        return _error_response("内部错误", 500)


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

    # 安全加固: 记录使用需要管理员权限
    admin_err = _require_admin_token(request)
    if admin_err:
        return JSONResponse({"ok": False, "error": admin_err}, status_code=403)

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

        # 租户隔离
        _rec_base = _tenant_base_dir(request)
        manager = MemoryFeedbackManager(
            feedback_dir=_rec_base / "memory_feedback",
            stats_dir=_rec_base / "memory_usage_stats",
        )
        manager.record_usage(
            memory_id=memory_id,
            employee=employee,
            relevance_score=relevance_score,
        )

        return JSONResponse({"ok": True})

    except Exception:
        logger.exception("记录使用失败")
        return _error_response("内部错误", 500)


async def _handle_memory_low_quality(request: Any, ctx: _AppContext) -> Any:
    """获取低质量记忆 — GET /api/memory/usage/low-quality.

    查询参数:
        employee (optional): 按员工过滤
        min_uses (optional): 最少使用次数，默认 5
        max_helpful_ratio (optional): 最大有帮助比例，默认 0.3

    返回:
        {"ok": true, "memories": [...], "total": 3}
    """

    from crew.memory_feedback import MemoryFeedbackManager

    try:
        # 安全加固: 需要管理员权限
        admin_err = _require_admin_token(request)
        if admin_err:
            return JSONResponse({"error": admin_err}, status_code=403)

        employee = request.query_params.get("employee")
        min_uses = int(request.query_params.get("min_uses", "5"))
        max_helpful_ratio = float(request.query_params.get("max_helpful_ratio", "0.3"))

        # 租户隔离
        _lq_base = _tenant_base_dir(request)
        manager = MemoryFeedbackManager(
            feedback_dir=_lq_base / "memory_feedback",
            stats_dir=_lq_base / "memory_usage_stats",
        )
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

    except Exception:
        logger.exception("获取低质量记忆失败")
        return _error_response("内部错误", 500)


async def _handle_memory_popular(request: Any, ctx: _AppContext) -> Any:
    """获取热门记忆 — GET /api/memory/usage/popular.

    查询参数:
        employee (optional): 按员工过滤
        limit (optional): 最大返回数量，默认 10

    返回:
        {"ok": true, "memories": [...], "total": 10}
    """

    from crew.memory_feedback import MemoryFeedbackManager

    try:
        # 安全加固: 需要管理员权限
        admin_err = _require_admin_token(request)
        if admin_err:
            return JSONResponse({"error": admin_err}, status_code=403)

        employee = request.query_params.get("employee")
        limit = _safe_limit(request.query_params.get("limit", "10"), default=10)

        # 租户隔离
        _pop_base = _tenant_base_dir(request)
        manager = MemoryFeedbackManager(
            feedback_dir=_pop_base / "memory_feedback",
            stats_dir=_pop_base / "memory_usage_stats",
        )
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

    except Exception:
        logger.exception("获取热门记忆失败")
        return _error_response("内部错误", 500)


async def _handle_memory_feedback_summary(request: Any, ctx: _AppContext) -> Any:
    """获取反馈汇总 — GET /api/memory/feedback/summary.

    查询参数:
        employee (optional): 按员工过滤

    返回:
        {"ok": true, "summary": {...}}
    """

    from crew.memory_feedback import MemoryFeedbackManager

    try:
        # 安全加固: 需要管理员权限
        admin_err = _require_admin_token(request)
        if admin_err:
            return JSONResponse({"error": admin_err}, status_code=403)

        employee = request.query_params.get("employee")

        # 租户隔离
        _sum_base = _tenant_base_dir(request)
        manager = MemoryFeedbackManager(
            feedback_dir=_sum_base / "memory_feedback",
            stats_dir=_sum_base / "memory_usage_stats",
        )
        summary = manager.get_feedback_summary(employee=employee)

        return JSONResponse(
            {
                "ok": True,
                "summary": summary,
            }
        )

    except Exception:
        logger.exception("获取反馈汇总失败")
        return _error_response("内部错误", 500)
