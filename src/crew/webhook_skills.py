"""Skills API handlers — HTTP 端点."""

import logging
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from crew.memory import get_memory_store
from crew.skills import Skill, SkillAction, SkillMetadata, SkillStore, SkillTrigger
from crew.skills_engine import SkillsEngine
from crew.tenant import get_current_tenant
from crew.webhook_context import _AppContext

logger = logging.getLogger(__name__)


def _tenant_id_for_store(request: Any) -> str | None:
    """从请求获取租户 ID，admin 返回 None（向后兼容）."""
    tenant = get_current_tenant(request)
    return None if tenant.is_admin else tenant.tenant_id


def _get_skills_engine(request: Any, ctx: _AppContext) -> SkillsEngine:
    """获取 SkillsEngine 实例."""
    # 每次都创建新实例，使用 ctx.project_dir
    skill_store = SkillStore(project_dir=ctx.project_dir)
    memory_store = get_memory_store(
        project_dir=ctx.project_dir, tenant_id=_tenant_id_for_store(request)
    )
    return SkillsEngine(skill_store, memory_store)


async def _handle_skill_create(request: Request, ctx: _AppContext) -> JSONResponse:
    """创建 Skill.

    POST /api/employees/{employee_name}/skills
    Body: {
        "name": "query-before-code",
        "version": "0.1.0",
        "description": "...",
        "trigger": {...},
        "actions": [...]
    }
    """
    try:
        tenant = get_current_tenant(request)
        if not tenant.is_admin:
            return JSONResponse({"error": "admin access required"}, status_code=403)

        employee_name = request.path_params.get("employee_name")
        if not employee_name:
            return JSONResponse({"error": "employee_name required"}, status_code=400)

        body = await request.json()

        # 构建 Skill 对象
        skill = Skill(
            name=body["name"],
            version=body.get("version", "0.1.0"),
            employee=employee_name,
            description=body["description"],
            trigger=SkillTrigger(**body["trigger"]),
            actions=[SkillAction(**a) for a in body["actions"]],
            metadata=SkillMetadata(**body.get("metadata", {})),
            enabled=body.get("enabled", True),
        )

        # 保存
        engine = _get_skills_engine(request, ctx)
        created_skill = engine.skill_store.create_skill(skill)

        return JSONResponse(
            {
                "ok": True,
                "skill_id": created_skill.skill_id,
                "employee": created_skill.employee,
                "name": created_skill.name,
                "version": created_skill.version,
                "created_at": created_skill.metadata.created_at,
            }
        )

    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.error(f"Failed to create skill: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


async def _handle_skill_list(request: Request, ctx: _AppContext) -> JSONResponse:
    """列出 Skills.

    GET /api/employees/{employee_name}/skills
    """
    try:
        employee_name = request.path_params.get("employee_name")

        engine = _get_skills_engine(request, ctx)
        skills = engine.skill_store.list_skills(employee_name)

        return JSONResponse(
            {
                "employee": employee_name,
                "skills": [
                    {
                        "skill_id": s.skill_id,
                        "name": s.name,
                        "version": s.version,
                        "category": s.metadata.category,
                        "priority": s.metadata.priority,
                        "enabled": s.enabled,
                    }
                    for s in skills
                ],
                "total": len(skills),
            }
        )

    except Exception as e:
        logger.error(f"Failed to list skills: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


async def _handle_skill_get(request: Request, ctx: _AppContext) -> JSONResponse:
    """获取 Skill 详情.

    GET /api/employees/{employee_name}/skills/{skill_name}
    """
    try:
        employee_name = request.path_params.get("employee_name")
        skill_name = request.path_params.get("skill_name")

        if not employee_name or not skill_name:
            return JSONResponse({"error": "employee_name and skill_name required"}, status_code=400)

        engine = _get_skills_engine(request, ctx)
        skill = engine.skill_store.get_skill(employee_name, skill_name)

        if not skill:
            return JSONResponse({"error": "Skill not found"}, status_code=404)

        return JSONResponse(skill.model_dump())

    except Exception as e:
        logger.error(f"Failed to get skill: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


async def _handle_skill_update(request: Request, ctx: _AppContext) -> JSONResponse:
    """更新 Skill.

    PUT /api/employees/{employee_name}/skills/{skill_name}
    Body: {
        "version": "0.2.0",
        "description": "...",
        ...
    }
    """
    try:
        tenant = get_current_tenant(request)
        if not tenant.is_admin:
            return JSONResponse({"error": "admin access required"}, status_code=403)

        employee_name = request.path_params.get("employee_name")
        skill_name = request.path_params.get("skill_name")

        if not employee_name or not skill_name:
            return JSONResponse({"error": "employee_name and skill_name required"}, status_code=400)

        body = await request.json()

        engine = _get_skills_engine(request, ctx)
        updated_skill = engine.skill_store.update_skill(employee_name, skill_name, body)

        return JSONResponse(
            {
                "ok": True,
                "skill_id": updated_skill.skill_id,
                "version": updated_skill.version,
                "updated_at": updated_skill.metadata.updated_at,
            }
        )

    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        logger.error(f"Failed to update skill: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


async def _handle_skill_delete(request: Request, ctx: _AppContext) -> JSONResponse:
    """删除 Skill.

    DELETE /api/employees/{employee_name}/skills/{skill_name}
    """
    try:
        tenant = get_current_tenant(request)
        if not tenant.is_admin:
            return JSONResponse({"error": "admin access required"}, status_code=403)

        employee_name = request.path_params.get("employee_name")
        skill_name = request.path_params.get("skill_name")

        if not employee_name or not skill_name:
            return JSONResponse({"error": "employee_name and skill_name required"}, status_code=400)

        engine = _get_skills_engine(request, ctx)
        deleted = engine.skill_store.delete_skill(employee_name, skill_name)

        if not deleted:
            return JSONResponse({"error": "Skill not found"}, status_code=404)

        return JSONResponse({"ok": True})

    except Exception as e:
        logger.error(f"Failed to delete skill: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


async def _handle_skills_check_triggers(request: Request, ctx: _AppContext) -> JSONResponse:
    """检查应该触发的 Skills.

    POST /api/skills/check-triggers
    Body: {
        "employee": "赵云帆",
        "task": "帮我写一个新的 API endpoint",
        "context": {...}
    }
    """
    try:
        body = await request.json()
        employee = body.get("employee")
        task = body.get("task")
        context = body.get("context", {})

        if not employee or not task:
            return JSONResponse({"error": "employee and task required"}, status_code=400)

        engine = _get_skills_engine(request, ctx)
        triggered = engine.check_triggers(employee, task, context)

        return JSONResponse(
            {
                "employee": employee,
                "triggered_skills": [
                    {
                        "skill_id": skill.skill_id,
                        "name": skill.name,
                        "priority": skill.metadata.priority,
                        "match_score": score,
                        "reason": f"Matched with score {score:.2f}",
                    }
                    for skill, score in triggered
                ],
                "total": len(triggered),
            }
        )

    except Exception as e:
        logger.error(f"Failed to check triggers: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


async def _handle_skills_execute(request: Request, ctx: _AppContext) -> JSONResponse:
    """执行 Skill.

    POST /api/skills/execute
    Body: {
        "skill_id": "skill_abc123",
        "employee": "赵云帆",
        "context": {...}
    }
    """
    try:
        body = await request.json()
        skill_id = body.get("skill_id")
        employee = body.get("employee")
        context = body.get("context", {})

        if not skill_id or not employee:
            return JSONResponse({"error": "skill_id and employee required"}, status_code=400)

        engine = _get_skills_engine(request, ctx)

        # 找到对应的 skill
        skills = engine.skill_store.list_skills(employee)
        skill = next((s for s in skills if s.skill_id == skill_id), None)

        if not skill:
            return JSONResponse({"error": "Skill not found"}, status_code=404)

        # 执行
        result = engine.execute_skill(skill, employee, context)

        # 记录触发历史
        engine.record_trigger(
            skill=skill,
            employee=employee,
            task=context.get("task", ""),
            match_score=1.0,  # 手动执行，分数为 1.0
            execution_result=result,
        )

        return JSONResponse(
            {
                "ok": True,
                "skill_id": skill.skill_id,
                "skill_name": skill.name,
                **result,
            }
        )

    except Exception as e:
        logger.error(f"Failed to execute skill: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


async def _handle_skills_stats(request: Request, ctx: _AppContext) -> JSONResponse:
    """获取 Skills 统计.

    GET /api/skills/stats
    """
    try:
        engine = _get_skills_engine(request, ctx)
        stats = engine.skill_store.get_stats()

        return JSONResponse(stats)

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


async def _handle_skills_trigger_history(request: Request, ctx: _AppContext) -> JSONResponse:
    """获取触发历史.

    GET /api/skills/trigger-history?employee=赵云帆&limit=50
    """
    try:
        params = request.query_params
        employee = params.get("employee")
        skill_name = params.get("skill_name")
        limit = int(params.get("limit", "50"))
        since = params.get("since")

        engine = _get_skills_engine(request, ctx)
        triggers = engine.skill_store.get_trigger_history(
            employee=employee,
            skill_name=skill_name,
            limit=limit,
            since=since,
        )

        return JSONResponse(
            {
                "triggers": [t.model_dump() for t in triggers],
                "total": len(triggers),
                "limit": limit,
            }
        )

    except Exception as e:
        logger.error(f"Failed to get trigger history: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)
