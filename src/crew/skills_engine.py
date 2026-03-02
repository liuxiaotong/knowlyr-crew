"""Skills 触发引擎 — 判断和执行 Skills."""

import logging
import time
from typing import Any

from crew.memory import MemoryStore
from crew.skills import Skill, SkillAction, SkillStore, SkillTriggerRecord

logger = logging.getLogger(__name__)


class SkillsEngine:
    """Skills 触发和执行引擎."""

    def __init__(self, skill_store: SkillStore, memory_store: MemoryStore):
        self.skill_store = skill_store
        self.memory_store = memory_store

    def check_triggers(
        self, employee: str, task: str, context: dict[str, Any] | None = None
    ) -> list[tuple[Skill, float]]:
        """检查应该触发的 Skills.

        返回: [(skill, match_score), ...]
        """
        context = context or {}
        skills = self.skill_store.list_skills(employee)
        triggered = []

        for skill in skills:
            if not skill.enabled:
                continue

            match_score = self._should_trigger(skill, task, context)
            if match_score > 0:
                triggered.append((skill, match_score))

        # 按优先级和匹配分数排序
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        triggered.sort(
            key=lambda x: (priority_order.get(x[0].metadata.priority, 0), x[1]),
            reverse=True,
        )

        return triggered

    def _should_trigger(self, skill: Skill, task: str, context: dict[str, Any]) -> float:
        """判断是否应该触发 Skill.

        返回: 匹配分数 (0-1)，0 表示不触发
        """
        trigger = skill.trigger

        if trigger.type == "always":
            return 1.0

        elif trigger.type == "keyword":
            return self._keyword_match(trigger.keywords, task)

        elif trigger.type == "semantic":
            # TODO: 实现语义匹配（需要 embedding）
            # 暂时降级为关键词匹配
            logger.warning(
                f"Semantic trigger not implemented for {skill.name}, falling back to keyword"
            )
            # 从 description 中提取关键词作为临时方案
            keywords = self._extract_keywords_from_description(skill.description)
            return self._keyword_match(keywords, task)

        return 0.0

    def _keyword_match(self, keywords: list[str], task: str) -> float:
        """关键词匹配."""
        if not keywords:
            return 0.0

        task_lower = task.lower()
        matched = sum(1 for kw in keywords if kw.lower() in task_lower)

        if matched == 0:
            return 0.0

        # 匹配分数 = 匹配的关键词数 / 总关键词数
        return min(matched / len(keywords), 1.0)

    def _extract_keywords_from_description(self, description: str) -> list[str]:
        """从 description 中提取关键词（临时方案）."""
        # 提取引号中的内容作为关键词
        import re

        keywords = re.findall(r'"([^"]+)"', description)
        return keywords

    def execute_skill(
        self, skill: Skill, employee: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """执行 Skill.

        返回: {
            "executed_actions": [...],
            "enhanced_context": {...}
        }
        """
        context = context or {}
        start_time = time.time()

        executed_actions = []
        enhanced_context = {}

        for action in skill.actions:
            try:
                result = self._execute_action(action, employee, context)
                executed_actions.append(
                    {"type": action.type, "status": "success", "result": result}
                )

                # 合并到 enhanced_context
                if action.type == "query_memory":
                    enhanced_context.setdefault("memories", []).extend(
                        result.get("memories", [])
                    )
                elif action.type == "load_checklist":
                    enhanced_context.setdefault("checklist_items", []).extend(
                        result.get("items", [])
                    )

            except Exception as e:
                logger.error(f"Failed to execute action {action.type}: {e}")
                executed_actions.append(
                    {"type": action.type, "status": "failure", "error": str(e)}
                )

        execution_time_ms = int((time.time() - start_time) * 1000)

        return {
            "executed_actions": executed_actions,
            "enhanced_context": enhanced_context,
            "execution_time_ms": execution_time_ms,
        }

    def _execute_action(
        self, action: SkillAction, employee: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """执行单个 Action."""
        if action.type == "query_memory":
            return self._execute_query_memory(action, employee, context)
        elif action.type == "load_checklist":
            return self._execute_load_checklist(action, employee, context)
        elif action.type == "read_wiki":
            return self._execute_read_wiki(action, employee, context)
        else:
            raise ValueError(f"Unknown action type: {action.type}")

    def _execute_query_memory(
        self, action: SkillAction, employee: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """执行 query_memory 动作."""
        params = action.params

        # 从 params 中提取参数
        # 注意：MemoryStore.query 不支持自由文本查询，只支持 category 过滤
        # 这里简化实现，使用 category 参数
        category = params.get("category")
        limit = params.get("limit", 10)

        # 查询记忆
        memories = self.memory_store.query(
            employee=employee,
            category=category,
            limit=limit,
        )

        return {"memories": [m.model_dump() for m in memories], "count": len(memories)}

    def _execute_load_checklist(
        self, action: SkillAction, employee: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """执行 load_checklist 动作."""
        params = action.params
        section = params.get("section", "")

        # TODO: 从 soul.md 中提取检查清单
        # 暂时返回空列表
        logger.warning(f"load_checklist not fully implemented for section: {section}")

        return {"items": [], "section": section}

    def _execute_read_wiki(
        self, action: SkillAction, employee: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """执行 read_wiki 动作."""
        params = action.params
        path = params.get("path", "")

        # TODO: 读取 Wiki 内容
        logger.warning(f"read_wiki not fully implemented for path: {path}")

        return {"content": "", "path": path}

    def _replace_template_vars(self, template: str, context: dict[str, Any]) -> str:
        """替换模板变量."""
        # 替换 {{task_keywords}}
        if "{{task_keywords}}" in template:
            keywords = context.get("task_keywords", [])
            template = template.replace("{{task_keywords}}", " ".join(keywords))

        return template

    def record_trigger(
        self,
        skill: Skill,
        employee: str,
        task: str,
        match_score: float,
        execution_result: dict[str, Any],
    ) -> None:
        """记录触发历史."""
        record = SkillTriggerRecord(
            skill_id=skill.skill_id,
            employee=employee,
            task=task,
            match_score=match_score,
            execution_status="success"
            if all(a["status"] == "success" for a in execution_result["executed_actions"])
            else "failure",
            execution_time_ms=execution_result.get("execution_time_ms", 0),
            actions_executed=len(execution_result["executed_actions"]),
            result=execution_result,
        )

        self.skill_store.record_trigger(record)
