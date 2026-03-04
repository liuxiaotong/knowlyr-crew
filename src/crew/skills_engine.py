"""Skills 触发引擎 — 判断和执行 Skills."""

import logging
import time
from typing import Any

from crew.skills import Skill, SkillAction, SkillStore, SkillTriggerRecord

logger = logging.getLogger(__name__)


class SkillsEngine:
    """Skills 触发和执行引擎."""

    def __init__(self, skill_store: SkillStore, memory_store: Any):
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

        # 信息分级过滤：从 context 中读取 channel，计算有效许可
        clearance_kwargs: dict[str, Any] = {}
        channel = context.get("channel", "")
        if channel:
            from crew.classification import get_effective_clearance

            clearance = get_effective_clearance(employee, channel)
            clearance_kwargs["classification_max"] = clearance["classification_max"]
            clearance_kwargs["allowed_domains"] = clearance["allowed_domains"]
            clearance_kwargs["include_confidential"] = clearance["include_confidential"]

        # 查询记忆
        memories = self.memory_store.query(
            employee=employee,
            category=category,
            limit=limit,
            **clearance_kwargs,
        )

        # Phase 4：审计日志（skills_engine 层有 channel 信息）
        from crew.classification import CHANNEL_SOURCE_TYPE

        source_type = CHANNEL_SOURCE_TYPE.get(channel, "external") if channel else "internal"
        logger.info(
            "memory_audit: employee=%s channel=%s source_type=%s "
            "classification_max=%s allowed_domains=%s returned=%d",
            employee,
            channel,
            source_type,
            clearance_kwargs.get("classification_max", "none"),
            clearance_kwargs.get("allowed_domains", "none"),
            len(memories),
        )

        # 兼容 MemoryStore（文件版返回 MemoryEntry）和 MemoryStoreDB（返回 dict）
        result_list = []
        for m in memories:
            if isinstance(m, dict):
                result_list.append(m)
            else:
                result_list.append(m.model_dump())

        return {"memories": result_list, "count": len(result_list)}

    def _execute_load_checklist(
        self, action: SkillAction, employee: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """执行 load_checklist 动作 - 从员工 soul 中提取检查清单."""
        params = action.params
        section = params.get("section", "工作检查清单")

        try:
            # 使用 config_store 获取 soul
            from crew.config_store import get_soul

            soul_data = get_soul(employee)
            if not soul_data:
                logger.warning(f"Soul not found for {employee}")
                return {"items": [], "section": section}

            soul_content = soul_data.get("content", "")

            if not soul_content or len(soul_content) < 100:
                logger.warning(f"Empty or invalid soul content for {employee}")
                return {"items": [], "section": section}

            # 解析 Markdown 提取检查清单
            items = self._parse_checklist_from_markdown(soul_content, section)

            return {"items": items, "section": section, "count": len(items)}

        except Exception as e:
            logger.warning(f"load_checklist failed: {e}")
            return {"items": [], "section": section}

    def _parse_checklist_from_markdown(self, content: str, section: str) -> list[str]:
        """从 Markdown 内容中提取检查清单."""
        lines = content.split("\n")
        items = []
        in_section = False
        current_subsection = ""

        for line in lines:
            # 检测章节标题（## 开头，不是 ###）
            if line.startswith("## ") and section in line:
                in_section = True
                continue

            # 遇到下一个同级标题（## 开头，不是 ###），退出
            if in_section and line.startswith("## ") and section not in line:
                break

            # 在目标章节内
            if in_section:
                # 检测子章节（### 开头）
                if line.startswith("### "):
                    current_subsection = line.strip("# ").strip()
                    continue

                # 提取检查项（- [ ] 格式）
                stripped = line.strip()
                if stripped.startswith("- [ ]"):
                    item_text = stripped[5:].strip()
                    if current_subsection:
                        items.append(f"[{current_subsection}] {item_text}")
                    else:
                        items.append(item_text)

        return items

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
