"""灵魂自动进化 -- 从记忆中提炼行为准则候选.

高频使用的 pattern 自动晋升为 soul.md 行为准则候选，
被多条 correction 推翻的旧规则自动标记归档候选。
候选存入数据库，由管理员审批后才真正更新 soul。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from crew.config_store import get_config, put_config

logger = logging.getLogger(__name__)


@dataclass
class EvolutionCandidate:
    """进化候选项."""

    id: str = ""
    employee: str = ""
    action: str = ""  # "promote" 或 "archive"
    source_type: str = ""  # "pattern" 或 "correction"
    source_ids: list[str] = field(default_factory=list)
    content: str = ""  # 候选的行为准则文本
    reason: str = ""  # 为什么推荐这个变更
    confidence: float = 0.0  # 置信度 0-1
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典."""
        return asdict(self)


def _call_llm(prompt: str) -> str:
    """调用 LLM 生成文本.

    复用 memory_pipeline 中的 LLM 调用模式。

    Args:
        prompt: 提示文本

    Returns:
        LLM 响应文本

    Raises:
        RuntimeError: API key 缺失或 SDK 未安装
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY 环境变量未设置")

    try:
        import httpx

        timeout = httpx.Timeout(30.0)
    except ImportError:
        timeout = None  # type: ignore[assignment]

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except ImportError as e:
        raise RuntimeError("anthropic SDK 未安装") from e


def _keyword_overlap(keywords_a: list[str], keywords_b: list[str]) -> float:
    """计算两组关键词的 Jaccard 重叠度.

    Args:
        keywords_a: 第一组关键词
        keywords_b: 第二组关键词

    Returns:
        0.0-1.0 的重叠度
    """
    if not keywords_a or not keywords_b:
        return 0.0
    set_a = {k.lower() for k in keywords_a}
    set_b = {k.lower() for k in keywords_b}
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _get_existing_candidate_ids(employee: str) -> set[str]:
    """获取该员工已有候选的 source_ids，避免重复推荐.

    Returns:
        已在候选列表中的 source memory ID 集合
    """
    raw = get_config("soul_evolution", f"{employee}_candidates")
    if not raw:
        return set()
    try:
        candidates = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return set()

    existing_ids: set[str] = set()
    for c in candidates:
        for sid in c.get("source_ids", []):
            existing_ids.add(sid)
    return existing_ids


def find_promotion_candidates(
    employee: str,
    store: Any,
    min_verified: int = 3,
) -> list[EvolutionCandidate]:
    """找到可晋升的 pattern.

    查询条件：
    - category = 'pattern'
    - verified_count >= min_verified
    - superseded_by = '' (未被取代)
    - 不在已有候选中（避免重复推荐）

    对每个符合条件的 pattern：
    - 用 _call_llm 把 pattern content 改写为 soul.md 行为准则格式
    - 生成 reason（引用 verified_count 和使用场景）
    - confidence = min(1.0, verified_count / 5)

    Args:
        employee: 员工名
        store: MemoryStoreDB 实例
        min_verified: 最低验证次数阈值

    Returns:
        晋升候选列表
    """
    from crew.database import get_connection

    employee_resolved = store._resolve_to_character_name(employee)
    existing_ids = _get_existing_candidate_ids(employee_resolved)

    # 直接 SQL 查询高频验证的 pattern
    try:
        import psycopg2.extras

        cursor_factory = psycopg2.extras.RealDictCursor
    except ImportError:
        cursor_factory = None  # type: ignore[assignment]

    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=cursor_factory) if cursor_factory else conn.cursor()
        cur.execute(
            """
            SELECT id, employee, created_at, category, content,
                   source_session, confidence, superseded_by, ttl_days,
                   importance, last_accessed, tags, shared, visibility,
                   trigger_condition, applicability, origin_employee, verified_count,
                   classification, domain, keywords, linked_memories
            FROM memories
            WHERE employee = %s AND tenant_id = %s
              AND category = 'pattern'
              AND verified_count >= %s
              AND (superseded_by = '' OR superseded_by IS NULL)
            ORDER BY verified_count DESC
            """,
            (employee_resolved, store._tenant_id, min_verified),
        )
        rows = cur.fetchall()

    entries = [store._row_to_entry(row) for row in rows]

    # 过滤已有候选
    entries = [e for e in entries if e.id not in existing_ids]

    candidates: list[EvolutionCandidate] = []
    now = datetime.now(timezone.utc).isoformat()

    for entry in entries:
        verified = entry.verified_count
        # 用 LLM 改写为 soul 行为准则格式
        prompt = (
            f"将以下经验总结改写为一条简短的行为准则（祈使语气，一句话，中文）：\n\n"
            f"{entry.content}\n\n"
            f"只输出改写后的行为准则，不要解释。"
        )
        try:
            soul_rule = _call_llm(prompt).strip()
        except Exception as e:
            logger.warning("LLM 改写失败，使用原文: %s", e)
            soul_rule = entry.content

        reason = f"该 pattern 已被验证 {verified} 次，置信度高，建议晋升为行为准则。"

        candidate = EvolutionCandidate(
            id=uuid.uuid4().hex[:12],
            employee=employee_resolved,
            action="promote",
            source_type="pattern",
            source_ids=[entry.id],
            content=soul_rule,
            reason=reason,
            confidence=min(1.0, verified / 5),
            created_at=now,
        )
        candidates.append(candidate)

    return candidates


def find_archive_candidates(
    employee: str,
    store: Any,
    min_corrections: int = 2,
) -> list[EvolutionCandidate]:
    """找到应归档的旧规则.

    逻辑：
    1. 查 category='correction' 的记忆
    2. 按 tags 聚类（复用 _keyword_overlap）
    3. 同一主题的 corrections 数 >= min_corrections -> 生成 archive 候选
    4. content = 被推翻的规则描述
    5. reason = 引用具体 corrections

    Args:
        employee: 员工名
        store: MemoryStoreDB 实例
        min_corrections: 最低纠正次数阈值

    Returns:
        归档候选列表
    """
    # 查询所有 correction 记忆
    corrections = store.query(employee, category="correction", limit=200)
    if not corrections:
        return []

    existing_ids = _get_existing_candidate_ids(store._resolve_to_character_name(employee))

    # 按 tags 聚类
    clusters: list[list[Any]] = []
    for corr in corrections:
        if corr.id in existing_ids:
            continue

        placed = False
        for cluster in clusters:
            # 与 cluster 中第一条的 tags 比较
            overlap = _keyword_overlap(corr.tags, cluster[0].tags)
            if overlap >= 0.3:
                cluster.append(corr)
                placed = True
                break
        if not placed:
            clusters.append([corr])

    candidates: list[EvolutionCandidate] = []
    now = datetime.now(timezone.utc).isoformat()
    employee_resolved = store._resolve_to_character_name(employee)

    for cluster in clusters:
        if len(cluster) < min_corrections:
            continue

        # 合并 cluster 中的 correction 内容
        source_ids = [c.id for c in cluster]
        contents = [c.content for c in cluster]
        combined = "; ".join(contents[:5])

        reason = (
            f"共有 {len(cluster)} 条纠正记忆指向同一主题，"
            f"建议归档相关旧规则。涉及记忆: {', '.join(source_ids[:5])}"
        )

        candidate = EvolutionCandidate(
            id=uuid.uuid4().hex[:12],
            employee=employee_resolved,
            action="archive",
            source_type="correction",
            source_ids=source_ids,
            content=combined,
            reason=reason,
            confidence=min(1.0, len(cluster) / 4),
            created_at=now,
        )
        candidates.append(candidate)

    return candidates


def run_evolution_review(
    store: Any | None = None,
    employee: str | None = None,
) -> dict[str, Any]:
    """生成进化候选并存入数据库.

    Args:
        store: MemoryStoreDB 实例（None 则自动创建）
        employee: 可选，指定单个员工（None 则审查所有有记忆的员工）

    Returns:
        {
            "employees_reviewed": N,
            "promote_candidates": N,
            "archive_candidates": N,
            "candidates": [candidate_dict, ...]
        }
    """
    if store is None:
        from crew.memory import get_memory_store

        store = get_memory_store()

    # 确定要审查的员工列表
    if employee:
        employees = [employee]
    else:
        employees = store.list_employees()

    total_promote = 0
    total_archive = 0
    all_candidates: list[dict[str, Any]] = []

    for emp in employees:
        emp_resolved = store._resolve_to_character_name(emp)

        # 读取已有候选
        raw = get_config("soul_evolution", f"{emp_resolved}_candidates")
        existing: list[dict[str, Any]] = []
        if raw:
            try:
                existing = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                existing = []

        # 查找新候选
        promote = find_promotion_candidates(emp, store)
        archive = find_archive_candidates(emp, store)

        new_candidates = [c.to_dict() for c in promote + archive]

        if new_candidates:
            # 合并到已有候选
            merged = existing + new_candidates
            put_config(
                "soul_evolution",
                f"{emp_resolved}_candidates",
                json.dumps(merged, ensure_ascii=False),
            )

        total_promote += len(promote)
        total_archive += len(archive)
        all_candidates.extend(new_candidates)

    return {
        "employees_reviewed": len(employees),
        "promote_candidates": total_promote,
        "archive_candidates": total_archive,
        "candidates": all_candidates,
    }
