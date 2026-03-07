"""记忆管线 — Reflect -> Connect -> Store 三步处理.

Phase 3-1: 核心理念是没有坏记忆，只有未加工的原始数据。
但不是所有数据都值得进入记忆库。

Reflect: 用 LLM 提取结构化笔记，决定是否存储
Connect: 用关键词匹配找到关联记忆，决定 merge/link/new
Store:   执行实际的数据库写入
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import httpx

from crew.memory import MemoryEntry

if TYPE_CHECKING:
    from crew.memory_store_db import MemoryStoreDB

logger = logging.getLogger(__name__)

# ── 数据结构 ──


@dataclass
class ReflectResult:
    """Reflect 阶段的输出：结构化笔记."""

    store: bool
    content: str
    category: Literal["decision", "finding", "pattern", "correction", "estimate"]
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    context: str = ""


@dataclass
class ConnectResult:
    """Connect 阶段的输出：关联决策."""

    action: Literal["merge", "link", "new"]
    entry: MemoryEntry
    merged_entry_id: str | None = None  # merge 时被更新的已有记忆 ID


# ── Reflect 阶段 ──

_REFLECT_PROMPT = """\
你是一个记忆提炼器。分析以下原始文本，判断是否值得存储为长期记忆。

规则：
- 值得存储的：决策及理由、发现的事实/规律、工作模式/最佳实践、纠正（之前错了现在对了）、估算经验
- 不值得存储的：纯过程性描述（"我打开了文件"）、重复已知信息、无事实/决策/模式的闲聊
- 提炼时要保留关键信息，去掉冗余

员工：{employee}

原始文本：
{raw_text}

请返回 JSON（不要加 markdown 标记）：
{{
  "store": true或false,
  "content": "提炼后的内容（简洁但完整）",
  "category": "decision/finding/pattern/correction/estimate",
  "keywords": ["关键词1", "关键词2"],
  "tags": ["标签1"],
  "context": "什么场景下产生的"
}}
"""

_MAX_INPUT_LENGTH = 500


def _call_llm(prompt: str) -> str:
    """调用 Anthropic API 获取 LLM 响应.

    使用 anthropic SDK（项目已有依赖），模型 claude-sonnet-4-20250514。

    Args:
        prompt: 发送给 LLM 的提示文本

    Returns:
        LLM 响应的文本内容

    Raises:
        RuntimeError: API key 缺失或 SDK 未安装
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY 环境变量未设置")

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, timeout=httpx.Timeout(30.0))
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except ImportError as e:
        raise RuntimeError("anthropic SDK 未安装") from e


def reflect(raw_text: str, employee: str) -> ReflectResult | None:
    """Reflect 阶段：调 LLM 提取结构化笔记.

    Args:
        raw_text: 原始文本（超过 500 字自动截断）
        employee: 员工标识符

    Returns:
        ReflectResult 结构化笔记，若 LLM 判断 store=false 则返回 None
    """
    # 输入截断
    truncated = raw_text[:_MAX_INPUT_LENGTH] if len(raw_text) > _MAX_INPUT_LENGTH else raw_text

    prompt = _REFLECT_PROMPT.format(employee=employee, raw_text=truncated)

    try:
        response_text = _call_llm(prompt)

        # 尝试解析 JSON（容忍 markdown 代码块包裹）
        text = response_text.strip()
        if text.startswith("```"):
            # 去掉 ```json ... ``` 包裹
            lines = text.split("\n")
            text = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            )

        data = json.loads(text)

        # 验证必需字段
        if not isinstance(data, dict) or "store" not in data:
            logger.warning("Reflect LLM 返回格式异常: %s", response_text[:200])
            return None

        if not data["store"]:
            logger.debug("Reflect 决定跳过: employee=%s", employee)
            return None

        # 校验 category
        valid_categories = {"decision", "finding", "pattern", "correction", "estimate"}
        category = data.get("category", "finding")
        if category not in valid_categories:
            category = "finding"

        return ReflectResult(
            store=True,
            content=data.get("content", truncated),
            category=category,
            keywords=data.get("keywords", []),
            tags=data.get("tags", []),
            context=data.get("context", ""),
        )

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning("Reflect 阶段解析失败: %s", e)
        return None
    except Exception as e:
        logger.error("Reflect 阶段异常: %s", e)
        return None


# ── Connect 阶段 ──


def _keyword_overlap(keywords_a: list[str], keywords_b: list[str]) -> float:
    """计算两组关键词的重叠度.

    Args:
        keywords_a: 第一组关键词
        keywords_b: 第二组关键词

    Returns:
        重叠度 0.0-1.0（交集/并集，Jaccard 系数）
    """
    if not keywords_a or not keywords_b:
        return 0.0
    set_a = {k.lower() for k in keywords_a}
    set_b = {k.lower() for k in keywords_b}
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _find_candidates_by_keywords(
    keywords: list[str], employee: str, store: MemoryStoreDB, limit: int = 5
) -> list[MemoryEntry]:
    """用 ILIKE 匹配关键词，查已有记忆候选.

    Args:
        keywords: 要匹配的关键词列表
        employee: 员工标识符
        store: 数据库存储实例
        limit: 最多返回候选数

    Returns:
        匹配到的记忆条目列表（最多 limit 条）
    """
    if not keywords:
        return []

    from crew.database import get_connection

    # 用 ANY 匹配：记忆的 keywords 数组中有任一元素 ILIKE 任一搜索关键词
    # 构建条件：对每个搜索关键词，检查 keywords 数组是否包含匹配项
    conditions = []
    params: list[str | int] = []

    employee_resolved = store._resolve_to_character_name(employee)

    for kw in keywords[:10]:  # 最多 10 个关键词避免查询过大
        conditions.append(
            "EXISTS (SELECT 1 FROM unnest(keywords) AS k WHERE k ILIKE %s)"
        )
        params.append(f"%{kw}%")

    if not conditions:
        return []

    keyword_condition = " OR ".join(conditions)

    with get_connection() as conn:
        import psycopg2.extras

        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            f"""
            SELECT id, employee, created_at, category, content,
                   source_session, confidence, superseded_by, ttl_days,
                   importance, last_accessed, tags, shared, visibility,
                   trigger_condition, applicability, origin_employee, verified_count,
                   classification, domain,
                   keywords, linked_memories
            FROM memories
            WHERE employee = %s
              AND tenant_id = %s
              AND (superseded_by = '' OR superseded_by IS NULL)
              AND ({keyword_condition})
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (employee_resolved, store._tenant_id, *params, limit),
        )
        rows = cur.fetchall()

    return [store._row_to_entry(row) for row in rows]


def connect(
    note: ReflectResult,
    employee: str,
    store: MemoryStoreDB,
    **store_kwargs,
) -> ConnectResult:
    """Connect 阶段：找到关联记忆，决定 merge/link/new.

    逻辑：
    - 用 note.keywords 做 ILIKE 匹配，查已有记忆 Top-5
    - 计算 keyword 重叠度：
      - >= 70% 且 category 相同 -> merge
      - 30%-70% 或 category 不同 -> link
      - < 30% 或无候选 -> new

    Args:
        note: Reflect 阶段输出的结构化笔记
        employee: 员工标识符
        store: 数据库存储实例
        **store_kwargs: 透传给 store.add() 的额外参数

    Returns:
        ConnectResult 包含 action 和结果 entry
    """
    candidates = _find_candidates_by_keywords(note.keywords, employee, store)

    if not candidates:
        # 无候选 -> new
        entry = _store_new(note, employee, store, **store_kwargs)
        return ConnectResult(action="new", entry=entry)

    # 找最高重叠度的候选
    best_overlap = 0.0
    best_candidate: MemoryEntry | None = None

    for candidate in candidates:
        overlap = _keyword_overlap(note.keywords, candidate.keywords)
        if overlap > best_overlap:
            best_overlap = overlap
            best_candidate = candidate

    if best_candidate is None:
        entry = _store_new(note, employee, store, **store_kwargs)
        return ConnectResult(action="new", entry=entry)

    if best_overlap >= 0.7 and best_candidate.category == note.category:
        # merge: 更新已有记忆的 content 和 keywords
        merged_content = f"{best_candidate.content}\n---\n{note.content}"
        merged_keywords = list(set(best_candidate.keywords + note.keywords))

        store.update(
            best_candidate.id,
            employee,
            content=merged_content,
        )
        try:
            store.update_keywords(best_candidate.id, employee, merged_keywords)
        except Exception:
            logger.warning(
                "merge: update_keywords 失败 entry_id=%s，content 已更新",
                best_candidate.id,
            )

        # 返回更新后的 entry
        updated_entry = best_candidate.model_copy(
            update={
                "content": merged_content,
                "keywords": merged_keywords,
            }
        )
        return ConnectResult(
            action="merge", entry=updated_entry, merged_entry_id=best_candidate.id
        )

    elif best_overlap >= 0.3:
        # link: 新建 + 双向 linked_memories
        new_entry = _store_new(note, employee, store, **store_kwargs)

        try:
            # 更新新记忆的 linked_memories
            store.update_linked_memories(
                new_entry.id, employee, [best_candidate.id]
            )

            # 更新旧记忆的 linked_memories（双向）
            old_linked = list(best_candidate.linked_memories) + [new_entry.id]
            store.update_linked_memories(
                best_candidate.id, employee, old_linked
            )

            # 更新旧记忆的 keywords（合并新关键词）
            merged_kw = list(set(best_candidate.keywords + note.keywords))
            store.update_keywords(best_candidate.id, employee, merged_kw)
        except Exception:
            logger.warning(
                "link: 关联更新部分失败 new_id=%s, candidate_id=%s，"
                "新记忆已写入但关联可能不完整",
                new_entry.id,
                best_candidate.id,
            )

        updated_entry = new_entry.model_copy(
            update={"linked_memories": [best_candidate.id]}
        )
        return ConnectResult(action="link", entry=updated_entry)

    else:
        # < 30% -> new
        entry = _store_new(note, employee, store, **store_kwargs)
        return ConnectResult(action="new", entry=entry)


def _store_new(
    note: ReflectResult,
    employee: str,
    store: MemoryStoreDB,
    **store_kwargs,
) -> MemoryEntry:
    """Store 阶段 - 新建记忆条目.

    Args:
        note: Reflect 阶段输出
        employee: 员工标识符
        store: 数据库存储实例
        **store_kwargs: 透传给 store.add() 的额外参数
            (source_session, ttl_days, shared, confidence 等)

    Returns:
        新创建的 MemoryEntry
    """
    entry = store.add(
        employee=employee,
        category=note.category,
        content=note.content,
        tags=note.tags,
        **store_kwargs,
    )
    # 更新 keywords（add() 默认是空的）
    if note.keywords:
        store.update_keywords(entry.id, employee, note.keywords)
        entry = entry.model_copy(update={"keywords": note.keywords})
    return entry


# ── 顶层入口 ──


def process_memory(
    raw_text: str,
    employee: str,
    store: MemoryStoreDB | None = None,
    skip_reflect: bool = False,
    source_session: str = "",
    **kwargs,
) -> MemoryEntry | None:
    """记忆管线顶层入口：Reflect -> Connect -> Store.

    Args:
        raw_text: 原始文本
        employee: 员工标识符
        store: 数据库存储实例（None 则自动获取）
        skip_reflect: 跳过 Reflect 阶段（给 add_memory 等已结构化的路径用）
        source_session: 来源会话 ID
        **kwargs: skip_reflect=True 时，用于构造 ReflectResult 的参数：
            content, category, keywords, tags, context
            以及透传给 store.add() 的参数：
            ttl_days, shared, confidence, trigger_condition,
            applicability, origin_employee, classification, domain

    Returns:
        处理后的 MemoryEntry，Reflect 决定跳过时返回 None
    """
    if store is None:
        from crew.memory import get_memory_store

        store = get_memory_store()

    # 分离 ReflectResult 参数和 store.add() 透传参数
    _reflect_keys = {"content", "category", "keywords", "tags", "context"}
    _store_extra = {k: v for k, v in kwargs.items() if k not in _reflect_keys}
    if source_session:
        _store_extra["source_session"] = source_session

    if skip_reflect:
        # 直接构造 ReflectResult
        note = ReflectResult(
            store=True,
            content=kwargs.get("content", raw_text),
            category=kwargs.get("category", "finding"),
            keywords=kwargs.get("keywords", []),
            tags=kwargs.get("tags", []),
            context=kwargs.get("context", ""),
        )
    else:
        note = reflect(raw_text, employee)
        if note is None:
            return None

    result = connect(note, employee, store, **_store_extra)
    return result.entry
