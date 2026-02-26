"""回复后处理 — 决定是否将对话产出推送到记忆.

推送门槛：预计 95% 对话不触发，只有含决策/纠正/长产出的对话才写入记忆。
"""

import logging
import re

from crew.memory import MemoryStore
from crew.memory_cache import invalidate

logger = logging.getLogger(__name__)

# 决策关键词（触发 decision 类记忆）
_DECISION_KEYWORDS = re.compile(r"决定|统一|不再|确定|方案定|最终选|采用|弃用|禁止")

# 纠正关键词（触发 correction 类记忆）
_CORRECTION_KEYWORDS = re.compile(r"其实|说错了|纠正|更正|之前错|搞错|误解|实际上应该")

# 产出长度阈值（字符数）
_LONG_OUTPUT_THRESHOLD = 200

# 最小对话轮数（低于此轮数不检查长产出）
_MIN_TURNS_FOR_FINDING = 3


def should_push(
    reply: str,
    turn_count: int = 1,
) -> tuple[bool, str]:
    """判断是否应该将回复推送到记忆.

    Args:
        reply: 回复文本
        turn_count: 对话轮数

    Returns:
        (should_push, category) — category 为 "decision" / "correction" / "finding" / ""
    """
    if not reply or not reply.strip():
        return False, ""

    # 决策词 → 推 decision
    if _DECISION_KEYWORDS.search(reply):
        return True, "decision"

    # 纠正词 → 推 correction
    if _CORRECTION_KEYWORDS.search(reply):
        return True, "correction"

    # 长产出 + 多轮对话 → 推 finding
    if turn_count >= _MIN_TURNS_FOR_FINDING and len(reply) > _LONG_OUTPUT_THRESHOLD:
        return True, "finding"

    return False, ""


def push_if_needed(
    *,
    employee: str,
    reply: str,
    turn_count: int = 1,
    session_id: str = "",
    store: MemoryStore | None = None,
    max_retries: int = 2,
    timeout: float = 10.0,
) -> bool:
    """检查并推送回复到记忆（如果满足门槛）.

    Args:
        employee: 员工名称
        reply: 回复文本
        turn_count: 对话轮数
        session_id: 会话 ID（用于去重）
        store: MemoryStore 实例
        max_retries: 写入失败重试次数
        timeout: 超时秒数（保留字段，当前同步写入）

    Returns:
        True 如果写入了记忆，False 如果未触发或写入失败
    """
    do_push, category = should_push(reply, turn_count)
    if not do_push or not category:
        return False

    if store is None:
        store = MemoryStore()

    # 幂等校验：用 session_id 去重
    if session_id:
        existing = store.query(employee, limit=50)
        for entry in existing:
            if entry.source_session == session_id and entry.category == category:
                logger.debug(
                    "记忆推送跳过（幂等）: employee=%s session=%s category=%s",
                    employee,
                    session_id,
                    category,
                )
                return False

    # 提取摘要（取前 300 字符作为记忆内容）
    summary = reply.strip()[:300]
    if len(reply.strip()) > 300:
        summary += "..."

    # 写入，带重试
    for attempt in range(max_retries + 1):
        try:
            store.add(
                employee=employee,
                category=category,
                content=summary,
                source_session=session_id,
                tags=["auto-push", f"turns-{turn_count}"],
            )
            # 写入后失效缓存
            invalidate(employee)
            logger.info(
                "记忆推送成功: employee=%s category=%s session=%s",
                employee,
                category,
                session_id,
            )
            return True
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    "记忆推送重试 (%d/%d): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
            else:
                logger.error(
                    "记忆推送最终失败: employee=%s category=%s error=%s",
                    employee,
                    category,
                    e,
                )

    return False
