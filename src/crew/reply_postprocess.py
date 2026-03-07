"""回复后处理 — 决定是否将对话产出推送到记忆.

推送门槛：预计 95% 对话不触发，只有含决策/纠正/长产出的对话才写入记忆。
"""

import atexit
import logging
import re
import threading
import weakref

from crew.memory import get_memory_store
from crew.memory_cache import invalidate
from crew.memory_pipeline import process_memory

# daemon=False 线程退出前等待完成，避免记忆写入被截断
_active_threads: list[weakref.ref] = []

def _join_active_threads():
    for ref in _active_threads:
        t = ref()
        if t is not None and t.is_alive():
            t.join(timeout=15)

atexit.register(_join_active_threads)

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


def _do_push(employee: str, reply: str, session_id: str, store) -> None:
    """后台线程执行记忆管线."""
    try:
        entry = process_memory(
            raw_text=reply,
            employee=employee,
            store=store,
            skip_reflect=False,
            source_session=session_id,
        )
        if entry:
            invalidate(employee)
            logger.info(
                "记忆管线推送成功: employee=%s id=%s",
                employee,
                entry.id,
            )
        else:
            logger.info("记忆管线决定跳过: employee=%s", employee)
    except Exception as e:
        logger.error("记忆管线推送失败: employee=%s error=%s", employee, e)


def push_if_needed(
    *,
    employee: str,
    reply: str,
    turn_count: int = 1,
    session_id: str = "",
    store=None,
    max_retries: int = 2,
    timeout: float = 10.0,
) -> bool:
    """检查并推送回复到记忆（如果满足门槛）.

    LLM 管线调用在后台线程异步执行，不阻塞调用方。

    Args:
        employee: 员工名称
        reply: 回复文本
        turn_count: 对话轮数
        session_id: 会话 ID（用于去重）
        store: MemoryStore 实例
        max_retries: 保留参数（已废弃，管线内部有错误处理）
        timeout: 保留参数（已废弃）

    Returns:
        True 如果已触发管线（不等待结果），False 如果未触发
    """
    do_push, category = should_push(reply, turn_count)
    if not do_push or not category:
        return False

    if store is None:
        store = get_memory_store()

    # 后台线程异步执行管线（LLM 调用 2-5 秒，不阻塞）
    t = threading.Thread(
        target=_do_push,
        args=(employee, reply, session_id, store),
        daemon=False,  # 非 daemon：进程退出前等待记忆写入完成
    )
    t.start()
    _active_threads.append(weakref.ref(t))
    return True
