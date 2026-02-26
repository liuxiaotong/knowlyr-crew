"""记忆缓存层 — 进程内缓存 + 增量检查，避免每次请求扫磁盘.

80% 请求缓存命中，零 I/O。prompt 注入恒定 <=800 token。
"""

import logging
import time
from dataclasses import dataclass, field

from crew.memory import MemoryStore

logger = logging.getLogger(__name__)

_TTL_SECONDS = 60  # 1 分钟内不重扫磁盘
_PROMPT_TOKEN_LIMIT = 800
_DEFAULT_LIMIT = 10
_DEFAULT_MIN_IMPORTANCE = 3


@dataclass
class MemorySnapshot:
    """单个员工的记忆快照."""

    employee: str
    entries: list = field(default_factory=list)  # 已过滤的 active 条目
    last_seq: int = 0  # JSONL 行数（版本号）
    fetched_at: float = 0.0  # monotonic time
    prompt_text: str = ""  # format_for_prompt 缓存


# 进程级缓存
_CACHE: dict[str, MemorySnapshot] = {}


def _count_lines(store: MemoryStore, employee: str) -> int:
    """快速统计 JSONL 文件行数（不解析 JSON）."""
    path = store._employee_file(employee)
    if not path.exists():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def _truncate_to_token_limit(text: str, limit: int = _PROMPT_TOKEN_LIMIT) -> str:
    """粗略按字符数截断（中文 1 字 ~= 1.5 token，英文 1 word ~= 1 token）.

    保守估算：每 2 字符约 1 token。
    """
    char_limit = limit * 2  # 粗略映射
    if len(text) <= char_limit:
        return text
    # 按行截断，保留完整行
    lines = text.split("\n")
    result_lines: list[str] = []
    total = 0
    for line in lines:
        if total + len(line) > char_limit:
            break
        result_lines.append(line)
        total += len(line) + 1  # +1 for newline
    return "\n".join(result_lines)


def get_prompt_cached(
    employee: str,
    query: str = "",
    *,
    store: MemoryStore | None = None,
    max_visibility: str = "open",
    team_members: list[str] | None = None,
    employee_tags: list[str] | None = None,
) -> str:
    """获取员工记忆的 prompt 文本，带进程内缓存.

    缓存策略：
    - TTL 内 + JSONL 行数不变 → 零 I/O 返回缓存
    - TTL 过期或行数变化 → 重新加载

    Args:
        employee: 员工名称
        query: 查询上下文（语义搜索用）
        store: MemoryStore 实例（不传则新建）
        max_visibility: 可见性上限
        team_members: 同团队成员
        employee_tags: 员工标签

    Returns:
        截断到 800 token 的 prompt 文本
    """
    if store is None:
        store = MemoryStore()

    now = time.monotonic()
    cached = _CACHE.get(employee)

    if cached is not None:
        age = now - cached.fetched_at
        if age < _TTL_SECONDS:
            # TTL 内，再检查行数是否变化
            current_seq = _count_lines(store, employee)
            if current_seq == cached.last_seq:
                logger.debug("记忆缓存命中: %s (age=%.1fs)", employee, age)
                return cached.prompt_text

    # 缓存未命中，重新加载
    logger.debug("记忆缓存未命中: %s，重新加载", employee)

    prompt_text = store.format_for_prompt(
        employee,
        limit=_DEFAULT_LIMIT,
        query=query,
        employee_tags=employee_tags,
        max_visibility=max_visibility,
        team_members=team_members,
    )
    prompt_text = _truncate_to_token_limit(prompt_text, _PROMPT_TOKEN_LIMIT)
    current_seq = _count_lines(store, employee)

    _CACHE[employee] = MemorySnapshot(
        employee=employee,
        entries=[],  # 不缓存条目本身，只缓存 prompt 文本
        last_seq=current_seq,
        fetched_at=now,
        prompt_text=prompt_text,
    )

    return prompt_text


def invalidate(employee: str) -> None:
    """手动失效指定员工的缓存（写入后调用）."""
    _CACHE.pop(employee, None)


def invalidate_all() -> None:
    """清空全部缓存."""
    _CACHE.clear()
