"""Token 成本追踪 — 模型单价表、成本估算、历史查询."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crew.task_registry import TaskRegistry

logger = logging.getLogger(__name__)

# ── 模型单价表（每 1K tokens，USD）──
# 来源：各厂商官方定价，2025-Q2
COST_PER_1K: dict[str, dict[str, float]] = {
    # Anthropic（Opus 4.6/4.5 = $5/$25 per M, Sonnet 4/4.5 = $3/$15, Haiku 4.5 = $1/$5）
    "claude-opus-4-6": {"input": 0.005, "output": 0.025},
    "claude-opus-4-20250514": {"input": 0.005, "output": 0.025},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-sonnet-4-5-20250929": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5-20251001": {"input": 0.001, "output": 0.005},
    # Moonshot / Kimi（kimi-k2.5 = ¥4.2/¥21 per M → $0.6/$3 per M）
    "kimi-k2.5": {"input": 0.0006, "output": 0.003},
    "kimi2.5": {"input": 0.0006, "output": 0.003},
    "moonshot-v1-auto": {"input": 0.0006, "output": 0.003},
    "moonshot-v1-32k": {"input": 0.000694, "output": 0.002778},
    "moonshot-v1-8k": {"input": 0.000278, "output": 0.001389},
    # OpenAI
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    # Google
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    # DeepSeek
    "deepseek-chat": {"input": 0.0002, "output": 0.0008},
}

# 代理商定价覆盖（base_url → 模型 → 价格）
PROXY_PRICE_OVERRIDES: dict[str, dict[str, dict[str, float]]] = {
    "https://aiberm.com/v1": {
        "claude-opus-4-6": {"input": 0.00095, "output": 0.00475},
        "claude-opus-4-20250514": {"input": 0.00095, "output": 0.00475},
        "claude-sonnet-4-20250514": {"input": 0.0003, "output": 0.0015},
        "claude-sonnet-4-5-20250929": {"input": 0.0003, "output": 0.0015},
        "claude-haiku-4-5-20251001": {"input": 0.0001, "output": 0.0005},
    },
}

_DEFAULT_PRICE = {"input": 0.001, "output": 0.003}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    base_url: str | None = None,
) -> float:
    """估算单次调用成本（USD）."""
    if base_url:
        proxy = PROXY_PRICE_OVERRIDES.get(base_url.rstrip("/"), {})
        prices = proxy.get(model) or COST_PER_1K.get(model, _DEFAULT_PRICE)
    else:
        prices = COST_PER_1K.get(model, _DEFAULT_PRICE)
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1000


def enrich_result_with_cost(result: dict[str, Any]) -> dict[str, Any]:
    """向任务结果中追加 cost_usd 字段."""
    model = result.get("model", "")
    input_tokens = result.get("input_tokens", 0)
    output_tokens = result.get("output_tokens", 0)
    base_url = result.get("base_url")
    if input_tokens or output_tokens:
        result["cost_usd"] = round(
            estimate_cost(model, input_tokens, output_tokens, base_url=base_url), 6,
        )
    return result


# 触发源分类常量
WORK_TRIGGERS = {"github", "openclaw", "generic", "direct", "cron", "delegate_async", "delegate_chain"}
CHAT_TRIGGERS = {"feishu"}


def query_cost_summary(
    registry: "TaskRegistry",
    employee: str | None = None,
    days: int = 7,
    source: str | None = None,
) -> dict[str, Any]:
    """从任务注册表查询成本汇总.

    Args:
        source: 过滤触发源 — "work"(正式任务) / "chat"(闲聊) / None(全部)
    """
    cutoff = datetime.now() - timedelta(days=days)

    by_employee: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}
    by_trigger: dict[str, dict[str, Any]] = {}
    total_cost = 0.0
    total_tasks = 0

    for record in registry.list_recent(n=500):
        if record.created_at < cutoff:
            continue
        if record.status != "completed" or not record.result:
            continue
        if employee and record.target_name != employee:
            continue
        # 按触发源过滤
        if source == "work" and record.trigger in CHAT_TRIGGERS:
            continue
        if source == "chat" and record.trigger not in CHAT_TRIGGERS:
            continue

        result = record.result
        emp_name = result.get("employee", record.target_name)
        model = result.get("model", "unknown")
        inp = result.get("input_tokens", 0)
        out = result.get("output_tokens", 0)
        cost = result.get("cost_usd", 0.0)

        if not cost and (inp or out):
            base_url = result.get("base_url")
            cost = estimate_cost(model, inp, out, base_url=base_url)

        # 按员工汇总
        if emp_name not in by_employee:
            by_employee[emp_name] = {
                "cost_usd": 0.0, "tasks": 0, "input_tokens": 0, "output_tokens": 0,
            }
        by_employee[emp_name]["cost_usd"] += cost
        by_employee[emp_name]["tasks"] += 1
        by_employee[emp_name]["input_tokens"] += inp
        by_employee[emp_name]["output_tokens"] += out

        # 按模型汇总
        if model not in by_model:
            by_model[model] = {"cost_usd": 0.0, "tasks": 0}
        by_model[model]["cost_usd"] += cost
        by_model[model]["tasks"] += 1

        # 按触发源汇总
        trigger = record.trigger or "unknown"
        if trigger not in by_trigger:
            by_trigger[trigger] = {"cost_usd": 0.0, "tasks": 0}
        by_trigger[trigger]["cost_usd"] += cost
        by_trigger[trigger]["tasks"] += 1

        total_cost += cost
        total_tasks += 1

    def _round_dict(d: dict) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}

    return {
        "period_days": days,
        "total_cost_usd": round(total_cost, 4),
        "total_tasks": total_tasks,
        "by_employee": {
            k: _round_dict(v)
            for k, v in sorted(by_employee.items(), key=lambda x: -x[1]["cost_usd"])
        },
        "by_model": {
            k: _round_dict(v)
            for k, v in sorted(by_model.items(), key=lambda x: -x[1]["cost_usd"])
        },
        "by_trigger": {
            k: _round_dict(v)
            for k, v in sorted(by_trigger.items(), key=lambda x: -x[1]["cost_usd"])
        },
    }


# ── 质量预评分解析 ──


def parse_quality_score(output: str) -> dict[str, Any] | None:
    """从员工输出末尾解析结构化质量评分.

    格式: {"score": 72, "critical": 3, "warning": 5, "suggestion": 4}
    """
    pattern = r'\{[^{}]*"score"\s*:\s*\d+[^{}]*\}'
    matches = re.findall(pattern, output or "")
    for m in reversed(matches):  # 从末尾开始找
        try:
            data = json.loads(m)
            if "score" in data and isinstance(data["score"], (int, float)):
                return data
        except json.JSONDecodeError:
            continue
    return None
