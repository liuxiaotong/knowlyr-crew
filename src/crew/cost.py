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
# 来源：各厂商官方定价，2025-Q1
COST_PER_1K: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-6": {"input": 0.015, "output": 0.075},
    "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-sonnet-4-5-20250929": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5-20251001": {"input": 0.0008, "output": 0.004},
    # Moonshot (Kimi) — ¥0.06/1K ≈ $0.0083/1K（按 7.2 汇率）
    "kimi-k2.5": {"input": 0.0083, "output": 0.0083},
    "kimi2.5": {"input": 0.0083, "output": 0.0083},  # aiberm 代理名
    "moonshot-v1-auto": {"input": 0.0083, "output": 0.0083},
    # OpenAI
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    # Google
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    # DeepSeek
    "deepseek-chat": {"input": 0.0002, "output": 0.0008},
}

_DEFAULT_PRICE = {"input": 0.001, "output": 0.003}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """估算单次调用成本（USD）."""
    prices = COST_PER_1K.get(model, _DEFAULT_PRICE)
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1000


def enrich_result_with_cost(result: dict[str, Any]) -> dict[str, Any]:
    """向任务结果中追加 cost_usd 字段."""
    model = result.get("model", "")
    input_tokens = result.get("input_tokens", 0)
    output_tokens = result.get("output_tokens", 0)
    if input_tokens or output_tokens:
        result["cost_usd"] = round(estimate_cost(model, input_tokens, output_tokens), 6)
    return result


def query_cost_summary(
    registry: "TaskRegistry",
    employee: str | None = None,
    days: int = 7,
) -> dict[str, Any]:
    """从任务注册表查询成本汇总."""
    cutoff = datetime.now() - timedelta(days=days)

    by_employee: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}
    total_cost = 0.0
    total_tasks = 0

    for record in registry.list_recent(n=500):
        if record.created_at < cutoff:
            continue
        if record.status != "completed" or not record.result:
            continue
        if employee and record.target_name != employee:
            continue

        result = record.result
        emp_name = result.get("employee", record.target_name)
        model = result.get("model", "unknown")
        inp = result.get("input_tokens", 0)
        out = result.get("output_tokens", 0)
        cost = result.get("cost_usd", 0.0)

        if not cost and (inp or out):
            cost = estimate_cost(model, inp, out)

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
