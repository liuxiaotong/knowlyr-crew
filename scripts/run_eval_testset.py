#!/usr/bin/env python3
"""标准测试集运行脚本 -- 用预定义测试用例评估员工能力.

用法:
    # 测试一个员工
    python scripts/run_eval_testset.py --employee code-reviewer

    # 测试所有员工
    python scripts/run_eval_testset.py --all

    # 跑单道题
    python scripts/run_eval_testset.py --employee code-reviewer --case cr-001

    # 只看测试计划不执行
    python scripts/run_eval_testset.py --employee code-reviewer --dry-run

输出:
    .crew/evaluations/testset-{date}.jsonl   -- 逐条评分结果
    终端输出对比报告（与上次测试结果比较）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# 项目路径
CREW_ROOT = Path(__file__).resolve().parent.parent
TESTSETS_DIR = Path(__file__).resolve().parent / "eval_testsets"
EVALUATIONS_DIR = CREW_ROOT / ".crew" / "evaluations"

# 确保 crew 模块和 scripts 可导入
sys.path.insert(0, str(CREW_ROOT / "src"))
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8765"


# ── 测试集加载 ──────────────────────────────────────────────────────


def load_testset(employee: str) -> dict[str, Any] | None:
    """加载指定员工的测试集 JSON."""
    path = TESTSETS_DIR / f"{employee}.json"
    if not path.exists():
        logger.error("测试集不存在: %s", path)
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_available_testsets() -> list[str]:
    """列出所有可用的测试集员工名."""
    if not TESTSETS_DIR.exists():
        return []
    return sorted(p.stem for p in TESTSETS_DIR.glob("*.json"))


# ── 运行测试 ────────────────────────────────────────────────────────


def run_test_case(
    employee: str,
    test_case: dict[str, Any],
    *,
    base_url: str = DEFAULT_BASE_URL,
    api_token: str = "",
    model: str = "",
    timeout: float = 120.0,
) -> dict[str, Any]:
    """对单道测试题调用 crew API 运行员工，返回结果.

    Returns:
        包含 output, behavior_matches, score 等字段的 dict
    """
    payload: dict[str, Any] = {
        "task": test_case["task"],
        "format": "markdown",
    }
    if model:
        payload["model"] = model

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    url = f"{base_url}/run/employee/{employee}"
    logger.info("  调用 %s ...", url)

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.warning("  HTTP 错误: %s %s", e.response.status_code, e.response.text[:200])
        return {
            "case_id": test_case["id"],
            "success": False,
            "error": f"HTTP {e.response.status_code}",
            "output": "",
            "behavior_matches": {},
        }
    except Exception as e:
        logger.warning("  请求失败: %s", e)
        return {
            "case_id": test_case["id"],
            "success": False,
            "error": str(e),
            "output": "",
            "behavior_matches": {},
        }

    # 提取输出文本
    output = data.get("result", "") or data.get("output", "") or data.get("content", "")

    # 基础行为匹配：检查输出是否包含期望行为的关键词
    behavior_matches = _check_behaviors(output, test_case.get("expected_behaviors", []))

    return {
        "case_id": test_case["id"],
        "success": True,
        "output": output,
        "behavior_matches": behavior_matches,
    }


def _check_behaviors(output: str, expected_behaviors: list[str]) -> dict[str, bool]:
    """检查输出是否包含期望行为的关键词.

    简单的字符串匹配，每个 expected_behavior 拆分为关键词，
    检查输出中是否包含大部分关键词。

    TODO: S7 — 当前为粗粒度关键词匹配，考虑引入语义相似度或 LLM 判断提升准确性。
    """
    output_lower = output.lower()
    results: dict[str, bool] = {}

    for behavior in expected_behaviors:
        # 将行为描述拆分为关键词
        keywords = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z_]+", behavior.lower())
        if not keywords:
            results[behavior] = False
            continue
        # 至少匹配一半关键词算通过
        matched = sum(1 for kw in keywords if kw in output_lower)
        results[behavior] = matched >= max(1, len(keywords) // 2)

    return results


# ── 评分 ────────────────────────────────────────────────────────────


def _score_test_result(
    employee: str,
    test_case: dict[str, Any],
    run_result: dict[str, Any],
) -> dict[str, Any]:
    """评分单条测试结果，复用 daily_eval 的评分逻辑."""
    from daily_eval import _fallback_score, score_trajectory

    if not run_result.get("success"):
        return {
            "total_score": 0.0,
            "outcome_score": 0.0,
            "process_score": 0.0,
            "rubric_scores": {},
            "engine": "error",
        }

    output = run_result.get("output", "")

    # 构造 trajectory 格式以复用 score_trajectory
    trajectory = {
        "task": test_case["task"],
        "steps": [
            {
                "tool": "respond",
                "params": {},
                "output": output,
            }
        ],
        "outcome": {"success": True},
        "metadata": {
            "employee": employee,
            "model": "",
            "timestamp": datetime.now().isoformat(),
        },
    }

    try:
        score_result = score_trajectory(trajectory, employee)
    except Exception:
        score_result = _fallback_score(trajectory)

    # 用行为匹配结果调整分数
    behavior_matches = run_result.get("behavior_matches", {})
    if behavior_matches:
        match_rate = sum(behavior_matches.values()) / len(behavior_matches)
        # 行为匹配占 30% 权重
        adjusted_total = score_result["total_score"] * 0.7 + match_rate * 0.3
        score_result["total_score"] = round(adjusted_total, 4)
        score_result["rubric_scores"]["behavior_match_rate"] = round(match_rate, 4)

    return score_result


# ── 结果输出 ─────────────────────────────────────────────────────────


def _write_results(results: list[dict[str, Any]], date_str: str) -> Path:
    """写入评分结果到 .crew/evaluations/testset-{date}.jsonl."""
    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVALUATIONS_DIR / f"testset-{date_str}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("结果写入: %s (%d 条)", output_path, len(results))
    return output_path


def _find_previous_results(date_str: str) -> list[dict[str, Any]] | None:
    """查找上一次的测试结果用于对比."""
    if not EVALUATIONS_DIR.exists():
        return None
    testset_files = sorted(EVALUATIONS_DIR.glob("testset-*.jsonl"))
    current_file = EVALUATIONS_DIR / f"testset-{date_str}.jsonl"
    # 找比当前日期早的最近一个文件
    previous = None
    for f in testset_files:
        if f == current_file:
            continue
        if f.stem < f"testset-{date_str}":
            previous = f
    if previous is None:
        return None
    results = []
    for line in previous.read_text("utf-8").strip().split("\n"):
        if line.strip():
            results.append(json.loads(line))
    return results


def _print_comparison(current: list[dict[str, Any]], previous: list[dict[str, Any]] | None) -> None:
    """打印当前结果与上次结果的对比."""
    print("\n" + "=" * 60)
    print("测试集评估结果")
    print("=" * 60)

    # 按员工分组
    by_employee: dict[str, list[dict[str, Any]]] = {}
    for r in current:
        emp = r.get("employee", "unknown")
        by_employee.setdefault(emp, []).append(r)

    # 上次数据按员工分组
    prev_by_employee: dict[str, float] = {}
    if previous:
        for r in previous:
            emp = r.get("employee", "unknown")
            prev_by_employee.setdefault(emp, []).append(r)
        prev_avg = {
            emp: sum(r["total_score"] for r in rs) / len(rs)
            for emp, rs in prev_by_employee.items()
        }
    else:
        prev_avg = {}

    print(f"\n{'员工':<25} {'本次均分':>8} {'上次均分':>8} {'变化':>8} {'通过/总数':>10}")
    print("-" * 65)

    for emp in sorted(by_employee):
        results = by_employee[emp]
        scores = [r["total_score"] for r in results]
        avg = sum(scores) / len(scores)

        passed = sum(1 for r in results if r.get("success", False))

        prev = prev_avg.get(emp)
        if prev is not None:
            delta = avg - prev
            delta_str = f"{delta:+.2f}"
        else:
            delta_str = "  N/A"

        prev_str = f"{prev:.2f}" if prev is not None else "   N/A"
        print(f"{emp:<25} {avg:>8.2f} {prev_str:>8} {delta_str:>8} {passed:>4}/{len(results):<4}")

    # 总体
    all_scores = [r["total_score"] for r in current]
    total_avg = sum(all_scores) / len(all_scores) if all_scores else 0
    print("-" * 65)
    print(f"{'总体':<25} {total_avg:>8.2f}")
    print()


# ── 主流程 ───────────────────────────────────────────────────────────


def run_testset(
    employee: str | None = None,
    case_id: str | None = None,
    *,
    run_all: bool = False,
    base_url: str = DEFAULT_BASE_URL,
    api_token: str = "",
    model: str = "",
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """运行标准测试集.

    Args:
        employee: 指定员工名
        case_id: 指定单道题 ID
        run_all: 测试所有员工
        base_url: crew API 地址
        api_token: API token
        model: 覆盖员工默认模型
        dry_run: 只打印测试计划不执行

    Returns:
        评分结果列表
    """
    # 确定要测试的员工列表
    if run_all:
        employees = list_available_testsets()
    elif employee:
        employees = [employee]
    else:
        logger.error("请指定 --employee 或 --all")
        return []

    if not employees:
        logger.error("没有找到可用的测试集")
        return []

    # 收集测试计划
    plan: list[tuple[str, dict[str, Any], dict[str, Any]]] = []  # (employee, testset, case)
    for emp in employees:
        testset = load_testset(emp)
        if testset is None:
            continue
        for tc in testset.get("test_cases", []):
            if case_id and tc["id"] != case_id:
                continue
            plan.append((emp, testset, tc))

    if not plan:
        logger.error("没有匹配的测试用例")
        return []

    # 打印测试计划
    logger.info("测试计划: %d 道题，%d 个员工", len(plan), len(employees))
    for emp, testset, tc in plan:
        char_name = testset.get("character_name", "")
        difficulty = tc.get("difficulty", "?")
        task_preview = tc["task"][:40].replace("\n", " ")
        logger.info("  [%s] %s(%s) %s: %s...", difficulty, emp, char_name, tc["id"], task_preview)

    if dry_run:
        logger.info("Dry run 模式，不执行测试")
        return []

    # 执行测试
    date_str = datetime.now().strftime("%Y-%m-%d")
    results: list[dict[str, Any]] = []

    for i, (emp, testset, tc) in enumerate(plan):
        logger.info("运行 [%d/%d] %s / %s ...", i + 1, len(plan), emp, tc["id"])

        run_result = run_test_case(
            emp,
            tc,
            base_url=base_url,
            api_token=api_token,
            model=model,
        )

        score_result = _score_test_result(emp, tc, run_result)

        result = {
            "employee": emp,
            "character_name": testset.get("character_name", ""),
            "domain": testset.get("domain", ""),
            "case_id": tc["id"],
            "difficulty": tc.get("difficulty", ""),
            "task": tc["task"],
            "expected_behaviors": tc.get("expected_behaviors", []),
            "success": run_result.get("success", False),
            "behavior_matches": run_result.get("behavior_matches", {}),
            "output_preview": (run_result.get("output", "") or "")[:200],
            **score_result,
            "scored_at": f"{date_str}T{datetime.now().strftime('%H:%M:%S')}",
        }
        results.append(result)

        behavior_matches = run_result.get("behavior_matches", {})
        matched = sum(behavior_matches.values()) if behavior_matches else 0
        total_behaviors = len(behavior_matches) if behavior_matches else 0
        logger.info(
            "  得分: %.2f | 行为匹配: %d/%d",
            score_result["total_score"],
            matched,
            total_behaviors,
        )

    # 写入结果
    _write_results(results, date_str)

    # 对比报告
    previous = _find_previous_results(date_str)
    _print_comparison(results, previous)

    return results


# ── CLI 入口 ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="标准测试集运行脚本")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--employee", help="指定员工名（如 code-reviewer）")
    group.add_argument("--all", action="store_true", dest="run_all", help="测试所有员工")
    parser.add_argument("--case", help="指定单道题 ID（如 cr-001）")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CREW_API_URL", DEFAULT_BASE_URL),
        help=f"crew API 地址（默认 {DEFAULT_BASE_URL}）",
    )
    parser.add_argument(
        "--api-token",
        default=os.environ.get("CREW_API_TOKEN", ""),
        help="API token（也可通过 CREW_API_TOKEN 环境变量）",
    )
    parser.add_argument("--model", default="", help="覆盖员工默认模型")
    parser.add_argument("--dry-run", action="store_true", help="只打印测试计划不执行")

    args = parser.parse_args()

    results = run_testset(
        employee=args.employee,
        case_id=args.case,
        run_all=args.run_all,
        base_url=args.base_url,
        api_token=args.api_token,
        model=args.model,
        dry_run=args.dry_run,
    )

    if results:
        total = len(results)
        success = sum(1 for r in results if r.get("success"))
        avg_score = sum(r["total_score"] for r in results) / total if total else 0
        print(f"\n=== 测试完成 ===")
        print(f"总数: {total}")
        print(f"成功: {success}")
        print(f"平均分: {avg_score:.2f}")


if __name__ == "__main__":
    main()
