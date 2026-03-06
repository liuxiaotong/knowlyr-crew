#!/usr/bin/env python3
"""Crew Skills 触发率测试脚本.

对每个 Skill 的正例/反例调用 API，统计触发率和误触率。
可独立运行，输出适合 CI 或人工查看。

用法:
    python scripts/test_skill_triggers.py
    python scripts/test_skill_triggers.py --testcase-dir scripts/skill_trigger_testcases
    python scripts/test_skill_triggers.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
DEFAULT_API_URL = "https://crew.knowlyr.com/api/skills/check-triggers"
DEFAULT_API_TOKEN = "X52I08vGWptmvtZxCMzX500odojsdv30k-gEq0G4sp8"
DEFAULT_TESTCASE_DIR = Path(__file__).parent / "skill_trigger_testcases"

# 阈值
POSITIVE_HIT_RATE_THRESHOLD = 0.80  # 正例命中率 ≥ 80%
FALSE_TRIGGER_RATE_THRESHOLD = 0.10  # 误触率 ≤ 10%
REQUEST_INTERVAL = 1.0  # 每次 API 调用间隔（秒），避免 429


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class CaseResult:
    """单条用例的检测结果."""

    text: str
    expected_trigger: bool
    actual_trigger: bool
    triggered_skills: list[str]

    @property
    def correct(self) -> bool:
        return self.expected_trigger == self.actual_trigger


@dataclass
class SkillTestResult:
    """单个 Skill 的测试汇总."""

    employee: str
    skill_name: str
    trigger_type: str
    positive_results: list[CaseResult] = field(default_factory=list)
    negative_results: list[CaseResult] = field(default_factory=list)

    @property
    def positive_hit_rate(self) -> float:
        if not self.positive_results:
            return 1.0
        hits = sum(1 for r in self.positive_results if r.actual_trigger)
        return hits / len(self.positive_results)

    @property
    def false_trigger_rate(self) -> float:
        if not self.negative_results:
            return 0.0
        false_triggers = sum(1 for r in self.negative_results if r.actual_trigger)
        return false_triggers / len(self.negative_results)

    @property
    def passed(self) -> bool:
        """判断此 Skill 是否通过测试."""
        # always 类型的 Skill 只检查正例
        if self.trigger_type == "always":
            return self.positive_hit_rate >= POSITIVE_HIT_RATE_THRESHOLD
        return (
            self.positive_hit_rate >= POSITIVE_HIT_RATE_THRESHOLD
            and self.false_trigger_rate <= FALSE_TRIGGER_RATE_THRESHOLD
        )

    @property
    def missed_positives(self) -> list[str]:
        return [r.text for r in self.positive_results if not r.actual_trigger]

    @property
    def false_negatives(self) -> list[str]:
        return [r.text for r in self.negative_results if r.actual_trigger]


# ---------------------------------------------------------------------------
# API 调用
# ---------------------------------------------------------------------------
def check_triggers(
    employee: str,
    task: str,
    api_url: str = DEFAULT_API_URL,
    api_token: str = DEFAULT_API_TOKEN,
    max_retries: int = 5,
) -> list[dict[str, Any]]:
    """调用 check-triggers API，返回 triggered_skills 数组.

    遇到 429 自动重试（指数退避）。
    """
    import urllib.error
    import urllib.request

    payload = json.dumps({"employee": employee, "task": task, "context": {}}).encode()

    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            api_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            return data.get("triggered_skills", [])
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries:
                wait = 3 * (attempt + 1)  # 3, 6, 9, 12, 15 秒
                time.sleep(wait)
                continue
            raise

    # unreachable, but keeps mypy happy
    return []  # pragma: no cover


# ---------------------------------------------------------------------------
# 测试用例加载
# ---------------------------------------------------------------------------
def load_testcases(testcase_dir: Path) -> list[dict[str, Any]]:
    """加载目录下所有 YAML 测试用例文件."""
    testcases: list[dict[str, Any]] = []
    for path in sorted(testcase_dir.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data and "skills" in data:
            testcases.append(data)
    return testcases


# ---------------------------------------------------------------------------
# 核心测试逻辑
# ---------------------------------------------------------------------------
def run_skill_test(
    employee: str,
    skill_config: dict[str, Any],
    api_url: str = DEFAULT_API_URL,
    api_token: str = DEFAULT_API_TOKEN,
    verbose: bool = False,
) -> SkillTestResult:
    """对单个 Skill 执行触发率测试."""
    skill_name: str = skill_config["skill"]
    trigger_type: str = skill_config.get("trigger_type", "keyword")
    positive_cases: list[str] = skill_config.get("positive_cases", [])
    negative_cases: list[str] = skill_config.get("negative_cases", [])

    result = SkillTestResult(
        employee=employee,
        skill_name=skill_name,
        trigger_type=trigger_type,
    )

    # 正例测试
    for text in positive_cases:
        time.sleep(REQUEST_INTERVAL)
        triggered = check_triggers(employee, text, api_url, api_token)
        triggered_names = [s["name"] for s in triggered]
        hit = skill_name in triggered_names
        result.positive_results.append(
            CaseResult(
                text=text,
                expected_trigger=True,
                actual_trigger=hit,
                triggered_skills=triggered_names,
            )
        )
        if verbose:
            mark = "OK" if hit else "MISS"
            print(f'    [{mark}] + "{text}" -> {triggered_names}')

    # 反例测试
    for text in negative_cases:
        time.sleep(REQUEST_INTERVAL)
        triggered = check_triggers(employee, text, api_url, api_token)
        triggered_names = [s["name"] for s in triggered]
        hit = skill_name in triggered_names
        result.negative_results.append(
            CaseResult(
                text=text,
                expected_trigger=False,
                actual_trigger=hit,
                triggered_skills=triggered_names,
            )
        )
        if verbose:
            mark = "OK" if not hit else "FALSE"
            print(f'    [{mark}] - "{text}" -> {triggered_names}')

    return result


def run_all_tests(
    testcase_dir: Path,
    api_url: str = DEFAULT_API_URL,
    api_token: str = DEFAULT_API_TOKEN,
    verbose: bool = False,
) -> list[SkillTestResult]:
    """执行全部测试用例，返回结果列表."""
    testcases = load_testcases(testcase_dir)
    all_results: list[SkillTestResult] = []

    for tc in testcases:
        employee = tc["employee"]
        for skill_config in tc["skills"]:
            skill_name = skill_config["skill"]
            if verbose:
                print(f"\n  [{employee}] {skill_name}:")
            result = run_skill_test(
                employee=employee,
                skill_config=skill_config,
                api_url=api_url,
                api_token=api_token,
                verbose=verbose,
            )
            all_results.append(result)

    return all_results


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------
def print_report(results: list[SkillTestResult]) -> bool:
    """打印测试报告，返回是否全部通过."""
    print("\n" + "=" * 72)
    print("  Crew Skills 触发率测试报告")
    print("=" * 72)

    total_pass = 0
    total_fail = 0

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if r.passed:
            total_pass += 1
        else:
            total_fail += 1

        pos_info = f"正例命中 {r.positive_hit_rate:.0%} ({len(r.positive_results)})"
        if r.trigger_type == "always":
            neg_info = "（always 类型，无反例）"
        else:
            neg_info = f"误触 {r.false_trigger_rate:.0%} ({len(r.negative_results)})"

        print(f"\n  [{status}] {r.employee} / {r.skill_name}")
        print(f"         {pos_info} | {neg_info}")

        # 输出错误详情
        if r.missed_positives:
            print("         漏触发:")
            for text in r.missed_positives:
                print(f"           - {text}")
        if r.false_negatives:
            print("         误触发:")
            for text in r.false_negatives:
                print(f"           - {text}")

    print("\n" + "-" * 72)
    all_passed = total_fail == 0
    verdict = "ALL PASSED" if all_passed else "FAILED"
    print(f"  结果: {verdict}  (通过 {total_pass} / 失败 {total_fail})")
    print(
        f"  阈值: 正例命中 >= {POSITIVE_HIT_RATE_THRESHOLD:.0%} | "
        f"误触率 <= {FALSE_TRIGGER_RATE_THRESHOLD:.0%}"
    )
    print("=" * 72 + "\n")

    return all_passed


# ---------------------------------------------------------------------------
# JSON 输出（适合 CI）
# ---------------------------------------------------------------------------
def to_json_report(results: list[SkillTestResult]) -> dict[str, Any]:
    """生成 JSON 格式的报告."""
    skills_report = []
    for r in results:
        skills_report.append(
            {
                "employee": r.employee,
                "skill": r.skill_name,
                "trigger_type": r.trigger_type,
                "passed": r.passed,
                "positive_hit_rate": round(r.positive_hit_rate, 4),
                "false_trigger_rate": round(r.false_trigger_rate, 4),
                "positive_total": len(r.positive_results),
                "negative_total": len(r.negative_results),
                "missed_positives": r.missed_positives,
                "false_negatives": r.false_negatives,
            }
        )

    all_passed = all(r.passed for r in results)
    return {
        "passed": all_passed,
        "total_skills": len(results),
        "pass_count": sum(1 for r in results if r.passed),
        "fail_count": sum(1 for r in results if not r.passed),
        "thresholds": {
            "positive_hit_rate": POSITIVE_HIT_RATE_THRESHOLD,
            "false_trigger_rate": FALSE_TRIGGER_RATE_THRESHOLD,
        },
        "skills": skills_report,
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Crew Skills 触发率测试")
    parser.add_argument(
        "--testcase-dir",
        type=Path,
        default=DEFAULT_TESTCASE_DIR,
        help="测试用例 YAML 目录",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="check-triggers API URL",
    )
    parser.add_argument(
        "--api-token",
        default=DEFAULT_API_TOKEN,
        help="API Bearer token",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="输出每条用例详情")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式报告")
    args = parser.parse_args()

    start = time.time()

    if args.verbose:
        print("开始执行 Skills 触发率测试...\n")

    results = run_all_tests(
        testcase_dir=args.testcase_dir,
        api_url=args.api_url,
        api_token=args.api_token,
        verbose=args.verbose,
    )

    elapsed = time.time() - start

    if args.json:
        report = to_json_report(results)
        report["elapsed_seconds"] = round(elapsed, 2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        all_passed = print_report(results)
        print(f"  耗时: {elapsed:.1f}s\n")
        if not all_passed:
            sys.exit(1)


if __name__ == "__main__":
    main()
