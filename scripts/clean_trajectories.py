#!/usr/bin/env python3
"""一次性轨迹数据清洗脚本.

对已入库的 .crew/trajectories/trajectories.jsonl 做三类清洗:

1. task 是 dict → 提取 description 字段
2. task 以"你是"开头且超 200 字 → 尝试从中提取 ## 任务 后面的实际任务描述
3. 空壳轨迹（steps 全空）→ 移除

用法:
    # 预览（dry-run，不实际修改文件）
    python scripts/clean_trajectories.py

    # 实际执行
    python scripts/clean_trajectories.py --apply

    # 指定文件路径
    python scripts/clean_trajectories.py --file /path/to/trajectories.jsonl --apply
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

CREW_ROOT = Path(__file__).resolve().parent.parent

# 确保 crew 模块可导入
sys.path.insert(0, str(CREW_ROOT / "src"))

DEFAULT_FILE = CREW_ROOT / ".crew" / "trajectories" / "trajectories.jsonl"

# ── 检测函数 ──────────────────────────────────────────────────────────


def _is_hollow(data: dict[str, Any]) -> bool:
    """判断轨迹是否为空壳 — 委托给 crew.trajectory 公共实现."""
    from crew.trajectory import is_hollow_trajectory

    return is_hollow_trajectory(data)


def _is_soul_prompt_task(task: str) -> bool:
    """判断 task 是否为 soul prompt（以"你是"开头且超 200 字）."""
    return isinstance(task, str) and task.startswith("你是") and len(task) > 200


def _extract_task_from_soul_prompt(text: str) -> str | None:
    """尝试从 soul prompt 中提取 ## 任务 — 委托给 crew.trajectory 公共实现."""
    from crew.trajectory import extract_task_from_soul_prompt

    return extract_task_from_soul_prompt(text)


def _get_task_string(data: dict[str, Any]) -> str:
    """从轨迹数据中提取 task 字符串（兼容 dict 和 string 格式）."""
    task = data.get("task")
    if isinstance(task, dict):
        return task.get("description", "")
    if isinstance(task, str):
        return task
    return ""


# ── 清洗主逻辑 ────────────────────────────────────────────────────────


def clean_file(file_path: Path, *, apply: bool = False) -> dict[str, Any]:
    """清洗单个 trajectories.jsonl 文件.

    Returns:
        统计信息 dict
    """
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return {"error": "file_not_found"}

    lines = file_path.read_text("utf-8").strip().split("\n")
    lines = [line for line in lines if line.strip()]

    stats = {
        "total": len(lines),
        "task_dict_fixed": 0,
        "soul_prompt_fixed": 0,
        "soul_prompt_needs_review": 0,
        "hollow_removed": 0,
        "parse_errors": 0,
        "kept": 0,
    }

    cleaned: list[str] = []
    needs_review: list[dict[str, Any]] = []

    for i, line in enumerate(lines):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            stats["parse_errors"] += 1
            continue

        # 1. 空壳检测（最先处理，直接丢弃）
        if _is_hollow(data):
            stats["hollow_removed"] += 1
            continue

        # 2. task 是 dict → 提取 description
        task = data.get("task")
        if isinstance(task, dict):
            desc = task.get("description", "")
            data["task"] = desc
            stats["task_dict_fixed"] += 1

        # 3. task 是 soul prompt → 尝试提取 ## 任务
        task_str = data.get("task", "")
        if isinstance(task_str, str) and _is_soul_prompt_task(task_str):
            extracted = _extract_task_from_soul_prompt(task_str)
            if extracted:
                data["task"] = extracted
                stats["soul_prompt_fixed"] += 1
            else:
                stats["soul_prompt_needs_review"] += 1
                needs_review.append({
                    "line": i + 1,
                    "task_preview": task_str[:100] + "...",
                    "employee": (
                        data.get("metadata", {}).get("employee")
                        or data.get("employee", "unknown")
                    ),
                })

        cleaned.append(json.dumps(data, ensure_ascii=False))

    stats["kept"] = len(cleaned)

    # 写入
    if apply and cleaned:
        # 备份原文件
        backup_path = file_path.with_suffix(
            f".backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
        )
        shutil.copy2(file_path, backup_path)
        print(f"备份: {backup_path}")

        file_path.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
        print(f"已写入清洗后文件: {file_path}")

    # 打印统计
    print("\n=== 清洗统计 ===")
    print(f"总条数:           {stats['total']}")
    print(f"task dict→str:    {stats['task_dict_fixed']}")
    print(f"soul prompt 修复: {stats['soul_prompt_fixed']}")
    print(f"soul prompt 待审: {stats['soul_prompt_needs_review']}")
    print(f"空壳移除:         {stats['hollow_removed']}")
    print(f"解析错误:         {stats['parse_errors']}")
    print(f"保留条数:         {stats['kept']}")

    if needs_review:
        print(f"\n=== 需人工审查 ({len(needs_review)} 条) ===")
        for item in needs_review:
            print(f"  行 {item['line']} [{item['employee']}]: {item['task_preview']}")

    if not apply:
        print("\n(dry-run 模式，未修改文件。加 --apply 执行实际写入)")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="一次性轨迹数据清洗")
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_FILE,
        help=f"轨迹文件路径（默认 {DEFAULT_FILE}）",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="实际执行清洗（默认 dry-run 只打印统计）",
    )
    args = parser.parse_args()

    stats = clean_file(args.file, apply=args.apply)
    if stats.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
