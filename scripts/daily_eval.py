#!/usr/bin/env python3
"""每日自动评估脚本 -- 增量评分 + 高分范例导出 + 日报生成.

设计给叶心蕾（HR）定时触发，每天一次。

用法:
    # 评估昨天的轨迹（默认）
    python scripts/daily_eval.py

    # 评估指定日期
    python scripts/daily_eval.py --date 2026-02-21

    # 强制重跑（忽略游标，重新评分）
    python scripts/daily_eval.py --force

    # 同时启用 LLM Judge（默认只走规则层）
    python scripts/daily_eval.py --with-judge --model kimi-k2.5

幂等性:
    同一天跑两遍结果一样。评分结果按日期写入固定路径，重复执行会覆盖。
    游标记录已处理的最后一个 session 文件名，增量拉取新数据。

输出:
    .crew/evaluations/{date}.jsonl   -- 逐条评分结果
    .crew/exemplars/{employee}/      -- 高分轨迹 (reward >= 0.7)
    .crew/reports/daily/{date}.md    -- 文字日报
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# 项目路径
CREW_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = CREW_ROOT / ".crew" / "sessions"
TRAJECTORIES_FILE = CREW_ROOT / ".crew" / "trajectories" / "trajectories.jsonl"
EVALUATIONS_DIR = CREW_ROOT / ".crew" / "evaluations"
EXEMPLARS_DIR = CREW_ROOT / ".crew" / "exemplars"
REPORTS_DIR = CREW_ROOT / ".crew" / "reports" / "daily"
CURSOR_FILE = CREW_ROOT / ".crew" / "evaluations" / ".cursor"

# 确保 crew 模块可导入
sys.path.insert(0, str(CREW_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from crew.scoring import (  # noqa: E402, I001
    EXEMPLAR_THRESHOLD,
    _build_exemplar_content,
    _get_domain,
    score_trajectory,
    traj_employee,
)


# ── 游标管理 ─────────────────────────────────────────────────────────


def _read_cursor() -> str:
    """读取增量游标（上次处理的最后一个 session 文件名）."""
    if CURSOR_FILE.exists():
        return CURSOR_FILE.read_text("utf-8").strip()
    return ""


def _write_cursor(last_file: str) -> None:
    """写入增量游标."""
    CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    CURSOR_FILE.write_text(last_file + "\n", encoding="utf-8")


# ── 轨迹采集 ─────────────────────────────────────────────────────────


def _collect_from_sessions(target_date: str, cursor: str) -> list[dict[str, Any]]:
    """从 .crew/sessions/ 目录增量采集指定日期的轨迹.

    Args:
        target_date: 目标日期 "YYYYMMDD"
        cursor: 上次处理到的文件名（跳过 <= cursor 的文件）

    Returns:
        转换后的 trajectory 列表
    """
    from crew.session_converter import convert_session, is_organic

    trajectories = []
    if not SESSIONS_DIR.exists():
        return trajectories

    for f in sorted(SESSIONS_DIR.glob("*.jsonl")):
        # 文件名格式: YYYYMMDD-HHMMSS-uuid.jsonl
        file_date = f.stem[:8]

        # 只取目标日期的文件
        if file_date != target_date:
            continue

        # 增量: 跳过已处理的
        if cursor and f.stem <= cursor:
            continue

        # 只取 organic 数据
        try:
            first_line = f.read_text("utf-8").split("\n")[0]
            start = json.loads(first_line)
            if not is_organic(start):
                continue
        except Exception:
            continue

        traj = convert_session(f)
        if traj is not None:
            traj["_source_file"] = f.stem  # 记录来源文件，用于游标
            trajectories.append(traj)

    return trajectories


def _collect_from_trajectories_file(target_date: str) -> list[dict[str, Any]]:
    """从 .crew/trajectories/trajectories.jsonl 采集指定日期的轨迹.

    TrajectoryCollector 写出的数据，字段名和 session_converter 输出不同，需要适配。
    """
    trajectories = []
    if not TRAJECTORIES_FILE.exists():
        return trajectories

    for line in TRAJECTORIES_FILE.read_text("utf-8").strip().split("\n"):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        # 判断日期：从 task 对象或 metadata 中的 timestamp 提取
        timestamp = ""
        if isinstance(data.get("task"), dict):
            # agentrecorder Trajectory 格式
            timestamp = data.get("metadata", {}).get("timestamp", "")
        elif data.get("steps") and isinstance(data["steps"][0], dict):
            timestamp = data["steps"][0].get("timestamp", "")

        if timestamp and timestamp[:10].replace("-", "") != target_date:
            continue

        # 标准化为 RewardEngine 接受的格式
        traj = _normalize_trajectory(data)
        if traj is not None:
            trajectories.append(traj)

    return trajectories


def _normalize_trajectory(data: dict[str, Any]) -> dict[str, Any] | None:
    """将各种轨迹格式统一为 RewardEngine 接受的标准格式."""
    # 格式 A: session_converter 输出（已经是标准格式）
    if isinstance(data.get("task"), str) and isinstance(data.get("steps"), list):
        if data["steps"] and "tool" in data["steps"][0]:
            # 确保 metadata 中包含 channel 和 session_id
            meta = data.get("metadata", {})
            if "channel" not in meta or "session_id" not in meta:
                meta.setdefault("channel", "")
                meta.setdefault("session_id", meta.get("task_id", ""))
                data["metadata"] = meta
            return data

    # 格式 B: agentrecorder Trajectory（task 是对象）
    if isinstance(data.get("task"), dict):
        task_desc = data["task"].get("description", "")
        task_id = data["task"].get("task_id", "")
        steps = []
        for s in data.get("steps", []):
            tc = s.get("tool_call", {})
            tr = s.get("tool_result", {})
            steps.append(
                {
                    "tool": tc.get("name", "respond"),
                    "params": tc.get("parameters", {}),
                    "output": tr.get("output", ""),
                }
            )
        outcome = data.get("outcome", {})
        raw_meta = data.get("metadata", {})
        employee = raw_meta.get("employee", "")
        return {
            "task": task_desc,
            "steps": steps,
            "outcome": {"success": outcome.get("success", True)},
            "metadata": {
                "employee": employee,
                "model": data.get("model", ""),
                "timestamp": "",
                "channel": raw_meta.get("channel", ""),
                "session_id": task_id,
            },
        }

    # 格式 C: TrajectoryCollector fallback JSON
    if "employee" in data and "task" in data and isinstance(data.get("task"), str):
        steps = []
        for s in data.get("steps", []):
            steps.append(
                {
                    "tool": s.get("tool_name", "respond"),
                    "params": s.get("tool_params", {}),
                    "output": s.get("tool_output", ""),
                }
            )
        return {
            "task": data["task"],
            "steps": steps,
            "outcome": {"success": data.get("success", True)},
            "metadata": {
                "employee": data["employee"],
                "model": data.get("model", ""),
                "timestamp": "",
                "channel": data.get("channel", ""),
                "session_id": data.get("task_id", ""),
            },
        }

    return None




# ── 日报生成 ─────────────────────────────────────────────────────────


def _generate_report(
    date_str: str,
    results: list[dict[str, Any]],
) -> str:
    """生成 Markdown 日报."""
    lines = [
        f"# 每日评估日报 {date_str}",
        "",
        f"- 评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 处理条数: {len(results)}",
    ]

    if not results:
        lines.append("")
        lines.append("今日无新增轨迹。")
        return "\n".join(lines)

    # 使用的引擎
    engines = {r.get("engine", "?") for r in results}
    lines.append(f"- 评分引擎: {', '.join(engines)}")

    # 总体统计
    scores = [r["total_score"] for r in results]
    lines.extend(
        [
            f"- 平均分: {statistics.mean(scores):.2f}",
            f"- 最高分: {max(scores):.2f}",
            f"- 最低分: {min(scores):.2f}",
            "",
        ]
    )

    # 按员工分组
    by_employee: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        emp = r.get("employee", "unknown")
        by_employee.setdefault(emp, []).append(r)

    lines.append("## 各员工表现")
    lines.append("")
    lines.append("| 员工 | 条数 | 平均分 | 最高 | 最低 | 域 |")
    lines.append("|------|------|--------|------|------|----|")

    employee_stats = []
    for emp, emp_results in sorted(by_employee.items()):
        emp_scores = [r["total_score"] for r in emp_results]
        avg = statistics.mean(emp_scores)
        domain = _get_domain(emp)
        lines.append(
            f"| {emp} | {len(emp_results)} | {avg:.2f} | "
            f"{max(emp_scores):.2f} | {min(emp_scores):.2f} | {domain} |"
        )
        employee_stats.append((emp, avg, len(emp_results)))

    lines.append("")

    # Top 表现（按平均分排序，取前 5）
    sorted_by_avg = sorted(employee_stats, key=lambda x: -x[1])
    if len(sorted_by_avg) > 1:
        lines.append("## Top 表现")
        lines.append("")
        for emp, avg, count in sorted_by_avg[:5]:
            lines.append(f"- **{emp}**: {avg:.2f} ({count} 条)")
        lines.append("")

    # Bottom 表现（平均分 < 0.5 的员工）
    weak = [(emp, avg, count) for emp, avg, count in sorted_by_avg if avg < 0.5]
    if weak:
        lines.append("## 需关注")
        lines.append("")
        for emp, avg, count in weak:
            lines.append(f"- **{emp}**: {avg:.2f} ({count} 条) -- 平均分低于 0.5")
        lines.append("")

    # 高分范例统计
    exemplar_count = sum(1 for r in results if r["total_score"] >= EXEMPLAR_THRESHOLD)
    if exemplar_count:
        lines.append("## 高分范例")
        lines.append("")
        lines.append(
            f"本日 {exemplar_count} 条轨迹达到范例标准 (>= {EXEMPLAR_THRESHOLD})，已导出到 exemplars/ 目录。"
        )
        lines.append("")

    return "\n".join(lines)


# ── 主流程 ───────────────────────────────────────────────────────────


def run_daily_eval(
    target_date: str | None = None,
    force: bool = False,
    use_judge: bool = False,
    model_name: str = "",
    provider: str = "openai",
    base_url: str | None = None,
    api_key: str | None = None,
    employee_filter: str | None = None,
) -> dict[str, Any]:  # noqa: PLR0913
    """执行每日评估.

    Args:
        target_date: 目标日期 "YYYYMMDD"，默认为昨天
        force: 强制重跑（忽略游标）
        use_judge: 是否启用 LLM Judge
        model_name: LLM Judge 模型名
        provider: LLM 提供商
        base_url: API base URL
        api_key: API key
        employee_filter: 只评估指定员工（slug 名），None 表示全部

    Returns:
        汇总统计 dict
    """
    if target_date is None:
        yesterday = datetime.now() - timedelta(days=1)
        target_date = yesterday.strftime("%Y%m%d")

    date_display = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:8]}"
    logger.info("每日评估开始: %s", date_display)

    # 读游标（增量）
    cursor = "" if force else _read_cursor()
    if cursor:
        logger.info("增量模式: 从 %s 之后开始", cursor[:20])

    # Step 1: 采集轨迹
    trajectories = _collect_from_sessions(target_date, cursor)
    traj_file_trajs = _collect_from_trajectories_file(target_date)
    trajectories.extend(traj_file_trajs)

    logger.info(
        "采集到 %d 条轨迹 (sessions: %d, trajectories.jsonl: %d)",
        len(trajectories),
        len(trajectories) - len(traj_file_trajs),
        len(traj_file_trajs),
    )

    if not trajectories:
        logger.info("无新增轨迹，生成空日报")
        # 仍然生成日报（记录"今日无数据"）
        _write_report(date_display, [])
        return {"date": date_display, "total": 0, "scored": 0}

    # 过滤无归属轨迹
    SKIP_EMPLOYEES = {"unknown", "unknown-agent", ""}
    before_filter = len(trajectories)
    trajectories = [
        t for t in trajectories
        if traj_employee(t) not in SKIP_EMPLOYEES
    ]
    if before_filter != len(trajectories):
        logger.info("过滤无归属轨迹: %d → %d", before_filter, len(trajectories))

    # 过滤 training channel（训练/测试数据不参与正式评分）
    SKIP_CHANNELS = {"training", "test", "debug"}
    before_channel = len(trajectories)
    trajectories = [
        t for t in trajectories
        if (t.get("metadata", {}).get("channel") or "") not in SKIP_CHANNELS
    ]
    if before_channel != len(trajectories):
        logger.info("过滤非工作channel: %d → %d", before_channel, len(trajectories))

    # 按员工过滤
    if employee_filter:
        before_emp = len(trajectories)
        trajectories = [
            t for t in trajectories
            if traj_employee(t) == employee_filter
        ]
        logger.info("员工过滤 [%s]: %d → %d", employee_filter, before_emp, len(trajectories))

    # 过滤空壳轨迹（steps 全空 = 采集 bug，不值得评分）
    from crew.trajectory import is_hollow_trajectory

    before_hollow = len(trajectories)
    trajectories = [t for t in trajectories if not is_hollow_trajectory(t)]
    if before_hollow != len(trajectories):
        logger.info("过滤空壳轨迹: %d → %d", before_hollow, len(trajectories))

    # Step 2: 逐条评分（增量写入：每条评完立刻追加到 JSONL）
    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    # 文件名含员工名（按人跑时隔离）或 all
    eval_suffix = employee_filter or "all"
    eval_path = EVALUATIONS_DIR / f"{date_display}.{eval_suffix}.jsonl"
    # force 模式清空旧文件，否则追加
    if force:
        eval_path.write_text("", encoding="utf-8")

    results = []
    last_source_file = cursor
    for i, traj in enumerate(trajectories):
        employee = traj.get("metadata", {}).get("employee", "unknown")
        task_preview = (traj.get("task", "") or "")[:50]
        logger.info("评分 [%d/%d] %s: %s...", i + 1, len(trajectories), employee, task_preview)

        try:
            score_result = score_trajectory(
                traj,
                employee,
                use_judge=use_judge,
                model_name=model_name,
                provider=provider,
                base_url=base_url,
                api_key=api_key,
            )
        except Exception as e:
            logger.warning("评分失败 [%s] %s: %s", employee, task_preview, e)
            continue

        result = {
            "employee": employee,
            "domain": _get_domain(employee),
            "task": traj.get("task", ""),
            "model": traj.get("metadata", {}).get("model", ""),
            "session_id": traj.get("metadata", {}).get("session_id", ""),
            **score_result,
            "scored_at": f"{date_display}T00:00:00",
        }
        results.append(result)

        # 增量写入：每条评完立刻追加
        with open(eval_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # 更新游标
        source_file = traj.get("_source_file", "")
        if source_file and source_file > last_source_file:
            last_source_file = source_file

    logger.info("评分完成: %d/%d 成功", len(results), len(trajectories))

    # Step 3: 汇总写入（兼容旧格式 {date}.jsonl，按人模式跳过全局文件）
    _write_evaluations(date_display, results, employee_filter=employee_filter)

    # Step 4: 导出高分范例
    exemplar_count = _export_exemplars(results)
    logger.info("高分范例: %d 条", exemplar_count)

    # Step 5: 生成日报
    _write_report(date_display, results)

    # Step 6: 更新游标（仅全量模式更新，按人模式跳过避免竞争）
    if not employee_filter and last_source_file and last_source_file != cursor:
        _write_cursor(last_source_file)
        logger.info("游标更新: %s", last_source_file[:20])

    return {
        "date": date_display,
        "total": len(trajectories),
        "scored": len(results),
        "exemplars": exemplar_count,
    }


def _write_evaluations(
    date_display: str,
    results: list[dict[str, Any]],
    employee_filter: str | None = None,
) -> None:
    """写入评分结果到 .crew/evaluations/{date}.jsonl（覆盖写入，保证幂等）.

    按人模式只写 per-employee 文件，不写全局汇总（避免多进程覆盖冲突）。
    """
    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    if not employee_filter:
        output_path = EVALUATIONS_DIR / f"{date_display}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("评分结果: %s (%d 条)", output_path, len(results))


def _export_exemplars(results: list[dict[str, Any]]) -> int:
    """将高分轨迹导出到 .crew/exemplars/{employee_name}/ 并写入 MemoryStore."""
    count = 0
    memory_store = None
    try:
        from crew.memory import MemoryStore

        memory_store = MemoryStore(project_dir=CREW_ROOT)
    except Exception as e:
        logger.warning("MemoryStore 初始化失败，跳过记忆写入: %s", e)

    for r in results:
        if r["total_score"] >= EXEMPLAR_THRESHOLD:
            employee = r["employee"]
            emp_dir = EXEMPLARS_DIR / employee
            emp_dir.mkdir(parents=True, exist_ok=True)
            # 用 session_id 或 scored_at 做文件名，保证幂等
            filename = r.get("session_id") or r["scored_at"].replace(":", "-")
            output_path = emp_dir / f"{filename}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=2)
            count += 1

            # 写入 MemoryStore（去重：同一 session 不重复写入）
            if memory_store is not None:
                _write_exemplar_memory(memory_store, r)

    return count


def _write_exemplar_memory(
    memory_store: Any,
    result: dict[str, Any],
) -> None:
    """将单条高分范例写入 MemoryStore（去重）."""
    employee = result["employee"]
    session_id = result.get("session_id") or ""

    # 去重：用 source_session 检查是否已写入过
    if session_id:
        try:
            existing = memory_store.query(
                employee,
                category="finding",
                limit=100,
            )
            for entry in existing:
                if entry.source_session == session_id and "exemplar" in (entry.tags or []):
                    logger.debug("范例记忆已存在，跳过: %s/%s", employee, session_id)
                    return
        except Exception:
            pass

    content = _build_exemplar_content(result)
    try:
        memory_store.add(
            employee=employee,
            category="finding",
            content=content,
            source_session=session_id,
            tags=["exemplar", "high-score"],
            origin_employee=employee,
        )
        logger.debug("范例记忆写入: %s", employee)
    except Exception as e:
        logger.warning("范例记忆写入失败 [%s]: %s", employee, e)


def _write_report(date_display: str, results: list[dict[str, Any]]) -> None:
    """生成并写入日报到 .crew/reports/daily/{date}.md."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_text = _generate_report(date_display, results)
    output_path = REPORTS_DIR / f"{date_display}.md"
    output_path.write_text(report_text, encoding="utf-8")
    logger.info("日报: %s", output_path)


# ── CLI 入口 ─────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="每日自动评估脚本")
    parser.add_argument("--date", help="目标日期 YYYYMMDD（默认昨天）")
    parser.add_argument("--force", action="store_true", help="强制重跑（忽略游标）")
    parser.add_argument(
        "--with-judge", action="store_true", help="启用 LLM Judge（默认只走规则层）"
    )
    parser.add_argument("--model", default="kimi-k2.5", help="LLM Judge 模型（默认 kimi-k2.5）")
    parser.add_argument("--provider", default="openai", help="LLM 提供商（默认 openai）")
    parser.add_argument("--base-url", help="API base URL")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--employee", help="只评估指定员工（slug 名）")

    args = parser.parse_args()

    summary = run_daily_eval(
        target_date=args.date,
        force=args.force,
        use_judge=args.with_judge,
        model_name=args.model,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
        employee_filter=args.employee,
    )

    print("\n=== 评估完成 ===")
    print(f"日期: {summary['date']}")
    print(f"轨迹总数: {summary['total']}")
    print(f"成功评分: {summary['scored']}")
    if summary.get("exemplars"):
        print(f"高分范例: {summary['exemplars']}")


if __name__ == "__main__":
    main()
