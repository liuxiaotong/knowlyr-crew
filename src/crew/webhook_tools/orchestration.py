"""编排工具函数 — 异步委派、会议、Pipeline、定时任务、文件操作."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from crew.webhook_context import _ID_API_BASE, _ID_API_TOKEN

if TYPE_CHECKING:
    from crew.webhook_context import _AppContext

logger = logging.getLogger(__name__)


async def _tool_delegate_async(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """异步委派 — 立即返回 task_id，后台执行."""
    if ctx is None:
        return "错误: 上下文不可用"

    employee_name = args.get("employee_name", "")
    task_desc = args.get("task", "")

    from crew.discovery import discover_employees

    discovery = discover_employees(project_dir=ctx.project_dir)
    if discovery.get(employee_name) is None:
        available = ", ".join(sorted(discovery.employees.keys()))
        return f"错误：未找到员工 '{employee_name}'。可用员工：{available}"

    record = ctx.registry.create(
        trigger="delegate_async",
        target_type="employee",
        target_name=employee_name,
        args={"task": task_desc},
    )
    from crew.webhook_executor import _execute_task
    asyncio.create_task(_execute_task(ctx, record.task_id, agent_id=agent_id))
    return f"已异步委派给 {employee_name}。任务 ID: {record.task_id}"



async def _tool_check_task(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查询任务状态和结果."""
    if ctx is None:
        return "错误: 上下文不可用"

    task_id = args.get("task_id", "")
    record = ctx.registry.get(task_id)
    if record is None:
        return f"未找到任务: {task_id}"

    lines = [
        f"任务 ID: {record.task_id}",
        f"状态: {record.status}",
        f"类型: {record.target_type}",
        f"目标: {record.target_name}",
        f"创建: {record.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if record.completed_at:
        lines.append(f"完成: {record.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if record.status == "completed" and record.result:
        if record.target_type == "meeting":
            synthesis = record.result.get("synthesis", "")
            lines.append(f"\n会议综合结论:\n{synthesis[:1000]}")
        else:
            content = record.result.get("content", "")
            lines.append(f"\n执行结果:\n{content[:1000]}")
    elif record.status == "failed":
        lines.append(f"错误: {record.error}")

    return "\n".join(lines)



async def _tool_list_tasks(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """列出最近的任务."""
    if ctx is None:
        return "错误: 上下文不可用"

    status_filter = args.get("status")
    type_filter = args.get("type")
    limit = int(args.get("limit", 10))

    if status_filter:
        tasks = ctx.registry.list_by_status(status_filter, limit=limit)
    elif type_filter:
        tasks = ctx.registry.list_by_type(type_filter, limit=limit)
    else:
        tasks = ctx.registry.list_recent(n=limit)

    if not tasks:
        return "暂无任务记录。"

    _icons = {"pending": "⏳", "running": "▶️", "completed": "✅", "failed": "❌"}
    lines = [f"最近任务（共 {len(tasks)} 条）:"]
    for r in tasks:
        icon = _icons.get(r.status, "•")
        t = r.created_at.strftime("%m-%d %H:%M")
        lines.append(f"{icon} [{r.task_id}] {r.target_name} ({r.status}) - {t}")
    return "\n".join(lines)


# _execute_meeting is in crew.webhook (execution engine, not a tool)


async def _tool_organize_meeting(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """组织多员工会议（异步）."""
    if ctx is None:
        return "错误: 上下文不可用"

    employees_raw = args.get("employees", [])
    if isinstance(employees_raw, str):
        employees_list = [e.strip() for e in employees_raw.split(",") if e.strip()]
    else:
        employees_list = list(employees_raw)
    topic = args.get("topic", "")
    goal = args.get("goal", "")
    rounds = int(args.get("rounds", 2))

    if not employees_list or not topic:
        return "错误: 必须提供 employees 和 topic"

    from crew.discovery import discover_employees

    discovery = discover_employees(project_dir=ctx.project_dir)
    missing = [e for e in employees_list if discovery.get(e) is None]
    if missing:
        available = ", ".join(sorted(discovery.employees.keys()))
        return f"错误：未找到员工 {', '.join(missing)}。可用员工：{available}"

    meeting_name = f"{'、'.join(employees_list[:3])}{'等' if len(employees_list) > 3 else ''}会议"
    record = ctx.registry.create(
        trigger="organize_meeting",
        target_type="meeting",
        target_name=meeting_name,
        args={
            "employees": ",".join(employees_list),
            "topic": topic,
            "goal": goal,
            "rounds": str(rounds),
        },
    )
    from crew.webhook_executor import _execute_task
    asyncio.create_task(_execute_task(ctx, record.task_id, agent_id=agent_id))
    return (
        f"已组织会议。会议 ID: {record.task_id}\n"
        f"参会者: {', '.join(employees_list)}\n"
        f"议题: {topic}\n"
        f"轮次: {rounds}（异步执行中）"
    )



async def _tool_check_meeting(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查询会议进展（check_task 别名）."""
    return await _tool_check_task(args, agent_id=agent_id, ctx=ctx)


# ── Pipeline / Chain / 定时 / 文件 / 数据 / 日程 ──



async def _tool_run_pipeline(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """执行预定义流水线（异步）."""
    if ctx is None:
        return "错误: 上下文不可用"

    name = args.get("name", "")
    pipeline_args = args.get("args", {})
    if isinstance(pipeline_args, str):
        import json as _json
        try:
            pipeline_args = _json.loads(pipeline_args)
        except Exception:
            pipeline_args = {}

    from crew.pipeline import discover_pipelines

    pipelines = discover_pipelines(project_dir=ctx.project_dir)
    if name not in pipelines:
        available = ", ".join(sorted(pipelines.keys()))
        return f"错误：未找到流水线 '{name}'。可用：{available}"

    record = ctx.registry.create(
        trigger="tool",
        target_type="pipeline",
        target_name=name,
        args={k: str(v) for k, v in pipeline_args.items()},
    )
    from crew.webhook_executor import _execute_task
    asyncio.create_task(_execute_task(ctx, record.task_id, agent_id=agent_id))
    return f"已启动流水线 {name}。任务 ID: {record.task_id}（异步执行中）"


# _execute_chain is in crew.webhook (execution engine, not a tool)


async def _tool_delegate_chain(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """顺序委派链（异步）."""
    if ctx is None:
        return "错误: 上下文不可用"

    steps = args.get("steps", [])
    if not steps or not isinstance(steps, list):
        return "错误: 必须提供 steps 列表"

    from crew.discovery import discover_employees

    discovery = discover_employees(project_dir=ctx.project_dir)
    missing = [s["employee_name"] for s in steps if discovery.get(s.get("employee_name", "")) is None]
    if missing:
        return f"错误：未找到员工 {', '.join(missing)}"

    names = [s["employee_name"] for s in steps]
    chain_name = " → ".join(names)
    record = ctx.registry.create(
        trigger="delegate_chain",
        target_type="chain",
        target_name=chain_name,
        args={"steps_json": __import__("json").dumps(steps, ensure_ascii=False)},
    )
    from crew.webhook_executor import _execute_task
    asyncio.create_task(_execute_task(ctx, record.task_id, agent_id=agent_id))
    return f"已启动委派链: {chain_name}。任务 ID: {record.task_id}（异步执行中）"



async def _tool_schedule_task(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """创建定时任务."""
    if ctx is None or ctx.scheduler is None:
        return "错误: 定时任务调度器未启用"

    name = args.get("name", "")
    cron_expr = args.get("cron", "")
    emp_name = args.get("employee_name", "")
    task_desc = args.get("task", "")

    if not all([name, cron_expr, emp_name, task_desc]):
        return "错误: 必须提供 name、cron、employee_name、task"

    # 检查 cron 表达式
    try:
        from croniter import croniter
        if not croniter.is_valid(cron_expr):
            return f"错误: 无效的 cron 表达式 '{cron_expr}'"
    except ImportError:
        return "错误: croniter 未安装"

    # 检查名称冲突
    existing = [s.name for s in ctx.scheduler.schedules]
    if name in existing:
        return f"错误: 已存在同名定时任务 '{name}'"

    from crew.cron_config import CronSchedule

    schedule = CronSchedule(
        name=name,
        cron=cron_expr,
        target_type="employee",
        target_name=emp_name,
        args={"task": task_desc, "format": "memo"},
    )
    await ctx.scheduler.add_schedule(schedule)
    return f"已创建定时任务 '{name}'（{cron_expr} → {emp_name}）"



async def _tool_list_schedules(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """列出定时任务."""
    if ctx is None or ctx.scheduler is None:
        return "定时任务调度器未启用，暂无定时任务。"

    runs = ctx.scheduler.get_next_runs()
    if not runs:
        return "暂无定时任务。"

    lines = [f"定时任务（共 {len(runs)} 个）:"]
    for r in runs:
        if "error" in r:
            lines.append(f"  ❌ {r['name']}: {r['error']}")
        else:
            lines.append(
                f"  ⏰ {r['name']} ({r['cron']}) → {r['target_type']}/{r['target_name']}"
                f"  下次: {r['next_run']}"
            )
    return "\n".join(lines)



async def _tool_cancel_schedule(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """取消定时任务."""
    if ctx is None or ctx.scheduler is None:
        return "错误: 定时任务调度器未启用"

    name = args.get("name", "")
    if not name:
        return "错误: 必须提供任务名称"

    removed = await ctx.scheduler.remove_schedule(name)
    if removed:
        return f"已取消定时任务 '{name}'"
    return f"未找到定时任务 '{name}'"



async def _tool_agent_file_read(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """读取项目目录内的文件."""
    if ctx is None:
        return "错误: 上下文不可用"

    rel_path = args.get("path", "")
    if not rel_path:
        return "错误: 必须提供 path"

    project_dir = ctx.project_dir or Path(".")
    full_path = (project_dir / rel_path).resolve()

    # 安全检查
    if not full_path.is_relative_to(project_dir.resolve()):
        return "错误: 路径不在项目目录内"

    if not full_path.is_file():
        return f"错误: 文件不存在 — {rel_path}"

    start = args.get("start_line")
    end = args.get("end_line")

    try:
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)

        if start is not None:
            start = max(1, int(start))
        else:
            start = 1
        if end is not None:
            end = min(total, int(end))
        else:
            end = min(total, start + 4999)  # 最多 5000 行

        selected = lines[start - 1 : end]
        numbered = [f"{start + i:5d}│{line}" for i, line in enumerate(selected)]
        header = f"文件: {rel_path} ({total} 行, 显示 {start}-{end})\n"
        return header + "\n".join(numbered)
    except Exception as e:
        return f"读取失败: {e}"



async def _tool_agent_file_grep(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """在项目目录内搜索文件内容."""
    if ctx is None:
        return "错误: 上下文不可用"

    pattern = args.get("pattern", "")
    if not pattern:
        return "错误: 必须提供 pattern"

    project_dir = ctx.project_dir or Path(".")
    search_path = project_dir
    rel = args.get("path", "")
    if rel:
        search_path = (project_dir / rel).resolve()
        if not search_path.is_relative_to(project_dir.resolve()):
            return "错误: 路径不在项目目录内"

    file_pattern = args.get("file_pattern", "")

    import subprocess

    cmd = ["grep", "-rn", "--max-count=200", "--include", file_pattern or "*", pattern, str(search_path)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10,
            cwd=str(project_dir),
        )
        output = result.stdout.strip()
        if not output:
            return f"未找到匹配: {pattern}"

        # 将绝对路径替换为相对路径
        output = output.replace(str(project_dir) + "/", "")
        lines = output.splitlines()
        if len(lines) > 100:
            return "\n".join(lines[:100]) + f"\n\n... 共 {len(lines)} 条匹配（已截断前 100 条）"
        return output
    except subprocess.TimeoutExpired:
        return "搜索超时（10s），请缩小搜索范围"
    except Exception as e:
        return f"搜索失败: {e}"



async def _tool_query_data(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查询细粒度业务数据."""
    metric = args.get("metric", "")
    period = args.get("period", "week")
    group_by = args.get("group_by", "")

    import httpx

    params = {"metric": metric, "period": period}
    if group_by:
        params["group_by"] = group_by

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{_ID_API_BASE}/api/stats/query",
                params=params,
                headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
            )
            if resp.status_code == 200:
                return resp.text
            return f"查询失败 (HTTP {resp.status_code}): {resp.text[:200]}"
    except Exception as e:
        return f"查询失败: {e}"



async def _tool_find_free_time(
    args: dict, *, agent_id: int | None = None, ctx: "_AppContext | None" = None,
) -> str:
    """查询飞书用户共同空闲时间."""
    if ctx is None or ctx.feishu_token_mgr is None:
        return "错误: 飞书未配置"

    user_ids = args.get("user_ids", [])
    if isinstance(user_ids, str):
        user_ids = [u.strip() for u in user_ids.split(",") if u.strip()]
    days = int(args.get("days", 7))
    duration = int(args.get("duration_minutes", 60))

    if not user_ids:
        return "错误: 必须提供 user_ids"

    from crew.feishu import get_freebusy

    import time as _time

    now = int(_time.time())
    end = now + days * 86400

    try:
        busy_data = await get_freebusy(ctx.feishu_token_mgr, user_ids, now, end)
        if "error" in busy_data:
            return f"查询忙闲失败: {busy_data['error']}"

        # 计算空闲交集
        from datetime import datetime, timedelta

        # busy_data["freebusy_list"] = [{user_id, busy: [{start_time, end_time}]}]
        all_busy = []
        for user_info in busy_data.get("freebusy_list", []):
            for slot in user_info.get("busy", []):
                all_busy.append((int(slot["start_time"]), int(slot["end_time"])))

        # 合并重叠时段
        all_busy.sort()
        merged = []
        for s, e in all_busy:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # 在工作时间（9-18点）找空闲
        free_slots = []
        check_time = now
        for day_offset in range(days):
            day_start = datetime.fromtimestamp(now) + timedelta(days=day_offset)
            work_start = day_start.replace(hour=9, minute=0, second=0).timestamp()
            work_end = day_start.replace(hour=18, minute=0, second=0).timestamp()

            cursor = int(work_start)
            for bs, be in merged:
                if be <= cursor or bs >= work_end:
                    continue
                if bs > cursor:
                    gap = bs - cursor
                    if gap >= duration * 60:
                        free_slots.append((cursor, bs))
                cursor = max(cursor, int(be))
            if cursor < work_end and (work_end - cursor) >= duration * 60:
                free_slots.append((cursor, int(work_end)))

        if not free_slots:
            return f"未来 {days} 天内没有找到 {duration} 分钟的共同空闲时段。"

        lines = [f"共同空闲时段（{duration} 分钟以上）:"]
        for s, e in free_slots[:10]:
            st = datetime.fromtimestamp(s)
            et = datetime.fromtimestamp(e)
            lines.append(f"  {st.strftime('%m-%d %H:%M')} ~ {et.strftime('%H:%M')}")
        if len(free_slots) > 10:
            lines.append(f"  ... 共 {len(free_slots)} 个时段")
        return "\n".join(lines)
    except Exception as e:
        return f"查询空闲时间失败: {e}"


# _MAX_TOOL_ROUNDS and other constants moved to crew.webhook_context


# ── Tool handlers（调用 knowlyr-id API）──



HANDLERS: dict[str, object] = {
    "delegate_async": _tool_delegate_async,
    "check_task": _tool_check_task,
    "list_tasks": _tool_list_tasks,
    "organize_meeting": _tool_organize_meeting,
    "check_meeting": _tool_check_meeting,
    "run_pipeline": _tool_run_pipeline,
    "delegate_chain": _tool_delegate_chain,
    "schedule_task": _tool_schedule_task,
    "list_schedules": _tool_list_schedules,
    "cancel_schedule": _tool_cancel_schedule,
    "agent_file_read": _tool_agent_file_read,
    "agent_file_grep": _tool_agent_file_grep,
    "query_data": _tool_query_data,
    "find_free_time": _tool_find_free_time,
}
