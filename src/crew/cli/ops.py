"""运维命令 — agents, cron, eval, trajectory, template, export, sync, log, session 等."""

import json
import sys
from datetime import datetime
from pathlib import Path

import click

from crew.cli import (
    _default_display_name,
    _employee_root,
    _parse_variables,
    _suggest_similar,
)
from crew.discovery import discover_employees
from crew.session_recorder import SessionRecorder
from crew.template_manager import apply_template, discover_templates

# ── template 命令 ──


@click.group()
def template():
    """模板管理 — 快速复用员工骨架/提示."""
    pass


@template.command("list")
def template_list():
    """列出可用模板（内置 + 全局 + 项目）."""
    templates = discover_templates()
    if not templates:
        click.echo("未找到模板。")
        click.echo("在 .crew/templates/ 中放置自定义模板即可。")
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="模板库")
        table.add_column("名称", style="cyan")
        table.add_column("来源", style="green")
        table.add_column("路径")
        for record in templates.values():
            table.add_row(record.name, record.layer, str(record.path))
        console.print(table)
    except ImportError:
        for record in templates.values():
            click.echo(f"{record.name} ({record.layer}) — {record.path}")


@template.command("apply")
@click.argument("template_name")
@click.option("--employee", type=str, help="要生成的员工名称（会自动设置常用变量）")
@click.option("--var", "variables", multiple=True, help="额外模板变量，格式 key=value")
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="输出文件路径（默认 private/employees/<employee>.md）",
)
@click.option("--force", is_flag=True, help="如目标已存在则覆盖")
def template_apply(
    template_name: str,
    employee: str | None,
    variables: tuple[str, ...],
    output: str | None,
    force: bool,
):
    """渲染模板并输出到 private/employees/ 目录."""
    parsed_vars = _parse_variables(variables)

    defaults: dict[str, str] = {}
    if employee:
        defaults["name"] = employee
        defaults.setdefault("display_name", _default_display_name(employee))
        defaults.setdefault("character_name", "")
        defaults.setdefault("description", "请输入角色描述")
        defaults.setdefault("tags", "[]")
        defaults.setdefault("triggers", "[]")
        defaults.setdefault("tools", "[]")
        defaults.setdefault("context", "[]")

    merged_vars = {**defaults, **parsed_vars}

    target_path = Path(output) if output else None
    if target_path is None:
        filename = merged_vars.get("name") or template_name
        target_path = _employee_root() / f"{filename}.md"

    try:
        result_path = apply_template(
            template_name,
            variables=merged_vars,
            output=Path(target_path),
            overwrite=force,
        )
    except FileExistsError as e:
        raise click.ClickException(str(e))
    except FileNotFoundError as e:
        raise click.ClickException(str(e))

    click.echo(f"已写入: {result_path}")


# ── Skills 导出命令 ──


@click.command("export")
@click.argument("name")
@click.option(
    "-d",
    "--dir",
    "project_dir",
    type=click.Path(),
    default=None,
    help="项目根目录（默认当前目录）",
)
def export_cmd(name: str, project_dir: str | None):
    """导出单个员工到 .claude/skills/<name>/SKILL.md."""
    from crew.skill_converter import export_employee

    result = discover_employees()
    emp = result.get(name)

    if emp is None:
        click.echo(f"未找到员工: {name}", err=True)
        sys.exit(1)

    pdir = Path(project_dir) if project_dir else Path.cwd()
    path = export_employee(emp, pdir)
    click.echo(f"已导出: {path}")


@click.command("export-all")
@click.option(
    "-d",
    "--dir",
    "project_dir",
    type=click.Path(),
    default=None,
    help="项目根目录（默认当前目录）",
)
def export_all_cmd(project_dir: str | None):
    """导出所有员工到 .claude/skills/."""
    from crew.skill_converter import export_all

    result = discover_employees()
    employees = list(result.employees.values())

    if not employees:
        click.echo("未找到员工。", err=True)
        return

    pdir = Path(project_dir) if project_dir else Path.cwd()
    paths = export_all(employees, pdir)
    for p in paths:
        click.echo(f"已导出: {p}")
    click.echo(f"\n共导出 {len(paths)} 个员工到 .claude/skills/")


@click.command()
@click.option("--clean", is_flag=True, help="删除不再存在的孤儿技能目录")
@click.option(
    "-d",
    "--dir",
    "project_dir",
    type=click.Path(),
    default=None,
    help="项目根目录（默认当前目录）",
)
def sync(clean: bool, project_dir: str | None):
    """同步所有员工到 .claude/skills/ 目录.

    --clean 会删除 .claude/skills/ 中不再对应任何员工的孤儿目录。
    """
    from crew.skill_converter import sync_skills

    result = discover_employees()
    employees = list(result.employees.values())

    if not employees:
        click.echo("未找到员工。", err=True)
        return

    pdir = Path(project_dir) if project_dir else Path.cwd()
    report = sync_skills(employees, pdir, clean=clean)

    for p in report["exported"]:
        click.echo(f"已同步: {p}")
    for p in report["removed"]:
        click.echo(f"已删除: {p}")

    click.echo(f"\n同步完成: {len(report['exported'])} 导出, {len(report['removed'])} 删除")


# ── log 子命令组 ──


@click.group()
def log():
    """工作日志管理."""
    pass


@log.command("list")
@click.option("--employee", type=str, default=None, help="按员工过滤")
@click.option("-n", "--limit", type=int, default=20, help="返回条数")
def log_list(employee: str | None, limit: int):
    """查看工作日志列表."""
    from crew.log import WorkLogger

    work_logger = WorkLogger()
    sessions = work_logger.list_sessions(employee_name=employee, limit=limit)

    if not sessions:
        click.echo("暂无工作日志。")
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="工作日志")
        table.add_column("Session ID", style="cyan")
        table.add_column("员工", style="green")
        table.add_column("开始时间")
        table.add_column("条目数", justify="right")

        for s in sessions:
            table.add_row(
                s["session_id"],
                s["employee_name"],
                s["started_at"],
                str(s["entries"]),
            )

        console.print(table)
    except ImportError:
        for s in sessions:
            click.echo(
                f"{s['session_id']}  {s['employee_name']}  {s['started_at']}  ({s['entries']} 条)"
            )


@log.command("show")
@click.argument("session_id")
def log_show(session_id: str):
    """查看某次工作的详细日志."""
    from crew.log import WorkLogger

    work_logger = WorkLogger()
    entries = work_logger.get_session(session_id)

    if not entries:
        click.echo(f"未找到 session: {session_id}", err=True)
        sys.exit(1)

    for entry in entries:
        ts = entry.get("timestamp", "")
        action = entry.get("action", "")
        detail = entry.get("detail", "")
        click.echo(f"[{ts}] {action}: {detail}")


# ── session 子命令组 ──


@click.group(name="session")
def session_group():
    """会话记录管理 — JSONL 轨迹."""
    pass


@session_group.command("list")
@click.option(
    "--type",
    "session_type",
    type=str,
    default=None,
    help="按类型过滤 (employee/pipeline/discussion)",
)
@click.option("--subject", type=str, default=None, help="按 subject 过滤 (员工/流水线名称)")
@click.option("-n", "--limit", type=int, default=20, help="返回条数")
@click.option(
    "-f", "--format", "output_format", type=click.Choice(["table", "json"]), default="table"
)
def session_list(session_type: str | None, subject: str | None, limit: int, output_format: str):
    """列出最近的会话记录."""
    recorder = SessionRecorder()
    sessions = recorder.list_sessions(limit=limit, session_type=session_type, subject=subject)

    if not sessions:
        click.echo("暂无会话记录。")
        return

    if output_format == "json":
        click.echo(json.dumps(sessions, ensure_ascii=False, indent=2))
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="会话记录")
        table.add_column("Session ID", style="cyan")
        table.add_column("类型")
        table.add_column("Subject")
        table.add_column("开始时间")

        for item in sessions:
            table.add_row(
                item["session_id"],
                item.get("session_type", ""),
                item.get("subject", ""),
                item.get("started_at", ""),
            )

        console.print(table)
    except ImportError:
        for item in sessions:
            click.echo(
                f"{item['session_id']}  {item.get('session_type', '')}  {item.get('subject', '')}  {item.get('started_at', '')}"
            )


@session_group.command("show")
@click.argument("session_id")
def session_show(session_id: str):
    """查看某次会话的完整轨迹."""
    recorder = SessionRecorder()
    entries = recorder.read_session(session_id)
    if not entries:
        click.echo(f"未找到 session: {session_id}", err=True)
        sys.exit(1)

    for entry in entries:
        ts = entry.get("timestamp", "")
        event = entry.get("event", "")
        if event == "message":
            role = entry.get("role", "")
            click.echo(f"[{ts}] ({role})\n{entry.get('content', '')}\n")
        else:
            meta = entry.get("metadata") or {}
            detail = entry.get("detail") or ""
            status = entry.get("status")
            meta_part = f" {json.dumps(meta, ensure_ascii=False)}" if meta else ""
            status_part = f" status={status}" if status else ""
            detail_part = f" detail={detail}" if detail else ""
            click.echo(f"[{ts}] {event}{status_part}{detail_part}{meta_part}")


# ── agents 子命令组 ──


@click.group()
def agents():
    """Agent 管理."""


@agents.command("list")
@click.option(
    "-f", "--format", "output_format", type=click.Choice(["table", "json"]), default="table"
)
def agents_list_cmd(output_format: str):
    """列出所有本地员工（含 agent_id 绑定信息）."""
    from crew.discovery import discover_employees

    result = discover_employees()
    employees = list(result.employees.values())

    if not employees:
        click.echo("暂无员工", err=True)
        return

    if output_format == "json":
        data = [
            {
                "name": emp.name,
                "display_name": emp.effective_display_name,
                "agent_id": emp.agent_id,
                "agent_status": emp.agent_status,
                "model": emp.model,
                "tags": emp.tags,
            }
            for emp in employees
        ]
        click.echo(json.dumps(data, ensure_ascii=False, indent=2))
        return

    from rich.console import Console
    from rich.table import Table

    table = Table(title="Employees")
    table.add_column("Name", style="cyan")
    table.add_column("Display", style="bold")
    table.add_column("Agent ID")
    table.add_column("Status")
    table.add_column("Model")
    table.add_column("Tags")

    _status_styles = {"active": "green", "frozen": "yellow", "inactive": "red"}
    for emp in employees:
        status = emp.agent_status or "active"
        style = _status_styles.get(status, "dim")
        table.add_row(
            emp.name,
            emp.effective_display_name,
            str(emp.agent_id) if emp.agent_id else "-",
            f"[{style}]{status}[/{style}]",
            emp.model or "-",
            ", ".join(emp.tags) if emp.tags else "-",
        )

    Console(stderr=True).print(table)


def _find_employee(result, name: str):
    """按 name、character_name 或 trigger 查找员工."""
    emp = result.get(name)
    if emp:
        return emp
    # 按 character_name 查找
    for e in result.employees.values():
        if getattr(e, "character_name", None) == name:
            return e
    return None


def _update_employee_status(emp, new_status: str) -> bool:
    """更新员工 employee.yaml 中的 agent_status 字段."""
    if not emp.source_path:
        return False
    src = emp.source_path if emp.source_path.is_dir() else emp.source_path.parent
    config_path = src / "employee.yaml"
    if not config_path.exists():
        return False
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        return False
    config["agent_status"] = new_status
    config_path.write_text(
        yaml.dump(config, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return True


@agents.command("freeze")
@click.argument("names", nargs=-1, required=True)
@click.option("--force", is_flag=True, help="跳过确认")
def agents_freeze_cmd(names: tuple[str, ...], force: bool):
    """冻结员工（修改本地 agent_status 为 frozen）."""
    from crew.discovery import discover_employees

    result = discover_employees(cache_ttl=0)
    targets = []
    for name in names:
        emp = _find_employee(result, name)
        if not emp:
            candidates = list(result.employees.keys())
            click.echo(f"未找到员工: {name}{_suggest_similar(name, candidates)}", err=True)
            raise SystemExit(1)
        targets.append(emp)

    if not force:
        click.echo("即将冻结以下员工:", err=True)
        for emp in targets:
            click.echo(f"  {emp.character_name or emp.name}", err=True)
        if not click.confirm("确认冻结？"):
            click.echo("已取消", err=True)
            return

    for emp in targets:
        if emp.agent_status == "frozen":
            click.echo(
                f"- {emp.character_name or emp.name} 已处于冻结状态，跳过",
                err=True,
            )
            continue
        ok = _update_employee_status(emp, "frozen")
        if ok:
            click.echo(f"✓ {emp.character_name or emp.name} 已冻结", err=True)
        else:
            click.echo(f"⚠ {emp.character_name or emp.name} 冻结失败（无 employee.yaml）", err=True)


@agents.command("unfreeze")
@click.argument("names", nargs=-1, required=True)
@click.option("--force", is_flag=True, help="跳过确认")
def agents_unfreeze_cmd(names: tuple[str, ...], force: bool):
    """解冻员工（修改本地 agent_status 为 active）."""
    from crew.discovery import discover_employees

    result = discover_employees(cache_ttl=0)
    targets = []
    for name in names:
        emp = _find_employee(result, name)
        if not emp:
            candidates = list(result.employees.keys())
            click.echo(f"未找到员工: {name}{_suggest_similar(name, candidates)}", err=True)
            raise SystemExit(1)
        targets.append(emp)

    if not force:
        click.echo("即将解冻以下员工:", err=True)
        for emp in targets:
            click.echo(f"  {emp.character_name or emp.name}", err=True)
        if not click.confirm("确认解冻？"):
            click.echo("已取消", err=True)
            return

    for emp in targets:
        if emp.agent_status == "active":
            click.echo(
                f"- {emp.character_name or emp.name} 已处于活跃状态，跳过",
                err=True,
            )
            continue
        ok = _update_employee_status(emp, "active")
        if ok:
            click.echo(f"✓ {emp.character_name or emp.name} 已解冻", err=True)
        else:
            click.echo(f"⚠ {emp.character_name or emp.name} 解冻失败（无 employee.yaml）", err=True)


@agents.command("status")
@click.argument("name")
def agents_status_cmd(name: str):
    """查看员工状态（本地数据）."""
    from crew.discovery import discover_employees

    result = discover_employees()
    emp = _find_employee(result, name)
    if not emp:
        candidates = list(result.employees.keys())
        click.echo(f"未找到员工: {name}{_suggest_similar(name, candidates)}", err=True)
        raise SystemExit(1)

    click.echo(f"员工 '{emp.name}' 状态:")
    click.echo(f"  Name:         {emp.name}")
    click.echo(f"  Display:      {emp.effective_display_name}")
    if emp.character_name:
        click.echo(f"  Character:    {emp.character_name}")
    click.echo(f"  Status:       {emp.agent_status or 'active'}")
    if emp.agent_id is not None:
        click.echo(f"  Agent ID:     {emp.agent_id}")
    click.echo(f"  Model:        {emp.model or '-'}")
    click.echo(f"  Description:  {emp.description or '-'}")
    tags = ", ".join(emp.tags) if emp.tags else "-"
    click.echo(f"  Tags:         {tags}")
    click.echo(f"  Source:       {emp.source_path or '-'}")
    click.echo(f"  Layer:        {emp.source_layer or '-'}")


# ── cron 命令 ──


@click.group(name="cron")
def cron_group():
    """Cron 调度管理 — 查看和预览定时任务."""


@cron_group.command("list")
@click.option("-d", "--project-dir", type=click.Path(path_type=Path), default=None)
def cron_list(project_dir):
    """列出所有 cron 计划任务."""
    from crew.cron_config import load_cron_config

    config = load_cron_config(project_dir)
    if not config.schedules:
        click.echo("未配置 cron 计划任务。")
        click.echo("在 .crew/cron.yaml 中添加 schedules 配置。")
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Cron 计划任务")
        table.add_column("名称", style="cyan")
        table.add_column("Cron 表达式", style="green")
        table.add_column("目标类型")
        table.add_column("目标名称", style="bold")
        table.add_column("参数")

        for s in config.schedules:
            args_str = ", ".join(f"{k}={v}" for k, v in s.args.items()) if s.args else "-"
            table.add_row(s.name, s.cron, s.target_type, s.target_name, args_str)

        console.print(table)
    except ImportError:
        for s in config.schedules:
            click.echo(f"  {s.name}: {s.cron} → {s.target_type}/{s.target_name}")


@cron_group.command("preview")
@click.option("-n", "--next", "count", default=5, type=int, help="显示未来 N 次触发")
@click.option("-d", "--project-dir", type=click.Path(path_type=Path), default=None)
def cron_preview(count, project_dir):
    """预览未来 N 次 cron 触发时间."""
    from crew.cron_config import load_cron_config

    config = load_cron_config(project_dir)
    if not config.schedules:
        click.echo("未配置 cron 计划任务。")
        return

    try:
        from croniter import croniter
    except ImportError:
        click.echo("croniter 未安装。请运行: pip install knowlyr-crew[webhook]", err=True)
        sys.exit(1)

    now = datetime.now()

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"未来 {count} 次触发时间")
        table.add_column("任务名称", style="cyan")
        table.add_column("Cron", style="green")
        table.add_column("目标", style="bold")
        for i in range(1, count + 1):
            table.add_column(f"#{i}")

        for s in config.schedules:
            try:
                cron = croniter(s.cron, now)
                times = [cron.get_next(datetime).strftime("%m-%d %H:%M") for _ in range(count)]
            except Exception:
                times = ["[无效表达式]"] * count
            table.add_row(s.name, s.cron, f"{s.target_type}/{s.target_name}", *times)

        console.print(table)
    except ImportError:
        for s in config.schedules:
            try:
                cron = croniter(s.cron, now)
                times = [cron.get_next(datetime).strftime("%Y-%m-%d %H:%M") for _ in range(count)]
                click.echo(f"  {s.name} ({s.cron}):")
                for i, t in enumerate(times, 1):
                    click.echo(f"    #{i}: {t}")
            except Exception:
                click.echo(f"  {s.name}: 无效 cron 表达式 '{s.cron}'")


# ── eval 子命令组 ──


@click.group(name="eval")
def eval_group():
    """决策评估 — 追踪决策质量、回溯评估."""
    pass


@eval_group.command("track")
@click.argument("employee")
@click.option(
    "--category",
    "-c",
    required=True,
    type=click.Choice(["estimate", "recommendation", "commitment"]),
    help="决策类别",
)
@click.option("--content", "-m", required=True, help="决策内容")
@click.option("--expected", "-e", default="", help="预期结果")
@click.option("--meeting-id", default="", help="来源会议 ID")
def eval_track(employee: str, category: str, content: str, expected: str, meeting_id: str):
    """记录一个待评估的决策."""
    from crew.evaluation import EvaluationEngine

    engine = EvaluationEngine()
    decision = engine.track(
        employee=employee,
        category=category,
        content=content,
        expected_outcome=expected,
        meeting_id=meeting_id,
    )
    click.echo(f"已记录: [{decision.id}] ({decision.category}) {decision.content}")


@eval_group.command("list")
@click.option("--employee", default=None, help="按员工过滤")
@click.option(
    "--status", type=click.Choice(["pending", "evaluated"]), default=None, help="按状态过滤"
)
@click.option("-n", "--limit", type=int, default=20, help="返回条数")
def eval_list(employee: str | None, status: str | None, limit: int):
    """列出决策记录."""
    from crew.evaluation import EvaluationEngine

    engine = EvaluationEngine()
    decisions = engine.list_decisions(employee=employee, status=status, limit=limit)
    if not decisions:
        click.echo("暂无决策记录。")
        return

    for d in decisions:
        status_icon = "✓" if d.status == "evaluated" else "○"
        click.echo(f"  {status_icon} [{d.id}] {d.employee} ({d.category}): {d.content}")


@eval_group.command("run")
@click.argument("decision_id")
@click.option("--actual", "-a", required=True, help="实际结果")
@click.option("--evaluation", "-e", default="", help="评估结论（可选）")
def eval_run(decision_id: str, actual: str, evaluation: str):
    """评估一个决策并将结论写入记忆."""
    from crew.evaluation import EvaluationEngine

    engine = EvaluationEngine()
    result = engine.evaluate(decision_id, actual_outcome=actual, evaluation=evaluation)
    if result is None:
        click.echo(f"未找到决策: {decision_id}", err=True)
        sys.exit(1)
    click.echo(f"已评估: [{result.id}] {result.evaluation}")
    click.echo("评估结论已写入员工记忆。")


@eval_group.command("prompt")
@click.argument("decision_id")
def eval_prompt(decision_id: str):
    """生成回溯评估 prompt."""
    from crew.evaluation import EvaluationEngine

    engine = EvaluationEngine()
    prompt = engine.generate_evaluation_prompt(decision_id)
    if prompt is None:
        click.echo(f"未找到决策: {decision_id}", err=True)
        sys.exit(1)
    click.echo(prompt)


# ── trajectory 子命令组 ──


@click.group()
def trajectory():
    """轨迹管理 — 查看、打分、导出训练数据."""
    pass


@trajectory.command("list")
@click.option("-n", "--limit", type=int, default=20, help="显示条数")
def trajectory_list(limit: int):
    """列出已录制的轨迹."""
    traj_file = Path(".crew/trajectories/trajectories.jsonl")
    if not traj_file.exists():
        click.echo("暂无轨迹数据。")
        click.echo("使用 knowlyr-crew run <employee> --execute 录制首条轨迹。")
        return

    entries = []
    for line in traj_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))

    if not entries:
        click.echo("轨迹文件为空。")
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"已录制轨迹 (共 {len(entries)} 条)")
        table.add_column("#", justify="right", style="dim")
        table.add_column("员工", style="cyan")
        table.add_column("任务", style="green", max_width=40)
        table.add_column("模型", style="yellow")
        table.add_column("步数", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("结果", justify="center")

        for i, entry in enumerate(entries[-limit:], 1):
            agent = entry.get("agent", "")
            task_desc = entry.get("task", {}).get("description", "")
            model = entry.get("model", "")
            steps = entry.get("steps", [])
            outcome = entry.get("outcome", {})
            success = outcome.get("success", False)
            tokens = outcome.get("total_tokens", 0)

            table.add_row(
                str(i),
                agent.replace("crew/", ""),
                task_desc[:40],
                model.split("/")[-1][:20] if "/" in model else model[:20],
                str(len(steps)),
                str(tokens),
                "[green]OK[/green]" if success else "[red]FAIL[/red]",
            )

        console.print(table)
    except ImportError:
        for i, entry in enumerate(entries[-limit:], 1):
            agent = entry.get("agent", "")
            task = entry.get("task", {}).get("description", "")[:40]
            steps = len(entry.get("steps", []))
            click.echo(f"  {i}. {agent} | {task} | {steps} steps")

    click.echo(f"\n文件: {traj_file}", err=True)


@trajectory.command("score")
@click.option("--all", "score_all", is_flag=True, help="打分所有轨迹")
@click.option("-n", "--last", type=int, default=0, help="打分最后 N 条")
@click.option("--provider", default="openai", help="LLM judge provider (openai/anthropic)")
@click.option("--model", "judge_model", default=None, help="LLM judge 模型名")
@click.option("--base-url", default=None, help="OpenAI 兼容 API base URL")
def trajectory_score(
    score_all: bool, last: int, provider: str, judge_model: str | None, base_url: str | None
):
    """对轨迹进行 Reward 打分."""
    import os

    traj_file = Path(".crew/trajectories/trajectories.jsonl")
    if not traj_file.exists():
        click.echo("暂无轨迹数据。", err=True)
        sys.exit(1)

    try:
        from agentrecorder.schema import Trajectory
        from agentreward import RewardEngine
        from agentreward.config import RewardConfig

        trajectories = Trajectory.from_jsonl(traj_file)
    except ImportError as e:
        click.echo(f"Error: 依赖未安装 — {e}", err=True)
        click.echo("请运行: pip install knowlyr-recorder knowlyr-reward", err=True)
        sys.exit(1)

    if not trajectories:
        click.echo("轨迹文件为空。")
        return

    if last > 0:
        trajectories = trajectories[-last:]
    elif not score_all:
        trajectories = trajectories[-1:]
        click.echo("提示: 默认只打分最后一条。用 --all 打分全部，或 -n 5 打分最后 5 条。", err=True)

    # crew 的 AI 员工是对话类 agent，使用 conversation 领域评估
    config_kwargs: dict = {"provider": provider, "domain": "conversation"}
    if provider == "openai":
        if not base_url and os.environ.get("MOONSHOT_API_KEY"):
            config_kwargs["base_url"] = "https://api.moonshot.cn/v1"
            config_kwargs["api_key"] = os.environ["MOONSHOT_API_KEY"]
            config_kwargs["model_name"] = judge_model or "moonshot-v1-32k"
        elif base_url:
            config_kwargs["base_url"] = base_url
            if judge_model:
                config_kwargs["model_name"] = judge_model
        elif judge_model:
            config_kwargs["model_name"] = judge_model
    elif judge_model:
        config_kwargs["model_name"] = judge_model

    engine = RewardEngine(RewardConfig(**config_kwargs))
    for traj in trajectories:
        traj_dict = traj.model_dump() if hasattr(traj, "model_dump") else traj
        result = engine.score(traj_dict)
        agent = traj.agent.replace("crew/", "")
        task = (
            traj.task.description[:30] if hasattr(traj.task, "description") else str(traj.task)[:30]
        )
        click.echo(f"\n{agent} | {task}")
        click.echo(f"  总分: {result.total_score:.2f}")
        click.echo(f"  结果分: {result.outcome_score:.2f}")
        click.echo(f"  过程分: {result.process_score:.2f}")
        click.echo(f"  步骤数: {len(result.step_rewards)}")
        # 显示各维度平均分
        if result.step_rewards:
            all_rubric_ids: list[str] = []
            for sr in result.step_rewards:
                for rid in sr.rubric_scores:
                    if rid not in all_rubric_ids:
                        all_rubric_ids.append(rid)
            if all_rubric_ids:
                click.echo("  维度:")
                for rid in all_rubric_ids:
                    scores = [sr.rubric_scores.get(rid, 0) for sr in result.step_rewards]
                    avg = sum(scores) / len(scores)
                    click.echo(f"    {rid}: {avg:.2f}")


@trajectory.command("export")
@click.option(
    "-f", "--format", "fmt", type=click.Choice(["sft", "dpo"]), default="sft", help="导出格式"
)
@click.option(
    "-o", "--output", "output_path", type=click.Path(), required=True, help="输出文件路径"
)
def trajectory_export(fmt: str, output_path: str):
    """导出轨迹为训练数据."""
    traj_file = Path(".crew/trajectories/trajectories.jsonl")
    if not traj_file.exists():
        click.echo("暂无轨迹数据。", err=True)
        sys.exit(1)

    try:
        from trajectoryhub import DatasetExporter
    except ImportError:
        click.echo("Error: knowlyr-hub 未安装。请运行: pip install knowlyr-hub", err=True)
        sys.exit(1)

    exporter = DatasetExporter(str(traj_file.parent))
    if fmt == "sft":
        exporter.export_sft(output_path)
    else:
        exporter.export_dpo(output_path)

    click.echo(f"已导出 {fmt.upper()} 数据: {output_path}")


@trajectory.command("convert")
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="输出文件路径 (默认追加到 .crew/trajectories/trajectories.jsonl)",
)
def trajectory_convert(output: str | None):
    """将已有 sessions（含 --execute 回复）转为标准轨迹格式."""
    try:
        from agentrecorder import Recorder
        from agentrecorder.adapters import CrewAdapter
    except ImportError:
        click.echo("Error: knowlyr-recorder 未安装。请运行: pip install knowlyr-recorder", err=True)
        sys.exit(1)

    session_dir = Path(".crew/sessions")
    if not session_dir.is_dir():
        click.echo("未找到 .crew/sessions/ 目录。")
        return

    adapter = CrewAdapter()
    recorder = Recorder(adapter)
    output_path = output or ".crew/trajectories/trajectories.jsonl"

    converted = 0
    for f in sorted(session_dir.glob("*.jsonl")):
        if adapter.validate(str(f)):
            traj = recorder.convert(str(f))
            traj.to_jsonl(output_path)
            converted += 1

    if converted:
        click.echo(f"已转换 {converted} 条 session → {output_path}")
    else:
        click.echo("未找到包含执行回复的 session。")
        click.echo("使用 knowlyr-crew run <employee> --execute 产生可转换的 session。")


@trajectory.command("stats")
def trajectory_stats():
    """显示轨迹数据统计."""
    from collections import Counter

    traj_file = Path(".crew/trajectories/trajectories.jsonl")
    if not traj_file.exists():
        click.echo("暂无轨迹数据。")
        return

    entries = []
    for line in traj_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))

    if not entries:
        click.echo("轨迹文件为空。")
        return

    # 统计
    employee_counter = Counter()
    total_steps = 0
    total_tokens = 0
    success_count = 0

    for entry in entries:
        agent = entry.get("agent", "").replace("crew/", "")
        employee_counter[agent] += 1
        total_steps += len(entry.get("steps", []))
        total_tokens += entry.get("outcome", {}).get("total_tokens", 0)
        if entry.get("outcome", {}).get("success"):
            success_count += 1

    click.echo(f"轨迹总数: {len(entries)}")
    click.echo(f"总步数:   {total_steps}")
    click.echo(f"总 Tokens: {total_tokens:,}")
    click.echo(
        f"成功率:   {success_count}/{len(entries)} ({100 * success_count / len(entries):.0f}%)"
    )
    click.echo("\n按员工统计:")
    for emp, count in employee_counter.most_common():
        click.echo(f"  {emp}: {count} 条")
