"""Crew CLI — 命令行界面."""

import json
import logging
import sys
from pathlib import Path

import click

from crew import __version__
from crew.discovery import discover_employees
from crew.engine import CrewEngine
from crew.parser import parse_employee, validate_employee


def _setup_logging(verbose: bool) -> None:
    """配置日志级别."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(version=__version__, prog_name="knowlyr-crew")
@click.option("-v", "--verbose", is_flag=True, default=False, help="显示详细日志")
@click.pass_context
def main(ctx: click.Context, verbose: bool):
    """Crew — 数字员工管理框架

    用 Markdown 定义数字员工，在 Claude Code 等 AI 工具中加载使用。
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@main.command("list")
@click.option("--tag", type=str, default=None, help="按标签过滤")
@click.option(
    "--layer", type=click.Choice(["builtin", "global", "skill", "project"]),
    default=None, help="按来源层过滤",
)
@click.option(
    "-f", "--format", "output_format",
    type=click.Choice(["table", "json"]),
    default="table", help="输出格式",
)
@click.pass_context
def list_cmd(ctx: click.Context, tag: str | None, layer: str | None, output_format: str):
    """列出所有可用员工."""
    result = discover_employees()

    employees = list(result.employees.values())

    # 过滤
    if tag:
        employees = [e for e in employees if tag in e.tags]
    if layer:
        employees = [e for e in employees if e.source_layer == layer]

    if not employees:
        click.echo("未找到员工。", err=True)
        return

    if output_format == "json":
        data = [
            {
                "name": e.name,
                "display_name": e.effective_display_name,
                "description": e.description,
                "tags": e.tags,
                "triggers": e.triggers,
                "layer": e.source_layer,
            }
            for e in employees
        ]
        click.echo(json.dumps(data, ensure_ascii=False, indent=2))
        return

    # table 格式
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="数字员工")
        table.add_column("名称", style="cyan")
        table.add_column("显示名", style="green")
        table.add_column("来源", style="yellow")
        table.add_column("标签", style="blue")
        table.add_column("描述")

        for emp in employees:
            table.add_row(
                emp.name,
                emp.effective_display_name,
                emp.source_layer,
                ", ".join(emp.tags),
                emp.description,
            )

        console.print(table)
    except ImportError:
        # fallback：无 rich 时用纯文本
        click.echo(f"{'名称':<20} {'来源':<8} {'描述'}")
        click.echo("-" * 60)
        for emp in employees:
            click.echo(f"{emp.name:<20} {emp.source_layer:<8} {emp.description}")

    # 冲突信息
    if result.conflicts and ctx.obj.get("verbose"):
        click.echo(f"\n⚠ {len(result.conflicts)} 个冲突:", err=True)
        for c in result.conflicts:
            click.echo(f"  {c}", err=True)


@main.command()
@click.argument("name")
def show(name: str):
    """查看员工详情."""
    result = discover_employees()
    emp = result.get(name)

    if emp is None:
        click.echo(f"未找到员工: {name}", err=True)
        sys.exit(1)

    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel

        console = Console()

        # 元信息
        meta_lines = [
            f"**名称**: {emp.name}",
            f"**显示名**: {emp.effective_display_name}",
            f"**版本**: {emp.version}",
            f"**描述**: {emp.description}",
            f"**来源**: {emp.source_layer} ({emp.source_path})",
        ]
        if emp.tags:
            meta_lines.append(f"**标签**: {', '.join(emp.tags)}")
        if emp.triggers:
            meta_lines.append(f"**触发词**: {', '.join(emp.triggers)}")
        if emp.tools:
            meta_lines.append(f"**需要工具**: {', '.join(emp.tools)}")
        if emp.context:
            meta_lines.append(f"**预读上下文**: {', '.join(emp.context)}")
        if emp.args:
            meta_lines.append("**参数**:")
            for arg in emp.args:
                req = " (必填)" if arg.required else ""
                default = f" [默认: {arg.default}]" if arg.default else ""
                meta_lines.append(f"  - `{arg.name}`: {arg.description}{req}{default}")

        console.print(Panel(Markdown("\n".join(meta_lines)), title=emp.effective_display_name))
        console.print()
        console.print(Markdown(emp.body))
    except ImportError:
        click.echo(f"=== {emp.effective_display_name} ===")
        click.echo(f"名称: {emp.name}")
        click.echo(f"描述: {emp.description}")
        click.echo(f"来源: {emp.source_layer}")
        if emp.args:
            click.echo("参数:")
            for arg in emp.args:
                click.echo(f"  {arg.name}: {arg.description}")
        click.echo()
        click.echo(emp.body)


@main.command()
@click.argument("name")
@click.argument("positional_args", nargs=-1)
@click.option("--arg", "named_args", multiple=True, help="命名参数 (key=value)")
@click.option("--agent-id", type=int, default=None, help="绑定 knowlyr-id Agent ID")
@click.option("--raw", is_flag=True, help="输出原始渲染结果（不包裹 prompt 格式）")
@click.option("--copy", "to_clipboard", is_flag=True, help="复制到剪贴板")
@click.option("-o", "--output", type=click.Path(), help="输出到文件")
def run(
    name: str,
    positional_args: tuple[str, ...],
    named_args: tuple[str, ...],
    agent_id: int | None,
    raw: bool,
    to_clipboard: bool,
    output: str | None,
):
    """加载员工并生成 prompt.

    NAME 为员工名称或触发别名。后续参数作为位置参数传递。
    """
    result = discover_employees()
    emp = result.get(name)

    if emp is None:
        click.echo(f"未找到员工: {name}", err=True)
        sys.exit(1)

    engine = CrewEngine()

    # 解析命名参数
    args_dict: dict[str, str] = {}
    for item in named_args:
        if "=" in item:
            k, v = item.split("=", 1)
            args_dict[k] = v

    # 位置参数也填充到对应的 args.name
    for i, val in enumerate(positional_args):
        if i < len(emp.args):
            args_dict.setdefault(emp.args[i].name, val)

    # 校验参数
    errors = engine.validate_args(emp, args=args_dict)
    if errors:
        for err in errors:
            click.echo(f"参数错误: {err}", err=True)
        sys.exit(1)

    # 获取 Agent 身份（可选）
    agent_identity = None
    if agent_id is not None:
        try:
            from crew.id_client import fetch_agent_identity
            agent_identity = fetch_agent_identity(agent_id)
            if agent_identity is None:
                click.echo(f"Warning: 无法获取 Agent {agent_id} 身份，继续生成 prompt", err=True)
        except ImportError:
            click.echo("Warning: httpx 未安装，无法连接 knowlyr-id", err=True)

    # 生成
    if raw:
        text = engine.render(emp, args=args_dict, positional=list(positional_args))
    else:
        text = engine.prompt(
            emp, args=args_dict, positional=list(positional_args),
            agent_identity=agent_identity,
        )

    # 记录工作日志
    try:
        from crew.log import WorkLogger

        work_logger = WorkLogger()
        session_id = work_logger.create_session(emp.name, args=args_dict, agent_id=agent_id)
        work_logger.add_entry(session_id, "prompt_generated", f"via CLI, {len(text)} chars")
    except Exception:
        pass  # 日志失败不影响主流程

    # 发送心跳（可选）
    if agent_id is not None:
        try:
            from crew.id_client import send_heartbeat
            send_heartbeat(agent_id, detail=f"employee={emp.name}")
        except Exception:
            pass  # 心跳失败不影响主流程

    # 输出
    if output:
        Path(output).write_text(text, encoding="utf-8")
        click.echo(f"已写入: {output}", err=True)
    elif to_clipboard:
        try:
            import subprocess
            proc = subprocess.run(["pbcopy"], input=text.encode(), check=True)
            click.echo("已复制到剪贴板。", err=True)
        except Exception:
            click.echo(text)
            click.echo("（复制到剪贴板失败，已输出到终端）", err=True)
    else:
        click.echo(text)


@main.command()
@click.argument("path", type=click.Path(exists=True))
def validate(path: str):
    """校验 EMPLOYEE.md 文件或目录."""
    target = Path(path)
    files = list(target.glob("*.md")) if target.is_dir() else [target]

    total = 0
    passed = 0

    for f in files:
        if f.name.startswith("_") or f.name == "README.md":
            continue
        total += 1
        try:
            emp = parse_employee(f)
            errors = validate_employee(emp)
            if errors:
                click.echo(f"✗ {f.name}: {'; '.join(errors)}")
            else:
                click.echo(f"✓ {f.name} ({emp.effective_display_name})")
                passed += 1
        except ValueError as e:
            click.echo(f"✗ {f.name}: {e}")

    click.echo(f"\n{passed}/{total} 通过校验")
    if passed < total:
        sys.exit(1)


@main.command()
@click.option("--employee", type=str, default=None, help="创建指定员工的模板")
def init(employee: str | None):
    """初始化 .crew/ 目录或创建员工模板."""
    crew_dir = Path.cwd() / ".crew"
    crew_dir.mkdir(exist_ok=True)

    if employee:
        # 创建员工模板
        template = f"""---
name: {employee}
display_name: {employee}
description: 在此填写一句话描述
tags: []
triggers: []
args:
  - name: target
    description: 目标
    required: true
output:
  format: markdown
---

# 角色定义

你是……

## 工作流程

1. 第一步
2. 第二步
3. 第三步

## 输出格式

按需定义输出格式。
"""
        out_path = crew_dir / f"{employee}.md"
        if out_path.exists():
            click.echo(f"文件已存在: {out_path}", err=True)
            sys.exit(1)
        out_path.write_text(template, encoding="utf-8")
        click.echo(f"已创建: {out_path}")
    else:
        click.echo(f"已初始化: {crew_dir}/")
        click.echo("使用 --employee <name> 创建员工模板。")


# ── Skills 导出命令 ──


@main.command("export")
@click.argument("name")
@click.option(
    "-d", "--dir", "project_dir", type=click.Path(),
    default=None, help="项目根目录（默认当前目录）",
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


@main.command("export-all")
@click.option(
    "-d", "--dir", "project_dir", type=click.Path(),
    default=None, help="项目根目录（默认当前目录）",
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


@main.command()
@click.option("--clean", is_flag=True, help="删除不再存在的孤儿技能目录")
@click.option(
    "-d", "--dir", "project_dir", type=click.Path(),
    default=None, help="项目根目录（默认当前目录）",
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


@main.group()
def log():
    """工作日志管理."""
    pass


@log.command("list")
@click.option("--employee", type=str, default=None, help="按员工过滤")
@click.option("-n", "--limit", type=int, default=20, help="返回条数")
def log_list(employee: str | None, limit: int):
    """查看工作日志列表."""
    from crew.log import WorkLogger

    logger = WorkLogger()
    sessions = logger.list_sessions(employee_name=employee, limit=limit)

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
            click.echo(f"{s['session_id']}  {s['employee_name']}  {s['started_at']}  ({s['entries']} 条)")


@log.command("show")
@click.argument("session_id")
def log_show(session_id: str):
    """查看某次工作的详细日志."""
    from crew.log import WorkLogger

    logger = WorkLogger()
    entries = logger.get_session(session_id)

    if not entries:
        click.echo(f"未找到 session: {session_id}", err=True)
        sys.exit(1)

    for entry in entries:
        ts = entry.get("timestamp", "")
        action = entry.get("action", "")
        detail = entry.get("detail", "")
        click.echo(f"[{ts}] {action}: {detail}")


# ── mcp 命令 ──


@main.command()
def mcp():
    """启动 MCP Server（stdio 模式）."""
    from crew.mcp_server import main as mcp_main
    mcp_main()
