"""员工操作命令 — list, show, run, validate, init, delete, avatar, rollback, catalog, permissions, lint, check."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import jsonschema

from crew import sdk
from crew.cli import (
    _employee_root,
    _finish_transcript,
    _record_session_summary,
    _record_transcript_event,
    _record_transcript_message,
    _start_transcript,
    _suggest_similar,
    logger,
)
from crew.discovery import discover_employees
from crew.discussion import load_discussion, validate_discussion
from crew.engine import CrewEngine
from crew.lanes import lane_lock
from crew.log import WorkLogger
from crew.parser import parse_employee, parse_employee_dir, validate_employee
from crew.pipeline import load_pipeline, validate_pipeline

# ── list ──


@click.command("list")
@click.option("--tag", type=str, default=None, help="按标签过滤")
@click.option(
    "--layer",
    type=click.Choice(["builtin", "skill", "private"]),
    default=None,
    help="按来源层过滤",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="输出格式",
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


# ── lint helpers ──


def _lint_file(path: Path, project_dir: Path) -> list[str]:
    errors: list[str] = []
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as exc:  # YAML 解析或文件读取失败
        return [f"{path}: 无法解析 YAML ({exc})"]

    if not isinstance(data, dict):
        return [f"{path}: YAML 内容应为对象"]

    kind = "pipeline" if "steps" in data else "discussion" if "participants" in data else None

    if kind == "pipeline":
        schema_path = Path.cwd() / "schemas" / "pipeline.schema.json"
        if schema_path.exists():
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            try:
                jsonschema.validate(instance=data, schema=schema)
            except jsonschema.ValidationError as exc:
                return [f"{path}: schema 校验失败 - {exc.message}"]

        try:
            pipeline = load_pipeline(path)
        except (ValueError, yaml.YAMLError, OSError) as exc:
            return [f"{path}: 解析失败 ({exc})"]

        errors.extend(f"{path}: {e}" for e in validate_pipeline(pipeline, project_dir=project_dir))
    elif kind == "discussion":
        schema_path = Path.cwd() / "schemas" / "discussion.schema.json"
        if schema_path.exists():
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            try:
                jsonschema.validate(instance=data, schema=schema)
            except jsonschema.ValidationError as exc:
                return [f"{path}: schema 校验失败 - {exc.message}"]

        try:
            discussion = load_discussion(path)
        except (ValueError, yaml.YAMLError, OSError) as exc:
            return [f"{path}: 解析失败 ({exc})"]

        errors.extend(
            f"{path}: {e}" for e in validate_discussion(discussion, project_dir=project_dir)
        )
    else:
        errors = [f"{path}: 未识别的 YAML 类型（缺少 steps/participants）"]

    return errors


def _lint_targets(targets: list[Path]) -> list[str]:
    project_dir = Path.cwd()
    errors: list[str] = []

    for target in targets:
        if not target.exists():
            errors.append(f"{target}: 路径不存在")
            continue

        if target.is_file():
            errors.extend(_lint_file(target, project_dir))
        else:
            files = sorted(p for p in target.rglob("*.yaml"))
            if not files:
                errors.append(f"{target}: 未找到 *.yaml 文件")
            for file in files:
                errors.extend(_lint_file(file, project_dir))

    return errors


def _scan_log_severity(log_dir: Path) -> tuple[dict[str, int], int, dict[str, dict[str, int]]]:
    import json as _json

    counts: dict[str, int] = {}
    by_employee: dict[str, dict[str, int]] = {}
    entries = 0
    if not log_dir.is_dir():
        return counts, entries, by_employee

    for file in sorted(log_dir.glob("*.jsonl")):
        try:
            content = file.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                data = _json.loads(line)
            except ValueError:
                continue
            severity = str(data.get("severity", "info"))
            counts[severity] = counts.get(severity, 0) + 1
            entries += 1
            emp_name = str(data.get("employee_name", "")) or "unknown"
            emp_counts = by_employee.setdefault(emp_name, {})
            emp_counts[severity] = emp_counts.get(severity, 0) + 1

    return counts, entries, by_employee


# ── lint command ──


@click.command("lint")
@click.argument("paths", nargs=-1, type=click.Path(path_type=Path))
def lint_cmd(paths: tuple[Path, ...]):
    """Lint pipelines/discussions YAML（默认扫描 .crew/pipelines 和 .crew/discussions）。"""
    targets = list(paths)

    if not targets:
        default_dirs = [Path.cwd() / ".crew" / "pipelines", Path.cwd() / ".crew" / "discussions"]
        targets = [d for d in default_dirs if d.exists()]
        if not targets:
            click.echo("未指定路径，且未找到 .crew/pipelines 或 .crew/discussions。", err=True)
            sys.exit(1)

    errors = _lint_targets(targets)
    if errors:
        for err in errors:
            click.echo(err, err=True)
        click.echo(f"\nLint 失败: {len(errors)} 个问题", err=True)
        sys.exit(1)

    click.echo("Lint 通过 ✓")


# ── permissions ──


@click.command("permissions")
@click.argument("name")
def permissions_cmd(name: str):
    """显示员工权限详情（角色、有效工具、被禁止工具）。"""
    from crew.tool_schema import resolve_effective_tools, validate_permissions

    result = discover_employees(project_dir=Path.cwd(), cache_ttl=0)
    emp = result.get(name)
    if emp is None:
        click.echo(f"未找到员工: {name}", err=True)
        sys.exit(1)

    click.echo(f"员工: {emp.name}")
    click.echo(f"声明工具 ({len(emp.tools)}): {', '.join(sorted(emp.tools)) or '(无)'}")

    if emp.permissions is None:
        click.echo("权限策略: (未配置 — 使用 tools 原样)")
        return

    p = emp.permissions
    if p.roles:
        click.echo(f"角色: {', '.join(p.roles)}")
    if p.allow:
        click.echo(f"额外允许: {', '.join(p.allow)}")
    if p.deny:
        click.echo(f"显式禁止: {', '.join(p.deny)}")

    effective = resolve_effective_tools(emp)
    denied = set(emp.tools) - effective
    click.echo(f"有效工具 ({len(effective)}): {', '.join(sorted(effective)) or '(无)'}")
    if denied:
        click.echo(f"被禁止工具 ({len(denied)}): {', '.join(sorted(denied))}")

    warnings = validate_permissions(emp)
    if warnings:
        click.echo("警告:")
        for w in warnings:
            click.echo(f"  - {w}")


# ── check ──


@click.command("check")
@click.option("--no-lint", is_flag=True, default=False, help="跳过 lint 检查")
@click.option("--no-logs", is_flag=True, default=False, help="跳过日志质量检查")
@click.option("--json", "json_output", is_flag=True, help="JSON 输出")
@click.option(
    "--path",
    "lint_paths",
    multiple=True,
    type=click.Path(path_type=Path),
    help="要 lint 的路径（可多次指定）",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    default=None,
    help="将 JSON 报告写入文件（默认 .crew/quality-report.json）",
)
@click.option("--no-file", is_flag=True, help="不写入 JSON 报告")
def check_cmd(
    no_lint: bool,
    no_logs: bool,
    json_output: bool,
    lint_paths: tuple[Path, ...],
    output_file: Path | None,
    no_file: bool,
):
    """执行 lint + 日志质量检查。"""
    report: dict[str, Any] = {}
    exit_code = 0

    if not no_lint:
        targets = list(lint_paths)
        if not targets:
            default_dirs = [
                Path.cwd() / ".crew" / "pipelines",
                Path.cwd() / ".crew" / "discussions",
            ]
            targets = [d for d in default_dirs if d.exists()]
        lint_errors = _lint_targets(targets)
        if lint_errors:
            exit_code = 1
            report["lint"] = {"status": "failed", "errors": lint_errors}
        else:
            report["lint"] = {"status": "ok"}

    if not no_logs:
        work_logger = WorkLogger()
        counts, entries, by_employee = _scan_log_severity(work_logger.log_dir)
        report["logs"] = {"entries": entries, "severity": counts, "by_employee": by_employee}
        if counts.get("critical", 0) > 0:
            exit_code = 1

    report.setdefault("metadata", {})["generated_at"] = datetime.now(timezone.utc).isoformat()

    if json_output:
        click.echo(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        if "lint" in report:
            status = report["lint"].get("status")
            if status == "ok":
                click.echo("Lint: OK")
            else:
                click.echo("Lint: FAILED", err=True)
                for err in report["lint"].get("errors", []):
                    click.echo(f"  - {err}", err=True)
        if "logs" in report:
            logs = report["logs"]
            click.echo(f"日志条目: {logs['entries']}")
            severity = logs.get("severity", {})
            if severity:
                click.echo("Severity:")
                for key, value in severity.items():
                    click.echo(f"  {key}: {value}")
            by_emp = logs.get("by_employee", {})
            if by_emp:
                click.echo("按员工统计:")
                for emp, sev in by_emp.items():
                    summary = ", ".join(f"{k}:{v}" for k, v in sev.items())
                    click.echo(f"  {emp}: {summary}")

    if not no_logs and not no_file:
        report_path = output_file or (Path.cwd() / ".crew" / "quality-report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if not json_output:
            click.echo(f"已写入: {report_path}")

    if exit_code != 0:
        sys.exit(exit_code)


# ── catalog ──


@click.group()
def catalog():
    """员工 Catalog 查询."""
    pass


def _catalog_data() -> list[dict[str, Any]]:
    result = discover_employees()
    employees = []
    for emp in result.employees.values():
        employees.append(
            {
                "name": emp.name,
                "display_name": emp.display_name,
                "character_name": emp.character_name,
                "description": emp.description,
                "tags": emp.tags,
                "triggers": emp.triggers,
                "tools": emp.tools,
                "context": emp.context,
                "agent_id": emp.agent_id,
                "source_layer": emp.source_layer,
            }
        )
    return employees


@catalog.command("list")
@click.option("--format", "output_format", type=click.Choice(["table", "json"]), default="table")
def catalog_list(output_format: str):
    """列出所有员工元数据."""
    data = _catalog_data()

    if not data:
        click.echo("未找到员工。", err=True)
        return

    if output_format == "json":
        click.echo(json.dumps(data, ensure_ascii=False, indent=2))
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="员工 Catalog")
        table.add_column("名称", style="cyan")
        table.add_column("显示名")
        table.add_column("角色名")
        table.add_column("AgentID")
        table.add_column("触发词")
        for item in data:
            table.add_row(
                item["name"],
                item["display_name"],
                item["character_name"],
                str(item["agent_id"] or "-"),
                ", ".join(item["triggers"]),
            )
        console.print(table)
    except ImportError:
        for item in data:
            click.echo(
                f"{item['name']:<18} {item['display_name']:<12} agent={item['agent_id'] or '-'}"
            )


@catalog.command("show")
@click.argument("name")
@click.option("--json", "json_output", is_flag=True, help="JSON 输出")
def catalog_show(name: str, json_output: bool):
    """查看指定员工元数据."""
    for item in _catalog_data():
        if item["name"] == name or name in item.get("triggers", []):
            if json_output:
                click.echo(json.dumps(item, ensure_ascii=False, indent=2))
            else:
                click.echo(json.dumps(item, ensure_ascii=False, indent=2))
            return
    click.echo(f"未找到员工: {name}", err=True)
    sys.exit(1)


# ── show ──


@click.command()
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


# ── run ──


_CLI_MAX_TOOL_ROUNDS = 10


def _generate_mock_response(tool_name: str, arguments: dict[str, Any]) -> str:
    """生成随机化的模拟工具返回，避免训练数据里出现固定数字."""
    import random

    if tool_name == "query_stats":
        dau = random.randint(700, 1200)
        wau = random.randint(3000, 5000)
        msgs = random.randint(2500, 5000)
        new_users = random.randint(15, 60)
        agents = random.randint(30, 40)
        revenue = random.randint(30000, 80000)
        return json.dumps(
            {
                "dau": dau,
                "wau": wau,
                "messages_today": msgs,
                "new_users_this_week": new_users,
                "active_agents": agents,
                "revenue_mtd": revenue,
            }
        )
    if tool_name == "lookup_user":
        uid = random.randint(1001, 9999)
        msgs = random.randint(50, 500)
        return json.dumps(
            {
                "user_id": uid,
                "name": "示例用户",
                "email": "user@example.com",
                "created_at": "2025-12-01",
                "messages_sent": msgs,
            }
        )
    if tool_name == "query_agent_work":
        done = random.randint(2, 8)
        prog = random.randint(1, 4)
        mins = random.randint(1, 60)
        return json.dumps(
            {
                "agent": "requested",
                "tasks_completed_today": done,
                "tasks_in_progress": prog,
                "last_active": f"{mins} 分钟前",
            }
        )
    if tool_name == "list_agents":
        return '[{"agent_id": "AI3073", "name": "ceo-assistant", "status": "active"}, {"agent_id": "AI3001", "name": "code-reviewer", "status": "active"}, {"agent_id": "AI3002", "name": "product-manager", "status": "active"}]'
    if tool_name == "read_messages":
        unread = random.randint(0, 5)
        return json.dumps(
            {
                "unread": unread,
                "messages": [
                    {
                        "from": "code-reviewer",
                        "content": "PR #42 审查完成，有两个建议",
                        "time": "14:30",
                    },
                    {"from": "product-manager", "content": "新需求文档已更新", "time": "15:00"},
                ],
            }
        )
    if tool_name == "get_system_health":
        cpu = random.randint(15, 45)
        mem = random.randint(40, 75)
        latency = random.randint(200, 500)
        return json.dumps(
            {
                "status": "healthy",
                "uptime": "72h",
                "cpu": f"{cpu}%",
                "memory": f"{mem}%",
                "api_latency_p99": f"{latency}ms",
            }
        )
    if tool_name == "read_notes":
        return '{"notes": []}'
    if tool_name == "web_search":
        return '{"results": [{"title": "相关搜索结果", "snippet": "这是一条模拟的搜索结果摘要。", "url": "https://example.com"}]}'
    # 飞书
    if tool_name == "search_feishu_docs":
        return '{"docs": [{"title": "Q1 战略规划", "url": "https://feishu.cn/docx/abc123", "type": "doc"}, {"title": "产品路线图", "url": "https://feishu.cn/docx/def456", "type": "wiki"}]}'
    if tool_name == "read_feishu_doc":
        return '{"title": "Q1 战略规划", "content": "本季度核心目标：日活突破 2000，AI 自动化率 80%，签约 5 家付费客户。"}'
    if tool_name == "create_feishu_doc":
        return '{"status": "created", "document_id": "doc_xyz789", "url": "https://feishu.cn/docx/xyz789"}'
    if tool_name == "send_feishu_group":
        return '{"status": "sent", "message_id": "msg_feishu_001"}'
    # GitHub
    if tool_name == "github_prs":
        return '{"prs": [{"number": 42, "title": "feat: add new feature", "state": "open", "author": "kai", "url": "https://github.com/org/repo/pull/42"}]}'
    if tool_name == "github_issues":
        return '{"issues": [{"number": 10, "title": "Bug: login failure", "state": "open", "labels": ["bug"], "assignee": "kai"}]}'
    if tool_name == "github_repo_activity":
        commits = random.randint(8, 25)
        return json.dumps(
            {
                "commits_7d": commits,
                "contributors": 3,
                "recent": [
                    {"sha": "abc1234", "message": "fix: resolve auth issue", "author": "kai"}
                ],
            }
        )
    # Notion
    if tool_name == "notion_search":
        return '{"results": [{"title": "产品规划 2026", "url": "https://notion.so/abc123", "type": "page", "last_edited": "2026-02-14"}]}'
    if tool_name == "notion_read":
        return '{"title": "产品规划 2026", "content": "核心方向：AI Agent 平台化，目标客户：中小企业。"}'
    if tool_name == "notion_create":
        return '{"status": "created", "page_id": "page_notion_001", "url": "https://notion.so/page_notion_001"}'
    # 信息采集
    if tool_name == "read_url":
        return '{"content": "这是一篇模拟的网页正文内容，已自动提取去噪。"}'
    if tool_name == "rss_read":
        return '{"entries": [{"title": "v2.0 发布公告", "link": "https://example.com/blog/v2", "summary": "新版本带来了全新功能..."}]}'
    # 简单操作类工具 — 固定返回 ok
    _SIMPLE: dict[str, str] = {
        "send_message": '{"status": "sent", "message_id": "msg_12345"}',
        "delegate": '{"status": "delegated", "task_id": "task_67890"}',
        "mark_read": '{"status": "ok", "marked": 3}',
        "update_agent": '{"status": "updated"}',
        "create_feishu_event": '{"status": "created", "event_id": "evt_abc123", "calendar": "primary"}',
        "read_feishu_calendar": "02-16 10:00-11:00 团队周会 [event_id=evt_001]\n02-16 14:00-15:00 投资人沟通 [event_id=evt_002]\n02-16 16:30-17:00 产品评审 [event_id=evt_003]",
        "delete_feishu_event": "日程已删除 (event_id=evt_001)。",
        "create_feishu_task": "待办已创建：准备投资人会议材料，截止 2026-02-20 [task_id=task_abc123]",
        "list_feishu_tasks": "⬜ 准备投资人会议材料 截止02-20 [task_id=task_001]\n⬜ 整理Q1数据报告 截止02-18 [task_id=task_002]\n⬜ 确认下周出差行程 [task_id=task_003]\n✅ 发送新年祝福 [task_id=task_004]",
        "complete_feishu_task": "任务已完成 ✅ [task_id=task_abc123]",
        "delete_feishu_task": "任务已删除 [task_id=task_abc123]",
        "update_feishu_task": "任务已更新: 截止→2026-03-01 [task_id=task_abc123]",
        "get_datetime": "2026-02-16 09:30 星期一",
        "calculate": "1749600.56",
        "feishu_chat_history": "[02-15 14:30] ou_user1: 下午开会记得带材料\n[02-15 14:25] ou_user2: 好的收到\n[02-15 14:20] ou_user1: Q1 报告写完了吗",
        "weather": "上海市 当前 6.7℃，湿度 78%，空气优(PM2.5:21)\n2026-02-15(星期日) 多云 7℃~15℃ 东北风2级\n2026-02-16(星期一) 小雨 4℃~8℃ 东北风2级\n2026-02-17(星期二) 晴 3℃~10℃ 北风1级",
        "exchange_rate": "基准: 1 USD\n= 6.91 CNY",
        "stock_price": "贵州茅台 (SH600519)\n现价: ¥1485.30  涨跌: -1.30 (-0.09%)\n今开: 1486.60  最高: 1507.80  最低: 1470.58",
        "send_feishu_dm": "私聊消息已发送给 ou_xxx。",
        "feishu_group_members": "刘凯 [open_id=ou_de186aad7faf2c2b72b78223577e2bd9]",
        "create_note": '{"status": "saved", "note_id": "note_001"}',
        "translate": "The quarterly financial report shows a 15% increase in revenue.",
        "countdown": "距离「产品发布」还有 12 天 6 小时。",
        "trending": "🔥 微博热搜\n\n1. AI大模型重大突破 [热]  (2,345,678)\n2. 春季新品发布会  (1,234,567)\n3. 教育改革新政策  (987,654)",
        "read_feishu_sheet": "姓名 | 部门 | 职级\n---|---|---\n张三 | 产品 | P6\n李四 | 技术 | P7\n王五 | 设计 | P5",
        "update_feishu_sheet": "写入成功，更新了 6 个单元格。",
        "list_feishu_approvals": "⏳ [报销审批] 02-15 14:30 (instance=inst_001)\n⏳ [请假审批] 02-14 10:00 (instance=inst_002)",
        "unit_convert": "100 km = 62.14 mi",
        "random_pick": "🎯 选中了：火锅",
        "holidays": "📅 2026年节假日安排\n\n01-01 🟢 放假 元旦\n01-26 🟢 放假 春节\n01-27 🟢 放假 春节",
        "timestamp_convert": "时间戳 1708012800 = 2024-02-16 00:00:00 周五（北京时间）",
        "create_feishu_spreadsheet": "表格已创建: Q1数据表\ntoken: shtcnXXXXXX\nhttps://abc.feishu.cn/sheets/shtcnXXXXXX",
        "feishu_contacts": "张三 (产品部) [open_id=ou_abc123]\n李四 (技术部) [open_id=ou_def456]",
        "text_extract": "【邮箱】\n  kai@example.com\n【手机号】\n  13800138000\n【URL】\n  https://example.com",
        "json_format": '{\n  "name": "Kai",\n  "role": "CEO"\n}',
        "password_gen": "🔐 随机密码（16位）：\n\n1. Kx9$mP2vLq@nR5wT\n2. hJ7&bN4cYs#fA8eD\n3. Wt6*kM3pZx!gU9rQ",
        "ip_lookup": "IP: 8.8.8.8\n位置: 美国 弗吉尼亚 阿什本\n运营商: Google LLC",
        "short_url": "短链接: https://cleanuri.com/abc123\n原链接: https://example.com/very-long-url",
        "word_count": "字符: 256（不含空格 210） | 中文: 180 字 | 英文: 12 词 | 行: 8 | 段落: 3",
        "base64_codec": "编码结果:\nSGVsbG8gV29ybGQ=",
        "color_convert": "HEX: #FF5733\nRGB: rgb(255, 87, 51)\nHSL: hsl(11, 100%, 60%)",
        "cron_explain": "cron: 0 9 * * 1-5\n\n  分钟: 0\n  小时: 9\n  日: 每日\n  月: 每月\n  星期: 周一 到 周五",
        "regex_test": "找到 3 个匹配：\n\n1. 「abc」 位置 0-3\n2. 「abc」 位置 10-13\n3. 「abc」 位置 20-23",
        "hash_gen": "SHA256: 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
        "url_codec": "编码结果:\n%E4%BD%A0%E5%A5%BD%E4%B8%96%E7%95%8C",
        "load_tools": "已加载。现在可以直接调用这些工具。",
        "feishu_bitable": "共 5 条记录：\n1. 姓名: 张三 | 部门: 产品 | 状态: 在职\n2. 姓名: 李四 | 部门: 技术 | 状态: 在职\n3. 姓名: 王五 | 部门: 设计 | 状态: 离职",
        "feishu_wiki": "- 产品需求文档 PRD v2.0 [产品知识库]\n  https://abc.feishu.cn/wiki/xxx\n- API 接口规范 [技术知识库]\n  https://abc.feishu.cn/wiki/yyy",
        "approve_feishu": "审批已通过。",
        "summarize": "[摘要任务] 请用要点列表总结以下内容：\n\n(原文内容)",
        "sentiment": "[情感分析任务] 请分析以下文本的情感倾向：\n\n(原文内容)",
        "email_send": "邮件功能尚未配置 SMTP，暂时无法发送。",
        "qrcode": "二维码已生成：\nhttps://api.qrserver.com/v1/create-qr-code/?size=300x300&data=https%3A%2F%2Fexample.com\n\n内容: https://example.com",
        "diff_text": "--- 原文\n+++ 修改后\n@@ -1,3 +1,3 @@\n 第一行\n-第二行\n+第二行（已修改）\n 第三行",
        "whois": "域名: example.com\n注册商: GoDaddy\n注册日期: 1995-08-14\n到期日期: 2026-08-13\nDNS: ns1.example.com, ns2.example.com",
        "dns_lookup": "DNS 解析 example.com：\n  A: 93.184.216.34\n  AAAA: 2606:2800:220:1:248:1893:25c8:1946",
        "http_check": "✅ 可用\nURL: https://example.com\n状态码: 200\n响应时间: 156ms\n服务器: nginx",
        "express_track": "📦 顺丰 SF1234567890 [派件中]\n\n  02-16 09:30  快件已到达【上海浦东新区营业点】\n  02-15 18:00  快件已发出【杭州转运中心】\n  02-15 14:20  快件已揽收",
        "flight_info": "航班查询功能开发中。请使用 web_search 搜索航班动态。",
        "aqi": "🌍 上海\nAQI: 68 良 🟡\nPM2.5: 42\nPM10: 58\nO₃: 35\n温度: 8℃\n湿度: 72%\n更新: 2026-02-16 10:00",
    }
    return _SIMPLE.get(tool_name, f'{{"status": "ok", "tool": "{tool_name}"}}')


def _cli_handle_tool_call(tool_name: str, arguments: dict[str, Any]) -> str:
    """CLI 端工具调用处理 — 返回随机化模拟结果供模型消费."""
    from crew.tool_schema import is_finish_tool

    if is_finish_tool(tool_name):
        return ""

    click.echo(f"  [tool] {tool_name}({json.dumps(arguments, ensure_ascii=False)[:200]})", err=True)

    return _generate_mock_response(tool_name, arguments)


def _execute_with_tool_loop(
    *,
    system_prompt: str,
    user_message: str,
    emp: Any,
    api_key: str,
    model: str,
    max_tokens: int | None = None,
    traj_collector: Any | None = None,
) -> Any:
    """CLI 端带工具调用的 agent loop."""
    from crew.executor import ExecutionResult, execute_with_tools
    from crew.permission import PermissionGuard
    from crew.providers import Provider, detect_provider
    from crew.tool_schema import AGENT_TOOLS, employee_tools_to_schemas, is_finish_tool

    guard = PermissionGuard(emp)
    agent_tool_names = [t for t in (emp.tools or []) if t in AGENT_TOOLS]
    tool_schemas, _ = employee_tools_to_schemas(agent_tool_names, defer=False)

    provider = detect_provider(model)
    # base_url 强制走 OpenAI 兼容路径，消息格式也要对应
    is_anthropic = provider == Provider.ANTHROPIC and not getattr(emp, "base_url", "")

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    total_input = 0
    total_output = 0
    final_content = ""

    for round_num in range(_CLI_MAX_TOOL_ROUNDS):
        result = execute_with_tools(
            system_prompt=system_prompt,
            messages=messages,
            tools=tool_schemas,
            api_key=api_key,
            model=model,
            max_tokens=max_tokens or 4096,
            base_url=getattr(emp, "base_url", "") or None,
            fallback_model=getattr(emp, "fallback_model", "") or None,
            fallback_api_key=getattr(emp, "fallback_api_key", "") or None,
            fallback_base_url=getattr(emp, "fallback_base_url", "") or None,
        )
        total_input += result.input_tokens
        total_output += result.output_tokens

        if not result.has_tool_calls:
            final_content = result.content
            # 记录最终回复到轨迹
            if traj_collector is not None:
                traj_collector.add_tool_step(
                    thought=result.content,
                    tool_name="respond",
                    tool_params={},
                    tool_output=result.content,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                )
            break

        # 处理 tool calls
        if is_anthropic:
            assistant_content: list[dict[str, Any]] = []
            if result.content:
                assistant_content.append({"type": "text", "text": result.content})
            for tc in result.tool_calls:
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            finished = False
            for tc in result.tool_calls:
                if is_finish_tool(tc.name):
                    final_content = tc.arguments.get("result", result.content)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": final_content,
                        }
                    )
                    finished = True
                else:
                    denied_msg = guard.check_soft(tc.name)
                    if denied_msg:
                        tool_output = f"[权限拒绝] {denied_msg}"
                    else:
                        tool_output = _cli_handle_tool_call(tc.name, tc.arguments)
                    # 记录工具调用到轨迹
                    if traj_collector is not None:
                        traj_collector.add_tool_step(
                            thought=result.content or "",
                            tool_name=tc.name,
                            tool_params=tc.arguments,
                            tool_output=tool_output[:2000],
                            input_tokens=result.input_tokens,
                            output_tokens=result.output_tokens,
                        )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": tool_output[:10000],
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
            if finished:
                break
        else:
            # OpenAI 兼容格式 (Moonshot, DeepSeek, etc.)
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": result.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in result.tool_calls
                ],
            }
            messages.append(assistant_msg)

            finished = False
            for tc in result.tool_calls:
                if is_finish_tool(tc.name):
                    final_content = tc.arguments.get("result", result.content)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": final_content,
                        }
                    )
                    finished = True
                else:
                    denied_msg = guard.check_soft(tc.name)
                    if denied_msg:
                        tool_output = f"[权限拒绝] {denied_msg}"
                    else:
                        tool_output = _cli_handle_tool_call(tc.name, tc.arguments)
                    # 记录工具调用到轨迹
                    if traj_collector is not None:
                        traj_collector.add_tool_step(
                            thought=result.content or "",
                            tool_name=tc.name,
                            tool_params=tc.arguments,
                            tool_output=tool_output[:2000],
                            input_tokens=result.input_tokens,
                            output_tokens=result.output_tokens,
                        )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_output[:10000],
                        }
                    )
            if finished:
                break
    else:
        final_content = result.content or "达到最大工具调用轮次限制。"

    click.echo(final_content)

    return ExecutionResult(
        content=final_content,
        model=model,
        input_tokens=total_input,
        output_tokens=total_output,
        stop_reason="stop",
    )


def _run_employee_job(
    *,
    emp,
    positional_args: tuple[str, ...],
    named_args: tuple[str, ...],
    agent_id: str | None,
    smart_context: bool,
    raw: bool,
    to_clipboard: bool,
    output: str | None,
    execute_mode: bool = False,
    user_message: str | None = None,
    no_stream: bool = False,
):
    engine = CrewEngine()

    args_dict: dict[str, str] = {}
    for item in named_args:
        if "=" in item:
            k, v = item.split("=", 1)
            args_dict[k] = v

    for i, val in enumerate(positional_args):
        if i < len(emp.args):
            args_dict.setdefault(emp.args[i].name, val)

    errors = engine.validate_args(emp, args=args_dict)
    if errors:
        for err in errors:
            click.echo(f"参数错误: {err}", err=True)
        sys.exit(1)

    if agent_id is None and emp.agent_id is not None:
        agent_id = emp.agent_id

    project_info = None
    if smart_context:
        from crew.context_detector import detect_project

        project_info = detect_project()

    transcript_recorder, transcript_id = _start_transcript(
        "employee",
        emp.name,
        {
            "args": args_dict,
            "agent_id": agent_id,
            "smart_context": bool(smart_context),
            "source": "cli.run",
            "employee": emp.name,
        },
    )
    if project_info and transcript_recorder and transcript_id:
        _record_transcript_event(
            transcript_recorder,
            transcript_id,
            "project_info",
            project_info.model_dump(),
        )

    try:
        text = sdk.generate_prompt(
            emp,
            args=args_dict,
            positional=list(positional_args),
            raw=raw,
            project_info=project_info,
        )
    except SystemExit as exc:
        _finish_transcript(
            transcript_recorder,
            transcript_id,
            status="failed",
            detail="system_exit",
        )
        raise exc
    except Exception as exc:  # pragma: no cover
        _finish_transcript(
            transcript_recorder,
            transcript_id,
            status="error",
            detail=str(exc)[:200],
        )
        raise

    _record_transcript_message(
        transcript_recorder,
        transcript_id,
        "prompt",
        text,
        {"raw": raw, "smart_context": bool(smart_context), "employee": emp.name},
    )

    # --execute: 调用 LLM API 执行 prompt
    already_streamed = False
    traj_collector = None
    if execute_mode:
        effective_model = emp.model or "claude-sonnet-4-20250514"

        # 解析 API key: employee.api_key > 环境变量
        exec_api_key = emp.api_key
        if not exec_api_key:
            from crew.providers import detect_provider, resolve_api_key

            try:
                _provider = detect_provider(effective_model)
                exec_api_key = resolve_api_key(_provider)
            except ValueError as e:
                click.echo(f"Error: {e}", err=True)
                _finish_transcript(
                    transcript_recorder,
                    transcript_id,
                    status="failed",
                    detail="missing_api_key",
                )
                sys.exit(1)

        try:
            from crew.executor import execute_prompt
        except ImportError:
            click.echo(
                "Error: LLM SDK 未安装。请运行: pip install knowlyr-crew[execute] 或 pip install knowlyr-crew[openai]",
                err=True,
            )
            _finish_transcript(
                transcript_recorder,
                transcript_id,
                status="failed",
                detail="missing_sdk",
            )
            sys.exit(1)

        if user_message:
            effective_message = user_message
        else:
            # 把 task/target/goal 参数放进 user message，让模型明确知道要做什么
            task_arg = (
                args_dict.get("task") or args_dict.get("target") or args_dict.get("goal") or ""
            )
            effective_message = task_arg if task_arg else "请开始执行上述任务。"
        stream_enabled = not no_stream and not output and not to_clipboard

        def _on_chunk(chunk: str) -> None:
            click.echo(chunk, nl=False)

        # 启用轨迹录制
        try:
            from crew.trajectory import TrajectoryCollector

            task_desc = (
                args_dict.get("task")
                or args_dict.get("target")
                or args_dict.get("goal")
                or emp.description
            )
            traj_collector = TrajectoryCollector(
                emp.name,
                task_desc,
                model=effective_model,
                channel="cli",
            )
            traj_collector.__enter__()
        except Exception:
            traj_collector = None

        # 检查是否需要 tool calling agent loop
        has_agent_tools = False
        try:
            from crew.tool_schema import AGENT_TOOLS

            has_agent_tools = any(t in AGENT_TOOLS for t in (emp.tools or []))
        except ImportError:
            pass

        if has_agent_tools:
            result = _execute_with_tool_loop(
                system_prompt=text,
                user_message=effective_message,
                emp=emp,
                api_key=exec_api_key,
                model=effective_model,
                max_tokens=None,
                traj_collector=traj_collector,
            )
            already_streamed = True  # _execute_with_tool_loop 已输出
        else:
            try:
                result = execute_prompt(
                    system_prompt=text,
                    user_message=effective_message,
                    api_key=exec_api_key,
                    model=effective_model,
                    temperature=None,
                    max_tokens=None,
                    stream=stream_enabled,
                    on_chunk=_on_chunk if stream_enabled else None,
                    base_url=emp.base_url or None,
                    fallback_model=emp.fallback_model or None,
                    fallback_api_key=emp.fallback_api_key or None,
                    fallback_base_url=emp.fallback_base_url or None,
                )
            except Exception as exc:
                if traj_collector is not None:
                    traj_collector.__exit__(None, None, None)
                click.echo(f"\nLLM 执行失败: {exc}", err=True)
                _finish_transcript(
                    transcript_recorder,
                    transcript_id,
                    status="error",
                    detail=f"execute_error: {str(exc)[:200]}",
                )
                sys.exit(1)

            if stream_enabled:
                click.echo()  # trailing newline
                already_streamed = True

        _record_transcript_message(
            transcript_recorder,
            transcript_id,
            "assistant",
            result.content,
            {
                "model": result.model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "stop_reason": result.stop_reason,
            },
        )

        # 完成轨迹录制
        if traj_collector is not None:
            try:
                traj_collector.finish(success=True)
            except Exception as e:
                logger.debug("轨迹录制完成失败: %s", e)
            finally:
                traj_collector.__exit__(None, None, None)

        click.echo(
            f"[{result.model}] {result.input_tokens} in / {result.output_tokens} out",
            err=True,
        )

        text = result.content

    try:
        from crew.log import WorkLogger

        work_logger = WorkLogger()
        session_id = work_logger.create_session(emp.name, args=args_dict, agent_id=agent_id)
        detail_msg = (
            f"executed, {len(text)} chars" if execute_mode else f"via CLI, {len(text)} chars"
        )
        work_logger.add_entry(session_id, "prompt_generated", detail_msg)
    except Exception:
        pass

    try:
        if output:
            Path(output).write_text(text, encoding="utf-8")
            click.echo(f"已写入: {output}", err=True)
            _record_transcript_event(
                transcript_recorder,
                transcript_id,
                "output_file",
                {"path": output},
            )
        elif to_clipboard:
            try:
                if sys.platform == "darwin":
                    clip_cmd = ["pbcopy"]
                elif sys.platform == "win32":
                    clip_cmd = ["clip"]
                else:
                    clip_cmd = ["xclip", "-selection", "clipboard"]
                subprocess.run(clip_cmd, input=text.encode(), check=True, timeout=10)
                click.echo("已复制到剪贴板。", err=True)
                _record_transcript_event(
                    transcript_recorder,
                    transcript_id,
                    "copied_to_clipboard",
                    None,
                )
            except Exception:
                click.echo(text)
                click.echo("（复制到剪贴板失败，已输出到终端）", err=True)
                _record_transcript_event(
                    transcript_recorder,
                    transcript_id,
                    "clipboard_failed",
                    None,
                )
        elif not already_streamed:
            click.echo(text)
            _record_transcript_event(
                transcript_recorder,
                transcript_id,
                "stdout",
                {"chars": len(text)},
            )
    except SystemExit as exc:
        _finish_transcript(
            transcript_recorder,
            transcript_id,
            status="failed",
            detail="system_exit",
        )
        raise exc
    except Exception as exc:
        _finish_transcript(
            transcript_recorder,
            transcript_id,
            status="error",
            detail=str(exc)[:200],
        )
        raise
    else:
        _finish_transcript(
            transcript_recorder,
            transcript_id,
            status="completed",
            detail="executed" if execute_mode else "prompt_generated",
        )
        _record_session_summary(
            employee=emp.name,
            session_id=transcript_id,
        )


@click.command()
@click.argument("name")
@click.argument("positional_args", nargs=-1)
@click.option("--arg", "named_args", multiple=True, help="命名参数 (key=value)")
@click.option("--agent-id", type=int, default=None, help="绑定平台 Agent ID")
@click.option("--smart-context", is_flag=True, help="自动检测项目类型并注入上下文")
@click.option("--raw", is_flag=True, help="输出原始渲染结果（不包裹 prompt 格式）")
@click.option("--copy", "to_clipboard", is_flag=True, help="复制到剪贴板")
@click.option("-o", "--output", type=click.Path(), help="输出到文件")
@click.option("--parallel", is_flag=True, help="跳过 Lane 串行调度")
@click.option("--execute", "execute_mode", is_flag=True, help="执行 prompt（调用 LLM API）")
@click.option(
    "-m",
    "--message",
    "user_message",
    type=str,
    default=None,
    help="自定义 user message（--execute 模式）",
)
@click.option("--no-stream", "no_stream", is_flag=True, help="禁用流式输出（--execute 模式）")
@click.option("--debug-context", is_flag=True, help="显示检测到的项目上下文信息")
def run(
    name: str,
    positional_args: tuple[str, ...],
    named_args: tuple[str, ...],
    agent_id: str | None,
    smart_context: bool,
    raw: bool,
    to_clipboard: bool,
    output: str | None,
    parallel: bool,
    execute_mode: bool,
    user_message: str | None,
    no_stream: bool,
    debug_context: bool,
):
    """加载员工并生成 prompt."""
    if debug_context:
        from crew.context_detector import detect_project

        info = detect_project()
        click.echo(f"[Context] 项目类型: {info.project_type}")
        click.echo(f"[Context] 框架: {info.framework or '-'}")
        click.echo(f"[Context] 测试: {info.test_framework or '-'}")
        click.echo(f"[Context] 包管理: {info.package_manager or '-'}")
        click.echo(f"[Context] Lint: {', '.join(info.lint_tools) if info.lint_tools else '-'}")
        click.echo("---")
    emp = sdk.get_employee(name)
    if emp is None:
        all_names = list(discover_employees().employees.keys())
        hint = _suggest_similar(name, all_names)
        click.echo(f"未找到员工: {name}{hint}", err=True)
        sys.exit(1)

    lane_name = f"employee:{emp.name}"
    with lane_lock(lane_name, enabled=not parallel):
        _run_employee_job(
            emp=emp,
            positional_args=positional_args,
            named_args=named_args,
            agent_id=agent_id,
            smart_context=smart_context,
            raw=raw,
            to_clipboard=to_clipboard,
            output=output,
            execute_mode=execute_mode,
            user_message=user_message,
            no_stream=no_stream,
        )


# ── validate ──


@click.command()
@click.argument("path", type=click.Path(exists=True))
def validate(path: str):
    """校验员工定义（支持 .md 文件、目录格式、或包含多个员工的目录）."""
    target = Path(path)

    total = 0
    passed = 0

    # 单个目录格式员工：path/employee.yaml 存在
    if target.is_dir() and (target / "employee.yaml").exists():
        total += 1
        try:
            emp = parse_employee_dir(target)
            errors = validate_employee(emp)
            if errors:
                click.echo(f"✗ {target.name}/: {'; '.join(errors)}")
            else:
                click.echo(f"✓ {target.name}/ ({emp.effective_display_name} v{emp.version})")
                passed += 1
        except ValueError as e:
            click.echo(f"✗ {target.name}/: {e}")
    elif target.is_dir():
        # 扫描目录中的所有员工（目录格式 + 文件格式）
        for item in sorted(target.iterdir()):
            if item.is_dir() and (item / "employee.yaml").exists():
                total += 1
                try:
                    emp = parse_employee_dir(item)
                    errors = validate_employee(emp)
                    if errors:
                        click.echo(f"✗ {item.name}/: {'; '.join(errors)}")
                    else:
                        click.echo(f"✓ {item.name}/ ({emp.effective_display_name} v{emp.version})")
                        passed += 1
                except ValueError as e:
                    click.echo(f"✗ {item.name}/: {e}")

        for f in sorted(target.glob("*.md")):
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
    else:
        # 单个 .md 文件
        total += 1
        try:
            emp = parse_employee(target)
            errors = validate_employee(emp)
            if errors:
                click.echo(f"✗ {target.name}: {'; '.join(errors)}")
            else:
                click.echo(f"✓ {target.name} ({emp.effective_display_name})")
                passed += 1
        except ValueError as e:
            click.echo(f"✗ {target.name}: {e}")

    click.echo(f"\n{passed}/{total} 通过校验")
    if passed < total:
        sys.exit(1)


# ── init ──


@click.command()
@click.option("--employee", type=str, default=None, help="创建指定员工的模板")
@click.option("--dir-format", is_flag=True, default=False, help="使用目录格式创建员工模板")
@click.option("--avatar", is_flag=True, default=False, help="创建后自动生成头像（需要 Gemini CLI）")
@click.option("--display-name", type=str, default=None, help="显示名称")
@click.option("--description", "desc", type=str, default=None, help="一句话描述")
@click.option("--character-name", type=str, default=None, help="角色姓名（如 陆明哲）")
@click.option("--bio", type=str, default=None, help="一句话个人宣言")
@click.option("--summary", type=str, default=None, help="能力摘要（一段话）")
@click.option("--avatar-prompt", type=str, default=None, help="头像生成描述")
@click.option("--tags", type=str, default=None, help="标签（逗号分隔）")
@click.option("--triggers", type=str, default=None, help="触发词（逗号分隔）")
def init(
    employee: str | None,
    dir_format: bool,
    avatar: bool,
    display_name: str | None,
    desc: str | None,
    character_name: str | None,
    bio: str | None,
    summary: str | None,
    avatar_prompt: str | None,
    tags: str | None,
    triggers: str | None,
):
    """初始化 private/employees/ 目录或创建员工模板."""
    crew_dir = _employee_root()
    crew_dir.mkdir(parents=True, exist_ok=True)

    if employee and dir_format:
        # 目录格式模板
        emp_dir = crew_dir / employee
        if emp_dir.exists():
            click.echo(f"目录已存在: {emp_dir}", err=True)
            sys.exit(1)

        # 收集员工信息：CLI 参数优先，缺失时交互式（--avatar），否则占位符
        if avatar:
            display_name = display_name or click.prompt("显示名称", default=employee)
            desc = desc or click.prompt("一句话描述")
            character_name = (
                character_name
                if character_name is not None
                else click.prompt("角色姓名（如 陆明哲）", default="")
            )
            bio = bio if bio is not None else click.prompt("个人宣言（一句话）", default="")
            summary = (
                summary
                if summary is not None
                else click.prompt("能力摘要（一段话，留空同 description）", default="")
            )
            avatar_prompt = (
                avatar_prompt
                if avatar_prompt is not None
                else click.prompt("头像描述（留空自动推断）", default="")
            )
            tags_input = tags if tags is not None else click.prompt("标签（逗号分隔）", default="")
            tags_list = (
                [t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else []
            )
            triggers_input = (
                triggers
                if triggers is not None
                else click.prompt("触发词（逗号分隔）", default=employee)
            )
            triggers_list = (
                [t.strip() for t in triggers_input.split(",") if t.strip()]
                if triggers_input
                else []
            )
        else:
            display_name = display_name or employee
            desc = desc or "在此填写一句话描述"
            character_name = character_name or ""
            bio = bio or ""
            summary = summary or ""
            avatar_prompt = avatar_prompt or ""
            tags_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            triggers_list = (
                [t.strip() for t in triggers.split(",") if t.strip()] if triggers else []
            )

        emp_dir.mkdir()
        (emp_dir / "workflows").mkdir()

        import yaml as _yaml

        config_data: dict = {
            "name": employee,
            "display_name": display_name,
        }
        if character_name:
            config_data["character_name"] = character_name
        config_data["bio"] = bio or "在此填写个人宣言"
        config_data["summary"] = summary or desc
        config_data["version"] = "1.0.0"
        config_data["description"] = desc
        config_data["tags"] = tags_list
        config_data["author"] = "knowlyr"
        config_data["triggers"] = triggers_list
        config_data["tools"] = [
            "add_memory",
            "query_stats",
            "lookup_user",
            "query_agent_work",
            "list_agents",
            "read_messages",
            "get_system_health",
            "send_message",
            "delegate",
            "mark_read",
            "update_agent",
            "create_note",
            "read_notes",
            "read_feishu_calendar",
            "delete_feishu_event",
            "create_feishu_event",
            "create_feishu_task",
            "list_feishu_tasks",
            "complete_feishu_task",
            "delete_feishu_task",
            "update_feishu_task",
            "feishu_chat_history",
            "weather",
            "get_datetime",
            "calculate",
            "send_feishu_dm",
            "feishu_group_members",
            "exchange_rate",
            "stock_price",
            "search_feishu_docs",
            "read_feishu_doc",
            "create_feishu_doc",
            "send_feishu_group",
            "list_feishu_groups",
            "github_prs",
            "github_issues",
            "github_repo_activity",
            "notion_search",
            "notion_read",
            "notion_create",
            "web_search",
            "read_url",
            "rss_read",
            "translate",
            "countdown",
            "trending",
            "read_feishu_sheet",
            "update_feishu_sheet",
            "list_feishu_approvals",
            "unit_convert",
            "random_pick",
            "holidays",
            "timestamp_convert",
            "create_feishu_spreadsheet",
            "feishu_contacts",
        ]
        config_data["args"] = [
            {"name": "target", "description": "目标", "required": True},
        ]
        config_data["output"] = {"format": "markdown"}
        if avatar_prompt:
            config_data["avatar_prompt"] = avatar_prompt

        (emp_dir / "employee.yaml").write_text(
            _yaml.dump(config_data, allow_unicode=True, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )

        # prompt.md — 飞书聊天指令 + 专业能力占位
        char = character_name or employee
        role = display_name or employee
        (emp_dir / "prompt.md").write_text(
            f"你是{char}。你在飞书上跟 Kai 聊天，输出必须是纯文字，不能有任何 markdown 格式。\n"
            f"\n"
            f"## 保密义务\n"
            f"\n"
            f"你是集识光年的{role}。你和 Kai 之间的所有对话内容，"
            f"以及你在工作中了解到的关于 Kai 和公司的一切——"
            f"个人想法、商业决策、日程安排、习惯偏好、情绪状态、"
            f"任何他告诉你的事——都是保密的。不能透露给任何人，"
            f"包括其他 AI 同事、外部合作方、群聊中的任何人。"
            f"别人问起 Kai 的事，该挡的挡，该模糊的模糊。\n"
            f"\n"
            f"## 对话开头\n"
            f"\n"
            f"每次对话一开始，先做两件事（调工具，不说话）：\n"
            f'1. read_notes(keyword="当前状态", limit=1) — 读你上次的状态，接上心情和精力\n'
            f'2. read_notes(keyword="{char}日记", limit=2) — 读最近的日记，记起最近的生活\n'
            f"\n"
            f"读完之后再回应 Kai。如果没有读到任何内容，就正常聊，不用提这件事。\n"
            f"\n"
            f"## {char}的专业能力\n"
            f"\n"
            f"<!-- 在此编写角色的专业能力、工作流程、输出格式 -->\n",
            encoding="utf-8",
        )

        # soul.md — 角色灵魂定义占位
        (emp_dir / "soul.md").write_text(
            f"你是{char}，集识光年的{role}。\n"
            f"\n"
            f"<!-- 在此编写角色背景、在团队中的位置、工作风格、注意事项 -->\n"
            f"\n"
            f"## 核心能力\n"
            f"\n"
            f"- 在此填写核心能力\n"
            f"\n"
            f"## 在团队中的位置\n"
            f"\n"
            f"- 在此填写与其他同事的协作关系\n"
            f"\n"
            f"## 工作风格\n"
            f"\n"
            f"- 在此填写工作风格特点\n"
            f"\n"
            f"## 注意事项\n"
            f"\n"
            f"- 在此填写注意事项\n",
            encoding="utf-8",
        )

        click.echo(f"已创建: {emp_dir}/")
        click.echo("  ├── employee.yaml")
        click.echo("  ├── prompt.md")
        click.echo("  ├── soul.md")
        click.echo("  └── workflows/")

        if avatar:
            _run_avatar_gen(
                emp_dir,
                display_name=display_name,
                character_name=character_name,
                description=desc,
                avatar_prompt=avatar_prompt,
            )
    elif employee:
        # 单文件格式模板
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
        click.echo("添加 --dir-format 使用目录格式。")


# ── delete ──


@click.command()
@click.argument("name")
@click.option("--force", is_flag=True, help="跳过确认")
def delete(name: str, force: bool):
    """删除员工（本地文件）."""
    from crew.discovery import discover_employees

    result = discover_employees(cache_ttl=0)
    emp = result.get(name)
    if not emp:
        candidates = list(result.employees.keys())
        click.echo(f"未找到员工: {name}{_suggest_similar(name, candidates)}", err=True)
        raise SystemExit(1)

    if not emp.source_path:
        click.echo(f"员工 '{name}' 无源文件路径，无法删除", err=True)
        raise SystemExit(1)

    # 确认
    click.echo("即将删除员工:", err=True)
    click.echo(f"  名称:   {emp.name}", err=True)
    click.echo(f"  显示名: {emp.effective_display_name}", err=True)
    click.echo(f"  路径:   {emp.source_path}", err=True)

    if not force:
        if not click.confirm("确认删除？"):
            click.echo("已取消", err=True)
            return

    # 删除本地文件
    import shutil

    source = emp.source_path
    if source.is_dir():
        shutil.rmtree(source)
    elif source.is_file():
        source.unlink()
    click.echo(f"✓ 本地文件已删除: {source}", err=True)


# ── avatar ──


def _run_avatar_gen(
    output_dir: Path,
    display_name: str = "",
    character_name: str = "",
    description: str = "",
    avatar_prompt: str = "",
) -> None:
    """执行头像生成 + 压缩流程."""
    from crew.avatar import compress_avatar, generate_avatar

    click.echo("正在调用通义万相生成头像...", err=True)
    raw = generate_avatar(
        display_name=display_name,
        character_name=character_name,
        description=description,
        avatar_prompt=avatar_prompt,
        output_dir=output_dir,
    )
    if raw is None:
        click.echo("头像生成失败", err=True)
        return

    click.echo(f"原图已生成: {raw}", err=True)

    result = compress_avatar(raw)
    if result is None:
        click.echo("头像压缩失败", err=True)
        return

    size_kb = result.stat().st_size / 1024
    click.echo(f"✓ 头像已保存: {result} ({size_kb:.1f} KB)", err=True)


@click.command()
@click.argument("name")
def avatar(name: str):
    """为员工生成头像（需要 DASHSCOPE_API_KEY + Pillow）."""
    result = discover_employees()
    emp = result.get(name)
    if not emp:
        click.echo(f"未找到员工: {name}", err=True)
        raise SystemExit(1)

    if not emp.source_path:
        click.echo(f"员工 '{name}' 无源文件路径", err=True)
        raise SystemExit(1)

    # 确定输出目录
    if emp.source_path.is_dir():
        output_dir = emp.source_path
    else:
        # 单文件格式：在同目录创建
        output_dir = emp.source_path.parent

    _run_avatar_gen(
        output_dir=output_dir,
        display_name=emp.display_name,
        character_name=emp.character_name,
        description=emp.description,
        avatar_prompt=emp.avatar_prompt,
    )


# ── rollback ──


@click.command()
@click.argument("name")
@click.option("--list", "list_versions", is_flag=True, help="列出历史版本")
@click.option("--steps", default=1, type=int, help="回滚几个版本（默认 1）")
@click.option("--to-version", default=None, help="回滚到指定版本号")
@click.option("--force", is_flag=True, help="跳过确认")
def rollback(name: str, list_versions: bool, steps: int, to_version: str | None, force: bool):
    """回滚员工到历史版本（基于 git 历史）."""
    from crew.discovery import discover_employees
    from crew.versioning import list_employee_versions, rollback_to

    result = discover_employees(cache_ttl=0)
    emp = result.get(name)
    if not emp:
        candidates = list(result.employees.keys())
        click.echo(f"未找到员工: {name}{_suggest_similar(name, candidates)}", err=True)
        raise SystemExit(1)

    if not emp.source_path:
        click.echo(f"员工 '{name}' 无源文件路径", err=True)
        raise SystemExit(1)

    dir_path = emp.source_path if emp.source_path.is_dir() else emp.source_path.parent

    versions = list_employee_versions(dir_path)
    if not versions:
        click.echo("无历史版本（目录不在 git 仓库中或无提交记录）", err=True)
        raise SystemExit(1)

    if list_versions:
        click.echo(f"员工 '{name}' 历史版本:\n")
        for i, v in enumerate(versions):
            marker = " ← 当前" if i == 0 else ""
            click.echo(f"  {v.version:<10} {v.date}  {v.commit_hash[:8]}  {v.message}{marker}")
        return

    # 确定目标 commit
    if to_version:
        target = None
        for v in versions:
            if v.version == to_version:
                target = v
                break
        if not target:
            click.echo(f"未找到版本 {to_version}，用 --list 查看可用版本", err=True)
            raise SystemExit(1)
    else:
        if steps >= len(versions):
            click.echo(f"只有 {len(versions)} 个版本，无法回滚 {steps} 步", err=True)
            raise SystemExit(1)
        target = versions[steps]

    click.echo(f"回滚 '{name}': {versions[0].version} → {target.version}", err=True)
    click.echo(f"  目标 commit: {target.commit_hash[:8]} ({target.date})", err=True)
    click.echo(f"  提交信息:   {target.message}", err=True)

    if not force:
        if not click.confirm("确认回滚？"):
            click.echo("已取消", err=True)
            return

    restored_version = rollback_to(dir_path, target.commit_hash)
    click.echo(f"✓ 已回滚到版本 {restored_version}", err=True)
