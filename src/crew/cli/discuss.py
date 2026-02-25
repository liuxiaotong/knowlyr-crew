"""讨论会命令 — discuss group + meetings group + changelog + ingest."""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

from crew.cli import (
    _finish_transcript,
    _record_session_summary,
    _record_transcript_event,
    _record_transcript_message,
    _start_transcript,
)
from crew.lanes import lane_lock


@click.group()
def discuss():
    """讨论会管理 — 多员工多轮讨论."""
    pass


@discuss.command("list")
def discuss_list():
    """列出所有可用讨论会."""
    from crew.discussion import discover_discussions, load_discussion

    discussions = discover_discussions()
    if not discussions:
        click.echo("未找到讨论会。")
        click.echo("在 .crew/discussions/ 中创建 YAML 文件，或使用内置讨论会。")
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="讨论会")
        table.add_column("名称", style="cyan")
        table.add_column("描述", style="green")
        table.add_column("参与者", justify="right")
        table.add_column("轮次", justify="right")
        table.add_column("来源")

        for name, path in discussions.items():
            try:
                d = load_discussion(path)
                source = "内置" if "employees/discussions" in str(path) else "项目"
                rounds_count = d.rounds if isinstance(d.rounds, int) else len(d.rounds)
                table.add_row(
                    name,
                    d.description,
                    str(len(d.participants)),
                    str(rounds_count),
                    source,
                )
            except Exception:
                table.add_row(name, "[解析失败]", "-", "-", str(path))

        console.print(table)
    except ImportError:
        for name, path in discussions.items():
            click.echo(f"  {name} — {path}")


@discuss.command("show")
@click.argument("name")
def discuss_show(name: str):
    """查看讨论会详情."""
    from crew.discussion import discover_discussions, load_discussion

    discussions = discover_discussions()
    if name not in discussions:
        click.echo(f"未找到讨论会: {name}", err=True)
        sys.exit(1)

    d = load_discussion(discussions[name])
    click.echo(f"讨论会: {d.name}")
    click.echo(f"描述: {d.description}")
    click.echo(f"议题: {d.topic}")
    if d.goal:
        click.echo(f"目标: {d.goal}")
    rounds_count = d.rounds if isinstance(d.rounds, int) else len(d.rounds)
    click.echo(f"轮次: {rounds_count}")
    click.echo(f"输出格式: {d.output_format}")
    click.echo()

    role_labels = {"moderator": "主持人", "speaker": "发言人", "recorder": "记录员"}
    for i, p in enumerate(d.participants, 1):
        focus_str = f" — {p.focus}" if p.focus else ""
        click.echo(f"  {i}. {p.employee} ({role_labels[p.role]}){focus_str}")


@discuss.command("run")
@click.argument("name_or_path")
@click.option("--arg", "named_args", multiple=True, help="参数 (key=value)")
@click.option("--agent-id", type=int, default=None, help="绑定平台 Agent ID")
@click.option("--smart-context/--no-smart-context", default=True, help="自动检测项目类型")
@click.option(
    "--orchestrated",
    is_flag=True,
    default=False,
    help="编排模式：生成独立 prompt 计划（每个参会者独立推理）",
)
@click.option("-o", "--output", type=click.Path(), help="输出到文件")
@click.option("--parallel", is_flag=True, help="跳过 Lane 串行调度")
def discuss_run(
    name_or_path: str,
    named_args: tuple[str, ...],
    agent_id: str | None,
    smart_context: bool,
    orchestrated: bool,
    output: str | None,
    parallel: bool,
):
    """生成讨论会 prompt.

    NAME_OR_PATH 可以是讨论会名称或 YAML 文件路径。
    """
    from crew.discussion import (
        discover_discussions,
        load_discussion,
        validate_discussion,
    )

    # 解析讨论会
    path = Path(name_or_path)
    if path.exists() and path.suffix in (".yaml", ".yml"):
        d = load_discussion(path)
    else:
        discussions = discover_discussions()
        if name_or_path not in discussions:
            click.echo(f"未找到讨论会: {name_or_path}", err=True)
            sys.exit(1)
        d = load_discussion(discussions[name_or_path])

    # 校验
    errors = validate_discussion(d)
    if errors:
        for err in errors:
            click.echo(f"校验错误: {err}", err=True)
        sys.exit(1)

    # 解析参数
    initial_args: dict[str, str] = {}
    for item in named_args:
        if "=" in item:
            k, v = item.split("=", 1)
            initial_args[k] = v

    rounds_count = d.rounds if isinstance(d.rounds, int) else len(d.rounds)

    transcript_recorder, transcript_id = _start_transcript(
        "discussion",
        d.name,
        {
            "participants": len(d.participants),
            "rounds": rounds_count,
            "initial_args": initial_args,
            "agent_id": agent_id,
            "smart_context": bool(smart_context),
            "mode": "plan" if orchestrated else "prompt",
            "source": "cli.discuss.run",
        },
    )
    memory_key = f"discussion:{d.name}"

    with lane_lock(f"discussion:{d.name}", enabled=not parallel):
        if orchestrated:
            click.echo(
                f"生成编排式讨论: {d.name} ({len(d.participants)} 人, {rounds_count} 轮)",
                err=True,
            )
            _run_discussion_plan(
                discussion=d,
                initial_args=initial_args,
                agent_id=agent_id,
                smart_context=smart_context,
                output=output,
                transcript_recorder=transcript_recorder,
                transcript_id=transcript_id,
                memory_key=memory_key,
            )
        else:
            click.echo(
                f"生成讨论会: {d.name} ({len(d.participants)} 人, {rounds_count} 轮)",
                err=True,
            )
            _run_discussion_prompt(
                discussion=d,
                initial_args=initial_args,
                agent_id=agent_id,
                smart_context=smart_context,
                output=output,
                transcript_recorder=transcript_recorder,
                transcript_id=transcript_id,
                memory_key=memory_key,
            )


def _run_discussion_plan(
    *,
    discussion,
    initial_args: dict[str, str],
    agent_id: str | None,
    smart_context: bool,
    output: str | None,
    transcript_recorder,
    transcript_id,
    memory_key: str | None = None,
):
    from crew.discussion import render_discussion_plan

    try:
        plan = render_discussion_plan(
            discussion,
            initial_args=initial_args,
            agent_id=agent_id,
            smart_context=smart_context,
        )
        result_text = plan.model_dump_json(indent=2)
        _record_transcript_message(
            transcript_recorder,
            transcript_id,
            "discussion_plan",
            result_text,
            None,
        )
        if output:
            Path(output).write_text(result_text, encoding="utf-8")
            click.echo(f"\n已写入: {output}", err=True)
            _record_transcript_event(
                transcript_recorder,
                transcript_id,
                "output_file",
                {"path": output},
            )
        else:
            click.echo(result_text)
            _record_transcript_event(
                transcript_recorder,
                transcript_id,
                "stdout",
                {"chars": len(result_text)},
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
            detail="discussion_plan",
        )
        if memory_key:
            _record_session_summary(
                employee=memory_key,
                session_id=transcript_id,
            )


def _run_discussion_prompt(
    *,
    discussion,
    initial_args: dict[str, str],
    agent_id: str | None,
    smart_context: bool,
    output: str | None,
    transcript_recorder,
    transcript_id,
    memory_key: str | None = None,
):
    from crew.discussion import render_discussion

    try:
        prompt = render_discussion(
            discussion,
            initial_args=initial_args,
            agent_id=agent_id,
            smart_context=smart_context,
        )
        _record_transcript_message(
            transcript_recorder,
            transcript_id,
            "prompt",
            prompt,
            None,
        )
        if output:
            Path(output).write_text(prompt, encoding="utf-8")
            click.echo(f"\n已写入: {output}", err=True)
            _record_transcript_event(
                transcript_recorder,
                transcript_id,
                "output_file",
                {"path": output},
            )
        else:
            click.echo(prompt)
            _record_transcript_event(
                transcript_recorder,
                transcript_id,
                "stdout",
                {"chars": len(prompt)},
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
            detail="discussion_prompt",
        )
        if memory_key:
            _record_session_summary(
                employee=memory_key,
                session_id=transcript_id,
            )


@discuss.command("adhoc")
@click.option("-e", "--employees", required=True, help="员工名称（逗号分隔）")
@click.option("-t", "--topic", required=True, help="议题")
@click.option("-g", "--goal", default="", help="目标")
@click.option("-r", "--rounds", type=int, default=2, help="轮次数（默认 2）")
@click.option(
    "--round-template",
    type=str,
    default=None,
    help="轮次模板 (standard, brainstorm-to-decision, adversarial)",
)
@click.option(
    "--output-format",
    type=click.Choice(["decision", "transcript", "summary"]),
    default="summary",
    help="输出格式",
)
@click.option("--agent-id", type=int, default=None, help="绑定平台 Agent ID")
@click.option("--smart-context/--no-smart-context", default=True, help="自动检测项目类型")
@click.option(
    "--orchestrated",
    is_flag=True,
    default=False,
    help="编排模式：生成独立 prompt 计划（每个参会者独立推理）",
)
@click.option("-o", "--output", type=click.Path(), help="输出到文件")
@click.option("--parallel", is_flag=True, help="跳过 Lane 串行调度")
def discuss_adhoc(
    employees: str,
    topic: str,
    goal: str,
    rounds: int,
    round_template: str | None,
    output_format: str,
    agent_id: str | None,
    smart_context: bool,
    orchestrated: bool,
    output: str | None,
    parallel: bool,
):
    """发起即席讨论（无需 YAML 定义）.

    示例:
        crew discuss adhoc -e "code-reviewer,test-engineer" -t "auth 模块代码质量"
        crew discuss adhoc -e "hr-manager" -t "招聘方案"
    """
    from crew.discussion import (
        create_adhoc_discussion,
        validate_discussion,
    )

    emp_list = [e.strip() for e in employees.split(",") if e.strip()]
    if not emp_list:
        click.echo("错误: 至少指定 1 个员工", err=True)
        sys.exit(1)

    d = create_adhoc_discussion(
        employees=emp_list,
        topic=topic,
        goal=goal,
        rounds=rounds,
        output_format=output_format,
        round_template=round_template,
    )

    errors = validate_discussion(d)
    if errors:
        for err in errors:
            click.echo(f"校验错误: {err}", err=True)
        sys.exit(1)

    mode_label = "1v1 会议" if d.effective_mode == "meeting" else "讨论会"
    rounds_count = d.rounds if isinstance(d.rounds, int) else len(d.rounds)

    transcript_recorder, transcript_id = _start_transcript(
        "discussion",
        f"adhoc:{topic}",
        {
            "participants": len(emp_list),
            "rounds": rounds_count,
            "agent_id": agent_id,
            "smart_context": bool(smart_context),
            "mode": "plan" if orchestrated else "prompt",
            "source": "cli.discuss.adhoc",
        },
    )
    memory_key = f"discussion:adhoc:{topic}"

    with lane_lock(f"discussion:adhoc:{topic}", enabled=not parallel):
        if orchestrated:
            click.echo(
                f"生成编排式{mode_label}: {len(emp_list)} 人, {rounds_count} 轮",
                err=True,
            )
            _run_discussion_plan(
                discussion=d,
                initial_args={},
                agent_id=agent_id,
                smart_context=smart_context,
                output=output,
                transcript_recorder=transcript_recorder,
                transcript_id=transcript_id,
                memory_key=memory_key,
            )
        else:
            click.echo(
                f"生成即席{mode_label}: {len(emp_list)} 人, {rounds_count} 轮",
                err=True,
            )
            _run_discussion_prompt(
                discussion=d,
                initial_args={},
                agent_id=agent_id,
                smart_context=smart_context,
                output=output,
                transcript_recorder=transcript_recorder,
                transcript_id=transcript_id,
                memory_key=memory_key,
            )


@discuss.command("history")
@click.option("-n", "--limit", type=int, default=20, help="显示条数")
@click.option("--keyword", type=str, default=None, help="按关键词搜索")
def discuss_history(limit: int, keyword: str | None):
    """查看历史会议记录."""
    from crew.meeting_log import MeetingLogger

    logger = MeetingLogger()
    records = logger.list(limit=limit, keyword=keyword)

    if not records:
        click.echo("暂无会议记录。")
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="会议历史")
        table.add_column("ID", style="cyan")
        table.add_column("名称", style="green")
        table.add_column("议题")
        table.add_column("参与者", justify="right")
        table.add_column("模式")
        table.add_column("时间")

        for r in records:
            table.add_row(
                r.meeting_id,
                r.name,
                r.topic[:40],
                str(len(r.participants)),
                r.mode,
                r.started_at[:16],
            )
        console.print(table)
    except ImportError:
        for r in records:
            click.echo(f"  {r.meeting_id}  {r.name}  {r.topic[:40]}  ({r.mode})")


@discuss.command("view")
@click.argument("meeting_id")
def discuss_view(meeting_id: str):
    """查看某次会议的完整记录."""
    from crew.meeting_log import MeetingLogger

    logger = MeetingLogger()
    result = logger.get(meeting_id)

    if result is None:
        click.echo(f"未找到会议: {meeting_id}", err=True)
        sys.exit(1)

    record, content = result
    click.echo(f"会议 ID: {record.meeting_id}")
    click.echo(f"名称: {record.name}")
    click.echo(f"议题: {record.topic}")
    click.echo(f"参与者: {', '.join(record.participants)}")
    click.echo(f"模式: {record.mode}")
    click.echo(f"时间: {record.started_at}")
    click.echo()
    click.echo(content)


@discuss.command("ingest")
@click.argument("json_file", required=False, type=click.Path(exists=True))
@click.option("--stdin", "use_stdin", is_flag=True, help="从 stdin 读取 JSON")
def discuss_ingest(json_file: str | None, use_stdin: bool):
    """导入外部讨论（如 Claude Code 会话）到员工记忆.

    接受 JSON 格式的讨论数据，写入每位参与者的记忆并保存会议记录。
    支持从文件或 stdin 读取。
    """
    import json as _json

    from crew.discussion_ingest import DiscussionIngestor, DiscussionInput

    if use_stdin:
        raw = sys.stdin.read()
    elif json_file:
        raw = Path(json_file).read_text(encoding="utf-8")
    else:
        click.echo("请指定 JSON 文件路径或使用 --stdin 从标准输入读取。", err=True)
        sys.exit(1)

    try:
        data = DiscussionInput(**_json.loads(raw))
    except Exception as e:
        click.echo(f"JSON 解析失败: {e}", err=True)
        sys.exit(1)

    ingestor = DiscussionIngestor()
    results = ingestor.ingest(data)

    click.echo(f"讨论已保存: {data.topic}")
    click.echo(f"  会议 ID: {results.get('meeting_id', 'N/A')}")
    click.echo(f"  本地记忆: {results['memories_written']} 条")
    if results.get("synced_to_crew"):
        click.echo("  线上同步: 已同步 (crew.knowlyr.com)")
    elif results["memories_written"] > 0:
        click.echo("  线上同步: 未同步 (检查 CREW_REMOTE_URL / CREW_API_TOKEN)")
    for p in results["participants"]:
        click.echo(f"    - {p['name']} ({p['slug']})")


# ── meetings 子命令组 ──


@click.group()
def meetings():
    """会议记录与文稿导出."""
    pass


@meetings.command("list")
@click.option("-n", "--limit", type=int, default=20, help="显示条数")
def meetings_list(limit: int):
    """列出会议历史（MeetingLogger)."""
    from crew.meeting_log import MeetingLogger

    logger = MeetingLogger()
    records = logger.list(limit=limit)
    if not records:
        click.echo("暂无会议记录。")
        return

    for r in records:
        click.echo(f"{r.meeting_id}  {r.name}  {r.topic[:40]}  ({r.mode})")


@meetings.command("export")
@click.option("--meeting-id", required=True, help="要导出的会议 ID")
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="输出文件（默认 .crew/meetings/<id>-summary.md）",
)
@click.option("--with-meta", is_flag=True, help="在导出的 Markdown 中包含元信息")
def meetings_export(meeting_id: str, output: Path | None, with_meta: bool):
    """将会议记录导出为 Markdown 文件."""
    from crew.meeting_log import MeetingLogger

    logger = MeetingLogger()
    result = logger.get(meeting_id)
    if result is None:
        click.echo(f"未找到会议: {meeting_id}", err=True)
        sys.exit(1)

    record, content = result
    target = output or (Path.cwd() / ".crew" / "meetings" / f"{meeting_id}-summary.md")
    target.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    if with_meta:
        lines.extend(
            [
                f"# 会议：{record.name}",
                "",
                f"- ID: {record.meeting_id}",
                f"- 议题: {record.topic}",
                f"- 参与者: {', '.join(record.participants)}",
                f"- 模式: {record.mode}",
                f"- 时间: {record.started_at}",
                "",
                "---",
                "",
            ]
        )
    lines.append(content)
    target.write_text("\n".join(lines), encoding="utf-8")
    click.echo(f"已导出: {target}")


# ── changelog ──


@click.command("changelog")
@click.option("--since", type=str, default=None, help="git log since (如 v0.1.0)")
@click.option("-n", "--limit", type=int, default=10, help="返回最近 N 条")
@click.option("-o", "--output", type=click.Path(path_type=Path), default="CHANGELOG_DRAFT.md")
def changelog_draft(since: str | None, limit: int, output: Path):
    """根据 git log 生成 changelog 草稿."""
    cmd = ["git", "log", "-n", str(limit), "--pretty=format:%h %s"]
    if since:
        cmd = ["git", "log", f"{since}..HEAD", "--pretty=format:%h %s"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception as exc:
        lines = [f"(git log 失败: {exc})"]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    content = [f"## {now} Changelog", ""]
    for line in lines:
        content.append(f"- {line}")
    output.write_text("\n".join(content) + "\n", encoding="utf-8")
    click.echo(f"已生成: {output}")
