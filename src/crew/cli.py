"""Crew CLI — 命令行界面."""

import json
import jsonschema
import logging
import sys
from pathlib import Path
from typing import Any

import click

from crew import __version__
from crew import sdk
from crew.discovery import discover_employees
from crew.engine import CrewEngine
from crew.log import WorkLogger
from crew.lanes import LaneLock, lane_lock
from crew.session_recorder import SessionRecorder
from datetime import datetime, timezone
import subprocess
from crew.parser import parse_employee, parse_employee_dir, validate_employee
from crew.template_manager import apply_template, discover_templates
from crew.pipeline import load_pipeline, validate_pipeline
from crew.discussion import load_discussion, validate_discussion
from crew.session_summary import SessionMemoryWriter

EMPLOYEE_SUBDIR = Path("private") / "employees"


def _employee_root() -> Path:
    """返回当前项目的员工根目录."""
    return Path.cwd() / EMPLOYEE_SUBDIR


def _setup_logging(verbose: bool) -> None:
    """配置日志级别."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _start_transcript(
    session_type: str,
    subject: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[SessionRecorder | None, str | None]:
    """尝试创建会话记录，失败时返回 (None, None)。"""
    try:
        recorder = SessionRecorder()
        session_id = recorder.start(session_type, subject, metadata or {})
        return recorder, session_id
    except Exception:
        return None, None


def _record_transcript_event(
    recorder: SessionRecorder | None,
    session_id: str | None,
    event: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    if recorder is None or session_id is None:
        return
    try:
        recorder.record_event(session_id, event, metadata)
    except Exception:
        pass


def _record_transcript_message(
    recorder: SessionRecorder | None,
    session_id: str | None,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    if recorder is None or session_id is None:
        return
    try:
        recorder.record_message(session_id, role, content, metadata)
    except Exception:
        pass


def _finish_transcript(
    recorder: SessionRecorder | None,
    session_id: str | None,
    *,
    status: str,
    detail: str,
) -> None:
    if recorder is None or session_id is None:
        return
    try:
        recorder.finish(session_id, status=status, detail=detail)
    except Exception:
        pass


def _record_session_summary(
    *,
    employee: str,
    session_id: str | None,
    agent_id: int | None = None,
) -> None:
    try:
        summary = SessionMemoryWriter().capture(
            employee=employee,
            session_id=session_id,
        )
    except Exception:
        summary = None

    if summary and agent_id:
        try:
            from crew.id_client import append_agent_memory

            append_agent_memory(agent_id, summary)
        except Exception:
            pass


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
    "--layer", type=click.Choice(["builtin", "skill", "private"]),
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


def _lint_file(path: Path, project_dir: Path) -> list[str]:
    errors: list[str] = []
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - YAML parse error surfaced below
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
        except Exception as exc:
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
        except Exception as exc:
            return [f"{path}: 解析失败 ({exc})"]

        errors.extend(f"{path}: {e}" for e in validate_discussion(discussion, project_dir=project_dir))
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


@main.command("lint")
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


@main.command("check")
@click.option("--no-lint", is_flag=True, default=False, help="跳过 lint 检查")
@click.option("--no-logs", is_flag=True, default=False, help="跳过日志质量检查")
@click.option("--json", "json_output", is_flag=True, help="JSON 输出")
@click.option("--path", "lint_paths", multiple=True, type=click.Path(path_type=Path), help="要 lint 的路径（可多次指定）")
@click.option("--output-file", type=click.Path(path_type=Path), default=None, help="将 JSON 报告写入文件（默认 .crew/quality-report.json）")
@click.option("--no-file", is_flag=True, help="不写入 JSON 报告")
def check_cmd(no_lint: bool, no_logs: bool, json_output: bool, lint_paths: tuple[Path, ...], output_file: Path | None, no_file: bool):
    """执行 lint + 日志质量检查。"""
    report: dict[str, Any] = {}
    exit_code = 0

    if not no_lint:
        targets = list(lint_paths)
        if not targets:
            default_dirs = [Path.cwd() / ".crew" / "pipelines", Path.cwd() / ".crew" / "discussions"]
            targets = [d for d in default_dirs if d.exists()]
        lint_errors = _lint_targets(targets)
        if lint_errors:
            exit_code = 1
            report["lint"] = {"status": "failed", "errors": lint_errors}
        else:
            report["lint"] = {"status": "ok"}

    if not no_logs:
        logger = WorkLogger()
        counts, entries, by_employee = _scan_log_severity(logger.log_dir)
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


@main.group()
def catalog():
    """员工 Catalog 查询."""
    pass


def _catalog_data() -> list[dict[str, Any]]:
    result = discover_employees()
    employees = []
    for emp in result.employees.values():
        employees.append({
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
        })
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
            click.echo(f"{item['name']:<18} {item['display_name']:<12} agent={item['agent_id'] or '-'}")


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
@click.option("--smart-context", is_flag=True, help="自动检测项目类型并注入上下文")
@click.option("--raw", is_flag=True, help="输出原始渲染结果（不包裹 prompt 格式）")
@click.option("--copy", "to_clipboard", is_flag=True, help="复制到剪贴板")
@click.option("-o", "--output", type=click.Path(), help="输出到文件")
@click.option("--parallel", is_flag=True, help="跳过 Lane 串行调度")
@click.option("--execute", "execute_mode", is_flag=True, help="执行 prompt（调用 LLM API）")
@click.option("-m", "--message", "user_message", type=str, default=None, help="自定义 user message（--execute 模式）")
@click.option("--no-stream", "no_stream", is_flag=True, help="禁用流式输出（--execute 模式）")
def run(
    name: str,
    positional_args: tuple[str, ...],
    named_args: tuple[str, ...],
    agent_id: int | None,
    smart_context: bool,
    raw: bool,
    to_clipboard: bool,
    output: str | None,
    parallel: bool,
    execute_mode: bool,
    user_message: str | None,
    no_stream: bool,
):
    """加载员工并生成 prompt."""
    emp = sdk.get_employee(name)
    if emp is None:
        click.echo(f"未找到员工: {name}", err=True)
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


def _run_employee_job(
    *,
    emp,
    positional_args: tuple[str, ...],
    named_args: tuple[str, ...],
    agent_id: int | None,
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

    agent_identity = None
    if agent_id is not None:
        try:
            from crew.id_client import fetch_agent_identity

            agent_identity = fetch_agent_identity(agent_id)
            if agent_identity is None:
                click.echo(
                    f"Warning: 无法获取 Agent {agent_id} 身份，继续生成 prompt",
                    err=True,
                )
        except ImportError:
            click.echo("Warning: httpx 未安装，无法连接 knowlyr-id", err=True)

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
            agent_identity=agent_identity,
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
    if execute_mode:
        if agent_identity is None or not agent_identity.api_key:
            click.echo(
                "Error: --execute 需要 Agent 身份且已配置 api_key。"
                "请确认 --agent-id 并在 knowlyr-id 中设置 api_key。",
                err=True,
            )
            _finish_transcript(
                transcript_recorder, transcript_id,
                status="failed", detail="missing_api_key",
            )
            sys.exit(1)

        try:
            from crew.executor import execute_prompt
        except ImportError:
            click.echo(
                "Error: anthropic SDK 未安装。请运行: pip install knowlyr-crew[execute]",
                err=True,
            )
            _finish_transcript(
                transcript_recorder, transcript_id,
                status="failed", detail="missing_anthropic",
            )
            sys.exit(1)

        effective_model = agent_identity.model or emp.model or "claude-sonnet-4-20250514"
        effective_message = user_message or "请开始执行上述任务。"
        stream_enabled = not no_stream and not output and not to_clipboard

        def _on_chunk(chunk: str) -> None:
            click.echo(chunk, nl=False)

        try:
            result = execute_prompt(
                system_prompt=text,
                user_message=effective_message,
                api_key=agent_identity.api_key,
                model=effective_model,
                temperature=agent_identity.temperature,
                max_tokens=agent_identity.max_tokens,
                stream=stream_enabled,
                on_chunk=_on_chunk if stream_enabled else None,
            )
        except Exception as exc:
            click.echo(f"\nLLM 执行失败: {exc}", err=True)
            _finish_transcript(
                transcript_recorder, transcript_id,
                status="error", detail=f"execute_error: {str(exc)[:200]}",
            )
            sys.exit(1)

        if stream_enabled:
            click.echo()  # trailing newline
            already_streamed = True

        _record_transcript_message(
            transcript_recorder, transcript_id,
            "assistant", result.content,
            {
                "model": result.model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "stop_reason": result.stop_reason,
            },
        )

        click.echo(
            f"[{result.model}] {result.input_tokens} in / {result.output_tokens} out",
            err=True,
        )

        text = result.content

    try:
        from crew.log import WorkLogger

        work_logger = WorkLogger()
        session_id = work_logger.create_session(emp.name, args=args_dict, agent_id=agent_id)
        detail_msg = f"executed, {len(text)} chars" if execute_mode else f"via CLI, {len(text)} chars"
        work_logger.add_entry(session_id, "prompt_generated", detail_msg)
    except Exception:
        pass

    if agent_id is not None:
        try:
            from crew.id_client import send_heartbeat

            send_heartbeat(agent_id, detail=f"employee={emp.name}")
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
                subprocess.run(["pbcopy"], input=text.encode(), check=True)
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
            agent_id=agent_id,
        )


@main.command()
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


@main.command()
@click.option("--employee", type=str, default=None, help="创建指定员工的模板")
@click.option("--dir-format", is_flag=True, default=False, help="使用目录格式创建员工模板")
@click.option("--avatar", is_flag=True, default=False, help="创建后自动生成头像（需要 Gemini CLI）")
@click.option("--display-name", type=str, default=None, help="显示名称")
@click.option("--description", "desc", type=str, default=None, help="一句话描述")
@click.option("--character-name", type=str, default=None, help="角色姓名（如 陆明哲）")
@click.option("--avatar-prompt", type=str, default=None, help="头像生成描述")
@click.option("--tags", type=str, default=None, help="标签（逗号分隔）")
def init(employee: str | None, dir_format: bool, avatar: bool,
         display_name: str | None, desc: str | None, character_name: str | None,
         avatar_prompt: str | None, tags: str | None):
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
            character_name = character_name if character_name is not None else click.prompt("角色姓名（如 陆明哲）", default="")
            avatar_prompt = avatar_prompt if avatar_prompt is not None else click.prompt("头像描述（留空自动推断）", default="")
            tags_input = tags if tags is not None else click.prompt("标签（逗号分隔）", default="")
            tags_list = [t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else []
        else:
            display_name = display_name or employee
            desc = desc or "在此填写一句话描述"
            character_name = character_name or ""
            avatar_prompt = avatar_prompt or ""
            tags_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        emp_dir.mkdir()

        import yaml as _yaml
        config_data = {
            "name": employee,
            "display_name": display_name,
            "description": desc,
            "version": "1.0",
            "tags": tags_list,
            "triggers": [],
            "args": [{"name": "target", "description": "目标", "required": True}],
            "output": {"format": "markdown"},
        }
        if character_name:
            config_data["character_name"] = character_name
        if avatar_prompt:
            config_data["avatar_prompt"] = avatar_prompt

        (emp_dir / "employee.yaml").write_text(
            _yaml.dump(config_data, allow_unicode=True, sort_keys=False,
                       default_flow_style=False),
            encoding="utf-8",
        )
        (emp_dir / "prompt.md").write_text(
            """# 角色定义

你是……

## 工作流程

1. 第一步
2. 第二步
3. 第三步

## 输出格式

按需定义输出格式。
""",
            encoding="utf-8",
        )
        click.echo(f"已创建: {emp_dir}/")
        click.echo(f"  ├── employee.yaml")
        click.echo(f"  └── prompt.md")

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


# ── 模板命令 ──


def _parse_variables(items: tuple[str, ...]) -> dict[str, str]:
    """解析 key=value 形式的变量。"""
    variables: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise click.BadParameter("变量格式应为 key=value")
        key, value = item.split("=", 1)
        variables[key] = value
    return variables


def _default_display_name(slug: str) -> str:
    return slug.replace("-", " ").title()


@main.group()
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
    "-o", "--output", type=click.Path(), default=None,
    help="输出文件路径（默认 private/employees/<employee>.md）",
)
@click.option("--force", is_flag=True, help="如目标已存在则覆盖")
def template_apply(template_name: str, employee: str | None, variables: tuple[str, ...],
                   output: str | None, force: bool):
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


# ── session 子命令组 ──


@main.group(name="session")
def session_group():
    """会话记录管理 — JSONL 轨迹."""
    pass


@session_group.command("list")
@click.option("--type", "session_type", type=str, default=None, help="按类型过滤 (employee/pipeline/discussion)")
@click.option("--subject", type=str, default=None, help="按 subject 过滤 (员工/流水线名称)")
@click.option("-n", "--limit", type=int, default=20, help="返回条数")
@click.option("-f", "--format", "output_format", type=click.Choice(["table", "json"]), default="table")
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
                f"{item['session_id']}  {item.get('session_type','')}  {item.get('subject','')}  {item.get('started_at','')}"
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


# ── pipeline 子命令组 ──


@main.group()
def pipeline():
    """流水线管理 — 多员工顺序执行."""
    pass


@pipeline.command("list")
def pipeline_list():
    """列出所有可用流水线."""
    from crew.pipeline import discover_pipelines, load_pipeline

    pipelines = discover_pipelines()
    if not pipelines:
        click.echo("未找到流水线。")
        click.echo("在 .crew/pipelines/ 中创建 YAML 文件，或使用内置流水线。")
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="流水线")
        table.add_column("名称", style="cyan")
        table.add_column("描述", style="green")
        table.add_column("步骤数", justify="right")
        table.add_column("来源")

        for name, path in pipelines.items():
            try:
                p = load_pipeline(path)
                source = "内置" if "employees/pipelines" in str(path) else "项目"
                table.add_row(name, p.description, str(len(p.steps)), source)
            except Exception:
                table.add_row(name, "[解析失败]", "-", str(path))

        console.print(table)
    except ImportError:
        for name, path in pipelines.items():
            click.echo(f"  {name} — {path}")


@pipeline.command("show")
@click.argument("name")
def pipeline_show(name: str):
    """查看流水线详情."""
    from crew.pipeline import discover_pipelines, load_pipeline

    pipelines = discover_pipelines()
    if name not in pipelines:
        click.echo(f"未找到流水线: {name}", err=True)
        sys.exit(1)

    p = load_pipeline(pipelines[name])
    click.echo(f"流水线: {p.name}")
    click.echo(f"描述: {p.description}")
    click.echo(f"步骤: {len(p.steps)}")
    click.echo()
    for i, step in enumerate(p.steps, 1):
        args_str = ", ".join(f"{k}={v}" for k, v in step.args.items())
        click.echo(f"  {i}. {step.employee}" + (f" ({args_str})" if args_str else ""))


@pipeline.command("run")
@click.argument("name_or_path")
@click.option("--arg", "named_args", multiple=True, help="参数 (key=value)")
@click.option("--agent-id", type=int, default=None, help="绑定 knowlyr-id Agent ID")
@click.option("--smart-context/--no-smart-context", default=True, help="自动检测项目类型")
@click.option("-o", "--output", type=click.Path(), help="输出到文件")
@click.option("--parallel", is_flag=True, help="跳过 Lane 串行调度")
def pipeline_run(name_or_path: str, named_args: tuple[str, ...], agent_id: int | None,
                 smart_context: bool, output: str | None, parallel: bool):
    """执行流水线.

    NAME_OR_PATH 可以是流水线名称或 YAML 文件路径。
    """
    from crew.pipeline import discover_pipelines, load_pipeline, run_pipeline

    path_obj = Path(name_or_path)
    if path_obj.exists() and path_obj.suffix in (".yaml", ".yml"):
        p = load_pipeline(path_obj)
    else:
        pipelines = discover_pipelines()
        if name_or_path not in pipelines:
            click.echo(f"未找到流水线: {name_or_path}", err=True)
            sys.exit(1)
        p = load_pipeline(pipelines[name_or_path])

    initial_args: dict[str, str] = {}
    for item in named_args:
        if "=" in item:
            k, v = item.split("=", 1)
            initial_args[k] = v

    lane = LaneLock(f"pipeline:{p.name}") if not parallel else None
    if lane:
        lane.acquire()
    try:
        click.echo(f"执行流水线: {p.name} ({len(p.steps)} 步)", err=True)

        transcript_recorder, transcript_id = _start_transcript(
            "pipeline",
            p.name,
            {
                "steps": len(p.steps),
                "agent_id": agent_id,
                "initial_args": initial_args,
                "smart_context": bool(smart_context),
                "source": "cli.pipeline.run",
            },
        )

        try:
            results = run_pipeline(
                p, initial_args=initial_args,
                agent_id=agent_id, smart_context=smart_context,
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

        parts = []
        for i, r in enumerate(results, 1):
            header = f"{'=' * 60}\n## 步骤 {i}: {r['employee']}\n{'=' * 60}"
            parts.append(f"{header}\n\n{r['prompt']}")
            click.echo(f"  ✓ 步骤 {i}: {r['employee']}", err=True)
            _record_transcript_message(
                transcript_recorder,
                transcript_id,
                "step",
                r.get("prompt", ""),
                {
                    "step_index": i,
                    "employee": r.get("employee"),
                    "args": r.get("args", {}),
                    "error": r.get("error", False),
                },
            )

        combined = "\n\n".join(parts)

        try:
            if output:
                Path(output).write_text(combined, encoding="utf-8")
                click.echo(f"\n已写入: {output}", err=True)
                _record_transcript_event(
                    transcript_recorder,
                    transcript_id,
                    "output_file",
                    {"path": output},
                )
            else:
                click.echo(combined)
                _record_transcript_event(
                    transcript_recorder,
                    transcript_id,
                    "stdout",
                    {"chars": len(combined)},
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
                detail=f"{len(results)} steps",
            )
            _record_session_summary(
                employee=f"pipeline:{p.name}",
                session_id=transcript_id,
                agent_id=agent_id,
            )
    finally:
        if lane:
            lane.release()


# ── discuss 子命令组 ──


@main.group()
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
                    name, d.description, str(len(d.participants)),
                    str(rounds_count), source,
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
@click.option("--agent-id", type=int, default=None, help="绑定 knowlyr-id Agent ID")
@click.option("--smart-context/--no-smart-context", default=True, help="自动检测项目类型")
@click.option("--orchestrated", is_flag=True, default=False,
              help="编排模式：生成独立 prompt 计划（每个参会者独立推理）")
@click.option("-o", "--output", type=click.Path(), help="输出到文件")
@click.option("--parallel", is_flag=True, help="跳过 Lane 串行调度")
def discuss_run(name_or_path: str, named_args: tuple[str, ...], agent_id: int | None,
                smart_context: bool, orchestrated: bool, output: str | None,
                parallel: bool):
    """生成讨论会 prompt.

    NAME_OR_PATH 可以是讨论会名称或 YAML 文件路径。
    """
    from crew.discussion import (
        discover_discussions,
        load_discussion,
        render_discussion,
        render_discussion_plan,
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
    agent_id: int | None,
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
                agent_id=agent_id,
            )


def _run_discussion_prompt(
    *,
    discussion,
    initial_args: dict[str, str],
    agent_id: int | None,
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
                agent_id=agent_id,
            )


@discuss.command("adhoc")
@click.option("-e", "--employees", required=True, help="员工名称（逗号分隔）")
@click.option("-t", "--topic", required=True, help="议题")
@click.option("-g", "--goal", default="", help="目标")
@click.option("-r", "--rounds", type=int, default=2, help="轮次数（默认 2）")
@click.option("--round-template", type=str, default=None,
              help="轮次模板 (standard, brainstorm-to-decision, adversarial)")
@click.option("--output-format", type=click.Choice(["decision", "transcript", "summary"]),
              default="summary", help="输出格式")
@click.option("--agent-id", type=int, default=None, help="绑定 knowlyr-id Agent ID")
@click.option("--smart-context/--no-smart-context", default=True, help="自动检测项目类型")
@click.option("--orchestrated", is_flag=True, default=False,
              help="编排模式：生成独立 prompt 计划（每个参会者独立推理）")
@click.option("-o", "--output", type=click.Path(), help="输出到文件")
@click.option("--parallel", is_flag=True, help="跳过 Lane 串行调度")
def discuss_adhoc(employees: str, topic: str, goal: str, rounds: int,
                  round_template: str | None, output_format: str,
                  agent_id: int | None, smart_context: bool, orchestrated: bool,
                  output: str | None, parallel: bool):
    """发起即席讨论（无需 YAML 定义）.

    示例:
        crew discuss adhoc -e "code-reviewer,test-engineer" -t "auth 模块代码质量"
        crew discuss adhoc -e "hr-manager" -t "招聘方案"
    """
    from crew.discussion import (
        create_adhoc_discussion,
        render_discussion,
        render_discussion_plan,
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


# ── meetings 子命令组 ──


@main.group()
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
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None, help="输出文件（默认 .crew/meetings/<id>-summary.md）")
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
        lines.extend([
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
        ])
    lines.append(content)
    target.write_text("\n".join(lines), encoding="utf-8")
    click.echo(f"已导出: {target}")


# ── changelog ──


@main.command("changelog")
@click.option("--since", type=str, default=None, help="git log since (如 v0.1.0)")
@click.option("-n", "--limit", type=int, default=10, help="返回最近 N 条")
@click.option("-o", "--output", type=click.Path(path_type=Path), default="CHANGELOG_DRAFT.md")
def changelog_draft(since: str | None, limit: int, output: Path):
    """根据 git log 生成 changelog 草稿."""
    cmd = ["git", "log", f"-n", str(limit), "--pretty=format:%h %s"]
    if since:
        cmd = ["git", "log", f"{since}..HEAD", "--pretty=format:%h %s"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception as exc:
        lines = [f"(git log 失败: {exc})"]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    content = [f"## {now} Changelog", ""]
    for line in lines:
        content.append(f"- {line}")
    output.write_text("\n".join(content) + "\n", encoding="utf-8")
    click.echo(f"已生成: {output}")


# ── memory 子命令组 ──


@main.group()
def memory():
    """员工记忆管理 — 持久化经验存储."""
    pass


@memory.command("list")
def memory_list():
    """列出有记忆的员工."""
    from crew.memory import MemoryStore

    store = MemoryStore()
    employees = store.list_employees()
    if not employees:
        click.echo("暂无记忆数据。")
        return
    for emp in employees:
        entries = store.query(emp)
        click.echo(f"  {emp}: {len(entries)} 条记忆")


@memory.command("show")
@click.argument("employee")
@click.option("--category", type=click.Choice(["decision", "estimate", "finding", "correction"]),
              default=None, help="按类别过滤")
@click.option("-n", "--limit", type=int, default=20, help="返回条数")
def memory_show(employee: str, category: str | None, limit: int):
    """查看员工记忆."""
    from crew.memory import MemoryStore

    store = MemoryStore()
    entries = store.query(employee, category=category, limit=limit)
    if not entries:
        click.echo(f"员工 '{employee}' 暂无记忆。")
        return

    for entry in entries:
        conf = f" [{entry.confidence:.0%}]" if entry.confidence < 1.0 else ""
        click.echo(f"  [{entry.id}] ({entry.category}){conf} {entry.content}")


@memory.command("add")
@click.argument("employee")
@click.option("--category", "-c", required=True,
              type=click.Choice(["decision", "estimate", "finding", "correction"]),
              help="记忆类别")
@click.option("--content", "-m", required=True, help="记忆内容")
def memory_add(employee: str, category: str, content: str):
    """手动添加员工记忆."""
    from crew.memory import MemoryStore

    store = MemoryStore()
    entry = store.add(employee=employee, category=category, content=content)
    click.echo(f"已添加: [{entry.id}] ({entry.category}) {entry.content}")


@memory.command("correct")
@click.argument("employee")
@click.argument("old_id")
@click.option("--content", "-m", required=True, help="纠正后的内容")
def memory_correct(employee: str, old_id: str, content: str):
    """纠正一条记忆."""
    from crew.memory import MemoryStore

    store = MemoryStore()
    new_entry = store.correct(employee=employee, old_id=old_id, new_content=content)
    if new_entry is None:
        click.echo(f"未找到记忆: {old_id}", err=True)
        sys.exit(1)
    click.echo(f"已纠正: [{new_entry.id}] {new_entry.content}")


@memory.command("index")
def memory_index_cmd():
    """重建混合搜索索引."""
    from crew.memory_index import MemorySearchIndex

    index = MemorySearchIndex()
    stats = index.rebuild()
    click.echo(
        f"索引完成: 记忆 {stats.memory_entries} 条, 会话 {stats.session_messages} 条"
    )


@memory.command("search")
@click.argument("query")
@click.option("--employee", type=str, default=None, help="按员工过滤")
@click.option("--kind", type=click.Choice(["memory", "session"]), default=None,
              help="数据类型过滤")
@click.option("-n", "--limit", type=int, default=5, help="返回条数")
@click.option("--json", "json_output", is_flag=True, help="JSON 输出")
def memory_search(query: str, employee: str | None, kind: str | None,
                  limit: int, json_output: bool):
    """搜索持久记忆 + 会话记录."""
    from crew.memory_index import MemorySearchIndex

    index = MemorySearchIndex()
    results = index.search(query, limit=limit, employee=employee, kind=kind)

    if not results:
        click.echo("未找到匹配项。")
        return

    if json_output:
        click.echo(json.dumps(results, ensure_ascii=False, indent=2))
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="记忆搜索结果")
        table.add_column("类型", style="cyan")
        table.add_column("员工")
        table.add_column("标题")
        table.add_column("摘要")

        for item in results:
            snippet = item.get("snippet") or item.get("content", "")
            table.add_row(
                item.get("kind", ""),
                item.get("employee", ""),
                item.get("title", ""),
                snippet[:120],
            )

        console.print(table)
    except ImportError:
        for item in results:
            snippet = item.get("snippet") or item.get("content", "")
            click.echo(
                f"[{item.get('kind','')}] {item.get('employee','')} - {item.get('title','')}\n  {snippet}"
            )


# ── eval 子命令组 ──


@main.group(name="eval")
def eval_group():
    """决策评估 — 追踪决策质量、回溯评估."""
    pass


@eval_group.command("track")
@click.argument("employee")
@click.option("--category", "-c", required=True,
              type=click.Choice(["estimate", "recommendation", "commitment"]),
              help="决策类别")
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
@click.option("--status", type=click.Choice(["pending", "evaluated"]), default=None, help="按状态过滤")
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


# ── register 命令 ──


@main.command()
@click.argument("name")
@click.option("--dry-run", is_flag=True, help="预览操作但不执行")
def register(name: str, dry_run: bool):
    """将员工注册为 knowlyr-id Agent 并保存 agent_id."""
    from crew.discovery import discover_employees

    result = discover_employees()
    emp = result.get(name)
    if not emp:
        click.echo(f"未找到员工: {name}", err=True)
        raise SystemExit(1)

    if emp.agent_id is not None:
        click.echo(f"员工 '{emp.name}' 已绑定 Agent #{emp.agent_id}", err=True)
        raise SystemExit(1)

    nickname = emp.character_name or emp.display_name or emp.name
    title = emp.display_name or emp.name
    capabilities = emp.description
    domains = emp.tags[:5] if emp.tags else []
    model = emp.model

    click.echo(f"注册员工 \"{emp.name}\" 到 knowlyr-id...", err=True)
    click.echo(f"  nickname:     {nickname}", err=True)
    click.echo(f"  title:        {title}", err=True)
    click.echo(f"  capabilities: {capabilities}", err=True)
    if domains:
        click.echo(f"  domains:      {', '.join(domains)}", err=True)
    if model:
        click.echo(f"  model:        {model}", err=True)

    # 检查头像
    avatar_b64 = _load_avatar_base64(emp)
    if avatar_b64:
        click.echo(f"  avatar:       ✓ (avatar.webp)", err=True)

    if dry_run:
        click.echo("\n(dry-run 模式，未执行注册)", err=True)
        return

    from crew.id_client import register_agent

    agent_id = register_agent(
        nickname=nickname,
        title=title,
        capabilities=capabilities,
        domains=domains,
        model=model,
        avatar_base64=avatar_b64,
    )
    if agent_id is None:
        click.echo("注册失败（检查 KNOWLYR_ID_URL 和 AGENT_API_TOKEN 环境变量）", err=True)
        raise SystemExit(1)

    click.echo(f"✓ 已注册为 Agent #{agent_id}", err=True)

    # 回写 agent_id 到源文件
    if emp.source_path and emp.source_layer in ("global", "project"):
        _write_agent_id(emp, agent_id)
    else:
        click.echo(f"  提示: 请手动在员工定义中添加 agent_id: {agent_id}", err=True)


def _load_avatar_base64(emp) -> str | None:
    """从员工目录加载 avatar.webp 并 base64 编码."""
    if not emp.source_path:
        return None
    avatar_dir = emp.source_path if emp.source_path.is_dir() else emp.source_path.parent
    avatar_path = avatar_dir / "avatar.webp"
    if not avatar_path.exists():
        return None
    import base64
    return base64.b64encode(avatar_path.read_bytes()).decode()


def _write_agent_id(emp, agent_id: int) -> None:
    """将 agent_id 回写到员工定义文件."""
    import yaml

    source = emp.source_path
    if source is None:
        return

    if source.is_dir():
        # 目录格式: 更新 employee.yaml
        config_path = source / "employee.yaml"
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["agent_id"] = agent_id
            config_path.write_text(
                yaml.dump(config, allow_unicode=True, sort_keys=False,
                          default_flow_style=False),
                encoding="utf-8",
            )
            click.echo(f"✓ agent_id 已写入 {config_path}", err=True)
    elif source.is_file() and source.suffix == ".md":
        # 单文件格式: 在 frontmatter 末尾 --- 之前插入 agent_id
        import re
        content = source.read_text(encoding="utf-8")
        # 匹配第二个 ---
        match = re.match(r"(---\n.*?)(---)", content, re.DOTALL)
        if match:
            new_content = f"{match.group(1)}agent_id: {agent_id}\n{match.group(2)}{content[match.end():]}"
            source.write_text(new_content, encoding="utf-8")
            click.echo(f"✓ agent_id 已写入 {source}", err=True)


# ── avatar 命令 ──


def _run_avatar_gen(
    output_dir: Path,
    display_name: str = "",
    character_name: str = "",
    description: str = "",
    avatar_prompt: str = "",
) -> None:
    """执行头像生成 + 压缩流程."""
    from crew.avatar import generate_avatar, compress_avatar

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


@main.command()
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


# ── agents 子命令组 ──


@main.group()
def agents():
    """Agent 管理（与 knowlyr-id 交互）."""


@agents.command("list")
@click.option("-f", "--format", "output_format",
              type=click.Choice(["table", "json"]), default="table")
def agents_list_cmd(output_format: str):
    """列出 knowlyr-id 中的所有 Agent."""
    from crew.id_client import list_agents

    data = list_agents()
    if data is None:
        click.echo("获取失败（检查 KNOWLYR_ID_URL 和 AGENT_API_TOKEN 环境变量）", err=True)
        raise SystemExit(1)

    if not data:
        click.echo("暂无 Agent", err=True)
        return

    if output_format == "json":
        import json
        click.echo(json.dumps(data, ensure_ascii=False, indent=2))
        return

    from rich.console import Console
    from rich.table import Table

    table = Table(title="Agents")
    table.add_column("ID", style="cyan")
    table.add_column("Nickname", style="bold")
    table.add_column("Title")
    table.add_column("Domains")
    table.add_column("Status", style="green")
    table.add_column("Heartbeat")

    for agent in data:
        table.add_row(
            str(agent.get("id", "")),
            agent.get("nickname", ""),
            agent.get("title", ""),
            ", ".join(agent.get("domains", [])),
            agent.get("status", ""),
            str(agent.get("heartbeat_count", 0)),
        )

    Console(stderr=True).print(table)


@agents.command("status")
@click.argument("target")
@click.option("--employee", "by_employee", is_flag=True, help="target 为员工名称而非 agent_id")
@click.option("--heartbeat", is_flag=True, help="额外发送一次心跳用于连通性检测")
def agents_status_cmd(target: str, by_employee: bool, heartbeat: bool):
    """检查 knowlyr-id Agent 状态."""
    from crew.id_client import fetch_agent_identity, send_heartbeat

    agent_id: int
    employee_name: str | None = None

    if by_employee:
        result = discover_employees()
        emp = result.get(target)
        if not emp:
            click.echo(f"未找到员工: {target}", err=True)
            raise SystemExit(1)
        if getattr(emp, "agent_id", None) is None:
            click.echo(f"员工 '{emp.name}' 未绑定 Agent", err=True)
            raise SystemExit(1)
        agent_id = int(emp.agent_id)
        employee_name = emp.name
    else:
        try:
            agent_id = int(target)
        except ValueError:
            click.echo("请提供有效的 Agent ID（整数）", err=True)
            raise SystemExit(1)

    identity = fetch_agent_identity(agent_id)
    if identity is None:
        click.echo(
            f"无法获取 Agent #{agent_id} 状态（检查 KNOWLYR_ID_URL / AGENT_API_TOKEN）",
            err=True,
        )
        raise SystemExit(1)

    click.echo(f"Agent #{agent_id} 状态:")
    if employee_name:
        click.echo(f"  Employee:    {employee_name}")
    click.echo(f"  Nickname:    {identity.nickname or '-'}")
    click.echo(f"  Title:       {identity.title or '-'}")
    click.echo(f"  Status:      {identity.agent_status or '-'}")
    click.echo(f"  Model:       {identity.model or '-'}")
    domains = ", ".join(identity.domains) if identity.domains else "-"
    click.echo(f"  Domains:     {domains}")
    mem = (identity.memory or "").strip()
    click.echo(f"  Memory Size: {len(mem)} chars")
    if mem:
        preview = mem.splitlines()[0]
        truncated = preview[:80]
        suffix = "…" if len(preview) > len(truncated) or len(mem) > len(preview) else ""
        click.echo(f"  Memory Peek: {truncated}{suffix}")

    if heartbeat:
        ok = send_heartbeat(agent_id, detail="cli.agents.status")
        status = "OK" if ok else "FAILED"
        line = f"  Heartbeat:  {status}"
        if ok:
            click.echo(line)
        else:
            click.echo(line, err=True)


@agents.command("sync")
@click.argument("name")
def agents_sync_cmd(name: str):
    """同步员工元数据到 knowlyr-id Agent."""
    from crew.discovery import discover_employees
    from crew.id_client import update_agent

    result = discover_employees()
    emp = result.get(name)
    if not emp:
        click.echo(f"未找到员工: {name}", err=True)
        raise SystemExit(1)

    if emp.agent_id is None:
        click.echo(f"员工 '{emp.name}' 未绑定 Agent（先执行 knowlyr-crew register {name}）", err=True)
        raise SystemExit(1)

    nickname = emp.character_name or emp.display_name or emp.name
    title = emp.display_name or emp.name
    capabilities = emp.description
    domains = emp.tags[:5] if emp.tags else []
    system_prompt = emp.body if emp.body else None

    avatar_b64 = _load_avatar_base64(emp)

    click.echo(f"同步 \"{emp.name}\" (Agent #{emp.agent_id}) 到 knowlyr-id...", err=True)
    click.echo(f"  nickname:      {nickname}", err=True)
    click.echo(f"  title:         {title}", err=True)
    click.echo(f"  capabilities:  {capabilities}", err=True)
    if domains:
        click.echo(f"  domains:       {', '.join(domains)}", err=True)
    if system_prompt:
        click.echo(f"  system_prompt: {len(system_prompt)} 字符", err=True)
    if avatar_b64:
        click.echo("  avatar:        ✓", err=True)

    ok = update_agent(
        agent_id=emp.agent_id,
        nickname=nickname,
        title=title,
        capabilities=capabilities,
        domains=domains,
        model=emp.model or None,
        system_prompt=system_prompt,
        avatar_base64=avatar_b64,
    )
    if not ok:
        click.echo("同步失败", err=True)
        raise SystemExit(1)

    click.echo("✓ 同步完成", err=True)


# ── mcp 命令 ──


@main.command()
def mcp():
    """启动 MCP Server（stdio 模式）."""
    from crew.mcp_server import main as mcp_main
    mcp_main()
