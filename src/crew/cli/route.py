"""协作流程命令 — route group."""

import sys
from pathlib import Path
from typing import Any

import click

from crew.cli import (
    _finish_transcript,
    _record_session_summary,
    _record_transcript_message,
    _start_transcript,
)


@click.group()
def route():
    """协作流程管理 — 按路由模板发起多员工协作."""
    pass


@route.command("list")
@click.option("--format", "output_format", type=click.Choice(["table", "json"]), default="table")
def route_list(output_format: str):
    """列出所有协作流程模板."""
    from crew.organization import load_organization
    from crew.paths import resolve_project_dir

    org = load_organization(project_dir=resolve_project_dir())
    templates = org.routing_templates
    if not templates:
        click.echo("未找到协作流程模板。")
        click.echo("在 organization.yaml 的 routing_templates 中定义。")
        return

    if output_format == "json":
        import json as _json

        data = []
        for name, tmpl in templates.items():
            human_count = sum(1 for s in tmpl.steps if s.human)
            data.append(
                {
                    "name": name,
                    "label": tmpl.label,
                    "steps": len(tmpl.steps),
                    "human_steps": human_count,
                }
            )
        click.echo(_json.dumps(data, ensure_ascii=False, indent=2))
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="协作流程")
        table.add_column("名称", style="cyan")
        table.add_column("标签", style="green")
        table.add_column("步骤数", justify="right")
        table.add_column("人工节点", justify="right")

        for name, tmpl in templates.items():
            human_count = sum(1 for s in tmpl.steps if s.human)
            table.add_row(
                name, tmpl.label, str(len(tmpl.steps)), str(human_count) if human_count else "—"
            )
        console.print(table)
    except ImportError:
        for name, tmpl in templates.items():
            human_count = sum(1 for s in tmpl.steps if s.human)
            click.echo(f"  {name:20s} {tmpl.label:12s} {len(tmpl.steps)} 步  {human_count} 人工")


@route.command("show")
@click.argument("name")
def route_show(name: str):
    """查看协作流程详情."""
    from crew.organization import load_organization
    from crew.paths import resolve_project_dir

    org = load_organization(project_dir=resolve_project_dir())
    tmpl = org.routing_templates.get(name)
    if not tmpl:
        available = list(org.routing_templates.keys())
        click.echo(f"未找到流程: {name}", err=True)
        if available:
            click.echo(f"可用: {', '.join(available)}", err=True)
        sys.exit(1)

    click.echo(f"{tmpl.label}（{len(tmpl.steps)} 步）\n")
    for i, step in enumerate(tmpl.steps, 1):
        # 执行人
        if step.human:
            executor = "[人工判断]"
        elif step.employee:
            executor = step.employee
        elif step.employees:
            executor = ", ".join(step.employees)
        elif step.team:
            executor = f"团队:{step.team}"
        else:
            executor = "未指定"

        # 标记
        tags = []
        if step.human:
            tags.append("人工")
        if step.approval:
            tags.append("需审批")
        if step.optional:
            tags.append("可选")
        tag_str = f" [{', '.join(tags)}]" if tags else ""

        click.echo(f"  {i}. {step.role:16s} → {executor}{tag_str}")
        if step.description:
            click.echo(f"     {step.description}")


@route.command("run")
@click.argument("name")
@click.argument("task")
@click.option("--override", "overrides_raw", multiple=True, help="覆盖执行人 (role=employee)")
@click.option("--agent-id", type=int, default=None, help="绑定平台 Agent ID")
@click.option("--smart-context/--no-smart-context", default=True, help="自动检测项目类型")
@click.option("-o", "--output", type=click.Path(), help="输出到文件")
@click.option("--execute", is_flag=True, help="执行模式 — 自动调用 LLM 串联执行")
@click.option("--model", type=str, default=None, help="LLM 模型标识符（execute 模式）")
@click.option("--remote", is_flag=True, help="远程执行 — 发到 crew 服务器（支持审批检查点）")
def route_run(
    name: str,
    task: str,
    overrides_raw: tuple[str, ...],
    agent_id: str | None,
    smart_context: bool,
    output: str | None,
    execute: bool,
    model: str | None,
    remote: bool,
):
    """执行协作流程.

    NAME 是路由模板名称（如 code_change）。
    TASK 是任务描述。

    默认 prompt-only 模式，加 --execute 启用 LLM 自动执行。
    加 --remote 发到 crew 服务器执行（支持审批检查点）。
    """
    from crew.organization import load_organization

    # 解析 overrides
    overrides: dict[str, str] = {}
    for item in overrides_raw:
        if "=" in item:
            k, v = item.split("=", 1)
            overrides[k] = v

    # 远程执行模式
    if remote:
        _route_run_remote(name, task, overrides)
        return

    # 本地执行 — 展开模板为 Pipeline 然后复用 run_pipeline
    from crew.paths import resolve_project_dir

    org = load_organization(project_dir=resolve_project_dir())
    tmpl = org.routing_templates.get(name)
    if not tmpl:
        available = list(org.routing_templates.keys())
        click.echo(f"未找到流程: {name}", err=True)
        if available:
            click.echo(f"可用: {', '.join(available)}", err=True)
        sys.exit(1)

    # 展开模板为步骤列表
    from crew.models import PipelineStep as _PStep

    pipeline_steps: list[_PStep] = []
    skipped: list[str] = []

    for step in tmpl.steps:
        if step.optional and step.role not in overrides:
            skipped.append(f"{step.role}（可选）")
            continue
        if step.human:
            skipped.append(f"{step.role}（人工）")
            continue

        emp_name = overrides.get(step.role)
        if not emp_name:
            if step.employee:
                emp_name = step.employee
            elif step.employees:
                emp_name = step.employees[0]
            elif step.team:
                members = org.get_team_members(step.team)
                emp_name = members[0] if members else None
        if not emp_name:
            skipped.append(f"{step.role}（无执行人）")
            continue

        # 构造步骤任务描述
        step_task = f"[{step.role}] {task}"
        if pipeline_steps:
            step_task += "\n\n上一步结果:\n$prev"

        pipeline_steps.append(
            _PStep(
                employee=emp_name,
                id=step.role,
                args={"task": step_task},
            )
        )

    if not pipeline_steps:
        click.echo("模板展开后无可执行步骤。", err=True)
        sys.exit(1)

    if skipped:
        click.echo(f"跳过: {', '.join(skipped)}", err=True)

    # 构造 Pipeline 对象
    from crew.pipeline import Pipeline, _flatten_results, run_pipeline

    p = Pipeline(
        name=f"route:{name}",
        description=f"{tmpl.label} — {task[:60]}",
        steps=pipeline_steps,
    )

    # execute 模式需要 API key
    api_key = None
    if execute:
        from crew.providers import detect_provider, resolve_api_key

        effective_model = model or "claude-sonnet-4-20250514"
        try:
            provider = detect_provider(effective_model)
            api_key = resolve_api_key(provider)
        except ValueError as e:
            click.echo(f"错误: {e}", err=True)
            sys.exit(1)

    mode_label = "execute" if execute else "prompt-only"
    click.echo(f"执行流程: {tmpl.label} ({len(pipeline_steps)} 步, {mode_label})", err=True)

    transcript_recorder, transcript_id = _start_transcript(
        "route",
        name,
        {
            "steps": len(pipeline_steps),
            "skipped": skipped,
            "agent_id": agent_id,
            "task": task[:200],
            "execute": execute,
            "source": "cli.route.run",
        },
    )

    def _on_step(r, _checkpoint=None):
        status = "✗" if r.error else "✓"
        suffix = ""
        if execute and not r.error:
            suffix = f" ({r.input_tokens}+{r.output_tokens} tokens, {r.duration_ms}ms)"
        click.echo(f"  {status} {r.employee}{suffix}", err=True)
        _record_transcript_message(
            transcript_recorder,
            transcript_id,
            "step",
            r.output if execute else r.prompt,
            {
                "step_index": r.step_index,
                "employee": r.employee,
                "error": r.error,
                "execute": execute,
            },
        )

    try:
        result = run_pipeline(
            p,
            initial_args={"task": task},
            agent_id=agent_id,
            smart_context=smart_context,
            execute=execute,
            api_key=api_key,
            model=model,
            on_step_complete=_on_step,
        )
    except Exception as exc:
        _finish_transcript(
            transcript_recorder, transcript_id, status="error", detail=str(exc)[:200]
        )
        raise

    all_results = _flatten_results(result.steps)
    parts = []
    for r in all_results:
        content = r.output if execute else r.prompt
        header = f"{'=' * 60}\n## [{r.employee}]\n{'=' * 60}"
        parts.append(f"{header}\n\n{content}")

    combined = "\n\n".join(parts)

    if execute:
        click.echo(
            f"\n总计: {result.total_input_tokens}+{result.total_output_tokens} tokens, "
            f"{result.total_duration_ms}ms",
            err=True,
        )

    if output:
        Path(output).write_text(combined, encoding="utf-8")
        click.echo(f"\n已写入: {output}", err=True)
    else:
        click.echo(combined)

    _finish_transcript(
        transcript_recorder,
        transcript_id,
        status="completed",
        detail=f"{len(all_results)} steps, mode={result.mode}",
    )
    _record_session_summary(
        employee=f"route:{name}",
        session_id=transcript_id,
    )


def _route_run_remote(name: str, task: str, overrides: dict[str, str]) -> None:
    """远程执行路由模板 — 发到 crew 服务器."""
    import os

    crew_url = os.environ.get("CREW_REMOTE_URL", "")
    crew_token = os.environ.get("CREW_API_TOKEN", "")
    if not crew_url or not crew_token:
        click.echo(
            "远程执行需要设置环境变量:\n"
            "  CREW_REMOTE_URL=https://crew.knowlyr.com\n"
            "  CREW_API_TOKEN=<your-token>",
            err=True,
        )
        sys.exit(1)

    try:
        import httpx
    except ImportError:
        click.echo("远程执行需要 httpx: pip install httpx", err=True)
        sys.exit(1)

    import json as _json

    url = f"{crew_url.rstrip('/')}/run/route/{name}"
    payload: dict[str, Any] = {"args": {"task": task}}
    if overrides:
        payload["overrides"] = overrides

    click.echo(f"发送到 {url} ...", err=True)
    try:
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {crew_token}"},
            json=payload,
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        click.echo(_json.dumps(data, ensure_ascii=False, indent=2))
    except httpx.HTTPStatusError as e:
        click.echo(f"服务器返回 {e.response.status_code}: {e.response.text}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"请求失败: {e}", err=True)
        sys.exit(1)
