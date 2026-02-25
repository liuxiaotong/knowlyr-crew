"""流水线命令 — pipeline group + checkpoint subgroup."""

import asyncio
import sys
from pathlib import Path

import click

from crew.cli import (
    _finish_transcript,
    _record_session_summary,
    _record_transcript_event,
    _record_transcript_message,
    _start_transcript,
    _suggest_similar,
)
from crew.lanes import LaneLock


@click.group()
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
        hint = _suggest_similar(name, list(pipelines.keys()))
        click.echo(f"未找到流水线: {name}{hint}", err=True)
        sys.exit(1)

    from crew.models import ConditionalStep, LoopStep, ParallelGroup

    p = load_pipeline(pipelines[name])
    click.echo(f"流水线: {p.name}")
    click.echo(f"描述: {p.description}")
    click.echo(f"步骤: {len(p.steps)}")
    click.echo()
    for i, item in enumerate(p.steps, 1):
        if isinstance(item, ParallelGroup):
            click.echo(f"  {i}. [并行]")
            for j, sub in enumerate(item.parallel, 1):
                id_str = f" id={sub.id}" if sub.id else ""
                args_str = ", ".join(f"{k}={v}" for k, v in sub.args.items())
                click.echo(
                    f"     {i}.{j} {sub.employee}{id_str}" + (f" ({args_str})" if args_str else "")
                )
        elif isinstance(item, ConditionalStep):
            body = item.condition
            matcher = (
                f"contains '{body.contains}'" if body.contains else f"matches '{body.matches}'"
            )
            click.echo(f"  {i}. [条件] check={body.check} {matcher}")
            click.echo("     then:")
            for j, sub in enumerate(body.then, 1):
                id_str = f" id={sub.id}" if sub.id else ""
                click.echo(f"       {i}.T{j} {sub.employee}{id_str}")
            if body.else_:
                click.echo("     else:")
                for j, sub in enumerate(body.else_, 1):
                    id_str = f" id={sub.id}" if sub.id else ""
                    click.echo(f"       {i}.E{j} {sub.employee}{id_str}")
        elif isinstance(item, LoopStep):
            body = item.loop
            matcher = (
                f"contains '{body.until.contains}'"
                if body.until.contains
                else f"matches '{body.until.matches}'"
            )
            click.echo(f"  {i}. [循环] max={body.max_iterations}")
            for j, sub in enumerate(body.steps, 1):
                id_str = f" id={sub.id}" if sub.id else ""
                click.echo(f"     {i}.L{j} {sub.employee}{id_str}")
            click.echo(f"     until: {matcher}")
        else:
            id_str = f" id={item.id}" if item.id else ""
            args_str = ", ".join(f"{k}={v}" for k, v in item.args.items())
            click.echo(f"  {i}. {item.employee}{id_str}" + (f" ({args_str})" if args_str else ""))


@pipeline.command("graph")
@click.argument("name")
@click.option("-o", "--output", type=click.Path(), help="输出到文件")
def pipeline_graph(name: str, output: str | None):
    """生成流水线 Mermaid 流程图."""
    from crew.pipeline import discover_pipelines, load_pipeline, pipeline_to_mermaid

    pipelines = discover_pipelines()
    if name not in pipelines:
        hint = _suggest_similar(name, list(pipelines.keys()))
        click.echo(f"未找到流水线: {name}{hint}", err=True)
        sys.exit(1)

    p = load_pipeline(pipelines[name])
    mermaid = pipeline_to_mermaid(p)

    if output:
        Path(output).write_text(mermaid, encoding="utf-8")
        click.echo(f"已保存到 {output}")
    else:
        click.echo(mermaid)


@pipeline.group("checkpoint")
def pipeline_checkpoint():
    """管理流水线断点."""


@pipeline_checkpoint.command("list")
@click.option("-d", "--project-dir", type=click.Path(path_type=Path), default=None)
def checkpoint_list(project_dir):
    """列出有断点的 pipeline 任务."""
    from crew.task_registry import TaskRegistry

    base = Path(project_dir) if project_dir else Path.cwd()
    persist_path = base / ".crew" / "tasks.jsonl"
    if not persist_path.exists():
        click.echo("未找到任务记录文件 (.crew/tasks.jsonl)")
        return

    registry = TaskRegistry(persist_path=persist_path)
    tasks = [t for t in registry.list_recent(n=100) if t.checkpoint and t.target_type == "pipeline"]
    if not tasks:
        click.echo("暂无带断点的 pipeline 任务。")
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Pipeline 断点")
        table.add_column("Task ID", style="cyan")
        table.add_column("Pipeline", style="green")
        table.add_column("状态")
        table.add_column("已完成步骤", justify="right")

        for t in tasks:
            completed = len(t.checkpoint.get("completed_steps", []))
            table.add_row(t.task_id, t.target_name, t.status, str(completed))

        console.print(table)
    except ImportError:
        for t in tasks:
            completed = len(t.checkpoint.get("completed_steps", []))
            click.echo(f"  {t.task_id}  {t.target_name}  {t.status}  步骤={completed}")


@pipeline_checkpoint.command("resume")
@click.argument("task_id")
@click.option("--model", type=str, default=None, help="LLM 模型标识符")
@click.option("--retry-failed", is_flag=True, help="从第一个失败步骤重新执行")
@click.option("--no-fail-fast", is_flag=True, help="步骤失败时继续执行后续步骤")
@click.option("-d", "--project-dir", type=click.Path(path_type=Path), default=None)
def checkpoint_resume(
    task_id: str, model: str | None, retry_failed: bool, no_fail_fast: bool, project_dir
):
    """从断点恢复 pipeline 执行."""
    from crew.pipeline import aresume_pipeline, discover_pipelines, load_pipeline
    from crew.task_registry import TaskRegistry

    base = Path(project_dir) if project_dir else Path.cwd()
    persist_path = base / ".crew" / "tasks.jsonl"
    if not persist_path.exists():
        click.echo("未找到任务记录文件", err=True)
        sys.exit(1)

    registry = TaskRegistry(persist_path=persist_path)
    record = registry.get(task_id)
    if record is None:
        click.echo(f"未找到任务: {task_id}", err=True)
        sys.exit(1)

    if not record.checkpoint:
        click.echo("该任务没有断点数据", err=True)
        sys.exit(1)

    checkpoint = dict(record.checkpoint)

    # --retry-failed: 回退到第一个失败步骤
    if retry_failed:
        completed = checkpoint.get("completed_steps", [])
        first_error_idx = None
        for idx, item in enumerate(completed):
            entries = item if isinstance(item, list) else [item]
            if any(e.get("error") for e in entries):
                first_error_idx = idx
                break
        if first_error_idx is not None:
            checkpoint["completed_steps"] = completed[:first_error_idx]
            checkpoint["next_step_i"] = first_error_idx
            # 重算 flat_index
            flat = 0
            for item in checkpoint["completed_steps"]:
                flat += len(item) if isinstance(item, list) else 1
            checkpoint["next_flat_index"] = flat
            click.echo(f"回退到步骤 {first_error_idx + 1}（第一个失败步骤）")
        else:
            click.echo("没有找到失败步骤，从上次断点继续")

    pipeline_name = checkpoint.get("pipeline_name", record.target_name)
    pipelines = discover_pipelines(project_dir=base)
    if pipeline_name not in pipelines:
        click.echo(f"未找到 pipeline: {pipeline_name}", err=True)
        sys.exit(1)

    p = load_pipeline(pipelines[pipeline_name])
    total_steps = len(p.steps)
    restored = checkpoint.get("next_step_i", 0)

    last_step_i = [restored]  # 按 checkpoint 的 next_step_i 跟踪顶层步骤

    def _on_step(step_result, checkpoint_data):
        registry.update_checkpoint(task_id, checkpoint_data)
        current_step = checkpoint_data.get("next_step_i", last_step_i[0])
        branch = f" [{step_result.branch}]" if step_result.branch else ""
        status = " [失败]" if step_result.error else ""
        if current_step != last_step_i[0]:
            last_step_i[0] = current_step
        click.echo(f"  步骤 {current_step}/{total_steps}: {step_result.employee}{branch}{status}")

    click.echo(f"恢复 pipeline: {pipeline_name} (task={task_id})")
    if restored > 0:
        click.echo(f"  已恢复 {restored} 步，从步骤 {restored + 1} 继续")
    result = asyncio.run(
        aresume_pipeline(
            p,
            checkpoint=checkpoint,
            initial_args=record.args,
            project_dir=base,
            execute=bool(model),
            fail_fast=not no_fail_fast,
            model=model,
            on_step_complete=_on_step,
        )
    )

    has_errors = any(
        (r.error if not isinstance(r, list) else any(sub.error for sub in r)) for r in result.steps
    )
    if has_errors:
        registry.update(task_id, "failed", result=result.model_dump(mode="json"))
        click.echo(f"Pipeline 执行有失败步骤 (共 {len(result.steps)} 步)", err=True)
        sys.exit(1)
    else:
        registry.update(task_id, "completed", result=result.model_dump(mode="json"))
        click.echo(f"完成! 共 {len(result.steps)} 步")


@pipeline.command("run")
@click.argument("name_or_path")
@click.option("--arg", "named_args", multiple=True, help="参数 (key=value)")
@click.option("--agent-id", type=int, default=None, help="绑定平台 Agent ID")
@click.option("--smart-context/--no-smart-context", default=True, help="自动检测项目类型")
@click.option("-o", "--output", type=click.Path(), help="输出到文件")
@click.option("--parallel", is_flag=True, help="跳过 Lane 串行调度")
@click.option("--execute", is_flag=True, help="执行模式 — 自动调用 LLM 串联执行")
@click.option("--model", type=str, default=None, help="LLM 模型标识符（execute 模式）")
def pipeline_run(
    name_or_path: str,
    named_args: tuple[str, ...],
    agent_id: str | None,
    smart_context: bool,
    output: str | None,
    parallel: bool,
    execute: bool,
    model: str | None,
):
    """执行流水线.

    NAME_OR_PATH 可以是流水线名称或 YAML 文件路径。
    默认 prompt-only 模式，加 --execute 启用 LLM 自动执行。
    """
    from crew.pipeline import _flatten_results, discover_pipelines, load_pipeline, run_pipeline

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

    lane = LaneLock(f"pipeline:{p.name}") if not parallel else None
    if lane:
        lane.acquire()
    try:
        mode_label = "execute" if execute else "prompt-only"
        click.echo(f"执行流水线: {p.name} ({len(p.steps)} 步, {mode_label})", err=True)

        transcript_recorder, transcript_id = _start_transcript(
            "pipeline",
            p.name,
            {
                "steps": len(p.steps),
                "agent_id": agent_id,
                "initial_args": initial_args,
                "smart_context": bool(smart_context),
                "execute": execute,
                "source": "cli.pipeline.run",
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
                    "args": r.args,
                    "error": r.error,
                    "execute": execute,
                },
            )

        try:
            result = run_pipeline(
                p,
                initial_args=initial_args,
                agent_id=agent_id,
                smart_context=smart_context,
                execute=execute,
                api_key=api_key,
                model=model,
                on_step_complete=_on_step,
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

        # 组装输出
        all_results = _flatten_results(result.steps)
        parts = []
        for r in all_results:
            content = r.output if execute else r.prompt
            header = f"{'=' * 60}\n## 步骤 {r.step_index + 1}: {r.employee}\n{'=' * 60}"
            parts.append(f"{header}\n\n{content}")

        combined = "\n\n".join(parts)

        # token 统计（execute 模式）
        if execute:
            click.echo(
                f"\n总计: {result.total_input_tokens}+{result.total_output_tokens} tokens, "
                f"{result.total_duration_ms}ms",
                err=True,
            )

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
                detail=f"{len(all_results)} steps, mode={result.mode}",
            )
            _record_session_summary(
                employee=f"pipeline:{p.name}",
                session_id=transcript_id,
            )
    finally:
        if lane:
            lane.release()
