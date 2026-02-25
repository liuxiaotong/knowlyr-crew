"""服务器命令 — serve + mcp + agent group."""

import sys
from pathlib import Path

import click


@click.command()
@click.option("--host", default="0.0.0.0", help="监听地址")
@click.option("--port", default=8765, type=int, help="监听端口")
@click.option(
    "--token",
    default=None,
    envvar="CREW_API_TOKEN",
    help="Bearer token（未设置则不启用认证）",
)
@click.option(
    "-d",
    "--project-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="项目目录（默认当前目录）",
)
@click.option("--no-cron", is_flag=True, help="禁用 cron 调度器")
@click.option(
    "--cors-origin",
    multiple=True,
    envvar="CREW_CORS_ORIGINS",
    help="允许的 CORS 来源（可多次指定，如 --cors-origin https://antgather.knowlyr.com）",
)
def serve(host, port, token, project_dir, no_cron, cors_origin):
    """启动 Webhook 服务器（含 Cron 调度）."""
    from crew.webhook import serve_webhook

    serve_webhook(
        host=host,
        port=port,
        project_dir=project_dir,
        token=token,
        enable_cron=not no_cron,
        cors_origins=list(cors_origin) if cors_origin else None,
    )


@click.command()
@click.option(
    "-t",
    "--transport",
    type=click.Choice(["stdio", "sse", "http"]),
    default="stdio",
    help="传输协议: stdio（默认）/ sse / http",
)
@click.option("--host", default="127.0.0.1", help="监听地址（sse/http 模式）")
@click.option("--port", default=8000, type=int, help="监听端口（sse/http 模式）")
@click.option(
    "-d",
    "--project-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="项目目录（默认当前目录）",
)
@click.option(
    "--api-token",
    default=None,
    envvar="KNOWLYR_CREW_API_TOKEN",
    help="Bearer token（未设置则不启用认证）",
)
def mcp(transport, host, port, project_dir, api_token):
    """启动 MCP Server."""
    import asyncio

    if transport == "sse":
        from crew.mcp_server import serve_sse

        asyncio.run(serve_sse(project_dir, host, port, api_token))
    elif transport == "http":
        from crew.mcp_server import serve_http

        asyncio.run(serve_http(project_dir, host, port, api_token))
    else:
        from crew.mcp_server import serve

        asyncio.run(serve(project_dir))


# ── Agent 命令组 ──


@click.group()
def agent():
    """Agent 模式 — 在 Docker 沙箱中自主执行任务."""
    pass


@agent.command(name="run")
@click.argument("employee_name")
@click.option("--task", "-t", required=True, help="任务描述")
@click.option("--model", "-m", default="claude-sonnet-4-5-20250929", help="模型 ID")
@click.option("--max-steps", default=30, type=int, help="最大步数")
@click.option("--repo", default="", help="Git 仓库 (owner/repo)")
@click.option("--base-commit", default="", help="基准 commit")
@click.option("--image", default="python:3.11-slim", help="Docker 镜像")
@click.option("--project-dir", type=click.Path(path_type=Path), default=None)
def agent_run(employee_name, task, model, max_steps, repo, base_commit, image, project_dir):
    """在 Docker 沙箱中执行员工任务.

    示例:
        knowlyr-crew agent run code-reviewer -t "审查 src/auth.py"
        knowlyr-crew agent run test-engineer -t "为 src/utils.py 补充单测" --max-steps 50
    """
    from rich.console import Console

    console = Console()

    try:
        from agentsandbox import SandboxEnv
        from agentsandbox.config import SandboxConfig, TaskConfig
    except ImportError:
        console.print("[red]knowlyr-gym 未安装。请运行: pip install knowlyr-crew[agent][/red]")
        raise SystemExit(1)

    from crew.agent_bridge import create_crew_agent

    steps_log: list[dict] = []

    def on_step(step_num, tool_name, params):
        steps_log.append({"step": step_num, "tool": tool_name, "params": params})
        param_str = ", ".join(f"{k}={v!r}" for k, v in list(params.items())[:3])
        console.print(f"  [dim]Step {step_num}:[/dim] [cyan]{tool_name}[/cyan]({param_str})")

    console.print(f"\n[bold]Agent 执行: {employee_name}[/bold]")
    console.print(f"任务: {task}")
    console.print(f"模型: {model} | 最大步数: {max_steps}\n")

    agent_fn = create_crew_agent(
        employee_name,
        task,
        model=model,
        project_dir=project_dir,
        on_step=on_step,
    )

    s_config = SandboxConfig(image=image)
    t_config = TaskConfig(repo_url=repo, base_commit=base_commit, description=task)
    env = SandboxEnv(config=s_config, task_config=t_config, max_steps=max_steps)

    try:
        ts = env.reset()
        while not ts.done:
            action = agent_fn(ts.observation)
            ts = env.step(action)
    finally:
        env.close()

    console.print()
    if ts.terminated:
        console.print(f"[green]✓ 任务完成[/green] ({len(steps_log)} 步)")
    else:
        console.print(f"[yellow]⚠ 达到最大步数[/yellow] ({max_steps})")

    if ts.observation:
        console.print(f"\n[bold]结果:[/bold]\n{ts.observation[:2000]}")
