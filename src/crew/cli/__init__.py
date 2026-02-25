"""Crew CLI — 命令行界面."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from crew import __version__
from crew.session_recorder import SessionRecorder
from crew.session_summary import SessionMemoryWriter

logger = logging.getLogger(__name__)

EMPLOYEE_SUBDIR = Path("private") / "employees"


def _suggest_similar(name: str, candidates: list[str]) -> str:
    """查找相似名称，返回提示文本."""
    import difflib

    close = difflib.get_close_matches(name, candidates, n=3, cutoff=0.5)
    if close:
        return f"\n类似的名称: {', '.join(close)}"
    return ""


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
    except Exception as e:
        logger.debug("记录 transcript 事件失败: %s", e)


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
    except Exception as e:
        logger.debug("记录 transcript 消息失败: %s", e)


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
    except Exception as e:
        logger.debug("结束 transcript 失败: %s", e)


def _record_session_summary(
    *,
    employee: str,
    session_id: str | None,
) -> None:
    try:
        SessionMemoryWriter().capture(
            employee=employee,
            session_id=session_id,
        )
    except Exception as e:
        logger.debug("采集 session 摘要失败: %s", e)


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


# ── 注册子模块命令 ──
# 使用延迟 import 避免在顶层加载所有子模块

def _register_commands() -> None:
    """注册所有子模块的命令到 main group."""
    from crew.cli.employee import (
        avatar,
        catalog,
        check_cmd,
        delete,
        init,
        lint_cmd,
        list_cmd,
        permissions_cmd,
        rollback,
        run,
        show,
        validate,
    )
    from crew.cli.pipeline import pipeline
    from crew.cli.route import route
    from crew.cli.discuss import changelog_draft, discuss, meetings
    from crew.cli.memory import memory
    from crew.cli.server import agent, mcp, serve
    from crew.cli.ops import (
        agents,
        cron_group,
        eval_group,
        export_all_cmd,
        export_cmd,
        log,
        session_group,
        sync,
        template,
        trajectory,
    )

    # employee.py 的命令
    main.add_command(list_cmd, "list")
    main.add_command(show)
    main.add_command(run)
    main.add_command(validate)
    main.add_command(init)
    main.add_command(delete)
    main.add_command(avatar)
    main.add_command(rollback)
    main.add_command(catalog)
    main.add_command(permissions_cmd, "permissions")
    main.add_command(lint_cmd, "lint")
    main.add_command(check_cmd, "check")

    # pipeline.py
    main.add_command(pipeline)

    # route.py
    main.add_command(route)

    # discuss.py
    main.add_command(discuss)
    main.add_command(meetings)
    main.add_command(changelog_draft, "changelog")

    # memory.py
    main.add_command(memory)

    # server.py
    main.add_command(serve)
    main.add_command(mcp)
    main.add_command(agent)

    # ops.py
    main.add_command(agents)
    main.add_command(cron_group, "cron")
    main.add_command(eval_group, "eval")
    main.add_command(export_cmd, "export")
    main.add_command(export_all_cmd, "export-all")
    main.add_command(sync)
    main.add_command(log)
    main.add_command(session_group, "session")
    main.add_command(template)
    main.add_command(trajectory)


_register_commands()
