"""数据查询工具函数 — 消息、笔记、用户、项目状态等."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from crew.webhook_context import (
    _ANTGATHER_API_TOKEN,
    _ANTGATHER_API_URL,
    _ID_API_BASE,
    _ID_API_TOKEN,
)

if TYPE_CHECKING:
    from crew.webhook_context import _AppContext

logger = logging.getLogger(__name__)


async def _tool_send_message(
    args: dict, *, agent_id: int | None = None, ctx: _AppContext | None = None
) -> str:
    """发私信 — 通过蚁聚 internal API."""
    import httpx

    sender = agent_id or args.get("sender_id", 0)
    recipient = args.get("recipient_id")
    content = args.get("content", "")

    if not _ANTGATHER_API_URL or not _ANTGATHER_API_TOKEN:
        return "发送失败: 蚁聚 API 未配置（需要 ANTGATHER_API_URL 和 ANTGATHER_API_TOKEN）"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{_ANTGATHER_API_URL}/api/internal/messages",
                json={
                    "sender_id": sender,
                    "recipient_id": recipient,
                    "content": content,
                    "msg_type": "private",
                },
                headers={"Authorization": f"Bearer {_ANTGATHER_API_TOKEN}"},
            )
            if resp.is_success:
                logger.info("send_message via antgather OK (sender=%s)", sender)
                return resp.text
            logger.error(
                "send_message via antgather error (%s): %s",
                resp.status_code,
                resp.text[:200],
            )
            return f"发送失败（HTTP {resp.status_code}）: {resp.text[:200]}"
    except httpx.HTTPError as e:
        logger.error("send_message via antgather failed: %s", e)
        return f"发送失败: {e}"


async def _tool_list_agents(
    args: dict, *, agent_id: int | None = None, ctx: _AppContext | None = None
) -> str:
    """查看所有 AI 同事的列表和当前状态（本地数据）."""
    import json

    from crew.discovery import discover_employees

    project_dir = ctx.project_dir if ctx and ctx.project_dir else None
    discovery = discover_employees(project_dir=project_dir)

    agents = []
    for name, emp in sorted(discovery.employees.items()):
        info = {
            "name": name,
            "display_name": emp.display_name or name,
            "title": emp.title or "",
            "status": emp.agent_status or "active",
            "model": emp.model or "",
            "tags": emp.tags or [],
        }
        agents.append(info)

    return json.dumps(agents, ensure_ascii=False, indent=2)


async def _tool_create_note(
    args: dict, *, agent_id: int | None = None, ctx: _AppContext | None = None
) -> str:
    """保存备忘/笔记到 .crew/notes/."""
    import re
    from datetime import datetime

    title = args.get("title", "untitled")
    content = args.get("content", "")
    tags = args.get("tags", "")
    visibility = args.get("visibility", "open")

    if not content:
        return "错误：content 不能为空"

    # 确定项目目录
    project_dir = ctx.project_dir if ctx and ctx.project_dir else Path(".")

    # sanitize filename
    safe_title = re.sub(r"[^\w\u4e00-\u9fff-]", "-", title)[:60].strip("-")
    date_str = datetime.now().strftime("%Y%m%d-%H%M")
    filename = f"{date_str}-{safe_title}.md"

    notes_dir = project_dir / ".crew" / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    # frontmatter + content
    lines = [
        "---",
        f"title: {title}",
        f"date: {datetime.now().isoformat()}",
    ]
    if tags:
        lines.append(f"tags: [{tags}]")
    if visibility == "private":
        lines.append("visibility: private")
    lines.extend(["---", "", content])

    note_path = notes_dir / filename
    note_path.write_text("\n".join(lines), encoding="utf-8")
    return f"笔记已保存: {filename}"


# ── 过渡期：人类数据仍在 knowlyr-id（id 管人类） ──


async def _tool_lookup_user(
    args: dict, *, agent_id: int | None = None, ctx: _AppContext | None = None
) -> str:
    """按昵称查用户详情（过渡期走 knowlyr-id）."""
    import httpx

    name = args.get("name", "")
    if not name:
        return "错误：需要 name 参数"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{_ID_API_BASE}/api/stats/user",
                params={"q": name},
                headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
            )
            return resp.text
    except httpx.HTTPError as e:
        return f"查询失败: {e}"


async def _tool_read_notes(
    args: dict, *, agent_id: int | None = None, ctx: _AppContext | None = None
) -> str:
    """列出最近笔记，可选按关键词过滤."""
    keyword = args.get("keyword", "")
    limit = min(args.get("limit", 10), 20)
    max_visibility = args.get("_max_visibility", "open")

    notes_dir = (ctx.project_dir if ctx and ctx.project_dir else Path(".")) / ".crew" / "notes"
    if not notes_dir.exists():
        return "暂无笔记"

    files = sorted(notes_dir.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
    results = []
    for f in files:
        if len(results) >= limit:
            break
        content = f.read_text(encoding="utf-8")
        # 可见性过滤: max_visibility="open" 时跳过 private 笔记
        if max_visibility != "private" and "visibility: private" in content:
            continue
        if keyword and keyword.lower() not in content.lower():
            continue
        results.append(f"【{f.stem}】\n{content[:200]}")

    return "\n---\n".join(results) if results else "没有匹配的笔记"


# 本地路径（Mac 开发机）和服务器路径
_PROJECT_STATUS_PATHS = [
    Path.home() / ".claude/projects/-Users-liukai/memory/project-status.md",
    Path("/opt/knowlyr-crew/data/project-status.md"),
]
_PROJECT_STATUS_SCRIPT = Path.home() / ".claude/scripts/project-status.sh"


def _find_report() -> Path | None:
    """按优先级查找报告文件."""
    for p in _PROJECT_STATUS_PATHS:
        if p.exists():
            return p
    return None


async def _tool_project_status(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """查询 knowlyr 项目状态."""
    import asyncio
    import os
    from datetime import datetime

    refresh = args.get("refresh", False)

    # refresh 只在本地可用（需要 git 仓库和脚本）
    if refresh:
        if not _PROJECT_STATUS_SCRIPT.exists():
            return "refresh 仅在本地开发机可用（服务器无项目仓库）。去掉 refresh 可读取缓存报告。"
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash",
                str(_PROJECT_STATUS_SCRIPT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            if proc.returncode != 0:
                return f"脚本执行失败 (exit {proc.returncode}): {stderr.decode()[:500]}"
        except asyncio.TimeoutError:
            return "脚本执行超时（60秒）。请稍后重试。"
        except Exception as e:
            return f"脚本执行失败: {e}"

    report = _find_report()
    if report is None:
        if not refresh and _PROJECT_STATUS_SCRIPT.exists():
            return await _tool_project_status(
                {"refresh": True},
                agent_id=agent_id,
                ctx=ctx,
            )
        return "报告文件不存在。请在本地运行 ~/.claude/scripts/project-status.sh 生成。"

    try:
        content = report.read_text(encoding="utf-8")
        if len(content) > 9500:
            content = content[:9500] + "\n\n[已截断]"
        mtime = os.path.getmtime(report)
        cache_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        suffix = "（刚刷新）" if refresh else f"（缓存于 {cache_time}）"
        return f"{content}\n\n---\n{suffix}"
    except Exception as e:
        return f"读取报告失败: {e}"


HANDLERS: dict[str, object] = {
    "send_message": _tool_send_message,
    "list_agents": _tool_list_agents,
    "create_note": _tool_create_note,
    "lookup_user": _tool_lookup_user,
    "read_notes": _tool_read_notes,
    "project_status": _tool_project_status,
}
