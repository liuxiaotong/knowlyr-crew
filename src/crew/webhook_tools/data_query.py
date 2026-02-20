"""数据查询工具函数 — 统计、消息、笔记、用户、系统健康等."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from crew.webhook_context import _ID_API_BASE, _ID_API_TOKEN

if TYPE_CHECKING:
    from crew.webhook_context import _AppContext


async def _tool_query_stats(
    args: dict, *, agent_id: int | None = None, ctx: _AppContext | None = None
) -> str:
    """调用 knowlyr-id /api/stats/briefing."""
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_ID_API_BASE}/api/stats/briefing",
            headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
        )
        return resp.text


async def _tool_send_message(
    args: dict, *, agent_id: int | None = None, ctx: _AppContext | None = None
) -> str:
    """调用 knowlyr-id /api/messages/agent-send."""
    import httpx

    sender = agent_id or args.get("sender_id", 0)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{_ID_API_BASE}/api/messages/agent-send",
            json={
                "sender_id": sender,
                "recipient_id": args.get("recipient_id"),
                "content": args.get("content", ""),
            },
            headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
        )
        return resp.text


async def _tool_list_agents(
    args: dict, *, agent_id: int | None = None, ctx: _AppContext | None = None
) -> str:
    """调用 knowlyr-id /api/agents."""
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_ID_API_BASE}/api/agents",
            headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
        )
        return resp.text


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


async def _tool_lookup_user(
    args: dict, *, agent_id: int | None = None, ctx: _AppContext | None = None
) -> str:
    """按昵称查用户详情."""
    import httpx

    name = args.get("name", "")
    if not name:
        return "错误：需要 name 参数"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_ID_API_BASE}/api/stats/user",
            params={"q": name},
            headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
        )
        return resp.text


async def _tool_query_agent_work(
    args: dict, *, agent_id: int | None = None, ctx: _AppContext | None = None
) -> str:
    """查 AI 同事最近工作记录."""
    import httpx

    name = args.get("name", "")
    days = args.get("days", 3)
    if not name:
        return "错误：需要 name 参数"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_ID_API_BASE}/api/stats/agent-work",
            params={"name": name, "days": days},
            headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
        )
        return resp.text


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


async def _tool_read_messages(
    args: dict, *, agent_id: int | None = None, ctx: _AppContext | None = None
) -> str:
    """查 Kai 的未读消息概要."""
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_ID_API_BASE}/api/stats/unread",
            params={"user_id": 1},
            headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
        )
        return resp.text


async def _tool_get_system_health(
    args: dict, *, agent_id: int | None = None, ctx: _AppContext | None = None
) -> str:
    """查服务器健康状态."""
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_ID_API_BASE}/api/stats/system-health",
            headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
        )
        return resp.text


async def _tool_mark_read(
    args: dict, *, agent_id: int | None = None, ctx: _AppContext | None = None
) -> str:
    """标消息已读."""
    import httpx

    mark_all = args.get("all", False)
    sender_name = args.get("sender_name", "")

    payload: dict[str, Any] = {"user_id": 1}

    if mark_all:
        payload["all"] = True
    elif sender_name:
        # 先查 sender_id
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{_ID_API_BASE}/api/stats/user",
                params={"q": sender_name},
                headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
            )
            try:
                users = resp.json()
                if not users:
                    return f"找不到用户「{sender_name}」"
                payload["sender_id"] = users[0]["id"]
            except Exception:
                return f"查询用户失败: {resp.text[:200]}"
    else:
        return "需要指定 sender_name 或 all=true"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{_ID_API_BASE}/api/messages/mark-read-batch",
            json=payload,
            headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
        )
        return resp.text


async def _tool_update_agent(
    args: dict, *, agent_id: int | None = None, ctx: _AppContext | None = None
) -> str:
    """管理 AI 同事."""
    import httpx

    target_id = args.get("agent_id")
    if not target_id:
        return "需要 agent_id 参数"

    update_data: dict[str, Any] = {}
    if args.get("status"):
        update_data["agent_status"] = args["status"]
    if args.get("memory"):
        update_data["memory"] = args["memory"]

    if not update_data:
        return "没有要更新的内容（需要 status 或 memory）"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.put(
            f"{_ID_API_BASE}/api/agents/{target_id}",
            json=update_data,
            headers={"Authorization": f"Bearer {_ID_API_TOKEN}"},
        )
        return resp.text


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
    "query_stats": _tool_query_stats,
    "send_message": _tool_send_message,
    "list_agents": _tool_list_agents,
    "create_note": _tool_create_note,
    "lookup_user": _tool_lookup_user,
    "query_agent_work": _tool_query_agent_work,
    "read_notes": _tool_read_notes,
    "read_messages": _tool_read_messages,
    "get_system_health": _tool_get_system_health,
    "project_status": _tool_project_status,
    "mark_read": _tool_mark_read,
    "update_agent": _tool_update_agent,
}
