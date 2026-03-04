"""飞书工具函数 — 日历、任务、文档、表格、群聊、审批等."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from crew.feishu import get_feishu_client

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from crew.webhook_context import _AppContext


async def _tool_create_feishu_event(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """在飞书日历创建日程."""
    from datetime import datetime, timedelta
    from datetime import timezone as _tz

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置，无法创建日程。"

    summary = (args.get("summary") or "").strip()
    date_str = (args.get("date") or "").strip()
    start_hour = args.get("start_hour", 9)
    start_minute = args.get("start_minute", 0)
    duration = args.get("duration_minutes", 60)
    description = args.get("description", "")

    if not summary:
        return "缺少日程标题。"
    if not date_str:
        return "缺少日期。"

    try:
        start_hour = int(start_hour)
        start_minute = int(start_minute)
        duration = int(duration)
    except (TypeError, ValueError):
        return "时间参数格式不对。"

    tz_cn = _tz(timedelta(hours=8))
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        start_time = d.replace(hour=start_hour, minute=start_minute, tzinfo=tz_cn)
    except ValueError:
        return f"日期格式不对: {date_str}，需要 YYYY-MM-DD。"

    end_time = start_time + timedelta(minutes=max(duration, 15))

    from crew.feishu import add_attendees_to_event, create_calendar_event

    cal_id = (ctx.feishu_config.calendar_id if ctx.feishu_config else "") or ""
    if not cal_id:
        logger.warning("日历创建: calendar_id 为空，将依赖环境变量 FEISHU_CALENDAR_ID")
    result = await create_calendar_event(
        token_mgr=ctx.feishu_token_mgr,
        summary=summary,
        start_timestamp=int(start_time.timestamp()),
        end_timestamp=int(end_time.timestamp()),
        description=description,
        calendar_id=cal_id,
    )

    if result.get("ok"):
        event_id = result.get("event_id", "")
        logger.info("日历日程创建成功: event_id=%s summary=%s", event_id, summary)
        # 自动邀请日历所有者（让日程出现在他/她的日历上）
        owner_id = (ctx.feishu_config.owner_open_id if ctx.feishu_config else "") or ""
        if not owner_id:
            logger.warning("日历创建: owner_open_id 为空，日程不会自动邀请所有者")
        if owner_id and event_id and cal_id:
            att_result = await add_attendees_to_event(
                token_mgr=ctx.feishu_token_mgr,
                calendar_id=cal_id,
                event_id=event_id,
                attendee_open_ids=[owner_id],
            )
            if not att_result.get("ok"):
                logger.warning("日程创建成功但邀请参与者失败: %s", att_result.get("error"))
        end_str = end_time.strftime("%H:%M")
        start_str = start_time.strftime("%H:%M")
        return f"日程已创建：{date_str} {start_str}-{end_str}《{summary}》"
    else:
        return f"创建失败: {result.get('error', '未知错误')}"


# ── 飞书日程查询/删除 ──


async def _tool_read_feishu_calendar(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """查看飞书日历日程."""
    import os
    from datetime import datetime, timedelta
    from datetime import timezone as _tz

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    tz_cn = _tz(timedelta(hours=8))
    date_str = (args.get("date") or "").strip()
    days = max(int(args.get("days", 1)), 1)

    if date_str:
        try:
            start_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=tz_cn)
        except ValueError:
            return f"日期格式不对: {date_str}，需要 YYYY-MM-DD。"
    else:
        start_dt = datetime.now(tz_cn).replace(hour=0, minute=0, second=0, microsecond=0)

    end_dt = start_dt + timedelta(days=days)

    cal_id = (ctx.feishu_config.calendar_id if ctx.feishu_config else "") or os.environ.get(
        "FEISHU_CALENDAR_ID", ""
    )
    if not cal_id:
        return "未配置日历 ID。"

    token = await ctx.feishu_token_mgr.get_token()
    try:
        client = get_feishu_client()
        resp = await client.get(
            f"https://open.feishu.cn/open-apis/calendar/v4/calendars/{cal_id}/events",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "start_time": str(int(start_dt.timestamp())),
                "end_time": str(int(end_dt.timestamp())),
                "page_size": 50,
            },
            timeout=15.0,
        )
        data = resp.json()

        if data.get("code") != 0:
            return f"查询失败: {data.get('msg', '未知错误')}"

        items = data.get("data", {}).get("items", [])
        if not items:
            date_range = date_str or start_dt.strftime("%Y-%m-%d")
            if days > 1:
                date_range += f" ~ {end_dt.strftime('%Y-%m-%d')}"
            return f"{date_range} 没有日程。"

        lines = []
        for ev in items:
            summary = ev.get("summary", "无标题")
            event_id = ev.get("event_id", "")
            st = ev.get("start_time", {})
            et = ev.get("end_time", {})
            # timestamp or date
            if st.get("timestamp"):
                s = datetime.fromtimestamp(int(st["timestamp"]), tz=tz_cn)
                e = datetime.fromtimestamp(int(et.get("timestamp", st["timestamp"])), tz=tz_cn)
                time_str = f"{s.strftime('%m-%d %H:%M')}-{e.strftime('%H:%M')}"
            elif st.get("date"):
                time_str = f"{st['date']} 全天"
            else:
                time_str = "时间未知"
            lines.append(f"{time_str} {summary} [event_id={event_id}]")

        return "\n".join(lines)
    except Exception as e:
        return f"查询失败: {e}"


async def _tool_delete_feishu_event(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """删除飞书日历日程."""
    import os

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    event_id = (args.get("event_id") or "").strip()
    if not event_id:
        return "需要 event_id 参数。先用 read_feishu_calendar 查到 event_id。"

    cal_id = (ctx.feishu_config.calendar_id if ctx.feishu_config else "") or os.environ.get(
        "FEISHU_CALENDAR_ID", ""
    )
    if not cal_id:
        return "未配置日历 ID。"

    token = await ctx.feishu_token_mgr.get_token()
    try:
        client = get_feishu_client()
        resp = await client.delete(
            f"https://open.feishu.cn/open-apis/calendar/v4/calendars/{cal_id}/events/{event_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
        data = resp.json()

        if data.get("code") == 0:
            return f"日程已删除 (event_id={event_id})。"
        return f"删除失败: {data.get('msg', '未知错误')}"
    except Exception as e:
        return f"删除失败: {e}"


# ── 飞书待办任务 ──


async def _tool_create_feishu_task(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """在飞书创建待办任务."""
    from datetime import datetime, timedelta
    from datetime import timezone as _tz

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    summary = (args.get("summary") or "").strip()
    if not summary:
        return "需要任务标题。"

    due_str = (args.get("due") or "").strip()
    description = args.get("description", "")

    body: dict[str, Any] = {"summary": summary}  # noqa: F821
    if description:
        body["description"] = description
    if due_str:
        tz_cn = _tz(timedelta(hours=8))
        try:
            due_dt = datetime.strptime(due_str, "%Y-%m-%d").replace(
                hour=23, minute=59, tzinfo=tz_cn
            )
            body["due"] = {"timestamp": str(int(due_dt.timestamp())), "is_all_day": True}
        except ValueError:
            return f"截止日期格式不对: {due_str}，需要 YYYY-MM-DD。"

    token = await ctx.feishu_token_mgr.get_token()
    try:
        client = get_feishu_client()
        resp = await client.post(
            "https://open.feishu.cn/open-apis/task/v2/tasks",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=body,
            timeout=15.0,
        )
        data = resp.json()

        if data.get("code") == 0:
            task = data.get("data", {}).get("task", {})
            task_id = task.get("guid", "")
            due_info = f"，截止 {due_str}" if due_str else ""
            return f"待办已创建：{summary}{due_info} [task_id={task_id}]"
        return f"创建失败: {data.get('msg', '未知错误')}"
    except Exception as e:
        return f"创建失败: {e}"


async def _tool_list_feishu_tasks(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """查看飞书待办任务列表."""
    from datetime import datetime, timedelta
    from datetime import timezone as _tz

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    limit = min(int(args.get("limit", 20)), 50)

    token = await ctx.feishu_token_mgr.get_token()
    try:
        client = get_feishu_client()
        resp = await client.get(
            "https://open.feishu.cn/open-apis/task/v2/tasks",
            headers={"Authorization": f"Bearer {token}"},
            params={"page_size": limit},
            timeout=15.0,
        )
        data = resp.json()

        if data.get("code") != 0:
            return f"查询失败: {data.get('msg', '未知错误')}"

        items = data.get("data", {}).get("items", [])
        if not items:
            return "没有待办任务。"

        tz_cn = _tz(timedelta(hours=8))
        lines = []
        for task in items:
            summary = task.get("summary", "无标题")
            completed_at = task.get("completed_at", "0")
            status = "✅" if completed_at and completed_at != "0" else "⬜"
            due = task.get("due", {})
            due_str = ""
            if due and due.get("timestamp") and due["timestamp"] != "0":
                due_dt = datetime.fromtimestamp(int(due["timestamp"]), tz=tz_cn)
                due_str = f" 截止{due_dt.strftime('%m-%d')}"
            task_id = task.get("guid", "")
            lines.append(f"{status} {summary}{due_str} [task_id={task_id}]")

        return "\n".join(lines)
    except Exception as e:
        return f"查询失败: {e}"


async def _tool_complete_feishu_task(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """完成飞书待办任务."""

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    task_id = (args.get("task_id") or "").strip()
    if not task_id:
        return "需要 task_id。先用 list_feishu_tasks 查看任务列表。"

    token = await ctx.feishu_token_mgr.get_token()
    try:
        client = get_feishu_client()
        resp = await client.post(
            f"https://open.feishu.cn/open-apis/task/v2/tasks/{task_id}/complete",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
        data = resp.json()

        if data.get("code") == 0:
            return f"任务已完成 ✅ [task_id={task_id}]"
        return f"操作失败: {data.get('msg', '未知错误')}"
    except Exception as e:
        return f"操作失败: {e}"


async def _tool_delete_feishu_task(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """删除飞书待办任务."""

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    task_id = (args.get("task_id") or "").strip()
    if not task_id:
        return "需要 task_id。先用 list_feishu_tasks 查看任务列表。"

    token = await ctx.feishu_token_mgr.get_token()
    try:
        client = get_feishu_client()
        resp = await client.delete(
            f"https://open.feishu.cn/open-apis/task/v2/tasks/{task_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15.0,
        )
        data = resp.json()

        if data.get("code") == 0:
            return f"任务已删除 [task_id={task_id}]"
        return f"删除失败: {data.get('msg', '未知错误')}"
    except Exception as e:
        return f"删除失败: {e}"


async def _tool_update_feishu_task(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """更新飞书待办任务."""
    from datetime import datetime, timedelta
    from datetime import timezone as _tz

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    task_id = (args.get("task_id") or "").strip()
    if not task_id:
        return "需要 task_id。先用 list_feishu_tasks 查看任务列表。"

    summary = (args.get("summary") or "").strip()
    due_str = (args.get("due") or "").strip()
    description = args.get("description")

    body: dict[str, Any] = {}  # noqa: F821
    update_fields: list[str] = []
    if summary:
        body["summary"] = summary
        update_fields.append("summary")
    if description is not None and description != "":
        body["description"] = description
        update_fields.append("description")
    if due_str:
        tz_cn = _tz(timedelta(hours=8))
        try:
            due_dt = datetime.strptime(due_str, "%Y-%m-%d").replace(
                hour=23, minute=59, tzinfo=tz_cn
            )
            body["due"] = {"timestamp": str(int(due_dt.timestamp())), "is_all_day": True}
            update_fields.append("due")
        except ValueError:
            return f"截止日期格式不对: {due_str}，需要 YYYY-MM-DD。"

    if not body:
        return "需要至少一个要更新的字段（summary/due/description）。"

    token = await ctx.feishu_token_mgr.get_token()
    try:
        client = get_feishu_client()
        resp = await client.patch(
            f"https://open.feishu.cn/open-apis/task/v2/tasks/{task_id}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            params={"update_fields": ",".join(update_fields)},
            json=body,
            timeout=15.0,
        )
        data = resp.json()

        if data.get("code") == 0:
            parts = []
            if summary:
                parts.append(f"标题→{summary}")
            if due_str:
                parts.append(f"截止→{due_str}")
            if description is not None and description != "":
                parts.append("描述已更新")
            return f"任务已更新: {', '.join(parts)} [task_id={task_id}]"
        return f"更新失败: {data.get('msg', '未知错误')}"
    except Exception as e:
        return f"更新失败: {e}"


async def _tool_feishu_chat_history(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """读取飞书群/会话最近消息."""
    import json as _json
    from datetime import datetime, timedelta
    from datetime import timezone as _tz

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    chat_id = (args.get("chat_id") or "").strip()
    if not chat_id:
        return "需要 chat_id。先用 list_feishu_groups 查群列表。"

    limit = min(int(args.get("limit", 10)), 50)
    token = await ctx.feishu_token_mgr.get_token()
    tz_cn = _tz(timedelta(hours=8))
    try:
        client = get_feishu_client()
        resp = await client.get(
            "https://open.feishu.cn/open-apis/im/v1/messages",
            headers={"Authorization": f"Bearer {token}"},
            params={
                "container_id_type": "chat",
                "container_id": chat_id,
                "page_size": limit,
                "sort_type": "ByCreateTimeDesc",
            },
            timeout=15.0,
        )
        data = resp.json()

        if data.get("code") != 0:
            return f"查询失败: {data.get('msg', '未知错误')}"

        items = data.get("data", {}).get("items", [])
        if not items:
            return "没有消息记录。"

        lines = []
        for msg in items:
            sender = msg.get("sender", {}).get("id", "?")
            msg_type = msg.get("msg_type", "text")
            create_time = msg.get("create_time", "")
            time_str = ""
            if create_time:
                try:
                    dt = datetime.fromtimestamp(int(create_time) / 1000, tz=tz_cn)
                    time_str = dt.strftime("%m-%d %H:%M")
                except (ValueError, OSError):
                    pass

            body = msg.get("body", {}).get("content", "")
            try:
                content_obj = _json.loads(body) if body else {}
                text = content_obj.get("text", "")
            except _json.JSONDecodeError:
                text = body

            if msg_type == "image":
                text = "[图片]"
            elif msg_type == "file":
                text = "[文件]"
            elif msg_type == "sticker":
                text = "[表情]"
            elif not text:
                text = f"[{msg_type}]"

            # 截断长消息
            if len(text) > 100:
                text = text[:100] + "…"
            lines.append(f"[{time_str}] {sender}: {text}")

        return "\n".join(lines)
    except Exception as e:
        return f"查询失败: {e}"


# ── 天气工具 ──

# 城市代码已移至 _constants.py


async def _tool_search_feishu_docs(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """搜索飞书云文档."""

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置，无法搜索文档。"

    query = (args.get("query") or "").strip()
    count = min(args.get("count", 10), 20)
    if not query:
        return "搜索关键词不能为空。"

    token = await ctx.feishu_token_mgr.get_token()
    client = get_feishu_client()
    resp = await client.post(
        "https://open.feishu.cn/open-apis/suite/docs-api/search/object",
        json={"query": query, "count": count, "offset": 0},
        headers={"Authorization": f"Bearer {token}"},
    )
    data = resp.json()
    if not data.get("data", {}).get("docs_entities"):
        return "没有找到匹配的文档。"

    lines = []
    for doc in data["data"]["docs_entities"][:count]:
        title = doc.get("title", "无标题")
        url = doc.get("url", "")
        doc_type = doc.get("docs_type", "")
        lines.append(f"[{doc_type}] {title}\n{url}")
    return "\n---\n".join(lines)


async def _tool_read_feishu_doc(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """读取飞书文档内容."""

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置，无法读取文档。"

    doc_id = (args.get("document_id") or "").strip()
    if not doc_id:
        return "缺少 document_id。"

    token = await ctx.feishu_token_mgr.get_token()
    client = get_feishu_client()
    resp = await client.get(
        f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_id}/raw_content",
        headers={"Authorization": f"Bearer {token}"},
    )
    data = resp.json()
    content = data.get("data", {}).get("content", "")
    if not content:
        return f"文档 {doc_id} 内容为空或无权限访问。"
    if len(content) > 9500:
        return content[:9500] + f"\n\n[内容已截断，共 {len(content)} 字符]"
    return content


async def _tool_create_feishu_doc(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """在飞书创建新文档."""

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置，无法创建文档。"

    title = (args.get("title") or "").strip()
    content = args.get("content", "")
    folder_token = args.get("folder_token", "")
    if not title:
        return "缺少文档标题。"

    token = await ctx.feishu_token_mgr.get_token()
    create_body: dict = {"title": title}
    if folder_token:
        create_body["folder_token"] = folder_token

    client = get_feishu_client()
    resp = await client.post(
        "https://open.feishu.cn/open-apis/docx/v1/documents",
        json=create_body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    data = resp.json()
    doc_id = data.get("data", {}).get("document", {}).get("document_id", "")
    if not doc_id:
        return f"创建文档失败: {data.get('msg', '未知错误')}"

    if content:
        await client.post(
            f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_id}/blocks/{doc_id}/children",
            json={
                "children": [
                    {
                        "block_type": 2,
                        "text": {"elements": [{"text_run": {"content": content}}]},
                    }
                ],
            },
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )

    url = f"https://feishu.cn/docx/{doc_id}"
    return f"文档已创建：{title}\n{url}"


_URL_RE = re.compile(r"https?://\S+")


async def _fetch_og_meta(url: str) -> dict[str, str]:
    """抓取页面 Open Graph 元数据（title / description / image）."""
    meta: dict[str, str] = {}
    try:
        client = get_feishu_client()
        resp = await client.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5.0,
            follow_redirects=True,
        )
        html = resp.text[:20_000]  # 只看前 20k，够提取 meta 了

        # og:title / og:description / og:image
        for prop in ("title", "description", "image"):
            m = re.search(
                rf'<meta\s[^>]*property=["\']og:{prop}["\'][^>]*content=["\']([^"\']+)["\']',
                html,
                re.IGNORECASE,
            )
            if not m:
                m = re.search(
                    rf'<meta\s[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:{prop}["\']',
                    html,
                    re.IGNORECASE,
                )
            if m:
                meta[prop] = m.group(1).strip()

        # 兜底：用 <title> 标签
        if "title" not in meta:
            m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
            if m:
                meta["title"] = m.group(1).strip()
    except Exception:
        pass
    return meta


def _build_link_card(url: str, text: str, og: dict[str, str] | None = None) -> dict:
    """根据链接和 OG 元数据构建飞书卡片."""
    og = og or {}
    is_feishu = "feishu.cn/" in url

    # 标题优先级：消息附带文字 > og:title > 兜底
    user_title = text.replace(url, "").strip().strip("：:—\n")
    title = (
        user_title
        or og.get("title")
        or ("飞书文档" if is_feishu else url.split("//")[-1].split("?")[0])
    )
    description = og.get("description", "")

    btn_label = "打开文档" if is_feishu else "打开链接"
    color = "blue" if is_feishu else "grey"

    elements: list[dict] = []
    if description:
        elements.append(
            {
                "tag": "div",
                "text": {"tag": "plain_text", "content": description[:200]},
            }
        )
    elements.append(
        {
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": btn_label},
                    "url": url,
                    "type": "primary",
                },
            ],
        }
    )

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": color,
        },
        "elements": elements,
    }


async def _tool_send_feishu_group(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """发消息到飞书群（含链接时自动发卡片，带页面预览）."""
    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置，无法发群消息。"

    chat_id = (args.get("chat_id") or "").strip()
    text = (args.get("text") or "").strip()
    if not chat_id or not text:
        return "需要 chat_id 和 text。"

    from crew.feishu import send_feishu_message, send_feishu_text

    # 检测链接 — 有则抓取页面元数据并发卡片，无则发纯文本
    m = _URL_RE.search(text)
    if m:
        url = m.group(0)
        og = await _fetch_og_meta(url)
        card = _build_link_card(url, text, og)
        result = await send_feishu_message(
            ctx.feishu_token_mgr,
            chat_id,
            content=card,
            msg_type="interactive",
        )
    else:
        result = await send_feishu_text(ctx.feishu_token_mgr, chat_id, text)

    if result.get("code") == 0 or result.get("ok"):
        return f"消息已发送到群 {chat_id}。"
    return f"发送失败: {result.get('msg') or result.get('error', '未知错误')}"


async def _tool_send_feishu_file(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """上传文件并发送到飞书群."""
    import json as _json

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置，无法发文件。"

    chat_id = (args.get("chat_id") or "").strip()
    file_name = (args.get("file_name") or "").strip()
    content = args.get("content", "")
    if not chat_id or not file_name or not content:
        return "需要 chat_id、file_name 和 content。"

    token = await ctx.feishu_token_mgr.get_token()
    base = "https://open.feishu.cn/open-apis"

    try:
        client = get_feishu_client()
        # 1. 上传文件
        resp = await client.post(
            f"{base}/im/v1/files",
            headers={"Authorization": f"Bearer {token}"},
            data={"file_type": "stream", "file_name": file_name},
            files={"file": (file_name, content.encode("utf-8"))},
        )
        data = resp.json()
        if data.get("code") != 0:
            return f"文件上传失败: {data.get('msg', '未知错误')}"

        file_key = data.get("data", {}).get("file_key", "")
        if not file_key:
            return "文件上传成功但未返回 file_key。"

        # 2. 发送文件消息到群
        resp = await client.post(
            f"{base}/im/v1/messages",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            params={"receive_id_type": "chat_id"},
            json={
                "receive_id": chat_id,
                "msg_type": "file",
                "content": _json.dumps({"file_key": file_key}),
            },
        )
        data = resp.json()

        if data.get("code") == 0:
            return f"文件 {file_name} 已发送到群 {chat_id}。"
        return f"文件发送失败: {data.get('msg', '未知错误')}"
    except Exception as e:
        return f"发送文件失败: {e}"


async def _tool_list_feishu_groups(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """列出机器人加入的所有飞书群."""

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    token = await ctx.feishu_token_mgr.get_token()
    try:
        client = get_feishu_client()
        resp = await client.get(
            "https://open.feishu.cn/open-apis/im/v1/chats",
            headers={"Authorization": f"Bearer {token}"},
            params={"page_size": 50},
            timeout=15.0,
        )
        data = resp.json()
        if data.get("code") != 0:
            return f"查询失败: {data.get('msg', '未知错误')}"
        items = data.get("data", {}).get("items", [])
        if not items:
            return "机器人没有加入任何群。"
        lines = []
        for item in items:
            name = item.get("name", "未命名")
            chat_id = item.get("chat_id", "")
            lines.append(f"{name} — {chat_id}")
        return "\n".join(lines)
    except Exception as e:
        return f"查询失败: {e}"


async def _tool_send_feishu_dm(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """给飞书用户发私聊消息."""
    import json as _json

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    open_id = (args.get("open_id") or "").strip()
    text = (args.get("text") or "").strip()
    if not open_id:
        return "需要 open_id。先用 feishu_group_members 查成员列表获取 open_id。"
    if not text:
        return "消息内容不能为空。"

    token = await ctx.feishu_token_mgr.get_token()
    try:
        client = get_feishu_client()
        resp = await client.post(
            "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "receive_id": open_id,
                "msg_type": "text",
                "content": _json.dumps({"text": text}),
            },
            timeout=15.0,
        )
        data = resp.json()

        if data.get("code") == 0:
            return f"私聊消息已发送给 {open_id}。"
        return f"发送失败: {data.get('msg', '未知错误')}"
    except Exception as e:
        return f"发送失败: {e}"


async def _tool_feishu_group_members(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """查看飞书群成员列表."""

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    chat_id = (args.get("chat_id") or "").strip()
    if not chat_id:
        return "需要 chat_id。先用 list_feishu_groups 查群列表。"

    token = await ctx.feishu_token_mgr.get_token()
    try:
        client = get_feishu_client()
        resp = await client.get(
            f"https://open.feishu.cn/open-apis/im/v1/chats/{chat_id}/members",
            headers={"Authorization": f"Bearer {token}"},
            params={"page_size": 50},
            timeout=15.0,
        )
        data = resp.json()

        if data.get("code") != 0:
            return f"查询失败: {data.get('msg', '未知错误')}"

        items = data.get("data", {}).get("items", [])
        if not items:
            return "群里没有成员（或无权限查看）。"

        lines = []
        for m in items:
            name = m.get("name", "未知")
            mid = m.get("member_id", "")
            lines.append(f"{name} [open_id={mid}]")
        return "\n".join(lines)
    except Exception as e:
        return f"查询失败: {e}"


# ── GitHub 工具 ──


async def _tool_read_feishu_sheet(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """读取飞书表格数据."""

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    ss_token = (args.get("spreadsheet_token") or "").strip()
    if not ss_token:
        return "缺少 spreadsheet_token。"

    sheet_id = (args.get("sheet_id") or "").strip()
    range_str = (args.get("range") or "A1:Z100").strip()

    token = await ctx.feishu_token_mgr.get_token()
    headers = {"Authorization": f"Bearer {token}"}
    base = "https://open.feishu.cn/open-apis"

    try:
        client = get_feishu_client()
        if not sheet_id:
            meta_resp = await client.get(
                f"{base}/sheets/v3/spreadsheets/{ss_token}/sheets/query",
                headers=headers,
            )
            meta = meta_resp.json()
            sheets = meta.get("data", {}).get("sheets", [])
            if not sheets:
                return "该表格没有工作表。"
            sheet_id = sheets[0].get("sheet_id", "")

        full_range = f"{sheet_id}!{range_str}"
        resp = await client.get(
            f"{base}/sheets/v2/spreadsheets/{ss_token}/values/{full_range}",
            headers=headers,
            params={"valueRenderOption": "ToString"},
        )
        data = resp.json()

        if data.get("code") != 0:
            return f"读取失败: {data.get('msg', '未知错误')}"

        values = data.get("data", {}).get("valueRange", {}).get("values", [])
        if not values:
            return "表格为空或指定范围无数据。"

        lines = []
        for i, row in enumerate(values[:100]):
            cells = [str(c) if c is not None else "" for c in row]
            lines.append(" | ".join(cells))
            if i == 0:
                lines.append("-" * min(len(lines[0]), 80))
        result = "\n".join(lines)
        if len(result) > 9500:
            result = result[:9500] + "\n\n[数据已截断]"
        return result
    except Exception as e:
        return f"读取表格失败: {e}"


async def _tool_update_feishu_sheet(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """写入飞书表格数据."""
    import json as _json

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    ss_token = (args.get("spreadsheet_token") or "").strip()
    range_str = (args.get("range") or "").strip()
    values_str = (args.get("values") or "").strip()
    sheet_id = (args.get("sheet_id") or "").strip()

    if not ss_token:
        return "缺少 spreadsheet_token。"
    if not range_str:
        return "缺少 range（如 A1:C3）。"
    if not values_str:
        return "缺少 values（JSON 二维数组）。"

    try:
        values = _json.loads(values_str)
        if not isinstance(values, list):
            return "values 必须是二维数组。"
    except _json.JSONDecodeError as e:
        return f"values JSON 解析失败: {e}"

    token = await ctx.feishu_token_mgr.get_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    base = "https://open.feishu.cn/open-apis"

    try:
        client = get_feishu_client()
        if not sheet_id:
            meta_resp = await client.get(
                f"{base}/sheets/v3/spreadsheets/{ss_token}/sheets/query",
                headers={"Authorization": f"Bearer {token}"},
            )
            meta = meta_resp.json()
            sheets = meta.get("data", {}).get("sheets", [])
            if not sheets:
                return "该表格没有工作表。"
            sheet_id = sheets[0].get("sheet_id", "")

        full_range = f"{sheet_id}!{range_str}"
        resp = await client.put(
            f"{base}/sheets/v2/spreadsheets/{ss_token}/values",
            headers=headers,
            json={
                "valueRange": {
                    "range": full_range,
                    "values": values,
                },
            },
        )
        data = resp.json()

        if data.get("code") != 0:
            return f"写入失败: {data.get('msg', '未知错误')}"

        updated = data.get("data", {}).get("updatedCells", 0)
        return f"写入成功，更新了 {updated} 个单元格。"
    except Exception as e:
        return f"写入表格失败: {e}"


# ── 飞书审批工具 ──


async def _tool_list_feishu_approvals(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """查看飞书审批列表."""

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    status = (args.get("status") or "PENDING").strip().upper()
    limit = min(args.get("limit", 10) or 10, 20)

    token = await ctx.feishu_token_mgr.get_token()
    headers = {"Authorization": f"Bearer {token}"}
    base = "https://open.feishu.cn/open-apis"

    try:
        client = get_feishu_client()
        # 先获取审批定义列表
        resp = await client.get(
            f"{base}/approval/v4/approvals",
            headers=headers,
            params={"page_size": 20},
        )
        data = resp.json()

        if data.get("code") != 0:
            return f"获取审批失败: {data.get('msg', '未知错误')}"

        approvals = data.get("data", {}).get("approval_list", [])
        if not approvals:
            return "没有找到审批流程。"

        # 遍历审批定义，查实例
        all_instances: list[str] = []
        for appr in approvals[:5]:  # 只查前 5 个审批定义
            code = appr.get("approval_code", "")
            name = appr.get("approval_name", "未命名")
            if not code:
                continue
            params: dict[str, Any] = {  # noqa: F821
                "approval_code": code,
                "page_size": limit,
            }
            if status != "ALL":
                params["status"] = status
            inst_resp = await client.get(
                f"{base}/approval/v4/instances",
                headers=headers,
                params=params,
            )
            inst_data = inst_resp.json()
            instances = inst_data.get("data", {}).get("instance_list", [])
            for inst in instances:
                inst_code = inst.get("instance_code", "")
                inst_status = inst.get("status", "")
                start_time = inst.get("start_time", "")
                # 转换时间戳
                ts_str = ""
                if start_time:
                    try:
                        from datetime import datetime, timedelta
                        from datetime import timezone as _tz

                        ts = int(start_time) // 1000 if len(start_time) > 10 else int(start_time)
                        dt = datetime.fromtimestamp(ts, _tz(timedelta(hours=8)))
                        ts_str = dt.strftime("%m-%d %H:%M")
                    except (ValueError, OSError):
                        ts_str = start_time
                status_icon = {"PENDING": "⏳", "APPROVED": "✅", "REJECTED": "❌"}.get(
                    inst_status, "📋"
                )
                all_instances.append(f"{status_icon} [{name}] {ts_str} (instance={inst_code})")
                if len(all_instances) >= limit:
                    break
            if len(all_instances) >= limit:
                break

        if not all_instances:
            label = {"PENDING": "待审批", "APPROVED": "已通过", "REJECTED": "已拒绝"}.get(
                status, ""
            )
            return f"没有{label}的审批。"

        return "\n".join(all_instances)
    except Exception as e:
        return f"获取审批失败: {e}"


# ── 实用工具 ──

# 单位换算表已移至 _constants.py


async def _tool_create_feishu_spreadsheet(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """在飞书创建新表格."""

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    title = (args.get("title") or "").strip()
    folder_token = (args.get("folder_token") or "").strip()
    if not title:
        return "需要表格标题。"

    token = await ctx.feishu_token_mgr.get_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        body: dict[str, Any] = {"title": title}  # noqa: F821
        if folder_token:
            body["folder_token"] = folder_token

        client = get_feishu_client()
        resp = await client.post(
            "https://open.feishu.cn/open-apis/sheets/v3/spreadsheets",
            headers=headers,
            json=body,
        )
        data = resp.json()

        if data.get("code") != 0:
            return f"创建失败: {data.get('msg', '未知错误')}"

        ss = data.get("data", {}).get("spreadsheet", {})
        ss_token = ss.get("spreadsheet_token", "")
        url = ss.get("url", "")
        return f"表格已创建: {title}\ntoken: {ss_token}\n{url}"
    except Exception as e:
        return f"创建表格失败: {e}"


# ── 飞书通讯录搜索 ──


async def _tool_feishu_contacts(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """飞书通讯录搜索."""

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    query = (args.get("query") or "").strip()
    limit = min(args.get("limit", 5) or 5, 20)
    if not query:
        return "需要搜索关键词。"

    token = await ctx.feishu_token_mgr.get_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        client = get_feishu_client()
        resp = await client.post(
            "https://open.feishu.cn/open-apis/search/v1/user",
            headers=headers,
            json={"query": query, "page_size": limit},
            timeout=15.0,
        )
        data = resp.json()

        if data.get("code") != 0:
            # 如果没有搜索权限，回退到群成员查找
            return f"通讯录搜索暂不可用({data.get('code')}): {data.get('msg', '')}。可以用 feishu_group_members 从群里查人。"

        users = data.get("data", {}).get("users", [])
        if not users:
            return f"未找到「{query}」。"

        lines = []
        for u in users:
            name = u.get("name", "未知")
            dept = u.get("department", {}).get("name", "")
            open_id = u.get("open_id", "")
            line = f"{name}"
            if dept:
                line += f" ({dept})"
            if open_id:
                line += f" [open_id={open_id}]"
            lines.append(line)
        return "\n".join(lines)
    except Exception as e:
        return f"搜索通讯录失败: {e}"


# ── 文本 & 开发工具 ──


async def _tool_feishu_bitable(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """读取飞书多维表格."""

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    app_token = (args.get("app_token") or "").strip()
    table_id = (args.get("table_id") or "").strip()
    if not app_token or not table_id:
        return "需要 app_token 和 table_id。"

    limit = min(args.get("limit", 20) or 20, 100)
    filter_str = (args.get("filter") or "").strip()

    token = await ctx.feishu_token_mgr.get_token()
    headers = {"Authorization": f"Bearer {token}"}
    base = "https://open.feishu.cn/open-apis"

    try:
        params: dict[str, Any] = {"page_size": limit}  # noqa: F821
        if filter_str:
            params["filter"] = filter_str

        client = get_feishu_client()
        resp = await client.get(
            f"{base}/bitable/v1/apps/{app_token}/tables/{table_id}/records",
            headers=headers,
            params=params,
        )
        data = resp.json()

        if data.get("code") != 0:
            return f"读取多维表格失败: {data.get('msg', '未知错误')}"

        records = data.get("data", {}).get("items", [])
        if not records:
            return "表格中没有数据。"

        lines: list[str] = []
        for i, rec in enumerate(records, 1):
            fields = rec.get("fields", {})
            parts = [f"{k}: {v}" for k, v in fields.items()]
            lines.append(f"{i}. {' | '.join(parts)}")

        total = data.get("data", {}).get("total", len(records))
        lines.insert(0, f"共 {total} 条记录（显示前 {len(records)} 条）：\n")
        return "\n".join(lines)
    except Exception as e:
        return f"读取多维表格失败: {e}"


async def _tool_feishu_wiki(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """搜索飞书知识库."""

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    query = (args.get("query") or "").strip()
    if not query:
        return "需要搜索关键词。"

    limit = min(args.get("limit", 10) or 10, 20)

    token = await ctx.feishu_token_mgr.get_token()
    headers = {"Authorization": f"Bearer {token}"}
    base = "https://open.feishu.cn/open-apis"

    try:
        client = get_feishu_client()
        resp = await client.post(
            f"{base}/wiki/v2/spaces/search",
            headers=headers,
            json={"query": query, "page_size": limit},
        )
        data = resp.json()

        if data.get("code") != 0:
            return f"搜索知识库失败: {data.get('msg', '未知错误')}"

        items = data.get("data", {}).get("items", [])
        if not items:
            return f"知识库中没有找到「{query}」相关内容。"

        lines: list[str] = []
        for item in items:
            title = item.get("title", "无标题")
            url = item.get("url", "")
            space = item.get("space_name", "")
            line = f"- {title}"
            if space:
                line += f" [{space}]"
            if url:
                line += f"\n  {url}"
            lines.append(line)
        return "\n".join(lines)
    except Exception as e:
        return f"搜索知识库失败: {e}"


async def _tool_approve_feishu(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """操作飞书审批（通过/拒绝）."""

    if not ctx or not ctx.feishu_token_mgr:
        return "飞书未配置。"

    instance_id = (args.get("instance_id") or "").strip()
    action = (args.get("action") or "").strip().lower()
    comment = (args.get("comment") or "").strip()

    if not instance_id:
        return "需要审批实例 ID。"
    if action not in ("approve", "reject"):
        return "action 必须是 approve 或 reject。"

    token = await ctx.feishu_token_mgr.get_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    base = "https://open.feishu.cn/open-apis"
    action_cn = "通过" if action == "approve" else "拒绝"

    try:
        client = get_feishu_client()
        # 1. 获取审批实例详情，拿到 approval_code 和待处理任务节点
        inst_resp = await client.get(
            f"{base}/approval/v4/instances/{instance_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        inst_data = inst_resp.json()
        if inst_data.get("code") != 0:
            return f"获取审批实例失败: {inst_data.get('msg', '未知错误')}"

        instance = inst_data.get("data", {})
        approval_code = instance.get("approval_code", "")

        # 2. 从 task_list 找 PENDING 状态的审批节点
        task_list = instance.get("task_list", [])
        pending_task = None
        for t in task_list:
            if t.get("status") == "PENDING":
                pending_task = t
                break

        if not pending_task:
            return "没有待处理的审批节点，可能已被处理。"

        task_node_id = pending_task.get("id", "")
        user_id = pending_task.get("user_id", "")

        # 3. 调用审批/拒绝 API
        resp = await client.post(
            f"{base}/approval/v4/tasks/{action}",
            headers=headers,
            json={
                "approval_code": approval_code,
                "instance_code": instance_id,
                "user_id": user_id,
                "task_id": task_node_id,
                "comment": comment or action_cn,
            },
        )
        data = resp.json()

        if data.get("code") == 0:
            return f"审批已{action_cn}。"
        return f"审批操作失败: {data.get('msg', '未知错误')}"
    except Exception as e:
        return f"审批操作失败: {e}"


HANDLERS: dict[str, object] = {
    "create_feishu_event": _tool_create_feishu_event,
    "read_feishu_calendar": _tool_read_feishu_calendar,
    "delete_feishu_event": _tool_delete_feishu_event,
    "create_feishu_task": _tool_create_feishu_task,
    "list_feishu_tasks": _tool_list_feishu_tasks,
    "complete_feishu_task": _tool_complete_feishu_task,
    "delete_feishu_task": _tool_delete_feishu_task,
    "update_feishu_task": _tool_update_feishu_task,
    "feishu_chat_history": _tool_feishu_chat_history,
    "search_feishu_docs": _tool_search_feishu_docs,
    "read_feishu_doc": _tool_read_feishu_doc,
    "create_feishu_doc": _tool_create_feishu_doc,
    "send_feishu_group": _tool_send_feishu_group,
    "send_feishu_file": _tool_send_feishu_file,
    "list_feishu_groups": _tool_list_feishu_groups,
    "send_feishu_dm": _tool_send_feishu_dm,
    "feishu_group_members": _tool_feishu_group_members,
    "read_feishu_sheet": _tool_read_feishu_sheet,
    "update_feishu_sheet": _tool_update_feishu_sheet,
    "list_feishu_approvals": _tool_list_feishu_approvals,
    "create_feishu_spreadsheet": _tool_create_feishu_spreadsheet,
    "feishu_contacts": _tool_feishu_contacts,
    "feishu_bitable": _tool_feishu_bitable,
    "feishu_wiki": _tool_feishu_wiki,
    "approve_feishu": _tool_approve_feishu,
}
