"""Crew MCP Server — Model Context Protocol 服务."""

import json
import logging
import os
import threading
import time as _time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ── 远程 Memory API 客户端 ──────────────────────────────────────


def _get_remote_memory_config() -> tuple[str, str] | None:
    """从环境变量获取远程 API 配置，返回 (base_url, token) 或 None."""
    base_url = os.environ.get("CREW_API_URL", os.environ.get("CREW_REMOTE_URL", "")).rstrip("/")
    token = os.environ.get("CREW_API_TOKEN", "")
    if base_url and token:
        return (base_url, token)
    return None


async def _remote_memory_add(
    base_url: str,
    token: str,
    *,
    employee: str,
    category: str,
    content: str,
    source_session: str = "",
    ttl_days: int = 0,
    tags: list[str] | None = None,
    shared: bool = False,
    trigger_condition: str = "",
    applicability: list[str] | None = None,
    origin_employee: str = "",
) -> dict:
    """通过远程 API 写入记忆，返回响应 dict."""
    import httpx

    payload = {
        "employee": employee,
        "category": category,
        "content": content,
        "source_session": source_session,
        "ttl_days": ttl_days,
        "tags": tags or [],
        "shared": shared,
        "trigger_condition": trigger_condition,
        "applicability": applicability or [],
        "origin_employee": origin_employee,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{base_url}/api/memory/add",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json()


async def _remote_memory_query(
    base_url: str,
    token: str,
    *,
    employee: str,
    category: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """通过远程 API 查询记忆，返回条目列表."""
    import httpx

    params: dict[str, str] = {"employee": employee, "limit": str(limit)}
    if category:
        params["category"] = category
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url}/api/memory/query",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("entries", [])


# ── Wiki Files API 客户端 ─────────────────────────────────────


def _get_wiki_config() -> tuple[str, str] | None:
    """从环境变量获取 Wiki API 配置，返回 (base_url, token) 或 None."""
    base_url = os.environ.get("WIKI_API_URL", "").rstrip("/")
    token = os.environ.get("WIKI_API_TOKEN", "")
    if base_url and token:
        return (base_url, token)
    return None


async def _wiki_upload_file(
    base_url: str,
    token: str,
    *,
    file_bytes: bytes,
    filename: str,
    content_type: str = "application/octet-stream",
    space_slug: str = "",
    doc_slug: str = "",
) -> dict:
    """上传文件到 Wiki，返回响应 dict."""
    import httpx

    files = {"file": (filename, file_bytes, content_type)}
    data: dict[str, str] = {}
    if space_slug:
        data["space_slug"] = space_slug
    if doc_slug:
        data["doc_slug"] = doc_slug
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{base_url}/api/wiki/files/upload",
            files=files,
            data=data,
            headers={"X-Wiki-Token": token},
        )
        resp.raise_for_status()
        return resp.json()


async def _wiki_get_file(
    base_url: str,
    token: str,
    *,
    file_id: int,
) -> dict:
    """获取 Wiki 文件元数据 + 签名 URL + text_content."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url}/api/wiki/files/{file_id}",
            headers={"X-Wiki-Token": token},
        )
        resp.raise_for_status()
        return resp.json()


async def _wiki_list_files(
    base_url: str,
    token: str,
    *,
    space_slug: str = "",
    doc_slug: str = "",
    mime_type: str = "",
    page: int = 1,
    page_size: int = 20,
) -> dict:
    """列出 Wiki 文件."""
    import httpx

    params: dict[str, str] = {
        "page": str(page),
        "page_size": str(page_size),
    }
    if space_slug:
        params["space_slug"] = space_slug
    if doc_slug:
        params["doc_slug"] = doc_slug
    if mime_type:
        params["mime_type"] = mime_type
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url}/api/wiki/files",
            params=params,
            headers={"X-Wiki-Token": token},
        )
        resp.raise_for_status()
        return resp.json()


async def _wiki_delete_file(
    base_url: str,
    token: str,
    *,
    file_id: int,
) -> dict:
    """删除 Wiki 文件，返回响应 dict."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.delete(
            f"{base_url}/api/wiki/files/{file_id}",
            headers={"X-Wiki-Token": token},
        )
        resp.raise_for_status()
        return resp.json()


# ── Wiki Admin API 客户端（文档 CRUD）──────────────────────────


def _get_wiki_admin_config() -> tuple[str, str] | None:
    """从环境变量获取 Wiki Admin API 配置，返回 (base_url, token) 或 None.

    base_url 复用 WIKI_API_URL（指向蚁聚后端）。
    token 优先取 WIKI_ADMIN_TOKEN，fallback 到 ANTGATHER_API_TOKEN（即蚁聚 INTERNAL_SERVICE_TOKEN）。
    """
    base_url = os.environ.get("WIKI_API_URL", "").rstrip("/")
    token = os.environ.get("WIKI_ADMIN_TOKEN", "") or os.environ.get("ANTGATHER_API_TOKEN", "")
    if base_url and token:
        return (base_url, token)
    return None


async def _wiki_list_spaces(base_url: str, token: str) -> list[dict]:
    """查询 Wiki 空间列表，返回 spaces 数组."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url}/api/admin/wiki/spaces",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("spaces", [])


async def _wiki_resolve_space_id(base_url: str, token: str, space_slug: str) -> int | None:
    """通过 slug 查找 space_id，未找到返回 None."""
    spaces = await _wiki_list_spaces(base_url, token)
    for s in spaces:
        if s.get("slug") == space_slug:
            return s["id"]
    return None


async def _wiki_create_doc(
    base_url: str,
    token: str,
    *,
    space_id: int,
    title: str,
    slug: str,
    content: str = "",
    ai_content: str = "",
    content_type: str = "markdown",
    excerpt: str = "",
    visibility: str = "internal",
    parent_id: int | None = None,
    sort_order: int = 0,
    is_pinned: bool = False,
) -> dict:
    """创建 Wiki 文档页，返回响应 dict."""
    import httpx

    body: dict = {
        "space_id": space_id,
        "title": title,
        "slug": slug,
    }
    if content:
        body["content"] = content
    if ai_content:
        body["ai_content"] = ai_content
    if content_type != "markdown":
        body["content_type"] = content_type
    if excerpt:
        body["excerpt"] = excerpt
    if visibility != "internal":
        body["visibility"] = visibility
    if parent_id is not None:
        body["parent_id"] = parent_id
    if sort_order != 0:
        body["sort_order"] = sort_order
    if is_pinned:
        body["is_pinned"] = is_pinned

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{base_url}/api/admin/wiki/docs",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json()


async def _wiki_update_doc(
    base_url: str,
    token: str,
    *,
    doc_id: int,
    title: str = "",
    slug: str = "",
    content: str | None = None,
    ai_content: str | None = None,
    content_type: str = "",
    excerpt: str | None = None,
    visibility: str = "",
    parent_id: int | None = None,
    sort_order: int | None = None,
    is_pinned: bool | None = None,
) -> dict:
    """更新 Wiki 文档页（部分更新），返回响应 dict."""
    import httpx

    body: dict = {}
    if title:
        body["title"] = title
    if slug:
        body["slug"] = slug
    if content is not None:
        body["content"] = content
    if ai_content is not None:
        body["ai_content"] = ai_content
    if content_type:
        body["content_type"] = content_type
    if excerpt is not None:
        body["excerpt"] = excerpt
    if visibility:
        body["visibility"] = visibility
    if parent_id is not None:
        body["parent_id"] = parent_id
    if sort_order is not None:
        body["sort_order"] = sort_order
    if is_pinned is not None:
        body["is_pinned"] = is_pinned

    if not body:
        return {"ok": True, "message": "没有需要更新的字段"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.put(
            f"{base_url}/api/admin/wiki/docs/{doc_id}",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json()


async def _wiki_find_doc_by_slug(
    base_url: str,
    token: str,
    *,
    space_slug: str,
    doc_slug: str,
) -> dict | None:
    """通过 space_slug + doc_slug 查找文档，返回 doc dict 或 None."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url}/api/admin/wiki/docs",
            params={"space_slug": space_slug, "slug": doc_slug},
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        docs = data.get("docs", [])
        if docs:
            return docs[0]
        return None


# ── 远程 KV API 客户端 ──────────────────────────────────────────


async def _remote_kv_put(base_url: str, token: str, *, key: str, content: str) -> dict:
    """通过远程 API 写入 KV，返回响应 dict."""
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.put(
            f"{base_url}/api/kv/{key}",
            content=content.encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "text/plain; charset=utf-8",
            },
        )
        resp.raise_for_status()
        return resp.json()


async def _remote_kv_get(base_url: str, token: str, *, key: str) -> str:
    """通过远程 API 读取 KV，返回文件内容字符串. 不存在时抛 httpx.HTTPStatusError."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url}/api/kv/{key}",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.text


async def _remote_kv_list(base_url: str, token: str, *, prefix: str = "") -> list[str]:
    """通过远程 API 列出 KV keys，返回 key 列表."""
    import httpx

    params: dict[str, str] = {}
    if prefix:
        params["prefix"] = prefix
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url}/api/kv/",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("keys", [])


# ── 远程 Employee API 客户端 ──────────────────────────────────


async def _remote_list_employees(
    base_url: str,
    token: str,
    *,
    tag: str | None = None,
) -> list[dict]:
    """通过远程 API 列出员工，返回员工列表."""
    import httpx

    params: dict[str, str] = {}
    if tag:
        params["tag"] = tag
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url}/api/employees",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("items", [])


async def _remote_get_employee(
    base_url: str,
    token: str,
    *,
    name: str,
) -> dict:
    """通过远程 API 获取单个员工详情，返回完整 dict."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url}/api/employees/{name}",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json()


async def _remote_run_employee(
    base_url: str,
    token: str,
    *,
    name: str,
    args: dict | None = None,
    agent_id: str | None = None,
) -> str:
    """通过远程 API 获取员工 prompt 文本."""
    import httpx

    params: dict[str, str] = {}
    if args:
        for k, v in args.items():
            params[f"arg_{k}"] = v
    if agent_id:
        params["agent_id"] = agent_id
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{base_url}/api/employees/{name}/prompt",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("prompt", "")


# ── 远程 Pipelines / Discussions / Decisions / Meetings / WorkLog / Permission API 客户端 ──


async def _remote_list_pipelines(base_url: str, token: str) -> list[dict]:
    """通过远程 API 列出流水线."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url}/api/config/pipelines",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json().get("items", [])


async def _remote_list_discussions(base_url: str, token: str) -> list[dict]:
    """通过远程 API 列出讨论会."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url}/api/config/discussions",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json().get("items", [])


async def _remote_run_discussion_prompt(
    base_url: str,
    token: str,
    *,
    name: str,
    args: dict | None = None,
    agent_id: str | None = None,
    smart_context: bool = True,
) -> str:
    """通过远程 API 获取预定义讨论会 prompt（非编排模式）."""
    import httpx

    params: dict[str, str] = {}
    if args:
        for k, v in args.items():
            params[f"arg_{k}"] = v
    if agent_id:
        params["agent_id"] = agent_id
    if not smart_context:
        params["smart_context"] = "false"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{base_url}/api/discussions/{name}/prompt",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("prompt", "")


async def _remote_run_discussion_plan(
    base_url: str,
    token: str,
    *,
    name: str,
    args: dict | None = None,
    agent_id: str | None = None,
    smart_context: bool = True,
) -> dict:
    """通过远程 API 获取预定义讨论会编排计划."""
    import httpx

    params: dict[str, str] = {}
    if args:
        for k, v in args.items():
            params[f"arg_{k}"] = v
    if agent_id:
        params["agent_id"] = agent_id
    if not smart_context:
        params["smart_context"] = "false"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{base_url}/api/discussions/{name}/plan",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json()


async def _remote_track_decision(
    base_url: str,
    token: str,
    *,
    employee: str,
    category: str,
    content: str,
    expected_outcome: str = "",
    meeting_id: str = "",
) -> dict:
    """通过远程 API 记录决策."""
    import httpx

    payload = {
        "employee": employee,
        "category": category,
        "content": content,
        "expected_outcome": expected_outcome,
        "meeting_id": meeting_id,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{base_url}/api/decisions/track",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json()


async def _remote_evaluate_decision(
    base_url: str,
    token: str,
    *,
    decision_id: str,
    actual_outcome: str,
    evaluation: str = "",
) -> dict:
    """通过远程 API 评估决策."""
    import httpx

    payload: dict[str, str] = {"actual_outcome": actual_outcome}
    if evaluation:
        payload["evaluation"] = evaluation
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{base_url}/api/decisions/{decision_id}/evaluate",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json()


async def _remote_list_meeting_history(
    base_url: str,
    token: str,
    *,
    limit: int = 20,
    keyword: str | None = None,
) -> list[dict]:
    """通过远程 API 列出会议历史."""
    import httpx

    params: dict[str, str] = {"limit": str(limit)}
    if keyword:
        params["keyword"] = keyword
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url}/api/meetings",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json().get("items", [])


async def _remote_get_meeting_detail(
    base_url: str,
    token: str,
    *,
    meeting_id: str,
) -> dict:
    """通过远程 API 获取会议详情."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url}/api/meetings/{meeting_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json()


async def _remote_get_work_log(
    base_url: str,
    token: str,
    *,
    employee_name: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """通过远程 API 获取工作日志."""
    import httpx

    params: dict[str, str] = {"limit": str(limit)}
    if employee_name:
        params["employee_name"] = employee_name
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url}/api/work-log",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json().get("items", [])


async def _remote_get_permission_matrix(
    base_url: str,
    token: str,
    *,
    employee: str | None = None,
) -> list[dict]:
    """通过远程 API 获取权限矩阵."""
    import httpx

    params: dict[str, str] = {}
    if employee:
        params["employee"] = employee
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url}/api/permission-matrix",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.json().get("items", [])


try:
    from mcp.server import InitializationOptions, Server
    from mcp.server.lowlevel.helper_types import ReadResourceContents
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        GetPromptResult,
        Prompt,
        PromptArgument,
        PromptMessage,
        Resource,
        ServerCapabilities,
        TextContent,
        Tool,
    )

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from crew.context_detector import detect_project
from crew.discovery import discover_employees
from crew.engine import CrewEngine
from crew.exceptions import EmployeeNotFoundError
from crew.log import WorkLogger
from crew.pipeline import (
    arun_pipeline,
    discover_pipelines,
    load_pipeline,
    validate_pipeline,
)


class ToolMetricsCollector:
    """Tool 调用使用率埋点收集器 — 线程安全.

    内存热缓存 + EventCollector 持久化双写。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._start_time = _time.monotonic()
        self._total_calls = 0
        # per-tool stats: {tool_name: {calls, success, failed, total_ms, last_called, errors: {type: count}}}
        self._by_tool: dict[str, dict] = {}

    def record(
        self,
        *,
        tool_name: str,
        duration_ms: float,
        success: bool,
        error_type: str = "",
    ) -> None:
        """记录一次 Tool 调用（内存 + 持久化双写）."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._total_calls += 1
            if tool_name not in self._by_tool:
                self._by_tool[tool_name] = {
                    "calls": 0,
                    "success": 0,
                    "failed": 0,
                    "total_ms": 0.0,
                    "last_called": "",
                    "errors": {},
                }
            stats = self._by_tool[tool_name]
            stats["calls"] += 1
            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1
                if error_type:
                    stats["errors"][error_type] = stats["errors"].get(error_type, 0) + 1
            stats["total_ms"] += duration_ms
            stats["last_called"] = now

        # 持久化双写（写入失败不影响主流程）
        try:
            from crew.event_collector import get_event_collector

            ec = get_event_collector()
            ec.record(
                event_type="tool_call",
                event_name=tool_name,
                duration_ms=duration_ms,
                success=success,
                error_type=error_type,
                source="mcp",
            )
        except Exception:
            pass  # 持久化失败不影响主流程

    def snapshot(self, tool_name: str | None = None) -> dict:
        """返回 Tool 使用率快照（内存热缓存）.

        Args:
            tool_name: 可选，过滤单个 Tool
        """
        with self._lock:
            result: dict = {
                "uptime_seconds": round(_time.monotonic() - self._start_time),
                "total_tool_calls": self._total_calls,
                "tools": {},
            }
            items = self._by_tool.items()
            if tool_name:
                items = [(k, v) for k, v in items if k == tool_name]
            for name, stats in items:
                avg_ms = round(stats["total_ms"] / stats["calls"], 1) if stats["calls"] else 0
                result["tools"][name] = {
                    "calls": stats["calls"],
                    "success": stats["success"],
                    "failed": stats["failed"],
                    "avg_duration_ms": avg_ms,
                    "last_called": stats["last_called"],
                    "errors": dict(stats["errors"]),
                }
            return result

    def snapshot_persistent(
        self,
        tool_name: str | None = None,
        since: str | None = None,
    ) -> dict:
        """从 events 表读取持久化统计.

        Args:
            tool_name: 可选，过滤单个 Tool
            since: 可选，ISO 8601 时间戳，只统计此时间之后的事件
        """
        try:
            from crew.event_collector import get_event_collector

            ec = get_event_collector()
            agg = ec.aggregate(event_type="tool_call", since=since)
            tools: dict = {}
            total = 0
            for row in agg:
                if tool_name and row["event_name"] != tool_name:
                    continue
                total += row["count"]
                tools[row["event_name"]] = {
                    "calls": row["count"],
                    "success": row["success_count"],
                    "failed": row["fail_count"],
                    "avg_duration_ms": row["avg_duration_ms"],
                    "last_called": row["last_seen"],
                }
            return {
                "source": "persistent",
                "total_tool_calls": total,
                "tools": tools,
            }
        except Exception:
            return {
                "source": "persistent",
                "total_tool_calls": 0,
                "tools": {},
                "error": "unavailable",
            }

    def reset(self) -> None:
        """重置内存统计."""
        with self._lock:
            self._start_time = _time.monotonic()
            self._total_calls = 0
            self._by_tool.clear()


# 全局 Tool 指标收集器单例
_tool_metrics = ToolMetricsCollector()


def get_tool_metrics_collector() -> ToolMetricsCollector:
    """获取全局 Tool 指标收集器."""
    return _tool_metrics


def _get_version() -> str:
    """读取包版本."""
    try:
        from importlib.metadata import version

        return version("knowlyr-crew")
    except Exception:
        return "unknown"


def create_server(project_dir: Path | None = None) -> "Server":
    """创建 MCP 服务器实例."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-crew[mcp]")

    _project_dir = project_dir  # captured in closure

    server = Server("crew")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """列出可用的工具."""
        return [
            Tool(
                name="list_employees",
                description="列出所有可用的数字员工",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tag": {
                            "type": "string",
                            "description": "按标签过滤（可选）",
                        },
                    },
                },
            ),
            Tool(
                name="get_employee",
                description="获取数字员工的完整定义",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "员工名称或触发别名",
                        },
                    },
                    "required": ["name"],
                },
            ),
            Tool(
                name="run_employee",
                description="加载数字员工并生成可执行的 prompt",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "员工名称或触发别名",
                        },
                        "args": {
                            "type": "object",
                            "description": "传递给员工的参数（key-value）",
                            "additionalProperties": {"type": "string"},
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "绑定的平台 Agent ID（可选）",
                        },
                    },
                    "required": ["name"],
                },
            ),
            Tool(
                name="get_work_log",
                description="查看数字员工的工作日志",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "employee_name": {
                            "type": "string",
                            "description": "按员工过滤（可选）",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "返回条数（默认 10）",
                            "default": 10,
                        },
                    },
                },
            ),
            Tool(
                name="detect_project",
                description="检测当前项目类型、框架、包管理器等信息",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_dir": {
                            "type": "string",
                            "description": "项目目录路径（默认当前目录）",
                        },
                    },
                },
            ),
            Tool(
                name="list_pipelines",
                description="列出所有可用的流水线",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="run_pipeline",
                description="执行流水线 — 支持 prompt-only 模式和 execute 模式（自动调用 LLM 串联执行）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "流水线名称或 YAML 文件路径",
                        },
                        "args": {
                            "type": "object",
                            "description": "传递给流水线的参数（key-value）",
                            "additionalProperties": {"type": "string"},
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "绑定的平台 Agent ID（可选）",
                        },
                        "smart_context": {
                            "type": "boolean",
                            "description": "自动检测项目类型（默认 true）",
                            "default": True,
                        },
                        "execute": {
                            "type": "boolean",
                            "description": "执行模式 — 自动调用 LLM 串联执行（默认 false）",
                            "default": False,
                        },
                        "model": {
                            "type": "string",
                            "description": "LLM 模型标识符（execute 模式使用）",
                        },
                    },
                    "required": ["name"],
                },
            ),
            Tool(
                name="list_discussions",
                description="列出所有可用的讨论会",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="run_discussion",
                description="生成讨论会 prompt — 支持预定义 YAML 或即席讨论（employees+topic）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "讨论会名称或 YAML 文件路径（与 employees+topic 二选一）",
                        },
                        "employees": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "即席讨论的员工列表（与 name 二选一）",
                        },
                        "topic": {
                            "type": "string",
                            "description": "即席讨论的议题（与 employees 搭配使用）",
                        },
                        "goal": {
                            "type": "string",
                            "description": "讨论目标（可选）",
                        },
                        "rounds": {
                            "type": "integer",
                            "description": "讨论轮次（默认 2，即席讨论时使用）",
                        },
                        "round_template": {
                            "type": "string",
                            "description": "轮次模板 (standard, brainstorm-to-decision, adversarial)",
                        },
                        "args": {
                            "type": "object",
                            "description": "传递给讨论会的参数（key-value）",
                            "additionalProperties": {"type": "string"},
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "绑定的平台 Agent ID（可选）",
                        },
                        "smart_context": {
                            "type": "boolean",
                            "description": "自动检测项目类型（默认 true）",
                            "default": True,
                        },
                        "orchestrated": {
                            "type": "boolean",
                            "description": "编排模式：每个参会者独立推理（默认 false）",
                            "default": False,
                        },
                    },
                },
            ),
            Tool(
                name="add_memory",
                description="为员工添加一条持久化记忆",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "employee": {
                            "type": "string",
                            "description": "员工名称",
                        },
                        "category": {
                            "type": "string",
                            "enum": ["decision", "estimate", "finding", "correction", "pattern"],
                            "description": "记忆类别",
                        },
                        "content": {
                            "type": "string",
                            "description": "记忆内容",
                        },
                        "source_session": {
                            "type": "string",
                            "description": "来源 session ID（可选）",
                        },
                        "ttl_days": {
                            "type": "integer",
                            "description": "生存期天数 (0=永不过期，默认 0)",
                            "default": 0,
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "语义标签列表（可选）",
                        },
                        "shared": {
                            "type": "boolean",
                            "description": "是否加入共享记忆池（默认 false，pattern 类型自动为 true）",
                            "default": False,
                        },
                        "trigger_condition": {
                            "type": "string",
                            "description": "触发条件：什么场景下该用此模式（仅 pattern 类型）",
                        },
                        "applicability": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "适用范围：角色/领域标签列表（仅 pattern 类型）",
                        },
                        "origin_employee": {
                            "type": "string",
                            "description": "来源员工名（仅 pattern 类型，默认当前员工）",
                        },
                    },
                    "required": ["employee", "category", "content"],
                },
            ),
            Tool(
                name="query_memory",
                description="查询员工的持久化记忆",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "employee": {
                            "type": "string",
                            "description": "员工名称",
                        },
                        "category": {
                            "type": "string",
                            "enum": ["decision", "estimate", "finding", "correction", "pattern"],
                            "description": "按类别过滤（可选）",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "最大返回条数（默认 20）",
                            "default": 20,
                        },
                    },
                    "required": ["employee"],
                },
            ),
            Tool(
                name="track_decision",
                description="记录一个待评估的决策（来自会议或日常工作）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "employee": {
                            "type": "string",
                            "description": "提出决策的员工名称",
                        },
                        "category": {
                            "type": "string",
                            "enum": ["estimate", "recommendation", "commitment"],
                            "description": "决策类别",
                        },
                        "content": {
                            "type": "string",
                            "description": "决策内容",
                        },
                        "expected_outcome": {
                            "type": "string",
                            "description": "预期结果（可选）",
                        },
                        "meeting_id": {
                            "type": "string",
                            "description": "来源会议 ID（可选）",
                        },
                    },
                    "required": ["employee", "category", "content"],
                },
            ),
            Tool(
                name="evaluate_decision",
                description="评估一个决策 — 记录实际结果并将经验写入员工记忆",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "decision_id": {
                            "type": "string",
                            "description": "决策 ID",
                        },
                        "actual_outcome": {
                            "type": "string",
                            "description": "实际结果",
                        },
                        "evaluation": {
                            "type": "string",
                            "description": "评估结论（可选，为空则自动生成）",
                        },
                    },
                    "required": ["decision_id", "actual_outcome"],
                },
            ),
            Tool(
                name="list_meeting_history",
                description="查看讨论会历史记录",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "返回条数（默认 20）",
                            "default": 20,
                        },
                        "keyword": {
                            "type": "string",
                            "description": "按关键词过滤",
                        },
                    },
                },
            ),
            Tool(
                name="get_meeting_detail",
                description="获取某次讨论会的完整记录",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "meeting_id": {
                            "type": "string",
                            "description": "会议 ID",
                        },
                    },
                    "required": ["meeting_id"],
                },
            ),
            Tool(
                name="list_tool_schemas",
                description="列出所有可用的工具定义（名称和描述）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "description": "按角色预设过滤（如 profile-engineer, memory, github）",
                        },
                    },
                },
            ),
            Tool(
                name="get_permission_matrix",
                description="查看员工权限矩阵 — 每位员工的有效工具集和权限策略",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "employee": {
                            "type": "string",
                            "description": "查询指定员工（可选，不传则列出所有）",
                        },
                    },
                },
            ),
            Tool(
                name="get_audit_log",
                description="查询工具调用审计日志",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "employee": {
                            "type": "string",
                            "description": "按员工过滤（可选）",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "返回条数（默认 50）",
                            "default": 50,
                        },
                        "denied_only": {
                            "type": "boolean",
                            "description": "仅返回被拒绝的调用（默认 false）",
                            "default": False,
                        },
                    },
                },
            ),
            Tool(
                name="get_tool_metrics",
                description="查询 MCP Tool 使用率统计 — 调用次数、成功/失败、平均耗时等（支持持久化历史数据）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "按工具名过滤（可选，不传则返回所有工具的统计）",
                        },
                        "since": {
                            "type": "string",
                            "description": "ISO 8601 时间戳，只统计此时间之后的事件（可选，不传则统计全部历史）",
                        },
                    },
                },
            ),
            Tool(
                name="query_events",
                description="查询统一埋点事件 — 支持按 event_type / event_name / 时间范围过滤",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "event_type": {
                            "type": "string",
                            "description": "事件类型过滤（tool_call / employee_run / pipeline_run / discussion_run 等）",
                        },
                        "event_name": {
                            "type": "string",
                            "description": "事件名称过滤（如工具名、员工名等）",
                        },
                        "since": {
                            "type": "string",
                            "description": "起始时间 ISO 8601（可选）",
                        },
                        "until": {
                            "type": "string",
                            "description": "截止时间 ISO 8601（可选）",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "最大返回条数（默认 100）",
                            "default": 100,
                        },
                        "aggregate": {
                            "type": "boolean",
                            "description": "是否返回聚合统计而非原始事件（默认 false）",
                            "default": False,
                        },
                    },
                },
            ),
            Tool(
                name="put_config",
                description="写入配置文件到 KV 存储（用于跨机器同步 CLAUDE.md / MEMORY.md 等）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "存储路径 key，如 config/knowlyr-crew/CLAUDE.md",
                        },
                        "content": {
                            "type": "string",
                            "description": "文件内容",
                        },
                    },
                    "required": ["key", "content"],
                },
            ),
            Tool(
                name="get_config",
                description="从 KV 存储读取配置文件",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "存储路径 key，如 config/knowlyr-crew/CLAUDE.md",
                        },
                    },
                    "required": ["key"],
                },
            ),
            Tool(
                name="list_configs",
                description="列出 KV 存储中指定前缀下的所有 key",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prefix": {
                            "type": "string",
                            "description": "前缀过滤，如 config/（可选，不传则列出全部）",
                        },
                    },
                },
            ),
            Tool(
                name="wiki_upload",
                description="上传文件到 Wiki — 支持本地文件路径或 base64 内容",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "本地文件路径（与 content+filename 二选一）",
                        },
                        "content": {
                            "type": "string",
                            "description": "base64 编码的文件内容（与 file_path 二选一）",
                        },
                        "filename": {
                            "type": "string",
                            "description": "文件名（使用 content 时必填）",
                        },
                        "space_slug": {
                            "type": "string",
                            "description": "Wiki 空间 slug（可选）",
                        },
                        "doc_slug": {
                            "type": "string",
                            "description": "Wiki 文档 slug（可选）",
                        },
                    },
                },
            ),
            Tool(
                name="wiki_read_file",
                description="读取 Wiki 文件 — 返回文本内容（如有）和签名 URL",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": ["string", "integer"],
                            "description": "文件 ID（与 filename 二选一）",
                        },
                        "filename": {
                            "type": "string",
                            "description": "文件名模糊搜索（与 file_id 二选一）",
                        },
                    },
                },
            ),
            Tool(
                name="wiki_list_files",
                description="列出 Wiki 文件 — 支持按空间、文档、MIME 类型过滤",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "space_slug": {
                            "type": "string",
                            "description": "按 Wiki 空间 slug 过滤（可选）",
                        },
                        "doc_slug": {
                            "type": "string",
                            "description": "按 Wiki 文档 slug 过滤（可选）",
                        },
                        "mime_type": {
                            "type": "string",
                            "description": "按 MIME 类型过滤（可选）",
                        },
                        "page": {
                            "type": "integer",
                            "description": "页码（默认 1）",
                            "default": 1,
                        },
                        "page_size": {
                            "type": "integer",
                            "description": "每页条数（默认 20）",
                            "default": 20,
                        },
                    },
                },
            ),
            Tool(
                name="wiki_delete_file",
                description="删除 Wiki 文件 — 按 file_id 删除 OSS 文件和数据库记录",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": ["integer", "string"],
                            "description": "文件 ID（必填）",
                        },
                    },
                    "required": ["file_id"],
                },
            ),
            Tool(
                name="wiki_create_doc",
                description="创建 Wiki 文档页 — 在指定空间下新建文档",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "space_slug": {
                            "type": "string",
                            "description": "Wiki 空间 slug（如 dev, company, internal, projects）",
                        },
                        "title": {
                            "type": "string",
                            "description": "文档标题（1-200字）",
                        },
                        "slug": {
                            "type": "string",
                            "description": "文档 slug（1-200字，空间内唯一，用于 URL）",
                        },
                        "content": {
                            "type": "string",
                            "description": "文档内容（Markdown，最大200000字，可选）",
                        },
                        "ai_content": {
                            "type": "string",
                            "description": "AI 版内容（信息密度更高，可选）",
                        },
                        "excerpt": {
                            "type": "string",
                            "description": "摘要（最多500字，可选）",
                        },
                        "visibility": {
                            "type": "string",
                            "description": "可见性：draft/internal/public/open（默认 internal）",
                            "default": "internal",
                        },
                        "parent_id": {
                            "type": "integer",
                            "description": "父文档 ID（可选）",
                        },
                        "sort_order": {
                            "type": "integer",
                            "description": "排序序号（默认 0）",
                            "default": 0,
                        },
                        "is_pinned": {
                            "type": "boolean",
                            "description": "是否置顶（默认 false）",
                            "default": False,
                        },
                    },
                    "required": ["space_slug", "title", "slug"],
                },
            ),
            Tool(
                name="wiki_update_doc",
                description="更新已有 Wiki 文档页 — 支持按 doc_id 或 space_slug+slug 定位",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "doc_id": {
                            "type": "integer",
                            "description": "文档 ID（与 space_slug+slug 二选一）",
                        },
                        "space_slug": {
                            "type": "string",
                            "description": "Wiki 空间 slug（与 doc_id 二选一，需配合 slug 使用）",
                        },
                        "slug": {
                            "type": "string",
                            "description": "文档 slug（与 space_slug 配合定位文档）",
                        },
                        "title": {
                            "type": "string",
                            "description": "新标题（可选）",
                        },
                        "new_slug": {
                            "type": "string",
                            "description": "新 slug（可选，用于重命名）",
                        },
                        "content": {
                            "type": "string",
                            "description": "新内容（Markdown，可选）",
                        },
                        "ai_content": {
                            "type": "string",
                            "description": "新 AI 版内容（可选）",
                        },
                        "excerpt": {
                            "type": "string",
                            "description": "新摘要（可选）",
                        },
                        "visibility": {
                            "type": "string",
                            "description": "新可见性：draft/internal/public/open（可选）",
                        },
                        "parent_id": {
                            "type": "integer",
                            "description": "新父文档 ID（可选）",
                        },
                        "sort_order": {
                            "type": "integer",
                            "description": "新排序序号（可选）",
                        },
                        "is_pinned": {
                            "type": "boolean",
                            "description": "是否置顶（可选）",
                        },
                    },
                },
            ),
            Tool(
                name="get_soul",
                description="读取员工灵魂配置（soul.md）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "employee_name": {
                            "type": "string",
                            "description": "员工名称",
                        },
                    },
                    "required": ["employee_name"],
                },
            ),
            Tool(
                name="update_soul",
                description="更新员工灵魂配置（自动版本递增 + 历史记录）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "employee_name": {
                            "type": "string",
                            "description": "员工名称",
                        },
                        "content": {
                            "type": "string",
                            "description": "soul.md 完整内容",
                        },
                        "updated_by": {
                            "type": "string",
                            "description": "更新者（可选）",
                        },
                    },
                    "required": ["employee_name", "content"],
                },
            ),
            Tool(
                name="create_employee",
                description="创建新的 AI 员工（含头像生成、文件系统创建）",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "员工标识（slug，仅 [a-z0-9-]）",
                        },
                        "character_name": {
                            "type": "string",
                            "description": "角色名/中文名",
                        },
                        "display_name": {
                            "type": "string",
                            "description": "显示名称（可选）",
                        },
                        "description": {
                            "type": "string",
                            "description": "职责描述",
                        },
                        "model": {
                            "type": "string",
                            "description": "使用的模型（默认 claude-sonnet-4-6）",
                        },
                        "model_tier": {
                            "type": "string",
                            "description": "模型档位（默认 claude）",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "标签列表",
                        },
                        "soul_content": {
                            "type": "string",
                            "description": "初始 soul 配置",
                        },
                        "avatar_prompt": {
                            "type": "string",
                            "description": "头像生成 prompt（可选）",
                        },
                        "agent_status": {
                            "type": "string",
                            "enum": ["active", "frozen", "inactive"],
                            "description": "员工状态（默认 active）",
                        },
                    },
                    "required": ["name", "character_name", "soul_content"],
                },
            ),
            Tool(
                name="create_discussion",
                description="创建讨论会配置",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "讨论会名称",
                        },
                        "yaml_content": {
                            "type": "string",
                            "description": "YAML 配置内容",
                        },
                        "description": {
                            "type": "string",
                            "description": "描述（可选）",
                        },
                    },
                    "required": ["name", "yaml_content"],
                },
            ),
            Tool(
                name="update_discussion",
                description="更新讨论会配置",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "讨论会名称",
                        },
                        "yaml_content": {
                            "type": "string",
                            "description": "YAML 配置内容",
                        },
                        "description": {
                            "type": "string",
                            "description": "描述（可选）",
                        },
                    },
                    "required": ["name", "yaml_content"],
                },
            ),
            Tool(
                name="create_pipeline",
                description="创建流水线配置",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "流水线名称",
                        },
                        "yaml_content": {
                            "type": "string",
                            "description": "YAML 配置内容",
                        },
                        "description": {
                            "type": "string",
                            "description": "描述（可选）",
                        },
                    },
                    "required": ["name", "yaml_content"],
                },
            ),
            Tool(
                name="update_pipeline",
                description="更新流水线配置",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "流水线名称",
                        },
                        "yaml_content": {
                            "type": "string",
                            "description": "YAML 配置内容",
                        },
                        "description": {
                            "type": "string",
                            "description": "描述（可选）",
                        },
                    },
                    "required": ["name", "yaml_content"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """调用工具."""
        logger.info("tool_call: %s", name)
        t0 = _time.monotonic()
        success = True
        error_type = ""
        try:
            result = await _handle_tool(name, arguments)
            return result
        except (KeyboardInterrupt, SystemExit):
            raise
        except (ValueError, TypeError, KeyError) as exc:
            # 预期的业务错误
            success = False
            error_type = type(exc).__name__
            logger.warning("tool_call_error: %s - %s", name, exc)
            return [TextContent(type="text", text=f"参数错误: {exc}")]
        except Exception as exc:
            # 未预期的系统错误
            success = False
            error_type = type(exc).__name__
            logger.exception("tool_call_fatal: %s", name)
            return [TextContent(type="text", text=f"内部错误: {name}")]
        finally:
            try:
                duration_ms = (_time.monotonic() - t0) * 1000
                _tool_metrics.record(
                    tool_name=name,
                    duration_ms=duration_ms,
                    success=success,
                    error_type=error_type,
                )
            except Exception:
                pass  # 埋点不能影响主流程

    async def _handle_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "list_employees":
            tag = arguments.get("tag")
            # 优先走远程 API
            remote_cfg = _get_remote_memory_config()
            if remote_cfg:
                try:
                    items = await _remote_list_employees(
                        remote_cfg[0],
                        remote_cfg[1],
                        tag=tag,
                    )
                    return [
                        TextContent(
                            type="text", text=json.dumps(items, ensure_ascii=False, indent=2)
                        )
                    ]
                except Exception as exc:
                    logger.warning("远程 list_employees 失败，fallback 到本地: %s", exc)
            # fallback 到本地
            result = discover_employees(project_dir=_project_dir)
            employees = list(result.employees.values())
            if tag:
                employees = [e for e in employees if tag in e.tags]
            data = [
                {
                    "name": e.name,
                    "display_name": e.effective_display_name,
                    "character_name": e.character_name,
                    "description": e.description,
                    "tags": e.tags,
                    "triggers": e.triggers,
                    "model": e.model,
                    "layer": e.source_layer,
                    "avatar_url": f"/static/avatars/{e.agent_id}.webp" if e.agent_id else None,
                }
                for e in employees
            ]
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        elif name == "get_employee":
            emp_name = arguments["name"]
            # 优先走远程 API
            remote_cfg = _get_remote_memory_config()
            if remote_cfg:
                try:
                    data = await _remote_get_employee(
                        remote_cfg[0],
                        remote_cfg[1],
                        name=emp_name,
                    )
                    return [
                        TextContent(
                            type="text", text=json.dumps(data, ensure_ascii=False, indent=2)
                        )
                    ]
                except Exception as exc:
                    logger.warning("远程 get_employee(%s) 失败，fallback 到本地: %s", emp_name, exc)
            # fallback 到本地
            result = discover_employees(project_dir=_project_dir)
            emp = result.get(emp_name)
            if emp is None:
                return [TextContent(type="text", text=f"未找到员工: {emp_name}")]
            data = emp.model_dump(
                mode="json",
                exclude={
                    "source_path",
                    "api_key",
                    "fallback_api_key",
                    "fallback_base_url",
                },
            )
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        elif name == "run_employee":
            emp_name = arguments["name"]
            emp_args = arguments.get("args", {})
            agent_id = arguments.get("agent_id")
            # 优先走远程 API 拿 prompt
            remote_cfg = _get_remote_memory_config()
            if remote_cfg:
                try:
                    prompt = await _remote_run_employee(
                        remote_cfg[0],
                        remote_cfg[1],
                        name=emp_name,
                        args=emp_args,
                        agent_id=agent_id,
                    )
                    if prompt:
                        # 记录工作日志
                        try:
                            log = WorkLogger(project_dir=_project_dir)
                            sid = log.create_session(emp_name, args=emp_args, agent_id=agent_id)
                            log.add_entry(sid, "prompt_generated", f"{len(prompt)} chars (remote)")
                        except Exception:
                            pass  # 日志失败不影响主流程
                        return [TextContent(type="text", text=prompt)]
                except Exception as exc:
                    logger.warning("远程 run_employee(%s) 失败，fallback 到本地: %s", emp_name, exc)
            # fallback 到本地
            result = discover_employees(project_dir=_project_dir)
            emp = result.get(emp_name)
            if emp is None:
                return [TextContent(type="text", text=f"未找到员工: {emp_name}")]
            engine = CrewEngine(project_dir=_project_dir)
            errors = engine.validate_args(emp, args=emp_args)
            if errors:
                return [TextContent(type="text", text=f"参数错误: {'; '.join(errors)}")]

            # 智能上下文检测
            project_info = detect_project(_project_dir)

            prompt = engine.prompt(emp, args=emp_args, project_info=project_info)

            # 记录工作日志
            try:
                log = WorkLogger(project_dir=_project_dir)
                sid = log.create_session(emp.name, args=emp_args, agent_id=agent_id)
                log.add_entry(sid, "prompt_generated", f"{len(prompt)} chars")
            except Exception:
                pass  # 日志失败不影响主流程

            return [TextContent(type="text", text=prompt)]

        elif name == "get_work_log":
            emp_name = arguments.get("employee_name")
            limit = arguments.get("limit", 10)
            # 优先走远程 API
            remote_cfg = _get_remote_memory_config()
            if remote_cfg:
                try:
                    items = await _remote_get_work_log(
                        remote_cfg[0],
                        remote_cfg[1],
                        employee_name=emp_name,
                        limit=limit,
                    )
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(items, ensure_ascii=False, indent=2),
                        )
                    ]
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "远程 get_work_log 失败，fallback 到本地: %s", exc
                    )
            # fallback 到本地
            work_logger = WorkLogger(project_dir=_project_dir)
            sessions = work_logger.list_sessions(employee_name=emp_name, limit=limit)
            return [
                TextContent(
                    type="text",
                    text=json.dumps(sessions, ensure_ascii=False, indent=2),
                )
            ]

        elif name == "detect_project":
            arg_project_dir = arguments.get("project_dir")
            if arg_project_dir:
                p = Path(arg_project_dir).resolve()
                if not p.is_relative_to(_project_dir):
                    return [TextContent(type="text", text="路径不在项目目录范围内")]
            info = detect_project(Path(arg_project_dir) if arg_project_dir else _project_dir)
            data = info.model_dump(mode="json")
            data["display_label"] = info.display_label
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        elif name == "list_pipelines":
            # 优先走远程 API
            remote_cfg = _get_remote_memory_config()
            if remote_cfg:
                try:
                    items = await _remote_list_pipelines(remote_cfg[0], remote_cfg[1])
                    return [
                        TextContent(
                            type="text", text=json.dumps(items, ensure_ascii=False, indent=2)
                        )
                    ]
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "远程 list_pipelines 失败，fallback 到本地: %s", exc
                    )
            # fallback 到本地
            pipelines = discover_pipelines(project_dir=_project_dir)

            def _step_summary(s):
                if hasattr(s, "employee"):
                    return s.employee
                if hasattr(s, "parallel"):
                    return [sub.employee for sub in s.parallel]
                if hasattr(s, "condition"):
                    return {"condition": [sub.employee for sub in s.condition.then]}
                if hasattr(s, "loop"):
                    return {"loop": [sub.employee for sub in s.loop.steps]}
                return "unknown"

            data = []
            for pname, ppath in pipelines.items():
                pl = load_pipeline(ppath)
                data.append(
                    {
                        "name": pname,
                        "description": pl.description,
                        "steps": [_step_summary(s) for s in pl.steps],
                        "path": str(ppath),
                    }
                )
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        elif name == "run_pipeline":
            pl_name = arguments["name"]
            pl_args = arguments.get("args", {})
            agent_id = arguments.get("agent_id")
            smart_context = arguments.get("smart_context", True)
            execute = arguments.get("execute", False)
            pl_model = arguments.get("model")

            # 查找流水线
            pl_path = Path(pl_name)
            if pl_path.is_absolute() and not pl_path.resolve().is_relative_to(_project_dir):
                return [TextContent(type="text", text="路径不在项目目录范围内")]
            if not pl_path.exists():
                pipelines = discover_pipelines(project_dir=_project_dir)
                if pl_name in pipelines:
                    pl_path = pipelines[pl_name]
                else:
                    return [TextContent(type="text", text=f"未找到流水线: {pl_name}")]

            pipeline = load_pipeline(pl_path)
            errors = validate_pipeline(pipeline, project_dir=_project_dir)
            if errors:
                return [TextContent(type="text", text=f"流水线校验失败: {'; '.join(errors)}")]

            # execute 模式需要 API key（自动从环境变量解析）
            api_key = None
            if execute:
                from crew.providers import detect_provider, resolve_api_key

                eff_model = pl_model or "claude-sonnet-4-20250514"
                try:
                    _prov = detect_provider(eff_model)
                    api_key = resolve_api_key(_prov)
                except ValueError as e:
                    return [TextContent(type="text", text=f"错误: {e}")]

            result = await arun_pipeline(
                pipeline,
                initial_args=pl_args,
                agent_id=agent_id,
                smart_context=smart_context,
                project_dir=_project_dir,
                execute=execute,
                api_key=api_key,
                model=pl_model,
            )
            return [TextContent(type="text", text=result.model_dump_json(indent=2))]

        elif name == "list_discussions":
            # 优先走远程 API
            remote_cfg = _get_remote_memory_config()
            if remote_cfg:
                try:
                    items = await _remote_list_discussions(remote_cfg[0], remote_cfg[1])
                    return [
                        TextContent(
                            type="text", text=json.dumps(items, ensure_ascii=False, indent=2)
                        )
                    ]
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "远程 list_discussions 失败，fallback 到本地: %s", exc
                    )
            # fallback 到本地
            from crew.discussion import discover_discussions, load_discussion

            discussions = discover_discussions(project_dir=_project_dir)
            data = []
            for dname, dpath in discussions.items():
                try:
                    d = load_discussion(dpath)
                    rounds_count = d.rounds if isinstance(d.rounds, int) else len(d.rounds)
                    data.append(
                        {
                            "name": dname,
                            "description": d.description,
                            "participants": [p.employee for p in d.participants],
                            "rounds": rounds_count,
                            "path": str(dpath),
                        }
                    )
                except Exception:
                    data.append({"name": dname, "error": "解析失败", "path": str(dpath)})
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        elif name == "run_discussion":
            from crew.discussion import (
                create_adhoc_discussion,
                discover_discussions,
                load_discussion,
                render_discussion,
                render_discussion_plan,
                validate_discussion,
            )

            d_args = arguments.get("args", {})
            agent_id = arguments.get("agent_id")
            smart_context = arguments.get("smart_context", True)
            is_orchestrated = arguments.get("orchestrated", False)

            employees_list = arguments.get("employees")
            adhoc_topic = arguments.get("topic")

            if employees_list and adhoc_topic:
                # 即席讨论模式 — 保持本地（不走远程）
                discussion = create_adhoc_discussion(
                    employees=employees_list,
                    topic=adhoc_topic,
                    goal=arguments.get("goal", ""),
                    rounds=arguments.get("rounds", 2),
                    round_template=arguments.get("round_template"),
                )
            elif "name" in arguments:
                d_name = arguments["name"]
                # 预定义讨论会 — 优先走远程 API
                remote_cfg = _get_remote_memory_config()
                if remote_cfg:
                    try:
                        if is_orchestrated:
                            plan_data = await _remote_run_discussion_plan(
                                remote_cfg[0],
                                remote_cfg[1],
                                name=d_name,
                                args=d_args,
                                agent_id=agent_id,
                                smart_context=smart_context,
                            )
                            return [
                                TextContent(
                                    type="text",
                                    text=json.dumps(plan_data, ensure_ascii=False, indent=2),
                                )
                            ]
                        else:
                            prompt = await _remote_run_discussion_prompt(
                                remote_cfg[0],
                                remote_cfg[1],
                                name=d_name,
                                args=d_args,
                                agent_id=agent_id,
                                smart_context=smart_context,
                            )
                            return [TextContent(type="text", text=prompt)]
                    except Exception as exc:
                        logging.getLogger(__name__).warning(
                            "远程 run_discussion(%s) 失败，fallback 到本地: %s",
                            d_name,
                            exc,
                        )
                # fallback 到本地
                d_path = Path(d_name)
                if d_path.is_absolute() and not d_path.resolve().is_relative_to(_project_dir):
                    return [TextContent(type="text", text="路径不在项目目录范围内")]
                if not d_path.exists():
                    discussions = discover_discussions(project_dir=_project_dir)
                    if d_name in discussions:
                        d_path = discussions[d_name]
                    else:
                        return [TextContent(type="text", text=f"未找到讨论会: {d_name}")]
                discussion = load_discussion(d_path)
            else:
                return [TextContent(type="text", text="请提供 name 或 employees+topic")]

            errors = validate_discussion(discussion, project_dir=_project_dir)
            if errors:
                return [TextContent(type="text", text=f"讨论会校验失败: {'; '.join(errors)}")]

            if is_orchestrated:
                plan = render_discussion_plan(
                    discussion,
                    initial_args=d_args,
                    agent_id=agent_id,
                    smart_context=smart_context,
                    project_dir=_project_dir,
                )
                return [
                    TextContent(
                        type="text",
                        text=plan.model_dump_json(indent=2),
                    )
                ]
            else:
                prompt = render_discussion(
                    discussion,
                    initial_args=d_args,
                    agent_id=agent_id,
                    smart_context=smart_context,
                    project_dir=_project_dir,
                )
                return [TextContent(type="text", text=prompt)]

        elif name == "add_memory":
            remote_cfg = _get_remote_memory_config()
            if remote_cfg:
                # 远程 API 路径
                base_url, api_token = remote_cfg
                try:
                    result = await _remote_memory_add(
                        base_url,
                        api_token,
                        employee=arguments["employee"],
                        category=arguments["category"],
                        content=arguments["content"],
                        source_session=arguments.get("source_session", ""),
                        ttl_days=arguments.get("ttl_days", 0),
                        tags=arguments.get("tags"),
                        shared=arguments.get("shared", False),
                        trigger_condition=arguments.get("trigger_condition", ""),
                        applicability=arguments.get("applicability"),
                        origin_employee=arguments.get("origin_employee", ""),
                    )
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(result, ensure_ascii=False, indent=2),
                        )
                    ]
                except Exception as exc:
                    logging.getLogger(__name__).warning("远程 memory add 失败: %s", exc)
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {"error": f"远程写入失败: {exc}", "remote_url": base_url},
                                ensure_ascii=False,
                            ),
                        )
                    ]
            else:
                # 本地 fallback（未配置远程 API 时）
                from crew.memory import MemoryStore

                store = MemoryStore(project_dir=_project_dir)
                entry = store.add(
                    employee=arguments["employee"],
                    category=arguments["category"],
                    content=arguments["content"],
                    source_session=arguments.get("source_session", ""),
                    ttl_days=arguments.get("ttl_days", 0),
                    tags=arguments.get("tags"),
                    shared=arguments.get("shared", False),
                    trigger_condition=arguments.get("trigger_condition", ""),
                    applicability=arguments.get("applicability"),
                    origin_employee=arguments.get("origin_employee", ""),
                )
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(entry.model_dump(), ensure_ascii=False, indent=2),
                    )
                ]

        elif name == "query_memory":
            remote_cfg = _get_remote_memory_config()
            if remote_cfg:
                # 远程 API 路径
                base_url, api_token = remote_cfg
                try:
                    entries_data = await _remote_memory_query(
                        base_url,
                        api_token,
                        employee=arguments["employee"],
                        category=arguments.get("category"),
                        limit=arguments.get("limit", 20),
                    )
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(entries_data, ensure_ascii=False, indent=2),
                        )
                    ]
                except Exception as exc:
                    logging.getLogger(__name__).warning("远程 memory query 失败: %s", exc)
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {"error": f"远程查询失败: {exc}", "remote_url": base_url},
                                ensure_ascii=False,
                            ),
                        )
                    ]
            else:
                # 本地 fallback（未配置远程 API 时）
                from crew.memory import MemoryStore

                store = MemoryStore(project_dir=_project_dir)
                entries = store.query(
                    employee=arguments["employee"],
                    category=arguments.get("category"),
                    limit=arguments.get("limit", 20),
                )
                data = [e.model_dump() for e in entries]
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(data, ensure_ascii=False, indent=2),
                    )
                ]

        elif name == "track_decision":
            # 优先走远程 API
            remote_cfg = _get_remote_memory_config()
            if remote_cfg:
                try:
                    result = await _remote_track_decision(
                        remote_cfg[0],
                        remote_cfg[1],
                        employee=arguments["employee"],
                        category=arguments["category"],
                        content=arguments["content"],
                        expected_outcome=arguments.get("expected_outcome", ""),
                        meeting_id=arguments.get("meeting_id", ""),
                    )
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(result, ensure_ascii=False, indent=2),
                        )
                    ]
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "远程 track_decision 失败，fallback 到本地: %s", exc
                    )
            # fallback 到本地
            from crew.evaluation import EvaluationEngine

            engine = EvaluationEngine(project_dir=_project_dir)
            decision = engine.track(
                employee=arguments["employee"],
                category=arguments["category"],
                content=arguments["content"],
                expected_outcome=arguments.get("expected_outcome", ""),
                meeting_id=arguments.get("meeting_id", ""),
            )
            return [
                TextContent(
                    type="text",
                    text=json.dumps(decision.model_dump(), ensure_ascii=False, indent=2),
                )
            ]

        elif name == "evaluate_decision":
            decision_id = arguments["decision_id"]
            # 优先走远程 API
            remote_cfg = _get_remote_memory_config()
            if remote_cfg:
                try:
                    result = await _remote_evaluate_decision(
                        remote_cfg[0],
                        remote_cfg[1],
                        decision_id=decision_id,
                        actual_outcome=arguments["actual_outcome"],
                        evaluation=arguments.get("evaluation", ""),
                    )
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(result, ensure_ascii=False, indent=2),
                        )
                    ]
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "远程 evaluate_decision(%s) 失败，fallback 到本地: %s",
                        decision_id,
                        exc,
                    )
            # fallback 到本地
            from crew.evaluation import EvaluationEngine

            engine = EvaluationEngine(project_dir=_project_dir)
            decision = engine.evaluate(
                decision_id=decision_id,
                actual_outcome=arguments["actual_outcome"],
                evaluation=arguments.get("evaluation", ""),
            )
            if decision is None:
                return [TextContent(type="text", text=f"未找到决策: {decision_id}")]
            return [
                TextContent(
                    type="text",
                    text=json.dumps(decision.model_dump(), ensure_ascii=False, indent=2),
                )
            ]

        elif name == "list_meeting_history":
            limit = arguments.get("limit", 20)
            keyword = arguments.get("keyword")
            # 优先走远程 API
            remote_cfg = _get_remote_memory_config()
            if remote_cfg:
                try:
                    items = await _remote_list_meeting_history(
                        remote_cfg[0],
                        remote_cfg[1],
                        limit=limit,
                        keyword=keyword,
                    )
                    return [
                        TextContent(
                            type="text", text=json.dumps(items, ensure_ascii=False, indent=2)
                        )
                    ]
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "远程 list_meeting_history 失败，fallback 到本地: %s", exc
                    )
            # fallback 到本地
            from crew.meeting_log import MeetingLogger

            meeting_logger = MeetingLogger(project_dir=_project_dir)
            records = meeting_logger.list(limit=limit, keyword=keyword)
            data = [r.model_dump() for r in records]
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        elif name == "get_meeting_detail":
            meeting_id = arguments["meeting_id"]
            # 优先走远程 API
            remote_cfg = _get_remote_memory_config()
            if remote_cfg:
                try:
                    data = await _remote_get_meeting_detail(
                        remote_cfg[0],
                        remote_cfg[1],
                        meeting_id=meeting_id,
                    )
                    return [
                        TextContent(
                            type="text", text=json.dumps(data, ensure_ascii=False, indent=2)
                        )
                    ]
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "远程 get_meeting_detail(%s) 失败，fallback 到本地: %s",
                        meeting_id,
                        exc,
                    )
            # fallback 到本地
            from crew.meeting_log import MeetingLogger

            meeting_logger = MeetingLogger(project_dir=_project_dir)
            result = meeting_logger.get(meeting_id)
            if result is None:
                return [TextContent(type="text", text=f"未找到会议: {meeting_id}")]
            record, content = result
            data = {**record.model_dump(), "content": content}
            return [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, indent=2))]

        elif name == "list_tool_schemas":
            from crew.tool_schema import _TOOL_SCHEMAS, TOOL_ROLE_PRESETS

            role = arguments.get("role")
            if role:
                preset = TOOL_ROLE_PRESETS.get(role)
                if preset is None:
                    available = sorted(TOOL_ROLE_PRESETS.keys())
                    return [
                        TextContent(
                            type="text",
                            text=f"未知角色: {role}\n可用角色: {', '.join(available)}",
                        )
                    ]
                tool_names = sorted(preset)
            else:
                tool_names = sorted(_TOOL_SCHEMAS.keys())
            data = [
                {"name": t, "description": _TOOL_SCHEMAS[t]["description"]}
                for t in tool_names
                if t in _TOOL_SCHEMAS
            ]
            return [
                TextContent(
                    type="text",
                    text=json.dumps(data, ensure_ascii=False, indent=2),
                )
            ]

        elif name == "get_permission_matrix":
            emp_name = arguments.get("employee")
            # 优先走远程 API
            remote_cfg = _get_remote_memory_config()
            if remote_cfg:
                try:
                    items = await _remote_get_permission_matrix(
                        remote_cfg[0],
                        remote_cfg[1],
                        employee=emp_name,
                    )
                    return [
                        TextContent(
                            type="text", text=json.dumps(items, ensure_ascii=False, indent=2)
                        )
                    ]
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "远程 get_permission_matrix 失败，fallback 到本地: %s", exc
                    )
            # fallback 到本地
            from crew.tool_schema import TOOL_ROLE_PRESETS, resolve_effective_tools

            result = discover_employees(project_dir=_project_dir)
            employees = list(result.employees.values())
            if emp_name:
                emp = result.get(emp_name)
                if emp is None:
                    return [TextContent(type="text", text=f"未找到员工: {emp_name}")]
                employees = [emp]
            matrix = []
            for emp in employees:
                effective = resolve_effective_tools(emp)
                entry = {
                    "name": emp.name,
                    "display_name": emp.effective_display_name,
                    "tools_declared": len(emp.tools),
                    "tools_effective": len(effective),
                    "permissions": (
                        emp.permissions.model_dump(mode="json") if emp.permissions else None
                    ),
                    "effective_tools": sorted(effective),
                }
                matrix.append(entry)
            return [
                TextContent(
                    type="text",
                    text=json.dumps(matrix, ensure_ascii=False, indent=2),
                )
            ]

        elif name == "get_audit_log":
            from crew.permission import get_audit_logger

            audit = get_audit_logger()
            log_path = audit._ensure_dir()
            if not log_path.exists():
                return [TextContent(type="text", text="暂无审计日志")]
            emp_filter = arguments.get("employee")
            limit = arguments.get("limit", 50)
            denied_only = arguments.get("denied_only", False)
            records = []
            for line in log_path.read_text(encoding="utf-8").strip().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if emp_filter and record.get("employee") != emp_filter:
                    continue
                if denied_only and record.get("allowed", True):
                    continue
                records.append(record)
            records = records[-limit:]
            return [
                TextContent(
                    type="text",
                    text=json.dumps(records, ensure_ascii=False, indent=2),
                )
            ]

        elif name == "get_tool_metrics":
            tool_filter = arguments.get("tool_name")
            since = arguments.get("since")
            if since:
                # 有时间范围：从持久化读取
                data = _tool_metrics.snapshot_persistent(tool_name=tool_filter, since=since)
            else:
                # 无时间范围：返回 memory + persistent 两层数据
                memory_snap = _tool_metrics.snapshot(tool_name=tool_filter)
                persistent_snap = _tool_metrics.snapshot_persistent(tool_name=tool_filter)
                data = {
                    "memory": {
                        "source": "memory",
                        "total_tool_calls": memory_snap["total_tool_calls"],
                        "uptime_seconds": memory_snap["uptime_seconds"],
                        "tools": memory_snap["tools"],
                    },
                    "persistent": persistent_snap,
                }
            return [
                TextContent(
                    type="text",
                    text=json.dumps(data, ensure_ascii=False, indent=2),
                )
            ]

        elif name == "query_events":
            from crew.event_collector import get_event_collector

            ec = get_event_collector()
            is_aggregate = arguments.get("aggregate", False)
            if is_aggregate:
                data = ec.aggregate(
                    event_type=arguments.get("event_type"),
                    since=arguments.get("since"),
                )
            else:
                limit = min(arguments.get("limit", 100), 1000)
                data = ec.query(
                    event_type=arguments.get("event_type"),
                    event_name=arguments.get("event_name"),
                    since=arguments.get("since"),
                    until=arguments.get("until"),
                    limit=limit,
                )
            return [
                TextContent(
                    type="text",
                    text=json.dumps(data, ensure_ascii=False, indent=2),
                )
            ]

        elif name == "put_config":
            key = arguments["key"]
            content = arguments["content"]
            remote_cfg = _get_remote_memory_config()
            if remote_cfg:
                base_url, api_token = remote_cfg
                try:
                    result = await _remote_kv_put(base_url, api_token, key=key, content=content)
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(result, ensure_ascii=False, indent=2),
                        )
                    ]
                except Exception as exc:
                    logging.getLogger(__name__).warning("远程 KV put 失败: %s", exc)
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {"error": f"远程写入失败: {exc}", "remote_url": base_url},
                                ensure_ascii=False,
                            ),
                        )
                    ]
            else:
                # 本地 fallback
                import re as _re_kv

                _KV_KEY_RE = _re_kv.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_/-]*$")
                if ".." in key or "." in key or key.startswith("/") or not _KV_KEY_RE.match(key):
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps({"error": "invalid key"}, ensure_ascii=False),
                        )
                    ]
                base_dir = (_project_dir or Path(".")) / ".crew" / "kv"
                file_path = (base_dir / key).resolve()

                # 先验证路径，再操作
                try:
                    file_path.relative_to(base_dir.resolve())
                except ValueError:
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {"error": "path traversal detected"}, ensure_ascii=False
                            ),
                        )
                    ]

                # 检查是否为符号链接
                if file_path.is_symlink():
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps({"error": "symlink not allowed"}, ensure_ascii=False),
                        )
                    ]

                file_path.parent.mkdir(parents=True, exist_ok=True)
                raw_bytes = content.encode("utf-8")
                file_path.write_bytes(raw_bytes)
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"ok": True, "key": key, "size": len(raw_bytes)}, ensure_ascii=False
                        ),
                    )
                ]

        elif name == "get_config":
            key = arguments["key"]
            remote_cfg = _get_remote_memory_config()
            if remote_cfg:
                base_url, api_token = remote_cfg
                try:
                    text = await _remote_kv_get(base_url, api_token, key=key)
                    return [TextContent(type="text", text=text)]
                except Exception as exc:
                    logging.getLogger(__name__).warning("远程 KV get 失败: %s", exc)
                    error_msg = str(exc)
                    if "404" in error_msg:
                        return [
                            TextContent(
                                type="text",
                                text=json.dumps(
                                    {"error": "not found", "key": key}, ensure_ascii=False
                                ),
                            )
                        ]
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {"error": f"远程读取失败: {exc}", "remote_url": base_url},
                                ensure_ascii=False,
                            ),
                        )
                    ]
            else:
                # 本地 fallback
                import re as _re_kv

                _KV_KEY_RE = _re_kv.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_./-]*$")
                if ".." in key or key.startswith("/") or not _KV_KEY_RE.match(key):
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps({"error": "invalid key"}, ensure_ascii=False),
                        )
                    ]
                base_dir = (_project_dir or Path(".")) / ".crew" / "kv"
                file_path = base_dir / key
                try:
                    file_path.resolve().relative_to(base_dir.resolve())
                except ValueError:
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {"error": "path traversal detected"}, ensure_ascii=False
                            ),
                        )
                    ]
                if not file_path.is_file():
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps({"error": "not found", "key": key}, ensure_ascii=False),
                        )
                    ]
                return [TextContent(type="text", text=file_path.read_text(encoding="utf-8"))]

        elif name == "list_configs":
            prefix = arguments.get("prefix", "")
            remote_cfg = _get_remote_memory_config()
            if remote_cfg:
                base_url, api_token = remote_cfg
                try:
                    keys = await _remote_kv_list(base_url, api_token, prefix=prefix)
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {"ok": True, "keys": keys}, ensure_ascii=False, indent=2
                            ),
                        )
                    ]
                except Exception as exc:
                    logging.getLogger(__name__).warning("远程 KV list 失败: %s", exc)
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {"error": f"远程列表失败: {exc}", "remote_url": base_url},
                                ensure_ascii=False,
                            ),
                        )
                    ]
            else:
                # 本地 fallback
                if prefix and (".." in prefix or prefix.startswith("/")):
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps({"error": "invalid prefix"}, ensure_ascii=False),
                        )
                    ]
                base_dir = (_project_dir or Path(".")) / ".crew" / "kv"
                scan_dir = base_dir / prefix if prefix else base_dir
                try:
                    scan_dir.resolve().relative_to(base_dir.resolve())
                except ValueError:
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {"error": "path traversal detected"}, ensure_ascii=False
                            ),
                        )
                    ]
                keys: list[str] = []
                if scan_dir.is_dir():
                    for p in sorted(scan_dir.rglob("*")):
                        if p.is_file():
                            rel = p.relative_to(base_dir)
                            keys.append(str(rel))
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"ok": True, "keys": keys}, ensure_ascii=False, indent=2),
                    )
                ]

        elif name == "wiki_upload":
            wiki_cfg = _get_wiki_config()
            if not wiki_cfg:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "Wiki API 未配置，请设置 WIKI_API_URL 和 WIKI_API_TOKEN 环境变量"
                            },
                            ensure_ascii=False,
                        ),
                    )
                ]
            base_url, wiki_token = wiki_cfg
            file_path_arg = arguments.get("file_path")
            content_arg = arguments.get("content")
            filename_arg = arguments.get("filename")
            space_slug = arguments.get("space_slug", "")
            doc_slug = arguments.get("doc_slug", "")

            if file_path_arg:
                # 从本地文件读取
                fp = Path(file_path_arg).expanduser().resolve()
                if not fp.is_file():
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {"error": f"文件不存在: {file_path_arg}"}, ensure_ascii=False
                            ),
                        )
                    ]
                file_bytes = fp.read_bytes()
                filename = fp.name
                # 猜测 MIME 类型
                import mimetypes

                ct, _ = mimetypes.guess_type(filename)
                content_type = ct or "application/octet-stream"
            elif content_arg and filename_arg:
                # 从 base64 内容解码
                import base64

                try:
                    file_bytes = base64.b64decode(content_arg)
                except Exception:
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps({"error": "base64 解码失败"}, ensure_ascii=False),
                        )
                    ]
                filename = filename_arg
                import mimetypes

                ct, _ = mimetypes.guess_type(filename)
                content_type = ct or "application/octet-stream"
            else:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": "请提供 file_path 或 content+filename"},
                            ensure_ascii=False,
                        ),
                    )
                ]

            try:
                result = await _wiki_upload_file(
                    base_url,
                    wiki_token,
                    file_bytes=file_bytes,
                    filename=filename,
                    content_type=content_type,
                    space_slug=space_slug,
                    doc_slug=doc_slug,
                )
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result, ensure_ascii=False, indent=2),
                    )
                ]
            except Exception as exc:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"上传失败: {exc}"},
                            ensure_ascii=False,
                        ),
                    )
                ]

        elif name == "wiki_read_file":
            wiki_cfg = _get_wiki_config()
            if not wiki_cfg:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "Wiki API 未配置，请设置 WIKI_API_URL 和 WIKI_API_TOKEN 环境变量"
                            },
                            ensure_ascii=False,
                        ),
                    )
                ]
            base_url, wiki_token = wiki_cfg
            file_id_arg = arguments.get("file_id")
            filename_arg = arguments.get("filename")

            if file_id_arg is not None:
                # 直接按 ID 查
                try:
                    fid = int(file_id_arg)
                except (ValueError, TypeError):
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {"error": f"无效的 file_id: {file_id_arg}"}, ensure_ascii=False
                            ),
                        )
                    ]
                try:
                    result = await _wiki_get_file(base_url, wiki_token, file_id=fid)
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(result, ensure_ascii=False, indent=2),
                        )
                    ]
                except Exception as exc:
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps({"error": f"获取文件失败: {exc}"}, ensure_ascii=False),
                        )
                    ]
            elif filename_arg:
                # 按文件名模糊搜索：先 list 再找匹配
                try:
                    list_result = await _wiki_list_files(base_url, wiki_token, page=1, page_size=50)
                    files = list_result.get("files", list_result.get("items", []))
                    # 模糊匹配：文件名包含搜索词（大小写不敏感）
                    keyword = filename_arg.lower()
                    matched = [
                        f
                        for f in files
                        if keyword in f.get("filename", "").lower()
                        or keyword in f.get("original_filename", "").lower()
                    ]
                    if not matched:
                        return [
                            TextContent(
                                type="text",
                                text=json.dumps(
                                    {
                                        "error": f"未找到匹配的文件: {filename_arg}",
                                        "total_files": len(files),
                                    },
                                    ensure_ascii=False,
                                ),
                            )
                        ]
                    if len(matched) == 1:
                        # 只有一个匹配，直接获取详情
                        fid = matched[0].get("id") or matched[0].get("file_id")
                        result = await _wiki_get_file(base_url, wiki_token, file_id=int(fid))
                        return [
                            TextContent(
                                type="text",
                                text=json.dumps(result, ensure_ascii=False, indent=2),
                            )
                        ]
                    else:
                        # 多个匹配，返回列表让用户选择
                        return [
                            TextContent(
                                type="text",
                                text=json.dumps(
                                    {
                                        "message": f"找到 {len(matched)} 个匹配文件，请用 file_id 指定",
                                        "matches": matched,
                                    },
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                            )
                        ]
                except Exception as exc:
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps({"error": f"搜索文件失败: {exc}"}, ensure_ascii=False),
                        )
                    ]
            else:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": "请提供 file_id 或 filename"},
                            ensure_ascii=False,
                        ),
                    )
                ]

        elif name == "wiki_list_files":
            wiki_cfg = _get_wiki_config()
            if not wiki_cfg:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "Wiki API 未配置，请设置 WIKI_API_URL 和 WIKI_API_TOKEN 环境变量"
                            },
                            ensure_ascii=False,
                        ),
                    )
                ]
            base_url, wiki_token = wiki_cfg
            try:
                result = await _wiki_list_files(
                    base_url,
                    wiki_token,
                    space_slug=arguments.get("space_slug", ""),
                    doc_slug=arguments.get("doc_slug", ""),
                    mime_type=arguments.get("mime_type", ""),
                    page=arguments.get("page", 1),
                    page_size=arguments.get("page_size", 20),
                )
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result, ensure_ascii=False, indent=2),
                    )
                ]
            except Exception as exc:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": f"列表失败: {exc}"}, ensure_ascii=False),
                    )
                ]

        elif name == "wiki_delete_file":
            file_id_arg = arguments.get("file_id")
            if file_id_arg is None:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": "缺少必填参数: file_id"}, ensure_ascii=False),
                    )
                ]
            try:
                fid = int(file_id_arg)
            except (ValueError, TypeError):
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"无效的 file_id: {file_id_arg}"}, ensure_ascii=False
                        ),
                    )
                ]

            # 优先走远程 Crew API
            remote_cfg = _get_remote_memory_config()
            if remote_cfg:
                import httpx

                try:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        resp = await client.delete(
                            f"{remote_cfg[0]}/api/wiki/files/{fid}",
                            headers={"Authorization": f"Bearer {remote_cfg[1]}"},
                        )
                        resp.raise_for_status()
                        return [
                            TextContent(
                                type="text",
                                text=json.dumps(resp.json(), ensure_ascii=False, indent=2),
                            )
                        ]
                except Exception as exc:
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps({"error": f"删除文件失败: {exc}"}, ensure_ascii=False),
                        )
                    ]

            # 本地: 直接调用 Wiki 后端
            wiki_cfg = _get_wiki_config()
            if not wiki_cfg:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "Wiki API 未配置，请设置 WIKI_API_URL 和 WIKI_API_TOKEN 环境变量"
                            },
                            ensure_ascii=False,
                        ),
                    )
                ]
            base_url, wiki_token = wiki_cfg
            try:
                result = await _wiki_delete_file(base_url, wiki_token, file_id=fid)
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result, ensure_ascii=False, indent=2),
                    )
                ]
            except Exception as exc:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": f"删除文件失败: {exc}"}, ensure_ascii=False),
                    )
                ]

        elif name == "wiki_create_doc":
            admin_cfg = _get_wiki_admin_config()
            if not admin_cfg:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "Wiki Admin API 未配置，请设置 WIKI_API_URL 和 WIKI_ADMIN_TOKEN（或 ANTGATHER_API_TOKEN）环境变量"
                            },
                            ensure_ascii=False,
                        ),
                    )
                ]
            base_url, admin_token = admin_cfg
            space_slug = arguments.get("space_slug", "")
            title = arguments.get("title", "")
            slug = arguments.get("slug", "")

            if not space_slug or not title or not slug:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": "缺少必填参数：space_slug, title, slug"},
                            ensure_ascii=False,
                        ),
                    )
                ]

            # slug 转 space_id
            try:
                space_id = await _wiki_resolve_space_id(base_url, admin_token, space_slug)
            except Exception as exc:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"查询空间列表失败: {exc}"},
                            ensure_ascii=False,
                        ),
                    )
                ]

            if space_id is None:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": f"未找到空间: {space_slug}，可用空间请通过 wiki_list_files 查看"
                            },
                            ensure_ascii=False,
                        ),
                    )
                ]

            try:
                result = await _wiki_create_doc(
                    base_url,
                    admin_token,
                    space_id=space_id,
                    title=title,
                    slug=slug,
                    content=arguments.get("content", ""),
                    ai_content=arguments.get("ai_content", ""),
                    excerpt=arguments.get("excerpt", ""),
                    visibility=arguments.get("visibility", "internal"),
                    parent_id=arguments.get("parent_id"),
                    sort_order=arguments.get("sort_order", 0),
                    is_pinned=arguments.get("is_pinned", False),
                )
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result, ensure_ascii=False, indent=2),
                    )
                ]
            except Exception as exc:
                import httpx as _httpx

                # 409 slug 冲突 → 自动 fallback 到更新已有文档
                if isinstance(exc, _httpx.HTTPStatusError) and exc.response.status_code == 409:
                    try:
                        conflict_detail = exc.response.json().get("detail", {})
                        existing = conflict_detail.get("existing_doc", {}) if isinstance(conflict_detail, dict) else {}
                        existing_id = existing.get("id")
                    except Exception:
                        existing_id = None

                    if existing_id:
                        try:
                            update_result = await _wiki_update_doc(
                                base_url,
                                admin_token,
                                doc_id=existing_id,
                                title=title,
                                content=arguments.get("content") or None,
                                ai_content=arguments.get("ai_content") or None,
                                excerpt=arguments.get("excerpt") or None,
                            )
                            update_result["_fallback"] = f"slug '{slug}' 已存在 (doc_id={existing_id})，已自动更新"
                            return [
                                TextContent(
                                    type="text",
                                    text=json.dumps(update_result, ensure_ascii=False, indent=2),
                                )
                            ]
                        except Exception as update_exc:
                            return [
                                TextContent(
                                    type="text",
                                    text=json.dumps(
                                        {"error": f"slug 冲突后 fallback 更新也失败: {update_exc}"},
                                        ensure_ascii=False,
                                    ),
                                )
                            ]

                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"创建文档失败: {exc}"},
                            ensure_ascii=False,
                        ),
                    )
                ]

        elif name == "get_soul":
            remote_cfg = _get_remote_memory_config()
            if not remote_cfg:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": "远程 API 未配置"}, ensure_ascii=False),
                    )
                ]
            base_url, api_token = remote_cfg
            employee_name = arguments["employee_name"]
            try:
                import httpx

                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(
                        f"{base_url}/api/souls/{employee_name}",
                        headers={"Authorization": f"Bearer {api_token}"},
                    )
                    if resp.status_code == 404:
                        return [
                            TextContent(
                                type="text",
                                text=json.dumps(
                                    {"error": f"soul not found: {employee_name}"},
                                    ensure_ascii=False,
                                ),
                            )
                        ]
                    resp.raise_for_status()
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(resp.json(), ensure_ascii=False, indent=2),
                        )
                    ]
            except Exception as exc:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": str(exc)}, ensure_ascii=False),
                    )
                ]

        elif name == "update_soul":
            remote_cfg = _get_remote_memory_config()
            if not remote_cfg:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": "远程 API 未配置"}, ensure_ascii=False),
                    )
                ]
            base_url, api_token = remote_cfg
            employee_name = arguments["employee_name"]
            content = arguments["content"]
            updated_by = arguments.get("updated_by", "")
            try:
                import httpx

                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.put(
                        f"{base_url}/api/souls/{employee_name}",
                        json={"content": content, "updated_by": updated_by},
                        headers={"Authorization": f"Bearer {api_token}"},
                    )
                    resp.raise_for_status()
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(resp.json(), ensure_ascii=False, indent=2),
                        )
                    ]
            except Exception as exc:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": str(exc)}, ensure_ascii=False),
                    )
                ]

        elif name == "create_employee":
            remote_cfg = _get_remote_memory_config()
            if not remote_cfg:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": "远程 API 未配置"}, ensure_ascii=False),
                    )
                ]
            base_url, api_token = remote_cfg
            try:
                import httpx

                payload = {
                    "name": arguments["name"],
                    "character_name": arguments["character_name"],
                    "soul_content": arguments["soul_content"],
                    "display_name": arguments.get("display_name", ""),
                    "description": arguments.get("description", ""),
                    "model": arguments.get("model", "claude-sonnet-4-6"),
                    "model_tier": arguments.get("model_tier", "claude"),
                    "tags": arguments.get("tags", []),
                    "avatar_prompt": arguments.get("avatar_prompt", ""),
                    "agent_status": arguments.get("agent_status", "active"),
                }

                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        f"{base_url}/api/employees",
                        json=payload,
                        headers={"Authorization": f"Bearer {api_token}"},
                    )
                    resp.raise_for_status()
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(resp.json(), ensure_ascii=False, indent=2),
                        )
                    ]
            except Exception as exc:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": str(exc)}, ensure_ascii=False),
                    )
                ]

        elif name == "create_discussion":
            remote_cfg = _get_remote_memory_config()
            if not remote_cfg:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": "远程 API 未配置"}, ensure_ascii=False),
                    )
                ]
            base_url, api_token = remote_cfg
            try:
                import httpx

                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(
                        f"{base_url}/api/config/discussions",
                        json={
                            "name": arguments["name"],
                            "yaml_content": arguments["yaml_content"],
                            "description": arguments.get("description", ""),
                        },
                        headers={"Authorization": f"Bearer {api_token}"},
                    )
                    resp.raise_for_status()
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(resp.json(), ensure_ascii=False, indent=2),
                        )
                    ]
            except Exception as exc:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": str(exc)}, ensure_ascii=False),
                    )
                ]

        elif name == "update_discussion":
            remote_cfg = _get_remote_memory_config()
            if not remote_cfg:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": "远程 API 未配置"}, ensure_ascii=False),
                    )
                ]
            base_url, api_token = remote_cfg
            try:
                import httpx

                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.put(
                        f"{base_url}/api/config/discussions/{arguments['name']}",
                        json={
                            "yaml_content": arguments["yaml_content"],
                            "description": arguments.get("description"),
                        },
                        headers={"Authorization": f"Bearer {api_token}"},
                    )
                    resp.raise_for_status()
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(resp.json(), ensure_ascii=False, indent=2),
                        )
                    ]
            except Exception as exc:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": str(exc)}, ensure_ascii=False),
                    )
                ]

        elif name == "create_pipeline":
            remote_cfg = _get_remote_memory_config()
            if not remote_cfg:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": "远程 API 未配置"}, ensure_ascii=False),
                    )
                ]
            base_url, api_token = remote_cfg
            try:
                import httpx

                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(
                        f"{base_url}/api/config/pipelines",
                        json={
                            "name": arguments["name"],
                            "yaml_content": arguments["yaml_content"],
                            "description": arguments.get("description", ""),
                        },
                        headers={"Authorization": f"Bearer {api_token}"},
                    )
                    resp.raise_for_status()
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(resp.json(), ensure_ascii=False, indent=2),
                        )
                    ]
            except Exception as exc:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": str(exc)}, ensure_ascii=False),
                    )
                ]

        elif name == "update_pipeline":
            remote_cfg = _get_remote_memory_config()
            if not remote_cfg:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": "远程 API 未配置"}, ensure_ascii=False),
                    )
                ]
            base_url, api_token = remote_cfg
            try:
                import httpx

                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.put(
                        f"{base_url}/api/config/pipelines/{arguments['name']}",
                        json={
                            "yaml_content": arguments["yaml_content"],
                            "description": arguments.get("description"),
                        },
                        headers={"Authorization": f"Bearer {api_token}"},
                    )
                    resp.raise_for_status()
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(resp.json(), ensure_ascii=False, indent=2),
                        )
                    ]
            except Exception as exc:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({"error": str(exc)}, ensure_ascii=False),
                    )
                ]

        elif name == "wiki_update_doc":
            admin_cfg = _get_wiki_admin_config()
            if not admin_cfg:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "error": "Wiki Admin API 未配置，请设置 WIKI_API_URL 和 WIKI_ADMIN_TOKEN（或 ANTGATHER_API_TOKEN）环境变量"
                            },
                            ensure_ascii=False,
                        ),
                    )
                ]
            base_url, admin_token = admin_cfg
            doc_id = arguments.get("doc_id")
            space_slug = arguments.get("space_slug", "")
            slug = arguments.get("slug", "")

            # 定位文档：优先 doc_id，其次 space_slug + slug
            if doc_id is not None:
                try:
                    doc_id = int(doc_id)
                except (ValueError, TypeError):
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {"error": f"无效的 doc_id: {doc_id}"}, ensure_ascii=False
                            ),
                        )
                    ]
            elif space_slug and slug:
                try:
                    doc = await _wiki_find_doc_by_slug(
                        base_url, admin_token, space_slug=space_slug, doc_slug=slug
                    )
                except Exception as exc:
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps({"error": f"查找文档失败: {exc}"}, ensure_ascii=False),
                        )
                    ]
                if doc is None:
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps(
                                {"error": f"未找到文档: space='{space_slug}', slug='{slug}'"},
                                ensure_ascii=False,
                            ),
                        )
                    ]
                doc_id = doc["id"]
            else:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": "请提供 doc_id 或 space_slug+slug 来定位文档"},
                            ensure_ascii=False,
                        ),
                    )
                ]

            try:
                result = await _wiki_update_doc(
                    base_url,
                    admin_token,
                    doc_id=doc_id,
                    title=arguments.get("title", ""),
                    slug=arguments.get("new_slug", ""),
                    content=arguments.get("content"),
                    ai_content=arguments.get("ai_content"),
                    excerpt=arguments.get("excerpt"),
                    visibility=arguments.get("visibility", ""),
                    parent_id=arguments.get("parent_id"),
                    sort_order=arguments.get("sort_order"),
                    is_pinned=arguments.get("is_pinned"),
                )
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result, ensure_ascii=False, indent=2),
                    )
                ]
            except Exception as exc:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": f"更新文档失败: {exc}"},
                            ensure_ascii=False,
                        ),
                    )
                ]

        return [TextContent(type="text", text=f"未知工具: {name}")]

    # ── MCP Prompts: 每个员工 = 一个可调用的 prompt ──

    @server.list_prompts()
    async def list_prompts() -> list[Prompt]:
        """列出所有员工作为 MCP Prompts."""
        result = discover_employees(project_dir=_project_dir)
        prompts = []
        for emp in result.employees.values():
            arguments = [
                PromptArgument(
                    name=a.name,
                    description=a.description,
                    required=a.required,
                )
                for a in emp.args
            ]
            prompts.append(
                Prompt(
                    name=emp.name,
                    title=emp.effective_display_name,
                    description=emp.description,
                    arguments=arguments or None,
                )
            )
        return prompts

    @server.get_prompt()
    async def get_prompt(
        name: str,
        arguments: dict[str, str] | None,
    ) -> GetPromptResult:
        """获取渲染后的 prompt."""
        result = discover_employees(project_dir=_project_dir)
        emp = result.get(name)
        if emp is None:
            raise EmployeeNotFoundError(name)

        engine = CrewEngine(project_dir=_project_dir)
        args = arguments or {}
        errors = engine.validate_args(emp, args=args)
        if errors:
            raise ValueError(f"参数错误: {'; '.join(errors)}")

        rendered = engine.prompt(emp, args=args)
        return GetPromptResult(
            description=emp.description,
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=rendered),
                ),
            ],
        )

    # ── MCP Resources: 员工定义的原始 Markdown ──

    def _build_employee_context_markdown(emp, project_dir: Path | None) -> str:
        """构建员工运行时上下文 Markdown — 供 Claude Code 直接当上下文用.

        复用 get_prompt_cached() 获取记忆（60s TTL 缓存），
        复用 _handle_employee_state 的 soul/notes 读取逻辑。
        """
        from crew.memory import MemoryStore

        parts: list[str] = []

        # 标题 + 基本信息
        display = emp.character_name or emp.effective_display_name
        parts.append(f"# {display} — {emp.description}")
        parts.append("")
        parts.append(f"**状态**: {emp.agent_status} | **模型**: {emp.model or '默认'}")

        # Soul（人设）
        soul = ""
        if emp.source_path:
            soul_path = emp.source_path / "soul.md"
            if soul_path.exists():
                soul = soul_path.read_text(encoding="utf-8").strip()
        if soul:
            parts.append("")
            parts.append("## Soul")
            parts.append(soul)

        # 近期记忆 — 复用 get_prompt_cached（带 60s TTL 缓存）
        store = MemoryStore(project_dir=project_dir)
        memories = store.query(
            emp.name,
            limit=10,
            max_visibility="open",
            sort_by="importance",
            min_importance=3,
            update_access=False,
        )
        if memories:
            parts.append("")
            parts.append("## 近期记忆")
            for m in memories:
                date_str = m.created_at[:10] if m.created_at else ""
                parts.append(f"- [{m.category}] {date_str}: {m.content}")

        # Notes
        notes_dir = (project_dir or Path.cwd()) / ".crew" / "notes"
        recent_notes: list[str] = []
        if notes_dir.is_dir():
            note_files = sorted(notes_dir.glob("*.md"), reverse=True)
            for nf in note_files[:5]:
                text = nf.read_text(encoding="utf-8")
                if "visibility: private" in text:
                    continue
                if emp.character_name in text or emp.name in text:
                    # 取前 500 字符作为摘要
                    recent_notes.append(text[:500])
        if recent_notes:
            parts.append("")
            parts.append("## Notes")
            for note in recent_notes:
                parts.append(f"- {note}")

        return "\n".join(parts)

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        """列出所有员工定义 + active 员工的运行时上下文."""
        result = discover_employees(project_dir=_project_dir)
        resources: list[Resource] = []

        for emp in result.employees.values():
            # 员工定义（原有）
            resources.append(
                Resource(
                    uri=f"crew://employee/{emp.name}",
                    name=emp.name,
                    title=emp.effective_display_name,
                    description=emp.description,
                    mimeType="text/markdown",
                )
            )
            # 运行时上下文（仅 active 员工）
            if emp.agent_status == "active":
                resources.append(
                    Resource(
                        uri=f"crew://employee-context/{emp.name}",
                        name=f"{emp.name}-context",
                        title=f"{emp.effective_display_name} 运行时上下文",
                        description=f"{emp.effective_display_name} 的角色设定、近期记忆和笔记",
                        mimeType="text/markdown",
                    )
                )

        return resources

    @server.read_resource()
    async def read_resource(uri) -> list[ReadResourceContents]:
        """读取员工定义或运行时上下文."""
        uri_str = str(uri)

        # ── employee-context: 运行时上下文 ──
        context_prefix = "crew://employee-context/"
        if uri_str.startswith(context_prefix):
            emp_name = uri_str[len(context_prefix) :]
            result = discover_employees(project_dir=_project_dir)
            emp = result.get(emp_name)
            if emp is None:
                raise EmployeeNotFoundError(emp_name)

            content = _build_employee_context_markdown(emp, _project_dir)
            return [ReadResourceContents(content=content, mime_type="text/markdown")]

        # ── employee: 原始定义 ──
        prefix = "crew://employee/"
        if not uri_str.startswith(prefix):
            raise ValueError(f"未知资源: {uri_str}")

        emp_name = uri_str[len(prefix) :]
        result = discover_employees(project_dir=_project_dir)
        emp = result.get(emp_name)
        if emp is None:
            raise EmployeeNotFoundError(emp_name)

        if emp.source_path and emp.source_path.exists():
            if emp.source_path.is_dir():
                # 目录格式：返回拼接后的完整内容（即 body）
                content = emp.body
            else:
                content = emp.source_path.read_text(encoding="utf-8")
        else:
            content = emp.body

        return [ReadResourceContents(content=content, mime_type="text/markdown")]

    return server


async def serve(project_dir: Path | None = None):
    """启动 MCP 服务器（stdio 传输）."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-crew[mcp]")

    server = create_server(project_dir=project_dir)
    init_options = InitializationOptions(
        server_name="crew",
        server_version="0.1.1",
        capabilities=ServerCapabilities(
            tools={},
            resources={},
            prompts={},
        ),
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)


async def serve_sse(
    project_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    api_token: str | None = None,
):
    """启动 MCP 服务器（SSE 传输 — 兼容 Claude Desktop / Cursor）."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-crew[mcp]")

    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route

    server = create_server(project_dir=project_dir)
    sse = SseServerTransport("/messages/")
    _start = _time.monotonic()

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    async def health(request):
        emp_count = len(discover_employees(project_dir=project_dir).employees)
        return JSONResponse(
            {
                "status": "ok",
                "version": _get_version(),
                "employees": emp_count,
                "uptime_seconds": round(_time.monotonic() - _start),
            }
        )

    async def metrics(request):
        from crew.metrics import get_collector

        return JSONResponse(get_collector().snapshot())

    async def sse_lifespan(app):
        yield

    app = Starlette(
        routes=[
            Route("/health", endpoint=health),
            Route("/metrics", endpoint=metrics),
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
        lifespan=sse_lifespan,
    )
    if api_token:
        from crew.auth import BearerTokenMiddleware

        app.add_middleware(BearerTokenMiddleware, token=api_token)

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()


async def serve_http(
    project_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    api_token: str | None = None,
):
    """启动 MCP 服务器（Streamable HTTP 传输 — MCP 最新规范）."""
    if not HAS_MCP:
        raise ImportError("MCP 未安装。请运行: pip install knowlyr-crew[mcp]")

    import uvicorn
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route

    server = create_server(project_dir=project_dir)
    session_manager = StreamableHTTPSessionManager(app=server)
    _start = _time.monotonic()

    async def handle_mcp(scope, receive, send):
        await session_manager.handle_request(scope, receive, send)

    async def health(request):
        emp_count = len(discover_employees(project_dir=project_dir).employees)
        return JSONResponse(
            {
                "status": "ok",
                "version": _get_version(),
                "employees": emp_count,
                "uptime_seconds": round(_time.monotonic() - _start),
            }
        )

    async def metrics(request):
        from crew.metrics import get_collector

        return JSONResponse(get_collector().snapshot())

    async def lifespan(app):
        async with session_manager.run():
            yield

    app = Starlette(
        routes=[
            Route("/health", endpoint=health),
            Route("/metrics", endpoint=metrics),
            Mount("/mcp", app=handle_mcp),
        ],
        lifespan=lifespan,
    )
    if api_token:
        from crew.auth import BearerTokenMiddleware

        app.add_middleware(BearerTokenMiddleware, token=api_token)

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()


def main():
    """CLI 入口: knowlyr-crew mcp."""
    import asyncio
    import os

    transport = os.environ.get("KNOWLYR_CREW_TRANSPORT", "stdio")
    project_dir = os.environ.get("KNOWLYR_CREW_PROJECT_DIR")
    host = os.environ.get("KNOWLYR_CREW_HOST", "127.0.0.1")
    port = int(os.environ.get("KNOWLYR_CREW_PORT", "8000"))
    api_token = os.environ.get("KNOWLYR_CREW_API_TOKEN")
    pd = Path(project_dir) if project_dir else None

    if transport == "sse":
        asyncio.run(serve_sse(pd, host, port, api_token))
    elif transport == "http":
        asyncio.run(serve_http(pd, host, port, api_token))
    else:
        asyncio.run(serve(pd))
