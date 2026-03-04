"""核心工具 — 员工管理、记忆、基础执行能力."""

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
    args: dict, *, agent_id: str | None = None, ctx: _AppContext | None = None
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
    args: dict, *, agent_id: str | None = None, ctx: _AppContext | None = None
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
            "title": emp.summary or emp.display_name or "",
            "status": emp.agent_status or "active",
            "model": emp.model or "",
            "tags": emp.tags or [],
        }
        agents.append(info)

    return json.dumps(agents, ensure_ascii=False, indent=2)


async def _tool_create_note(
    args: dict, *, agent_id: str | None = None, ctx: _AppContext | None = None
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
    args: dict, *, agent_id: str | None = None, ctx: _AppContext | None = None
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
    args: dict, *, agent_id: str | None = None, ctx: _AppContext | None = None
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
    agent_id: str | None = None,
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


# ── 基础执行能力工具 ──


async def _tool_get_datetime(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """获取当前准确日期时间."""
    from datetime import datetime, timedelta
    from datetime import timezone as _tz

    tz_cn = _tz(timedelta(hours=8))
    now = datetime.now(tz_cn)
    weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()]
    return f"{now.strftime('%Y-%m-%d %H:%M')} {weekday}"


async def _tool_calculate(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """安全计算数学表达式."""
    import ast
    import math
    import operator

    expr = (args.get("expression") or "").strip()
    if not expr:
        return "需要一个数学表达式，如 1+2*3 或 (100*1.15**12)。"

    # 安全求值：只允许数学运算
    _OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    _FUNCS = {
        "abs": abs,
        "round": round,
        "int": int,
        "float": float,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "max": max,
        "min": min,
        "sum": sum,
        "pow": pow,
    }

    def _eval(node: ast.AST) -> Any:  # noqa: F821
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"不允许的常量: {node.value}")
        if isinstance(node, ast.BinOp):
            op = _OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的运算: {type(node.op).__name__}")
            left, right = _eval(node.left), _eval(node.right)
            # 防止巨指数 DoS（如 2**999999999）
            if isinstance(node.op, ast.Pow):
                if isinstance(right, (int, float)) and abs(right) > 10000:
                    raise ValueError(f"指数过大: {right}（上限 10000）")
            return op(left, right)
        if isinstance(node, ast.UnaryOp):
            op = _OPS.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的运算: {type(node.op).__name__}")
            return op(_eval(node.operand))
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
                fn = _FUNCS[node.func.id]
                args_vals = [_eval(a) for a in node.args]
                return fn(*args_vals)
            raise ValueError(f"不允许的函数: {ast.dump(node.func)}")
        if isinstance(node, ast.Name):
            if node.id in _FUNCS:
                return _FUNCS[node.id]
            raise ValueError(f"未知变量: {node.id}")
        if isinstance(node, ast.Tuple):
            return tuple(_eval(e) for e in node.elts)
        if isinstance(node, ast.List):
            return [_eval(e) for e in node.elts]
        raise ValueError(f"不支持的语法: {type(node).__name__}")

    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval(tree)
        # 格式化结果
        if isinstance(result, float):
            if result == int(result) and abs(result) < 1e15:
                return str(int(result))
            return f"{result:,.6g}"
        return str(result)
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError) as e:
        return f"计算错误: {e}"


async def _tool_web_search(
    args: dict, *, agent_id: str | None = None, ctx: _AppContext | None = None
) -> str:
    """搜索互联网（Bing cn）."""
    import re

    import httpx

    query = args.get("query", "")
    max_results = min(args.get("max_results", 5), 10)
    if not query:
        return "错误：query 不能为空"

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://cn.bing.com/search",
                params={"q": query, "count": max_results},
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                    "Accept-Language": "zh-CN,zh;q=0.9",
                },
            )

        results: list[str] = []
        for block in re.finditer(r'<li class="b_algo".*?</li>', resp.text, re.DOTALL):
            if len(results) >= max_results:
                break
            title_m = re.search(
                r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
                block.group(),
                re.DOTALL,
            )
            snippet_m = re.search(r"<p[^>]*>(.*?)</p>", block.group(), re.DOTALL)
            if title_m:
                href = title_m.group(1)
                title = re.sub(r"<[^>]+>", "", title_m.group(2)).strip()
                snippet = re.sub(r"<[^>]+>", "", snippet_m.group(1)).strip() if snippet_m else ""
                if title or snippet:
                    results.append(f"{title}\n{snippet}\n{href}")

        if not results:
            return f"未找到关于「{query}」的搜索结果"
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"搜索失败: {e}"


async def _tool_read_url(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """读取网页正文."""
    import ipaddress
    import re
    import socket
    from urllib.parse import urlparse

    import httpx

    url = (args.get("url") or "").strip()
    if not url:
        return "缺少 URL。"

    # SSRF 防护：仅允许 http/https，阻止私有/保留 IP
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "仅支持 http/https 协议。"
    hostname = parsed.hostname or ""
    if not hostname:
        return "无效 URL。"

    # 先解析 DNS，再校验 IP，防止 DNS rebinding 绕过
    try:
        addr_infos = socket.getaddrinfo(
            hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
        )
    except socket.gaierror:
        return f"DNS 解析失败: {hostname}"
    for family, _type, _proto, _canonname, sockaddr in addr_infos:
        try:
            addr = ipaddress.ip_address(sockaddr[0])
            if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
                return "不允许访问内网地址。"
        except ValueError:
            pass
    if hostname in ("localhost", "metadata.google.internal"):
        return "不允许访问内网地址。"

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        html = resp.text
    except Exception as e:
        return f"请求失败: {e}"

    # 简单的 HTML → 纯文本提取
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<nav[^>]*>.*?</nav>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<footer[^>]*>.*?</footer>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<header[^>]*>.*?</header>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # 提取 <article> 或 <main>，否则取 <body>
    for tag in ("article", "main"):
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", html, re.DOTALL | re.IGNORECASE)
        if m:
            html = m.group(1)
            break

    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return "无法提取页面内容。"
    if len(text) > 9500:
        return text[:9500] + f"\n\n[内容已截断，共 {len(text)} 字符]"
    return text


HANDLERS: dict[str, object] = {
    "send_message": _tool_send_message,
    "list_agents": _tool_list_agents,
    "create_note": _tool_create_note,
    "lookup_user": _tool_lookup_user,
    "read_notes": _tool_read_notes,
    "project_status": _tool_project_status,
    "get_datetime": _tool_get_datetime,
    "calculate": _tool_calculate,
    "web_search": _tool_web_search,
    "read_url": _tool_read_url,
}
