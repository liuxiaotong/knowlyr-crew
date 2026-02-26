"""外部服务工具函数 — Notion、RSS、翻译、天气、搜索等."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crew.webhook_context import (
    _NOTION_API_BASE,
    _NOTION_API_KEY,
    _NOTION_VERSION,
)
from crew.webhook_tools.feishu import _CITY_CODES

if TYPE_CHECKING:
    from crew.webhook_context import _AppContext


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


async def _tool_weather(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """查天气（国内城市）."""
    import httpx

    city = (args.get("city") or "").strip().replace("市", "").replace("省", "")
    if not city:
        return "需要城市名，如：上海、北京、杭州。"

    code = _CITY_CODES.get(city)
    if not code:
        avail = "、".join(list(_CITY_CODES.keys())[:20]) + "…"
        return f"暂不支持「{city}」，支持的城市：{avail}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"http://t.weather.itboy.net/api/weather/city/{code}")
            data = resp.json()

        if data.get("status") != 200:
            return f"查询失败: {data.get('message', '未知错误')}"

        info = data.get("data", {})
        city_name = data.get("cityInfo", {}).get("city", city)
        now_temp = info.get("wendu", "?")
        humidity = info.get("shidu", "?")
        quality = info.get("quality", "")
        pm25 = info.get("pm25", "")

        forecast = info.get("forecast", [])
        lines = [f"{city_name} 当前 {now_temp}℃，湿度 {humidity}"]
        if quality:
            lines[0] += f"，空气{quality}"
            if pm25:
                lines[0] += f"(PM2.5:{pm25})"

        days = min(int(args.get("days", 3)), 7)
        for day in forecast[:days]:
            high = day.get("high", "").replace("高温 ", "")
            low = day.get("low", "").replace("低温 ", "")
            weather_type = day.get("type", "")
            wind = day.get("fx", "")
            wind_level = day.get("fl", "")
            date = day.get("ymd", "")
            week = day.get("week", "")
            lines.append(f"{date}({week}) {weather_type} {low}~{high} {wind}{wind_level}")

        return "\n".join(lines)
    except Exception as e:
        return f"天气查询失败: {e}"


async def _tool_exchange_rate(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """查汇率."""
    import httpx

    base = (args.get("from") or args.get("base") or "USD").upper().strip()
    targets = (args.get("to") or "CNY").upper().strip()
    target_list = [t.strip() for t in targets.split(",") if t.strip()]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"https://api.exchangerate-api.com/v4/latest/{base}")
            data = resp.json()

        if "rates" not in data:
            return f"查询失败: 不支持货币 {base}"

        rates = data["rates"]
        lines = [f"基准: 1 {base}"]
        for t in target_list:
            rate = rates.get(t)
            if rate is not None:
                lines.append(f"= {rate} {t}")
            else:
                lines.append(f"{t}: 不支持")
        return "\n".join(lines)
    except Exception as e:
        return f"汇率查询失败: {e}"


async def _tool_stock_price(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """查股价（A股/美股）."""
    import re

    import httpx

    symbol = (args.get("symbol") or "").strip()
    if not symbol:
        return "需要股票代码，如：sh600519（茅台）、gb_aapl（苹果）。"

    # 规范化：纯数字默认为沪市A股
    sym = symbol.lower()
    if re.match(r"^\d{6}$", sym):
        sym = f"sh{sym}" if sym.startswith("6") else f"sz{sym}"
    elif re.match(r"^[a-z]{1,5}$", sym):
        sym = f"gb_{sym}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"https://hq.sinajs.cn/list={sym}",
                headers={"Referer": "https://finance.sina.com.cn"},
            )
        text = resp.text.strip()
        if not text or '=""' in text:
            return f"未找到股票: {symbol}"

        # 解析 Sina 行情数据
        m = re.search(r'="(.+)"', text)
        if not m:
            return f"数据解析失败: {symbol}"

        fields = m.group(1).split(",")
        if sym.startswith("gb_"):
            # 美股格式: 名称,现价,涨跌幅%,时间,涨跌额,开盘,最高,最低,...
            if len(fields) < 4:
                return f"数据不完整: {symbol}"
            name, price, change_pct = fields[0], fields[1], fields[2]
            return f"{name} ({symbol.upper()})\n现价: ${price}  涨跌: {change_pct}%"
        else:
            # A股格式: 名称,今开,昨收,现价,最高,最低,...
            if len(fields) < 6:
                return f"数据不完整: {symbol}"
            name, today_open, prev_close, price, high, low = fields[:6]
            try:
                change = float(price) - float(prev_close)
                change_pct = change / float(prev_close) * 100
                sign = "+" if change >= 0 else ""
                return f"{name} ({symbol.upper()})\n现价: ¥{price}  涨跌: {sign}{change:.2f} ({sign}{change_pct:.2f}%)\n今开: {today_open}  最高: {high}  最低: {low}"
            except (ValueError, ZeroDivisionError):
                return f"{name} ({symbol.upper()})\n现价: ¥{price}"
    except Exception as e:
        return f"股价查询失败: {e}"


def _notion_blocks_to_text(blocks: list[dict]) -> str:
    """将 Notion blocks 转为纯文本."""
    parts = []
    for block in blocks:
        btype = block.get("type", "")
        data = block.get(btype, {})
        rich_text = data.get("rich_text", [])
        text = "".join(rt.get("plain_text", "") for rt in rich_text)
        if btype.startswith("heading"):
            parts.append(f"\n{text}\n")
        elif btype == "to_do":
            checked = "x" if data.get("checked") else " "
            parts.append(f"[{checked}] {text}")
        elif btype == "code":
            parts.append(f"```\n{text}\n```")
        elif text:
            parts.append(text)
    return "\n".join(parts)


async def _tool_notion_search(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """搜索 Notion 页面."""
    import httpx

    if not _NOTION_API_KEY:
        return "Notion 未配置。请设置 NOTION_API_KEY 环境变量。"

    query = (args.get("query") or "").strip()
    limit = min(args.get("limit", 10), 20)
    if not query:
        return "搜索关键词不能为空。"

    headers = {
        "Authorization": f"Bearer {_NOTION_API_KEY}",
        "Notion-Version": _NOTION_VERSION,
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{_NOTION_API_BASE}/search",
            json={"query": query, "page_size": limit},
            headers=headers,
        )
    if resp.status_code != 200:
        return f"Notion API 错误 {resp.status_code}: {resp.text[:500]}"

    results = resp.json().get("results", [])
    if not results:
        return "没有找到匹配的页面。"

    lines = []
    for r in results[:limit]:
        obj_type = r.get("object", "page")
        url = r.get("url", "")
        edited = r.get("last_edited_time", "")[:10]
        # 提取标题
        props = r.get("properties", {})
        title_prop = props.get("title", props.get("Name", {}))
        title_arr = title_prop.get("title", []) if isinstance(title_prop, dict) else []
        title = "".join(t.get("plain_text", "") for t in title_arr) or "无标题"
        lines.append(f"[{obj_type}] {title} (edited: {edited})\n{url}")
    return "\n---\n".join(lines)


async def _tool_notion_read(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """读取 Notion 页面内容."""
    import httpx

    if not _NOTION_API_KEY:
        return "Notion 未配置。请设置 NOTION_API_KEY 环境变量。"

    page_id = (args.get("page_id") or "").strip().replace("-", "")
    if not page_id:
        return "缺少 page_id。"

    headers = {
        "Authorization": f"Bearer {_NOTION_API_KEY}",
        "Notion-Version": _NOTION_VERSION,
    }
    all_blocks: list[dict] = []
    next_cursor = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        for _ in range(5):  # 最多取 5 页
            params: dict = {"page_size": 100}
            if next_cursor:
                params["start_cursor"] = next_cursor
            resp = await client.get(
                f"{_NOTION_API_BASE}/blocks/{page_id}/children",
                params=params,
                headers=headers,
            )
            if resp.status_code != 200:
                return f"Notion API 错误 {resp.status_code}: {resp.text[:500]}"
            data = resp.json()
            all_blocks.extend(data.get("results", []))
            if not data.get("has_more"):
                break
            next_cursor = data.get("next_cursor")

    text = _notion_blocks_to_text(all_blocks)
    if not text.strip():
        return f"页面 {page_id} 内容为空。"
    if len(text) > 9500:
        return text[:9500] + f"\n\n[内容已截断，共 {len(text)} 字符]"
    return text


async def _tool_notion_create(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """在 Notion 创建新页面."""
    import httpx

    if not _NOTION_API_KEY:
        return "Notion 未配置。请设置 NOTION_API_KEY 环境变量。"

    parent_id = (args.get("parent_id") or "").strip().replace("-", "")
    title = (args.get("title") or "").strip()
    content = args.get("content", "")
    if not parent_id or not title:
        return "需要 parent_id 和 title。"

    children = []
    for para in content.split("\n\n")[:100]:
        if para.strip():
            children.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": para.strip()}}]
                    },
                }
            )

    headers = {
        "Authorization": f"Bearer {_NOTION_API_KEY}",
        "Notion-Version": _NOTION_VERSION,
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{_NOTION_API_BASE}/pages",
            json={
                "parent": {"page_id": parent_id},
                "properties": {"title": {"title": [{"text": {"content": title}}]}},
                "children": children,
            },
            headers=headers,
        )
    if resp.status_code not in (200, 201):
        return f"Notion 创建失败 {resp.status_code}: {resp.text[:500]}"

    url = resp.json().get("url", "")
    return f"页面已创建：{title}\n{url}"


# ── 信息采集工具 ──


async def _tool_read_url(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """读取网页正文."""
    import ipaddress
    import re
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

    import socket

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


async def _tool_rss_read(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """读取 RSS/Atom 订阅源."""
    import re

    try:
        import defusedxml.ElementTree as ET
    except ImportError:
        import logging as _logging
        import xml.etree.ElementTree as ET  # noqa: S405

        _logging.getLogger(__name__).warning(
            "defusedxml 未安装，RSS 解析使用标准库 xml.etree.ElementTree（无 XXE 防护）。"
            "建议安装: pip install 'knowlyr-crew[webhook]'"
        )

    import httpx

    url = (args.get("url") or "").strip()
    limit = min(args.get("limit", 10), 30)
    if not url:
        return "缺少 RSS URL。"

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(resp.text)
    except Exception as e:
        return f"RSS 解析失败: {e}"

    entries = []
    # RSS 2.0: <channel><item>
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = (item.findtext("description") or "")[:200].strip()
        desc = re.sub(r"<[^>]+>", "", desc)  # strip HTML
        if title:
            entries.append(f"{title}\n{desc}\n{link}")

    # Atom: <entry>
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
        title = (
            entry.findtext("atom:title", "", ns)
            or entry.findtext("{http://www.w3.org/2005/Atom}title")
            or ""
        ).strip()
        link_el = entry.find("atom:link", ns) or entry.find("{http://www.w3.org/2005/Atom}link")
        link = link_el.get("href", "") if link_el is not None else ""
        summary = (
            entry.findtext("atom:summary", "", ns)
            or entry.findtext("{http://www.w3.org/2005/Atom}summary")
            or ""
        )[:200].strip()
        summary = re.sub(r"<[^>]+>", "", summary)
        if title:
            entries.append(f"{title}\n{summary}\n{link}")

    if not entries:
        return "订阅源中没有找到条目。"

    return "\n---\n".join(entries[:limit])


# ── 生活助手工具 ──


async def _tool_translate(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """中英互译（MyMemory API）."""
    import httpx

    text = (args.get("text") or "").strip()
    if not text:
        return "需要翻译的文本。"
    if len(text) > 2000:
        return "文本过长，最多 2000 字符。"

    from_lang = (args.get("from_lang") or "auto").strip().lower()
    to_lang = (args.get("to_lang") or "").strip().lower()

    # 自动检测：CJK 占比 > 30% → 中→英，否则英→中
    if from_lang == "auto":
        cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        if cjk / max(len(text), 1) > 0.3:
            from_lang, to_lang = "zh-CN", to_lang or "en-GB"
        else:
            from_lang, to_lang = "en-GB", to_lang or "zh-CN"
    else:
        _lang_map = {
            "zh": "zh-CN",
            "en": "en-GB",
            "ja": "ja-JP",
            "ko": "ko-KR",
            "fr": "fr-FR",
            "de": "de-DE",
        }
        from_lang = _lang_map.get(from_lang, from_lang)
        to_lang = (
            _lang_map.get(to_lang, to_lang)
            if to_lang
            else ("en-GB" if "zh" in from_lang else "zh-CN")
        )

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.mymemory.translated.net/get",
                params={"q": text, "langpair": f"{from_lang}|{to_lang}"},
            )
            data = resp.json()
        translated = data.get("responseData", {}).get("translatedText", "")
        if not translated:
            return "翻译失败，未获得结果。"
        return translated
    except Exception as e:
        return f"翻译失败: {e}"


async def _tool_countdown(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """计算距离目标日期的倒计时."""
    from datetime import datetime, timedelta
    from datetime import timezone as _tz

    tz_cn = _tz(timedelta(hours=8))
    date_str = (args.get("date") or "").strip()
    event = (args.get("event") or "").strip()

    if not date_str:
        return "需要目标日期，格式 YYYY-MM-DD。"

    try:
        target = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=tz_cn)
    except ValueError:
        return f"日期格式不对: {date_str}，需要 YYYY-MM-DD。"

    now = datetime.now(tz_cn)
    delta = target - now
    label = f"「{event}」" if event else date_str

    if delta.total_seconds() < 0:
        days = abs(delta.days)
        return f"{label} 已经过去了 {days} 天。"

    days = delta.days
    hours = delta.seconds // 3600
    if days == 0:
        return f"距离 {label} 还有 {hours} 小时。"
    return f"距离 {label} 还有 {days} 天 {hours} 小时。"


async def _tool_trending(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """热搜聚合（微博 / 知乎）."""
    import httpx

    platform = (args.get("platform") or "weibo").strip().lower()
    limit = min(args.get("limit", 15) or 15, 30)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            if platform == "zhihu":
                resp = await client.get(
                    "https://www.zhihu.com/api/v3/feed/topstory/hot-lists/total",
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                data = resp.json()
                items = data.get("data", [])[:limit]
                if not items:
                    return "知乎热榜暂无数据。"
                lines = []
                for i, item in enumerate(items, 1):
                    target = item.get("target", {})
                    title = target.get("title", "")
                    excerpt = target.get("excerpt", "")[:60]
                    lines.append(f"{i}. {title}\n   {excerpt}")
                return "📊 知乎热榜\n\n" + "\n".join(lines)
            else:
                # 微博热搜
                resp = await client.get(
                    "https://weibo.com/ajax/side/hotSearch",
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                data = resp.json()
                items = data.get("data", {}).get("realtime", [])[:limit]
                if not items:
                    return "微博热搜暂无数据。"
                lines = []
                for i, item in enumerate(items, 1):
                    word = item.get("word", "")
                    num = item.get("num", 0)
                    label_name = item.get("label_name", "")
                    tag = f" [{label_name}]" if label_name else ""
                    lines.append(f"{i}. {word}{tag}  ({num:,})")
                return "🔥 微博热搜\n\n" + "\n".join(lines)
    except Exception as e:
        return f"获取热搜失败: {e}"


# ── 飞书表格工具 ──


async def _tool_summarize(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """长文摘要（由模型自身完成）."""
    text = (args.get("text") or "").strip()
    if not text:
        return "需要文本。"
    if len(text) > 50000:
        text = text[:50000] + "...(已截断)"

    style = (args.get("style") or "bullet").strip().lower()
    style_map = {
        "bullet": "用要点列表总结",
        "paragraph": "用一段话总结",
        "oneline": "用一句话总结",
    }
    instruction = style_map.get(style, style_map["bullet"])
    return f"[摘要任务] 请{instruction}以下内容：\n\n{text}"


async def _tool_sentiment(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """情感分析（由模型自身完成）."""
    text = (args.get("text") or "").strip()
    if not text:
        return "需要文本。"
    if len(text) > 10000:
        text = text[:10000] + "...(已截断)"

    return (
        f"[情感分析任务] 请分析以下文本的情感倾向（正面/负面/中性）、语气和关键情绪词：\n\n{text}"
    )


async def _tool_email_send(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """发送邮件（暂未对接 SMTP）."""
    to = (args.get("to") or "").strip()
    subject = (args.get("subject") or "").strip()
    if not to or not subject:
        return "需要收件人和主题。"
    return "邮件功能尚未配置 SMTP，暂时无法发送。请直接通过飞书或其他方式联系。"


async def _tool_qrcode(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """生成二维码."""
    from urllib.parse import quote

    data = (args.get("data") or "").strip()
    if not data:
        return "需要编码内容。"

    size = args.get("size", 300) or 300
    encoded = quote(data, safe="")
    url = f"https://api.qrserver.com/v1/create-qr-code/?size={size}x{size}&data={encoded}"
    return f"二维码已生成：\n{url}\n\n内容: {data}"


async def _tool_express_track(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """快递物流查询."""
    import httpx

    number = (args.get("number") or "").strip()
    if not number:
        return "需要快递单号。"

    company = (args.get("company") or "").strip()

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # 快递100 auto API
            url = "https://www.kuaidi100.com/query"
            params = {"type": company or "auto", "postid": number}
            resp = await client.get(url, params=params)
            data = resp.json()

        if data.get("status") != "200" and not data.get("data"):
            # 尝试备用格式
            msg = data.get("message") or data.get("msg") or "未查到物流信息"
            return f"查询失败: {msg}"

        traces = data.get("data", [])
        if not traces:
            return f"快递单号 {number} 暂无物流信息。"

        com_name = data.get("com", company or "未知")
        state_map = {
            "0": "运输中",
            "1": "揽收",
            "2": "疑难",
            "3": "已签收",
            "4": "退签",
            "5": "派件中",
            "6": "退回",
        }
        state = state_map.get(str(data.get("state", "")), "未知")

        lines = [f"📦 {com_name} {number} [{state}]", ""]
        for t in traces[:10]:
            time_str = t.get("ftime") or t.get("time", "")
            context = t.get("context", "")
            lines.append(f"  {time_str}  {context}")
        return "\n".join(lines)
    except Exception as e:
        return f"快递查询失败: {e}"


async def _tool_flight_info(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """航班查询（暂用 web_search 代理）."""
    flight_no = (args.get("flight_no") or "").strip().upper()
    if not flight_no:
        return "需要航班号。"

    date = (args.get("date") or "").strip()
    return f"航班查询功能开发中。请使用 web_search 搜索「{flight_no} {date} 航班动态」获取信息。"


async def _tool_aqi(
    args: dict,
    *,
    agent_id: str | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """空气质量查询."""
    import httpx

    city = (args.get("city") or "").strip()
    if not city:
        return "需要城市名。"

    # 中文城市名映射
    city_map = {
        "上海": "shanghai",
        "北京": "beijing",
        "广州": "guangzhou",
        "深圳": "shenzhen",
        "杭州": "hangzhou",
        "成都": "chengdu",
        "重庆": "chongqing",
        "武汉": "wuhan",
        "南京": "nanjing",
        "西安": "xian",
        "苏州": "suzhou",
        "天津": "tianjin",
        "长沙": "changsha",
        "郑州": "zhengzhou",
        "青岛": "qingdao",
        "大连": "dalian",
        "厦门": "xiamen",
        "昆明": "kunming",
        "合肥": "hefei",
        "福州": "fuzhou",
    }
    query = city_map.get(city, city)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"https://api.waqi.info/feed/{query}/",
                params={"token": "demo"},
            )
            data = resp.json()

        if data.get("status") != "ok":
            return f"未找到 {city} 的空气质量数据。"

        d = data["data"]
        aqi_val = d.get("aqi", "N/A")
        station = d.get("city", {}).get("name", city)
        time_str = d.get("time", {}).get("s", "")

        # AQI 等级
        if isinstance(aqi_val, int):
            if aqi_val <= 50:
                level = "优 🟢"
            elif aqi_val <= 100:
                level = "良 🟡"
            elif aqi_val <= 150:
                level = "轻度污染 🟠"
            elif aqi_val <= 200:
                level = "中度污染 🔴"
            elif aqi_val <= 300:
                level = "重度污染 🟤"
            else:
                level = "严重污染 ⚫"
        else:
            level = ""

        iaqi = d.get("iaqi", {})
        lines = [f"🌍 {station}", f"AQI: {aqi_val} {level}"]
        if iaqi.get("pm25"):
            lines.append(f"PM2.5: {iaqi['pm25'].get('v', 'N/A')}")
        if iaqi.get("pm10"):
            lines.append(f"PM10: {iaqi['pm10'].get('v', 'N/A')}")
        if iaqi.get("o3"):
            lines.append(f"O₃: {iaqi['o3'].get('v', 'N/A')}")
        if iaqi.get("t"):
            lines.append(f"温度: {iaqi['t'].get('v', 'N/A')}℃")
        if iaqi.get("h"):
            lines.append(f"湿度: {iaqi['h'].get('v', 'N/A')}%")
        if time_str:
            lines.append(f"更新: {time_str}")
        return "\n".join(lines)
    except Exception as e:
        return f"空气质量查询失败: {e}"


HANDLERS: dict[str, object] = {
    "web_search": _tool_web_search,
    "weather": _tool_weather,
    "exchange_rate": _tool_exchange_rate,
    "stock_price": _tool_stock_price,
    "notion_search": _tool_notion_search,
    "notion_read": _tool_notion_read,
    "notion_create": _tool_notion_create,
    "read_url": _tool_read_url,
    "rss_read": _tool_rss_read,
    "translate": _tool_translate,
    "countdown": _tool_countdown,
    "trending": _tool_trending,
    "summarize": _tool_summarize,
    "sentiment": _tool_sentiment,
    "email_send": _tool_email_send,
    "qrcode": _tool_qrcode,
    "express_track": _tool_express_track,
    "flight_info": _tool_flight_info,
    "aqi": _tool_aqi,
}
