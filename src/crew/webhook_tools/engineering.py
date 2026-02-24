"""工程/实用工具函数 — 编码、哈希、正则、DNS、时间等."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crew.webhook_context import _AppContext


async def _tool_get_datetime(
    args: dict,
    *,
    agent_id: int | None = None,
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
    agent_id: int | None = None,
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

    def _eval(node: ast.AST) -> Any:
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
            return op(_eval(node.left), _eval(node.right))
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


# ── 飞书文档工具 ──


async def _tool_unit_convert(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """单位换算."""
    value = args.get("value")
    if value is None:
        return "缺少数值。"
    value = float(value)

    from_u = (args.get("from_unit") or "").strip().lower().replace("°", "").replace(" ", "")
    to_u = (args.get("to_unit") or "").strip().lower().replace("°", "").replace(" ", "")

    if not from_u or not to_u:
        return "需要原单位和目标单位。"

    # 温度特殊处理
    if from_u in ("c", "celsius") and to_u in ("f", "fahrenheit"):
        result = value * 9 / 5 + 32
        return f"{value}°C = {result:.2f}°F"
    if from_u in ("f", "fahrenheit") and to_u in ("c", "celsius"):
        result = (value - 32) * 5 / 9
        return f"{value}°F = {result:.2f}°C"

    key = (from_u, to_u)
    factor = _UNIT_CONVERSIONS.get(key)
    if factor is None:
        return f"不支持 {from_u} → {to_u} 的换算。支持：km/mi, m/ft, kg/lb, l/gal, gb/mb, c/f 等。"

    result = value * factor
    if result == int(result) and abs(result) < 1e15:
        return f"{value} {from_u} = {int(result)} {to_u}"
    return f"{value} {from_u} = {result:,.4g} {to_u}"


async def _tool_random_pick(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """随机选择 / 掷骰子."""
    import random

    options_str = (args.get("options") or "").strip()
    count = max(args.get("count", 1) or 1, 1)

    if not options_str:
        # 掷骰子
        result = random.randint(1, 6)
        return f"🎲 掷出了 {result} 点！"

    options = [o.strip() for o in options_str.replace("，", ",").split(",") if o.strip()]
    if len(options) < 2:
        return "至少需要两个选项。"

    count = min(count, len(options))
    picks = random.sample(options, count)
    if count == 1:
        return f"🎯 选中了：{picks[0]}"
    return f"🎯 选中了：{'、'.join(picks)}"


async def _tool_holidays(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """查询中国法定节假日."""
    from datetime import datetime, timedelta
    from datetime import timezone as _tz

    import httpx

    tz_cn = _tz(timedelta(hours=8))
    now = datetime.now(tz_cn)
    year = args.get("year") or now.year
    month = args.get("month") or 0

    # 使用 timor.tech 免费节假日 API（国内可用）
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if month:
                resp = await client.get(
                    f"https://timor.tech/api/holiday/year/{year}/{month:02d}",
                )
            else:
                resp = await client.get(
                    f"https://timor.tech/api/holiday/year/{year}",
                )
            data = resp.json()

        if data.get("code") != 0:
            return f"查询失败: {data.get('msg', '未知错误')}"

        holidays_data = data.get("holiday", {})
        if not holidays_data:
            return f"{year}年{f'{month}月' if month else ''}没有节假日数据。"

        lines = []
        for date_str, info in sorted(holidays_data.items()):
            name = info.get("name", "")
            is_holiday = info.get("holiday", False)
            tag = "🟢 放假" if is_holiday else "🔴 补班"
            lines.append(f"{date_str} {tag} {name}")

        if not lines:
            return "没有找到节假日信息。"

        header = f"📅 {year}年{f'{month}月' if month else ''}节假日安排"
        return f"{header}\n\n" + "\n".join(lines)
    except Exception as e:
        return f"查询节假日失败: {e}"


async def _tool_timestamp_convert(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """Unix 时间戳 ↔ 可读时间互转."""
    from datetime import datetime, timedelta
    from datetime import timezone as _tz

    tz_cn = _tz(timedelta(hours=8))
    input_str = (args.get("input") or "").strip()
    if not input_str:
        return "需要时间戳或日期时间。"

    # 尝试解析为数字（时间戳）
    try:
        ts = int(input_str)
        # 毫秒级 → 秒级
        if ts > 1e12:
            ts = ts // 1000
        dt = datetime.fromtimestamp(ts, tz_cn)
        weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][dt.weekday()]
        return f"时间戳 {input_str} = {dt.strftime('%Y-%m-%d %H:%M:%S')} {weekday}（北京时间）"
    except (ValueError, OSError):
        pass

    # 尝试解析为日期时间
    fmts = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"]
    for fmt in fmts:
        try:
            dt = datetime.strptime(input_str, fmt).replace(tzinfo=tz_cn)
            ts = int(dt.timestamp())
            return f"{input_str} = 时间戳 {ts}（秒）/ {ts * 1000}（毫秒）"
        except ValueError:
            continue

    return f"无法解析: {input_str}。支持格式：Unix 时间戳 或 YYYY-MM-DD HH:MM:SS"


# ── 飞书表格创建 ──


async def _tool_text_extract(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """从文本中提取邮箱、手机号、URL、金额等."""
    import re

    text = args.get("text") or ""
    if not text:
        return "需要文本。"

    extract_type = (args.get("extract_type") or "all").strip().lower()

    results: dict[str, list[str]] = {}

    if extract_type in ("email", "all"):
        emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        if emails:
            results["邮箱"] = list(dict.fromkeys(emails))

    if extract_type in ("phone", "all"):
        phones = re.findall(r"1[3-9]\d{9}", text)
        # 也匹配带分隔的号码
        phones += re.findall(r"\+?\d{1,4}[-\s]?\d{3,4}[-\s]?\d{4}", text)
        if phones:
            results["手机号"] = list(dict.fromkeys(phones))

    if extract_type in ("url", "all"):
        urls = re.findall(r"https?://[^\s<>\"']+", text)
        if urls:
            results["URL"] = list(dict.fromkeys(urls))

    if extract_type in ("money", "all"):
        money = re.findall(
            r"[¥$￥]\s?[\d,]+\.?\d*|[\d,]+\.?\d*\s?(?:元|万|亿|美元|万元|亿元|USD|CNY|RMB)", text
        )
        if money:
            results["金额"] = list(dict.fromkeys(money))

    if not results:
        return "未提取到信息。"

    lines = []
    for category, items in results.items():
        lines.append(f"【{category}】")
        for item in items[:20]:
            lines.append(f"  {item}")
    return "\n".join(lines)


async def _tool_json_format(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """格式化 JSON."""
    import json as _json
    import re

    text = args.get("text") or ""
    if not text:
        return "需要 JSON 文本。"

    compact = args.get("compact", False)

    # 尝试直接解析
    try:
        obj = _json.loads(text)
    except _json.JSONDecodeError:
        # 尝试从文本中提取 JSON
        match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
        if not match:
            return "未找到有效的 JSON。"
        try:
            obj = _json.loads(match.group())
        except _json.JSONDecodeError as e:
            return f"JSON 解析失败: {e}"

    if compact:
        result = _json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    else:
        result = _json.dumps(obj, ensure_ascii=False, indent=2)

    if len(result) > 9500:
        result = result[:9500] + "\n\n[已截断]"
    return result


async def _tool_password_gen(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """生成安全随机密码."""
    import secrets
    import string

    length = max(min(args.get("length", 16) or 16, 128), 8)
    count = max(min(args.get("count", 3) or 3, 10), 1)
    no_symbols = args.get("no_symbols", False)

    chars = string.ascii_letters + string.digits
    if not no_symbols:
        chars += "!@#$%&*-_=+"

    passwords = []
    for _ in range(count):
        pw = "".join(secrets.choice(chars) for _ in range(length))
        passwords.append(pw)

    lines = [f"🔐 随机密码（{length}位）：", ""]
    for i, pw in enumerate(passwords, 1):
        lines.append(f"{i}. {pw}")
    return "\n".join(lines)


async def _tool_ip_lookup(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """查询 IP 地址归属地."""
    import httpx

    ip = (args.get("ip") or "").strip()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if ip:
                resp = await client.get(f"http://ip-api.com/json/{ip}?lang=zh-CN")
            else:
                resp = await client.get("http://ip-api.com/json/?lang=zh-CN")
            data = resp.json()

        if data.get("status") != "success":
            return f"查询失败: {data.get('message', '未知错误')}"

        query_ip = data.get("query", ip or "本机")
        country = data.get("country", "")
        region = data.get("regionName", "")
        city = data.get("city", "")
        isp = data.get("isp", "")
        org = data.get("org", "")

        location = " ".join(filter(None, [country, region, city]))
        lines = [f"IP: {query_ip}", f"位置: {location}"]
        if isp:
            lines.append(f"运营商: {isp}")
        if org and org != isp:
            lines.append(f"组织: {org}")
        return "\n".join(lines)
    except Exception as e:
        return f"查询 IP 失败: {e}"


async def _tool_short_url(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """生成短链接（cleanuri.com 免费 API）."""
    import httpx

    url = (args.get("url") or "").strip()
    if not url:
        return "需要 URL。"
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://cleanuri.com/api/v1/shorten",
                data={"url": url},
            )
            data = resp.json()

        short = data.get("result_url", "")
        if short:
            return f"短链接: {short}\n原链接: {url}"
        return f"生成失败: {data.get('error', '未知错误')}"
    except Exception as e:
        return f"生成短链接失败: {e}"


async def _tool_word_count(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """统计文本字数."""
    text = args.get("text") or ""
    if not text:
        return "需要文本。"

    # 总字符数（含空格）
    total_chars = len(text)
    # 不含空格
    no_space = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
    # 中文字数
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    # 英文单词数
    import re

    words = len(re.findall(r"[a-zA-Z]+", text))
    # 数字个数
    numbers = len(re.findall(r"\d+", text))
    # 行数
    lines = text.count("\n") + 1
    # 段落数
    paragraphs = len([p for p in text.split("\n\n") if p.strip()])

    parts = [
        f"字符: {total_chars}（不含空格 {no_space}）",
        f"中文: {cjk} 字",
        f"英文: {words} 词",
    ]
    if numbers:
        parts.append(f"数字: {numbers} 个")
    parts.append(f"行: {lines}")
    parts.append(f"段落: {paragraphs}")

    return " | ".join(parts)


# ── 编码 & 开发辅助工具 ──


async def _tool_base64_codec(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """Base64 编解码."""
    import base64

    text = args.get("text") or ""
    if not text:
        return "需要文本。"

    decode = args.get("decode", False)
    try:
        if decode:
            result = base64.b64decode(text).decode("utf-8", errors="replace")
            return f"解码结果:\n{result}"
        else:
            result = base64.b64encode(text.encode("utf-8")).decode()
            return f"编码结果:\n{result}"
    except Exception as e:
        return f"Base64 {'解码' if decode else '编码'}失败: {e}"


async def _tool_color_convert(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """颜色格式转换."""
    import re

    color = (args.get("color") or "").strip()
    if not color:
        return "需要颜色值。"

    r = g = b = 0

    # HEX
    hex_match = re.match(r"^#?([0-9a-fA-F]{6})$", color)
    if hex_match:
        h = hex_match.group(1)
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    else:
        # RGB
        rgb_match = re.match(r"rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", color, re.I)
        if rgb_match:
            r, g, b = int(rgb_match.group(1)), int(rgb_match.group(2)), int(rgb_match.group(3))
        else:
            # 3位 HEX
            short_match = re.match(r"^#?([0-9a-fA-F]{3})$", color)
            if short_match:
                h = short_match.group(1)
                r, g, b = int(h[0] * 2, 16), int(h[1] * 2, 16), int(h[2] * 2, 16)
            else:
                return f"无法解析颜色: {color}。支持 #FF5733、rgb(255,87,51) 格式。"

    # RGB → HSL
    r1, g1, b1 = r / 255, g / 255, b / 255
    mx, mn = max(r1, g1, b1), min(r1, g1, b1)
    l = (mx + mn) / 2
    if mx == mn:
        h_val = s = 0.0
    else:
        d = mx - mn
        s = d / (2 - mx - mn) if l > 0.5 else d / (mx + mn)
        if mx == r1:
            h_val = (g1 - b1) / d + (6 if g1 < b1 else 0)
        elif mx == g1:
            h_val = (b1 - r1) / d + 2
        else:
            h_val = (r1 - g1) / d + 4
        h_val /= 6

    hex_str = f"#{r:02X}{g:02X}{b:02X}"
    rgb_str = f"rgb({r}, {g}, {b})"
    hsl_str = f"hsl({int(h_val * 360)}, {int(s * 100)}%, {int(l * 100)}%)"

    return f"HEX: {hex_str}\nRGB: {rgb_str}\nHSL: {hsl_str}"


async def _tool_cron_explain(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """解释 cron 表达式."""
    expr = (args.get("expression") or "").strip()
    if not expr:
        return "需要 cron 表达式或自然语言描述。"

    # 简单自然语言 → cron 映射
    _NL_MAP = {
        "每分钟": "* * * * *",
        "每小时": "0 * * * *",
        "每天": "0 0 * * *",
        "每天早上9点": "0 9 * * *",
        "每天晚上10点": "0 22 * * *",
        "每周一": "0 0 * * 1",
        "工作日": "0 9 * * 1-5",
        "工作日早上9点": "0 9 * * 1-5",
        "每月1号": "0 0 1 * *",
        "每月15号": "0 0 15 * *",
    }

    for key, cron in _NL_MAP.items():
        if key in expr:
            return f"「{expr}」对应的 cron:\n{cron}"

    # 解析 cron 表达式
    parts = expr.split()
    if len(parts) not in (5, 6):
        return f"无法解析: {expr}。标准 cron 是 5 段（分 时 日 月 周），如 0 9 * * 1-5"

    fields = ["分钟", "小时", "日", "月", "星期"]
    if len(parts) == 6:
        fields = ["秒"] + fields

    _WEEKDAYS = {
        "0": "日",
        "1": "一",
        "2": "二",
        "3": "三",
        "4": "四",
        "5": "五",
        "6": "六",
        "7": "日",
    }

    lines = []
    for i, (p, name) in enumerate(zip(parts, fields, strict=False)):
        if p == "*":
            lines.append(f"  {name}: 每{name}")
        elif p.startswith("*/"):
            lines.append(f"  {name}: 每 {p[2:]} {name}")
        elif name == "星期" and "-" in p:
            start, end = p.split("-", 1)
            lines.append(
                f"  {name}: 周{_WEEKDAYS.get(start, start)} 到 周{_WEEKDAYS.get(end, end)}"
            )
        elif name == "星期":
            days = [f"周{_WEEKDAYS.get(d.strip(), d.strip())}" for d in p.split(",")]
            lines.append(f"  {name}: {','.join(days)}")
        else:
            lines.append(f"  {name}: {p}")

    return f"cron: {expr}\n\n" + "\n".join(lines)


async def _tool_regex_test(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """测试正则表达式."""
    import re

    pattern = args.get("pattern") or ""
    text = args.get("text") or ""
    replace = args.get("replace") or ""

    if not pattern:
        return "需要正则表达式。"
    if not text:
        return "需要测试文本。"

    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return f"正则语法错误: {e}"

    if replace:
        result = compiled.sub(replace, text)
        return f"替换结果:\n{result}"

    matches = list(compiled.finditer(text))
    if not matches:
        return "没有匹配。"

    lines = [f"找到 {len(matches)} 个匹配：", ""]
    for i, m in enumerate(matches[:20], 1):
        groups = m.groups()
        if groups:
            lines.append(f"{i}. 「{m.group()}」 groups={groups}")
        else:
            lines.append(f"{i}. 「{m.group()}」 位置 {m.start()}-{m.end()}")
    return "\n".join(lines)


async def _tool_hash_gen(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """计算文本哈希值."""
    import hashlib

    text = args.get("text") or ""
    if not text:
        return "需要文本。"

    algo = (args.get("algorithm") or "sha256").strip().lower()
    data = text.encode("utf-8")

    results = []
    if algo == "all" or algo == "md5":
        results.append(f"MD5:    {hashlib.md5(data).hexdigest()}")
    if algo == "all" or algo == "sha1":
        results.append(f"SHA1:   {hashlib.sha1(data).hexdigest()}")
    if algo == "all" or algo == "sha256":
        results.append(f"SHA256: {hashlib.sha256(data).hexdigest()}")

    if not results:
        # 默认 sha256
        results.append(f"SHA256: {hashlib.sha256(data).hexdigest()}")

    return "\n".join(results)


async def _tool_url_codec(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """URL 编解码."""
    from urllib.parse import quote, unquote

    text = args.get("text") or ""
    if not text:
        return "需要文本。"

    decode = args.get("decode", False)
    if decode:
        result = unquote(text)
        return f"解码结果:\n{result}"
    else:
        result = quote(text, safe="")
        return f"编码结果:\n{result}"


# ── 第 5 批工具 handler ──


async def _tool_diff_text(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """文本对比."""
    import difflib

    text1 = args.get("text1") or ""
    text2 = args.get("text2") or ""
    if not text1 and not text2:
        return "需要两段文本。"

    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)
    diff = list(difflib.unified_diff(lines1, lines2, fromfile="原文", tofile="修改后", lineterm=""))

    if not diff:
        return "两段文本完全相同。"
    return "\n".join(diff[:200])


async def _tool_whois(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """WHOIS 域名查询."""
    import httpx

    domain = (args.get("domain") or "").strip().lower()
    if not domain:
        return "需要域名。"
    # 去掉 http:// 等前缀
    domain = domain.replace("https://", "").replace("http://", "").split("/")[0]

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"https://whois.freeaitools.xyz/api/{domain}")
            if resp.status_code != 200:
                return f"WHOIS 查询失败 (HTTP {resp.status_code})"
            data = resp.json()

        lines = [f"域名: {domain}"]
        for key in ("registrar", "creation_date", "expiration_date", "name_servers", "status"):
            val = data.get(key)
            if val:
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val)
                label = {
                    "registrar": "注册商",
                    "creation_date": "注册日期",
                    "expiration_date": "到期日期",
                    "name_servers": "DNS",
                    "status": "状态",
                }.get(key, key)
                lines.append(f"{label}: {val}")
        return "\n".join(lines) if len(lines) > 1 else f"未找到 {domain} 的 WHOIS 信息。"
    except Exception as e:
        return f"WHOIS 查询失败: {e}"


async def _tool_dns_lookup(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """DNS 解析."""
    import asyncio
    import socket

    domain = (args.get("domain") or "").strip().lower()
    if not domain:
        return "需要域名。"
    domain = domain.replace("https://", "").replace("http://", "").split("/")[0]

    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: socket.getaddrinfo(domain, None, socket.AF_UNSPEC, socket.SOCK_STREAM),
        )
        seen: set[str] = set()
        lines = [f"DNS 解析 {domain}："]
        for family, _type, _proto, _canon, addr in results:
            ip = addr[0]
            if ip in seen:
                continue
            seen.add(ip)
            record_type = "A" if family == socket.AF_INET else "AAAA"
            lines.append(f"  {record_type}: {ip}")
        return "\n".join(lines) if len(lines) > 1 else f"未找到 {domain} 的 DNS 记录。"
    except socket.gaierror:
        return f"无法解析域名: {domain}"
    except Exception as e:
        return f"DNS 查询失败: {e}"


async def _tool_http_check(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """网站可用性检查."""
    import time

    import httpx

    url = (args.get("url") or "").strip()
    if not url:
        return "需要 URL。"
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    try:
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.head(url)
        elapsed = (time.monotonic() - start) * 1000

        status = resp.status_code
        ok = "✅ 可用" if 200 <= status < 400 else "❌ 异常"
        lines = [
            f"{ok}",
            f"URL: {url}",
            f"状态码: {status}",
            f"响应时间: {elapsed:.0f}ms",
        ]
        if resp.headers.get("server"):
            lines.append(f"服务器: {resp.headers['server']}")
        return "\n".join(lines)
    except httpx.ConnectTimeout:
        return f"❌ 连接超时: {url}"
    except httpx.ConnectError:
        return f"❌ 无法连接: {url}"
    except Exception as e:
        return f"❌ 检查失败: {e}"


# ── Python 代码执行沙箱 ──

# 允许导入的模块白名单
_ALLOWED_MODULES: set[str] = {
    "json",
    "re",
    "math",
    "datetime",
    "time",
    "collections",
    "itertools",
    "functools",
    "string",
    "textwrap",
    "decimal",
    "statistics",
    "random",
    "hashlib",
    "base64",
    "urllib",
    "urllib.parse",
    "html",
    "html.parser",
    "csv",
    "io",
    "typing",
    "dataclasses",
    "enum",
    "struct",
    "calendar",
    "operator",
    "copy",
    "pprint",
    "fractions",
    "bisect",
    "heapq",
}

# 禁止调用的内置函数
_BLOCKED_CALLS: set[str] = {
    "__import__",
    "exec",
    "eval",
    "compile",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "breakpoint",
    "exit",
    "quit",
}


def _validate_python_code(code: str) -> str | None:
    """AST 静态检查，返回 None 表示通过，否则返回错误消息."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"语法错误: {e}"

    for node in ast.walk(tree):
        # 检查 import
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                if mod not in _ALLOWED_MODULES:
                    return f"不允许导入 {alias.name}（安全限制）"
        if isinstance(node, ast.ImportFrom):
            if node.module:
                mod = node.module.split(".")[0]
                if mod not in _ALLOWED_MODULES:
                    return f"不允许导入 {node.module}（安全限制）"

        # 检查危险属性访问（os.system 等）
        if isinstance(node, ast.Attribute):
            _BLOCKED_ATTRS = {
                "system", "popen", "exec", "spawn", "call", "run", "Popen",
                "check_output", "check_call", "getstatusoutput", "execvp", "execve",
                "fork", "kill", "remove", "rmdir", "unlink", "rmtree",
            }
            if node.attr in _BLOCKED_ATTRS:
                return f"不允许调用 .{node.attr}()（安全限制）"

        # 检查危险函数调用
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BLOCKED_CALLS:
                return f"不允许调用 {node.func.id}()（安全限制）"
            # open() 只允许读模式
            if node.func.id == "open":
                # 有第二个参数且不是 'r'/'rb' → 拒绝
                if len(node.args) >= 2:
                    mode_arg = node.args[1]
                    if isinstance(mode_arg, ast.Constant) and mode_arg.value not in ("r", "rb"):
                        return "open() 只允许读模式（安全限制）"
                # 有 mode keyword
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        if kw.value.value not in ("r", "rb"):
                            return "open() 只允许读模式（安全限制）"

    return None


async def _tool_run_python(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,
) -> str:
    """在沙箱中执行 Python 代码片段."""
    import asyncio
    import sys
    import textwrap

    code = (args.get("code") or "").strip()
    if not code:
        return "需要 Python 代码。用 print() 输出结果。"

    # 1. AST 静态检查
    error = _validate_python_code(code)
    if error:
        return f"代码检查失败: {error}"

    # 2. 超时限制
    timeout = min(max(int(args.get("timeout", 30) or 30), 5), 60)

    # 3. 构建 wrapper（Linux 上加 resource limits）
    wrapper = textwrap.dedent("""\
        import sys
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (256*1024*1024, 256*1024*1024))
            resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))
        except (ImportError, ValueError, OSError):
            pass
        exec(compile({code!r}, "<run_python>", "exec"))
    """).format(timeout=timeout, code=code)

    # 4. subprocess 执行
    python = sys.executable or "python3"
    try:
        proc = await asyncio.create_subprocess_exec(
            python,
            "-c",
            wrapper,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout + 5,
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()  # type: ignore[union-attr]
        except ProcessLookupError:
            pass
        return f"执行超时（超过 {timeout} 秒）。"

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    # 5. 截断输出
    max_out, max_err = 10_000, 3_000
    if len(stdout) > max_out:
        stdout = stdout[:max_out] + "\n\n[输出已截断]"
    if len(stderr) > max_err:
        stderr = stderr[:max_err] + "\n\n[错误已截断]"

    parts: list[str] = []
    if stdout.strip():
        parts.append(stdout.strip())
    if stderr.strip():
        parts.append(f"[stderr]\n{stderr.strip()}")
    if proc.returncode and proc.returncode != 0 and not parts:
        parts.append(f"进程退出码: {proc.returncode}")
    if not parts:
        parts.append("（代码执行完毕，无输出）")

    return "\n".join(parts)


HANDLERS: dict[str, object] = {
    "get_datetime": _tool_get_datetime,
    "calculate": _tool_calculate,
    "unit_convert": _tool_unit_convert,
    "random_pick": _tool_random_pick,
    "holidays": _tool_holidays,
    "timestamp_convert": _tool_timestamp_convert,
    "text_extract": _tool_text_extract,
    "json_format": _tool_json_format,
    "password_gen": _tool_password_gen,
    "ip_lookup": _tool_ip_lookup,
    "short_url": _tool_short_url,
    "word_count": _tool_word_count,
    "base64_codec": _tool_base64_codec,
    "color_convert": _tool_color_convert,
    "cron_explain": _tool_cron_explain,
    "regex_test": _tool_regex_test,
    "hash_gen": _tool_hash_gen,
    "url_codec": _tool_url_codec,
    "diff_text": _tool_diff_text,
    "whois": _tool_whois,
    "dns_lookup": _tool_dns_lookup,
    "http_check": _tool_http_check,
    "run_python": _tool_run_python,
}
