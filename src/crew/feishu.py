"""飞书双向 Bot — 事件解析、token 管理、员工路由、消息发送."""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from crew.paths import resolve_project_dir

logger = logging.getLogger(__name__)

FEISHU_API_BASE = "https://open.feishu.cn/open-apis"
TOKEN_REFRESH_MARGIN = 300  # 过期前 5 分钟刷新
MESSAGE_SEND_TIMEOUT = 30.0


# ── 配置 ──


class FeishuConfig(BaseModel):
    """飞书 Bot 配置."""

    app_id: str = Field(default="", description="飞书应用 App ID")
    app_secret: str = Field(default="", description="飞书应用 App Secret")
    verification_token: str = Field(default="", description="事件订阅验证 token")
    encrypt_key: str = Field(default="", description="事件加密密钥（预留）")
    default_employee: str = Field(default="", description="未匹配 @mention 时的默认员工")
    calendar_id: str = Field(default="", description="飞书日历 ID（创建日程用）")


def load_feishu_config(project_dir: Path | None = None) -> FeishuConfig:
    """从 .crew/feishu.yaml 或环境变量加载配置.

    优先级: YAML 文件 > 环境变量.
    """
    base = resolve_project_dir(project_dir)
    config_path = base / ".crew" / "feishu.yaml"

    data: dict[str, Any] = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data = raw

    # 环境变量兜底
    if not data.get("app_id"):
        data["app_id"] = os.environ.get("FEISHU_APP_ID", "")
    if not data.get("app_secret"):
        data["app_secret"] = os.environ.get("FEISHU_APP_SECRET", "")
    if not data.get("verification_token"):
        data["verification_token"] = os.environ.get("FEISHU_VERIFICATION_TOKEN", "")
    if not data.get("encrypt_key"):
        data["encrypt_key"] = os.environ.get("FEISHU_ENCRYPT_KEY", "")
    if not data.get("calendar_id"):
        data["calendar_id"] = os.environ.get("FEISHU_CALENDAR_ID", "")

    return FeishuConfig(**data)


# ── 事件验证 ──


def verify_feishu_event(verification_token: str, token_in_payload: str) -> bool:
    """验证飞书事件回调的 token."""
    if not verification_token or not token_in_payload:
        return False
    return verification_token == token_in_payload


# ── Token 管理 ──


class FeishuTokenManager:
    """tenant_access_token 管理器 — 自动获取 + 缓存 + 到期前刷新."""

    def __init__(self, app_id: str, app_secret: str):
        self._app_id = app_id
        self._app_secret = app_secret
        self._token: str = ""
        self._expire_at: float = 0.0
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get_token(self) -> str:
        """获取有效的 tenant_access_token."""
        now = time.time()
        if self._token and now < self._expire_at:
            return self._token

        async with self._get_lock():
            now = time.time()
            if self._token and now < self._expire_at:
                return self._token
            return await self._refresh()

    async def _refresh(self) -> str:
        """POST /auth/v3/tenant_access_token/internal."""
        import httpx

        url = f"{FEISHU_API_BASE}/auth/v3/tenant_access_token/internal"
        payload = {"app_id": self._app_id, "app_secret": self._app_secret}

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            data = resp.json()

        code = data.get("code", -1)
        if code != 0:
            msg = data.get("msg", "unknown error")
            raise RuntimeError(f"获取飞书 tenant_access_token 失败: {msg} (code={code})")

        self._token = data["tenant_access_token"]
        expire_seconds = data.get("expire", 7200)
        self._expire_at = time.time() + expire_seconds - TOKEN_REFRESH_MARGIN
        logger.info("飞书 token 已刷新，有效期 %ds", expire_seconds)
        return self._token


# ── 事件解析 ──


class FeishuMessageEvent(BaseModel):
    """解析后的飞书消息事件."""

    message_id: str = Field(description="消息 ID")
    chat_id: str = Field(description="群聊 ID")
    chat_type: str = Field(default="group", description="p2p / group")
    sender_id: str = Field(default="", description="发送者 open_id")
    text: str = Field(description="纯文本消息内容（已去除 @mention 占位符）")
    mentions: list[dict[str, str]] = Field(
        default_factory=list, description="@mention 列表 [{key, id, name}]"
    )
    msg_type: str = Field(default="text", description="消息类型 text / image / ...")
    image_key: str = Field(default="", description="图片 key（仅 image 类型）")


def parse_message_event(payload: dict[str, Any]) -> FeishuMessageEvent | None:
    """从飞书事件回调 payload 中解析消息.

    支持 text 和 image 类型，其他类型返回 None.
    """
    event = payload.get("event", {})
    message = event.get("message", {})
    sender = event.get("sender", {})

    msg_type = message.get("message_type", "")

    common_kwargs = dict(
        message_id=message.get("message_id", ""),
        chat_id=message.get("chat_id", ""),
        chat_type=message.get("chat_type", "group"),
        sender_id=sender.get("sender_id", {}).get("open_id", ""),
        msg_type=msg_type,
    )

    if msg_type == "image":
        content_str = message.get("content", "{}")
        try:
            content = _json.loads(content_str)
        except _json.JSONDecodeError:
            content = {}
        image_key = content.get("image_key", "")

        # image 消息也可能有 mentions（群聊 @bot + 发图）
        raw_mentions = message.get("mentions", [])
        mentions = _parse_mentions(raw_mentions, "")

        return FeishuMessageEvent(
            **common_kwargs,
            text="",
            mentions=mentions,
            image_key=image_key,
        )

    if msg_type != "text":
        return None

    content_str = message.get("content", "{}")
    try:
        content = _json.loads(content_str)
    except _json.JSONDecodeError:
        content = {}

    text = content.get("text", "")

    raw_mentions = message.get("mentions", [])
    mentions = _parse_mentions(raw_mentions, text)
    # 从文本中移除 @mention 占位符
    for m in mentions:
        if m["key"]:
            text = text.replace(m["key"], "").strip()

    return FeishuMessageEvent(
        **common_kwargs,
        text=text,
        mentions=mentions,
    )


def _parse_mentions(raw_mentions: list, text: str) -> list[dict[str, str]]:
    """解析 @mention 列表."""
    mentions: list[dict[str, str]] = []
    for m in raw_mentions:
        mentions.append({
            "key": m.get("key", ""),
            "id": m.get("id", {}).get("open_id", ""),
            "name": m.get("name", ""),
        })
    return mentions


# ── 员工路由 ──


def resolve_employee_from_mention(
    mentions: list[dict[str, str]],
    text: str,
    discovery_result: Any,
    default_employee: str = "",
) -> tuple[str | None, str]:
    """从 @mention 名称匹配员工.

    匹配顺序:
    1. mention name == character_name
    2. mention name == display_name
    3. mention name == name 或 trigger (通过 DiscoveryResult.get)
    4. text 前缀匹配 employee name/trigger
    5. default_employee

    Returns:
        (employee_name, task_text) 或 (None, text)
    """
    for m in mentions:
        mention_name = m.get("name", "")
        if not mention_name:
            continue

        for emp in discovery_result.employees.values():
            if emp.character_name and emp.character_name == mention_name:
                return emp.name, text
            if emp.display_name and emp.display_name == mention_name:
                return emp.name, text

        emp = discovery_result.get(mention_name)
        if emp:
            return emp.name, text

    # text 前缀匹配
    for emp in discovery_result.employees.values():
        prefixes = [emp.name, emp.display_name, emp.character_name] + emp.triggers
        for prefix in prefixes:
            if prefix and text.startswith(prefix):
                remaining = text[len(prefix):].strip()
                return emp.name, remaining or text

    if default_employee:
        return default_employee, text

    return None, text


# ── 消息发送 ──


async def send_feishu_message(
    token_manager: FeishuTokenManager,
    chat_id: str,
    content: dict[str, Any] | str,
    msg_type: str = "interactive",
) -> dict[str, Any]:
    """通过 Message API 发送消息到飞书群."""
    import httpx

    token = await token_manager.get_token()
    url = f"{FEISHU_API_BASE}/im/v1/messages"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    params = {"receive_id_type": "chat_id"}

    content_str = content if isinstance(content, str) else _json.dumps(content, ensure_ascii=False)
    body = {
        "receive_id": chat_id,
        "msg_type": msg_type,
        "content": content_str,
    }

    async with httpx.AsyncClient(timeout=MESSAGE_SEND_TIMEOUT) as client:
        resp = await client.post(url, json=body, headers=headers, params=params)
        data = resp.json()

    code = data.get("code", -1)
    if code != 0:
        msg = data.get("msg", "unknown")
        logger.warning("飞书消息发送失败: %s (code=%d)", msg, code)

    return data


async def send_feishu_card(
    token_manager: FeishuTokenManager,
    chat_id: str,
    task_name: str,
    task_result: dict[str, Any] | None,
    task_error: str | None,
) -> dict[str, Any]:
    """构建卡片并发送（复用 delivery._build_feishu_card）."""
    from crew.delivery import _build_feishu_card

    card_payload = _build_feishu_card(task_name, task_result, task_error)
    return await send_feishu_message(
        token_manager,
        chat_id,
        content=card_payload["card"],
        msg_type="interactive",
    )


async def send_feishu_text(
    token_manager: FeishuTokenManager,
    chat_id: str,
    text: str,
) -> dict[str, Any]:
    """发送纯文本消息."""
    content = {"text": text}
    return await send_feishu_message(
        token_manager, chat_id, content=content, msg_type="text",
    )


async def send_feishu_reply(
    token_manager: FeishuTokenManager,
    message_id: str,
    text: str,
) -> dict[str, Any]:
    """回复指定消息（形成 thread）."""
    import httpx

    token = await token_manager.get_token()
    url = f"{FEISHU_API_BASE}/im/v1/messages/{message_id}/reply"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    body = {
        "msg_type": "text",
        "content": _json.dumps({"text": text}, ensure_ascii=False),
    }

    async with httpx.AsyncClient(timeout=MESSAGE_SEND_TIMEOUT) as client:
        resp = await client.post(url, json=body, headers=headers)
        data = resp.json()

    code = data.get("code", -1)
    if code != 0:
        msg = data.get("msg", "unknown")
        logger.warning("飞书回复发送失败: %s (code=%d)", msg, code)

    return data


# ── 日历日程 ──


async def create_calendar_event(
    token_mgr: "FeishuTokenManager",
    summary: str,
    start_timestamp: int,
    end_timestamp: int,
    description: str = "",
    calendar_id: str = "",
) -> dict[str, Any]:
    """在飞书日历创建日程.

    Args:
        token_mgr: FeishuTokenManager 实例
        summary: 日程标题
        start_timestamp: 开始时间 unix 秒
        end_timestamp: 结束时间 unix 秒
        description: 日程描述（可选）
        calendar_id: 日历 ID（不传则取环境变量 FEISHU_CALENDAR_ID）

    Returns:
        {"ok": True, "event_id": "...", "summary": "..."} 或
        {"ok": False, "error": "..."}
    """
    import httpx

    cal_id = calendar_id or os.environ.get("FEISHU_CALENDAR_ID", "")
    if not cal_id:
        return {"ok": False, "error": "未配置 FEISHU_CALENDAR_ID"}

    token = await token_mgr.get_token()
    body: dict[str, Any] = {
        "summary": summary[:256],
        "start": {"timestamp": str(start_timestamp)},
        "end": {"timestamp": str(end_timestamp)},
    }
    if description:
        body["description"] = description[:2048]

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{FEISHU_API_BASE}/calendar/v4/calendars/{cal_id}/events",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            data = resp.json()
            if data.get("code") != 0:
                return {"ok": False, "error": data.get("msg", "未知错误")}
            event = data.get("data", {}).get("event", {})
            return {
                "ok": True,
                "event_id": event.get("event_id", ""),
                "summary": event.get("summary", summary),
            }
    except Exception as e:
        logger.error("飞书创建日程失败: %s", e)
        return {"ok": False, "error": str(e)}


# ── 事件去重 ──


class EventDeduplicator:
    """基于 message_id 的事件去重器."""

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 1000):
        self._seen: dict[str, float] = {}
        self._ttl = ttl_seconds
        self._max_size = max_size

    def is_duplicate(self, message_id: str) -> bool:
        """检查并记录 message_id."""
        now = time.time()

        if len(self._seen) > self._max_size:
            cutoff = now - self._ttl
            self._seen = {k: v for k, v in self._seen.items() if v > cutoff}

        if message_id in self._seen:
            return True

        self._seen[message_id] = now
        return False
