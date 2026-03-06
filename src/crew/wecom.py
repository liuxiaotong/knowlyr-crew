"""企业微信 Bot -- 消息加解密、token 管理、XML 解析、消息发送."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import re
import struct
import threading
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from crew.paths import resolve_project_dir

logger = logging.getLogger(__name__)

# ── 共享 httpx 连接池 ──

import httpx

_wecom_client: httpx.AsyncClient | None = None
_wecom_client_lock = threading.Lock()


def get_wecom_client() -> httpx.AsyncClient:
    """获取共享的企微 httpx 客户端（连接池复用）."""
    global _wecom_client
    if _wecom_client is None or _wecom_client.is_closed:
        with _wecom_client_lock:
            if _wecom_client is None or _wecom_client.is_closed:
                _wecom_client = httpx.AsyncClient(
                    timeout=30.0,
                    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                )
    return _wecom_client


async def close_wecom_client() -> None:
    """关闭共享客户端（应用退出时调用）."""
    global _wecom_client
    if _wecom_client is not None:
        await _wecom_client.aclose()
        _wecom_client = None


WECOM_API_BASE = "https://qyapi.weixin.qq.com/cgi-bin"
TOKEN_REFRESH_MARGIN = 300  # 过期前 5 分钟刷新
_WECOM_TEXT_MAX_BYTES = 2048  # 企微文本消息限制 2048 字节


# ── 配置 ──


class WecomConfig(BaseModel):
    """企业微信 Bot 配置."""

    corp_id: str = Field(default="", description="企业 CorpID")
    agent_id: int = Field(default=0, description="应用 AgentId")
    secret: str = Field(default="", description="应用 Secret")
    token: str = Field(default="", description="回调 Token（签名验证）")
    encoding_aes_key: str = Field(default="", description="回调 EncodingAESKey（消息加解密）")
    default_employee: str = Field(default="", description="默认员工名")
    tenant_id: str = Field(default="", description="绑定的租户 ID（空=admin 租户）")
    # 通讯录同步（独立密钥）
    contact_secret: str = Field(default="", description="通讯录同步 Secret")
    contact_token: str = Field(default="", description="通讯录同步回调 Token")
    contact_encoding_aes_key: str = Field(default="", description="通讯录同步回调 EncodingAESKey")


def load_wecom_config(project_dir: Path | None = None) -> WecomConfig:
    """从 .crew/wecom.yaml 或环境变量加载配置.

    优先级: YAML 文件 > 环境变量.
    """
    base = resolve_project_dir(project_dir)
    config_path = base / ".crew" / "wecom.yaml"

    data: dict[str, Any] = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data = raw

    # 环境变量兜底
    if not data.get("corp_id"):
        data["corp_id"] = os.environ.get("WECOM_CORP_ID", "")
    if not data.get("agent_id"):
        _aid = os.environ.get("WECOM_AGENT_ID", "0")
        data["agent_id"] = int(_aid) if _aid.isdigit() else 0
    if not data.get("secret"):
        data["secret"] = os.environ.get("WECOM_SECRET", "")
    if not data.get("token"):
        data["token"] = os.environ.get("WECOM_TOKEN", "")
    if not data.get("encoding_aes_key"):
        data["encoding_aes_key"] = os.environ.get("WECOM_ENCODING_AES_KEY", "")

    return WecomConfig(**data)


# ── AES-256-CBC 加解密 (企微回调协议) ──


class WecomCrypto:
    """企业微信消息加解密器.

    使用 AES-256-CBC，密钥从 EncodingAESKey(Base64) 派生。
    明文格式: random(16) + msg_len(4, network order) + msg + corp_id
    """

    def __init__(self, encoding_aes_key: str, corp_id: str):
        self._corp_id = corp_id
        # EncodingAESKey 是 43 个字符的 Base64 编码（末尾补 =），解码后 32 bytes = AES-256
        self._key = base64.b64decode(encoding_aes_key + "=")
        if len(self._key) != 32:
            raise ValueError(f"EncodingAESKey 解码后应为 32 字节，实际 {len(self._key)}")

    def decrypt(self, encrypted_text: str) -> str:
        """解密企微加密消息，返回明文 XML/echostr."""
        from Crypto.Cipher import AES

        cipher = AES.new(self._key, AES.MODE_CBC, iv=self._key[:16])
        encrypted_bytes = base64.b64decode(encrypted_text)
        decrypted = cipher.decrypt(encrypted_bytes)
        # PKCS#7 去填充（验证 pad 值范围）
        pad_len = decrypted[-1]
        if pad_len < 1 or pad_len > 32:
            raise ValueError(f"PKCS#7 pad 值无效: {pad_len}（应为 1-32）")
        content = decrypted[:-pad_len]
        # 解析: random(16) + msg_len(4) + msg + corp_id
        msg_len = struct.unpack("!I", content[16:20])[0]
        msg = content[20 : 20 + msg_len].decode("utf-8")
        from_corp_id = content[20 + msg_len :].decode("utf-8")
        if from_corp_id != self._corp_id:
            raise ValueError(f"CorpID 不匹配: 期望 {self._corp_id}，实际 {from_corp_id}")
        return msg

    def encrypt(self, plaintext: str) -> str:
        """加密明文，返回 Base64 编码的密文."""
        from Crypto.Cipher import AES

        msg_bytes = plaintext.encode("utf-8")
        corp_bytes = self._corp_id.encode("utf-8")
        random_bytes = os.urandom(16)
        msg_len = struct.pack("!I", len(msg_bytes))
        content = random_bytes + msg_len + msg_bytes + corp_bytes
        # PKCS#7 填充到 AES block size
        # 企微回调协议要求 block_size=32（非标准 AES-128 的 16 字节），
        # 与 EncodingAESKey 解码后的 32 字节密钥长度一致。
        block_size = 32
        pad_len = block_size - (len(content) % block_size)
        content += bytes([pad_len]) * pad_len
        cipher = AES.new(self._key, AES.MODE_CBC, iv=self._key[:16])
        encrypted = cipher.encrypt(content)
        return base64.b64encode(encrypted).decode("utf-8")


# ── 签名验证 ──


def verify_wecom_signature(
    token: str, timestamp: str, nonce: str, msg_encrypt: str, signature: str
) -> bool:
    """验证企微回调签名.

    SHA1(sort(token, timestamp, nonce, msg_encrypt)) == signature
    """
    import hmac as _hmac

    parts = sorted([token, timestamp, nonce, msg_encrypt])
    digest = hashlib.sha1("".join(parts).encode("utf-8")).hexdigest()
    return _hmac.compare_digest(digest, signature)


def generate_wecom_signature(token: str, timestamp: str, nonce: str, msg_encrypt: str) -> str:
    """生成企微回调签名."""
    parts = sorted([token, timestamp, nonce, msg_encrypt])
    return hashlib.sha1("".join(parts).encode("utf-8")).hexdigest()


# ── Token 管理 ──


class WecomTokenManager:
    """access_token 管理器 -- 自动获取 + 缓存 + 到期前刷新."""

    def __init__(self, corp_id: str, secret: str):
        self._corp_id = corp_id
        self._secret = secret
        self._token: str = ""
        self._expire_at: float = 0.0
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get_token(self) -> str:
        """获取有效的 access_token."""
        now = time.time()
        if self._token and now < self._expire_at:
            return self._token

        async with self._get_lock():
            now = time.time()
            if self._token and now < self._expire_at:
                return self._token
            return await self._refresh()

    async def _refresh(self) -> str:
        """GET /gettoken?corpid=...&corpsecret=..."""
        url = f"{WECOM_API_BASE}/gettoken"
        params = {"corpid": self._corp_id, "corpsecret": self._secret}

        client = get_wecom_client()
        resp = await client.get(url, params=params)
        data = resp.json()

        errcode = data.get("errcode", -1)
        if errcode != 0:
            errmsg = data.get("errmsg", "unknown error")
            raise RuntimeError(f"获取企微 access_token 失败: {errmsg} (errcode={errcode})")

        self._token = data["access_token"]
        expire_seconds = data.get("expires_in", 7200)
        self._expire_at = time.time() + expire_seconds - TOKEN_REFRESH_MARGIN
        logger.info("企微 token 已刷新，有效期 %ds", expire_seconds)
        return self._token


# ── XML 解析 (defusedxml 防 XXE) ──


def parse_wecom_message(xml_text: str) -> dict[str, str]:
    """解析企微回调 XML，返回字段字典.

    支持字段: ToUserName, FromUserName, CreateTime, MsgType, Content,
    MsgId, AgentID, Encrypt, PicUrl, MediaId 等.
    """
    import defusedxml.ElementTree as ET

    root = ET.fromstring(xml_text)
    result: dict[str, str] = {}
    for child in root:
        result[child.tag] = child.text or ""
    return result


def parse_wecom_encrypt_xml(xml_text: str) -> tuple[str, str]:
    """从企微加密回调 XML 中提取 Encrypt 和 ToUserName.

    Returns:
        (encrypt_content, to_user_name)
    """
    import defusedxml.ElementTree as ET

    root = ET.fromstring(xml_text)
    encrypt_node = root.find("Encrypt")
    to_user_node = root.find("ToUserName")
    encrypt_content = encrypt_node.text if encrypt_node is not None else ""
    to_user_name = to_user_node.text if to_user_node is not None else ""
    return encrypt_content, to_user_name


# ── 消息发送 ──


def _truncate_to_bytes(text: str, max_bytes: int) -> str:
    """将文本截断到不超过 max_bytes 个 UTF-8 字节."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    # 按字节截断，避免截断到 UTF-8 多字节序列中间
    truncated = encoded[:max_bytes]
    # 回退到最后一个完整 UTF-8 字符
    result = truncated.decode("utf-8", errors="ignore")
    return result + "..."


def _strip_markdown(text: str) -> str:
    """将 Markdown 格式转为企微友好的纯文本."""
    # 代码块: ```lang\n...\n``` -> 保留内容
    text = re.sub(
        r"```[a-zA-Z]*\n(.*?)```",
        lambda m: m.group(1).strip(),
        text,
        flags=re.DOTALL,
    )
    # 标题: ### Title -> Title
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # 粗体/斜体
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    # 行内代码
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # 链接: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # 图片: ![alt](url) -> 移除
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", text)
    # 水平线
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # HTML 标签
    text = re.sub(r"<[^>]+>", "", text)
    # 清理多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sanitize_wecom_text(text: str) -> str:
    """净化发送到企微的文本内容.

    - 清除 Markdown 格式
    - 去除 null bytes 和 C0 控制字符（保留换行符和制表符）
    - 截断到 2048 字节
    """
    if not text:
        return text
    cleaned = _strip_markdown(text)
    cleaned = "".join(c for c in cleaned if c in ("\n", "\r", "\t") or (ord(c) >= 0x20))
    cleaned = _truncate_to_bytes(cleaned, _WECOM_TEXT_MAX_BYTES)
    return cleaned


def strip_wecom_at_prefix(content: str, app_name: str = "") -> str:
    """从企微群聊消息中去除 @应用名 前缀，返回纯净的用户文本.

    企微群聊中 @应用 发送消息时，Content 格式为 "@应用名 消息内容" 或 "@应用名\\n消息内容"。
    此函数去除开头的 @xxx 前缀（无论是否匹配 app_name）。

    Args:
        content: 原始消息内容
        app_name: 应用名称（可选，用于精确匹配）

    Returns:
        去除 @前缀 后的消息文本
    """
    if not content:
        return content

    # 匹配开头的 @xxx 前缀（可能带空格或换行）
    # 格式: "@应用名 后续消息" 或 "@应用名\n后续消息"
    match = re.match(r"@\S+[\s]*", content)
    if match:
        remaining = content[match.end() :].strip()
        return remaining if remaining else content

    return content


async def send_wecom_text(
    token_manager: WecomTokenManager,
    user_id: str,
    agent_id: int,
    text: str,
) -> dict[str, Any]:
    """发送文本消息给企微用户（单聊）.

    通过 /cgi-bin/message/send 主动推送。
    """
    text = _sanitize_wecom_text(text)
    token = await token_manager.get_token()
    url = f"{WECOM_API_BASE}/message/send"
    params = {"access_token": token}
    body = {
        "touser": user_id,
        "msgtype": "text",
        "agentid": agent_id,
        "text": {"content": text},
    }

    client = get_wecom_client()
    resp = await client.post(url, json=body, params=params)
    data = resp.json()

    errcode = data.get("errcode", -1)
    if errcode != 0:
        errmsg = data.get("errmsg", "unknown")
        logger.warning("企微消息发送失败: %s (errcode=%d)", errmsg, errcode)

    return data


async def send_wecom_group_text(
    token_manager: WecomTokenManager,
    chat_id: str,
    text: str,
) -> dict[str, Any]:
    """发送文本消息到企微群聊.

    通过 /cgi-bin/appchat/send 推送到群聊。

    注意限制：
    - 应用的可见范围必须是根部门（通讯录根目录）
    - 只能发送到由该应用创建的群聊
    - 每企业消息发送量不可超过 2 万人次/分
    - 如果限制不满足，调用方应降级为 send_wecom_text 单聊回复发送者

    Args:
        token_manager: token 管理器
        chat_id: 群聊 ID
        text: 消息内容

    Returns:
        API 返回结果
    """
    text = _sanitize_wecom_text(text)
    token = await token_manager.get_token()
    url = f"{WECOM_API_BASE}/appchat/send"
    params = {"access_token": token}
    body = {
        "chatid": chat_id,
        "msgtype": "text",
        "text": {"content": text},
    }

    client = get_wecom_client()
    resp = await client.post(url, json=body, params=params)
    data = resp.json()

    errcode = data.get("errcode", -1)
    if errcode != 0:
        errmsg = data.get("errmsg", "unknown")
        logger.warning("企微群聊消息发送失败: %s (errcode=%d)", errmsg, errcode)

    return data
