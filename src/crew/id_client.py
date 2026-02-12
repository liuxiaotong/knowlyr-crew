"""knowlyr-id 客户端 — 可选的 Agent 身份对接."""

from __future__ import annotations

import logging
import os

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _get_config() -> tuple[str, str]:
    """读取环境配置."""
    base_url = os.environ.get("KNOWLYR_ID_URL", "http://localhost:8000")
    token = os.environ.get("AGENT_API_TOKEN", "")
    return base_url, token


def _get_httpx():
    """延迟导入 httpx，未安装时返回 None."""
    try:
        import httpx
        return httpx
    except ImportError:
        return None


class AgentIdentity(BaseModel):
    """从 knowlyr-id 获取的 Agent 身份信息."""

    agent_id: int
    nickname: str = ""
    title: str = ""
    bio: str = ""
    domains: list[str] = Field(default_factory=list)
    model: str = ""
    api_key: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    agent_status: str = "active"
    system_prompt: str = ""
    memory: str = ""


def fetch_agent_identity(agent_id: int) -> AgentIdentity | None:
    """从 knowlyr-id 获取 Agent 身份配置.

    环境变量:
        KNOWLYR_ID_URL: 服务地址（默认 http://localhost:8000）
        AGENT_API_TOKEN: Bearer token

    Returns:
        AgentIdentity 对象，失败返回 None（不抛异常）
    """
    httpx = _get_httpx()
    if httpx is None:
        logger.warning("httpx 未安装，无法连接 knowlyr-id")
        return None

    base_url, token = _get_config()
    if not token:
        logger.warning("AGENT_API_TOKEN 未设置，跳过身份获取")
        return None

    url = f"{base_url}/api/agents/{agent_id}/config"
    try:
        resp = httpx.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("configured", False):
            logger.warning("Agent %s 尚未配置", agent_id)
            return AgentIdentity(
                agent_id=agent_id,
                nickname=data.get("nickname", ""),
            )

        return AgentIdentity(
            agent_id=agent_id,
            nickname=data.get("nickname", ""),
            title=data.get("title", ""),
            bio=data.get("bio", ""),
            domains=data.get("domains", []),
            model=data.get("model", ""),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            system_prompt=data.get("system_prompt", ""),
            memory=data.get("memory", ""),
        )
    except Exception as e:
        logger.warning("获取 Agent %s 身份失败: %s", agent_id, e)
        return None


def send_heartbeat(agent_id: int, detail: str = "") -> bool:
    """向 knowlyr-id 发送心跳（增加 daily_run_count）.

    Returns:
        True 成功, False 失败（不抛异常）
    """
    httpx = _get_httpx()
    if httpx is None or not _get_config()[1]:
        return False

    base_url, token = _get_config()
    url = f"{base_url}/api/agents/{agent_id}/heartbeat"
    try:
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Agent %s 心跳发送失败: %s", agent_id, e)
        return False


def register_agent(
    nickname: str,
    title: str = "",
    domains: list[str] | None = None,
    model: str = "",
    system_prompt: str = "",
    avatar_base64: str | None = None,
) -> int | None:
    """在 knowlyr-id 注册新 Agent.

    Returns:
        agent_id（成功）或 None（失败，不抛异常）
    """
    httpx = _get_httpx()
    if httpx is None:
        logger.warning("httpx 未安装，无法注册 Agent")
        return None

    base_url, token = _get_config()
    if not token:
        logger.warning("AGENT_API_TOKEN 未设置，跳过 Agent 注册")
        return None

    url = f"{base_url}/api/agents"
    payload: dict = {"nickname": nickname}
    if title:
        payload["title"] = title[:100]
    if domains:
        payload["domains"] = domains[:5]
    if model:
        payload["model"] = model
    if system_prompt:
        payload["system_prompt"] = system_prompt
    if avatar_base64 is not None:
        payload["avatar_base64"] = avatar_base64

    try:
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("agent_id")
    except Exception as e:
        logger.warning("Agent 注册失败: %s", e)
        return None


def update_agent(
    agent_id: int,
    nickname: str | None = None,
    title: str | None = None,
    domains: list[str] | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    memory: str | None = None,
    avatar_base64: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    agent_status: str | None = None,
) -> bool:
    """更新 knowlyr-id Agent 配置.

    Returns:
        True 成功, False 失败（不抛异常）
    """
    httpx = _get_httpx()
    if httpx is None or not _get_config()[1]:
        return False

    base_url, token = _get_config()
    url = f"{base_url}/api/agents/{agent_id}"

    payload: dict = {}
    if nickname is not None:
        payload["nickname"] = nickname
    if title is not None:
        payload["title"] = title[:100]
    if domains is not None:
        payload["domains"] = domains[:5]
    if model is not None:
        payload["model"] = model
    if system_prompt is not None:
        payload["system_prompt"] = system_prompt
    if memory is not None:
        payload["memory"] = memory
    if avatar_base64 is not None:
        payload["avatar_base64"] = avatar_base64
    if api_key is not None:
        payload["api_key"] = api_key
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if agent_status is not None:
        payload["agent_status"] = agent_status

    if not payload:
        return True  # nothing to update

    try:
        resp = httpx.put(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=10.0,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Agent %s 更新失败: %s", agent_id, e)
        return False


def list_agents() -> list[dict] | None:
    """列出 knowlyr-id 中所有活跃 Agent.

    Returns:
        Agent 列表（成功）或 None（失败，不抛异常）
    """
    httpx = _get_httpx()
    if httpx is None:
        logger.warning("httpx 未安装，无法列出 Agent")
        return None

    base_url, token = _get_config()
    if not token:
        logger.warning("AGENT_API_TOKEN 未设置，跳过 Agent 列表")
        return None

    url = f"{base_url}/api/agents"
    try:
        resp = httpx.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Agent 列表获取失败: %s", e)
        return None
