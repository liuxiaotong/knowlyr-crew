"""knowlyr-id 客户端 — 可选的 Agent 身份对接."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time as _time

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

AGENT_MEMORY_LIMIT = 2000


class WorkLogResult(BaseModel):
    """POST /api/work 的返回结果."""

    work_log_id: int
    is_exemplar: bool = False


class HumanFeedback(BaseModel):
    """人工评分反馈条目."""

    work_log_id: int
    task_type: str = ""
    auto_score: float | None = None
    human_score: float
    is_exemplar: bool = False
    output_preview: str = ""
    error: str = ""
    created_at: str = ""


class _CircuitBreaker:
    """Simple circuit breaker: after N consecutive failures, skip calls for cooldown_seconds."""

    def __init__(self, threshold: int = 3, cooldown: float = 30.0):
        self._threshold = threshold
        self._cooldown = cooldown
        self._failures = 0
        self._last_failure: float = 0.0
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        with self._lock:
            if self._failures < self._threshold:
                return False
            if _time.monotonic() - self._last_failure > self._cooldown:
                self._failures = 0
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._last_failure = _time.monotonic()


_breaker = _CircuitBreaker()


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


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


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
    crew_name: str = ""


def fetch_agent_identity(agent_id: int) -> AgentIdentity | None:
    """从 knowlyr-id 获取 Agent 身份配置.

    环境变量:
        KNOWLYR_ID_URL: 服务地址（默认 http://localhost:8000）
        AGENT_API_TOKEN: Bearer token

    Returns:
        AgentIdentity 对象，失败返回 None（不抛异常）
    """
    if _breaker.is_open():
        logger.debug("knowlyr-id 断路器打开，跳过请求")
        return None

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
        resp = httpx.get(url, headers=_headers(token), timeout=5.0)
        resp.raise_for_status()
        _breaker.record_success()
        return _parse_identity(agent_id, resp.json())
    except Exception as e:
        _breaker.record_failure()
        logger.warning("获取 Agent %s 身份失败: %s", agent_id, e)
        return None


def send_heartbeat(
    agent_id: int,
    detail: str = "",
    *,
    task_type: str = "",
    tokens_used: int = 0,
    model_used: str = "",
    execution_ms: int = 0,
    error: str = "",
) -> bool:
    """向 knowlyr-id 发送心跳（附带执行上下文）.

    Returns:
        True 成功, False 失败（不抛异常）
    """
    if _breaker.is_open():
        logger.debug("knowlyr-id 断路器打开，跳过请求")
        return False

    httpx = _get_httpx()
    if httpx is None or not _get_config()[1]:
        return False

    base_url, token = _get_config()
    url = f"{base_url}/api/agents/{agent_id}/heartbeat"
    payload: dict = {}
    if detail:
        payload["label"] = detail
    if task_type:
        payload["task_type"] = task_type
    if tokens_used:
        payload["tokens_used"] = tokens_used
    if model_used:
        payload["model_used"] = model_used
    if execution_ms:
        payload["execution_ms"] = execution_ms
    if error:
        payload["error"] = error[:200]
    try:
        resp = httpx.post(
            url,
            headers=_headers(token),
            json=payload if payload else None,
            timeout=5.0,
        )
        resp.raise_for_status()
        _breaker.record_success()
        return True
    except Exception as e:
        _breaker.record_failure()
        logger.warning("Agent %s 心跳发送失败: %s", agent_id, e)
        return False


def register_agent(
    nickname: str,
    title: str = "",
    capabilities: str = "",
    domains: list[str] | None = None,
    model: str = "",
    system_prompt: str = "",
    avatar_base64: str | None = None,
) -> int | None:
    """在 knowlyr-id 注册新 Agent.

    Returns:
        agent_id（成功）或 None（失败，不抛异常）
    """
    if _breaker.is_open():
        logger.debug("knowlyr-id 断路器打开，跳过请求")
        return None

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
    if capabilities:
        payload["capabilities"] = capabilities
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
            headers=_headers(token),
            json=payload,
            timeout=10.0,
        )
        resp.raise_for_status()
        _breaker.record_success()
        data = resp.json()
        return data.get("agent_id")
    except Exception as e:
        _breaker.record_failure()
        logger.warning("Agent 注册失败: %s", e)
        return None


def update_agent(
    agent_id: int,
    nickname: str | None = None,
    title: str | None = None,
    bio: str | None = None,
    capabilities: str | None = None,
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
    if _breaker.is_open():
        logger.debug("knowlyr-id 断路器打开，跳过请求")
        return False

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
    if bio is not None:
        payload["bio"] = bio
    if capabilities is not None:
        payload["capabilities"] = capabilities
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
            headers=_headers(token),
            json=payload,
            timeout=10.0,
        )
        resp.raise_for_status()
        _breaker.record_success()
        return True
    except Exception as e:
        _breaker.record_failure()
        logger.warning("Agent %s 更新失败: %s", agent_id, e)
        return False


def _combine_agent_memory(existing: str, addition: str, limit: int | None = None) -> str:
    """将新的摘要追加到已有记忆并在 limit 内裁剪."""
    limit = limit or AGENT_MEMORY_LIMIT
    existing = (existing or "").strip()
    addition = (addition or "").strip()
    chunks: list[str] = []
    if existing:
        pieces = [chunk.strip() for chunk in existing.split("\n\n") if chunk.strip()]
        if pieces:
            chunks.extend(pieces)
        else:
            chunks.append(existing)
    if addition:
        chunks.append(addition)
    if not chunks:
        return ""
    trimmed = "\n\n".join(chunks)
    removed = ""
    while len(trimmed) > limit and chunks:
        removed = chunks.pop(0)
        trimmed = "\n\n".join(chunks)
    if trimmed:
        return trimmed
    fallback = addition or removed
    return fallback[-limit:]


def append_agent_memory(agent_id: int, summary: str) -> bool:
    """将新的摘要拼接到 knowlyr-id Agent 的 memory 字段."""
    identity = fetch_agent_identity(agent_id)
    if identity is None:
        return False

    combined = _combine_agent_memory(identity.memory, summary)
    return update_agent(agent_id, memory=combined)


def list_agents() -> list[dict] | None:
    """列出 knowlyr-id 中所有活跃 Agent.

    Returns:
        Agent 列表（成功）或 None（失败，不抛异常）
    """
    if _breaker.is_open():
        logger.debug("knowlyr-id 断路器打开，跳过请求")
        return None

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
            headers=_headers(token),
            timeout=5.0,
        )
        resp.raise_for_status()
        _breaker.record_success()
        return resp.json()
    except Exception as e:
        _breaker.record_failure()
        logger.warning("Agent 列表获取失败: %s", e)
        return None


# ── 异步版本（供 MCP server / async 上下文使用）──


def _parse_identity(agent_id: int, data: dict) -> AgentIdentity:
    """从 API 响应解析 AgentIdentity."""
    if not data.get("configured", False):
        logger.warning("Agent %s 尚未配置", agent_id)
        return AgentIdentity(agent_id=agent_id, nickname=data.get("nickname", ""))
    return AgentIdentity(
        agent_id=agent_id,
        nickname=data.get("nickname", ""),
        title=data.get("title", ""),
        bio=data.get("bio", ""),
        domains=data.get("domains", []),
        model=data.get("model", ""),
        api_key=data.get("api_key", ""),
        temperature=data.get("temperature"),
        max_tokens=data.get("max_tokens"),
        system_prompt=data.get("system_prompt", ""),
        memory=data.get("memory", ""),
        crew_name=data.get("crew_name", ""),
    )


async def afetch_agent_identity(agent_id: int) -> AgentIdentity | None:
    """fetch_agent_identity 的异步版本."""
    if _breaker.is_open():
        logger.debug("knowlyr-id 断路器打开，跳过请求")
        return None
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
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=_headers(token), timeout=5.0)
            resp.raise_for_status()
            _breaker.record_success()
            return _parse_identity(agent_id, resp.json())
    except Exception as e:
        _breaker.record_failure()
        logger.warning("获取 Agent %s 身份失败: %s", agent_id, e)
        return None


async def asend_heartbeat(
    agent_id: int,
    detail: str = "",
    *,
    task_type: str = "",
    tokens_used: int = 0,
    model_used: str = "",
    execution_ms: int = 0,
    error: str = "",
) -> bool:
    """send_heartbeat 的异步版本."""
    if _breaker.is_open():
        logger.debug("knowlyr-id 断路器打开，跳过请求")
        return False
    httpx = _get_httpx()
    if httpx is None or not _get_config()[1]:
        return False
    base_url, token = _get_config()
    url = f"{base_url}/api/agents/{agent_id}/heartbeat"
    payload: dict = {}
    if detail:
        payload["label"] = detail
    if task_type:
        payload["task_type"] = task_type
    if tokens_used:
        payload["tokens_used"] = tokens_used
    if model_used:
        payload["model_used"] = model_used
    if execution_ms:
        payload["execution_ms"] = execution_ms
    if error:
        payload["error"] = error[:200]
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url, headers=_headers(token),
                json=payload if payload else None,
                timeout=5.0,
            )
            resp.raise_for_status()
            _breaker.record_success()
            return True
    except Exception as e:
        _breaker.record_failure()
        logger.warning("Agent %s 心跳发送失败: %s", agent_id, e)
        return False


async def aregister_agent(
    nickname: str,
    title: str = "",
    capabilities: str = "",
    domains: list[str] | None = None,
    model: str = "",
    system_prompt: str = "",
    avatar_base64: str | None = None,
) -> int | None:
    """register_agent 的异步版本."""
    if _breaker.is_open():
        logger.debug("knowlyr-id 断路器打开，跳过请求")
        return None
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
    if capabilities:
        payload["capabilities"] = capabilities
    if domains:
        payload["domains"] = domains[:5]
    if model:
        payload["model"] = model
    if system_prompt:
        payload["system_prompt"] = system_prompt
    if avatar_base64 is not None:
        payload["avatar_base64"] = avatar_base64
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=_headers(token), json=payload, timeout=10.0)
            resp.raise_for_status()
            _breaker.record_success()
            return resp.json().get("agent_id")
    except Exception as e:
        _breaker.record_failure()
        logger.warning("Agent 注册失败: %s", e)
        return None


async def aupdate_agent(
    agent_id: int,
    nickname: str | None = None,
    title: str | None = None,
    capabilities: str | None = None,
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
    """update_agent 的异步版本."""
    if _breaker.is_open():
        logger.debug("knowlyr-id 断路器打开，跳过请求")
        return False
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
    if capabilities is not None:
        payload["capabilities"] = capabilities
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
        return True
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.put(url, headers=_headers(token), json=payload, timeout=10.0)
            resp.raise_for_status()
            _breaker.record_success()
            return True
    except Exception as e:
        _breaker.record_failure()
        logger.warning("Agent %s 更新失败: %s", agent_id, e)
        return False


async def aappend_agent_memory(agent_id: int, summary: str) -> bool:
    """append_agent_memory 的异步版本."""
    identity = await afetch_agent_identity(agent_id)
    if identity is None:
        return False
    combined = _combine_agent_memory(identity.memory, summary)
    return await aupdate_agent(agent_id, memory=combined)


async def alist_agents() -> list[dict] | None:
    """list_agents 的异步版本."""
    if _breaker.is_open():
        logger.debug("knowlyr-id 断路器打开，跳过请求")
        return None
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
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=_headers(token), timeout=5.0)
            resp.raise_for_status()
            _breaker.record_success()
            return resp.json()
    except Exception as e:
        _breaker.record_failure()
        logger.warning("Agent 列表获取失败: %s", e)
        return None


# ── 范例获取 ──


def fetch_exemplars(agent_id: int, task_type: str, limit: int = 2) -> str:
    """从 knowlyr-id 获取 few-shot 范例提示文本.

    Returns:
        格式化的 markdown 字符串，无范例时返回空字符串
    """
    if _breaker.is_open():
        return ""
    httpx = _get_httpx()
    if httpx is None or not _get_config()[1]:
        return ""
    base_url, token = _get_config()
    url = f"{base_url}/api/work/exemplars"
    try:
        resp = httpx.get(
            url,
            headers=_headers(token),
            params={"agent_id": agent_id, "task_type": task_type, "limit": limit},
            timeout=5.0,
        )
        resp.raise_for_status()
        _breaker.record_success()
        return resp.json().get("prompt", "")
    except Exception as e:
        _breaker.record_failure()
        logger.warning("Agent %s 范例获取失败: %s", agent_id, e)
        return ""


async def afetch_exemplars(agent_id: int, task_type: str, limit: int = 2) -> str:
    """fetch_exemplars 的异步版本."""
    if _breaker.is_open():
        return ""
    httpx = _get_httpx()
    if httpx is None or not _get_config()[1]:
        return ""
    base_url, token = _get_config()
    url = f"{base_url}/api/work/exemplars"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                url,
                headers=_headers(token),
                params={"agent_id": agent_id, "task_type": task_type, "limit": limit},
                timeout=5.0,
            )
            resp.raise_for_status()
            _breaker.record_success()
            return resp.json().get("prompt", "")
    except Exception as e:
        _breaker.record_failure()
        logger.warning("Agent %s 范例获取失败: %s", agent_id, e)
        return ""


# ── 人工评分反馈 ──


def fetch_feedback(
    agent_id: int,
    task_type: str = "",
    limit: int = 5,
) -> list[HumanFeedback]:
    """从 knowlyr-id 获取最近的人工评分反馈.

    Returns:
        HumanFeedback 列表，失败返回空列表
    """
    if _breaker.is_open():
        return []
    httpx = _get_httpx()
    if httpx is None or not _get_config()[1]:
        return []
    base_url, token = _get_config()
    url = f"{base_url}/api/work/feedback"
    params: dict = {"agent_id": agent_id, "limit": limit}
    if task_type:
        params["task_type"] = task_type
    try:
        resp = httpx.get(url, headers=_headers(token), params=params, timeout=5.0)
        resp.raise_for_status()
        _breaker.record_success()
        return [HumanFeedback(**f) for f in resp.json().get("feedbacks", [])]
    except Exception as e:
        _breaker.record_failure()
        logger.warning("Agent %s 反馈获取失败: %s", agent_id, e)
        return []


async def afetch_feedback(
    agent_id: int,
    task_type: str = "",
    limit: int = 5,
) -> list[HumanFeedback]:
    """fetch_feedback 的异步版本."""
    if _breaker.is_open():
        return []
    httpx = _get_httpx()
    if httpx is None or not _get_config()[1]:
        return []
    base_url, token = _get_config()
    url = f"{base_url}/api/work/feedback"
    params: dict = {"agent_id": agent_id, "limit": limit}
    if task_type:
        params["task_type"] = task_type
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=_headers(token), params=params, timeout=5.0)
            resp.raise_for_status()
            _breaker.record_success()
            return [HumanFeedback(**f) for f in resp.json().get("feedbacks", [])]
    except Exception as e:
        _breaker.record_failure()
        logger.warning("Agent %s 反馈获取失败: %s", agent_id, e)
        return []


# ── 工作记录提交 ──


def _build_work_payload(
    agent_id: int,
    task_type: str,
    task_input: str,
    task_output: str,
    auto_score: float | None,
    crew_task_id: str,
    tokens_used: int,
    model_used: str,
    execution_ms: int,
    error: str,
) -> dict:
    payload: dict = {"agent_id": agent_id, "task_type": task_type}
    if task_input:
        payload["task_input"] = task_input[:10000]
    if task_output:
        payload["task_output"] = task_output[:10000]
    if auto_score is not None:
        payload["auto_score"] = auto_score
    if crew_task_id:
        payload["crew_task_id"] = crew_task_id
    if tokens_used:
        payload["tokens_used"] = tokens_used
    if model_used:
        payload["model_used"] = model_used
    if execution_ms:
        payload["execution_ms"] = execution_ms
    if error:
        payload["error"] = error[:2000]
    return payload


def log_work(
    agent_id: int,
    task_type: str,
    task_input: str = "",
    task_output: str = "",
    auto_score: float | None = None,
    crew_task_id: str = "",
    tokens_used: int = 0,
    model_used: str = "",
    execution_ms: int = 0,
    error: str = "",
) -> WorkLogResult | None:
    """向 knowlyr-id 提交工作记录.

    Returns:
        WorkLogResult（成功）或 None（失败，不抛异常）
    """
    if _breaker.is_open():
        logger.debug("knowlyr-id 断路器打开，跳过请求")
        return None
    httpx = _get_httpx()
    if httpx is None:
        logger.warning("httpx 未安装，无法提交工作记录")
        return None
    base_url, token = _get_config()
    if not token:
        logger.warning("AGENT_API_TOKEN 未设置，跳过工作记录提交")
        return None
    url = f"{base_url}/api/work"
    payload = _build_work_payload(
        agent_id, task_type, task_input, task_output,
        auto_score, crew_task_id, tokens_used, model_used, execution_ms, error,
    )
    try:
        resp = httpx.post(url, headers=_headers(token), json=payload, timeout=10.0)
        resp.raise_for_status()
        _breaker.record_success()
        data = resp.json()
        return WorkLogResult(
            work_log_id=data["work_log_id"],
            is_exemplar=data.get("is_exemplar", False),
        )
    except Exception as e:
        _breaker.record_failure()
        logger.warning("Agent %s 工作记录提交失败: %s", agent_id, e)
        return None


async def alog_work(
    agent_id: int,
    task_type: str,
    task_input: str = "",
    task_output: str = "",
    auto_score: float | None = None,
    crew_task_id: str = "",
    tokens_used: int = 0,
    model_used: str = "",
    execution_ms: int = 0,
    error: str = "",
) -> WorkLogResult | None:
    """log_work 的异步版本."""
    if _breaker.is_open():
        logger.debug("knowlyr-id 断路器打开，跳过请求")
        return None
    httpx = _get_httpx()
    if httpx is None:
        logger.warning("httpx 未安装，无法提交工作记录")
        return None
    base_url, token = _get_config()
    if not token:
        logger.warning("AGENT_API_TOKEN 未设置，跳过工作记录提交")
        return None
    url = f"{base_url}/api/work"
    payload = _build_work_payload(
        agent_id, task_type, task_input, task_output,
        auto_score, crew_task_id, tokens_used, model_used, execution_ms, error,
    )
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=_headers(token), json=payload, timeout=10.0)
            resp.raise_for_status()
            _breaker.record_success()
            data = resp.json()
            return WorkLogResult(
                work_log_id=data["work_log_id"],
                is_exemplar=data.get("is_exemplar", False),
            )
    except Exception as e:
        _breaker.record_failure()
        logger.warning("Agent %s 工作记录提交失败: %s", agent_id, e)
        return None


# ── 周期性心跳管理器 ──


class HeartbeatManager:
    """后台周期性心跳管理器，为所有 active agent 定期发送心跳."""

    def __init__(self, interval: float = 60.0):
        self._interval = interval
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """启动心跳循环。仅在 AGENT_API_TOKEN 存在时生效."""
        if not os.environ.get("AGENT_API_TOKEN"):
            logger.info("AGENT_API_TOKEN 未设置，心跳管理器未启动")
            return
        self._task = asyncio.create_task(self._loop(), name="heartbeat-manager")
        logger.info("心跳管理器已启动 (间隔 %.0fs)", self._interval)

    async def stop(self) -> None:
        """停止心跳循环."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("心跳管理器已停止")

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _loop(self) -> None:
        while True:
            try:
                agents = await alist_agents()
                if agents:
                    for agent in agents:
                        aid = agent.get("id")
                        status = agent.get("agent_status", "active")
                        if aid and status == "active":
                            await asend_heartbeat(aid, detail="periodic")
                            await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("周期心跳失败: %s", e)
            await asyncio.sleep(self._interval)
