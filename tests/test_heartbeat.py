"""测试周期性心跳管理器."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crew.id_client import HeartbeatManager


@pytest.fixture(autouse=True)
def _reset_breaker():
    """每个测试前重置断路器，避免测试间状态污染."""
    from crew.id_client import _breaker

    _breaker.record_success()
    yield
    _breaker.record_success()


def _make_mock_httpx(agents_response, *, heartbeat_ids=None, fail_list_until=0):
    """构造 mock httpx 模块，模拟 AsyncClient 的 get/post 行为.

    Args:
        agents_response: agent 列表 API 返回的数据
        heartbeat_ids: 可选列表，记录收到心跳的 agent_id
        fail_list_until: 前 N 次 list 调用抛异常
    """
    call_count = {"list": 0}

    async def mock_get(url, **kwargs):
        call_count["list"] += 1
        if call_count["list"] <= fail_list_until:
            raise ConnectionError("网络错误")
        resp = MagicMock()
        resp.json.return_value = agents_response
        resp.raise_for_status = MagicMock()
        return resp

    async def mock_post(url, **kwargs):
        if heartbeat_ids is not None:
            # 从 URL 提取 agent_id: .../api/agents/{aid}/heartbeat
            parts = url.rstrip("/").split("/")
            try:
                idx = parts.index("agents")
                aid = int(parts[idx + 1])
                heartbeat_ids.append(aid)
            except (ValueError, IndexError):
                pass
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        return resp

    # 构造 async context manager mock
    mock_client = AsyncMock()
    mock_client.get = mock_get
    mock_client.post = mock_post
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_httpx = MagicMock()
    mock_httpx.AsyncClient.return_value = mock_client

    return mock_httpx, call_count


class TestHeartbeatManager:
    """HeartbeatManager 单元测试."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """启动后发送心跳，停止后不再发送."""
        mock_agents = [
            {"id": 1, "nickname": "Alice", "agent_status": "active"},
            {"id": 2, "nickname": "Bob", "agent_status": "active"},
        ]
        heartbeat_ids = []
        mock_httpx, _ = _make_mock_httpx(mock_agents, heartbeat_ids=heartbeat_ids)

        with (
            patch.dict(os.environ, {"AGENT_API_TOKEN": "test-token"}, clear=False),
            patch("crew.id_client._get_httpx", return_value=mock_httpx),
        ):
            mgr = HeartbeatManager(interval=0.2)
            await mgr.start()
            assert mgr.running

            # 等待至少一轮心跳
            await asyncio.sleep(0.5)
            await mgr.stop()

            assert not mgr.running
            assert 1 in heartbeat_ids
            assert 2 in heartbeat_ids

    @pytest.mark.asyncio
    async def test_skips_inactive(self):
        """agent_status != 'active' 的 agent 不发心跳."""
        mock_agents = [
            {"id": 1, "nickname": "Alice", "agent_status": "active"},
            {"id": 2, "nickname": "Bob", "agent_status": "inactive"},
            {"id": 3, "nickname": "Carol", "agent_status": "suspended"},
        ]
        heartbeat_ids = []
        mock_httpx, _ = _make_mock_httpx(mock_agents, heartbeat_ids=heartbeat_ids)

        with (
            patch.dict(os.environ, {"AGENT_API_TOKEN": "test-token"}, clear=False),
            patch("crew.id_client._get_httpx", return_value=mock_httpx),
        ):
            mgr = HeartbeatManager(interval=0.2)
            await mgr.start()
            await asyncio.sleep(0.5)
            await mgr.stop()

        # 只有 agent 1 (active) 收到心跳
        assert 1 in heartbeat_ids
        assert 2 not in heartbeat_ids
        assert 3 not in heartbeat_ids

    @pytest.mark.asyncio
    async def test_no_token(self):
        """无 AGENT_API_TOKEN 时不启动."""
        with patch.dict(os.environ, {"AGENT_API_TOKEN": ""}, clear=False):
            mgr = HeartbeatManager(interval=1.0)
            await mgr.start()
            assert not mgr.running
            await mgr.stop()  # 应该不报错

    @pytest.mark.asyncio
    async def test_api_failure_continues(self):
        """API 失败不影响循环继续."""
        mock_agents = [{"id": 1, "agent_status": "active"}]
        heartbeat_ids = []
        mock_httpx, call_count = _make_mock_httpx(
            mock_agents,
            heartbeat_ids=heartbeat_ids,
            fail_list_until=1,  # 第一次 list 失败
        )

        with (
            patch.dict(os.environ, {"AGENT_API_TOKEN": "test-token"}, clear=False),
            patch("crew.id_client._get_httpx", return_value=mock_httpx),
        ):
            mgr = HeartbeatManager(interval=0.2)
            await mgr.start()
            await asyncio.sleep(0.8)
            await mgr.stop()

        # 第一次失败后应继续，后续成功
        assert call_count["list"] >= 2
        assert 1 in heartbeat_ids

    @pytest.mark.asyncio
    async def test_running_property(self):
        """running 属性反映实际状态."""
        mgr = HeartbeatManager()
        assert not mgr.running

        mock_httpx, _ = _make_mock_httpx([])

        with (
            patch.dict(os.environ, {"AGENT_API_TOKEN": "test-token"}, clear=False),
            patch("crew.id_client._get_httpx", return_value=mock_httpx),
        ):
            await mgr.start()
            assert mgr.running
            await mgr.stop()
            assert not mgr.running
