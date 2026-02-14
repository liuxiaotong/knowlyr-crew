"""测试周期性心跳管理器."""

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

from crew.id_client import HeartbeatManager


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

        async def _mock_heartbeat(agent_id, detail=""):
            heartbeat_ids.append(agent_id)
            return True

        with patch.dict(os.environ, {"AGENT_API_TOKEN": "test-token"}, clear=False), \
             patch("crew.id_client.alist_agents", new_callable=AsyncMock, return_value=mock_agents), \
             patch("crew.id_client.asend_heartbeat", side_effect=_mock_heartbeat):
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

        async def _mock_heartbeat(agent_id, detail=""):
            heartbeat_ids.append(agent_id)
            return True

        with patch.dict(os.environ, {"AGENT_API_TOKEN": "test-token"}, clear=False), \
             patch("crew.id_client.alist_agents", new_callable=AsyncMock, return_value=mock_agents), \
             patch("crew.id_client.asend_heartbeat", side_effect=_mock_heartbeat):
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
        call_count = 0

        async def _failing_list():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("网络错误")
            return [{"id": 1, "agent_status": "active"}]

        heartbeat_ids = []

        async def _mock_heartbeat(agent_id, detail=""):
            heartbeat_ids.append(agent_id)
            return True

        with patch.dict(os.environ, {"AGENT_API_TOKEN": "test-token"}, clear=False), \
             patch("crew.id_client.alist_agents", side_effect=_failing_list), \
             patch("crew.id_client.asend_heartbeat", side_effect=_mock_heartbeat):
            mgr = HeartbeatManager(interval=0.2)
            await mgr.start()
            await asyncio.sleep(0.8)
            await mgr.stop()

        # 第一次失败后应继续，后续成功
        assert call_count >= 2
        assert 1 in heartbeat_ids

    @pytest.mark.asyncio
    async def test_running_property(self):
        """running 属性反映实际状态."""
        mgr = HeartbeatManager()
        assert not mgr.running

        with patch.dict(os.environ, {"AGENT_API_TOKEN": "test-token"}, clear=False), \
             patch("crew.id_client.alist_agents", new_callable=AsyncMock, return_value=[]), \
             patch("crew.id_client.asend_heartbeat", new_callable=AsyncMock):
            await mgr.start()
            assert mgr.running
            await mgr.stop()
            assert not mgr.running
