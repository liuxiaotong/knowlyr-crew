"""测试 knowlyr-id 客户端."""

from unittest.mock import MagicMock, patch

import pytest

from crew.id_client import (
    AGENT_MEMORY_LIMIT,
    AgentIdentity,
    append_agent_memory,
    fetch_agent_identity,
    list_agents,
    register_agent,
    send_heartbeat,
    update_agent,
)


@pytest.fixture(autouse=True)
def _reset_breaker():
    """每个测试前重置断路器，避免测试间状态污染."""
    from crew.id_client import _breaker

    _breaker.record_success()  # 重置失败计数
    yield
    _breaker.record_success()  # 测试后也重置


class TestAgentIdentityModel:
    """测试 AgentIdentity 模型."""

    def test_basic_creation(self):
        """应能创建基本的 AgentIdentity."""
        identity = AgentIdentity(agent_id=3050, nickname="Alice")
        assert identity.agent_id == 3050
        assert identity.nickname == "Alice"
        assert identity.domains == []
        assert identity.temperature is None

    def test_full_creation(self):
        """应支持所有字段."""
        identity = AgentIdentity(
            agent_id=3050,
            nickname="Alice",
            title="Senior Reviewer",
            domains=["python", "security"],
            model="claude-opus-4-6",
            temperature=0.7,
            memory="历史记录...",
        )
        assert identity.title == "Senior Reviewer"
        assert identity.domains == ["python", "security"]
        assert identity.temperature == 0.7


class TestFetchAgentIdentity:
    """测试 Agent 身份获取."""

    @patch.dict("os.environ", {"AGENT_API_TOKEN": "test-token", "KNOWLYR_ID_URL": "http://test"})
    @patch("crew.id_client._get_httpx")
    def test_fetch_success(self, mock_httpx_fn):
        """正常获取应返回 AgentIdentity."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "configured": True,
            "nickname": "Alice",
            "title": "Reviewer",
            "model": "claude-opus-4-6",
            "domains": ["python"],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_resp
        mock_httpx_fn.return_value = mock_httpx

        result = fetch_agent_identity(3050)
        assert result is not None
        assert result.agent_id == 3050
        assert result.nickname == "Alice"
        assert result.model == "claude-opus-4-6"

    @patch.dict("os.environ", {"AGENT_API_TOKEN": ""})
    def test_fetch_no_token(self):
        """无 token 应返回 None."""
        result = fetch_agent_identity(3050)
        assert result is None

    @patch.dict("os.environ", {"AGENT_API_TOKEN": "test-token"})
    @patch("crew.id_client._get_httpx", return_value=None)
    def test_fetch_no_httpx(self, _):
        """httpx 未安装应返回 None."""
        result = fetch_agent_identity(3050)
        assert result is None

    @patch.dict("os.environ", {"AGENT_API_TOKEN": "test-token", "KNOWLYR_ID_URL": "http://test"})
    @patch("crew.id_client._get_httpx")
    def test_fetch_network_error(self, mock_httpx_fn):
        """网络错误应返回 None（不抛异常）."""
        mock_httpx = MagicMock()
        mock_httpx.get.side_effect = Exception("connection refused")
        mock_httpx_fn.return_value = mock_httpx

        result = fetch_agent_identity(3050)
        assert result is None

    @patch.dict("os.environ", {"AGENT_API_TOKEN": "test-token", "KNOWLYR_ID_URL": "http://test"})
    @patch("crew.id_client._get_httpx")
    def test_fetch_unconfigured_agent(self, mock_httpx_fn):
        """未配置的 agent 应返回基本身份."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "configured": False,
            "nickname": "Bob",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_resp
        mock_httpx_fn.return_value = mock_httpx

        result = fetch_agent_identity(3050)
        assert result is not None
        assert result.nickname == "Bob"
        assert result.model == ""


class TestSendHeartbeat:
    """测试心跳发送."""

    @patch.dict("os.environ", {"AGENT_API_TOKEN": "test-token", "KNOWLYR_ID_URL": "http://test"})
    @patch("crew.id_client._get_httpx")
    def test_heartbeat_success(self, mock_httpx_fn):
        """成功发送应返回 True."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = mock_resp
        mock_httpx_fn.return_value = mock_httpx

        assert send_heartbeat(3050) is True

    @patch.dict("os.environ", {"AGENT_API_TOKEN": ""})
    def test_heartbeat_no_token(self):
        """无 token 应返回 False."""
        assert send_heartbeat(3050) is False

    @patch.dict("os.environ", {"AGENT_API_TOKEN": "test-token", "KNOWLYR_ID_URL": "http://test"})
    @patch("crew.id_client._get_httpx")
    def test_heartbeat_network_error(self, mock_httpx_fn):
        """网络错误应返回 False（不抛异常）."""
        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = Exception("timeout")
        mock_httpx_fn.return_value = mock_httpx

        assert send_heartbeat(3050) is False


class TestRegisterAgent:
    """测试 Agent 注册."""

    @patch.dict("os.environ", {"AGENT_API_TOKEN": "test-token", "KNOWLYR_ID_URL": "http://test"})
    @patch("crew.id_client._get_httpx")
    def test_register_success(self, mock_httpx_fn):
        """成功注册应返回 agent_id."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"agent_id": 3055, "created": True}
        mock_resp.raise_for_status = MagicMock()
        mock_httpx = MagicMock()
        mock_httpx.post.return_value = mock_resp
        mock_httpx_fn.return_value = mock_httpx

        result = register_agent("测试员工", title="Tester", domains=["test"])
        assert result == 3055
        mock_httpx.post.assert_called_once()
        call_kwargs = mock_httpx.post.call_args
        assert call_kwargs.kwargs["json"]["nickname"] == "测试员工"

    @patch.dict("os.environ", {"AGENT_API_TOKEN": ""})
    def test_register_no_token(self):
        """无 token 应返回 None."""
        assert register_agent("Test") is None

    @patch.dict("os.environ", {"AGENT_API_TOKEN": "test-token"})
    @patch("crew.id_client._get_httpx", return_value=None)
    def test_register_no_httpx(self, _):
        """httpx 未安装应返回 None."""
        assert register_agent("Test") is None

    @patch.dict("os.environ", {"AGENT_API_TOKEN": "test-token", "KNOWLYR_ID_URL": "http://test"})
    @patch("crew.id_client._get_httpx")
    def test_register_network_error(self, mock_httpx_fn):
        """网络错误应返回 None."""
        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = Exception("connection refused")
        mock_httpx_fn.return_value = mock_httpx

        assert register_agent("Test") is None


class TestUpdateAgent:
    """测试 Agent 更新."""

    @patch.dict("os.environ", {"AGENT_API_TOKEN": "test-token", "KNOWLYR_ID_URL": "http://test"})
    @patch("crew.id_client._get_httpx")
    def test_update_success(self, mock_httpx_fn):
        """成功更新应返回 True."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"agent_id": 3050, "updated": True}
        mock_resp.raise_for_status = MagicMock()
        mock_httpx = MagicMock()
        mock_httpx.put.return_value = mock_resp
        mock_httpx_fn.return_value = mock_httpx

        assert update_agent(3050, nickname="New Name", title="New Title") is True


class TestAppendAgentMemory:
    """测试 append_agent_memory 行为."""

    @patch("crew.id_client.update_agent", return_value=True)
    @patch("crew.id_client.fetch_agent_identity")
    def test_append_basic(self, mock_fetch, mock_update):
        identity = AgentIdentity(agent_id=1, nickname="emp", memory="旧摘要")
        mock_fetch.return_value = identity
        ok = append_agent_memory(1, "新的总结")
        assert ok is True
        mock_update.assert_called_once()
        assert mock_update.call_args.kwargs["memory"] == "旧摘要\n\n新的总结"

    @patch("crew.id_client.update_agent", return_value=True)
    @patch("crew.id_client.fetch_agent_identity")
    def test_append_trims_oldest_chunk(self, mock_fetch, mock_update):
        identity = AgentIdentity(agent_id=1, nickname="emp", memory="第一条\n\n第二条")
        mock_fetch.return_value = identity
        with patch("crew.id_client.AGENT_MEMORY_LIMIT", 12):
            append_agent_memory(1, "第三条很长")
        sent = mock_update.call_args.kwargs["memory"]
        assert "第一条" not in sent
        assert sent.startswith("第二条")
        assert "第三条" in sent

    @patch("crew.id_client.update_agent", return_value=True)
    @patch("crew.id_client.fetch_agent_identity")
    def test_append_large_summary(self, mock_fetch, mock_update):
        identity = AgentIdentity(agent_id=1, nickname="emp", memory="")
        mock_fetch.return_value = identity
        huge = "X" * (AGENT_MEMORY_LIMIT + 100)
        append_agent_memory(1, huge)
        sent = mock_update.call_args.kwargs["memory"]
        assert len(sent) == AGENT_MEMORY_LIMIT
        assert sent == huge[-AGENT_MEMORY_LIMIT:]

    @patch.dict("os.environ", {"AGENT_API_TOKEN": "test-token"})
    def test_update_nothing(self):
        """无更新内容应返回 True."""
        assert update_agent(3050) is True

    @patch.dict("os.environ", {"AGENT_API_TOKEN": ""})
    def test_update_no_token(self):
        """无 token 应返回 False."""
        assert update_agent(3050, nickname="Test") is False

    @patch.dict("os.environ", {"AGENT_API_TOKEN": "test-token", "KNOWLYR_ID_URL": "http://test"})
    @patch("crew.id_client._get_httpx")
    def test_update_network_error(self, mock_httpx_fn):
        """网络错误应返回 False."""
        mock_httpx = MagicMock()
        mock_httpx.put.side_effect = Exception("timeout")
        mock_httpx_fn.return_value = mock_httpx

        assert update_agent(3050, title="Test") is False


class TestListAgents:
    """测试 Agent 列表."""

    @patch.dict("os.environ", {"AGENT_API_TOKEN": "test-token", "KNOWLYR_ID_URL": "http://test"})
    @patch("crew.id_client._get_httpx")
    def test_list_success(self, mock_httpx_fn):
        """成功获取应返回列表."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"id": 3050, "nickname": "Alice"},
            {"id": 3051, "nickname": "Bob"},
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_resp
        mock_httpx_fn.return_value = mock_httpx

        result = list_agents()
        assert result is not None
        assert len(result) == 2

    @patch.dict("os.environ", {"AGENT_API_TOKEN": ""})
    def test_list_no_token(self):
        """无 token 应返回 None."""
        assert list_agents() is None

    @patch.dict("os.environ", {"AGENT_API_TOKEN": "test-token"})
    @patch("crew.id_client._get_httpx", return_value=None)
    def test_list_no_httpx(self, _):
        """httpx 未安装应返回 None."""
        assert list_agents() is None
