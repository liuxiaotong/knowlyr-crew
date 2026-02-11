"""测试 knowlyr-id 客户端."""

from unittest.mock import MagicMock, patch

from crew.id_client import AgentIdentity, fetch_agent_identity, send_heartbeat


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
