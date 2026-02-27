"""测试 Memory 远程 API — 服务端端点 + MCP server 远程调用."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── 服务端端点测试 ──────────────────────────────────────────────

starlette = pytest.importorskip("starlette")

from starlette.testclient import TestClient

from crew.webhook import create_webhook_app
from crew.webhook_config import WebhookConfig

TOKEN = "test-memory-token"


def _make_client(project_dir=None):
    """创建测试客户端."""
    app = create_webhook_app(
        project_dir=project_dir or Path("/tmp/test-memory"),
        token=TOKEN,
        config=WebhookConfig(),
    )
    return TestClient(app)


class TestMemoryAddEndpoint:
    """POST /api/memory/add — 记忆写入端点."""

    def test_add_requires_auth(self):
        """未认证时返回 401."""
        client = _make_client()
        resp = client.post(
            "/api/memory/add",
            json={"employee": "test", "category": "finding", "content": "hello"},
        )
        assert resp.status_code == 401

    def test_add_missing_fields(self):
        """缺少必填字段时返回 400."""
        client = _make_client()
        resp = client.post(
            "/api/memory/add",
            json={"employee": "test"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 400
        assert "required" in resp.json()["error"]

    def test_add_invalid_category(self):
        """无效 category 返回 400."""
        client = _make_client()
        resp = client.post(
            "/api/memory/add",
            json={"employee": "test", "category": "invalid", "content": "hello"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 400
        assert "category" in resp.json()["error"]

    @patch("crew.memory.MemoryStore.add")
    @patch("crew.memory.MemoryStore.query")
    def test_add_basic(self, mock_query, mock_add):
        """基本写入成功."""
        from crew.memory import MemoryEntry

        mock_query.return_value = []
        mock_add.return_value = MemoryEntry(
            id="abc123", employee="backend-engineer", category="finding", content="测试记忆"
        )

        client = _make_client()
        resp = client.post(
            "/api/memory/add",
            json={
                "employee": "backend-engineer",
                "category": "finding",
                "content": "测试记忆",
            },
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["skipped"] is False
        assert data["entry_id"] == "abc123"

    @patch("crew.memory.MemoryStore.add")
    @patch("crew.memory.MemoryStore.query")
    def test_add_with_full_params(self, mock_query, mock_add):
        """传入全部参数（含 pattern 专有字段）成功."""
        from crew.memory import MemoryEntry

        mock_query.return_value = []
        mock_add.return_value = MemoryEntry(
            id="pat123", employee="backend-engineer", category="pattern", content="test"
        )

        client = _make_client()
        resp = client.post(
            "/api/memory/add",
            json={
                "employee": "backend-engineer",
                "category": "pattern",
                "content": "代码审查时先看测试覆盖率",
                "tags": ["code-review", "testing"],
                "ttl_days": 90,
                "shared": True,
                "trigger_condition": "代码审查场景",
                "applicability": ["backend", "frontend"],
                "origin_employee": "code-reviewer",
            },
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True

        # 验证 store.add 调用时传入了所有参数
        call_kwargs = mock_add.call_args[1]
        assert call_kwargs["ttl_days"] == 90
        assert call_kwargs["shared"] is True
        assert call_kwargs["trigger_condition"] == "代码审查场景"
        assert call_kwargs["applicability"] == ["backend", "frontend"]
        assert call_kwargs["origin_employee"] == "code-reviewer"

    @patch("crew.memory.MemoryStore.add")
    @patch("crew.memory.MemoryStore.query")
    def test_add_idempotent(self, mock_query, mock_add):
        """幂等检查：相同 source_session + category 不重复写入."""
        from crew.memory import MemoryEntry

        existing_entry = MemoryEntry(
            id="existing-id",
            employee="test",
            category="finding",
            content="old",
            source_session="sess-001",
        )
        mock_query.return_value = [existing_entry]

        client = _make_client()
        resp = client.post(
            "/api/memory/add",
            json={
                "employee": "test",
                "category": "finding",
                "content": "duplicate",
                "source_session": "sess-001",
            },
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["skipped"] is True
        assert data["existing_id"] == "existing-id"
        mock_add.assert_not_called()

    def test_add_invalid_json(self):
        """无效 JSON body 返回 400."""
        client = _make_client()
        resp = client.post(
            "/api/memory/add",
            content=b"not json",
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 400


class TestMemoryQueryEndpoint:
    """GET /api/memory/query — 记忆查询端点."""

    def test_query_requires_auth(self):
        """未认证时返回 401."""
        client = _make_client()
        resp = client.get("/api/memory/query?employee=test")
        assert resp.status_code == 401

    def test_query_missing_employee(self):
        """缺少 employee 参数返回 400."""
        client = _make_client()
        resp = client.get(
            "/api/memory/query",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 400
        assert "employee" in resp.json()["error"]

    @patch("crew.memory.MemoryStore.query")
    def test_query_basic(self, mock_query):
        """基本查询成功."""
        from crew.memory import MemoryEntry

        mock_query.return_value = [
            MemoryEntry(
                id="mem1",
                employee="backend-engineer",
                category="finding",
                content="测试发现",
            ),
        ]

        client = _make_client()
        resp = client.get(
            "/api/memory/query?employee=backend-engineer",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["total"] == 1
        assert len(data["entries"]) == 1
        assert data["entries"][0]["content"] == "测试发现"

    @patch("crew.memory.MemoryStore.query")
    def test_query_with_category_and_limit(self, mock_query):
        """传入 category 和 limit 参数."""
        mock_query.return_value = []

        client = _make_client()
        resp = client.get(
            "/api/memory/query?employee=test&category=decision&limit=5",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["total"] == 0

        # 验证传给 store.query 的参数
        call_kwargs = mock_query.call_args[1]
        assert call_kwargs["employee"] == "test"
        assert call_kwargs["category"] == "decision"
        assert call_kwargs["limit"] == 5

    @patch("crew.memory.MemoryStore.query")
    def test_query_empty_result(self, mock_query):
        """无记忆时返回空列表."""
        mock_query.return_value = []

        client = _make_client()
        resp = client.get(
            "/api/memory/query?employee=nobody",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["entries"] == []
        assert data["total"] == 0

    @patch("crew.memory.MemoryStore.query")
    def test_query_invalid_limit_falls_back(self, mock_query):
        """无效 limit 值回退到默认值 20."""
        mock_query.return_value = []

        client = _make_client()
        resp = client.get(
            "/api/memory/query?employee=test&limit=abc",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200

        call_kwargs = mock_query.call_args[1]
        assert call_kwargs["limit"] == 20


# ── MCP Server 远程 Memory 测试 ──────────────────────────────────

mcp_mod = pytest.importorskip("mcp")

from crew.mcp_server import (
    _get_remote_memory_config,
    _remote_memory_add,
    _remote_memory_query,
    create_server,
)


def _run(coro):
    """同步运行 async 函数."""
    return asyncio.run(coro)


class TestRemoteMemoryConfig:
    """测试远程 memory API 配置获取."""

    def test_no_env_returns_none(self, monkeypatch):
        """未设置环境变量时返回 None."""
        monkeypatch.delenv("CREW_API_URL", raising=False)
        monkeypatch.delenv("CREW_API_TOKEN", raising=False)
        assert _get_remote_memory_config() is None

    def test_only_url_returns_none(self, monkeypatch):
        """只设置 URL 缺少 token 时返回 None."""
        monkeypatch.setenv("CREW_API_URL", "https://crew.knowlyr.com")
        monkeypatch.delenv("CREW_API_TOKEN", raising=False)
        assert _get_remote_memory_config() is None

    def test_both_set_returns_tuple(self, monkeypatch):
        """URL 和 token 都设置时返回 (url, token)."""
        monkeypatch.setenv("CREW_API_URL", "https://crew.knowlyr.com/")
        monkeypatch.setenv("CREW_API_TOKEN", "my-token")
        result = _get_remote_memory_config()
        assert result == ("https://crew.knowlyr.com", "my-token")

    def test_trailing_slash_stripped(self, monkeypatch):
        """URL 末尾斜杠被去除."""
        monkeypatch.setenv("CREW_API_URL", "https://crew.knowlyr.com///")
        monkeypatch.setenv("CREW_API_TOKEN", "tok")
        url, _ = _get_remote_memory_config()
        assert not url.endswith("/")


class TestRemoteMemoryAdd:
    """测试 _remote_memory_add 函数（mock httpx）."""

    @patch("httpx.AsyncClient")
    def test_add_success(self, mock_client_cls):
        """远程写入成功返回响应 dict."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": True,
            "skipped": False,
            "entry_id": "remote-001",
            "employee": "test",
            "category": "finding",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = _run(
            _remote_memory_add(
                "https://crew.knowlyr.com",
                "test-token",
                employee="test",
                category="finding",
                content="远程记忆",
            )
        )
        assert result["ok"] is True
        assert result["entry_id"] == "remote-001"

        # 验证请求参数
        call_args = mock_client.post.call_args
        assert "/api/memory/add" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["employee"] == "test"
        assert payload["content"] == "远程记忆"
        assert "Bearer test-token" in call_args[1]["headers"]["Authorization"]

    @patch("httpx.AsyncClient")
    def test_add_with_pattern_fields(self, mock_client_cls):
        """pattern 类型记忆传入额外字段."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "entry_id": "pat-001"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        _run(
            _remote_memory_add(
                "https://crew.knowlyr.com",
                "tok",
                employee="test",
                category="pattern",
                content="always test first",
                ttl_days=30,
                shared=True,
                trigger_condition="before coding",
                applicability=["backend"],
                origin_employee="senior-dev",
            )
        )
        payload = mock_client.post.call_args[1]["json"]
        assert payload["ttl_days"] == 30
        assert payload["shared"] is True
        assert payload["trigger_condition"] == "before coding"
        assert payload["applicability"] == ["backend"]
        assert payload["origin_employee"] == "senior-dev"

    @patch("httpx.AsyncClient")
    def test_add_http_error(self, mock_client_cls):
        """HTTP 错误时抛异常."""
        import httpx

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(httpx.HTTPStatusError):
            _run(
                _remote_memory_add(
                    "https://crew.knowlyr.com",
                    "tok",
                    employee="test",
                    category="finding",
                    content="fail",
                )
            )


class TestRemoteMemoryQuery:
    """测试 _remote_memory_query 函数（mock httpx）."""

    @patch("httpx.AsyncClient")
    def test_query_success(self, mock_client_cls):
        """远程查询成功返回条目列表."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": True,
            "entries": [
                {"id": "mem1", "employee": "test", "category": "finding", "content": "hello"},
            ],
            "total": 1,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = _run(
            _remote_memory_query(
                "https://crew.knowlyr.com",
                "test-token",
                employee="test",
            )
        )
        assert len(result) == 1
        assert result[0]["content"] == "hello"

        # 验证请求参数
        call_args = mock_client.get.call_args
        assert "/api/memory/query" in call_args[0][0]
        assert call_args[1]["params"]["employee"] == "test"

    @patch("httpx.AsyncClient")
    def test_query_with_category(self, mock_client_cls):
        """传入 category 参数."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "entries": [], "total": 0}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        _run(
            _remote_memory_query(
                "https://crew.knowlyr.com",
                "tok",
                employee="test",
                category="decision",
                limit=5,
            )
        )
        params = mock_client.get.call_args[1]["params"]
        assert params["category"] == "decision"
        assert params["limit"] == "5"

    @patch("httpx.AsyncClient")
    def test_query_no_category(self, mock_client_cls):
        """不传 category 时请求参数中无该字段."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "entries": [], "total": 0}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        _run(
            _remote_memory_query(
                "https://crew.knowlyr.com",
                "tok",
                employee="test",
            )
        )
        params = mock_client.get.call_args[1]["params"]
        assert "category" not in params


class TestMCPMemoryToolRemoteIntegration:
    """测试 MCP server 中 memory 工具走远程 API 路径."""

    def setup_method(self):
        self.server = create_server()

    def _call_tool(self, name, arguments, server=None):
        from mcp.types import CallToolRequest, CallToolRequestParams

        s = server or self.server
        handler = s.request_handlers[CallToolRequest]
        return _run(
            handler(
                CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name=name, arguments=arguments),
                )
            )
        )

    @patch("crew.mcp_server._get_remote_memory_config")
    @patch("crew.mcp_server._remote_memory_add", new_callable=AsyncMock)
    def test_add_memory_uses_remote(self, mock_remote_add, mock_get_cfg):
        """配置了远程 API 时 add_memory 走远程路径."""
        mock_get_cfg.return_value = ("https://crew.knowlyr.com", "token")
        mock_remote_add.return_value = {
            "ok": True,
            "entry_id": "remote-001",
            "employee": "test",
            "category": "finding",
        }

        result = self._call_tool(
            "add_memory",
            {"employee": "test", "category": "finding", "content": "远程写入"},
        )
        text = result.root.content[0].text
        data = json.loads(text)
        assert data["ok"] is True
        assert data["entry_id"] == "remote-001"
        mock_remote_add.assert_called_once()

    @patch("crew.mcp_server._get_remote_memory_config")
    @patch("crew.mcp_server._remote_memory_query", new_callable=AsyncMock)
    def test_query_memory_uses_remote(self, mock_remote_query, mock_get_cfg):
        """配置了远程 API 时 query_memory 走远程路径."""
        mock_get_cfg.return_value = ("https://crew.knowlyr.com", "token")
        mock_remote_query.return_value = [
            {"id": "r1", "employee": "test", "category": "finding", "content": "远程记忆"},
        ]

        result = self._call_tool("query_memory", {"employee": "test"})
        text = result.root.content[0].text
        data = json.loads(text)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["content"] == "远程记忆"
        mock_remote_query.assert_called_once()

    @patch("crew.mcp_server._get_remote_memory_config")
    def test_add_memory_fallback_local(self, mock_get_cfg, tmp_path):
        """未配置远程 API 时 add_memory 走本地路径."""
        mock_get_cfg.return_value = None

        server = create_server(project_dir=tmp_path)
        result = self._call_tool(
            "add_memory",
            {
                "employee": "test-local",
                "category": "finding",
                "content": "本地记忆",
            },
            server=server,
        )
        text = result.root.content[0].text
        data = json.loads(text)
        assert data["employee"] == "test-local"
        assert data["content"] == "本地记忆"
        # 确认本地文件已写入
        assert (tmp_path / ".crew" / "memory" / "test-local.jsonl").exists()

    @patch("crew.mcp_server._get_remote_memory_config")
    @patch("crew.mcp_server._remote_memory_add", new_callable=AsyncMock)
    def test_add_memory_remote_error(self, mock_remote_add, mock_get_cfg):
        """远程写入失败时返回错误信息，不静默失败."""
        mock_get_cfg.return_value = ("https://crew.knowlyr.com", "token")
        mock_remote_add.side_effect = Exception("Connection refused")

        result = self._call_tool(
            "add_memory",
            {"employee": "test", "category": "finding", "content": "fail"},
        )
        text = result.root.content[0].text
        data = json.loads(text)
        assert "error" in data
        assert "Connection refused" in data["error"]

    @patch("crew.mcp_server._get_remote_memory_config")
    @patch("crew.mcp_server._remote_memory_query", new_callable=AsyncMock)
    def test_query_memory_remote_error(self, mock_remote_query, mock_get_cfg):
        """远程查询失败时返回错误信息，不静默失败."""
        mock_get_cfg.return_value = ("https://crew.knowlyr.com", "token")
        mock_remote_query.side_effect = Exception("Timeout")

        result = self._call_tool("query_memory", {"employee": "test"})
        text = result.root.content[0].text
        data = json.loads(text)
        assert "error" in data
        assert "Timeout" in data["error"]
