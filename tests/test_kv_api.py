"""测试 KV 存储 API — 服务端端点 + MCP server 远程调用."""

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

TOKEN = "test-kv-token"
ADMIN_TOKEN = "test-admin-token"


def _make_client(project_dir=None):
    """创建测试客户端."""
    import os

    os.environ["ADMIN_TOKEN"] = ADMIN_TOKEN
    app = create_webhook_app(
        project_dir=project_dir or Path("/tmp/test-kv"),
        token=TOKEN,
        config=WebhookConfig(),
    )
    return TestClient(app)


class TestKVPutEndpoint:
    """PUT /api/kv/{key} — KV 写入端点."""

    def test_put_requires_auth(self):
        """未认证时返回 401."""
        client = _make_client()
        resp = client.put(
            "/api/kv/config/test.md",
            content=b"hello",
            headers={"Content-Type": "text/plain"},
        )
        assert resp.status_code == 401

    def test_put_text_plain(self, tmp_path):
        """text/plain body 写入成功."""
        client = _make_client(project_dir=tmp_path)
        resp = client.put(
            "/api/kv/config/test.md",
            content=b"# Test Config\n\nHello world",
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "text/plain",
                "X-Admin-Token": ADMIN_TOKEN,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["key"] == "config/test.md"
        assert data["size"] > 0

        # 验证文件实际写入
        file_path = tmp_path / ".crew" / "kv" / "config" / "test.md"
        assert file_path.exists()
        assert file_path.read_text() == "# Test Config\n\nHello world"

    def test_put_json_body(self, tmp_path):
        """JSON body 写入成功."""
        client = _make_client(project_dir=tmp_path)
        resp = client.put(
            "/api/kv/config/global/CLAUDE.md",
            json={"content": "# Global Config"},
            headers={"Authorization": f"Bearer {TOKEN}", "X-Admin-Token": ADMIN_TOKEN},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["key"] == "config/global/CLAUDE.md"

        file_path = tmp_path / ".crew" / "kv" / "config" / "global" / "CLAUDE.md"
        assert file_path.exists()
        assert file_path.read_text() == "# Global Config"

    def test_put_overwrite(self, tmp_path):
        """覆盖写入."""
        client = _make_client(project_dir=tmp_path)
        headers = {
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "text/plain",
            "X-Admin-Token": ADMIN_TOKEN,
        }

        # 第一次写
        client.put("/api/kv/config/overwrite.md", content=b"v1", headers=headers)

        # 第二次覆盖
        resp = client.put("/api/kv/config/overwrite.md", content=b"v2", headers=headers)
        assert resp.status_code == 200

        file_path = tmp_path / ".crew" / "kv" / "config" / "overwrite.md"
        assert file_path.read_text() == "v2"

    def test_put_path_traversal_dotdot(self, tmp_path):
        """路径穿越检测 — '..' 拦截.

        注意：Starlette 路由会对 URL 中的 .. 做规范化处理，
        可能导致 404 而非 400。两种状态码都可接受。
        """
        client = _make_client(project_dir=tmp_path)
        resp = client.put(
            "/api/kv/config/../../../etc/passwd",
            content=b"hacked",
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "text/plain",
            },
        )
        assert resp.status_code in (400, 404)

    def test_put_path_traversal_in_key(self, tmp_path):
        """路径穿越检测 — 通过编码的 '..' 传入 handler."""
        client = _make_client(project_dir=tmp_path)
        # 直接在 key 中包含 ..（不经过 URL 规范化）
        resp = client.put(
            "/api/kv/config%2F..%2F..%2Fetc%2Fpasswd",
            content=b"hacked",
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "text/plain",
                "X-Admin-Token": ADMIN_TOKEN,
            },
        )
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_put_key_starting_with_slash(self, tmp_path):
        """key 以 '/' 开头被拒绝."""
        client = _make_client(project_dir=tmp_path)
        resp = client.put(
            "/api/kv//etc/passwd",
            content=b"bad",
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "text/plain",
                "X-Admin-Token": ADMIN_TOKEN,
            },
        )
        # Starlette 路由可能会 404 或者 handler 返回 400/403
        assert resp.status_code in (400, 403, 404)

    def test_put_empty_body(self, tmp_path):
        """空 body 返回 400."""
        client = _make_client(project_dir=tmp_path)
        resp = client.put(
            "/api/kv/config/empty.md",
            content=b"",
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "text/plain",
                "X-Admin-Token": ADMIN_TOKEN,
            },
        )
        assert resp.status_code == 400
        assert "empty" in resp.json().get("error", "")

    def test_put_json_missing_content(self, tmp_path):
        """JSON body 缺少 content 字段返回 400."""
        client = _make_client(project_dir=tmp_path)
        resp = client.put(
            "/api/kv/config/test.md",
            json={"wrong_field": "hello"},
            headers={"Authorization": f"Bearer {TOKEN}", "X-Admin-Token": ADMIN_TOKEN},
        )
        assert resp.status_code == 400


class TestKVGetEndpoint:
    """GET /api/kv/{key} — KV 读取端点."""

    def test_get_requires_auth(self):
        """未认证时返回 401."""
        client = _make_client()
        resp = client.get("/api/kv/config/test.md")
        assert resp.status_code == 401

    def test_get_success(self, tmp_path):
        """读取成功."""
        # 先写入文件
        kv_dir = tmp_path / ".crew" / "kv" / "config"
        kv_dir.mkdir(parents=True)
        (kv_dir / "test.md").write_text("# Hello", encoding="utf-8")

        client = _make_client(project_dir=tmp_path)
        resp = client.get(
            "/api/kv/config/test.md",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        assert resp.text == "# Hello"
        assert "text/plain" in resp.headers["content-type"]

    def test_get_not_found(self, tmp_path):
        """不存在的 key 返回 404."""
        client = _make_client(project_dir=tmp_path)
        resp = client.get(
            "/api/kv/config/nonexistent.md",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 404
        data = resp.json()
        assert data["ok"] is False
        assert data["error"] == "not found"

    def test_get_path_traversal(self, tmp_path):
        """路径穿越检测 — Starlette 可能规范化 URL 导致 404."""
        client = _make_client(project_dir=tmp_path)
        resp = client.get(
            "/api/kv/config/../../secret",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code in (400, 404)

    def test_get_path_traversal_encoded(self, tmp_path):
        """路径穿越检测 — 编码形式直达 handler."""
        client = _make_client(project_dir=tmp_path)
        resp = client.get(
            "/api/kv/config%2F..%2F..%2Fsecret",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 400

    def test_put_then_get_roundtrip(self, tmp_path):
        """写入后读取，内容一致."""
        client = _make_client(project_dir=tmp_path)
        headers = {"Authorization": f"Bearer {TOKEN}", "X-Admin-Token": ADMIN_TOKEN}
        content = "# CLAUDE.md\n\n这是中文内容 🎉\nLine 2"

        # PUT
        resp = client.put(
            "/api/kv/config/roundtrip.md",
            json={"content": content},
            headers=headers,
        )
        assert resp.status_code == 200

        # GET
        resp = client.get(
            "/api/kv/config/roundtrip.md",
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.text == content


class TestKVListEndpoint:
    """GET /api/kv/ — KV 列表端点."""

    def test_list_requires_auth(self):
        """未认证时返回 401."""
        client = _make_client()
        resp = client.get("/api/kv/")
        assert resp.status_code == 401

    def test_list_empty(self, tmp_path):
        """空目录返回空列表."""
        client = _make_client(project_dir=tmp_path)
        resp = client.get(
            "/api/kv/",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["keys"] == []

    def test_list_all(self, tmp_path):
        """列出所有 key."""
        kv_dir = tmp_path / ".crew" / "kv"
        (kv_dir / "config").mkdir(parents=True, exist_ok=True)
        (kv_dir / "config" / "a.md").write_text("aaa", encoding="utf-8")
        (kv_dir / "config" / "b.md").write_text("bbb", encoding="utf-8")
        (kv_dir / "other").mkdir(parents=True, exist_ok=True)
        (kv_dir / "other" / "c.md").write_text("ccc", encoding="utf-8")

        client = _make_client(project_dir=tmp_path)
        resp = client.get(
            "/api/kv/",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert len(data["keys"]) == 3
        assert "config/a.md" in data["keys"]
        assert "config/b.md" in data["keys"]
        assert "other/c.md" in data["keys"]

    def test_list_with_prefix(self, tmp_path):
        """按前缀过滤."""
        kv_dir = tmp_path / ".crew" / "kv"
        (kv_dir / "config").mkdir(parents=True)
        (kv_dir / "config" / "x.md").write_text("x", encoding="utf-8")
        (kv_dir / "other").mkdir(parents=True)
        (kv_dir / "other" / "y.md").write_text("y", encoding="utf-8")

        client = _make_client(project_dir=tmp_path)
        resp = client.get(
            "/api/kv/?prefix=config/",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["keys"] == ["config/x.md"]

    def test_list_prefix_traversal(self, tmp_path):
        """prefix 路径穿越拦截."""
        client = _make_client(project_dir=tmp_path)
        resp = client.get(
            "/api/kv/?prefix=../secret",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 400


# ── MCP Server KV 工具测试 ──────────────────────────────────────

mcp_mod = pytest.importorskip("mcp")

from crew.mcp_server import (
    _remote_kv_get,
    _remote_kv_list,
    _remote_kv_put,
    create_server,
)


def _run(coro):
    """同步运行 async 函数."""
    return asyncio.run(coro)


class TestRemoteKVPut:
    """测试 _remote_kv_put 函数（mock httpx）."""

    @patch("httpx.AsyncClient")
    def test_put_success(self, mock_client_cls):
        """远程 KV 写入成功."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "key": "config/test.md", "size": 12}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.put = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = _run(
            _remote_kv_put(
                "https://crew.knowlyr.com",
                "test-token",
                key="config/test.md",
                content="# Hello",
            )
        )
        assert result["ok"] is True
        assert result["key"] == "config/test.md"

        # 验证请求参数
        call_args = mock_client.put.call_args
        assert "/api/kv/config/test.md" in call_args[0][0]
        assert "Bearer test-token" in call_args[1]["headers"]["Authorization"]


class TestRemoteKVGet:
    """测试 _remote_kv_get 函数（mock httpx）."""

    @patch("httpx.AsyncClient")
    def test_get_success(self, mock_client_cls):
        """远程 KV 读取成功."""
        mock_response = MagicMock()
        mock_response.text = "# Config Content"
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = _run(
            _remote_kv_get(
                "https://crew.knowlyr.com",
                "test-token",
                key="config/test.md",
            )
        )
        assert result == "# Config Content"

    @patch("httpx.AsyncClient")
    def test_get_404(self, mock_client_cls):
        """远程 KV 读取 404."""
        import httpx

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock()
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with pytest.raises(httpx.HTTPStatusError):
            _run(
                _remote_kv_get(
                    "https://crew.knowlyr.com",
                    "tok",
                    key="nonexistent.md",
                )
            )


class TestRemoteKVList:
    """测试 _remote_kv_list 函数（mock httpx）."""

    @patch("httpx.AsyncClient")
    def test_list_success(self, mock_client_cls):
        """远程 KV 列表成功."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": True,
            "keys": ["config/a.md", "config/b.md"],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = _run(
            _remote_kv_list(
                "https://crew.knowlyr.com",
                "test-token",
                prefix="config/",
            )
        )
        assert len(result) == 2
        assert "config/a.md" in result

        # 验证请求参数
        call_args = mock_client.get.call_args
        assert "/api/kv/" in call_args[0][0]
        assert call_args[1]["params"]["prefix"] == "config/"


class TestMCPKVToolRemoteIntegration:
    """测试 MCP server 中 KV 工具走远程 API 路径."""

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
    @patch("crew.mcp_server._remote_kv_put", new_callable=AsyncMock)
    def test_put_config_uses_remote(self, mock_remote_put, mock_get_cfg):
        """配置了远程 API 时 put_config 走远程路径."""
        mock_get_cfg.return_value = ("https://crew.knowlyr.com", "token")
        mock_remote_put.return_value = {
            "ok": True,
            "key": "config/test.md",
            "size": 10,
        }

        result = self._call_tool(
            "put_config",
            {"key": "config/test.md", "content": "# Hello"},
        )
        text = result.root.content[0].text
        data = json.loads(text)
        assert data["ok"] is True
        assert data["key"] == "config/test.md"
        mock_remote_put.assert_called_once()

    @patch("crew.mcp_server._get_remote_memory_config")
    @patch("crew.mcp_server._remote_kv_get", new_callable=AsyncMock)
    def test_get_config_uses_remote(self, mock_remote_get, mock_get_cfg):
        """配置了远程 API 时 get_config 走远程路径."""
        mock_get_cfg.return_value = ("https://crew.knowlyr.com", "token")
        mock_remote_get.return_value = "# Config Content"

        result = self._call_tool("get_config", {"key": "config/test.md"})
        text = result.root.content[0].text
        assert text == "# Config Content"
        mock_remote_get.assert_called_once()

    @patch("crew.mcp_server._get_remote_memory_config")
    @patch("crew.mcp_server._remote_kv_list", new_callable=AsyncMock)
    def test_list_configs_uses_remote(self, mock_remote_list, mock_get_cfg):
        """配置了远程 API 时 list_configs 走远程路径."""
        mock_get_cfg.return_value = ("https://crew.knowlyr.com", "token")
        mock_remote_list.return_value = ["config/a.md", "config/b.md"]

        result = self._call_tool("list_configs", {"prefix": "config/"})
        text = result.root.content[0].text
        data = json.loads(text)
        assert data["ok"] is True
        assert len(data["keys"]) == 2
        mock_remote_list.assert_called_once()

    @patch("crew.mcp_server._get_remote_memory_config")
    def test_put_config_fallback_local(self, mock_get_cfg, tmp_path):
        """未配置远程 API 时 put_config 走本地路径."""
        mock_get_cfg.return_value = None

        server = create_server(project_dir=tmp_path)
        result = self._call_tool(
            "put_config",
            {"key": "config/local.md", "content": "# Local"},
            server=server,
        )
        text = result.root.content[0].text
        data = json.loads(text)
        assert data["ok"] is True
        assert data["key"] == "config/local.md"

        # 确认本地文件已写入
        assert (tmp_path / ".crew" / "kv" / "config" / "local.md").exists()

    @patch("crew.mcp_server._get_remote_memory_config")
    def test_get_config_fallback_local(self, mock_get_cfg, tmp_path):
        """未配置远程 API 时 get_config 走本地路径."""
        mock_get_cfg.return_value = None

        # 先写入文件
        kv_dir = tmp_path / ".crew" / "kv" / "config"
        kv_dir.mkdir(parents=True)
        (kv_dir / "local.md").write_text("# Local Content", encoding="utf-8")

        server = create_server(project_dir=tmp_path)
        result = self._call_tool(
            "get_config",
            {"key": "config/local.md"},
            server=server,
        )
        text = result.root.content[0].text
        assert text == "# Local Content"

    @patch("crew.mcp_server._get_remote_memory_config")
    def test_get_config_local_not_found(self, mock_get_cfg, tmp_path):
        """本地 fallback key 不存在时返回错误."""
        mock_get_cfg.return_value = None

        server = create_server(project_dir=tmp_path)
        result = self._call_tool(
            "get_config",
            {"key": "config/nonexistent.md"},
            server=server,
        )
        text = result.root.content[0].text
        data = json.loads(text)
        assert "error" in data
        assert "not found" in data["error"]

    @patch("crew.mcp_server._get_remote_memory_config")
    def test_list_configs_fallback_local(self, mock_get_cfg, tmp_path):
        """未配置远程 API 时 list_configs 走本地路径."""
        mock_get_cfg.return_value = None

        kv_dir = tmp_path / ".crew" / "kv" / "config"
        kv_dir.mkdir(parents=True)
        (kv_dir / "a.md").write_text("a", encoding="utf-8")
        (kv_dir / "b.md").write_text("b", encoding="utf-8")

        server = create_server(project_dir=tmp_path)
        result = self._call_tool(
            "list_configs",
            {"prefix": "config/"},
            server=server,
        )
        text = result.root.content[0].text
        data = json.loads(text)
        assert data["ok"] is True
        assert len(data["keys"]) == 2

    @patch("crew.mcp_server._get_remote_memory_config")
    @patch("crew.mcp_server._remote_kv_put", new_callable=AsyncMock)
    def test_put_config_remote_error(self, mock_remote_put, mock_get_cfg):
        """远程写入失败时返回错误信息."""
        mock_get_cfg.return_value = ("https://crew.knowlyr.com", "token")
        mock_remote_put.side_effect = Exception("Connection refused")

        result = self._call_tool(
            "put_config",
            {"key": "config/test.md", "content": "fail"},
        )
        text = result.root.content[0].text
        data = json.loads(text)
        assert "error" in data
        assert "Connection refused" in data["error"]

    @patch("crew.mcp_server._get_remote_memory_config")
    @patch("crew.mcp_server._remote_kv_get", new_callable=AsyncMock)
    def test_get_config_remote_error(self, mock_remote_get, mock_get_cfg):
        """远程读取失败时返回错误信息."""
        mock_get_cfg.return_value = ("https://crew.knowlyr.com", "token")
        mock_remote_get.side_effect = Exception("Timeout")

        result = self._call_tool("get_config", {"key": "config/test.md"})
        text = result.root.content[0].text
        data = json.loads(text)
        assert "error" in data
        assert "Timeout" in data["error"]
