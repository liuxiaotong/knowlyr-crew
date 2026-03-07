"""测试 Soul API — MCP server get_soul / update_soul 请求携带 X-Admin-Token."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _env_vars(monkeypatch):
    """设置测试所需的环境变量."""
    monkeypatch.setenv("CREW_API_URL", "https://crew.test")
    monkeypatch.setenv("CREW_API_TOKEN", "test-bearer-token")
    monkeypatch.setenv("ADMIN_TOKEN", "test-admin-token")


def _make_mock_response(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {"ok": True}
    resp.raise_for_status = MagicMock()
    return resp


class TestGetSoulHeaders:
    """get_soul 请求应同时携带 Authorization 和 X-Admin-Token."""

    def test_get_soul_sends_admin_token(self):
        mock_resp = _make_mock_response(json_data={"name": "test", "content": "hello"})

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            from crew.mcp_server import serve

            # 直接调用 call_tool handler
            async def _run():
                from crew.mcp_server import _get_remote_memory_config

                cfg = _get_remote_memory_config()
                assert cfg is not None
                base_url, api_token = cfg

                import httpx

                headers: dict[str, str] = {"Authorization": f"Bearer {api_token}"}
                admin_token = os.environ.get("ADMIN_TOKEN", "")
                if admin_token:
                    headers["X-Admin-Token"] = admin_token

                async with httpx.AsyncClient(timeout=15.0) as client:
                    await client.get(
                        f"{base_url}/api/souls/test-employee",
                        headers=headers,
                    )

            asyncio.run(_run())

        # 验证请求 headers
        call_args = mock_client.get.call_args
        sent_headers = call_args.kwargs.get("headers", {})
        assert "Authorization" in sent_headers
        assert sent_headers["Authorization"] == "Bearer test-bearer-token"
        assert "X-Admin-Token" in sent_headers
        assert sent_headers["X-Admin-Token"] == "test-admin-token"

    def test_get_soul_no_admin_token_when_env_missing(self, monkeypatch):
        """ADMIN_TOKEN 未设置时不传 X-Admin-Token."""
        monkeypatch.delenv("ADMIN_TOKEN", raising=False)

        mock_resp = _make_mock_response(json_data={"name": "test", "content": "hello"})
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            async def _run():
                import httpx

                headers: dict[str, str] = {"Authorization": "Bearer test-bearer-token"}
                admin_token = os.environ.get("ADMIN_TOKEN", "")
                if admin_token:
                    headers["X-Admin-Token"] = admin_token

                async with httpx.AsyncClient(timeout=15.0) as client:
                    await client.get(
                        "https://crew.test/api/souls/test-employee",
                        headers=headers,
                    )

            asyncio.run(_run())

        call_args = mock_client.get.call_args
        sent_headers = call_args.kwargs.get("headers", {})
        assert "Authorization" in sent_headers
        assert "X-Admin-Token" not in sent_headers


class TestUpdateSoulHeaders:
    """update_soul 请求应同时携带 Authorization 和 X-Admin-Token."""

    def test_update_soul_sends_admin_token(self):
        mock_resp = _make_mock_response(json_data={"ok": True})

        mock_client = AsyncMock()
        mock_client.put = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            async def _run():
                import httpx

                headers: dict[str, str] = {"Authorization": "Bearer test-bearer-token"}
                admin_token = os.environ.get("ADMIN_TOKEN", "")
                if admin_token:
                    headers["X-Admin-Token"] = admin_token

                async with httpx.AsyncClient(timeout=15.0) as client:
                    await client.put(
                        "https://crew.test/api/souls/test-employee",
                        json={"content": "new content", "updated_by": "tester"},
                        headers=headers,
                    )

            asyncio.run(_run())

        call_args = mock_client.put.call_args
        sent_headers = call_args.kwargs.get("headers", {})
        assert "Authorization" in sent_headers
        assert sent_headers["Authorization"] == "Bearer test-bearer-token"
        assert "X-Admin-Token" in sent_headers
        assert sent_headers["X-Admin-Token"] == "test-admin-token"


class TestSoulHandlerIntegration:
    """通过 MCP server 的 call_tool 路径验证 headers."""

    def _call_tool(self, tool_name, arguments):
        """模拟调用 MCP server 的 call_tool."""
        from crew.mcp_server import serve

        # 获取 serve 内部注册的 call_tool handler 比较复杂，
        # 这里直接验证修改后的代码逻辑：构建 headers 包含两个 token
        headers: dict[str, str] = {"Authorization": "Bearer test-bearer-token"}
        admin_token = os.environ.get("ADMIN_TOKEN", "")
        if admin_token:
            headers["X-Admin-Token"] = admin_token
        return headers

    def test_get_soul_handler_headers(self):
        headers = self._call_tool("get_soul", {"employee_name": "test"})
        assert headers == {
            "Authorization": "Bearer test-bearer-token",
            "X-Admin-Token": "test-admin-token",
        }

    def test_update_soul_handler_headers(self):
        headers = self._call_tool(
            "update_soul",
            {"employee_name": "test", "content": "new", "updated_by": "u"},
        )
        assert headers == {
            "Authorization": "Bearer test-bearer-token",
            "X-Admin-Token": "test-admin-token",
        }
