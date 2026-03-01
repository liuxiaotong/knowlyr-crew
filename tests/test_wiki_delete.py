"""测试 Wiki 文件删除 API."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("starlette")

from starlette.testclient import TestClient

from crew.webhook import create_webhook_app
from crew.webhook_config import WebhookConfig

TOKEN = "test-token-123"


def _make_client(token=TOKEN):
    """创建测试客户端."""
    app = create_webhook_app(
        project_dir=Path("/tmp/test"),
        token=token,
        config=WebhookConfig(),
    )
    return TestClient(app)


class TestWikiFileDelete:
    """Wiki 文件删除端点测试."""

    def test_delete_success(self):
        """DELETE 成功返回 200."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = lambda: None

        mock_client_instance = AsyncMock()
        mock_client_instance.delete = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.dict(
                "os.environ",
                {"WIKI_API_URL": "https://wiki.example.com", "WIKI_API_TOKEN": "wiki-token"},
            ),
            patch("httpx.AsyncClient", return_value=mock_client_instance),
        ):
            client = _make_client()
            resp = client.delete(
                "/api/wiki/files/42",
                headers={"Authorization": f"Bearer {TOKEN}"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["deleted_file_id"] == 42

    def test_delete_not_found(self):
        """DELETE 不存在的 file_id 返回 404."""
        mock_resp = AsyncMock()
        mock_resp.status_code = 404
        mock_resp.raise_for_status = AsyncMock()

        mock_client_instance = AsyncMock()
        mock_client_instance.delete = AsyncMock(return_value=mock_resp)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.dict(
                "os.environ",
                {"WIKI_API_URL": "https://wiki.example.com", "WIKI_API_TOKEN": "wiki-token"},
            ),
            patch("httpx.AsyncClient", return_value=mock_client_instance),
        ):
            client = _make_client()
            resp = client.delete(
                "/api/wiki/files/99999",
                headers={"Authorization": f"Bearer {TOKEN}"},
            )

        assert resp.status_code == 404
        data = resp.json()
        assert "not found" in data["error"]

    def test_delete_requires_auth(self):
        """DELETE 无 Bearer token 返回 401."""
        client = _make_client()
        resp = client.delete("/api/wiki/files/42")
        assert resp.status_code == 401

    def test_delete_wiki_not_configured(self):
        """Wiki 环境变量未配置时返回 500."""
        import os

        env_copy = os.environ.copy()
        env_copy.pop("WIKI_API_URL", None)
        env_copy.pop("WIKI_API_TOKEN", None)

        with patch.dict("os.environ", env_copy, clear=True):
            client = _make_client()
            resp = client.delete(
                "/api/wiki/files/42",
                headers={"Authorization": f"Bearer {TOKEN}"},
            )

        assert resp.status_code == 500
        assert "未配置" in resp.json()["error"]
