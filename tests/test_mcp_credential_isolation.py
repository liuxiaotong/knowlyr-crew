"""MCP credential isolation tests."""
from __future__ import annotations
import asyncio
import sqlite3
from unittest.mock import AsyncMock, patch
import pytest


class TestCredentialsCRUD:
    def _setup_db(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE IF NOT EXISTS user_mcp_credentials ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "user_id TEXT NOT NULL, "
            "mcp_server TEXT NOT NULL, "
            "access_token TEXT NOT NULL DEFAULT '', "
            "refresh_token TEXT NOT NULL DEFAULT '', "
            "token_expires_at TEXT, "
            "scopes TEXT NOT NULL DEFAULT '', "
            "tenant_id TEXT NOT NULL DEFAULT '', "
            "created_at TEXT DEFAULT CURRENT_TIMESTAMP, "
            "updated_at TEXT DEFAULT CURRENT_TIMESTAMP, "
            "UNIQUE(user_id, mcp_server))"
        )
        conn.commit()
        return conn

    def test_roundtrip(self, tmp_path):
        from crew.mcp_gateway.credentials import _decrypt, _encrypt
        conn = self._setup_db(tmp_path)
        t = "ya29.test"
        conn.execute(
            "INSERT INTO user_mcp_credentials (user_id, mcp_server, access_token, tenant_id) VALUES (?, ?, ?, ?)",
            ("u1", "gws", _encrypt(t), "admin"),
        )
        conn.commit()
        row = conn.execute(
            "SELECT access_token, tenant_id FROM user_mcp_credentials WHERE user_id=? AND mcp_server=?",
            ("u1", "gws"),
        ).fetchone()
        assert _decrypt(row["access_token"]) == t
        assert row["tenant_id"] == "admin"
        conn.close()

    def test_tenant_isolation(self, tmp_path):
        from crew.mcp_gateway.credentials import _encrypt
        conn = self._setup_db(tmp_path)
        conn.execute(
            "INSERT INTO user_mcp_credentials (user_id, mcp_server, access_token, tenant_id) VALUES (?, ?, ?, ?)",
            ("u1", "gws", _encrypt("t1"), "admin"),
        )
        conn.execute(
            "INSERT INTO user_mcp_credentials (user_id, mcp_server, access_token, tenant_id) VALUES (?, ?, ?, ?)",
            ("u2", "gws", _encrypt("t2"), "ent"),
        )
        conn.commit()
        rows = conn.execute("SELECT tenant_id FROM user_mcp_credentials ORDER BY user_id").fetchall()
        assert rows[0]["tenant_id"] == "admin"
        assert rows[1]["tenant_id"] == "ent"
        conn.close()


class TestTenantFiltering:
    def test_empty_allows_all(self):
        from crew.mcp_gateway.config import MCPServerConfig
        cfg = MCPServerConfig(name="gws", command="gws", allowed_tenants=[])
        assert cfg.is_tenant_allowed("admin")
        assert cfg.is_tenant_allowed(None)

    def test_restricted(self):
        from crew.mcp_gateway.config import MCPServerConfig
        cfg = MCPServerConfig(name="gws", command="gws", allowed_tenants=["admin"])
        assert cfg.is_tenant_allowed("admin")
        assert not cfg.is_tenant_allowed("other")
        assert not cfg.is_tenant_allowed(None)

    def test_yaml_loads(self, tmp_path):
        from crew.mcp_gateway.config import load_mcp_servers_config
        d = tmp_path / ".crew"
        d.mkdir()
        yaml_text = "servers:\n  gws:\n    command: gws\n    allowed_tenants: [\"admin\"]\n  s2:\n    command: s2\n"
        (d / "mcp_servers.yaml").write_text(yaml_text)
        r = load_mcp_servers_config(tmp_path)
        assert r["gws"].allowed_tenants == ["admin"]
        assert r["s2"].allowed_tenants == []


class TestHandlerIntegration:
    def _make(self, tenants=None):
        from crew.mcp_gateway.config import MCPServerConfig
        from crew.mcp_gateway.manager import MCPGatewayManager
        cfg = MCPServerConfig(name="gws", command="gws", allowed_tenants=tenants or [])
        mc = AsyncMock()
        mc.call_tool = AsyncMock(return_value="ok")
        mgr = MCPGatewayManager()
        h = mgr._make_tool_handler("gws", "cal", mc, cfg)
        return h, mc

    def test_tenant_rejected(self):
        h, _ = self._make(["admin"])
        loop = asyncio.new_event_loop()
        try:
            with patch("crew.mcp_gateway.manager.log_tool_call"):
                r = loop.run_until_complete(h({}, agent_id="a", tenant_id="bad", user_id="u"))
        finally:
            loop.close()
        assert "权限拒绝" in r

    def test_tenant_ok(self):
        h, mc = self._make(["admin"])
        loop = asyncio.new_event_loop()
        try:
            with patch("crew.mcp_gateway.manager.log_tool_call"):
                with patch("crew.mcp_gateway.credentials.has_credential", return_value=True):
                    r = loop.run_until_complete(h({}, agent_id="a", tenant_id="admin", user_id="u"))
        finally:
            loop.close()
        assert r == "ok"

    def test_no_cred(self):
        h, _ = self._make([])
        loop = asyncio.new_event_loop()
        try:
            with patch("crew.mcp_gateway.manager.log_tool_call"):
                with patch("crew.mcp_gateway.credentials.has_credential", return_value=False):
                    r = loop.run_until_complete(h({}, agent_id="a", tenant_id="t", user_id="u"))
        finally:
            loop.close()
        assert "凭据缺失" in r

    def test_no_user_skip_cred(self):
        h, mc = self._make([])
        loop = asyncio.new_event_loop()
        try:
            with patch("crew.mcp_gateway.manager.log_tool_call"):
                r = loop.run_until_complete(h({}, agent_id="a", tenant_id="t"))
        finally:
            loop.close()
        assert r == "ok"

    def test_no_user_restricted_tenant_rejected(self):
        h, _ = self._make(["admin"])
        loop = asyncio.new_event_loop()
        try:
            with patch("crew.mcp_gateway.manager.log_tool_call"):
                r = loop.run_until_complete(h({}, agent_id="a", tenant_id="admin"))
        finally:
            loop.close()
        assert "凭据缺失" in r
        assert "无法识别用户身份" in r

    def test_audit_fields(self):
        h, _ = self._make([])
        loop = asyncio.new_event_loop()
        try:
            with patch("crew.mcp_gateway.manager.log_tool_call") as ml:
                with patch("crew.mcp_gateway.credentials.has_credential", return_value=True):
                    loop.run_until_complete(h({}, agent_id="a", tenant_id="admin", user_id="u42"))
        finally:
            loop.close()
        ml.assert_called_once()
        kw = ml.call_args[1]
        assert kw["tenant_id"] == "admin"
        assert kw["user_id"] == "u42"


class TestAuditSig:
    def test_has_tenant_id(self):
        import inspect
        from crew.mcp_gateway.audit import log_tool_call
        sig = inspect.signature(log_tool_call)
        assert "tenant_id" in sig.parameters


class TestConfigDefaults:
    def test_default_empty(self):
        from crew.mcp_gateway.config import MCPServerConfig
        cfg = MCPServerConfig(name="t", command="t")
        assert cfg.allowed_tenants == []
