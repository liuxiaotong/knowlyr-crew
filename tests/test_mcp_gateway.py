"""MCP Gateway 模块测试."""

from __future__ import annotations

import asyncio
import os

import pytest

# ── config tests ──


class TestMCPConfig:
    """YAML 配置加载测试."""

    def test_load_missing_file(self, tmp_path):
        from crew.mcp_gateway.config import load_mcp_servers_config

        result = load_mcp_servers_config(tmp_path)
        assert result == {}

    def test_load_valid_config(self, tmp_path):
        from crew.mcp_gateway.config import load_mcp_servers_config

        crew_dir = tmp_path / ".crew"
        crew_dir.mkdir()
        config_file = crew_dir / "mcp_servers.yaml"
        config_file.write_text(
            """
servers:
  gws:
    command: gws
    args: ["mcp"]
    env:
      FOO: bar
    whitelist:
      - calendar_list_events
    timeout: 15
    description: Google Workspace
  slack:
    command: slack-mcp
    args: []
"""
        )

        result = load_mcp_servers_config(tmp_path)
        assert "gws" in result
        assert "slack" in result
        assert result["gws"].command == "gws"
        assert result["gws"].args == ["mcp"]
        assert result["gws"].whitelist == ["calendar_list_events"]
        assert result["gws"].timeout == 15
        assert result["slack"].timeout == 30  # default

    def test_env_var_resolution(self):
        from crew.mcp_gateway.config import MCPServerConfig

        os.environ["TEST_MCP_CMD"] = "/usr/bin/test-mcp"
        cfg = MCPServerConfig(
            name="test",
            command="${TEST_MCP_CMD}",
            args=["--flag"],
            env={"KEY": "${TEST_MCP_CMD}"},
        )
        assert cfg.resolved_command() == "/usr/bin/test-mcp"
        assert cfg.resolved_env()["KEY"] == "/usr/bin/test-mcp"
        del os.environ["TEST_MCP_CMD"]


# ── registry tests ──


class TestMCPRegistry:
    """工具注册表测试."""

    def test_namespace_format(self):
        from crew.mcp_gateway.registry import make_namespaced_name, parse_namespaced_name

        ns = make_namespaced_name("gws", "calendar_list_events")
        assert ns == "mcp__gws__calendar_list_events"

        parsed = parse_namespaced_name(ns)
        assert parsed == ("gws", "calendar_list_events")

    def test_parse_invalid(self):
        from crew.mcp_gateway.registry import parse_namespaced_name

        assert parse_namespaced_name("bash") is None
        assert parse_namespaced_name("mcp__only_two") is None
        assert parse_namespaced_name("wrong__gws__tool") is None

    def test_parse_with_underscores_in_tool_name(self):
        """W4: tool 名包含下划线时正确解析."""
        from crew.mcp_gateway.registry import parse_namespaced_name

        # tool 名中含单下划线应正常工作
        result = parse_namespaced_name("mcp__gws__calendar_list_events")
        assert result == ("gws", "calendar_list_events")

    def test_validate_names(self):
        from crew.mcp_gateway.registry import validate_server_name, validate_tool_name

        assert validate_server_name("gws")
        assert validate_server_name("google-workspace")
        assert not validate_server_name("")
        assert not validate_server_name("GWS")  # 大写不合法
        assert not validate_server_name("a" * 50)  # 太长
        assert not validate_server_name("-bad")  # 不能以 - 开头

        assert validate_tool_name("calendar_list_events")
        assert validate_tool_name("get-event")
        assert not validate_tool_name("")
        assert not validate_tool_name("UPPER")
        assert not validate_tool_name("has__double")  # W1: __ 拦截

    def test_register_and_inject(self):
        from crew.mcp_gateway.registry import MCPToolRegistry

        registry = MCPToolRegistry()

        async def mock_handler(args, *, agent_id=None, ctx=None):
            return "ok"

        schema = {
            "name": "calendar_list",
            "description": "List calendar events",
            "input_schema": {"type": "object", "properties": {}},
        }

        ns_name = registry.register_tool(
            server_name="gws",
            tool_name="calendar_list",
            schema=schema,
            handler=mock_handler,
        )
        assert ns_name == "mcp__gws__calendar_list"
        assert registry.tool_count == 1

        # 注入
        schemas = {}
        handlers = {}
        count = registry.inject_into(schemas, handlers)
        assert count == 1
        assert "mcp__gws__calendar_list" in schemas
        assert "mcp__gws__calendar_list" in handlers
        # schema name 被修正
        assert schemas["mcp__gws__calendar_list"]["name"] == "mcp__gws__calendar_list"

    def test_collision_detection(self):
        from crew.mcp_gateway.registry import MCPToolRegistry

        registry = MCPToolRegistry()

        async def mock_handler(args, *, agent_id=None, ctx=None):
            return "ok"

        existing = {"mcp__gws__calendar_list"}

        ns_name = registry.register_tool(
            server_name="gws",
            tool_name="calendar_list",
            schema={"name": "calendar_list"},
            handler=mock_handler,
            existing_tools=existing,
        )
        assert ns_name is None  # 碰撞拒绝

    def test_unregister_server(self):
        from crew.mcp_gateway.registry import MCPToolRegistry

        registry = MCPToolRegistry()

        async def h(args, *, agent_id=None, ctx=None):
            return ""

        registry.register_tool("gws", "tool-a", {"name": "tool-a"}, h)
        registry.register_tool("gws", "tool-b", {"name": "tool-b"}, h)
        registry.register_tool("slack", "tool-c", {"name": "tool-c"}, h)

        assert registry.tool_count == 3
        removed = registry.unregister_server("gws")
        assert len(removed) == 2
        assert registry.tool_count == 1


# ── credentials tests ──


class TestCredentials:
    """凭据加密/解密测试."""

    def test_encrypt_decrypt_roundtrip(self):
        from crew.mcp_gateway.credentials import _decrypt, _encrypt

        original = "my-secret-oauth-token-12345"
        encrypted = _encrypt(original)
        assert encrypted != original
        decrypted = _decrypt(encrypted)
        assert decrypted == original

    def test_encrypt_different_each_time(self):
        from crew.mcp_gateway.credentials import _encrypt

        e1 = _encrypt("same-token")
        e2 = _encrypt("same-token")
        # 由于随机 IV，每次加密结果不同
        assert e1 != e2

    def test_decrypt_tampered_fails(self):
        import base64

        from crew.mcp_gateway.credentials import _decrypt, _encrypt

        encrypted = _encrypt("secret")
        raw = base64.b64decode(encrypted)
        # 篡改一个字节
        tampered = bytes([raw[0] ^ 0xFF]) + raw[1:]
        with pytest.raises(Exception):
            _decrypt(base64.b64encode(tampered).decode())

    def test_store_and_get_credential(self, tmp_path):
        """端到端测试：直接用 sqlite3 验证 SQL 逻辑."""
        import sqlite3

        from crew.mcp_gateway.credentials import _decrypt, _encrypt

        db_path = tmp_path / "test_cred.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("""\
CREATE TABLE IF NOT EXISTS user_mcp_credentials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    server_name TEXT NOT NULL,
    credential_type TEXT NOT NULL DEFAULT 'oauth_token',
    encrypted_value TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, server_name)
)""")
        conn.commit()

        # 存储
        token = "oauth-token-abc"
        encrypted = _encrypt(token)
        conn.execute(
            "INSERT INTO user_mcp_credentials (user_id, server_name, encrypted_value) VALUES (?, ?, ?)",
            ("user-123", "gws", encrypted),
        )
        conn.commit()

        # 读取
        row = conn.execute(
            "SELECT encrypted_value FROM user_mcp_credentials WHERE user_id = ? AND server_name = ?",
            ("user-123", "gws"),
        ).fetchone()
        assert row is not None
        decrypted = _decrypt(row["encrypted_value"])
        assert decrypted == token

        # 不存在的凭据
        row2 = conn.execute(
            "SELECT encrypted_value FROM user_mcp_credentials WHERE user_id = ? AND server_name = ?",
            ("user-999", "gws"),
        ).fetchone()
        assert row2 is None

        conn.close()


# ── audit tests ──


class TestAudit:
    """审计日志测试."""

    def test_sanitize_error(self):
        from crew.mcp_gateway.audit import sanitize_error

        msg = "Error: token=sk-12345abc password='secret123'"
        sanitized = sanitize_error(msg)
        assert "sk-12345abc" not in sanitized
        assert "secret123" not in sanitized

    def test_sanitize_args(self):
        from crew.mcp_gateway.audit import sanitize_args

        args = {
            "query": "normal value",
            "token": "secret-token",
            "api_key": "key-123",
            "long_text": "x" * 600,
        }
        sanitized = sanitize_args(args)
        assert sanitized["query"] == "normal value"
        assert sanitized["token"] == "***REDACTED***"
        assert sanitized["api_key"] == "***REDACTED***"
        assert "truncated" in sanitized["long_text"]

    def test_sanitize_args_nested(self):
        """S3: 递归脱敏嵌套 dict/list 中的敏感字段."""
        from crew.mcp_gateway.audit import sanitize_args

        args = {
            "config": {
                "token": "nested-secret",
                "url": "https://example.com",
                "nested": {"password": "deep-secret", "ok": "fine"},
            },
            "items": [{"api_key": "list-secret", "name": "test"}],
        }
        sanitized = sanitize_args(args)
        assert sanitized["config"]["token"] == "***REDACTED***"
        assert sanitized["config"]["url"] == "https://example.com"
        assert sanitized["config"]["nested"]["password"] == "***REDACTED***"
        assert sanitized["config"]["nested"]["ok"] == "fine"
        assert sanitized["items"][0]["api_key"] == "***REDACTED***"
        assert sanitized["items"][0]["name"] == "test"


# ── circuit breaker tests ──


class TestCircuitBreaker:
    """熔断器测试."""

    def test_initial_closed(self):
        from crew.mcp_gateway.manager import CircuitBreaker

        cb = CircuitBreaker()
        assert cb.check_state() == "CLOSED"
        assert not cb.is_open()

    def test_opens_after_threshold(self):
        from crew.mcp_gateway.manager import CircuitBreaker

        cb = CircuitBreaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.check_state() == "CLOSED"
        cb.record_failure()
        assert cb.check_state() == "OPEN"
        assert cb.is_open()

    def test_success_resets(self):
        from crew.mcp_gateway.manager import CircuitBreaker

        cb = CircuitBreaker(threshold=2)
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert cb.check_state() == "CLOSED"  # reset 后只有 1 次失败

    def test_half_open_after_recovery(self):
        import time

        from crew.mcp_gateway.manager import CircuitBreaker

        cb = CircuitBreaker(threshold=1, recovery_timeout=0.1)
        cb.record_failure()
        assert cb.check_state() == "OPEN"
        time.sleep(0.15)
        assert cb.check_state() == "HALF_OPEN"


# ── manager tests ──


class TestMCPGatewayManager:
    """网关管理器测试."""

    def test_start_no_config(self, tmp_path):
        from crew.mcp_gateway.manager import MCPGatewayManager

        mgr = MCPGatewayManager(project_dir=tmp_path)
        loop = asyncio.new_event_loop()
        try:
            count = loop.run_until_complete(mgr.start())
        finally:
            loop.close()
        assert count == 0

    def test_get_status(self, tmp_path):
        from crew.mcp_gateway.manager import MCPGatewayManager

        mgr = MCPGatewayManager(project_dir=tmp_path)
        status = mgr.get_status()
        assert status["started"] is False
        assert status["total_tools"] == 0
