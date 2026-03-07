"""MCP Gateway Manager — 管理外部 MCP Server 连接和工具注册.

核心职责:
1. 连接 MCP Server（通过 stdio transport）
2. 发现工具（list_tools）并注册到 registry
3. 代理工具调用（call_tool），带超时 + 熔断 + 审计
4. 故障隔离和重连管理

安全基线:
- 工具白名单过滤
- 30s 超时保护
- 熔断器（连续失败 3 次后断路 60s）
- 异常信息脱敏
- 重连竞态保护（asyncio.Lock）
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

from crew.mcp_gateway.audit import log_tool_call, sanitize_error
from crew.mcp_gateway.config import MCPServerConfig, load_mcp_servers_config
from crew.mcp_gateway.registry import (
    MCPToolRegistry,
    make_namespaced_name,
)

# ── 熔断器 ──


class CircuitBreaker:
    """简单的熔断器实现.

    连续失败 ``threshold`` 次后进入 OPEN 状态，
    ``recovery_timeout`` 秒后自动进入 HALF_OPEN 状态（允许一次试探）。
    """

    def __init__(self, threshold: int = 3, recovery_timeout: float = 60.0):
        self.threshold = threshold
        self.recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._state = "CLOSED"  # CLOSED / OPEN / HALF_OPEN

    def check_state(self) -> str:
        """检查并更新熔断器状态（可能从 OPEN 转为 HALF_OPEN）."""
        if self._state == "OPEN":
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = "HALF_OPEN"
        return self._state

    @property
    def state(self) -> str:
        """返回当前状态（无副作用，仅读取）."""
        return self._state

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = "CLOSED"

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.threshold:
            self._state = "OPEN"
            logger.warning("熔断器触发 OPEN: 连续失败 %d 次", self._failure_count)

    def is_open(self) -> bool:
        """检查熔断器是否处于 OPEN 状态（会触发状态转换检查）."""
        return self.check_state() == "OPEN"


# ── MCP Server 连接 ──


class MCPServerConnection:
    """单个 MCP Server 的连接管理.

    封装 MCP SDK 的 stdio client，提供:
    - 连接/断开
    - list_tools
    - call_tool（带超时）
    - 重连竞态保护
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.name = config.name
        self._session: Any = None  # mcp.ClientSession
        self._read_stream: Any = None
        self._write_stream: Any = None
        self._process_ctx: Any = None  # context manager from stdio_client
        self._connected = False
        self._needs_reconnect = False
        self._connect_lock = asyncio.Lock()
        self.circuit_breaker = CircuitBreaker()

    @property
    def is_connected(self) -> bool:
        return self._connected and self._session is not None

    async def connect(self) -> bool:
        """连接到 MCP Server（带竞态保护）."""
        async with self._connect_lock:
            if self._needs_reconnect:
                self._connected = False
                self._needs_reconnect = False
                await self._cleanup()
            if self._connected:
                return True
            return await self._do_connect()

    async def _do_connect(self) -> bool:
        """实际连接逻辑."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_params = StdioServerParameters(
                command=self.config.resolved_command(),
                args=self.config.resolved_args(),
                env={**self.config.resolved_env()},
            )

            # stdio_client 是一个 async context manager
            # 我们需要手动管理它的生命周期
            self._process_ctx = stdio_client(server_params)
            streams = await self._process_ctx.__aenter__()
            self._read_stream, self._write_stream = streams

            self._session = ClientSession(self._read_stream, self._write_stream)
            await self._session.__aenter__()
            await self._session.initialize()

            self._connected = True
            logger.info("MCP server '%s' 已连接", self.name)
            return True

        except Exception as e:
            logger.error("MCP server '%s' 连接失败: %s", self.name, sanitize_error(str(e)))
            await self._cleanup()
            return False

    async def disconnect(self) -> None:
        """断开连接."""
        async with self._connect_lock:
            await self._cleanup()

    async def _cleanup(self) -> None:
        """清理连接资源."""
        self._connected = False
        if self._session is not None:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None
        if self._process_ctx is not None:
            try:
                await self._process_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._process_ctx = None
        self._read_stream = None
        self._write_stream = None

    async def list_tools(self) -> list[dict[str, Any]]:
        """获取 MCP Server 提供的工具列表."""
        if not self.is_connected:
            if not await self.connect():
                return []

        try:
            result = await asyncio.wait_for(
                self._session.list_tools(),
                timeout=self.config.timeout,
            )
            tools = []
            for tool in result.tools:
                tool_dict = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
                }
                tools.append(tool_dict)
            return tools
        except asyncio.TimeoutError:
            logger.error("MCP server '%s' list_tools 超时 (%ds)", self.name, self.config.timeout)
            return []
        except Exception as e:
            logger.error("MCP server '%s' list_tools 失败: %s", self.name, sanitize_error(str(e)))
            return []

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """调用 MCP Server 的工具.

        Args:
            tool_name: 原始工具名（不含命名空间前缀）
            arguments: 工具参数

        Returns:
            工具输出文本

        Raises:
            TimeoutError: 超时
            ConnectionError: 未连接
            RuntimeError: 调用失败
        """
        if self.circuit_breaker.is_open():
            raise RuntimeError(f"MCP server '{self.name}' 熔断中，请稍后重试")

        if not self.is_connected:
            if not await self.connect():
                raise ConnectionError(f"MCP server '{self.name}' 无法连接")

        try:
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments),
                timeout=self.config.timeout,
            )
            self.circuit_breaker.record_success()

            # 提取文本结果
            if hasattr(result, "content") and result.content:
                parts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                    elif hasattr(block, "data"):
                        parts.append(f"[binary data: {len(block.data)} bytes]")
                    else:
                        parts.append(str(block))
                return "\n".join(parts)
            return str(result)

        except asyncio.TimeoutError:
            self.circuit_breaker.record_failure()
            raise TimeoutError(
                f"MCP server '{self.name}' 工具 '{tool_name}' 超时 ({self.config.timeout}s)"
            )
        except Exception as e:
            self.circuit_breaker.record_failure()
            # 检查是否需要重连（W3: 在锁内设置标志）
            if "closed" in str(e).lower() or "broken" in str(e).lower():
                logger.info("MCP server '%s' 连接断开，标记为重连", self.name)
                self._needs_reconnect = True
            safe_msg = f"MCP server '{self.name}' 工具 '{tool_name}' 调用失败"
            raise RuntimeError(safe_msg) from e


# ── Gateway Manager ──


class MCPGatewayManager:
    """MCP Gateway 管理器 — 统一管理所有 MCP Server 连接.

    使用方式::

        manager = MCPGatewayManager(project_dir=Path("."))
        await manager.start()

        # 工具已自动注入到 _TOOL_SCHEMAS 和 _TOOL_HANDLERS
        # 调用方式和内置工具完全一致

        await manager.stop()
    """

    def __init__(self, project_dir: Any = None):
        from pathlib import Path

        self.project_dir = Path(project_dir) if project_dir else None
        self.registry = MCPToolRegistry()
        self._connections: dict[str, MCPServerConnection] = {}
        self._configs: dict[str, MCPServerConfig] = {}
        self._started = False

    async def start(self) -> int:
        """启动网关：加载配置 → 连接 servers → 注册工具.

        Returns:
            注册的工具总数
        """
        if self._started:
            logger.info("MCP Gateway 已启动，跳过")
            return self.registry.tool_count

        # 1. 加载配置
        self._configs = load_mcp_servers_config(self.project_dir)
        if not self._configs:
            logger.info("无 MCP server 配置，MCP Gateway 未启动")
            return 0

        # 2. 连接每个 server 并注册工具
        total_tools = 0
        for name, config in self._configs.items():
            count = await self._connect_and_register(name, config)
            total_tools += count

        self._started = True
        logger.info(
            "MCP Gateway 已启动: %d servers, %d tools",
            self.registry.server_count,
            self.registry.tool_count,
        )
        return total_tools

    async def _connect_and_register(self, name: str, config: MCPServerConfig) -> int:
        """连接单个 MCP server 并注册其工具."""
        conn = MCPServerConnection(config)
        self._connections[name] = conn

        if not await conn.connect():
            logger.warning("MCP server '%s' 连接失败，跳过工具注册", name)
            return 0

        # 获取工具列表
        raw_tools = await conn.list_tools()
        if not raw_tools:
            logger.warning("MCP server '%s' 无可用工具", name)
            return 0

        # 获取现有内置工具集（碰撞检测用）
        from crew.tool_schema import _TOOL_SCHEMAS

        existing = set(_TOOL_SCHEMAS.keys())

        registered = 0
        whitelist = set(config.whitelist) if config.whitelist else None

        for tool_info in raw_tools:
            raw_name = tool_info["name"]

            # 白名单过滤
            if whitelist and raw_name not in whitelist:
                logger.debug("MCP tool '%s.%s' 不在白名单中，跳过", name, raw_name)
                continue

            # 构建 handler
            handler = self._make_tool_handler(name, raw_name, conn, config)

            # 构建 Anthropic 格式 schema
            schema = {
                "name": raw_name,  # registry 会修正为命名空间名
                "description": (
                    f"[{config.description or name}] {tool_info.get('description', '')}"
                ).strip(),
                "input_schema": tool_info.get(
                    "input_schema",
                    {
                        "type": "object",
                        "properties": {},
                    },
                ),
            }

            ns_name = self.registry.register_tool(
                server_name=name,
                tool_name=raw_name,
                schema=schema,
                handler=handler,
                existing_tools=existing,
            )
            if ns_name:
                registered += 1

        logger.info(
            "MCP server '%s': %d/%d 工具已注册 (白名单=%s)",
            name,
            registered,
            len(raw_tools),
            "yes" if whitelist else "no",
        )
        return registered

    def _make_tool_handler(
        self, server_name: str, tool_name: str, conn: MCPServerConnection,
        config: MCPServerConfig,
    ) -> Any:
        """为一个 MCP 工具创建 handler（含租户检查 + 凭据检查）."""
        ns_name = make_namespaced_name(server_name, tool_name)

        async def _handler(
            args: dict[str, Any],
            *,
            agent_id: str | None = None,
            ctx: Any = None,
            tenant_id: str | None = None,
            user_id: str | None = None,
        ) -> str:
            start = time.monotonic()

            # 1. 租户级权限检查
            if not config.is_tenant_allowed(tenant_id):
                logger.warning("租户拒绝: tenant=%s server=%s", tenant_id, server_name)
                log_tool_call(employee_name=str(agent_id or ""), tenant_id=str(tenant_id or ""), user_id=str(user_id or ""), server_name=server_name, tool_name=tool_name, namespaced_name=ns_name, args=args, success=False, error_message="tenant_not_allowed", duration_ms=0)
                return f"[权限拒绝] 当前租户无权使用 {server_name} 服务。请联系管理员开通权限。"

            # 2. 用户级凭据检查
            if user_id:
                from crew.mcp_gateway.credentials import has_credential
                if not has_credential(user_id, server_name):
                    logger.warning("凭据缺失: user=%s server=%s", user_id, server_name)
                    log_tool_call(employee_name=str(agent_id or ""), tenant_id=str(tenant_id or ""), user_id=str(user_id or ""), server_name=server_name, tool_name=tool_name, namespaced_name=ns_name, args=args, success=False, error_message="credential_not_found", duration_ms=0)
                    return f"[凭据缺失] 您尚未授权 {server_name} 服务。请先在设置中绑定相应账号。"

            # 3. 执行调用
            try:
                result = await conn.call_tool(tool_name, args)
                duration = (time.monotonic() - start) * 1000
                log_tool_call(employee_name=str(agent_id or ""), tenant_id=str(tenant_id or ""), user_id=str(user_id or ""), server_name=server_name, tool_name=tool_name, namespaced_name=ns_name, args=args, success=True, duration_ms=duration)
                return result

            except Exception as e:
                duration = (time.monotonic() - start) * 1000
                error_msg = sanitize_error(str(e))
                log_tool_call(employee_name=str(agent_id or ""), tenant_id=str(tenant_id or ""), user_id=str(user_id or ""), server_name=server_name, tool_name=tool_name, namespaced_name=ns_name, args=args, success=False, error_message=error_msg, duration_ms=duration)
                return f"[MCP Error] {error_msg}"

        return _handler

    def inject_tools(self) -> int:
        """将已注册的 MCP 工具注入到全局 schemas 和 handlers.

        应在 ``start()`` 后调用。

        Returns:
            注入的工具数量
        """
        from crew.tool_schema import _TOOL_SCHEMAS

        # 获取当前 handlers（webhook_executor 已持有引用，直接修改 dict 生效）
        from crew.webhook_executor import _TOOL_HANDLERS

        return self.registry.inject_into(_TOOL_SCHEMAS, _TOOL_HANDLERS)

    def get_server_config(self, server_name: str) -> MCPServerConfig | None:
        """获取指定 MCP server 的配置."""
        return self._configs.get(server_name)

    async def stop(self) -> None:
        """关闭所有 MCP Server 连接."""
        for name, conn in self._connections.items():
            try:
                await conn.disconnect()
                logger.info("MCP server '%s' 已断开", name)
            except Exception as e:
                logger.warning("MCP server '%s' 断开异常: %s", name, e)

        self._connections.clear()
        self._started = False
        logger.info("MCP Gateway 已关闭")

    def get_status(self) -> dict[str, Any]:
        """返回网关状态."""
        servers = {}
        for name, conn in self._connections.items():
            servers[name] = {
                "connected": conn.is_connected,
                "circuit_breaker": conn.circuit_breaker.check_state(),
                "description": conn.config.description,
                "allowed_tenants": conn.config.allowed_tenants,
            }

        return {
            "started": self._started,
            "servers": servers,
            "total_tools": self.registry.tool_count,
            "registered_tools": self.registry.registered_tools(),
        }
