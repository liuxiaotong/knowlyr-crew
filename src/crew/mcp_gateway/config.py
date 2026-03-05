"""MCP Gateway 配置 — YAML 加载和 server 定义.

系统级 MCP server 配置文件路径: ``{project_dir}/.crew/mcp_servers.yaml``

示例::

    servers:
      gws:
        command: "gws"
        args: ["mcp"]
        env:
          GOOGLE_APPLICATION_CREDENTIALS: "${GOOGLE_APPLICATION_CREDENTIALS}"
        whitelist:
          - "calendar_list_events"
          - "calendar_get_event"
          - "gmail_list_messages"
          - "gmail_get_message"
        timeout: 30
        description: "Google Workspace (只读)"
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_env(value: str) -> str:
    """将 ``${ENV_VAR}`` 替换为环境变量值."""

    def _replacer(m: re.Match) -> str:
        return os.environ.get(m.group(1), "")

    return _ENV_VAR_RE.sub(_replacer, value)


@dataclass
class MCPServerConfig:
    """单个 MCP Server 的配置."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    whitelist: list[str] = field(default_factory=list)
    timeout: int = 30
    description: str = ""

    def resolved_env(self) -> dict[str, str]:
        """返回环境变量解析后的 env dict."""
        return {k: _resolve_env(v) for k, v in self.env.items()}

    def resolved_command(self) -> str:
        """返回环境变量解析后的 command."""
        return _resolve_env(self.command)

    def resolved_args(self) -> list[str]:
        """返回环境变量解析后的 args."""
        return [_resolve_env(a) for a in self.args]


def load_mcp_servers_config(
    project_dir: Path | None = None,
) -> dict[str, MCPServerConfig]:
    """从 YAML 文件加载 MCP server 配置.

    Returns:
        server_name -> MCPServerConfig 映射，文件不存在则返回空 dict。
    """
    base = project_dir or Path(".")
    config_path = base / ".crew" / "mcp_servers.yaml"
    if not config_path.is_file():
        logger.debug("MCP servers 配置不存在: %s", config_path)
        return {}

    try:
        import yaml
    except ImportError:
        logger.warning("pyyaml 未安装，无法加载 MCP servers 配置")
        return {}

    with open(config_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    servers_raw: dict[str, Any] = raw.get("servers", {})
    configs: dict[str, MCPServerConfig] = {}

    for name, cfg in servers_raw.items():
        if not isinstance(cfg, dict):
            logger.warning("MCP server '%s' 配置格式错误，跳过", name)
            continue
        configs[name] = MCPServerConfig(
            name=name,
            command=cfg.get("command", ""),
            args=cfg.get("args", []),
            env=cfg.get("env", {}),
            whitelist=cfg.get("whitelist", []),
            timeout=cfg.get("timeout", 30),
            description=cfg.get("description", ""),
        )
        logger.info(
            "MCP server 已加载: %s (command=%s, whitelist=%d tools)",
            name,
            cfg.get("command", ""),
            len(configs[name].whitelist),
        )

    return configs
