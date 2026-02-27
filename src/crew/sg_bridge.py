"""SG 转发 Bridge — 通过 SSH 将飞书消息转发到新加坡 Claude Code 节点处理.

主通道: SSH → claude -p '消息' → 获取回复
失败时: 抛出 SGBridgeError，调用方 fallback 到现有 crew 引擎

特性:
- SSH ControlMaster 长连接复用
- Circuit Breaker（连续失败自动熔断，定时恢复探测）
- 模型分层路由（haiku/sonnet/opus 按消息类型自动选择）
- 合理的超时控制
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from crew.paths import resolve_project_dir

logger = logging.getLogger(__name__)


# ── 异常 ──


class SGBridgeError(Exception):
    """SG Bridge 通用异常 — 调用方应 fallback."""


class SGBridgeTimeout(SGBridgeError):
    """SSH 或 claude 命令超时."""


class SGBridgeUnavailable(SGBridgeError):
    """SG 节点不可达（SSH 连接失败）."""


# ── 配置 ──


@dataclass
class SGBridgeConfig:
    """SG Bridge 配置."""

    # SSH 连接
    ssh_host: str = "43.106.24.105"
    ssh_user: str = "root"
    ssh_port: int = 22
    ssh_key_path: str = "~/.ssh/knowlyr-liukai.pem"

    # 超时（秒）
    ssh_connect_timeout: int = 10
    claude_timeout: int = 180  # claude -p 执行超时（含工具调用）
    health_check_timeout: int = 5

    # SSH ControlMaster
    control_path: str = "/tmp/ssh-sg-bridge-%r@%h:%p"
    control_persist: int = 600  # 10 分钟

    # Claude CLI 配置
    claude_bin: str = "claude"  # SG 上的 claude 命令路径
    claude_env_file: str = "/root/.claude/env.sh"  # SG 上的环境变量文件（SSH 非交互不加载 .profile）
    default_model: str = "sonnet"  # 默认模型标识
    # -p 模式下允许的工具（root 不能用 --dangerously-skip-permissions）
    allowed_tools: list[str] = field(
        default_factory=lambda: [
            "WebSearch", "WebFetch", "Read", "Grep", "Glob",
            "mcp__crew__query_memory", "mcp__crew__list_employees",
            "mcp__crew__get_work_log",
        ]
    )

    # Circuit Breaker
    cb_failure_threshold: int = 3  # 连续失败次数阈值
    cb_recovery_timeout: int = 300  # 熔断后恢复探测间隔（秒）

    # 模型分层关键词
    opus_keywords: list[str] = field(
        default_factory=lambda: [
            "架构",
            "重构",
            "设计方案",
            "技术选型",
            "复盘",
            "事后分析",
            "合同",
            "法务",
            "金额",
            "预算",
            "报价",
            "方案对比",
            "权衡",
            "trade-off",
            "tradeoff",
            "/model opus",
        ]
    )
    haiku_keywords: list[str] = field(
        default_factory=lambda: [
            "你好",
            "早",
            "晚安",
            "谢谢",
            "哈哈",
            "嗯",
            "好的",
            "收到",
            "了解",
            "ok",
            "hi",
            "hello",
            "嗨",
            "在吗",
            "在不在",
        ]
    )
    # 纯闲聊最大长度（超过此长度不走 haiku）
    haiku_max_length: int = 30

    # 是否启用 SG Bridge
    enabled: bool = True


def load_sg_bridge_config(project_dir: Path | None = None) -> SGBridgeConfig:
    """从 .crew/sg_bridge.yaml 加载配置，文件不存在则返回默认配置."""
    base = resolve_project_dir(project_dir)
    config_path = base / ".crew" / "sg_bridge.yaml"
    if not config_path.exists():
        return SGBridgeConfig(enabled=False)
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return SGBridgeConfig(enabled=False)
        return SGBridgeConfig(**data)
    except Exception as e:
        logger.warning("SG Bridge 配置加载失败: %s", e)
        return SGBridgeConfig(enabled=False)


# ── 模型路由 ──


class ModelTier(str, Enum):
    """模型分层."""

    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"


def select_model_tier(text: str, config: SGBridgeConfig) -> ModelTier:
    """根据消息内容选择模型分层.

    规则：
    1. 用户手动 /model opus → opus
    2. 命中重决策关键词 → opus
    3. 短消息 + 命中闲聊关键词 → haiku
    4. 默认 → sonnet
    """
    if not text:
        return ModelTier.SONNET

    t = text.strip().lower()

    # 手动指定
    if "/model opus" in t:
        return ModelTier.OPUS
    if "/model haiku" in t:
        return ModelTier.HAIKU
    if "/model sonnet" in t:
        return ModelTier.SONNET

    # 重决策关键词
    for kw in config.opus_keywords:
        if kw.lower() in t:
            return ModelTier.OPUS

    # 闲聊关键词（仅短消息）
    if len(text.strip()) <= config.haiku_max_length:
        for kw in config.haiku_keywords:
            if kw.lower() in t:
                return ModelTier.HAIKU

    return ModelTier.SONNET


def _clean_reply(text: str) -> str:
    """清理 claude -p 回复中不适合聊天的内容."""
    # 去掉【墨言】等前缀标记（CLAUDE.md 残留或模型习惯）
    text = re.sub(r"^【[^】]+】\s*", "", text.strip())
    # 去掉 Sources: 引用块（WebSearch 工具自动附加，聊天不需要）
    text = re.sub(r"\n*Sources?:\s*\n[-–•*\s].*", "", text, flags=re.DOTALL).strip()
    return text


def _strip_model_command(text: str) -> str:
    """从消息中移除 /model xxx 指令."""
    return re.sub(r"/model\s+(opus|sonnet|haiku)\b", "", text).strip()


# ── Circuit Breaker ──


class CBState(str, Enum):
    """Circuit Breaker 状态."""

    CLOSED = "closed"  # 正常运行
    OPEN = "open"  # 熔断中，拒绝请求
    HALF_OPEN = "half_open"  # 恢复探测，放行单个请求


class CircuitBreaker:
    """简单的 Circuit Breaker 实现.

    - CLOSED: 正常转发，记录连续失败次数
    - OPEN: 拒绝所有请求，等待 recovery_timeout 后进入 HALF_OPEN
    - HALF_OPEN: 放行一个请求，成功回到 CLOSED，失败回到 OPEN
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: int = 300,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CBState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._last_state_change: float = time.monotonic()

    @property
    def state(self) -> CBState:
        """获取当前状态（含自动状态转换）."""
        if self._state == CBState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = CBState.HALF_OPEN
                self._last_state_change = time.monotonic()
                logger.info("Circuit Breaker: OPEN → HALF_OPEN（恢复探测）")
        return self._state

    def allow_request(self) -> bool:
        """检查是否允许请求通过."""
        s = self.state
        if s == CBState.CLOSED:
            return True
        if s == CBState.HALF_OPEN:
            return True  # 放行探测请求
        return False  # OPEN 状态拒绝

    def record_success(self) -> None:
        """记录请求成功."""
        if self._state in (CBState.HALF_OPEN, CBState.CLOSED):
            self._failure_count = 0
            if self._state != CBState.CLOSED:
                logger.info("Circuit Breaker: %s → CLOSED（恢复正常）", self._state.value)
            self._state = CBState.CLOSED
            self._last_state_change = time.monotonic()

    def record_failure(self) -> None:
        """记录请求失败."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CBState.HALF_OPEN:
            # 探测失败，回到 OPEN
            self._state = CBState.OPEN
            self._last_state_change = time.monotonic()
            logger.warning("Circuit Breaker: HALF_OPEN → OPEN（探测失败）")
        elif self._failure_count >= self.failure_threshold:
            self._state = CBState.OPEN
            self._last_state_change = time.monotonic()
            logger.warning(
                "Circuit Breaker: CLOSED → OPEN（连续 %d 次失败）",
                self._failure_count,
            )

    def reset(self) -> None:
        """重置为初始状态."""
        self._state = CBState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0
        self._last_state_change = time.monotonic()


# ── SG Bridge 核心 ──


class SGBridge:
    """SG SSH 转发 Bridge.

    通过 SSH 连接到新加坡 Claude Code 节点，
    执行 `claude -p '消息'` 并捕获输出。

    特性：
    - SSH ControlMaster 长连接（避免每次重建 TCP + SSH 握手）
    - 超时控制（SSH 连接 + claude 执行分开计时）
    - Circuit Breaker（连续失败自动切 fallback）
    """

    def __init__(self, config: SGBridgeConfig | None = None):
        self.config = config or SGBridgeConfig()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.cb_failure_threshold,
            recovery_timeout=self.config.cb_recovery_timeout,
        )
        self._control_master_active = False

    def _ssh_base_args(self) -> list[str]:
        """SSH 基础参数（含 ControlMaster 配置）."""
        key_path = str(Path(self.config.ssh_key_path).expanduser())
        return [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={self.config.ssh_connect_timeout}",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            # ControlMaster 长连接
            "-o", f"ControlPath={self.config.control_path}",
            "-o", "ControlMaster=auto",
            "-o", f"ControlPersist={self.config.control_persist}",
            "-i", key_path,
            "-p", str(self.config.ssh_port),
        ]

    def _ssh_target(self) -> str:
        """SSH 目标 user@host."""
        return f"{self.config.ssh_user}@{self.config.ssh_host}"

    async def health_check(self) -> bool:
        """检查 SG 节点是否可达（SSH echo test）."""
        try:
            cmd = self._ssh_base_args() + [self._ssh_target(), "echo", "ok"]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.health_check_timeout,
            )
            return proc.returncode == 0 and b"ok" in stdout
        except (asyncio.TimeoutError, OSError) as e:
            logger.debug("SG health check 失败: %s", e)
            return False

    async def execute_claude(
        self,
        message: str,
        *,
        model_tier: ModelTier = ModelTier.SONNET,
        employee_context: str | None = None,
    ) -> str:
        """通过 SSH 在 SG 节点执行 claude -p 并返回回复文本.

        Args:
            message: 用户消息
            model_tier: 模型分层（haiku/sonnet/opus）
            employee_context: 员工上下文提示（可选，会加入 system prompt 前缀）

        Returns:
            claude 的回复文本

        Raises:
            SGBridgeTimeout: 执行超时
            SGBridgeUnavailable: SSH 连接失败
            SGBridgeError: 其他错误
        """
        # 构建 claude 命令
        # 使用 claude -p 非交互模式
        # 用 stdin pipe 传消息（避免多层 SSH 引号被吞）

        # 构建远程命令
        # cd /tmp: 脱离项目目录，避免读取 CLAUDE.md（里面有【墨言】前缀指令）
        # source env.sh: SSH 非交互不加载 .profile，需手动 source 环境变量
        env_prefix = f"cd /tmp && source {self.config.claude_env_file} && "
        claude_cmd_parts = [self.config.claude_bin, "-p"]

        # 添加模型标识（如果不是默认 sonnet）
        if model_tier == ModelTier.OPUS:
            claude_cmd_parts.extend(["--model", "opus"])
        elif model_tier == ModelTier.HAIKU:
            claude_cmd_parts.extend(["--model", "haiku"])
        # sonnet 是默认，不需要额外参数

        # 启用工具（root 不能用 --dangerously-skip-permissions，用 --allowedTools 代替）
        if self.config.allowed_tools:
            tools_str = ",".join(self.config.allowed_tools)
            claude_cmd_parts.extend(["--allowedTools", tools_str])

        # 员工身份注入（--append-system-prompt 不覆盖默认 system prompt）
        if employee_context:
            # 转义双引号，用双引号包裹
            escaped_ctx = employee_context.replace("\\", "\\\\").replace('"', '\\"')
            claude_cmd_parts.extend(["--append-system-prompt", f'"{escaped_ctx}"'])

        # 不拼消息到命令行，走 stdin（避免 shell 多层转义问题）
        remote_cmd = env_prefix + " ".join(claude_cmd_parts)

        # SSH 完整命令
        ssh_cmd = self._ssh_base_args() + [self._ssh_target(), remote_cmd]

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=message.encode("utf-8")),
                timeout=self.config.claude_timeout,
            )
        except asyncio.TimeoutError:
            # 尝试 kill 进程
            try:
                proc.kill()  # type: ignore[union-attr]
            except ProcessLookupError:
                pass
            raise SGBridgeTimeout(
                f"claude 执行超时（{self.config.claude_timeout}s）"
            )
        except OSError as e:
            raise SGBridgeUnavailable(f"SSH 连接失败: {e}")

        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            # 区分 SSH 连接错误和 claude 执行错误
            if "Connection refused" in stderr_text or "Connection timed out" in stderr_text:
                raise SGBridgeUnavailable(f"SSH 连接被拒: {stderr_text[:200]}")
            if "Permission denied" in stderr_text:
                raise SGBridgeUnavailable(f"SSH 认证失败: {stderr_text[:200]}")
            raise SGBridgeError(
                f"claude 执行失败 (rc={proc.returncode}): {stderr_text[:500]}"
            )

        output = stdout.decode("utf-8", errors="replace").strip()
        if not output:
            raise SGBridgeError("claude 返回空输出")

        return _clean_reply(output)

    async def close_control_master(self) -> None:
        """关闭 SSH ControlMaster 连接."""
        try:
            cmd = [
                "ssh",
                "-o", f"ControlPath={self.config.control_path}",
                "-O", "exit",
                self._ssh_target(),
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=5)
            self._control_master_active = False
        except Exception as e:
            logger.debug("关闭 SSH ControlMaster 失败: %s", e)


# ── 模块级单例 ──

_bridge_instance: SGBridge | None = None
_bridge_config: SGBridgeConfig | None = None


def get_sg_bridge(project_dir: Path | None = None) -> SGBridge | None:
    """获取 SG Bridge 单例（懒加载）.

    Returns:
        SGBridge 实例，如果未启用则返回 None
    """
    global _bridge_instance, _bridge_config
    if _bridge_config is None:
        _bridge_config = load_sg_bridge_config(project_dir)
    if not _bridge_config.enabled:
        return None
    if _bridge_instance is None:
        _bridge_instance = SGBridge(_bridge_config)
    return _bridge_instance


def reset_sg_bridge() -> None:
    """重置单例（测试用）."""
    global _bridge_instance, _bridge_config
    _bridge_instance = None
    _bridge_config = None


# ── 高层 API ──


async def sg_dispatch(
    message: str,
    *,
    project_dir: Path | None = None,
    employee_name: str | None = None,
    chat_context: str | None = None,
    message_history: list[dict] | None = None,
) -> str:
    """SG 转发主入口 — 成功返回回复文本，失败抛出 SGBridgeError.

    调用方应捕获 SGBridgeError 并 fallback 到现有 crew 引擎。

    Args:
        message: 用户消息文本
        project_dir: 项目目录
        employee_name: 目标员工名称（用于上下文）
        chat_context: 飞书对话场景上下文
        message_history: 飞书对话历史（最近几轮），每条 {"role": "user"|"assistant", "content": "..."}

    Returns:
        claude 的回复文本

    Raises:
        SGBridgeError: 所有可恢复错误（调用方应 fallback）
    """
    bridge = get_sg_bridge(project_dir)
    if bridge is None:
        raise SGBridgeError("SG Bridge 未启用")

    config = bridge.config

    # Circuit Breaker 检查
    if not bridge.circuit_breaker.allow_request():
        raise SGBridgeError("SG Bridge 熔断中，等待恢复")

    # 模型选择
    clean_message = _strip_model_command(message)
    model_tier = select_model_tier(message, config)

    # 组装 system prompt（员工身份 + 场景）
    system_parts: list[str] = []
    if employee_name:
        system_parts.append(
            f"你是{employee_name}，集识光年的 AI 员工。"
            "直接回答问题，不要加任何前缀标记（如【墨言】）。"
            "用中文自然对话。"
        )
    if chat_context:
        system_parts.append(chat_context)
    employee_ctx = "\n".join(system_parts) if system_parts else None

    # 组装发给 claude 的用户消息
    full_message = ""

    # 拼入对话历史（最多 6 条，避免超过 claude -p 的 stdin 限制）
    if message_history:
        recent = message_history[-6:]
        history_lines = []
        for msg in recent:
            role_label = "Kai" if msg.get("role") == "user" else (employee_name or "助手")
            history_lines.append(f"{role_label}: {msg.get('content', '')}")
        if history_lines:
            full_message += "[最近对话]\n" + "\n".join(history_lines) + "\n[当前消息]\n"

    full_message += clean_message if clean_message else message

    logger.info(
        "SG dispatch: model=%s msg=%s cb=%s",
        model_tier.value,
        message[:40],
        bridge.circuit_breaker.state.value,
    )

    try:
        reply = await bridge.execute_claude(
            full_message,
            model_tier=model_tier,
            employee_context=employee_ctx,
        )
        bridge.circuit_breaker.record_success()
        logger.info(
            "SG dispatch 成功: model=%s reply_len=%d",
            model_tier.value,
            len(reply),
        )
        return reply
    except SGBridgeError:
        bridge.circuit_breaker.record_failure()
        raise
    except Exception as e:
        bridge.circuit_breaker.record_failure()
        raise SGBridgeError(f"SG dispatch 未知错误: {e}") from e
