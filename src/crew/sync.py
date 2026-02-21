"""knowlyr-id 双向同步 — 批量推送本地员工数据 / 拉取运行时数据."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from crew.engine import CrewEngine
from crew.id_client import (
    AgentIdentity,
    fetch_agent_identity,
    list_agents,
    register_agent,
    update_agent,
)
from crew.parser import parse_employee_dir
from crew.versioning import compute_content_hash

logger = logging.getLogger(__name__)


@dataclass
class SyncReport:
    """同步结果汇总."""

    pushed: list[str] = field(default_factory=list)
    pulled: list[str] = field(default_factory=list)
    registered: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)


def _load_avatar_b64(emp_dir: Path) -> str | None:
    """从员工目录加载 avatar.webp 并 base64 编码."""
    avatar_path = emp_dir / "avatar.webp"
    if not avatar_path.exists():
        return None
    return base64.b64encode(avatar_path.read_bytes()).decode()


def _read_yaml_config(emp_dir: Path) -> dict:
    """读取 employee.yaml 原始字典."""
    config_path = emp_dir / "employee.yaml"
    if not config_path.exists():
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_yaml_field(emp_dir: Path, updates: dict) -> None:
    """更新 employee.yaml 中的指定字段."""
    config_path = emp_dir / "employee.yaml"
    if not config_path.exists():
        return
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        return
    config.update(updates)
    import os
    import tempfile

    content = yaml.dump(config, allow_unicode=True, sort_keys=False, default_flow_style=False)
    fd, tmp = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
    fd_closed = False
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        fd_closed = True
        os.replace(tmp, config_path)
    except Exception:
        if not fd_closed:
            os.close(fd)
        Path(tmp).unlink(missing_ok=True)
        raise


def _write_agent_id(emp_dir: Path, agent_id: int) -> None:
    """回写 agent_id 到 employee.yaml."""
    _write_yaml_field(emp_dir, {"agent_id": agent_id})


def sync_all(
    employees_dir: Path,
    *,
    dry_run: bool = False,
    push: bool = True,
    pull: bool = True,
    force: bool = False,
    register: bool = False,
) -> SyncReport:
    """批量同步 employees_dir 下所有员工到 knowlyr-id.

    Args:
        employees_dir: 员工目录（如 .crew/global/）
        dry_run: 仅预览，不执行任何写操作
        push: 推送本地数据到 id
        pull: 拉取 id 数据到本地
        force: 忽略 content_hash，强制推送
        register: 是否注册新员工（默认 False，只报告不注册）

    Returns:
        SyncReport 汇总
    """
    report = SyncReport()

    if not employees_dir.is_dir():
        report.errors.append(("", f"目录不存在: {employees_dir}"))
        return report

    # 1. 扫描本地员工
    local_employees: dict[int, tuple[Path, dict]] = {}  # agent_id → (dir, yaml_config)
    new_employees: list[tuple[Path, dict]] = []  # 无 agent_id 的新员工

    for item in sorted(employees_dir.iterdir()):
        if not item.is_dir() or not (item / "employee.yaml").exists():
            continue
        # 缺 prompt.md 的目录不参与同步（避免注册空壳 agent）
        if not (item / "prompt.md").exists():
            name = _read_yaml_config(item).get("name", item.name)
            report.skipped.append(f"{name} (缺少 prompt.md)")
            logger.warning("跳过 %s: 缺少 prompt.md，不参与同步", item)
            continue
        config = _read_yaml_config(item)
        agent_id = config.get("agent_id")
        if agent_id is not None:
            local_employees[int(agent_id)] = (item, config)
        else:
            new_employees.append((item, config))

    # 2. 获取远程 Agent 列表
    remote_agents = list_agents()
    if remote_agents is None:
        report.errors.append(("", "无法获取 Agent 列表（检查 KNOWLYR_ID_URL / AGENT_API_TOKEN）"))
        return report

    remote_map: dict[int, dict] = {}
    for agent in remote_agents:
        aid = agent.get("id")
        if aid is not None:
            remote_map[int(aid)] = agent

    engine = CrewEngine()

    # 3. 同步已绑定的员工（本地有 agent_id + id 有）
    for agent_id, (emp_dir, config) in local_employees.items():
        name = config.get("name", emp_dir.name)
        try:
            if push:
                _push_employee(
                    emp_dir, config, agent_id, engine, report, dry_run=dry_run, force=force
                )
            if pull:
                remote_agent = remote_map.get(agent_id)
                _pull_employee(
                    emp_dir, config, agent_id, report, dry_run=dry_run, remote_data=remote_agent
                )
        except Exception as e:
            report.errors.append((name, str(e)))
            logger.warning("同步 %s 失败: %s", name, e)

    # 4. 处理新员工（本地无 agent_id）
    if push and new_employees:
        if register:
            # 显式要求注册时才注册
            for emp_dir, config in new_employees:
                name = config.get("name", emp_dir.name)
                try:
                    _register_new(emp_dir, config, engine, report, dry_run=dry_run)
                except Exception as e:
                    report.errors.append((name, str(e)))
                    logger.warning("注册 %s 失败: %s", name, e)
        else:
            # 默认只报告，不注册
            for emp_dir, config in new_employees:
                name = config.get("name", emp_dir.name)
                report.skipped.append(f"{name} (未注册，用 `knowlyr-crew register {name}` 注册)")

    # 5. 禁用已删除的员工（id 有但本地无）
    if push:
        local_ids = set(local_employees.keys())
        for agent_id, agent_data in remote_map.items():
            if agent_id not in local_ids and agent_data.get("agent_status") not in (
                "inactive",
                "frozen",
            ):
                name = agent_data.get("nickname", str(agent_id))
                if dry_run:
                    report.disabled.append(f"{name} (#{agent_id}) [dry-run]")
                else:
                    ok = update_agent(agent_id, agent_status="inactive")
                    if ok:
                        report.disabled.append(f"{name} (#{agent_id})")
                    else:
                        report.errors.append((name, "禁用失败"))

    return report


def _push_employee(
    emp_dir: Path,
    config: dict,
    agent_id: int,
    engine: CrewEngine,
    report: SyncReport,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    """推送单个员工数据到 knowlyr-id."""
    name = config.get("name", emp_dir.name)

    # 跳过优化：content_hash 未变且非 force
    prompt_changed = True
    if not force:
        stored_hash = config.get("_content_hash", "")
        current_hash = compute_content_hash(emp_dir)
        # 如果我们有 stored_hash 且与 current 相同，可以考虑跳过
        # 但由于 yaml 字段（bio, display_name）也可能变，始终推送 metadata
        # 仅跳过 prompt + avatar 的推送
        prompt_changed = (stored_hash != current_hash) or not stored_hash

    # 构建 push 数据
    nickname = config.get("character_name") or config.get("display_name") or name
    title = config.get("display_name") or name
    bio = config.get("bio") or ""
    capabilities = config.get("description", "")
    tags = config.get("tags", [])
    domains = tags[:5] if tags else []

    # 渲染 prompt（仅当内容变化或 force 时）
    system_prompt = None
    if force or prompt_changed:
        try:
            emp = parse_employee_dir(emp_dir, source_layer="private")
            system_prompt = engine.prompt(emp)
        except Exception as e:
            logger.warning("渲染 %s prompt 失败: %s", name, e)
            system_prompt = None

    # 加载头像（仅当内容变化或 force 时）
    avatar_b64 = None
    if force or prompt_changed:
        avatar_b64 = _load_avatar_b64(emp_dir)

    if dry_run:
        label = f"{name} (#{agent_id})"
        if system_prompt:
            label += f" [prompt: {len(system_prompt)}字]"
        if avatar_b64:
            label += " [avatar]"
        report.pushed.append(f"{label} [dry-run]")
        return

    ok = update_agent(
        agent_id=agent_id,
        nickname=nickname,
        title=title,
        bio=bio,
        capabilities=capabilities,
        domains=domains,
        model=config.get("model") or None,
        system_prompt=system_prompt,
        avatar_base64=avatar_b64,
        crew_name=name,
        temperature=config.get("temperature"),
        max_tokens=config.get("max_tokens"),
    )
    if ok:
        report.pushed.append(name)
    else:
        report.errors.append((name, "推送失败"))


def _pull_employee(
    emp_dir: Path,
    config: dict,
    agent_id: int,
    report: SyncReport,
    *,
    dry_run: bool = False,
    remote_data: dict | None = None,
) -> None:
    """从 knowlyr-id 拉取运行时数据到本地.

    优先使用 remote_data（来自 list_agents 批量接口），
    仅在 remote_data 为 None 时才调 fetch_agent_identity（逐个接口，可能 404）。
    """
    name = config.get("name", emp_dir.name)

    if remote_data is not None:
        # 从 list_agents 批量数据构造（不含 memory/temperature，仅状态同步）
        identity = AgentIdentity(
            agent_id=agent_id,
            nickname=remote_data.get("nickname", ""),
            title=remote_data.get("title", ""),
            bio=remote_data.get("bio", ""),
            domains=remote_data.get("domains", []),
            model=remote_data.get("model", ""),
            agent_status=remote_data.get("status", "active"),
        )
    else:
        identity = fetch_agent_identity(agent_id)
        if identity is None:
            report.errors.append((name, f"无法获取 Agent #{agent_id} 身份"))
            return

    pulled = False

    # 拉取 memory → memory-id.md（仅 fetch_agent_identity 路径有数据）
    if identity.memory and identity.memory.strip():
        memory_path = emp_dir / "memory-id.md"
        existing = memory_path.read_text(encoding="utf-8").strip() if memory_path.exists() else ""
        if existing != identity.memory.strip():
            if not dry_run:
                memory_path.write_text(identity.memory.strip() + "\n", encoding="utf-8")
            pulled = True

    # 拉取 agent_status → employee.yaml（仅在状态变化时写入）
    _VALID_AGENT_STATUS = {"active", "frozen", "inactive"}
    yaml_updates: dict = {}
    if (
        identity.agent_status
        and identity.agent_status in _VALID_AGENT_STATUS
        and identity.agent_status != config.get("agent_status", "active")
    ):
        yaml_updates["agent_status"] = identity.agent_status

    # 拉取 temperature → employee.yaml（仅 fetch_agent_identity 路径有数据）
    if identity.temperature is not None and identity.temperature != config.get("temperature"):
        yaml_updates["temperature"] = identity.temperature

    if yaml_updates:
        if not dry_run:
            _write_yaml_field(emp_dir, yaml_updates)
        pulled = True

    if pulled:
        label = f"{name} (#{agent_id})"
        if dry_run:
            label += " [dry-run]"
        report.pulled.append(label)


def _register_new(
    emp_dir: Path,
    config: dict,
    engine: CrewEngine,
    report: SyncReport,
    *,
    dry_run: bool = False,
) -> None:
    """注册新员工到 knowlyr-id."""
    name = config.get("name", emp_dir.name)
    nickname = config.get("character_name") or config.get("display_name") or name
    title = config.get("display_name") or name
    capabilities = config.get("description", "")
    bio = config.get("bio", "")
    tags = config.get("tags", [])
    domains = tags[:5] if tags else []
    model = config.get("model", "")
    temperature = config.get("temperature")
    max_tokens = config.get("max_tokens")

    # 渲染 prompt
    system_prompt = ""
    try:
        emp = parse_employee_dir(emp_dir, source_layer="private")
        system_prompt = engine.prompt(emp)
    except Exception as e:
        logger.warning("渲染 %s prompt 失败: %s", name, e)

    avatar_b64 = _load_avatar_b64(emp_dir)

    if dry_run:
        report.registered.append(f"{name} [dry-run]")
        return

    agent_id = register_agent(
        nickname=nickname,
        title=title,
        capabilities=capabilities,
        domains=domains,
        model=model,
        system_prompt=system_prompt,
        avatar_base64=avatar_b64,
        crew_name=name,
        bio=bio,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if agent_id is None:
        report.errors.append((name, "注册失败"))
        return

    # 回写 agent_id
    _write_agent_id(emp_dir, agent_id)
    report.registered.append(f"{name} (#{agent_id})")
