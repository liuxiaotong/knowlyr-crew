"""发现机制 — 内置 + .claude/skills + private/employees."""

import logging
import threading
import time
from pathlib import Path
from typing import Any

from crew.employees import builtin_dir
from crew.models import DiscoveryResult, Employee
from crew.parser import parse_employee, parse_employee_dir, parse_skill, validate_employee

logger = logging.getLogger(__name__)

# ── TTL 缓存 ──
_cache: dict[str, tuple[float, DiscoveryResult]] = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 30.0  # seconds

# Claude Code Skills 目录
SKILLS_DIR_NAME = ".claude/skills"

# 自定义员工目录（相对项目根）
PRIVATE_DIR_NAME = "private/employees"


def _scan_directory(
    dir_path: Path,
    layer: str,
) -> list[Employee]:
    """扫描目录中的员工定义（支持目录格式和文件格式）.

    扫描顺序:
    1. 目录格式: <dir_path>/<name>/employee.yaml → parse_employee_dir()
    2. 文件格式: <dir_path>/<name>.md → parse_employee()（已被目录覆盖的跳过）

    Args:
        dir_path: 要扫描的目录
        layer: 来源层标识（builtin / global / project）

    Returns:
        成功解析的员工列表（跳过解析失败的文件）
    """
    employees = []
    if not dir_path.is_dir():
        return employees

    seen_names: set[str] = set()

    # 1. 扫描目录格式的员工
    for item in sorted(dir_path.iterdir()):
        if not item.is_dir() or not (item / "employee.yaml").exists():
            continue
        try:
            # 可写层自动版本管理
            if layer in ("private",):
                try:
                    from crew.versioning import check_and_bump
                    check_and_bump(item)
                except Exception as e:
                    logger.debug("版本检查失败 %s: %s", item.name, e)

            emp = parse_employee_dir(item, source_layer=layer)

            errors = validate_employee(emp)
            if errors:
                logger.warning("跳过 %s: %s", item, "; ".join(errors))
                continue

            # 权限配置校验（警告级别，不阻止加载）
            from crew.tool_schema import validate_permissions
            for w in validate_permissions(emp):
                logger.warning("权限配置 %s: %s", emp.name, w)

            employees.append(emp)
            seen_names.add(emp.name)
        except ValueError as e:
            logger.warning("跳过 %s: %s", item, e)
        except Exception as e:
            logger.warning("跳过 %s: 未知错误 %s", item, e, exc_info=True)

    # 2. 扫描文件格式的员工（向后兼容），跳过已被目录覆盖的
    for md_file in sorted(dir_path.glob("*.md")):
        if md_file.name.startswith("_") or md_file.name == "README.md":
            continue
        if md_file.stem in seen_names:
            continue  # 目录格式优先
        try:
            emp = parse_employee(md_file)
            emp.source_layer = layer
            emp.source_path = md_file

            errors = validate_employee(emp)
            if errors:
                logger.warning("跳过 %s: %s", md_file, "; ".join(errors))
                continue

            employees.append(emp)
        except ValueError as e:
            logger.warning("跳过 %s: %s", md_file, e)
        except Exception as e:
            logger.warning("跳过 %s: 未知错误 %s", md_file, e, exc_info=True)

    return employees


def _scan_skills_directory(
    skills_dir: Path,
) -> list[Employee]:
    """扫描 .claude/skills/ 目录中的 SKILL.md 文件.

    Claude Code Skills 目录结构: <skills_dir>/<name>/SKILL.md

    Args:
        skills_dir: .claude/skills/ 目录

    Returns:
        成功解析的员工列表
    """
    employees = []
    if not skills_dir.is_dir():
        return employees

    for child in sorted(skills_dir.iterdir()):
        if not child.is_dir():
            continue
        skill_file = child / "SKILL.md"
        if not skill_file.exists():
            continue
        try:
            emp = parse_skill(skill_file, skill_name=child.name)
            emp.source_layer = "skill"
            emp.source_path = skill_file

            errors = validate_employee(emp)
            if errors:
                logger.warning("跳过 %s: %s", skill_file, "; ".join(errors))
                continue

            employees.append(emp)
        except ValueError as e:
            logger.warning("跳过 %s: %s", skill_file, e)
        except Exception as e:
            logger.warning("跳过 %s: 未知错误 %s", skill_file, e, exc_info=True)

    return employees


def _merge_employee(
    emp: Employee,
    layer_name: str,
    employees: dict[str, Employee],
    trigger_map: dict[str, str],
    conflicts: list[dict],
) -> None:
    """合并单个员工到结果集（高层覆盖低层）."""
    if emp.name in employees:
        existing = employees[emp.name]
        conflicts.append({
            "name": emp.name,
            "winner": f"{layer_name}:{emp.source_path}",
            "loser": f"{existing.source_layer}:{existing.source_path}",
        })
        logger.info(
            "员工 '%s' 被 %s 层覆盖（原: %s 层）",
            emp.name, layer_name, existing.source_layer,
        )

    employees[emp.name] = emp

    for trigger in emp.triggers:
        if trigger in trigger_map and trigger_map[trigger] != emp.name:
            existing_name = trigger_map[trigger]
            conflicts.append({
                "type": "trigger",
                "trigger": trigger,
                "winner": emp.name,
                "loser": existing_name,
            })
        trigger_map[trigger] = emp.name


def discover_employees(
    project_dir: Path | None = None,
    *,
    cache_ttl: float | None = None,
) -> DiscoveryResult:
    """执行员工发现（内置 + skills + private/employees）.

    带 TTL 缓存（默认 30s），按 project_dir 分 key。
    cache_ttl=0 禁用缓存（适合 CLI / 测试场景）。

    优先级（高覆盖低）:
    1. 内置层: 包内 employees/*.md
    2. 技能层: {project_dir}/.claude/skills/<name>/SKILL.md
    3. private 层: {project_dir}/private/employees/
    """
    from crew.paths import resolve_project_dir
    root = resolve_project_dir(project_dir)

    ttl = cache_ttl if cache_ttl is not None else _CACHE_TTL
    key = str(root)
    now = time.monotonic()

    with _cache_lock:
        if ttl > 0 and key in _cache:
            ts, result = _cache[key]
            if now - ts < ttl:
                return result

    result = _discover_employees_uncached(root)

    if ttl > 0:
        with _cache_lock:
            _cache[key] = (now, result)

    return result


def _discover_employees_uncached(root: Path) -> DiscoveryResult:
    """实际执行文件系统扫描的内部函数."""
    employees: dict[str, Employee] = {}
    trigger_map: dict[str, str] = {}
    conflicts: list[dict[str, Any]] = []

    # 低优先级：内置层
    for emp in _scan_directory(builtin_dir(), "builtin"):
        _merge_employee(emp, "builtin", employees, trigger_map, conflicts)

    # 中优先级：.claude/skills/
    skills_dir = root / SKILLS_DIR_NAME
    for emp in _scan_skills_directory(skills_dir):
        _merge_employee(emp, "skill", employees, trigger_map, conflicts)

    # 高优先级：private/employees/
    private_dir = root / PRIVATE_DIR_NAME
    for emp in _scan_directory(private_dir, "private"):
        _merge_employee(emp, "private", employees, trigger_map, conflicts)

    return DiscoveryResult(employees=employees, conflicts=conflicts)


def get_employee(
    name_or_trigger: str,
    project_dir: Path | None = None,
) -> Employee | None:
    """按名称或触发别名查找员工.

    Args:
        name_or_trigger: 员工名称或触发别名
        project_dir: 项目根目录

    Returns:
        Employee 对象，未找到返回 None
    """
    result = discover_employees(project_dir=project_dir)
    return result.get(name_or_trigger)
