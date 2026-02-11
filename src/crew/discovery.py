"""四层发现机制 — 内置 + 全局 + skill + 项目."""

import logging
from pathlib import Path
from typing import Any

from crew.employees import builtin_dir
from crew.models import DiscoveryResult, Employee
from crew.parser import parse_employee, parse_skill, validate_employee

logger = logging.getLogger(__name__)

# 全局员工目录
GLOBAL_DIR = Path.home() / ".knowlyr" / "crew"

# 项目员工目录名
PROJECT_DIR_NAME = ".crew"

# Claude Code Skills 目录
SKILLS_DIR_NAME = ".claude/skills"


def _scan_directory(
    dir_path: Path,
    layer: str,
) -> list[Employee]:
    """扫描目录中的 .md 文件，解析为 Employee 列表.

    Args:
        dir_path: 要扫描的目录
        layer: 来源层标识（builtin / global / project）

    Returns:
        成功解析的员工列表（跳过解析失败的文件）
    """
    employees = []
    if not dir_path.is_dir():
        return employees

    for md_file in sorted(dir_path.glob("*.md")):
        if md_file.name.startswith("_") or md_file.name == "README.md":
            continue
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
            logger.warning("跳过 %s: 未知错误 %s", md_file, e)

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
            logger.warning("跳过 %s: 未知错误 %s", skill_file, e)

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
) -> DiscoveryResult:
    """执行完整的四层发现.

    优先级（高覆盖低）:
    1. 项目层: {project_dir}/.crew/*.md
    2. 技能层: {project_dir}/.claude/skills/<name>/SKILL.md
    3. 全局层: ~/.knowlyr/crew/*.md
    4. 内置层: 包内 employees/*.md

    Args:
        project_dir: 项目根目录，默认为当前工作目录

    Returns:
        DiscoveryResult 包含去重后的员工映射和冲突记录
    """
    if project_dir is None:
        project_dir = Path.cwd()

    # 按优先级从低到高扫描（低层先入，高层覆盖）
    md_layers: list[tuple[str, Path]] = [
        ("builtin", builtin_dir()),
        ("global", GLOBAL_DIR),
    ]

    skills_dir = project_dir / SKILLS_DIR_NAME
    project_crew_dir = project_dir / PROJECT_DIR_NAME

    employees: dict[str, Employee] = {}
    trigger_map: dict[str, str] = {}  # trigger -> employee name
    conflicts: list[dict[str, Any]] = []

    # 先扫描 .md 格式的层（builtin, global）
    for layer_name, layer_dir in md_layers:
        for emp in _scan_directory(layer_dir, layer_name):
            _merge_employee(emp, layer_name, employees, trigger_map, conflicts)

    # 扫描 .claude/skills/ 层
    for emp in _scan_skills_directory(skills_dir):
        _merge_employee(emp, "skill", employees, trigger_map, conflicts)

    # 扫描项目 .crew/ 层（最高优先级）
    for emp in _scan_directory(project_crew_dir, "project"):
        _merge_employee(emp, "project", employees, trigger_map, conflicts)

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
