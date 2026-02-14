"""Public SDK helpers for knowlyr-crew."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from crew.context_detector import detect_project
from crew.discovery import discover_employees
from crew.engine import CrewEngine
from crew.exceptions import EmployeeNotFoundError
from crew.models import Employee

try:  # pragma: no cover - optional dependency
    from crew.id_client import fetch_agent_identity
except Exception:  # pragma: no cover - optional
    fetch_agent_identity = None  # type: ignore


def list_employees(project_dir: Path | None = None) -> list[Employee]:
    """Return all employees discovered in the current project."""
    result = discover_employees(project_dir=project_dir)
    return list(result.employees.values())


def get_employee(name_or_trigger: str, project_dir: Path | None = None) -> Employee | None:
    """Fetch an employee by name or trigger."""
    result = discover_employees(project_dir=project_dir)
    return result.get(name_or_trigger)


def generate_prompt(
    employee: Employee,
    *,
    args: dict[str, str] | None = None,
    positional: Sequence[str] | None = None,
    raw: bool = False,
    agent_identity=None,
    project_info=None,
    project_dir: Path | None = None,
) -> str:
    """Render prompt or body for a given employee."""
    engine = CrewEngine(project_dir=project_dir)
    if raw:
        return engine.render(employee, args=args, positional=list(positional or []))
    return engine.prompt(
        employee,
        args=args,
        positional=list(positional or []),
        agent_identity=agent_identity,
        project_info=project_info,
    )


def generate_prompt_by_name(
    name_or_trigger: str,
    *,
    args: dict[str, str] | None = None,
    positional: Sequence[str] | None = None,
    raw: bool = False,
    agent_id: int | None = None,
    smart_context: bool = False,
    project_dir: Path | None = None,
):
    """High level helper to render prompt by employee name or trigger."""
    employee = get_employee(name_or_trigger, project_dir=project_dir)
    if employee is None:
        raise EmployeeNotFoundError(name_or_trigger)

    agent_identity = None
    if agent_id is not None and fetch_agent_identity:
        try:
            agent_identity = fetch_agent_identity(agent_id)
        except Exception:  # pragma: no cover
            agent_identity = None

    project_info = detect_project(project_dir) if smart_context else None

    return generate_prompt(
        employee,
        args=args,
        positional=positional,
        raw=raw,
        agent_identity=agent_identity,
        project_info=project_info,
        project_dir=project_dir,
    )


def run_pipeline_steps(
    steps: Iterable[tuple[str, dict[str, str]]],
    *,
    smart_context: bool = True,
    agent_id: int | None = None,
    project_dir: Path | None = None,
) -> list[str]:
    """Helper to render a sequence of (employee, args) prompts."""
    prompts: list[str] = []
    project_info = detect_project(project_dir) if smart_context else None
    for name, step_args in steps:
        employee = get_employee(name, project_dir=project_dir)
        if employee is None:
            raise EmployeeNotFoundError(name)

        agent_identity = None
        if agent_id is not None and fetch_agent_identity:
            try:
                agent_identity = fetch_agent_identity(agent_id)
            except Exception:  # pragma: no cover
                agent_identity = None

        prompts.append(
            generate_prompt(
                employee,
                args=step_args,
                raw=False,
                agent_identity=agent_identity,
                project_info=project_info,
                project_dir=project_dir,
            )
        )
    return prompts


__all__ = [
    "list_employees",
    "get_employee",
    "generate_prompt",
    "generate_prompt_by_name",
    "run_pipeline_steps",
]
