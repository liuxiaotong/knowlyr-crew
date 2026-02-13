"""Path helpers for crew resources."""

from __future__ import annotations

import os
from pathlib import Path

_GLOBAL_ENV_VAR = "KNOWLYR_CREW_GLOBAL_DIR"


def resolve_project_dir(project_dir: Path | None = None) -> Path:
    """Resolve project_dir, falling back to cwd if None."""
    return Path(project_dir) if project_dir else Path.cwd()


def _default_global_dir(project_dir: Path | None = None) -> Path:
    return resolve_project_dir(project_dir) / ".crew" / "global"


def get_global_dir(project_dir: Path | None = None) -> Path:
    """Return the directory that stores global-level crew resources."""
    configured = os.environ.get(_GLOBAL_ENV_VAR)
    if configured:
        return Path(configured).expanduser()
    return _default_global_dir(project_dir)


def get_global_templates_dir(project_dir: Path | None = None) -> Path:
    return get_global_dir(project_dir) / "templates"


def get_global_discussions_dir(project_dir: Path | None = None) -> Path:
    return get_global_dir(project_dir) / "discussions"


__all__ = [
    "resolve_project_dir",
    "get_global_dir",
    "get_global_templates_dir",
    "get_global_discussions_dir",
]
