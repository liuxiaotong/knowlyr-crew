"""Path helpers for crew resources."""

from __future__ import annotations

import os
from pathlib import Path

_GLOBAL_ENV_VAR = "KNOWLYR_CREW_GLOBAL_DIR"


def _default_global_dir() -> Path:
    return Path.cwd() / ".crew" / "global"


def get_global_dir() -> Path:
    """Return the directory that stores global-level crew resources."""
    configured = os.environ.get(_GLOBAL_ENV_VAR)
    if configured:
        return Path(configured).expanduser()
    return _default_global_dir()


def get_global_templates_dir() -> Path:
    return get_global_dir() / "templates"


def get_global_discussions_dir() -> Path:
    return get_global_dir() / "discussions"


__all__ = [
    "get_global_dir",
    "get_global_templates_dir",
    "get_global_discussions_dir",
]
