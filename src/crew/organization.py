"""组织架构加载 — 从 organization.yaml 读取团队、权限、路由."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import yaml

from crew.models import Organization

logger = logging.getLogger(__name__)

_cache: Organization | None = None
_cache_time: float = 0.0
_cache_lock = threading.Lock()
_CACHE_TTL = 30.0  # seconds


def load_organization(project_dir: Path | None = None) -> Organization:
    """加载组织架构配置.

    搜索顺序：
    1. {project_dir}/private/organization.yaml
    2. {project_dir}/.crew/organization.yaml
    3. 返回空 Organization（向后兼容）
    """
    global _cache, _cache_time

    with _cache_lock:
        if _cache is not None and (time.time() - _cache_time) < _CACHE_TTL:
            return _cache

        candidates: list[Path] = []
        if project_dir:
            candidates.append(project_dir / "private" / "organization.yaml")
            candidates.append(project_dir / ".crew" / "organization.yaml")

        for path in candidates:
            if path.is_file():
                try:
                    data = yaml.safe_load(path.read_text(encoding="utf-8"))
                    org = Organization(**(data or {}))
                    logger.info("组织架构已加载: %s", path)
                    _cache = org
                    _cache_time = time.time()
                    return org
                except Exception as e:
                    logger.warning("组织架构加载失败 (%s): %s", path, e)

        org = Organization()
        _cache = org
        _cache_time = time.time()
        return org


def invalidate_cache() -> None:
    """清除缓存（测试用）."""
    global _cache, _cache_time
    with _cache_lock:
        _cache = None
        _cache_time = 0.0
