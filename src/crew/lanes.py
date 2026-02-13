"""Lane-based 串行调度工具."""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

from crew.paths import resolve_project_dir

try:  # pragma: no cover - only used on POSIX
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore


class LaneLock:
    """基于文件锁的 Lane."""

    def __init__(self, lane_name: str, root: Path | None = None, *, project_dir: Path | None = None):
        sanitized = lane_name.replace("/", "_").replace(":", "_")
        base = root if root is not None else resolve_project_dir(project_dir) / ".crew" / "lanes"
        self.path = base / f"{sanitized}.lock"
        self._fh: "Any | None" = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")
        if fcntl:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(f"pid={os.getpid()}\n")
        fh.flush()
        self._fh = fh

    def release(self) -> None:
        if self._fh is None:
            return
        if fcntl:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        self._fh.close()
        self._fh = None

    def __enter__(self) -> "LaneLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


@contextmanager
def lane_lock(lane_name: str, enabled: bool = True, *, project_dir: Path | None = None) -> Iterator[None]:
    """按 lane_name 串行化执行."""
    if not enabled:
        with nullcontext():
            yield
        return
    lock = LaneLock(lane_name, project_dir=project_dir)
    with lock:
        yield


__all__ = ["lane_lock", "LaneLock"]
