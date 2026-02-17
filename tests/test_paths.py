"""路径辅助函数测试 — paths.py."""

import os
from pathlib import Path

import pytest

from crew.paths import (
    file_lock,
    get_global_dir,
    get_global_discussions_dir,
    get_global_templates_dir,
    resolve_project_dir,
)


class TestResolveProjectDir:
    def test_none_returns_cwd(self):
        result = resolve_project_dir(None)
        assert result == Path.cwd()

    def test_path_passthrough(self, tmp_path):
        result = resolve_project_dir(tmp_path)
        assert result == tmp_path

    def test_string_converted_to_path(self):
        result = resolve_project_dir(Path("/tmp/test"))
        assert isinstance(result, Path)
        assert str(result) == "/tmp/test"


class TestGetGlobalDir:
    def test_default_location(self, tmp_path):
        result = get_global_dir(tmp_path)
        assert result == tmp_path / "private" / "employees"

    def test_env_override(self, tmp_path, monkeypatch):
        custom_dir = tmp_path / "custom-global"
        monkeypatch.setenv("KNOWLYR_CREW_GLOBAL_DIR", str(custom_dir))
        result = get_global_dir(tmp_path)
        assert result == custom_dir

    def test_env_override_with_tilde(self, monkeypatch):
        monkeypatch.setenv("KNOWLYR_CREW_GLOBAL_DIR", "~/crew-global")
        result = get_global_dir()
        assert "~" not in str(result)  # expanduser 展开了


class TestGetGlobalTemplatesDir:
    def test_location(self, tmp_path):
        result = get_global_templates_dir(tmp_path)
        assert result == tmp_path / "private" / "employees" / "templates"


class TestGetGlobalDiscussionsDir:
    def test_location(self, tmp_path):
        result = get_global_discussions_dir(tmp_path)
        assert result == tmp_path / "private" / "employees" / "discussions"


class TestFileLock:
    def test_creates_lock_file(self, tmp_path):
        target = tmp_path / "data.json"
        target.write_text("{}", encoding="utf-8")
        lock_file = tmp_path / "data.json.lock"

        with file_lock(target):
            assert lock_file.exists()

    def test_creates_parent_dirs(self, tmp_path):
        target = tmp_path / "nested" / "deep" / "data.json"
        with file_lock(target):
            assert target.parent.exists()

    def test_reentrant_same_process(self, tmp_path):
        """同进程嵌套不同文件不死锁."""
        target_a = tmp_path / "a.json"
        target_b = tmp_path / "b.json"
        target_a.write_text("{}", encoding="utf-8")
        target_b.write_text("{}", encoding="utf-8")
        with file_lock(target_a):
            with file_lock(target_b):
                pass

    def test_body_executes(self, tmp_path):
        """锁内代码正常执行."""
        target = tmp_path / "data.json"
        target.write_text("{}", encoding="utf-8")
        executed = False
        with file_lock(target):
            executed = True
        assert executed is True
