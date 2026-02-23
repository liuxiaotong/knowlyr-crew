"""全局测试配置 — 用 fixture 员工替代已删除的内置员工."""

import os
from pathlib import Path

import pytest

_FIXTURES_EMPLOYEES = Path(__file__).parent / "fixtures" / "employees"


@pytest.fixture(autouse=True)
def _use_fixture_employees(monkeypatch):
    """所有测试自动使用 fixtures/employees/ 作为 builtin 员工目录."""
    monkeypatch.setattr("crew.employees.builtin_dir", lambda: _FIXTURES_EMPLOYEES)
    monkeypatch.setattr("crew.discovery.builtin_dir", lambda: _FIXTURES_EMPLOYEES)


@pytest.fixture(autouse=True)
def _force_sqlite_backend(monkeypatch):
    """测试环境强制使用 SQLite 后端，不依赖 PostgreSQL."""
    monkeypatch.setenv("CREW_USE_SQLITE", "1")
