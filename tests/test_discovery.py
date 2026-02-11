"""测试三层发现机制."""

import tempfile
from pathlib import Path

from crew.discovery import discover_employees, get_employee
from crew.employees import builtin_dir


class TestDiscovery:
    """测试员工发现."""

    def test_discover_builtin(self):
        """应能发现所有内置员工."""
        result = discover_employees(project_dir=Path("/nonexistent"))
        assert len(result.employees) >= 5
        assert "code-reviewer" in result.employees
        assert "test-engineer" in result.employees
        assert "doc-writer" in result.employees
        assert "refactor-guide" in result.employees
        assert "pr-creator" in result.employees

    def test_builtin_source_layer(self):
        """内置员工的 source_layer 应为 builtin 或 global（全局层可覆盖）."""
        result = discover_employees(project_dir=Path("/nonexistent"))
        builtin_names = ["code-reviewer", "test-engineer", "doc-writer", "refactor-guide", "pr-creator"]
        for name in builtin_names:
            assert result.employees[name].source_layer in ("builtin", "global")

    def test_project_overrides_builtin(self):
        """项目层员工应覆盖同名内置员工."""
        with tempfile.TemporaryDirectory() as tmpdir:
            crew_dir = Path(tmpdir) / ".crew"
            crew_dir.mkdir()
            (crew_dir / "code-reviewer.md").write_text(
                """---
name: code-reviewer
description: 自定义审查员
---

自定义内容。
""",
                encoding="utf-8",
            )

            result = discover_employees(project_dir=Path(tmpdir))
            emp = result.employees["code-reviewer"]
            assert emp.source_layer == "project"
            assert emp.description == "自定义审查员"
            assert len(result.conflicts) >= 1

    def test_get_employee_by_name(self):
        """应能按名称查找员工."""
        emp = get_employee("code-reviewer", project_dir=Path("/nonexistent"))
        assert emp is not None
        assert emp.name == "code-reviewer"

    def test_get_employee_by_trigger(self):
        """应能按触发别名查找员工."""
        emp = get_employee("review", project_dir=Path("/nonexistent"))
        assert emp is not None
        assert emp.name == "code-reviewer"

    def test_get_employee_not_found(self):
        """查找不存在的员工应返回 None."""
        emp = get_employee("nonexistent", project_dir=Path("/nonexistent"))
        assert emp is None

    def test_builtin_dir_exists(self):
        """内置目录应存在且包含 .md 文件."""
        d = builtin_dir()
        assert d.is_dir()
        md_files = list(d.glob("*.md"))
        assert len(md_files) >= 5
