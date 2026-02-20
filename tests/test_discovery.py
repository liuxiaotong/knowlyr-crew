"""测试四层发现机制（含目录格式支持）."""

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
        builtin_names = [
            "code-reviewer",
            "test-engineer",
            "doc-writer",
            "refactor-guide",
            "pr-creator",
        ]
        for name in builtin_names:
            assert result.employees[name].source_layer == "builtin"

    def test_private_overrides_builtin(self):
        """private 层员工应覆盖同名内置员工."""
        with tempfile.TemporaryDirectory() as tmpdir:
            crew_dir = Path(tmpdir) / "private" / "employees"
            crew_dir.mkdir(parents=True)
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
            assert emp.source_layer == "private"
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
        assert len(md_files) >= 1

    def test_discover_dir_format_in_private(self):
        """private 层应能发现目录格式的员工."""
        with tempfile.TemporaryDirectory() as tmpdir:
            crew_dir = Path(tmpdir) / "private" / "employees" / "my-worker"
            crew_dir.mkdir(parents=True)
            (crew_dir / "employee.yaml").write_text(
                "name: my-worker\ndescription: 目录格式员工\nversion: '1.0'\n",
                encoding="utf-8",
            )
            (crew_dir / "prompt.md").write_text(
                "# 测试\n\n这是目录格式员工。\n",
                encoding="utf-8",
            )

            result = discover_employees(project_dir=Path(tmpdir))
            assert "my-worker" in result.employees
            emp = result.employees["my-worker"]
            assert emp.source_layer == "private"
            assert emp.source_path == crew_dir

    def test_dir_format_overrides_md_format(self):
        """private 层同名时目录格式应覆盖文件格式."""
        with tempfile.TemporaryDirectory() as tmpdir:
            crew_dir = Path(tmpdir) / "private" / "employees"
            crew_dir.mkdir(parents=True)

            # 文件格式
            (crew_dir / "dup-worker.md").write_text(
                "---\nname: dup-worker\ndescription: 文件版\n---\n文件内容。\n",
                encoding="utf-8",
            )

            # 目录格式（同名）
            dir_emp = crew_dir / "dup-worker"
            dir_emp.mkdir()
            (dir_emp / "employee.yaml").write_text(
                "name: dup-worker\ndescription: 目录版\nversion: '1.0'\n",
                encoding="utf-8",
            )
            (dir_emp / "prompt.md").write_text("目录内容。\n", encoding="utf-8")

            result = discover_employees(project_dir=Path(tmpdir))
            emp = result.employees["dup-worker"]
            assert emp.description == "目录版"
            assert emp.source_path == dir_emp

    def test_mixed_formats_in_same_layer(self):
        """private 层中目录格式和文件格式员工应共存."""
        with tempfile.TemporaryDirectory() as tmpdir:
            crew_dir = Path(tmpdir) / "private" / "employees"
            crew_dir.mkdir(parents=True)

            # 文件格式员工
            (crew_dir / "file-worker.md").write_text(
                "---\nname: file-worker\ndescription: 文件员工\n---\n内容。\n",
                encoding="utf-8",
            )

            # 目录格式员工
            dir_emp = crew_dir / "dir-worker"
            dir_emp.mkdir()
            (dir_emp / "employee.yaml").write_text(
                "name: dir-worker\ndescription: 目录员工\nversion: '1.0'\n",
                encoding="utf-8",
            )
            (dir_emp / "prompt.md").write_text("目录内容。\n", encoding="utf-8")

            result = discover_employees(project_dir=Path(tmpdir))
            assert "file-worker" in result.employees
            assert "dir-worker" in result.employees


class TestDiscoveryCacheThreadSafety:
    """发现缓存线程安全."""

    def test_concurrent_discover(self, tmp_path):
        """多线程同时调用 discover_employees 不出错."""
        import threading

        results: list = []
        errors: list = []

        def _discover():
            try:
                r = discover_employees(project_dir=tmp_path, cache_ttl=0.5)
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_discover) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        assert len(results) == 4
