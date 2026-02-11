"""测试 Skills 发现层."""

import tempfile
from pathlib import Path

from crew.discovery import discover_employees, _scan_skills_directory


class TestSkillDiscovery:
    """测试 .claude/skills/ 发现层."""

    def test_scan_empty_dir(self):
        """空目录应返回空列表."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assert _scan_skills_directory(Path(tmpdir)) == []

    def test_scan_nonexistent_dir(self):
        """不存在的目录应返回空列表."""
        assert _scan_skills_directory(Path("/nonexistent/skills")) == []

    def test_scan_valid_skill(self):
        """应正确扫描 <name>/SKILL.md 结构."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)
            skill_dir = skills_dir / "my-scanner"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("""---
name: my-scanner
description: 扫描测试
allowed-tools: Read Grep
---

扫描内容。
""", encoding="utf-8")

            employees = _scan_skills_directory(skills_dir)
            assert len(employees) == 1
            assert employees[0].name == "my-scanner"
            assert employees[0].source_layer == "skill"
            assert employees[0].tools == ["file_read", "grep"]

    def test_skill_layer_in_discovery(self):
        """discover_employees 应包含 skill 层."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            skills_dir = project_dir / ".claude" / "skills" / "skill-test"
            skills_dir.mkdir(parents=True)
            (skills_dir / "SKILL.md").write_text("""---
name: skill-test
description: 发现层测试
---

技能正文。
""", encoding="utf-8")

            result = discover_employees(project_dir=project_dir)
            assert "skill-test" in result.employees
            assert result.employees["skill-test"].source_layer == "skill"

    def test_project_overrides_skill(self):
        """项目层 (.crew/) 应覆盖技能层 (.claude/skills/)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            # 创建 skill 层
            skill_dir = project_dir / ".claude" / "skills" / "my-emp"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("""---
name: my-emp
description: skill 层版本
---

技能版本。
""", encoding="utf-8")

            # 创建 project 层
            crew_dir = project_dir / ".crew"
            crew_dir.mkdir()
            (crew_dir / "my-emp.md").write_text("""---
name: my-emp
description: project 层版本
---

项目版本。
""", encoding="utf-8")

            result = discover_employees(project_dir=project_dir)
            emp = result.employees["my-emp"]
            assert emp.source_layer == "project"
            assert emp.description == "project 层版本"
            assert len(result.conflicts) >= 1

    def test_skip_invalid_skill(self):
        """无效的 SKILL.md 应被跳过."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)
            bad_dir = skills_dir / "bad-skill"
            bad_dir.mkdir()
            (bad_dir / "SKILL.md").write_text("not valid yaml", encoding="utf-8")

            employees = _scan_skills_directory(skills_dir)
            assert len(employees) == 0

    def test_skip_dir_without_skill_md(self):
        """没有 SKILL.md 的子目录应被跳过."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)
            (skills_dir / "just-a-dir").mkdir()
            (skills_dir / "another").mkdir()
            (skills_dir / "another" / "README.md").write_text("not a skill")

            employees = _scan_skills_directory(skills_dir)
            assert len(employees) == 0
