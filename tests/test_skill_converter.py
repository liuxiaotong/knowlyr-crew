"""测试 SKILL.md 双向转换."""

import tempfile
from pathlib import Path

from crew.models import SKILL_TO_TOOL, TOOL_TO_SKILL, Employee, EmployeeArg
from crew.parser import (
    _parse_allowed_tools,
    _parse_argument_hint,
    parse_skill,
    parse_skill_string,
)
from crew.skill_converter import (
    _tools_to_allowed_tools,
    employee_to_skill,
    export_all,
    export_employee,
    sync_skills,
    write_skill,
)


class TestToolMapping:
    """测试工具名映射."""

    def test_tool_to_skill_mapping(self):
        """TOOL_TO_SKILL 映射应覆盖所有常用工具."""
        assert TOOL_TO_SKILL["file_read"] == "Read"
        assert TOOL_TO_SKILL["file_write"] == "Write"
        assert TOOL_TO_SKILL["git"] == "Bash(git:*)"
        assert TOOL_TO_SKILL["bash"] == "Bash"
        assert TOOL_TO_SKILL["grep"] == "Grep"
        assert TOOL_TO_SKILL["glob"] == "Glob"

    def test_skill_to_tool_reverse(self):
        """SKILL_TO_TOOL 应为 TOOL_TO_SKILL 的反向映射."""
        for crew_name, skill_name in TOOL_TO_SKILL.items():
            assert SKILL_TO_TOOL[skill_name] == crew_name

    def test_tools_to_allowed_tools_string(self):
        """tools 列表应正确转为 allowed-tools 字符串."""
        assert _tools_to_allowed_tools(["file_read", "git"]) == "Read Bash(git:*)"
        assert _tools_to_allowed_tools([]) == ""
        assert _tools_to_allowed_tools(["bash"]) == "Bash"

    def test_tools_to_allowed_tools_dedup(self):
        """重复工具应去重."""
        result = _tools_to_allowed_tools(["file_read", "file_read"])
        assert result == "Read"

    def test_unknown_tool_passthrough(self):
        """未知工具名应直接透传."""
        result = _tools_to_allowed_tools(["file_read", "custom_tool"])
        assert result == "Read custom_tool"


class TestParseAllowedTools:
    """测试 allowed-tools 解析."""

    def test_simple_tools(self):
        """应解析简单空格分隔的工具."""
        result = _parse_allowed_tools("Read Grep Glob")
        assert result == ["file_read", "grep", "glob"]

    def test_parenthesized_tool(self):
        """应正确解析带括号的工具如 Bash(git:*)."""
        result = _parse_allowed_tools("Read Bash(git:*) Grep")
        assert result == ["file_read", "git", "grep"]

    def test_bare_bash(self):
        """应正确映射不带括号的 Bash."""
        result = _parse_allowed_tools("Bash")
        assert result == ["bash"]

    def test_empty_string(self):
        """空字符串应返回空列表."""
        assert _parse_allowed_tools("") == []
        assert _parse_allowed_tools("  ") == []

    def test_unknown_skill_passthrough(self):
        """未知的技能名应直接保留."""
        result = _parse_allowed_tools("Read CustomTool")
        assert result == ["file_read", "CustomTool"]


class TestParseArgumentHint:
    """测试 argument-hint 解析."""

    def test_required_arg(self):
        """<name> 应解析为必填参数."""
        args = _parse_argument_hint("<target>")
        assert len(args) == 1
        assert args[0].name == "target"
        assert args[0].required is True

    def test_optional_arg(self):
        """[name] 应解析为可选参数."""
        args = _parse_argument_hint("[mode]")
        assert len(args) == 1
        assert args[0].name == "mode"
        assert args[0].required is False

    def test_mixed_args(self):
        """混合必填和可选参数."""
        args = _parse_argument_hint("<target> [focus] [mode]")
        assert len(args) == 3
        assert args[0].name == "target"
        assert args[0].required is True
        assert args[1].name == "focus"
        assert args[1].required is False
        assert args[2].name == "mode"
        assert args[2].required is False

    def test_empty_hint(self):
        """空 hint 应返回空列表."""
        assert _parse_argument_hint("") == []
        assert _parse_argument_hint("  ") == []


class TestParseSkillString:
    """测试 SKILL.md 解析."""

    def test_basic_skill(self):
        """应解析基本 SKILL.md."""
        content = """---
name: code-reviewer
description: 审查代码变更
allowed-tools: Read Bash(git:*)
argument-hint: <target> [focus]
---

审查 $0 的代码变更，关注 $1。
"""
        emp = parse_skill_string(content)
        assert emp.name == "code-reviewer"
        assert emp.description == "审查代码变更"
        assert emp.tools == ["file_read", "git"]
        assert len(emp.args) == 2
        assert emp.args[0].name == "target"
        assert emp.args[0].required is True
        assert emp.args[1].name == "focus"
        assert emp.args[1].required is False
        assert emp.source_layer == "skill"
        # 位置变量应转为命名变量
        assert "$target" in emp.body
        assert "$focus" in emp.body

    def test_skill_with_metadata_comment(self):
        """应从 HTML 注释提取 Crew 元数据."""
        content = '''---
name: test-skill
description: 测试技能
---

<!-- knowlyr-crew metadata {"display_name": "测试员", "tags": ["test"], "triggers": ["t"]} -->

正文内容。
'''
        emp = parse_skill_string(content)
        assert emp.display_name == "测试员"
        assert emp.tags == ["test"]
        assert emp.triggers == ["t"]

    def test_skill_name_from_param(self):
        """skill_name 参数应覆盖 frontmatter 中的 name."""
        content = """---
description: 测试
---

正文。
"""
        emp = parse_skill_string(content, skill_name="my-skill")
        assert emp.name == "my-skill"

    def test_skill_missing_description(self):
        """缺少 description 应报错."""
        content = """---
name: bad-skill
---

正文。
"""
        import pytest
        with pytest.raises(ValueError, match="description"):
            parse_skill_string(content)

    def test_skill_no_tools(self):
        """没有 allowed-tools 应返回空列表."""
        content = """---
name: simple-skill
description: 简单技能
---

正文内容。
"""
        emp = parse_skill_string(content)
        assert emp.tools == []
        assert emp.args == []


class TestParseSkillFile:
    """测试 parse_skill 文件解析."""

    def test_parse_skill_file(self):
        """应从文件解析 SKILL.md，name 从目录名推断."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "my-skill"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text("""---
description: 从文件解析
allowed-tools: Read
---

文件内容。
""", encoding="utf-8")

            emp = parse_skill(skill_file)
            assert emp.name == "my-skill"
            assert emp.description == "从文件解析"
            assert emp.source_path == skill_file


class TestEmployeeToSkill:
    """测试 Employee -> SKILL.md 转换."""

    def _make_employee(self, **kwargs):
        defaults = {
            "name": "test-emp",
            "description": "测试员工",
            "body": "审查 $target 的变更。",
            "tools": ["file_read", "git"],
            "args": [
                EmployeeArg(name="target", description="审查目标", required=True),
                EmployeeArg(name="focus", description="关注点", required=False),
            ],
        }
        defaults.update(kwargs)
        return Employee(**defaults)

    def test_basic_conversion(self):
        """应生成合法的 SKILL.md 格式."""
        emp = self._make_employee()
        result = employee_to_skill(emp)
        assert "---" in result
        assert "name: test-emp" in result
        assert "description: 测试员工" in result
        assert "allowed-tools: Read Bash(git:*)" in result
        assert "argument-hint: <target> [focus]" in result

    def test_named_to_positional_vars(self):
        """命名变量应转为位置变量."""
        emp = self._make_employee()
        result = employee_to_skill(emp)
        assert "$0" in result
        assert "$target" not in result

    def test_metadata_comment(self):
        """display_name/tags/triggers 应保存在 HTML 注释中."""
        emp = self._make_employee(
            display_name="测试员",
            tags=["test", "demo"],
            triggers=["t"],
        )
        result = employee_to_skill(emp)
        assert "<!-- knowlyr-crew metadata" in result
        assert '"display_name": "测试员"' in result
        assert '"test"' in result
        assert '"demo"' in result
        assert '"triggers"' in result

    def test_no_metadata_comment_when_empty(self):
        """无额外元数据时不应有 HTML 注释."""
        emp = self._make_employee(display_name="", tags=[], triggers=[])
        result = employee_to_skill(emp)
        assert "knowlyr-crew metadata" not in result

    def test_no_tools_no_allowed_tools(self):
        """没有 tools 时不应有 allowed-tools 行."""
        emp = self._make_employee(tools=[])
        result = employee_to_skill(emp)
        assert "allowed-tools" not in result

    def test_no_args_no_argument_hint(self):
        """没有 args 时不应有 argument-hint 行."""
        emp = self._make_employee(args=[])
        result = employee_to_skill(emp)
        assert "argument-hint" not in result

    def test_model_in_frontmatter(self):
        """model 应出现在 frontmatter 中（而非仅在 metadata 注释）."""
        emp = self._make_employee(model="claude-opus-4-6")
        result = employee_to_skill(emp)
        lines = result.split("\n")
        # frontmatter 在两个 --- 之间
        in_frontmatter = False
        found = False
        for line in lines:
            if line.strip() == "---":
                in_frontmatter = not in_frontmatter
                continue
            if in_frontmatter and line.startswith("model:"):
                found = True
                assert "claude-opus-4-6" in line
        assert found, "model 未出现在 frontmatter 中"

    def test_no_model_no_frontmatter_line(self):
        """model 为空时不应有 model 行."""
        emp = self._make_employee(model="")
        result = employee_to_skill(emp)
        lines = result.split("\n")
        in_frontmatter = False
        for line in lines:
            if line.strip() == "---":
                in_frontmatter = not in_frontmatter
                continue
            if in_frontmatter:
                assert not line.startswith("model:"), "空 model 不应出现在 frontmatter"


class TestWriteAndExport:
    """测试文件写入和导出."""

    def _make_employee(self):
        return Employee(
            name="test-skill",
            description="测试技能",
            body="正文内容。",
        )

    def test_write_skill(self):
        """应写入 <dir>/<name>/SKILL.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir)
            emp = self._make_employee()
            path = write_skill(emp, skills_dir)

            assert path == skills_dir / "test-skill" / "SKILL.md"
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "name: test-skill" in content

    def test_export_employee(self):
        """应导出到 .claude/skills/<name>/SKILL.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            emp = self._make_employee()
            path = export_employee(emp, project_dir)

            expected = project_dir / ".claude" / "skills" / "test-skill" / "SKILL.md"
            assert path == expected
            assert path.exists()

    def test_export_all(self):
        """应批量导出所有员工."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            emps = [
                Employee(name="a", description="A", body="A body."),
                Employee(name="b", description="B", body="B body."),
            ]
            paths = export_all(emps, project_dir)
            assert len(paths) == 2
            assert all(p.exists() for p in paths)

    def test_sync_clean(self):
        """sync --clean 应删除孤儿目录."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            skills_dir = project_dir / ".claude" / "skills"

            # 先创建一个"孤儿"
            orphan_dir = skills_dir / "orphan"
            orphan_dir.mkdir(parents=True)
            (orphan_dir / "SKILL.md").write_text("---\nname: orphan\ndescription: x\n---\nhi\n")

            emp = self._make_employee()
            report = sync_skills([emp], project_dir, clean=True)

            assert len(report["exported"]) == 1
            assert len(report["removed"]) == 1
            assert not (orphan_dir / "SKILL.md").exists()

    def test_sync_no_clean(self):
        """sync 不带 --clean 不应删除孤儿."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            skills_dir = project_dir / ".claude" / "skills"

            orphan_dir = skills_dir / "orphan"
            orphan_dir.mkdir(parents=True)
            (orphan_dir / "SKILL.md").write_text("---\nname: orphan\ndescription: x\n---\nhi\n")

            emp = self._make_employee()
            report = sync_skills([emp], project_dir, clean=False)

            assert len(report["removed"]) == 0
            assert (orphan_dir / "SKILL.md").exists()


class TestRoundTrip:
    """测试 Employee -> SKILL.md -> Employee 往返转换."""

    def test_round_trip(self):
        """Employee -> SKILL.md -> Employee 核心字段应一致."""
        original = Employee(
            name="round-trip",
            display_name="往返测试",
            description="测试往返转换",
            tags=["test"],
            triggers=["rt"],
            tools=["file_read", "git", "bash"],
            args=[
                EmployeeArg(name="target", description="目标", required=True),
                EmployeeArg(name="mode", description="模式", required=False),
            ],
            body="请审查 $target，模式 $mode。",
        )

        # Employee -> SKILL.md
        skill_content = employee_to_skill(original)

        # SKILL.md -> Employee
        restored = parse_skill_string(skill_content, skill_name="round-trip")

        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.display_name == original.display_name
        assert restored.tags == original.tags
        assert restored.triggers == original.triggers
        assert restored.tools == original.tools
        assert len(restored.args) == len(original.args)
        assert restored.args[0].name == "target"
        assert restored.args[0].required is True
        assert restored.args[1].name == "mode"
        assert restored.args[1].required is False
        # 正文中命名变量应恢复
        assert "$target" in restored.body
        assert "$mode" in restored.body

    def test_round_trip_model(self):
        """model 应在 Employee -> SKILL.md -> Employee 往返中保留."""
        original = Employee(
            name="model-rt",
            description="测试 model 往返",
            model="claude-opus-4-6",
            body="你好。",
        )
        skill_content = employee_to_skill(original)
        assert "model: claude-opus-4-6" in skill_content

        restored = parse_skill_string(skill_content, skill_name="model-rt")
        assert restored.model == "claude-opus-4-6"
