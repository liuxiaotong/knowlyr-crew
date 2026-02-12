"""测试 EMPLOYEE.md / 目录格式 解析器."""

import shutil
from pathlib import Path

import pytest

from crew.parser import parse_employee, parse_employee_dir, parse_employee_string, validate_employee

FIXTURES = Path(__file__).parent / "fixtures"


class TestParseEmployee:
    """测试文件解析."""

    def test_parse_valid_file(self):
        emp = parse_employee(FIXTURES / "valid_employee.md")
        assert emp.name == "test-worker"
        assert emp.display_name == "测试员工"
        assert emp.description == "用于测试的员工定义"
        assert emp.tags == ["test"]
        assert emp.triggers == ["tw"]
        assert len(emp.args) == 2
        assert emp.args[0].name == "target"
        assert emp.args[0].required is True
        assert emp.args[1].name == "mode"
        assert emp.args[1].default == "normal"
        assert emp.output.format == "markdown"
        assert "$target" in emp.body
        assert emp.source_path == FIXTURES / "valid_employee.md"

    def test_parse_invalid_missing_description(self):
        with pytest.raises(ValueError, match="description"):
            parse_employee(FIXTURES / "invalid_employee.md")

    def test_parse_string_minimal(self):
        content = """---
name: minimal
description: 最小员工
---

做点什么。
"""
        emp = parse_employee_string(content)
        assert emp.name == "minimal"
        assert emp.description == "最小员工"
        assert emp.body == "做点什么。"
        assert emp.args == []
        assert emp.triggers == []
        assert emp.display_name == ""
        assert emp.effective_display_name == "minimal"

    def test_parse_no_frontmatter(self):
        with pytest.raises(ValueError, match="frontmatter"):
            parse_employee_string("这是纯文本，没有 frontmatter。")

    def test_parse_empty_body(self):
        content = """---
name: empty-body
description: 空正文
---
"""
        with pytest.raises(ValueError, match="正文"):
            parse_employee_string(content)

    def test_parse_source_layer(self):
        content = """---
name: layered
description: 测试来源层
---

内容。
"""
        emp = parse_employee_string(content, source_layer="project")
        assert emp.source_layer == "project"


class TestValidateEmployee:
    """测试员工校验."""

    def test_valid_employee(self):
        emp = parse_employee(FIXTURES / "valid_employee.md")
        errors = validate_employee(emp)
        assert errors == []

    def test_invalid_name_format(self):
        content = """---
name: Invalid_Name
description: 测试
---

内容。
"""
        emp = parse_employee_string(content)
        errors = validate_employee(emp)
        assert any("格式无效" in e for e in errors)

    def test_name_too_long(self):
        content = f"""---
name: {"a" * 65}
description: 测试
---

内容。
"""
        emp = parse_employee_string(content)
        errors = validate_employee(emp)
        assert any("64" in e for e in errors)

    def test_duplicate_arg_names(self):
        content = """---
name: dup-args
description: 测试
args:
  - name: target
  - name: target
---

内容。
"""
        emp = parse_employee_string(content)
        errors = validate_employee(emp)
        assert any("重复" in e for e in errors)

    def test_agent_id_parsed(self):
        """应解析 frontmatter 中的 agent_id."""
        content = "---\nname: test\ndescription: 测试\nagent_id: 3060\n---\n\n内容。\n"
        emp = parse_employee_string(content)
        assert emp.agent_id == 3060

    def test_agent_id_none_by_default(self):
        """未设置 agent_id 时应为 None."""
        content = "---\nname: test\ndescription: 测试\n---\n\n内容。\n"
        emp = parse_employee_string(content)
        assert emp.agent_id is None


class TestParseEmployeeDir:
    """测试目录格式解析."""

    def test_parse_valid_dir(self):
        emp = parse_employee_dir(FIXTURES / "valid_employee_dir")
        assert emp.name == "dir-worker"
        assert emp.display_name == "目录员工"
        assert emp.character_name == "小目"
        assert emp.description == "目录格式测试员工"
        assert emp.tags == ["test", "dir-format"]
        assert emp.triggers == ["dw"]
        assert len(emp.args) == 2
        assert emp.args[0].name == "target"
        assert emp.args[0].required is True
        assert emp.args[1].name == "mode"
        assert emp.args[1].default == "normal"
        assert emp.output.format == "markdown"
        assert "$target" in emp.body
        assert emp.source_path == FIXTURES / "valid_employee_dir"

    def test_parse_dir_with_extras(self):
        """带 workflows/ 和 adaptors/ 的目录应拼接到 body."""
        emp = parse_employee_dir(FIXTURES / "employee_with_extras")
        assert emp.name == "extras-worker"
        assert "核心提示词" in emp.body
        assert "分析工作流" in emp.body
        assert "Python 适配" in emp.body

    def test_parse_dir_body_order(self):
        """body 应按 prompt.md → workflows → adaptors 顺序拼接."""
        emp = parse_employee_dir(FIXTURES / "employee_with_extras")
        prompt_pos = emp.body.index("核心提示词")
        workflow_pos = emp.body.index("分析工作流")
        adaptor_pos = emp.body.index("Python 适配")
        assert prompt_pos < workflow_pos < adaptor_pos

    def test_parse_dir_source_layer(self):
        emp = parse_employee_dir(FIXTURES / "valid_employee_dir", source_layer="global")
        assert emp.source_layer == "global"

    def test_parse_dir_missing_yaml(self, tmp_path):
        """缺少 employee.yaml 应报错."""
        emp_dir = tmp_path / "broken"
        emp_dir.mkdir()
        (emp_dir / "prompt.md").write_text("内容")
        with pytest.raises(ValueError, match="employee.yaml"):
            parse_employee_dir(emp_dir)

    def test_parse_dir_missing_prompt(self, tmp_path):
        """缺少 prompt.md 应报错."""
        emp_dir = tmp_path / "broken"
        emp_dir.mkdir()
        (emp_dir / "employee.yaml").write_text("name: x\ndescription: y\n")
        with pytest.raises(ValueError, match="prompt.md"):
            parse_employee_dir(emp_dir)

    def test_parse_dir_missing_name(self, tmp_path):
        """employee.yaml 缺少 name 应报错."""
        emp_dir = tmp_path / "broken"
        emp_dir.mkdir()
        (emp_dir / "employee.yaml").write_text("description: y\n")
        (emp_dir / "prompt.md").write_text("内容")
        with pytest.raises(ValueError, match="name"):
            parse_employee_dir(emp_dir)

    def test_agent_id_none_by_default(self):
        """未设置 agent_id 时应为 None."""
        emp = parse_employee_dir(FIXTURES / "valid_employee_dir")
        assert emp.agent_id is None

    def test_agent_id_parsed(self, tmp_path):
        """应解析 employee.yaml 中的 agent_id."""
        emp_dir = tmp_path / "with-agent"
        emp_dir.mkdir()
        (emp_dir / "employee.yaml").write_text(
            "name: test-agent\ndescription: test\nagent_id: 3050\n"
        )
        (emp_dir / "prompt.md").write_text("内容")
        emp = parse_employee_dir(emp_dir)
        assert emp.agent_id == 3050

    def test_validate_dir_employee(self):
        """目录解析的员工应能通过校验."""
        emp = parse_employee_dir(FIXTURES / "valid_employee_dir")
        errors = validate_employee(emp)
        assert errors == []
