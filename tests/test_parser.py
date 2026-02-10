"""测试 EMPLOYEE.md 解析器."""

from pathlib import Path

import pytest

from crew.parser import parse_employee, parse_employee_string, validate_employee

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
