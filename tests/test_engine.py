"""测试执行引擎."""

from pathlib import Path

from crew.engine import CrewEngine
from crew.parser import parse_employee

FIXTURES = Path(__file__).parent / "fixtures"


class TestCrewEngine:
    """测试 CrewEngine."""

    def setup_method(self):
        self.engine = CrewEngine()
        self.employee = parse_employee(FIXTURES / "valid_employee.md")

    def test_render_named_args(self):
        result = self.engine.render(
            self.employee,
            args={"target": "src/main.py", "mode": "strict"},
        )
        assert "src/main.py" in result
        assert "strict" in result
        assert "$target" not in result
        assert "$mode" not in result

    def test_render_default_args(self):
        result = self.engine.render(
            self.employee,
            args={"target": "src/main.py"},
        )
        assert "src/main.py" in result
        assert "normal" in result  # mode 的默认值

    def test_render_positional_args(self):
        result = self.engine.render(
            self.employee,
            positional=["file1.py", "file2.py"],
        )
        assert "file1.py" in result  # $1
        assert "file2.py" in result  # $2

    def test_render_arguments_variable(self):
        result = self.engine.render(
            self.employee,
            positional=["file1.py", "file2.py"],
        )
        # $ARGUMENTS 应被替换为所有位置参数
        assert "$ARGUMENTS" not in result

    def test_render_env_variables(self):
        result = self.engine.render(
            self.employee,
            args={"target": "test"},
        )
        # {date} 应被替换为当前日期
        assert "{date}" not in result or "20" in result  # 日期格式含 20xx

    def test_validate_args_missing_required(self):
        errors = self.engine.validate_args(self.employee, args={})
        assert any("target" in e for e in errors)

    def test_validate_args_ok(self):
        errors = self.engine.validate_args(
            self.employee,
            args={"target": "test"},
        )
        assert errors == []

    def test_prompt_contains_role(self):
        result = self.engine.prompt(
            self.employee,
            args={"target": "test"},
        )
        assert "测试员工" in result  # display_name
        assert "用于测试的员工定义" in result  # description
        assert "test" in result  # 标签

    def test_prompt_output_constraints(self):
        result = self.engine.prompt(
            self.employee,
            args={"target": "test"},
        )
        # output.filename 非空，应有输出约束部分
        assert "输出约束" in result
        assert "markdown" in result
