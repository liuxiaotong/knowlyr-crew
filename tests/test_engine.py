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

    def test_prompt_with_agent_identity(self):
        """带 Agent 身份时 prompt 应包含身份信息."""
        from crew.id_client import AgentIdentity
        identity = AgentIdentity(
            agent_id=3050,
            nickname="Alice",
            title="Senior Reviewer",
            domains=["python", "security"],
            memory="偏好简洁的审查风格。",
        )
        result = self.engine.prompt(
            self.employee,
            args={"target": "test"},
            agent_identity=identity,
        )
        assert "Alice" in result
        assert "Senior Reviewer" in result
        assert "python" in result
        assert "Agent 记忆" in result
        assert "偏好简洁" in result

    def test_prompt_without_agent_identity(self):
        """不带 Agent 身份时 prompt 应与原来一致."""
        result = self.engine.prompt(
            self.employee,
            args={"target": "test"},
            agent_identity=None,
        )
        assert "**Agent**" not in result
        assert "Agent 记忆" not in result

    def test_prompt_with_project_info(self):
        """带项目信息时 prompt 应包含项目类型."""
        from crew.context_detector import ProjectInfo
        info = ProjectInfo(
            project_type="python",
            framework="fastapi",
            test_framework="pytest",
            lint_tools=["ruff", "mypy"],
            package_manager="uv",
        )
        result = self.engine.prompt(
            self.employee,
            args={"target": "test"},
            project_info=info,
        )
        assert "Python (Fastapi)" in result
        assert "pytest" in result
        assert "ruff" in result
        assert "uv" in result

    def test_prompt_without_project_info(self):
        """不带项目信息时 prompt 不应有项目类型段."""
        result = self.engine.prompt(
            self.employee,
            args={"target": "test"},
            project_info=None,
        )
        assert "**项目类型**" not in result

    def test_git_branch_failure_returns_empty(self):
        """_get_git_branch 在 subprocess 失败时返回空串."""
        from crew.engine import _get_git_branch
        from unittest.mock import patch

        with patch("crew.engine.subprocess.run", side_effect=FileNotFoundError):
            assert _get_git_branch() == ""

        with patch("crew.engine.subprocess.run", side_effect=OSError("err")):
            assert _get_git_branch() == ""

    def test_render_weekday_variable(self):
        """render() 应替换 {weekday} 为中文星期."""
        from crew.models import Employee, EmployeeArg

        emp = Employee(
            name="weekday-test",
            description="test",
            body="今天是{weekday}",
            args=[],
        )
        result = self.engine.render(emp)
        assert "星期" in result
        assert "{weekday}" not in result

    def test_prompt_memory_failure_logged(self, caplog):
        """MemoryStore 加载失败时不影响 prompt 生成."""
        import logging
        from unittest.mock import patch

        with caplog.at_level(logging.DEBUG, logger="crew.engine"):
            with patch("crew.memory.MemoryStore", side_effect=Exception("db error")):
                result = self.engine.prompt(
                    self.employee,
                    args={"target": "test"},
                )
        assert "测试员工" in result  # prompt 仍正常生成
        assert "历史经验" not in result  # 记忆段未注入
