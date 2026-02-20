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
        from unittest.mock import patch

        from crew.engine import _get_git_branch

        with patch("crew.engine.subprocess.run", side_effect=FileNotFoundError):
            assert _get_git_branch() == ""

        with patch("crew.engine.subprocess.run", side_effect=OSError("err")):
            assert _get_git_branch() == ""

    def test_prompt_with_kpi(self):
        """带 KPI 的员工 prompt 应包含 KPI 段."""
        from crew.models import Employee
        emp = Employee(
            name="kpi-test",
            description="测试 KPI 注入",
            body="正文内容",
            kpi=["指标A", "指标B"],
        )
        result = self.engine.prompt(emp)
        assert "**KPI**" in result
        assert "指标A" in result
        assert "指标B" in result

    def test_prompt_without_kpi(self):
        """无 KPI 的员工 prompt 不应有 KPI 段."""
        from crew.models import Employee
        emp = Employee(
            name="no-kpi",
            description="无 KPI",
            body="正文内容",
        )
        result = self.engine.prompt(emp)
        assert "**KPI**" not in result

    def test_render_weekday_variable(self):
        """render() 应替换 {weekday} 为中文星期."""
        from crew.models import Employee

        emp = Employee(
            name="weekday-test",
            description="test",
            body="今天是{weekday}",
            args=[],
        )
        result = self.engine.render(emp)
        assert "星期" in result
        assert "{weekday}" not in result

    def test_prompt_injects_corrections(self):
        """有 correction 记忆时 prompt 应包含'上次教训'section."""
        from unittest.mock import MagicMock, patch

        from crew.models import Employee

        emp = Employee(
            name="correction-test",
            description="测试自检注入",
            body="正文内容",
        )

        mock_entry = MagicMock()
        mock_entry.content = "[自检] 审查代码 | 通过: 文件覆盖 | 待改进: 并发安全; 错误处理"
        mock_entry.category = "correction"
        mock_entry.confidence = 0.7

        mock_store = MagicMock()
        mock_store.format_for_prompt.return_value = ""
        mock_store.query.return_value = [mock_entry]

        with patch("crew.memory.MemoryStore", return_value=mock_store):
            result = self.engine.prompt(emp)

        assert "上次教训" in result
        assert "并发安全; 错误处理" in result

    def test_prompt_no_corrections(self):
        """无 correction 记忆时不应有'上次教训'."""
        from unittest.mock import MagicMock, patch

        from crew.models import Employee

        emp = Employee(
            name="no-correction",
            description="无自检记忆",
            body="正文内容",
        )

        mock_store = MagicMock()
        mock_store.format_for_prompt.return_value = ""
        mock_store.query.return_value = []

        with patch("crew.memory.MemoryStore", return_value=mock_store):
            result = self.engine.prompt(emp)

        assert "上次教训" not in result

    def test_selfcheck_extraction_format(self):
        """自检提取应生成结构化格式."""
        import re

        output_text = (
            "一些输出内容\n\n"
            "## 完成后自检\n\n"
            "- [x] 文件覆盖率检查\n"
            "- [x] 安全关键词扫描\n"
            "- [ ] 并发安全\n"
            "- [ ] 错误处理\n"
        )
        check_match = re.search(
            r"##\s*完成后自检[^\n]*\n+((?:- \[.\].*\n?)+)",
            output_text,
        )
        assert check_match is not None
        check_lines = check_match.group(1).strip().split("\n")
        passed = []
        failed = []
        for cl in check_lines:
            cl = cl.strip()
            if cl.startswith("- [x]") or cl.startswith("- [X]"):
                passed.append(cl[5:].strip())
            elif cl.startswith("- [ ]"):
                failed.append(cl[5:].strip())

        parts = ["[自检] 审查代码"]
        if passed:
            parts.append(f"通过: {'; '.join(passed)}")
        if failed:
            parts.append(f"待改进: {'; '.join(failed)}")
        content = " | ".join(parts)

        assert content == "[自检] 审查代码 | 通过: 文件覆盖率检查; 安全关键词扫描 | 待改进: 并发安全; 错误处理"

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
