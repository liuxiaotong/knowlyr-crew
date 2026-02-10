"""测试 CLI 命令."""

import tempfile
from pathlib import Path

from click.testing import CliRunner

from crew.cli import main


class TestCLI:
    """测试 CLI."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_version(self):
        result = self.runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_list(self):
        result = self.runner.invoke(main, ["list", "-f", "json"])
        assert result.exit_code == 0
        assert "code-reviewer" in result.output
        assert "test-engineer" in result.output

    def test_list_filter_tag(self):
        result = self.runner.invoke(main, ["list", "--tag", "testing", "-f", "json"])
        assert result.exit_code == 0
        assert "test-engineer" in result.output
        assert "refactor-guide" not in result.output

    def test_list_filter_layer(self):
        result = self.runner.invoke(main, ["list", "--layer", "builtin", "-f", "json"])
        assert result.exit_code == 0
        assert "code-reviewer" in result.output

    def test_show(self):
        result = self.runner.invoke(main, ["show", "code-reviewer"])
        assert result.exit_code == 0
        assert "代码审查员" in result.output

    def test_show_not_found(self):
        result = self.runner.invoke(main, ["show", "nonexistent"])
        assert result.exit_code == 1

    def test_run(self):
        result = self.runner.invoke(main, ["run", "code-reviewer", "main"])
        assert result.exit_code == 0
        assert "代码审查员" in result.output
        assert "main" in result.output

    def test_run_with_trigger(self):
        result = self.runner.invoke(main, ["run", "review", "main"])
        assert result.exit_code == 0
        assert "代码审查员" in result.output

    def test_run_raw(self):
        result = self.runner.invoke(main, ["run", "code-reviewer", "main", "--raw"])
        assert result.exit_code == 0
        # raw 模式不含角色前言
        assert "# 代码审查员" not in result.output
        assert "main" in result.output

    def test_run_missing_arg(self):
        result = self.runner.invoke(main, ["run", "code-reviewer"])
        assert result.exit_code == 1
        assert "target" in result.output

    def test_run_not_found(self):
        result = self.runner.invoke(main, ["run", "nonexistent"])
        assert result.exit_code == 1

    def test_validate_builtin(self):
        from crew.employees import builtin_dir
        result = self.runner.invoke(main, ["validate", str(builtin_dir())])
        assert result.exit_code == 0
        assert "5/5 通过校验" in result.output

    def test_validate_invalid(self):
        fixtures = Path(__file__).parent / "fixtures"
        result = self.runner.invoke(main, ["validate", str(fixtures / "invalid_employee.md")])
        assert result.exit_code == 1

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(main, ["init"], catch_exceptions=False)
            # init 在当前目录创建 .crew/
            assert result.exit_code == 0

    def test_init_employee(self):
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = self.runner.invoke(
                    main, ["init", "--employee", "my-worker"],
                    catch_exceptions=False,
                )
                assert result.exit_code == 0
                assert (Path(tmpdir) / ".crew" / "my-worker.md").exists()
            finally:
                os.chdir(old_cwd)
