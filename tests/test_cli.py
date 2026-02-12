"""测试 CLI 命令."""

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from crew import __version__
from crew.cli import main
from crew.log import WorkLogger


class TestCLI:
    """测试 CLI."""

    def setup_method(self):
        self.runner = CliRunner()

    def test_version(self):
        result = self.runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

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
        # 测试按层过滤（仅 builtin / skill / private）
        for layer in ("builtin", "skill", "private"):
            result = self.runner.invoke(main, ["list", "--layer", layer, "-f", "json"])
            assert result.exit_code == 0
            if result.output.strip() and "未找到" not in result.output:
                assert f'"layer": "{layer}"' in result.output

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
        assert "通过校验" in result.output

    def test_validate_invalid(self):
        fixtures = Path(__file__).parent / "fixtures"
        result = self.runner.invoke(main, ["validate", str(fixtures / "invalid_employee.md")])
        assert result.exit_code == 1

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(main, ["init"], catch_exceptions=False)
            # init 在当前目录创建 private/employees/
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
                assert (Path(tmpdir) / "private" / "employees" / "my-worker.md").exists()
            finally:
                os.chdir(old_cwd)

    def test_template_list(self):
        result = self.runner.invoke(main, ["template", "list"])
        assert result.exit_code == 0
        assert "advanced-employee" in result.output

    def test_template_apply(self):
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                result = self.runner.invoke(
                    main,
                    [
                        "template",
                        "apply",
                        "advanced-employee",
                        "--employee",
                        "planner",
                        "--var",
                        "tags=['planning']",
                        "--force",
                    ],
                    catch_exceptions=False,
                )
                assert result.exit_code == 0
                out_file = Path(tmpdir) / "private" / "employees" / "planner.md"
                assert out_file.exists()
                content = out_file.read_text(encoding="utf-8")
                assert "planner" in content
                assert "planning" in content
            finally:
                os.chdir(old_cwd)

    def test_lint_pipeline_success(self):
        pipeline_yaml = """name: lint-demo\ndescription: test\nsteps:\n  - employee: product-manager\n"""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "demo.yaml"
            target.write_text(pipeline_yaml, encoding="utf-8")
            result = self.runner.invoke(main, ["lint", str(target)])
            assert result.exit_code == 0
            assert "Lint 通过" in result.output

    def test_lint_pipeline_failure(self):
        pipeline_yaml = """name: lint-demo\ndescription: test\nsteps:\n  - employee: unknown-emp\n"""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "demo.yaml"
            target.write_text(pipeline_yaml, encoding="utf-8")
            result = self.runner.invoke(main, ["lint", str(target)])
            assert result.exit_code == 1
            assert "unknown-emp" in result.output

    def test_lint_schema_violation(self):
        pipeline_yaml = """name: lint-demo\nsteps: invalid\n"""
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_dir = Path(tmpdir) / "schemas"
            schema_dir.mkdir()
            (schema_dir / "pipeline.schema.json").write_text(
                """{\n  \"type\": \"object\",\n  \"properties\": {\n    \"steps\": {\"type\": \"array\"}\n  }\n}\n""",
                encoding="utf-8",
            )
            target = Path(tmpdir) / "demo.yaml"
            target.write_text(pipeline_yaml, encoding="utf-8")
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(tmpdir)
                result = self.runner.invoke(main, ["lint", str(target)])
            finally:
                os.chdir(old_cwd)
            assert result.exit_code == 1
            assert "schema 校验失败" in result.output

    def test_run_smart_context(self):
        result = self.runner.invoke(
            main, ["run", "code-reviewer", "main", "--smart-context"],
        )
        assert result.exit_code == 0
        assert "代码审查员" in result.output

    def test_pipeline_list(self):
        result = self.runner.invoke(main, ["pipeline", "list"])
        assert result.exit_code == 0
        assert "review-test-pr" in result.output
        assert "full-review" in result.output

    def test_catalog_list_json(self):
        result = self.runner.invoke(main, ["catalog", "list", "--format", "json"])
        assert result.exit_code == 0
        assert "product-manager" in result.output

    def test_catalog_show(self):
        result = self.runner.invoke(main, ["catalog", "show", "product-manager", "--json"])
        assert result.exit_code == 0
        assert "\"product-manager\"" in result.output

    def test_check_json_output(self):
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                (Path(tmpdir) / ".crew" / "pipelines").mkdir(parents=True)
                (Path(tmpdir) / ".crew" / "pipelines" / "ok.yaml").write_text(
                    """name: lint-demo\nsteps:\n  - employee: product-manager\n""",
                    encoding="utf-8",
                )
                result = self.runner.invoke(main, ["check", "--json", "--no-logs", "--no-file"])
                assert result.exit_code == 0
                assert "\"lint\":" in result.output
            finally:
                os.chdir(old_cwd)

    def test_check_writes_report_file(self):
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                (Path(tmpdir) / ".crew" / "pipelines").mkdir(parents=True)
                (Path(tmpdir) / ".crew" / "pipelines" / "ok.yaml").write_text(
                    """name: lint-demo\nsteps:\n  - employee: product-manager\n""",
                    encoding="utf-8",
                )
                logger = WorkLogger()
                session = logger.create_session("product-manager")
                logger.add_entry(session, "action", severity="warning")
                result = self.runner.invoke(main, ["check"])
                assert result.exit_code == 0
                report_path = Path(".crew/quality-report.json")
                assert report_path.exists()
                data = json.loads(report_path.read_text(encoding="utf-8"))
                assert data["lint"]["status"] == "ok"
                assert "logs" in data
            finally:
                os.chdir(old_cwd)

    def test_pipeline_show(self):
        result = self.runner.invoke(main, ["pipeline", "show", "review-test-pr"])
        assert result.exit_code == 0
        assert "code-reviewer" in result.output

    def test_pipeline_show_not_found(self):
        result = self.runner.invoke(main, ["pipeline", "show", "nonexistent"])
        assert result.exit_code == 1

    def test_pipeline_run(self):
        builtin = Path(__file__).parent.parent / "src" / "crew" / "employees" / "pipelines" / "full-review.yaml"
        result = self.runner.invoke(
            main, ["pipeline", "run", str(builtin), "--arg", "target=main"],
        )
        assert result.exit_code == 0
        assert "code-reviewer" in result.output
        assert "refactor-guide" in result.output
        assert "test-engineer" in result.output
