"""测试 CLI 命令."""

import json
import tempfile
from pathlib import Path
import yaml
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
                    main,
                    ["init", "--employee", "my-worker"],
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
        pipeline_yaml = (
            """name: lint-demo\ndescription: test\nsteps:\n  - employee: product-manager\n"""
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "demo.yaml"
            target.write_text(pipeline_yaml, encoding="utf-8")
            result = self.runner.invoke(main, ["lint", str(target)])
            assert result.exit_code == 0
            assert "Lint 通过" in result.output

    def test_lint_pipeline_failure(self):
        pipeline_yaml = (
            """name: lint-demo\ndescription: test\nsteps:\n  - employee: unknown-emp\n"""
        )
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
            main,
            ["run", "code-reviewer", "main", "--smart-context"],
        )
        assert result.exit_code == 0
        assert "代码审查员" in result.output

    def test_session_cli_roundtrip(self):
        with self.runner.isolated_filesystem():
            run_result = self.runner.invoke(main, ["run", "code-reviewer", "main"])
            assert run_result.exit_code == 0

            list_result = self.runner.invoke(main, ["session", "list", "-f", "json"])
            assert list_result.exit_code == 0
            data = json.loads(list_result.output)
            assert len(data) == 1
            session_id = data[0]["session_id"]

            show_result = self.runner.invoke(main, ["session", "show", session_id])
            assert show_result.exit_code == 0
            assert "prompt" in show_result.output

    def test_run_writes_memory(self):
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(main, ["run", "code-reviewer", "main"])
            assert result.exit_code == 0
            mem_file = Path(".crew/memory/code-reviewer.jsonl")
            assert mem_file.exists()
            content = mem_file.read_text(encoding="utf-8")
            assert "main" in content or "参数" in content

    def test_pipeline_list(self):
        result = self.runner.invoke(main, ["pipeline", "list"])
        assert result.exit_code == 0

    def test_pipeline_run_parallel(self):
        import tempfile

        pipeline_yaml = """name: lane-demo
description: test
steps:
  - employee: code-reviewer
    args:
      target: main
"""
        with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as tmp:
            tmp.write(pipeline_yaml)
            tmp_path = tmp.name
        result = self.runner.invoke(
            main,
            ["pipeline", "run", tmp_path, "--parallel"],
        )
        assert result.exit_code == 0, result.output
        assert "步骤 1" in result.output

    def test_discuss_run_parallel(self):
        result = self.runner.invoke(
            main,
            [
                "discuss",
                "adhoc",
                "-e",
                "code-reviewer",
                "-e",
                "test-engineer",
                "-t",
                "代码质量",
                "--parallel",
            ],
        )
        assert result.exit_code == 0

    def test_discuss_adhoc_parallel(self):
        result = self.runner.invoke(
            main,
            ["discuss", "adhoc", "-e", "code-reviewer", "-t", "API 设计", "--parallel"],
        )
        assert result.exit_code == 0, result.output
        assert "即席" in result.output

    def test_catalog_list_json(self):
        result = self.runner.invoke(main, ["catalog", "list", "--format", "json"])
        assert result.exit_code == 0
        assert "product-manager" in result.output

    def test_catalog_show(self):
        result = self.runner.invoke(main, ["catalog", "show", "product-manager", "--json"])
        assert result.exit_code == 0
        assert '"product-manager"' in result.output

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
                assert result.exit_code == 0, result.output
                assert '"lint":' in result.output
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
                assert result.exit_code == 0, result.output
                report_path = Path(".crew/quality-report.json")
                assert report_path.exists()
                data = json.loads(report_path.read_text(encoding="utf-8"))
                assert data["lint"]["status"] == "ok"
                assert "logs" in data
            finally:
                os.chdir(old_cwd)

    def test_pipeline_show(self, tmp_path, monkeypatch):
        import yaml

        pl_dir = tmp_path / ".crew" / "pipelines"
        pl_dir.mkdir(parents=True)
        data = {"name": "test-pl", "steps": [{"employee": "code-reviewer"}]}
        (pl_dir / "test-pl.yaml").write_text(yaml.dump(data))
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(main, ["pipeline", "show", "test-pl"])
        assert result.exit_code == 0
        assert "code-reviewer" in result.output

    def test_pipeline_show_not_found(self):
        result = self.runner.invoke(main, ["pipeline", "show", "nonexistent"])
        assert result.exit_code == 1

    def test_pipeline_run(self, tmp_path):
        data = {
            "name": "test-run",
            "steps": [
                {"employee": "code-reviewer", "args": {"target": "main"}},
                {"employee": "test-engineer", "args": {"target": "main"}},
            ],
        }
        pl_file = tmp_path / "test-run.yaml"
        pl_file.write_text(yaml.dump(data))
        result = self.runner.invoke(
            main,
            ["pipeline", "run", str(pl_file), "--arg", "target=main"],
        )
        assert result.exit_code == 0
        assert "code-reviewer" in result.output
        assert "test-engineer" in result.output

    def test_pipeline_graph(self, tmp_path, monkeypatch):
        pl_dir = tmp_path / ".crew" / "pipelines"
        pl_dir.mkdir(parents=True)
        data = {
            "name": "test-graph",
            "steps": [
                {"employee": "code-reviewer"},
                {"employee": "test-engineer"},
            ],
        }
        (pl_dir / "test-graph.yaml").write_text(yaml.dump(data))
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(main, ["pipeline", "graph", "test-graph"])
        assert result.exit_code == 0
        assert "graph LR" in result.output
        assert "code-reviewer" in result.output
        assert "test-engineer" in result.output

    def test_pipeline_graph_not_found(self):
        result = self.runner.invoke(main, ["pipeline", "graph", "nonexistent"])
        assert result.exit_code == 1

    def test_pipeline_graph_with_parallel(self, tmp_path, monkeypatch):
        pl_dir = tmp_path / ".crew" / "pipelines"
        pl_dir.mkdir(parents=True)
        data = {
            "name": "test-par",
            "steps": [
                {"employee": "code-reviewer"},
                {
                    "parallel": [
                        {"employee": "test-engineer"},
                        {"employee": "refactor-guide"},
                    ]
                },
                {"employee": "pr-creator"},
            ],
        }
        (pl_dir / "test-par.yaml").write_text(yaml.dump(data))
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(main, ["pipeline", "graph", "test-par"])
        assert result.exit_code == 0
        assert "并行" in result.output
        assert "合并" in result.output

    def test_cron_list_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(main, ["cron", "list", "-d", str(tmp_path)])
        assert result.exit_code == 0
        assert "未配置" in result.output

    def test_cron_list_with_schedules(self, tmp_path, monkeypatch):
        cron_dir = tmp_path / ".crew"
        cron_dir.mkdir(parents=True)
        data = {
            "schedules": [
                {
                    "name": "daily-review",
                    "cron": "0 9 * * *",
                    "target_type": "pipeline",
                    "target_name": "code-review",
                    "args": {"target": "main"},
                },
            ],
        }
        (cron_dir / "cron.yaml").write_text(yaml.dump(data))
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(main, ["cron", "list", "-d", str(tmp_path)])
        assert result.exit_code == 0
        assert "daily-review" in result.output

    def test_cron_preview_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(main, ["cron", "preview", "-d", str(tmp_path)])
        assert result.exit_code == 0
        assert "未配置" in result.output

    def test_cron_preview_with_schedules(self, tmp_path, monkeypatch):
        cron_dir = tmp_path / ".crew"
        cron_dir.mkdir(parents=True)
        data = {
            "schedules": [
                {
                    "name": "hourly",
                    "cron": "0 * * * *",
                    "target_type": "employee",
                    "target_name": "test-engineer",
                },
            ],
        }
        (cron_dir / "cron.yaml").write_text(yaml.dump(data))
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(main, ["cron", "preview", "-n", "3", "-d", str(tmp_path)])
        assert result.exit_code == 0
        # Rich table 可能截断名称，检查有输出且无报错
        assert "触发时间" in result.output or "hourly" in result.output

    # ── Fuzzy matching tests ──

    def test_run_not_found_suggests_similar(self):
        result = self.runner.invoke(main, ["run", "code-reviwer"])
        assert result.exit_code == 1
        assert "类似的名称" in result.output

    def test_pipeline_show_not_found_suggests_similar(self, tmp_path, monkeypatch):
        pl_dir = tmp_path / ".crew" / "pipelines"
        pl_dir.mkdir(parents=True)
        data = {"name": "full-review", "steps": [{"employee": "code-reviewer"}]}
        (pl_dir / "full-review.yaml").write_text(yaml.dump(data))
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(main, ["pipeline", "show", "ful-review"])
        assert result.exit_code == 1
        assert "类似的名称" in result.output

    def test_pipeline_graph_not_found_suggests_similar(self, tmp_path, monkeypatch):
        pl_dir = tmp_path / ".crew" / "pipelines"
        pl_dir.mkdir(parents=True)
        data = {"name": "full-review", "steps": [{"employee": "code-reviewer"}]}
        (pl_dir / "full-review.yaml").write_text(yaml.dump(data))
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(main, ["pipeline", "graph", "ful-review"])
        assert result.exit_code == 1
        assert "类似的名称" in result.output

    # ── Debug context test ──

    def test_run_debug_context(self):
        result = self.runner.invoke(main, ["run", "code-reviewer", "main", "--debug-context"])
        assert result.exit_code == 0
        assert "[Context]" in result.output

    # ── Checkpoint tests ──

    def test_checkpoint_list_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(main, ["pipeline", "checkpoint", "list", "-d", str(tmp_path)])
        assert result.exit_code == 0
        assert "未找到" in result.output

    def test_checkpoint_list_with_data(self, tmp_path, monkeypatch):
        from crew.task_registry import TaskRegistry

        (tmp_path / ".crew").mkdir(parents=True)
        persist_path = tmp_path / ".crew" / "tasks.jsonl"
        registry = TaskRegistry(persist_path=persist_path)
        record = registry.create(
            trigger="direct",
            target_type="pipeline",
            target_name="full-review",
            args={"target": "main"},
        )
        registry.update_checkpoint(
            record.task_id,
            {
                "pipeline_name": "full-review",
                "completed_steps": [{"employee": "code-reviewer"}],
            },
        )
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(main, ["pipeline", "checkpoint", "list", "-d", str(tmp_path)])
        assert result.exit_code == 0
        assert "full-review" in result.output

    def test_checkpoint_resume_no_task(self, tmp_path, monkeypatch):
        (tmp_path / ".crew").mkdir(parents=True)
        (tmp_path / ".crew" / "tasks.jsonl").write_text("")
        monkeypatch.chdir(tmp_path)
        result = self.runner.invoke(
            main, ["pipeline", "checkpoint", "resume", "nonexist", "-d", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "未找到" in result.output

    # ── Memory repair test ──

    def test_memory_index_repair(self, tmp_path, monkeypatch):
        from crew.memory import MemoryStore

        monkeypatch.chdir(tmp_path)
        mem_dir = tmp_path / ".crew" / "memory"
        mem_dir.mkdir(parents=True)
        store = MemoryStore(memory_dir=mem_dir)
        store.add("bot", "finding", "test memory")

        # CLI's MemoryStore() will use cwd, which is tmp_path
        result = self.runner.invoke(main, ["memory", "index", "--repair"])
        assert result.exit_code == 0
        assert "修复完成" in result.output

    def test_changelog_draft_subprocess_has_timeout(self):
        """changelog_draft 的 subprocess.run 应带 timeout 参数."""
        from unittest.mock import MagicMock, patch

        mock_result = MagicMock()
        mock_result.stdout = "abc1234 test commit\n"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = self.runner.invoke(main, ["changelog-draft", "-n", "1"])
            if mock_run.called:
                _, kwargs = mock_run.call_args
                assert "timeout" in kwargs, "subprocess.run 应设置 timeout"
