"""测试流水线引擎."""

from pathlib import Path

import pytest
import yaml

from crew.pipeline import (
    Pipeline,
    PipelineStep,
    discover_pipelines,
    load_pipeline,
    run_pipeline,
    validate_pipeline,
)


class TestPipelineModels:
    """测试流水线数据模型."""

    def test_pipeline_step(self):
        step = PipelineStep(employee="code-reviewer", args={"target": "main"})
        assert step.employee == "code-reviewer"
        assert step.args == {"target": "main"}

    def test_pipeline_step_default_args(self):
        step = PipelineStep(employee="test-engineer")
        assert step.args == {}

    def test_pipeline(self):
        pl = Pipeline(
            name="test",
            description="测试流水线",
            steps=[PipelineStep(employee="code-reviewer")],
        )
        assert pl.name == "test"
        assert len(pl.steps) == 1


class TestLoadPipeline:
    """测试加载流水线."""

    def test_load_valid(self, tmp_path):
        data = {
            "name": "test-pl",
            "description": "测试",
            "steps": [
                {"employee": "code-reviewer", "args": {"target": "main"}},
                {"employee": "test-engineer"},
            ],
        }
        f = tmp_path / "test.yaml"
        f.write_text(yaml.dump(data, allow_unicode=True))
        pl = load_pipeline(f)
        assert pl.name == "test-pl"
        assert len(pl.steps) == 2
        assert pl.steps[0].args == {"target": "main"}

    def test_load_builtin_review_test_pr(self):
        builtin = Path(__file__).parent.parent / "src" / "crew" / "employees" / "pipelines" / "review-test-pr.yaml"
        pl = load_pipeline(builtin)
        assert pl.name == "review-test-pr"
        assert len(pl.steps) == 3
        assert pl.steps[0].employee == "code-reviewer"

    def test_load_builtin_full_review(self):
        builtin = Path(__file__).parent.parent / "src" / "crew" / "employees" / "pipelines" / "full-review.yaml"
        pl = load_pipeline(builtin)
        assert pl.name == "full-review"
        assert len(pl.steps) == 3


class TestValidatePipeline:
    """测试流水线校验."""

    def test_empty_steps(self):
        pl = Pipeline(name="empty", steps=[])
        errors = validate_pipeline(pl)
        assert len(errors) == 1
        assert "至少需要一个步骤" in errors[0]

    def test_valid_pipeline(self):
        pl = Pipeline(
            name="valid",
            steps=[PipelineStep(employee="code-reviewer")],
        )
        errors = validate_pipeline(pl)
        assert errors == []

    def test_unknown_employee(self):
        pl = Pipeline(
            name="bad",
            steps=[PipelineStep(employee="nonexistent-worker")],
        )
        errors = validate_pipeline(pl)
        assert len(errors) == 1
        assert "nonexistent-worker" in errors[0]


class TestRunPipeline:
    """测试执行流水线."""

    def test_run_basic(self):
        pl = Pipeline(
            name="test",
            steps=[
                PipelineStep(employee="code-reviewer", args={"target": "main"}),
            ],
        )
        outputs = run_pipeline(pl, smart_context=False)
        assert len(outputs) == 1
        assert outputs[0]["employee"] == "code-reviewer"
        assert "prompt" in outputs[0]
        assert "main" in outputs[0]["prompt"]

    def test_run_variable_resolution(self):
        pl = Pipeline(
            name="test",
            steps=[
                PipelineStep(employee="code-reviewer", args={"target": "$target"}),
            ],
        )
        outputs = run_pipeline(pl, initial_args={"target": "feat/login"}, smart_context=False)
        assert outputs[0]["args"]["target"] == "feat/login"

    def test_run_unknown_employee(self):
        pl = Pipeline(
            name="test",
            steps=[PipelineStep(employee="no-such-worker")],
        )
        outputs = run_pipeline(pl, smart_context=False)
        assert len(outputs) == 1
        assert outputs[0].get("error") is True
        assert "未找到" in outputs[0]["prompt"]

    def test_run_multi_step(self):
        pl = Pipeline(
            name="multi",
            steps=[
                PipelineStep(employee="code-reviewer", args={"target": "main"}),
                PipelineStep(employee="test-engineer", args={"target": "src/"}),
            ],
        )
        outputs = run_pipeline(pl, smart_context=False)
        assert len(outputs) == 2
        assert outputs[0]["employee"] == "code-reviewer"
        assert outputs[1]["employee"] == "test-engineer"


class TestDiscoverPipelines:
    """测试流水线发现."""

    def test_discover_builtin(self):
        pipelines = discover_pipelines()
        assert "review-test-pr" in pipelines
        assert "full-review" in pipelines

    def test_discover_project(self, tmp_path):
        pl_dir = tmp_path / ".crew" / "pipelines"
        pl_dir.mkdir(parents=True)
        data = {"name": "custom", "steps": [{"employee": "code-reviewer"}]}
        (pl_dir / "custom.yaml").write_text(yaml.dump(data))
        pipelines = discover_pipelines(project_dir=tmp_path)
        assert "custom" in pipelines

    def test_project_overrides_builtin(self, tmp_path):
        pl_dir = tmp_path / ".crew" / "pipelines"
        pl_dir.mkdir(parents=True)
        data = {"name": "full-review", "description": "自定义", "steps": [{"employee": "code-reviewer"}]}
        (pl_dir / "full-review.yaml").write_text(yaml.dump(data))
        pipelines = discover_pipelines(project_dir=tmp_path)
        # 项目流水线应覆盖内置
        pl = load_pipeline(pipelines["full-review"])
        assert pl.description == "自定义"
        assert len(pl.steps) == 1
