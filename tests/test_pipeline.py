"""测试流水线引擎."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from crew.models import ParallelGroup, PipelineStep, StepResult
from crew.pipeline import (
    Pipeline,
    _build_checkpoint,
    _resolve_output_refs,
    aresume_pipeline,
    arun_pipeline,
    discover_pipelines,
    load_pipeline,
    run_pipeline,
    validate_pipeline,
)


def _run(coro):
    return asyncio.run(coro)


class TestPipelineModels:
    """测试流水线数据模型."""

    def test_pipeline_step(self):
        step = PipelineStep(employee="code-reviewer", args={"target": "main"})
        assert step.employee == "code-reviewer"
        assert step.args == {"target": "main"}
        assert step.id == ""

    def test_pipeline_step_with_id(self):
        step = PipelineStep(employee="code-reviewer", id="review", args={"target": "main"})
        assert step.id == "review"

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

    def test_parallel_group(self):
        group = ParallelGroup(parallel=[
            PipelineStep(employee="a", id="a1"),
            PipelineStep(employee="b"),
        ])
        assert len(group.parallel) == 2
        assert group.parallel[0].id == "a1"

    def test_pipeline_with_parallel(self):
        pl = Pipeline(
            name="test",
            steps=[
                PipelineStep(employee="code-reviewer"),
                ParallelGroup(parallel=[
                    PipelineStep(employee="security-auditor"),
                    PipelineStep(employee="test-engineer"),
                ]),
                PipelineStep(employee="pr-creator"),
            ],
        )
        assert len(pl.steps) == 3
        assert isinstance(pl.steps[1], ParallelGroup)


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

    def test_load_with_id(self, tmp_path):
        data = {
            "name": "test-id",
            "steps": [
                {"employee": "code-reviewer", "id": "review", "args": {"target": "main"}},
            ],
        }
        f = tmp_path / "test.yaml"
        f.write_text(yaml.dump(data, allow_unicode=True))
        pl = load_pipeline(f)
        assert pl.steps[0].id == "review"

    def test_load_parallel(self, tmp_path):
        data = {
            "name": "test-parallel",
            "steps": [
                {"employee": "code-reviewer"},
                {"parallel": [
                    {"employee": "security-auditor", "id": "sec"},
                    {"employee": "test-engineer"},
                ]},
                {"employee": "pr-creator"},
            ],
        }
        f = tmp_path / "test.yaml"
        f.write_text(yaml.dump(data, allow_unicode=True))
        pl = load_pipeline(f)
        assert len(pl.steps) == 3
        assert isinstance(pl.steps[0], PipelineStep)
        assert isinstance(pl.steps[1], ParallelGroup)
        assert isinstance(pl.steps[2], PipelineStep)
        assert len(pl.steps[1].parallel) == 2
        assert pl.steps[1].parallel[0].id == "sec"



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

    def test_duplicate_id(self):
        pl = Pipeline(
            name="dup",
            steps=[
                PipelineStep(employee="code-reviewer", id="same"),
                PipelineStep(employee="test-engineer", id="same"),
            ],
        )
        errors = validate_pipeline(pl)
        assert any("重复" in e for e in errors)

    def test_validate_parallel(self):
        pl = Pipeline(
            name="par",
            steps=[
                ParallelGroup(parallel=[
                    PipelineStep(employee="code-reviewer", id="a"),
                    PipelineStep(employee="nonexistent-worker", id="b"),
                ]),
            ],
        )
        errors = validate_pipeline(pl)
        assert any("nonexistent-worker" in e for e in errors)


class TestResolveOutputRefs:
    """测试输出引用解析."""

    def test_prompt_mode_preserves_placeholders(self):
        result = _resolve_output_refs(
            "{prev}", {}, {}, "actual output", execute=False,
        )
        assert result == "{prev}"

    def test_execute_mode_resolves_prev(self):
        result = _resolve_output_refs(
            "Context: {prev}", {}, {}, "review result", execute=True,
        )
        assert result == "Context: review result"

    def test_execute_mode_resolves_by_id(self):
        result = _resolve_output_refs(
            "{steps.review.output}",
            {"review": "good code"},
            {},
            "",
            execute=True,
        )
        assert result == "good code"

    def test_execute_mode_resolves_by_index(self):
        result = _resolve_output_refs(
            "{steps.0.output}",
            {},
            {0: "step zero output"},
            "",
            execute=True,
        )
        assert result == "step zero output"

    def test_unresolved_ref_kept(self):
        result = _resolve_output_refs(
            "{steps.missing.output}",
            {},
            {},
            "",
            execute=True,
        )
        assert result == "{steps.missing.output}"

    def test_multiple_refs(self):
        result = _resolve_output_refs(
            "A: {steps.a.output}, B: {steps.b.output}",
            {"a": "alpha", "b": "beta"},
            {},
            "",
            execute=True,
        )
        assert result == "A: alpha, B: beta"


class TestRunPipeline:
    """测试执行流水线."""

    def test_run_basic(self):
        pl = Pipeline(
            name="test",
            steps=[
                PipelineStep(employee="code-reviewer", args={"target": "main"}),
            ],
        )
        result = run_pipeline(pl, smart_context=False)
        assert result.pipeline_name == "test"
        assert result.mode == "prompt"
        assert len(result.steps) == 1
        step = result.steps[0]
        assert isinstance(step, StepResult)
        assert step.employee == "code-reviewer"
        assert step.prompt
        assert "main" in step.prompt

    def test_run_variable_resolution(self):
        pl = Pipeline(
            name="test",
            steps=[
                PipelineStep(employee="code-reviewer", args={"target": "$target"}),
            ],
        )
        result = run_pipeline(pl, initial_args={"target": "feat/login"}, smart_context=False)
        assert result.steps[0].args["target"] == "feat/login"

    def test_run_unknown_employee(self):
        pl = Pipeline(
            name="test",
            steps=[PipelineStep(employee="no-such-worker")],
        )
        result = run_pipeline(pl, smart_context=False)
        assert len(result.steps) == 1
        assert result.steps[0].error is True
        assert "未找到" in result.steps[0].prompt

    def test_run_multi_step(self):
        pl = Pipeline(
            name="multi",
            steps=[
                PipelineStep(employee="code-reviewer", args={"target": "main"}),
                PipelineStep(employee="test-engineer", args={"target": "src/"}),
            ],
        )
        result = run_pipeline(pl, smart_context=False)
        assert len(result.steps) == 2
        assert result.steps[0].employee == "code-reviewer"
        assert result.steps[1].employee == "test-engineer"

    def test_run_parallel_group_prompt_mode(self):
        pl = Pipeline(
            name="par-test",
            steps=[
                ParallelGroup(parallel=[
                    PipelineStep(employee="code-reviewer", id="a", args={"target": "main"}),
                    PipelineStep(employee="test-engineer", id="b", args={"target": "main"}),
                ]),
            ],
        )
        result = run_pipeline(pl, smart_context=False)
        assert len(result.steps) == 1
        group = result.steps[0]
        assert isinstance(group, list)
        assert len(group) == 2
        assert group[0].step_id in ("a", "b")
        assert group[1].step_id in ("a", "b")

    def test_run_with_step_ids(self):
        pl = Pipeline(
            name="id-test",
            steps=[
                PipelineStep(employee="code-reviewer", id="review", args={"target": "main"}),
            ],
        )
        result = run_pipeline(pl, smart_context=False)
        assert result.steps[0].step_id == "review"

    def test_on_step_complete_callback(self):
        pl = Pipeline(
            name="cb-test",
            steps=[
                PipelineStep(employee="code-reviewer", args={"target": "main"}),
                PipelineStep(employee="test-engineer", args={"target": "main"}),
            ],
        )
        callback_results = []
        result = run_pipeline(
            pl, smart_context=False,
            on_step_complete=lambda r: callback_results.append(r.employee),
        )
        assert len(callback_results) == 2
        assert "code-reviewer" in callback_results

    def test_flat_index_with_parallel(self):
        pl = Pipeline(
            name="idx-test",
            steps=[
                PipelineStep(employee="code-reviewer"),
                ParallelGroup(parallel=[
                    PipelineStep(employee="test-engineer"),
                    PipelineStep(employee="code-reviewer"),
                ]),
                PipelineStep(employee="code-reviewer"),
            ],
        )
        result = run_pipeline(pl, smart_context=False)
        # Step 0: code-reviewer (index 0)
        # Parallel group: test-engineer (index 1), code-reviewer (index 2)
        # Step 2: code-reviewer (index 3)
        assert result.steps[0].step_index == 0
        group = result.steps[1]
        assert isinstance(group, list)
        indices = {r.step_index for r in group}
        assert indices == {1, 2}
        assert result.steps[2].step_index == 3


class TestDiscoverPipelines:
    """测试流水线发现."""

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


class TestBuildCheckpoint:
    """测试断点数据构建."""

    def test_basic(self):
        r1 = StepResult(employee="a", step_index=0, args={}, prompt="p1", output="out1")
        checkpoint = _build_checkpoint(
            "test-pl", [r1], {"a": "out1"}, {0: "out1"}, 1, 1,
        )
        assert checkpoint["pipeline_name"] == "test-pl"
        assert len(checkpoint["completed_steps"]) == 1
        assert checkpoint["next_flat_index"] == 1
        assert checkpoint["next_step_i"] == 1
        assert checkpoint["outputs_by_id"]["a"] == "out1"

    def test_with_parallel(self):
        r1 = StepResult(employee="a", step_index=0, args={}, prompt="p1", output="o1")
        r2 = StepResult(employee="b", step_index=1, args={}, prompt="p2", output="o2")
        checkpoint = _build_checkpoint(
            "test", [[r1, r2]], {}, {0: "o1", 1: "o2"}, 2, 1,
        )
        assert len(checkpoint["completed_steps"]) == 1
        assert isinstance(checkpoint["completed_steps"][0], list)
        assert len(checkpoint["completed_steps"][0]) == 2


class TestAsyncPipelineCheckpoint:
    """测试异步 pipeline 的 checkpoint 回调."""

    def test_on_step_complete_called(self):
        pl = Pipeline(
            name="cb-test",
            steps=[
                PipelineStep(employee="code-reviewer", args={"target": "main"}),
                PipelineStep(employee="test-engineer", args={"target": "main"}),
            ],
        )
        callbacks = []

        def on_complete(step_result, checkpoint):
            callbacks.append((step_result.employee, checkpoint))

        result = _run(arun_pipeline(pl, smart_context=False, on_step_complete=on_complete))
        assert len(callbacks) == 2
        # 第一次回调的 checkpoint
        assert callbacks[0][1]["next_step_i"] == 1
        assert callbacks[0][1]["next_flat_index"] == 1
        # 第二次回调的 checkpoint
        assert callbacks[1][1]["next_step_i"] == 2
        assert callbacks[1][1]["next_flat_index"] == 2

    def test_checkpoint_has_completed_steps(self):
        pl = Pipeline(
            name="cp-test",
            steps=[
                PipelineStep(employee="code-reviewer", id="review", args={"target": "main"}),
                PipelineStep(employee="test-engineer", args={"target": "main"}),
            ],
        )
        last_checkpoint = {}

        def on_complete(step_result, checkpoint):
            nonlocal last_checkpoint
            last_checkpoint = checkpoint

        _run(arun_pipeline(pl, smart_context=False, on_step_complete=on_complete))
        assert len(last_checkpoint["completed_steps"]) == 2
        assert "review" in last_checkpoint["outputs_by_id"]


class TestResumePipeline:
    """测试断点恢复."""

    def test_resume_from_checkpoint(self):
        """从第 1 步的 checkpoint 恢复，跳过第 0 步."""
        # 模拟 checkpoint: 第 0 步已完成
        r0 = StepResult(
            employee="code-reviewer", step_id="review", step_index=0,
            args={"target": "main"}, prompt="prompt0", output="review output",
        )
        checkpoint = _build_checkpoint(
            "test-pl", [r0], {"review": "review output"}, {0: "review output"}, 1, 1,
        )

        pl = Pipeline(
            name="test-pl",
            steps=[
                PipelineStep(employee="code-reviewer", id="review", args={"target": "main"}),
                PipelineStep(employee="test-engineer", args={"target": "main"}),
            ],
        )

        result = _run(aresume_pipeline(pl, checkpoint=checkpoint, smart_context=False))
        assert result.pipeline_name == "test-pl"
        assert result.mode == "execute"
        # 应该有 2 步结果（1 恢复 + 1 新执行）
        assert len(result.steps) == 2
        # 第 0 步是从 checkpoint 恢复的
        assert result.steps[0].employee == "code-reviewer"
        assert result.steps[0].output == "review output"
        # 第 1 步是新执行的（prompt-only，因为没有 executor）
        assert result.steps[1].employee == "test-engineer"

    def test_resume_all_completed(self):
        """所有步骤都已完成，恢复无操作."""
        r0 = StepResult(
            employee="code-reviewer", step_index=0,
            args={"target": "main"}, prompt="p0", output="o0",
        )
        r1 = StepResult(
            employee="test-engineer", step_index=1,
            args={"target": "main"}, prompt="p1", output="o1",
        )
        checkpoint = _build_checkpoint(
            "test-pl", [r0, r1], {}, {0: "o0", 1: "o1"}, 2, 2,
        )

        pl = Pipeline(
            name="test-pl",
            steps=[
                PipelineStep(employee="code-reviewer", args={"target": "main"}),
                PipelineStep(employee="test-engineer", args={"target": "main"}),
            ],
        )

        result = _run(aresume_pipeline(pl, checkpoint=checkpoint, smart_context=False))
        assert len(result.steps) == 2
        # 没有新执行的步骤，全部从 checkpoint 恢复
        assert result.steps[0].output == "o0"
        assert result.steps[1].output == "o1"

    def test_resume_empty_checkpoint(self):
        """空 checkpoint 等于从头执行."""
        checkpoint = {
            "pipeline_name": "test",
            "completed_steps": [],
            "outputs_by_id": {},
            "outputs_by_index": {},
            "next_flat_index": 0,
            "next_step_i": 0,
        }

        pl = Pipeline(
            name="test",
            steps=[
                PipelineStep(employee="code-reviewer", args={"target": "main"}),
            ],
        )

        result = _run(aresume_pipeline(pl, checkpoint=checkpoint, smart_context=False))
        assert len(result.steps) == 1
        assert result.steps[0].employee == "code-reviewer"
