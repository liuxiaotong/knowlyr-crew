"""测试流水线引擎."""

import asyncio
from unittest.mock import patch

import pytest
import yaml

from crew.models import (
    Condition,
    ConditionalBody,
    ConditionalStep,
    LoopBody,
    LoopStep,
    ParallelGroup,
    PipelineStep,
    StepResult,
)
from crew.pipeline import (
    Pipeline,
    _build_checkpoint,
    _compute_steps_hash,
    _evaluate_check,
    _resolve_output_refs,
    aresume_pipeline,
    arun_pipeline,
    discover_pipelines,
    load_pipeline,
    pipeline_to_mermaid,
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
        group = ParallelGroup(
            parallel=[
                PipelineStep(employee="a", id="a1"),
                PipelineStep(employee="b"),
            ]
        )
        assert len(group.parallel) == 2
        assert group.parallel[0].id == "a1"

    def test_pipeline_with_parallel(self):
        pl = Pipeline(
            name="test",
            steps=[
                PipelineStep(employee="code-reviewer"),
                ParallelGroup(
                    parallel=[
                        PipelineStep(employee="security-auditor"),
                        PipelineStep(employee="test-engineer"),
                    ]
                ),
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
                {
                    "parallel": [
                        {"employee": "security-auditor", "id": "sec"},
                        {"employee": "test-engineer"},
                    ]
                },
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
                ParallelGroup(
                    parallel=[
                        PipelineStep(employee="code-reviewer", id="a"),
                        PipelineStep(employee="nonexistent-worker", id="b"),
                    ]
                ),
            ],
        )
        errors = validate_pipeline(pl)
        assert any("nonexistent-worker" in e for e in errors)


class TestResolveOutputRefs:
    """测试输出引用解析."""

    def test_prompt_mode_preserves_placeholders(self):
        result = _resolve_output_refs(
            "{prev}",
            {},
            {},
            "actual output",
            execute=False,
        )
        assert result == "{prev}"

    def test_execute_mode_resolves_prev(self):
        result = _resolve_output_refs(
            "Context: {prev}",
            {},
            {},
            "review result",
            execute=True,
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
                ParallelGroup(
                    parallel=[
                        PipelineStep(employee="code-reviewer", id="a", args={"target": "main"}),
                        PipelineStep(employee="test-engineer", id="b", args={"target": "main"}),
                    ]
                ),
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
        run_pipeline(
            pl,
            smart_context=False,
            on_step_complete=lambda r, cp: callback_results.append(r.employee),
        )
        assert len(callback_results) == 2
        assert "code-reviewer" in callback_results

    def test_flat_index_with_parallel(self):
        pl = Pipeline(
            name="idx-test",
            steps=[
                PipelineStep(employee="code-reviewer"),
                ParallelGroup(
                    parallel=[
                        PipelineStep(employee="test-engineer"),
                        PipelineStep(employee="code-reviewer"),
                    ]
                ),
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
        data = {
            "name": "full-review",
            "description": "自定义",
            "steps": [{"employee": "code-reviewer"}],
        }
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
            "test-pl",
            [r1],
            {"a": "out1"},
            {0: "out1"},
            1,
            1,
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
            "test",
            [[r1, r2]],
            {},
            {0: "o1", 1: "o2"},
            2,
            1,
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

        _run(arun_pipeline(pl, smart_context=False, on_step_complete=on_complete))
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
            employee="code-reviewer",
            step_id="review",
            step_index=0,
            args={"target": "main"},
            prompt="prompt0",
            output="review output",
        )
        checkpoint = _build_checkpoint(
            "test-pl",
            [r0],
            {"review": "review output"},
            {0: "review output"},
            1,
            1,
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
            employee="code-reviewer",
            step_index=0,
            args={"target": "main"},
            prompt="p0",
            output="o0",
        )
        r1 = StepResult(
            employee="test-engineer",
            step_index=1,
            args={"target": "main"},
            prompt="p1",
            output="o1",
        )
        checkpoint = _build_checkpoint(
            "test-pl",
            [r0, r1],
            {},
            {0: "o0", 1: "o1"},
            2,
            2,
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


class TestResumeExecuteParam:
    """测试 aresume_pipeline 的 execute 参数."""

    def test_resume_prompt_only(self):
        """execute=False 时走 prompt-only 模式."""
        r0 = StepResult(
            employee="code-reviewer",
            step_index=0,
            args={"target": "main"},
            prompt="p0",
            output="o0",
        )
        checkpoint = _build_checkpoint("test", [r0], {}, {0: "o0"}, 1, 1)
        pl = Pipeline(
            name="test",
            steps=[
                PipelineStep(employee="code-reviewer", args={"target": "main"}),
                PipelineStep(employee="test-engineer", args={"target": "main"}),
            ],
        )
        result = _run(
            aresume_pipeline(pl, checkpoint=checkpoint, smart_context=False, execute=False)
        )
        assert result.mode == "prompt"
        assert len(result.steps) == 2

    def test_resume_execute_mode(self):
        """execute=True（默认）时 mode='execute'."""
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
            steps=[PipelineStep(employee="code-reviewer", args={"target": "main"})],
        )
        # execute=True 但无 API key → prompt-only 实际，但 mode 仍为 execute
        result = _run(
            aresume_pipeline(pl, checkpoint=checkpoint, smart_context=False, execute=True)
        )
        assert result.mode == "execute"


class TestFailFast:
    """测试 fail_fast 步骤失败中止."""

    def _make_error_step(self, employee, error_msg="test error"):
        """创建一个会返回错误的 StepResult."""
        return StepResult(
            employee=employee,
            step_index=0,
            args={},
            prompt="p",
            output="",
            error=True,
            error_message=error_msg,
        )

    def test_fail_fast_stops_pipeline(self):
        """fail_fast=True 时步骤失败后中止."""
        pl = Pipeline(
            name="ff-test",
            steps=[
                PipelineStep(employee="code-reviewer", args={"target": "main"}),
                PipelineStep(employee="test-engineer", args={"target": "main"}),
                PipelineStep(employee="doc-writer", args={"target": "main"}),
            ],
        )
        # 用 mock 让第 2 步失败
        original_exec = None

        def _patched_exec(step, index, engine, employees, *args, **kwargs):
            if step.employee == "test-engineer":
                return StepResult(
                    employee=step.employee,
                    step_id=step.id,
                    step_index=index,
                    args=step.args,
                    prompt="p",
                    output="",
                    error=True,
                    error_message="mock failure",
                )
            return original_exec(step, index, engine, employees, *args, **kwargs)

        import crew.pipeline as _pl_mod

        original_exec = _pl_mod._execute_single_step
        with patch.object(_pl_mod, "_execute_single_step", side_effect=_patched_exec):
            result = run_pipeline(pl, smart_context=False, fail_fast=True)

        # 第 3 步（doc-writer）不应被执行
        flat = [r for r in result.steps if not isinstance(r, list)]
        employees = [r.employee for r in flat]
        assert "code-reviewer" in employees
        assert "test-engineer" in employees
        assert "doc-writer" not in employees

    def test_fail_fast_disabled(self):
        """fail_fast=False 时步骤失败后继续."""
        pl = Pipeline(
            name="ff-off",
            steps=[
                PipelineStep(employee="code-reviewer", args={"target": "main"}),
                PipelineStep(employee="test-engineer", args={"target": "main"}),
            ],
        )

        import crew.pipeline as _pl_mod

        real_exec = _pl_mod._execute_single_step
        call_count = [0]

        def _counting_exec(step, index, engine, employees, *a, **kw):
            call_count[0] += 1
            if step.employee == "code-reviewer":
                return StepResult(
                    employee=step.employee,
                    step_id=step.id,
                    step_index=index,
                    args=step.args,
                    prompt="p",
                    output="",
                    error=True,
                    error_message="mock failure",
                )
            return real_exec(step, index, engine, employees, *a, **kw)

        with patch.object(_pl_mod, "_execute_single_step", side_effect=_counting_exec):
            run_pipeline(pl, smart_context=False, fail_fast=False)

        assert call_count[0] == 2  # 两步都执行了

    def test_fail_fast_saves_checkpoint(self):
        """fail_fast 失败时 checkpoint 包含已完成步骤."""
        pl = Pipeline(
            name="ff-cp",
            steps=[
                PipelineStep(employee="code-reviewer", id="r1", args={"target": "main"}),
                PipelineStep(employee="test-engineer", args={"target": "main"}),
            ],
        )
        checkpoints = []

        import crew.pipeline as _pl_mod

        real_exec = _pl_mod._execute_single_step

        def _failing_exec(step, index, engine, employees, *a, **kw):
            if step.employee == "test-engineer":
                return StepResult(
                    employee=step.employee,
                    step_id=step.id,
                    step_index=index,
                    args=step.args,
                    prompt="p",
                    output="",
                    error=True,
                    error_message="fail",
                )
            return real_exec(step, index, engine, employees, *a, **kw)

        with patch.object(_pl_mod, "_execute_single_step", side_effect=_failing_exec):
            run_pipeline(
                pl,
                smart_context=False,
                fail_fast=True,
                on_step_complete=lambda r, cp: checkpoints.append(cp),
            )

        # 应该有 2 个 checkpoint（两步都被记录了，只是第二步是失败的）
        assert len(checkpoints) == 2
        last_cp = checkpoints[-1]
        assert last_cp["next_step_i"] == 2


class TestSyncCheckpointCallback:
    """测试 run_pipeline 的 checkpoint 回调."""

    def test_checkpoint_callback(self):
        """run_pipeline 的 on_step_complete 接收 checkpoint."""
        pl = Pipeline(
            name="sync-cp",
            steps=[
                PipelineStep(employee="code-reviewer", args={"target": "main"}),
                PipelineStep(employee="test-engineer", args={"target": "main"}),
            ],
        )
        checkpoints = []
        run_pipeline(
            pl,
            smart_context=False,
            on_step_complete=lambda r, cp: checkpoints.append((r.employee, cp)),
        )
        assert len(checkpoints) == 2
        assert checkpoints[0][1]["next_step_i"] == 1
        assert checkpoints[1][1]["next_step_i"] == 2


class TestStepsHash:
    """测试 Pipeline 定义变更检测."""

    def test_steps_hash_in_checkpoint(self):
        """checkpoint 包含 steps_hash."""
        pl = Pipeline(
            name="hash-test",
            steps=[
                PipelineStep(employee="code-reviewer", args={"target": "main"}),
            ],
        )
        checkpoints = []
        run_pipeline(
            pl,
            smart_context=False,
            on_step_complete=lambda r, cp: checkpoints.append(cp),
        )
        assert "steps_hash" in checkpoints[0]
        assert len(checkpoints[0]["steps_hash"]) == 16

    def test_hash_changes_with_steps(self):
        """步骤不同时 hash 不同."""
        pl1 = Pipeline(
            name="h1",
            steps=[PipelineStep(employee="code-reviewer", args={"target": "main"})],
        )
        pl2 = Pipeline(
            name="h1",
            steps=[PipelineStep(employee="test-engineer", args={"target": "main"})],
        )
        assert _compute_steps_hash(pl1) != _compute_steps_hash(pl2)

    def test_hash_stable(self):
        """相同步骤的 hash 应该稳定."""
        pl = Pipeline(
            name="stable",
            steps=[PipelineStep(employee="code-reviewer", args={"target": "main"})],
        )
        assert _compute_steps_hash(pl) == _compute_steps_hash(pl)

    def test_resume_with_changed_pipeline(self):
        """Pipeline 定义变更时正常恢复（不阻止）."""
        r0 = StepResult(
            employee="code-reviewer",
            step_index=0,
            args={"target": "main"},
            prompt="p0",
            output="o0",
        )
        # 用旧 pipeline 的 hash 构建 checkpoint
        old_pl = Pipeline(
            name="changed",
            steps=[
                PipelineStep(employee="code-reviewer", args={"target": "main"}),
                PipelineStep(employee="test-engineer", args={"target": "main"}),
            ],
        )
        old_hash = _compute_steps_hash(old_pl)
        checkpoint = _build_checkpoint("changed", [r0], {}, {0: "o0"}, 1, 1, old_hash)

        # 用修改后的 pipeline 恢复
        new_pl = Pipeline(
            name="changed",
            steps=[
                PipelineStep(employee="code-reviewer", args={"target": "main"}),
                PipelineStep(employee="doc-writer", args={"target": "main"}),
            ],
        )

        result = _run(
            aresume_pipeline(new_pl, checkpoint=checkpoint, smart_context=False, execute=False)
        )
        assert len(result.steps) == 2
        assert result.steps[1].employee == "doc-writer"


class TestRetryFailed:
    """测试 --retry-failed 逻辑."""

    def test_retry_rollback_to_first_error(self):
        """回退到第一个失败步骤."""
        # 模拟 checkpoint: 步骤 0 成功，步骤 1 失败
        r0 = StepResult(
            employee="code-reviewer",
            step_index=0,
            args={"target": "main"},
            prompt="p0",
            output="o0",
        ).model_dump(mode="json")
        r1 = StepResult(
            employee="test-engineer",
            step_index=1,
            args={"target": "main"},
            prompt="p1",
            output="",
            error=True,
            error_message="original failure",
        ).model_dump(mode="json")

        checkpoint = {
            "pipeline_name": "retry-test",
            "completed_steps": [r0, r1],
            "outputs_by_id": {},
            "outputs_by_index": {"0": "o0", "1": ""},
            "next_flat_index": 2,
            "next_step_i": 2,
        }

        # 模拟 CLI retry-failed 逻辑
        completed = checkpoint["completed_steps"]
        first_error_idx = None
        for idx, item in enumerate(completed):
            entries = item if isinstance(item, list) else [item]
            if any(e.get("error") for e in entries):
                first_error_idx = idx
                break

        assert first_error_idx == 1
        checkpoint["completed_steps"] = completed[:first_error_idx]
        checkpoint["next_step_i"] = first_error_idx
        flat = sum(
            len(item) if isinstance(item, list) else 1 for item in checkpoint["completed_steps"]
        )
        checkpoint["next_flat_index"] = flat

        assert checkpoint["next_step_i"] == 1
        assert checkpoint["next_flat_index"] == 1
        assert len(checkpoint["completed_steps"]) == 1


class TestPipelineToMermaid:
    """pipeline_to_mermaid 流程图生成."""

    def test_single_step(self):
        pl = Pipeline(
            name="simple",
            steps=[PipelineStep(employee="code-reviewer", args={})],
        )
        mermaid = pipeline_to_mermaid(pl)
        assert "graph LR" in mermaid
        assert "开始" in mermaid
        assert "结束" in mermaid
        assert '"code-reviewer"' in mermaid

    def test_multiple_steps(self):
        pl = Pipeline(
            name="multi",
            steps=[
                PipelineStep(employee="code-reviewer", args={}),
                PipelineStep(employee="test-engineer", args={}),
            ],
        )
        mermaid = pipeline_to_mermaid(pl)
        assert '"code-reviewer"' in mermaid
        assert '"test-engineer"' in mermaid
        # 应有 S → s0 → s1 → E 的连接
        assert "S --> s0" in mermaid
        assert "s0 --> s1" in mermaid
        assert "s1 --> E" in mermaid

    def test_parallel_group(self):
        pl = Pipeline(
            name="parallel",
            steps=[
                ParallelGroup(
                    parallel=[
                        PipelineStep(employee="test-engineer", args={}),
                        PipelineStep(employee="refactor-guide", args={}),
                    ]
                ),
            ],
        )
        mermaid = pipeline_to_mermaid(pl)
        assert "并行" in mermaid
        assert "合并" in mermaid
        assert '"test-engineer"' in mermaid
        assert '"refactor-guide"' in mermaid

    def test_with_custom_ids(self):
        pl = Pipeline(
            name="ids",
            steps=[
                PipelineStep(employee="code-reviewer", id="review", args={}),
                PipelineStep(employee="test-engineer", id="test", args={}),
            ],
        )
        mermaid = pipeline_to_mermaid(pl)
        assert 'review["code-reviewer"]' in mermaid
        assert 'test["test-engineer"]' in mermaid
        assert "S --> review" in mermaid
        assert "review --> test" in mermaid
        assert "test --> E" in mermaid

    def test_mixed_sequential_and_parallel(self):
        pl = Pipeline(
            name="mixed",
            steps=[
                PipelineStep(employee="code-reviewer", id="review", args={}),
                ParallelGroup(
                    parallel=[
                        PipelineStep(employee="test-engineer", args={}),
                        PipelineStep(employee="refactor-guide", args={}),
                    ]
                ),
                PipelineStep(employee="pr-creator", id="pr", args={}),
            ],
        )
        mermaid = pipeline_to_mermaid(pl)
        # review → fork → parallel tasks → join → pr → end
        assert "review --> F1" in mermaid
        assert "J1" in mermaid
        assert 'pr["pr-creator"]' in mermaid
        assert "J1 --> pr" in mermaid
        assert "pr --> E" in mermaid


class TestOutputRefWarning:
    """输出引用解析 warning 日志."""

    def test_unresolved_ref_warns(self, caplog):
        """未解析的引用应产生 warning 日志."""
        import logging

        with caplog.at_level(logging.WARNING, logger="crew.pipeline"):
            result = _resolve_output_refs(
                "{steps.missing.output}",
                {},
                {},
                "",
                execute=True,
            )
        assert result == "{steps.missing.output}"
        assert "未解析输出引用" in caplog.text

    def test_unresolved_index_ref_warns(self, caplog):
        """未解析的索引引用应产生 warning 日志."""
        import logging

        with caplog.at_level(logging.WARNING, logger="crew.pipeline"):
            result = _resolve_output_refs(
                "{steps.99.output}",
                {},
                {},
                "",
                execute=True,
            )
        assert result == "{steps.99.output}"
        assert "未解析输出引用" in caplog.text

    def test_resolved_ref_no_warning(self, caplog):
        """已解析的引用不应产生 warning."""
        import logging

        with caplog.at_level(logging.WARNING, logger="crew.pipeline"):
            result = _resolve_output_refs(
                "{steps.review.output}",
                {"review": "good code"},
                {},
                "",
                execute=True,
            )
        assert result == "good code"
        assert "未解析输出引用" not in caplog.text

    def test_parallel_prev_contains_all(self):
        """并行组后 {prev} 应包含所有子步骤输出（通过 run_pipeline）."""
        pl = Pipeline(
            name="par-prev",
            steps=[
                ParallelGroup(
                    parallel=[
                        PipelineStep(employee="code-reviewer", id="a", args={"target": "main"}),
                        PipelineStep(employee="test-engineer", id="b", args={"target": "main"}),
                    ]
                ),
                PipelineStep(employee="doc-writer", args={"target": "main"}),
            ],
        )
        result = run_pipeline(pl, smart_context=False)
        assert len(result.steps) == 2
        # 并行组结果
        group = result.steps[0]
        assert isinstance(group, list)
        # 第二步应该存在（prompt-only 模式）
        assert result.steps[1].employee == "doc-writer"


class TestParallelStepErrorLogging:
    """并行步骤失败时产生 warning 日志."""

    def test_parallel_error_logged(self, caplog):
        """并行组中的步骤失败应记录 warning."""
        import logging

        # 使用不存在的员工触发 error
        pl = Pipeline(
            name="error-log-test",
            steps=[
                ParallelGroup(
                    parallel=[
                        PipelineStep(employee="nonexistent-employee", args={}),
                        PipelineStep(employee="code-reviewer", args={"target": "main"}),
                    ]
                ),
            ],
        )
        with caplog.at_level(logging.WARNING, logger="crew.pipeline"):
            result = run_pipeline(pl, smart_context=False)

        group = result.steps[0]
        assert isinstance(group, list)
        # 至少有一个错误步骤
        error_steps = [r for r in group if r.error]
        assert len(error_steps) >= 1
        assert "并行步骤" in caplog.text


class TestAsyncGatherTimeout:
    """异步 gather 超时保护."""

    def test_timeout_constant_exists(self):
        """超时常量应该有合理的值."""
        from crew.pipeline import _ASYNC_STEP_TIMEOUT

        assert _ASYNC_STEP_TIMEOUT > 0
        assert _ASYNC_STEP_TIMEOUT <= 3600  # 不超过 1 小时


# ── 条件分支 + 循环测试 ──


class TestConditionModel:
    """测试 Condition / ConditionalBody 数据模型."""

    def test_condition_contains(self):
        c = Condition(check="{prev}", contains="critical")
        assert c.contains == "critical"
        assert c.matches == ""

    def test_condition_matches(self):
        c = Condition(check="{prev}", matches=r"LGTM|approved")
        assert c.matches == r"LGTM|approved"

    def test_condition_both_raises(self):
        with pytest.raises(ValueError, match="之一"):
            Condition(check="{prev}", contains="a", matches="b")

    def test_condition_neither_raises(self):
        with pytest.raises(ValueError, match="之一"):
            Condition(check="{prev}")

    def test_evaluate_contains_true(self):
        c = Condition(check="{prev}", contains="LGTM")
        assert c.evaluate("代码审查: LGTM，可以合并") is True

    def test_evaluate_contains_false(self):
        c = Condition(check="{prev}", contains="LGTM")
        assert c.evaluate("需要修改") is False

    def test_evaluate_matches_true(self):
        c = Condition(check="{prev}", matches=r"score:\s*\d+")
        assert c.evaluate("score: 95") is True

    def test_evaluate_matches_false(self):
        c = Condition(check="{prev}", matches=r"score:\s*\d+")
        assert c.evaluate("no score here") is False

    def test_conditional_body(self):
        body = ConditionalBody(
            check="{prev}",
            contains="critical",
            then=[PipelineStep(employee="security-auditor")],
            **{"else": [PipelineStep(employee="code-reviewer")]},
        )
        assert len(body.then) == 1
        assert len(body.else_) == 1
        assert body.evaluate("this is critical") is True
        assert body.evaluate("all good") is False

    def test_conditional_body_no_else(self):
        body = ConditionalBody(
            check="{prev}",
            contains="x",
            then=[PipelineStep(employee="a")],
        )
        assert body.else_ == []

    def test_conditional_step(self):
        step = ConditionalStep(
            condition=ConditionalBody(
                check="{prev}",
                contains="yes",
                then=[PipelineStep(employee="a")],
            )
        )
        assert step.condition.check == "{prev}"

    def test_loop_body(self):
        body = LoopBody(
            steps=[PipelineStep(employee="code-reviewer", id="review")],
            until=Condition(check="{steps.review.output}", contains="LGTM"),
            max_iterations=3,
        )
        assert body.max_iterations == 3
        assert len(body.steps) == 1

    def test_loop_step(self):
        step = LoopStep(
            loop=LoopBody(
                steps=[PipelineStep(employee="a")],
                until=Condition(check="{prev}", contains="done"),
            )
        )
        assert step.loop.max_iterations == 5  # default

    def test_loop_max_iterations_bounds(self):
        with pytest.raises(Exception):
            LoopBody(
                steps=[PipelineStep(employee="a")],
                until=Condition(check="{prev}", contains="x"),
                max_iterations=0,
            )
        with pytest.raises(Exception):
            LoopBody(
                steps=[PipelineStep(employee="a")],
                until=Condition(check="{prev}", contains="x"),
                max_iterations=51,
            )

    def test_step_result_branch_field(self):
        r = StepResult(employee="a", step_index=0, args={}, prompt="p")
        assert r.branch == ""
        r2 = StepResult(employee="a", step_index=0, args={}, prompt="p", branch="then")
        assert r2.branch == "then"


class TestEvaluateCheck:
    """测试 _evaluate_check 辅助函数."""

    def test_prompt_only_always_true(self):
        assert _evaluate_check("{prev}", "x", "", {}, {}, "", execute=False) is True

    def test_contains_match(self):
        assert _evaluate_check("{prev}", "ok", "", {}, {}, "all ok", execute=True) is True

    def test_contains_no_match(self):
        assert _evaluate_check("{prev}", "ok", "", {}, {}, "bad", execute=True) is False

    def test_matches_regex(self):
        assert _evaluate_check("{prev}", "", r"\d+", {}, {}, "score: 42", execute=True) is True

    def test_matches_no_match(self):
        assert _evaluate_check("{prev}", "", r"\d+", {}, {}, "no numbers", execute=True) is False

    def test_resolves_step_ref(self):
        assert (
            _evaluate_check(
                "{steps.review.output}",
                "LGTM",
                "",
                {"review": "LGTM, ship it"},
                {},
                "",
                execute=True,
            )
            is True
        )


class TestLoadPipelineConditionLoop:
    """测试从 YAML 加载条件/循环步骤."""

    def test_load_condition(self, tmp_path):
        data = {
            "name": "cond-test",
            "steps": [
                {"employee": "classifier", "id": "classify"},
                {
                    "condition": {
                        "check": "{steps.classify.output}",
                        "contains": "critical",
                        "then": [{"employee": "security-auditor"}],
                        "else": [{"employee": "code-reviewer"}],
                    }
                },
            ],
        }
        f = tmp_path / "cond.yaml"
        f.write_text(yaml.dump(data, allow_unicode=True))
        pl = load_pipeline(f)
        assert len(pl.steps) == 2
        assert isinstance(pl.steps[0], PipelineStep)
        assert isinstance(pl.steps[1], ConditionalStep)
        assert pl.steps[1].condition.contains == "critical"
        assert len(pl.steps[1].condition.then) == 1
        assert len(pl.steps[1].condition.else_) == 1

    def test_load_condition_no_else(self, tmp_path):
        data = {
            "name": "cond-no-else",
            "steps": [
                {
                    "condition": {
                        "check": "{prev}",
                        "contains": "skip",
                        "then": [{"employee": "code-reviewer"}],
                    }
                },
            ],
        }
        f = tmp_path / "cond2.yaml"
        f.write_text(yaml.dump(data, allow_unicode=True))
        pl = load_pipeline(f)
        assert isinstance(pl.steps[0], ConditionalStep)
        assert pl.steps[0].condition.else_ == []

    def test_load_loop(self, tmp_path):
        data = {
            "name": "loop-test",
            "steps": [
                {
                    "loop": {
                        "steps": [
                            {"employee": "code-reviewer", "id": "review"},
                        ],
                        "until": {"check": "{steps.review.output}", "contains": "LGTM"},
                        "max_iterations": 3,
                    }
                },
            ],
        }
        f = tmp_path / "loop.yaml"
        f.write_text(yaml.dump(data, allow_unicode=True))
        pl = load_pipeline(f)
        assert len(pl.steps) == 1
        assert isinstance(pl.steps[0], LoopStep)
        assert pl.steps[0].loop.max_iterations == 3
        assert pl.steps[0].loop.until.contains == "LGTM"

    def test_load_mixed_all_types(self, tmp_path):
        data = {
            "name": "mixed",
            "steps": [
                {"employee": "classifier", "id": "c"},
                {
                    "parallel": [
                        {"employee": "a"},
                        {"employee": "b"},
                    ]
                },
                {
                    "condition": {
                        "check": "{prev}",
                        "contains": "yes",
                        "then": [{"employee": "x"}],
                    }
                },
                {
                    "loop": {
                        "steps": [{"employee": "y"}],
                        "until": {"check": "{prev}", "contains": "done"},
                    }
                },
                {"employee": "z"},
            ],
        }
        f = tmp_path / "mixed.yaml"
        f.write_text(yaml.dump(data, allow_unicode=True))
        pl = load_pipeline(f)
        assert len(pl.steps) == 5
        assert isinstance(pl.steps[0], PipelineStep)
        assert isinstance(pl.steps[1], ParallelGroup)
        assert isinstance(pl.steps[2], ConditionalStep)
        assert isinstance(pl.steps[3], LoopStep)
        assert isinstance(pl.steps[4], PipelineStep)


class TestValidateConditionLoop:
    """测试条件/循环步骤的校验."""

    def test_validate_condition_unknown_employee(self):
        pl = Pipeline(
            name="bad",
            steps=[
                ConditionalStep(
                    condition=ConditionalBody(
                        check="{prev}",
                        contains="x",
                        then=[PipelineStep(employee="nonexistent-worker")],
                    )
                )
            ],
        )
        errors = validate_pipeline(pl)
        assert any("nonexistent-worker" in e for e in errors)

    def test_validate_condition_else_unknown(self):
        pl = Pipeline(
            name="bad2",
            steps=[
                ConditionalStep(
                    condition=ConditionalBody(
                        check="{prev}",
                        contains="x",
                        then=[PipelineStep(employee="code-reviewer")],
                        **{"else": [PipelineStep(employee="nonexistent-worker")]},
                    )
                )
            ],
        )
        errors = validate_pipeline(pl)
        assert any("nonexistent-worker" in e for e in errors)

    def test_validate_loop_unknown_employee(self):
        pl = Pipeline(
            name="bad3",
            steps=[
                LoopStep(
                    loop=LoopBody(
                        steps=[PipelineStep(employee="nonexistent-worker")],
                        until=Condition(check="{prev}", contains="x"),
                    )
                )
            ],
        )
        errors = validate_pipeline(pl)
        assert any("nonexistent-worker" in e for e in errors)

    def test_validate_condition_valid(self):
        pl = Pipeline(
            name="ok",
            steps=[
                ConditionalStep(
                    condition=ConditionalBody(
                        check="{prev}",
                        contains="x",
                        then=[PipelineStep(employee="code-reviewer")],
                    )
                )
            ],
        )
        errors = validate_pipeline(pl)
        assert errors == []

    def test_validate_loop_valid(self):
        pl = Pipeline(
            name="ok",
            steps=[
                LoopStep(
                    loop=LoopBody(
                        steps=[PipelineStep(employee="code-reviewer")],
                        until=Condition(check="{prev}", contains="x"),
                    )
                )
            ],
        )
        errors = validate_pipeline(pl)
        assert errors == []


class TestRunPipelineCondition:
    """测试条件分支执行."""

    def test_condition_prompt_only_defaults_then(self):
        """prompt-only 模式默认走 then 分支."""
        pl = Pipeline(
            name="cond-prompt",
            steps=[
                PipelineStep(employee="code-reviewer", args={"target": "main"}),
                ConditionalStep(
                    condition=ConditionalBody(
                        check="{prev}",
                        contains="critical",
                        then=[PipelineStep(employee="test-engineer", args={"target": "main"})],
                        **{"else": [PipelineStep(employee="doc-writer", args={"target": "main"})]},
                    )
                ),
            ],
        )
        result = run_pipeline(pl, smart_context=False)
        assert result.pipeline_name == "cond-prompt"
        # prompt-only 默认走 then → test-engineer
        flat = [r for item in result.steps for r in (item if isinstance(item, list) else [item])]
        employees = [r.employee for r in flat]
        assert "test-engineer" in employees
        assert "doc-writer" not in employees

    def test_condition_then_branch_value(self):
        """条件分支中的步骤应有 branch='then' 标记."""
        pl = Pipeline(
            name="branch-label",
            steps=[
                ConditionalStep(
                    condition=ConditionalBody(
                        check="{prev}",
                        contains="x",
                        then=[PipelineStep(employee="code-reviewer", args={"target": "main"})],
                    )
                ),
            ],
        )
        result = run_pipeline(pl, smart_context=False)
        step = result.steps[0]
        if isinstance(step, list):
            step = step[0]
        assert step.branch == "then"

    def test_condition_flat_index(self):
        """条件分支后的 flat_index 应正确递增."""
        pl = Pipeline(
            name="idx",
            steps=[
                PipelineStep(employee="code-reviewer"),  # flat=0
                ConditionalStep(
                    condition=ConditionalBody(
                        check="{prev}",
                        contains="x",
                        then=[
                            PipelineStep(employee="test-engineer"),  # flat=1
                            PipelineStep(employee="doc-writer"),  # flat=2
                        ],
                    )
                ),
                PipelineStep(employee="code-reviewer"),  # flat=3
            ],
        )
        result = run_pipeline(pl, smart_context=False)
        assert result.steps[0].step_index == 0
        # 条件分支产生 2 个步骤
        group = result.steps[1]
        assert isinstance(group, list)
        assert len(group) == 2
        assert {r.step_index for r in group} == {1, 2}
        # 最后一步
        assert result.steps[2].step_index == 3

    def test_prev_after_condition(self):
        """条件分支后 {prev} 指向分支最后一步的输出."""
        pl = Pipeline(
            name="prev-cond",
            steps=[
                ConditionalStep(
                    condition=ConditionalBody(
                        check="{prev}",
                        contains="x",
                        then=[PipelineStep(employee="code-reviewer", args={"target": "main"})],
                    )
                ),
                PipelineStep(employee="test-engineer", args={"target": "main"}),
            ],
        )
        result = run_pipeline(pl, smart_context=False)
        # 两步都应成功执行（prompt-only）
        flat = [r for item in result.steps for r in (item if isinstance(item, list) else [item])]
        assert len(flat) == 2


class TestRunPipelineLoop:
    """测试循环执行."""

    def test_loop_prompt_only_single_iteration(self):
        """prompt-only 模式只执行一次迭代."""
        pl = Pipeline(
            name="loop-prompt",
            steps=[
                LoopStep(
                    loop=LoopBody(
                        steps=[
                            PipelineStep(employee="code-reviewer", args={"target": "main"}),
                            PipelineStep(employee="test-engineer", args={"target": "main"}),
                        ],
                        until=Condition(check="{prev}", contains="LGTM"),
                        max_iterations=5,
                    )
                ),
            ],
        )
        result = run_pipeline(pl, smart_context=False)
        # prompt-only: 只执行一次迭代 = 2 个步骤
        flat = [r for item in result.steps for r in (item if isinstance(item, list) else [item])]
        assert len(flat) == 2

    def test_loop_branch_label(self):
        """循环内步骤应有 branch='loop-N' 标记."""
        pl = Pipeline(
            name="loop-label",
            steps=[
                LoopStep(
                    loop=LoopBody(
                        steps=[PipelineStep(employee="code-reviewer", args={"target": "main"})],
                        until=Condition(check="{prev}", contains="LGTM"),
                        max_iterations=1,
                    )
                ),
            ],
        )
        result = run_pipeline(pl, smart_context=False)
        step = result.steps[0]
        if isinstance(step, list):
            step = step[0]
        assert step.branch == "loop-0"

    def test_loop_flat_index(self):
        """循环步骤的 flat_index 应正确递增."""
        pl = Pipeline(
            name="loop-idx",
            steps=[
                PipelineStep(employee="code-reviewer"),  # flat=0
                LoopStep(
                    loop=LoopBody(
                        steps=[
                            PipelineStep(employee="test-engineer")
                        ],  # flat=1 (single iteration prompt-only)
                        until=Condition(check="{prev}", contains="done"),
                    )
                ),
                PipelineStep(employee="code-reviewer"),  # flat=2
            ],
        )
        result = run_pipeline(pl, smart_context=False)
        assert result.steps[0].step_index == 0
        # loop 产生 1 步（单次迭代）
        loop_step = result.steps[1]
        if isinstance(loop_step, list):
            assert loop_step[0].step_index == 1
        else:
            assert loop_step.step_index == 1
        assert result.steps[2].step_index == 2


class TestAsyncConditionLoop:
    """测试异步版本的条件/循环."""

    def test_async_condition(self):
        pl = Pipeline(
            name="async-cond",
            steps=[
                ConditionalStep(
                    condition=ConditionalBody(
                        check="{prev}",
                        contains="x",
                        then=[PipelineStep(employee="code-reviewer", args={"target": "main"})],
                    )
                ),
            ],
        )
        result = _run(arun_pipeline(pl, smart_context=False))
        flat = [r for item in result.steps for r in (item if isinstance(item, list) else [item])]
        assert len(flat) == 1
        assert flat[0].employee == "code-reviewer"

    def test_async_loop(self):
        pl = Pipeline(
            name="async-loop",
            steps=[
                LoopStep(
                    loop=LoopBody(
                        steps=[PipelineStep(employee="code-reviewer", args={"target": "main"})],
                        until=Condition(check="{prev}", contains="LGTM"),
                        max_iterations=2,
                    )
                ),
            ],
        )
        result = _run(arun_pipeline(pl, smart_context=False))
        flat = [r for item in result.steps for r in (item if isinstance(item, list) else [item])]
        assert len(flat) >= 1

    def test_async_checkpoint_after_condition(self):
        """条件分支后 checkpoint 应正确保存."""
        pl = Pipeline(
            name="cp-cond",
            steps=[
                ConditionalStep(
                    condition=ConditionalBody(
                        check="{prev}",
                        contains="x",
                        then=[PipelineStep(employee="code-reviewer", args={"target": "main"})],
                    )
                ),
                PipelineStep(employee="test-engineer", args={"target": "main"}),
            ],
        )
        checkpoints = []

        def on_complete(step_result, checkpoint):
            checkpoints.append(checkpoint)

        _run(arun_pipeline(pl, smart_context=False, on_step_complete=on_complete))
        assert len(checkpoints) == 2


class TestMermaidConditionLoop:
    """测试条件/循环的 Mermaid 流程图生成."""

    def test_mermaid_condition(self):
        pl = Pipeline(
            name="cond",
            steps=[
                PipelineStep(employee="classifier", id="classify"),
                ConditionalStep(
                    condition=ConditionalBody(
                        check="{steps.classify.output}",
                        contains="critical",
                        then=[PipelineStep(employee="security-auditor")],
                        **{"else": [PipelineStep(employee="code-reviewer")]},
                    )
                ),
            ],
        )
        mermaid = pipeline_to_mermaid(pl)
        assert "contains 'critical'" in mermaid
        assert '"security-auditor"' in mermaid
        assert '"code-reviewer"' in mermaid
        assert "then" in mermaid
        assert "else" in mermaid
        assert "合并" in mermaid

    def test_mermaid_condition_no_else(self):
        pl = Pipeline(
            name="cond-no-else",
            steps=[
                ConditionalStep(
                    condition=ConditionalBody(
                        check="{prev}",
                        contains="skip",
                        then=[PipelineStep(employee="code-reviewer")],
                    )
                ),
            ],
        )
        mermaid = pipeline_to_mermaid(pl)
        assert "contains 'skip'" in mermaid
        assert '"code-reviewer"' in mermaid
        assert "then" in mermaid

    def test_mermaid_loop(self):
        pl = Pipeline(
            name="loop",
            steps=[
                LoopStep(
                    loop=LoopBody(
                        steps=[PipelineStep(employee="code-reviewer", id="review")],
                        until=Condition(check="{steps.review.output}", contains="LGTM"),
                        max_iterations=3,
                    )
                ),
            ],
        )
        mermaid = pipeline_to_mermaid(pl)
        assert "循环 (max 3)" in mermaid
        assert '"code-reviewer"' in mermaid
        assert "contains 'LGTM'" in mermaid
        assert "继续" in mermaid

    def test_mermaid_mixed(self):
        """混合所有类型的完整流程图."""
        pl = Pipeline(
            name="mixed",
            steps=[
                PipelineStep(employee="start-worker", id="start"),
                ParallelGroup(
                    parallel=[
                        PipelineStep(employee="a"),
                        PipelineStep(employee="b"),
                    ]
                ),
                ConditionalStep(
                    condition=ConditionalBody(
                        check="{prev}",
                        contains="yes",
                        then=[PipelineStep(employee="x")],
                        **{"else": [PipelineStep(employee="y")]},
                    )
                ),
                LoopStep(
                    loop=LoopBody(
                        steps=[PipelineStep(employee="z")],
                        until=Condition(check="{prev}", contains="done"),
                        max_iterations=2,
                    )
                ),
            ],
        )
        mermaid = pipeline_to_mermaid(pl)
        assert "graph LR" in mermaid
        assert "开始" in mermaid
        assert "结束" in mermaid
        assert "并行" in mermaid
        assert "合并" in mermaid  # both parallel and condition merges
        assert "循环" in mermaid
        assert "继续" in mermaid


class TestPipelineWithConditionLoop:
    """测试 Pipeline 模型直接构造条件/循环."""

    def test_pipeline_with_condition(self):
        pl = Pipeline(
            name="test",
            steps=[
                PipelineStep(employee="a"),
                ConditionalStep(
                    condition=ConditionalBody(
                        check="{prev}",
                        contains="x",
                        then=[PipelineStep(employee="b")],
                    )
                ),
            ],
        )
        assert len(pl.steps) == 2
        assert isinstance(pl.steps[1], ConditionalStep)

    def test_pipeline_with_loop(self):
        pl = Pipeline(
            name="test",
            steps=[
                LoopStep(
                    loop=LoopBody(
                        steps=[PipelineStep(employee="a")],
                        until=Condition(check="{prev}", contains="done"),
                    )
                ),
                PipelineStep(employee="b"),
            ],
        )
        assert len(pl.steps) == 2
        assert isinstance(pl.steps[0], LoopStep)
