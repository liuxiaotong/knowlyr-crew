"""轨迹录制测试 — trajectory.py."""

from crew.trajectory import TrajectoryCollector


class TestTrajectoryCollectorContextVar:
    def test_current_none_outside_context(self):
        """上下文外 current() 返回 None."""
        assert TrajectoryCollector.current() is None

    def test_current_inside_context(self):
        tc = TrajectoryCollector("emp", "task")
        with tc:
            assert TrajectoryCollector.current() is tc

    def test_current_restored_after_exit(self):
        tc = TrajectoryCollector("emp", "task")
        with tc:
            pass
        assert TrajectoryCollector.current() is None

    def test_nested_context(self):
        """嵌套上下文正确恢复."""
        tc1 = TrajectoryCollector("emp1", "task1")
        tc2 = TrajectoryCollector("emp2", "task2")
        with tc1:
            assert TrajectoryCollector.current() is tc1
            with tc2:
                assert TrajectoryCollector.current() is tc2
            assert TrajectoryCollector.current() is tc1
        assert TrajectoryCollector.current() is None


class TestAddPromptStep:
    def test_records_step(self):
        tc = TrajectoryCollector("emp", "task")
        tc.add_prompt_step("hello", "claude-sonnet", 100, 50)
        assert len(tc._steps) == 1
        step = tc._steps[0]
        assert step["thought"] == "hello"
        assert step["tool_name"] == "respond"
        assert step["token_count"] == 150

    def test_sets_model_on_first_call(self):
        tc = TrajectoryCollector("emp", "task")
        tc.add_prompt_step("a", "model-a", 10, 5)
        assert tc.model == "model-a"
        # 后续不覆盖
        tc.add_prompt_step("b", "model-b", 10, 5)
        assert tc.model == "model-a"

    def test_step_count_increments(self):
        tc = TrajectoryCollector("emp", "task")
        tc.add_prompt_step("a", "m", 10, 5)
        tc.add_prompt_step("b", "m", 10, 5)
        assert tc._steps[0]["step_id"] == 1
        assert tc._steps[1]["step_id"] == 2


class TestBeginCompleteToolStep:
    def test_begin_complete_cycle(self):
        tc = TrajectoryCollector("emp", "task")
        tc.begin_tool_step("thinking", "bash", {"cmd": "ls"}, 100, 50)
        assert tc._pending is not None
        assert len(tc._steps) == 0

        tc.complete_tool_step("file1.py\nfile2.py", 0)
        assert tc._pending is None
        assert len(tc._steps) == 1
        step = tc._steps[0]
        assert step["tool_name"] == "bash"
        assert step["tool_output"] == "file1.py\nfile2.py"
        assert step["tool_exit_code"] == 0

    def test_begin_without_complete_auto_saves(self):
        """连续 begin 自动保存上一个未完成步骤."""
        tc = TrajectoryCollector("emp", "task")
        tc.begin_tool_step("step1", "bash", {}, 10, 5)
        tc.begin_tool_step("step2", "grep", {}, 10, 5)

        assert len(tc._steps) == 1
        assert tc._steps[0]["tool_name"] == "bash"
        assert tc._steps[0]["tool_exit_code"] == -1  # 未完成标记

    def test_complete_without_begin_noop(self):
        tc = TrajectoryCollector("emp", "task")
        tc.complete_tool_step("output", 0)
        assert len(tc._steps) == 0

    def test_begin_sets_model(self):
        tc = TrajectoryCollector("emp", "task")
        tc.begin_tool_step("t", "bash", {}, model="claude-opus")
        assert tc.model == "claude-opus"


class TestAddToolStep:
    def test_one_shot(self):
        tc = TrajectoryCollector("emp", "task")
        tc.add_tool_step(
            thought="reading",
            tool_name="file_read",
            tool_params={"path": "/tmp/a"},
            tool_output="content",
            tool_exit_code=0,
            input_tokens=50,
            output_tokens=30,
        )
        assert len(tc._steps) == 1
        step = tc._steps[0]
        assert step["tool_name"] == "file_read"
        assert step["token_count"] == 80


class TestFinish:
    def test_empty_steps_returns_none(self):
        tc = TrajectoryCollector("emp", "task")
        result = tc.finish()
        assert result is None

    def test_finish_writes_jsonl(self, tmp_path):
        """finish 写入 JSONL 文件."""
        tc = TrajectoryCollector("emp", "task", output_dir=tmp_path)
        tc.add_prompt_step("hello", "model", 100, 50)
        result = tc.finish()

        assert result is not None
        output_file = tmp_path / "trajectories.jsonl"
        assert output_file.exists()

        # 结果可能是 Trajectory 对象或 dict（取决于 agentrecorder 是否安装）
        if isinstance(result, dict):
            assert result["employee"] == "emp"
            assert result["total_steps"] == 1
            assert result["total_tokens"] == 150
        else:
            # agentrecorder Trajectory 对象
            assert result.agent == "crew/emp"
            assert result.outcome.total_steps == 1
            assert result.outcome.total_tokens == 150

    def test_finish_saves_pending(self, tmp_path):
        """finish() 保存未完成的 pending step."""
        tc = TrajectoryCollector("emp", "task", output_dir=tmp_path)
        tc.begin_tool_step("thinking", "bash", {"cmd": "ls"}, 10, 5)
        result = tc.finish()

        assert result is not None
        if isinstance(result, dict):
            assert len(result["steps"]) == 1
            assert result["steps"][0]["tool_exit_code"] == -1
        else:
            assert len(result.steps) == 1

    def test_finish_appends_to_existing(self, tmp_path):
        """多次 finish 追加到同一文件."""
        tc1 = TrajectoryCollector("emp1", "task1", output_dir=tmp_path)
        tc1.add_prompt_step("a", "m", 10, 5)
        tc1.finish()

        tc2 = TrajectoryCollector("emp2", "task2", output_dir=tmp_path)
        tc2.add_prompt_step("b", "m", 10, 5)
        tc2.finish()

        output_file = tmp_path / "trajectories.jsonl"
        lines = [l for l in output_file.read_text("utf-8").strip().split("\n") if l]
        assert len(lines) == 2

    def test_finish_with_failure(self, tmp_path):
        tc = TrajectoryCollector("emp", "task", output_dir=tmp_path)
        tc.add_prompt_step("error", "m", 10, 5)
        result = tc.finish(success=False, score=0.3)

        if isinstance(result, dict):
            assert result["success"] is False
        else:
            assert result.outcome.success is False
            assert result.outcome.score == 0.3

    def test_finish_creates_output_dir(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "dir"
        tc = TrajectoryCollector("emp", "task", output_dir=nested)
        tc.add_prompt_step("x", "m", 10, 5)
        tc.finish()
        assert (nested / "trajectories.jsonl").exists()


class TestChannelMetadata:
    """channel 字段测试."""

    def test_default_channel_is_cli(self):
        tc = TrajectoryCollector("emp", "task")
        assert tc.channel == "cli"

    def test_custom_channel(self):
        tc = TrajectoryCollector("emp", "task", channel="feishu")
        assert tc.channel == "feishu"

    def test_channel_in_finish_output(self, tmp_path):
        tc = TrajectoryCollector("emp", "task", channel="delegate", output_dir=tmp_path)
        tc.add_prompt_step("hello", "model", 100, 50)
        result = tc.finish()
        if isinstance(result, dict):
            assert result["channel"] == "delegate"
        else:
            assert result.metadata["channel"] == "delegate"

    def test_all_channel_values(self):
        """所有预期通道值都能正常创建."""
        for ch in ("cli", "pipeline", "feishu", "delegate", "api", "site_message"):
            tc = TrajectoryCollector("emp", "task", channel=ch)
            assert tc.channel == ch


class TestToolLoopSummary:
    """工具循环汇总步骤测试（对应 _execute_employee_with_tools 的录制）."""

    def test_agent_loop_step(self):
        tc = TrajectoryCollector("code-reviewer", "审查代码", channel="feishu")
        tc.add_tool_step(
            thought="[tool-loop] 3 rounds",
            tool_name="agent_loop",
            tool_params={"employee": "code-reviewer", "rounds": 3},
            tool_output="审查完成，发现 2 个问题",
            tool_exit_code=0,
            input_tokens=5000,
            output_tokens=1000,
        )
        assert len(tc._steps) == 1
        assert tc._steps[0]["tool_name"] == "agent_loop"
        assert tc._steps[0]["token_count"] == 6000

    def test_mixed_prompt_and_tool_loop(self, tmp_path):
        """单轮回复 + 工具循环汇总可以共存."""
        tc = TrajectoryCollector("emp", "task", channel="delegate", output_dir=tmp_path)
        tc.add_prompt_step("先想一想", "claude", 100, 50)
        tc.add_tool_step(
            thought="[tool-loop] 2 rounds",
            tool_name="agent_loop",
            tool_params={"employee": "emp", "rounds": 2},
            tool_output="完成",
            input_tokens=3000,
            output_tokens=500,
        )
        result = tc.finish()
        assert result is not None
        if isinstance(result, dict):
            assert result["total_steps"] == 2
            assert result["total_tokens"] == 3650  # 150 + 3500
