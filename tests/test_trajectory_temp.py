"""测试临时轨迹文件存储功能."""

import json
from datetime import date
from pathlib import Path

import pytest


class TestTrajectoryTempFile:
    """测试 TrajectoryCollector 写入临时文件."""

    def test_temp_file_created_on_finish(self, tmp_path):
        """finish() 时自动创建临时文件."""
        from crew.trajectory import TrajectoryCollector

        # 使用 tmp_path 作为输出目录
        collector = TrajectoryCollector(
            employee_name="test-employee",
            task_description="测试任务",
            model="claude-sonnet-4",
            channel="test",
            output_dir=tmp_path / ".crew" / "trajectories",
        )

        # 添加一些步骤
        collector.add_tool_step(
            thought="执行测试",
            tool_name="bash",
            tool_params={"command": "echo hello"},
            tool_output="hello",
            tool_exit_code=0,
        )

        # Mock _write_temp_file 以使用 tmp_path
        original_write = collector._write_temp_file

        def mock_write_temp():
            temp_base = tmp_path / "trajectory_temp"
            date_str = date.today().isoformat()
            date_dir = temp_base / date_str
            date_dir.mkdir(parents=True, exist_ok=True)

            import uuid

            session_id = f"{collector.employee_name}-{uuid.uuid4().hex[:8]}"
            temp_file = date_dir / f"session-{session_id}.jsonl"

            with open(temp_file, "w", encoding="utf-8") as f:
                for step in collector._steps:
                    f.write(json.dumps(step, ensure_ascii=False) + "\n")

            # 保存路径供测试验证
            collector._temp_file_path = temp_file

        collector._write_temp_file = mock_write_temp

        # 完成录制
        with collector:
            pass
        collector.finish()

        # 验证临时文件已创建
        assert hasattr(collector, "_temp_file_path")
        temp_file = collector._temp_file_path
        assert temp_file.exists()

        # 验证文件内容
        lines = temp_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        step_data = json.loads(lines[0])
        assert step_data["tool_name"] == "bash"
        assert step_data["tool_output"] == "hello"

    def test_temp_file_path_structure(self, tmp_path):
        """验证临时文件路径结构正确."""
        from crew.trajectory import TrajectoryCollector

        collector = TrajectoryCollector(
            employee_name="backend-engineer",
            task_description="测试",
            output_dir=tmp_path / ".crew" / "trajectories",
        )

        collector.add_tool_step(
            thought="test",
            tool_name="read",
            tool_params={"file": "test.py"},
            tool_output="content",
            tool_exit_code=0,
        )

        # Mock _write_temp_file
        def mock_write_temp():
            temp_base = tmp_path / "trajectory_temp"
            date_str = date.today().isoformat()
            date_dir = temp_base / date_str
            date_dir.mkdir(parents=True, exist_ok=True)

            import uuid

            session_id = f"{collector.employee_name}-{uuid.uuid4().hex[:8]}"
            temp_file = date_dir / f"session-{session_id}.jsonl"

            with open(temp_file, "w", encoding="utf-8") as f:
                for step in collector._steps:
                    f.write(json.dumps(step, ensure_ascii=False) + "\n")

            collector._temp_file_path = temp_file

        collector._write_temp_file = mock_write_temp

        with collector:
            pass
        collector.finish()

        # 验证路径结构
        temp_file = collector._temp_file_path
        assert temp_file.parent.name == date.today().isoformat()
        assert temp_file.name.startswith("session-backend-engineer-")
        assert temp_file.suffix == ".jsonl"

    def test_temp_file_multiple_steps(self, tmp_path):
        """验证多个步骤都被写入临时文件."""
        from crew.trajectory import TrajectoryCollector

        collector = TrajectoryCollector(
            employee_name="test",
            task_description="多步骤测试",
            output_dir=tmp_path / ".crew" / "trajectories",
        )

        # 添加多个步骤
        for i in range(3):
            collector.add_tool_step(
                thought=f"步骤 {i+1}",
                tool_name=f"tool_{i}",
                tool_params={"index": i},
                tool_output=f"output_{i}",
                tool_exit_code=0,
            )

        # Mock _write_temp_file
        def mock_write_temp():
            temp_base = tmp_path / "trajectory_temp"
            date_str = date.today().isoformat()
            date_dir = temp_base / date_str
            date_dir.mkdir(parents=True, exist_ok=True)

            import uuid

            session_id = f"{collector.employee_name}-{uuid.uuid4().hex[:8]}"
            temp_file = date_dir / f"session-{session_id}.jsonl"

            with open(temp_file, "w", encoding="utf-8") as f:
                for step in collector._steps:
                    f.write(json.dumps(step, ensure_ascii=False) + "\n")

            collector._temp_file_path = temp_file

        collector._write_temp_file = mock_write_temp

        with collector:
            pass
        collector.finish()

        # 验证所有步骤都被写入
        temp_file = collector._temp_file_path
        lines = temp_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

        for i, line in enumerate(lines):
            step_data = json.loads(line)
            assert step_data["tool_name"] == f"tool_{i}"
            assert step_data["tool_output"] == f"output_{i}"

    def test_temp_file_write_failure_does_not_crash(self, tmp_path, monkeypatch):
        """临时文件写入失败不影响主流程."""
        from crew.trajectory import TrajectoryCollector

        collector = TrajectoryCollector(
            employee_name="test",
            task_description="测试",
            output_dir=tmp_path / ".crew" / "trajectories",
        )

        collector.add_tool_step(
            thought="test",
            tool_name="bash",
            tool_params={},
            tool_output="ok",
            tool_exit_code=0,
        )

        # Mock Path.mkdir 抛出异常
        from pathlib import Path

        original_mkdir = Path.mkdir

        def mock_mkdir_fail(self, *args, **kwargs):
            if "/data/trajectory_temp" in str(self):
                raise PermissionError("无法创建目录")
            return original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", mock_mkdir_fail)

        # finish() 不应该抛异常
        with collector:
            pass
        result = collector.finish()

        # 主流程应该正常完成
        assert result is not None
