"""轨迹数据集导出测试."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from crew.trajectory_export import TrajectoryExporter


@pytest.fixture
def temp_dirs(tmp_path):
    """创建临时目录."""
    archive_dir = tmp_path / "archive"
    annotations_dir = tmp_path / "annotations"
    archive_dir.mkdir()
    annotations_dir.mkdir()
    return archive_dir, annotations_dir


@pytest.fixture
def exporter(temp_dirs):
    """创建测试用的导出器."""
    archive_dir, annotations_dir = temp_dirs
    return TrajectoryExporter(
        archive_dir=archive_dir,
        annotations_dir=annotations_dir,
    )


@pytest.fixture
def sample_trajectories(temp_dirs):
    """创建示例轨迹数据（新格式）."""
    archive_dir, _ = temp_dirs

    # 创建日期目录
    date_dir = archive_dir / "2026-03-01"
    date_dir.mkdir()

    # 创建轨迹文件
    traj_file = date_dir / "赵云帆-test.jsonl"

    trajectories = [
        {
            "trajectory_id": "traj-001",
            "employee": "赵云帆",
            "task": "实现用户认证",
            "model": "claude-sonnet-4",
            "channel": "cli",
            "created_at": "2026-03-01T10:00:00Z",
            "success": True,
            "metadata": {
                "total_steps": 2,
                "total_tokens": 1000,
                "duration_ms": 5000,
            },
            "trajectory": [
                {
                    "step": 1,
                    "observation": "",
                    "thought": "检查现有认证逻辑",
                    "action": {
                        "tool": "Read",
                        "parameters": {},
                    },
                    "result": "已读取认证模块",
                    "success": True,
                    "timestamp": "2026-03-01T10:00:01Z",
                },
                {
                    "step": 2,
                    "observation": "",
                    "thought": "添加 JWT 验证",
                    "action": {
                        "tool": "Edit",
                        "parameters": {},
                    },
                    "result": "已更新认证逻辑",
                    "success": True,
                    "timestamp": "2026-03-01T10:00:02Z",
                },
            ],
        },
        {
            "trajectory_id": "traj-002",
            "employee": "卫子昂",
            "task": "优化前端性能",
            "model": "claude-sonnet-4",
            "channel": "cli",
            "created_at": "2026-03-01T11:00:00Z",
            "success": True,
            "metadata": {
                "total_steps": 1,
                "total_tokens": 800,
                "duration_ms": 3000,
            },
            "trajectory": [
                {
                    "step": 1,
                    "observation": "",
                    "thought": "分析性能瓶颈",
                    "action": {
                        "tool": "Bash",
                        "parameters": {},
                    },
                    "result": "发现渲染性能问题",
                    "success": True,
                    "timestamp": "2026-03-01T11:00:01Z",
                }
            ],
        },
    ]

    with open(traj_file, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    return trajectories


class TestTrajectoryExporter:
    """测试轨迹导出器."""

    def test_load_trajectories(self, exporter, sample_trajectories):
        """测试加载轨迹."""
        trajectories = exporter.load_trajectories()

        assert len(trajectories) == 2
        assert trajectories[0]["trajectory_id"] == "traj-001"
        assert trajectories[1]["trajectory_id"] == "traj-002"

    def test_load_trajectories_filter_by_employee(self, exporter, sample_trajectories):
        """测试按员工过滤."""
        trajectories = exporter.load_trajectories(employee="赵云帆")

        assert len(trajectories) == 1
        assert trajectories[0]["employee"] == "赵云帆"

    def test_convert_to_training_example(self, exporter, sample_trajectories):
        """测试转换为训练样本."""
        traj = sample_trajectories[0]
        example = exporter.convert_to_training_example(traj)

        assert example.input == "实现用户认证"
        assert "已更新认证逻辑" in example.output
        assert "Step 1" in example.reasoning
        assert "Step 2" in example.reasoning
        assert example.metadata["employee"] == "赵云帆"
        assert example.metadata["total_steps"] == 2

    def test_export_dataset(self, exporter, sample_trajectories, tmp_path):
        """测试导出数据集."""
        output_file = tmp_path / "dataset.jsonl"

        stats = exporter.export_dataset(output_file)

        assert stats["total"] == 2
        assert stats["exported"] == 2
        assert output_file.exists()

        # 验证输出格式
        lines = output_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

        example = json.loads(lines[0])
        assert "input" in example
        assert "output" in example
        assert "reasoning" in example
        assert "metadata" in example

    def test_export_dataset_with_max_samples(self, exporter, sample_trajectories, tmp_path):
        """测试限制导出数量."""
        output_file = tmp_path / "dataset.jsonl"

        stats = exporter.export_dataset(output_file, max_samples=1)

        assert stats["exported"] == 1

    def test_add_annotation(self, exporter):
        """测试添加标注."""
        annotation = exporter.add_annotation(
            trajectory_id="traj-001",
            quality_score=0.85,
            annotator="姜墨言",
            notes="高质量轨迹",
        )

        assert annotation.trajectory_id == "traj-001"
        assert annotation.quality_score == 0.85
        assert annotation.annotator == "姜墨言"

        # 验证文件已创建
        anno_file = exporter.annotations_dir / "traj-001.json"
        assert anno_file.exists()

    def test_get_annotation(self, exporter):
        """测试获取标注."""
        # 添加标注
        exporter.add_annotation(
            trajectory_id="traj-001",
            quality_score=0.9,
            annotator="姜墨言",
        )

        # 获取标注
        annotation = exporter.get_annotation("traj-001")

        assert annotation is not None
        assert annotation.quality_score == 0.9

    def test_get_annotation_not_found(self, exporter):
        """测试获取不存在的标注."""
        annotation = exporter.get_annotation("nonexistent")
        assert annotation is None

    def test_list_annotations(self, exporter):
        """测试列出标注."""
        # 添加多个标注
        exporter.add_annotation("traj-001", 0.9, "姜墨言")
        exporter.add_annotation("traj-002", 0.7, "林锐")
        exporter.add_annotation("traj-003", 0.5, "姜墨言")

        # 列出所有标注
        annotations = exporter.list_annotations()
        assert len(annotations) == 3

        # 按质量过滤
        high_quality = exporter.list_annotations(min_quality=0.8)
        assert len(high_quality) == 1
        assert high_quality[0].trajectory_id == "traj-001"

        # 按标注人过滤
        by_annotator = exporter.list_annotations(annotator="姜墨言")
        assert len(by_annotator) == 2

    def test_load_trajectories_with_quality_filter(self, exporter, sample_trajectories):
        """测试按质量过滤加载轨迹."""
        # 添加标注
        exporter.add_annotation("traj-001", 0.9, "姜墨言")
        exporter.add_annotation("traj-002", 0.5, "姜墨言")

        # 加载高质量轨迹
        trajectories = exporter.load_trajectories(min_quality=0.7)

        assert len(trajectories) == 1
        assert trajectories[0]["trajectory_id"] == "traj-001"
        assert trajectories[0]["quality_score"] == 0.9
