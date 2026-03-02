"""集成测试：轨迹提炼流程."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from crew.memory_drafts import MemoryDraftStore
from crew.trajectory_extractor import TrajectoryExtractor


@pytest.fixture
def temp_dirs(tmp_path):
    """创建临时目录."""
    trajectories_dir = tmp_path / "trajectories"
    drafts_dir = tmp_path / "drafts"
    trajectories_dir.mkdir()
    drafts_dir.mkdir()
    return trajectories_dir, drafts_dir


@pytest.fixture
def sample_trajectory_file(temp_dirs):
    """创建示例轨迹文件."""
    trajectories_dir, _ = temp_dirs
    trajectory_file = trajectories_dir / "trajectories.jsonl"

    # 写入示例轨迹
    trajectory = {
        "task_id": "test-123",
        "employee": "赵云帆",
        "task": "修复 API 认证 bug",
        "success": True,
        "metadata": {"timestamp": "2026-03-01T10:00:00"},
        "steps": [
            {
                "step_id": 1,
                "thought": "检查认证逻辑",
                "tool_name": "Read",
                "tool_params": {},
                "tool_output": "发现 token 验证有问题",
                "tool_exit_code": 0,
                "timestamp": "2026-03-01T10:00:00",
            },
            {
                "step_id": 2,
                "thought": "修复 token 验证",
                "tool_name": "Edit",
                "tool_params": {},
                "tool_output": "已修复",
                "tool_exit_code": 0,
                "timestamp": "2026-03-01T10:01:00",
            },
        ],
    }

    with open(trajectory_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(trajectory, ensure_ascii=False) + "\n")

    return trajectory_file


class TestTrajectoryExtractionFlow:
    """测试轨迹提炼完整流程."""

    def test_end_to_end_flow(self, temp_dirs, sample_trajectory_file):
        """测试端到端流程：分析 → 提取 → 创建草稿."""
        trajectories_dir, drafts_dir = temp_dirs

        # 1. 创建提炼器（使用 mock client）
        extractor = TrajectoryExtractor(api_key="test-key", value_threshold=0.7)
        mock_client = MagicMock()
        extractor._client = mock_client

        # 2. 创建草稿存储
        draft_store = MemoryDraftStore(drafts_dir=drafts_dir)

        # 3. 加载轨迹
        with open(sample_trajectory_file, encoding="utf-8") as f:
            trajectory = json.loads(f.readline())

        # 4. Mock 分析结果（高价值）
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"value_score": 0.85, "reasoning": "包含重要的 bug 修复"}')
        ]
        mock_client.messages.create.return_value = mock_response

        analysis = extractor.analyze_trajectory(trajectory)
        assert analysis["should_extract"] is True

        # 5. Mock 提取结果
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "memories": [
                            {
                                "category": "correction",
                                "content": "token 验证必须在 middleware 中进行",
                                "tags": ["auth", "bug"],
                                "confidence": 0.9,
                            }
                        ]
                    }
                )
            )
        ]

        memories = extractor.extract_memories(trajectory)
        assert len(memories) == 1

        # 6. 创建草稿
        for mem in memories:
            draft = draft_store.create_draft(
                employee=mem["employee"],
                category=mem["category"],
                content=mem["content"],
                tags=mem.get("tags", []),
                confidence=mem.get("confidence", 1.0),
                source_trajectory_id=mem.get("source_trajectory_id", ""),
            )

            assert draft.employee == "赵云帆"
            assert draft.category == "correction"
            assert "token" in draft.content
            assert draft.status == "pending"

        # 7. 验证草稿已创建
        drafts = draft_store.list_drafts()
        assert len(drafts) == 1
        assert drafts[0].employee == "赵云帆"

    def test_low_value_trajectory_skipped(self, temp_dirs, sample_trajectory_file):
        """测试低价值轨迹被跳过."""
        trajectories_dir, drafts_dir = temp_dirs

        extractor = TrajectoryExtractor(api_key="test-key", value_threshold=0.7)
        mock_client = MagicMock()
        extractor._client = mock_client

        draft_store = MemoryDraftStore(drafts_dir=drafts_dir)

        with open(sample_trajectory_file, encoding="utf-8") as f:
            trajectory = json.loads(f.readline())

        # Mock 低价值分析结果
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"value_score": 0.3, "reasoning": "常规操作"}')]
        mock_client.messages.create.return_value = mock_response

        analysis = extractor.analyze_trajectory(trajectory)
        assert analysis["should_extract"] is False

        # 不应该创建草稿
        drafts = draft_store.list_drafts()
        assert len(drafts) == 0
