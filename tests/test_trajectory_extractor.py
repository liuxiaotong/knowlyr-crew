"""轨迹提炼器测试."""

import json
from unittest.mock import MagicMock, patch

import pytest

from crew.trajectory_extractor import TrajectoryExtractor


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client."""
    client = MagicMock()
    return client


@pytest.fixture
def extractor(mock_anthropic_client):
    """创建测试用的提炼器."""
    extractor = TrajectoryExtractor(api_key="test-key", value_threshold=0.7)
    # 直接设置 _client，绕过懒加载
    extractor._client = mock_anthropic_client
    return extractor


@pytest.fixture
def sample_trajectory():
    """示例轨迹数据."""
    return {
        "task_id": "test-123",
        "employee": "赵云帆",
        "task": "实现用户认证 API",
        "success": True,
        "steps": [
            {
                "step_id": 1,
                "thought": "先检查现有的认证逻辑",
                "tool_name": "Read",
                "tool_params": {"file_path": "/root/auth.py"},
                "tool_output": "def authenticate(token): ...",
                "tool_exit_code": 0,
            },
            {
                "step_id": 2,
                "thought": "添加 JWT 验证",
                "tool_name": "Edit",
                "tool_params": {"file_path": "/root/auth.py"},
                "tool_output": "已更新",
                "tool_exit_code": 0,
            },
        ],
    }


class TestTrajectoryExtractor:
    """测试轨迹提炼器."""

    def test_init(self, extractor):
        """测试初始化."""
        assert extractor.api_key == "test-key"
        assert extractor.model == "claude-3-5-sonnet-20241022"
        assert extractor.value_threshold == 0.7

    def test_analyze_trajectory_high_value(self, extractor, mock_anthropic_client, sample_trajectory):
        """测试分析高价值轨迹."""
        # Mock Claude 返回
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"value_score": 0.85, "reasoning": "包含重要的认证逻辑实现"}')
        ]
        mock_anthropic_client.messages.create.return_value = mock_response

        result = extractor.analyze_trajectory(sample_trajectory)

        assert result["value_score"] == 0.85
        assert result["should_extract"] is True
        assert "认证" in result["reasoning"]

    def test_analyze_trajectory_low_value(self, extractor, mock_anthropic_client, sample_trajectory):
        """测试分析低价值轨迹."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"value_score": 0.3, "reasoning": "常规查询操作"}')
        ]
        mock_anthropic_client.messages.create.return_value = mock_response

        result = extractor.analyze_trajectory(sample_trajectory)

        assert result["value_score"] == 0.3
        assert result["should_extract"] is False

    def test_analyze_trajectory_invalid_json(self, extractor, mock_anthropic_client, sample_trajectory):
        """测试 Claude 返回无效 JSON."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="这不是 JSON")]
        mock_anthropic_client.messages.create.return_value = mock_response

        result = extractor.analyze_trajectory(sample_trajectory)

        assert result["value_score"] == 0.0
        assert result["should_extract"] is False
        assert "JSON 解析失败" in result["reasoning"]

    def test_extract_memories(self, extractor, mock_anthropic_client, sample_trajectory):
        """测试提取记忆."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    {
                        "memories": [
                            {
                                "category": "finding",
                                "content": "JWT 认证需要在 middleware 中验证 token",
                                "tags": ["auth", "jwt", "api"],
                                "confidence": 0.9,
                            },
                            {
                                "category": "pattern",
                                "content": "认证失败时返回 401 状态码",
                                "tags": ["auth", "http"],
                                "confidence": 1.0,
                            },
                        ]
                    }
                )
            )
        ]
        mock_anthropic_client.messages.create.return_value = mock_response

        memories = extractor.extract_memories(sample_trajectory)

        assert len(memories) == 2
        assert memories[0]["category"] == "finding"
        assert memories[0]["employee"] == "赵云帆"
        assert memories[0]["source_trajectory_id"] == "test-123"
        assert "JWT" in memories[0]["content"]
        assert memories[1]["category"] == "pattern"

    def test_extract_memories_empty(self, extractor, mock_anthropic_client, sample_trajectory):
        """测试提取不到记忆."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"memories": []}')]
        mock_anthropic_client.messages.create.return_value = mock_response

        memories = extractor.extract_memories(sample_trajectory)

        assert len(memories) == 0

    def test_extract_memories_invalid_format(self, extractor, mock_anthropic_client, sample_trajectory):
        """测试 Claude 返回格式异常."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"wrong_field": []}')]
        mock_anthropic_client.messages.create.return_value = mock_response

        memories = extractor.extract_memories(sample_trajectory)

        assert len(memories) == 0

    def test_summarize_steps(self, extractor):
        """测试步骤总结."""
        steps = [
            {
                "tool_name": "Read",
                "thought": "读取文件",
                "tool_output": "文件内容",
                "tool_exit_code": 0,
            },
            {
                "tool_name": "Edit",
                "thought": "修改代码",
                "tool_output": "已更新",
                "tool_exit_code": 0,
            },
        ]

        summary = extractor._summarize_steps(steps)

        assert "Read" in summary
        assert "Edit" in summary
        assert "✓" in summary

    def test_summarize_steps_with_failure(self, extractor):
        """测试包含失败步骤的总结."""
        steps = [
            {
                "tool_name": "Bash",
                "thought": "运行测试",
                "tool_output": "Error: test failed",
                "tool_exit_code": 1,
            }
        ]

        summary = extractor._summarize_steps(steps)

        assert "Bash" in summary
        assert "✗" in summary

    def test_summarize_steps_truncation(self, extractor):
        """测试步骤过多时的截断."""
        steps = [{"tool_name": f"Tool{i}", "thought": "test", "tool_output": "", "tool_exit_code": 0} for i in range(25)]

        summary = extractor._summarize_steps(steps)

        assert "还有 5 步" in summary
