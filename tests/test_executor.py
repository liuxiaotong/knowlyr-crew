"""测试 LLM 执行器."""

from unittest.mock import MagicMock, patch

import pytest

from crew.executor import ExecutionResult, execute_prompt


class TestExecutePrompt:
    """测试 execute_prompt."""

    @patch("crew.executor._get_anthropic")
    def test_non_streaming(self, mock_get):
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="Hello world")]
        mock_msg.model = "claude-sonnet-4-20250514"
        mock_msg.usage.input_tokens = 100
        mock_msg.usage.output_tokens = 10
        mock_msg.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_get.return_value = mock_anthropic

        result = execute_prompt(
            system_prompt="You are a test.",
            api_key="test-key",
            stream=False,
        )

        assert isinstance(result, ExecutionResult)
        assert result.content == "Hello world"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.input_tokens == 100
        assert result.output_tokens == 10
        assert result.stop_reason == "end_turn"

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a test."
        assert call_kwargs["messages"] == [{"role": "user", "content": "请开始执行上述任务。"}]

    @patch("crew.executor._get_anthropic")
    def test_streaming(self, mock_get):
        chunks_received = []

        mock_final = MagicMock()
        mock_final.model = "claude-sonnet-4-20250514"
        mock_final.usage.input_tokens = 50
        mock_final.usage.output_tokens = 5
        mock_final.stop_reason = "end_turn"

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Hello", " ", "world"])
        mock_stream.get_final_message.return_value = mock_final

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_get.return_value = mock_anthropic

        result = execute_prompt(
            system_prompt="test",
            api_key="key",
            stream=True,
            on_chunk=lambda c: chunks_received.append(c),
        )

        assert result.content == "Hello world"
        assert chunks_received == ["Hello", " ", "world"]
        assert result.input_tokens == 50

    def test_missing_sdk(self):
        with patch("crew.executor._get_anthropic", return_value=None):
            with pytest.raises(ImportError, match="anthropic SDK 未安装"):
                execute_prompt(system_prompt="test", api_key="key", stream=False)

    @patch("crew.executor._get_anthropic")
    def test_custom_params(self, mock_get):
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="ok")]
        mock_msg.model = "claude-opus-4-20250514"
        mock_msg.usage.input_tokens = 200
        mock_msg.usage.output_tokens = 20
        mock_msg.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_get.return_value = mock_anthropic

        result = execute_prompt(
            system_prompt="sys",
            user_message="custom message",
            api_key="key",
            model="claude-opus-4-20250514",
            temperature=0.5,
            max_tokens=8192,
            stream=False,
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-20250514"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 8192
        assert call_kwargs["messages"] == [{"role": "user", "content": "custom message"}]
        assert result.content == "ok"

    @patch("crew.executor._get_anthropic")
    def test_empty_response(self, mock_get):
        mock_msg = MagicMock()
        mock_msg.content = []
        mock_msg.model = "claude-sonnet-4-20250514"
        mock_msg.usage.input_tokens = 10
        mock_msg.usage.output_tokens = 0
        mock_msg.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_get.return_value = mock_anthropic

        result = execute_prompt(system_prompt="test", api_key="key", stream=False)
        assert result.content == ""
