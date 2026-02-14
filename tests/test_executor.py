"""测试 LLM 执行器."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crew.executor import ExecutionResult, execute_prompt, aexecute_prompt


class TestExecutePrompt:
    """测试 execute_prompt（Anthropic 路径）."""

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


class TestOpenAIExecute:
    """测试 execute_prompt（OpenAI 路径）."""

    @patch("crew.executor._get_openai")
    def test_non_streaming(self, mock_get):
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from GPT"
        mock_choice.finish_reason = "stop"

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "gpt-4o"
        mock_resp.usage.prompt_tokens = 100
        mock_resp.usage.completion_tokens = 20

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_get.return_value = mock_openai

        result = execute_prompt(
            system_prompt="You are a test.",
            api_key="sk-openai-key",
            model="gpt-4o",
            stream=False,
        )

        assert isinstance(result, ExecutionResult)
        assert result.content == "Hello from GPT"
        assert result.model == "gpt-4o"
        assert result.input_tokens == 100
        assert result.output_tokens == 20
        assert result.stop_reason == "stop"

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "请开始执行上述任务。"},
        ]

    @patch("crew.executor._get_openai")
    def test_streaming(self, mock_get):
        chunks_received = []

        # 模拟流式 chunk
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"
        chunk2.usage = None

        chunk3 = MagicMock()
        chunk3.choices = []
        chunk3.usage = MagicMock()
        chunk3.usage.prompt_tokens = 50
        chunk3.usage.completion_tokens = 5

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_get.return_value = mock_openai

        result = execute_prompt(
            system_prompt="test",
            api_key="key",
            model="gpt-4o",
            stream=True,
            on_chunk=lambda c: chunks_received.append(c),
        )

        assert result.content == "Hello world"
        assert chunks_received == ["Hello", " world"]
        assert result.input_tokens == 50
        assert result.output_tokens == 5

    @patch("crew.executor._get_openai")
    def test_custom_params(self, mock_get):
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_choice.finish_reason = "stop"

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "gpt-4-turbo"
        mock_resp.usage.prompt_tokens = 200
        mock_resp.usage.completion_tokens = 10

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_get.return_value = mock_openai

        result = execute_prompt(
            system_prompt="sys",
            user_message="custom",
            api_key="key",
            model="gpt-4-turbo",
            temperature=0.7,
            max_tokens=4096,
            stream=False,
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 4096
        assert result.content == "ok"

    def test_missing_sdk(self):
        with patch("crew.executor._get_openai", return_value=None):
            with pytest.raises(ImportError, match="openai SDK 未安装"):
                execute_prompt(
                    system_prompt="test", api_key="key", model="gpt-4o", stream=False,
                )

    @patch("crew.executor._get_openai")
    def test_empty_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.choices = []
        mock_resp.model = "gpt-4o"
        mock_resp.usage = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_get.return_value = mock_openai

        result = execute_prompt(
            system_prompt="test", api_key="key", model="gpt-4o", stream=False,
        )
        assert result.content == ""
        assert result.stop_reason == "unknown"


class TestDeepSeekExecute:
    """测试 execute_prompt（DeepSeek 路径 — 使用 OpenAI SDK + base_url）."""

    @patch("crew.executor._get_openai")
    def test_uses_deepseek_base_url(self, mock_get):
        mock_choice = MagicMock()
        mock_choice.message.content = "DeepSeek result"
        mock_choice.finish_reason = "stop"

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "deepseek-chat"
        mock_resp.usage.prompt_tokens = 80
        mock_resp.usage.completion_tokens = 15

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_get.return_value = mock_openai

        result = execute_prompt(
            system_prompt="test",
            api_key="ds-key",
            model="deepseek-chat",
            stream=False,
        )

        # 验证 base_url 传递
        client_call = mock_openai.OpenAI.call_args[1]
        assert client_call["base_url"] == "https://api.deepseek.com"
        assert client_call["api_key"] == "ds-key"
        assert result.content == "DeepSeek result"

    @patch("crew.executor._get_openai")
    def test_deepseek_coder(self, mock_get):
        mock_choice = MagicMock()
        mock_choice.message.content = "code result"
        mock_choice.finish_reason = "stop"

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "deepseek-coder"
        mock_resp.usage.prompt_tokens = 60
        mock_resp.usage.completion_tokens = 10

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_get.return_value = mock_openai

        result = execute_prompt(
            system_prompt="test", api_key="key", model="deepseek-coder", stream=False,
        )
        assert result.content == "code result"

        # 验证使用 DeepSeek base_url
        client_call = mock_openai.OpenAI.call_args[1]
        assert "deepseek" in client_call["base_url"]


class TestProviderDispatch:
    """测试 provider 自动分派."""

    @patch("crew.executor._get_anthropic")
    def test_claude_model_routes_to_anthropic(self, mock_get):
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="ok")]
        mock_msg.model = "claude-sonnet-4-20250514"
        mock_msg.usage.input_tokens = 10
        mock_msg.usage.output_tokens = 5
        mock_msg.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_get.return_value = mock_anthropic

        result = execute_prompt(
            system_prompt="test", api_key="key", model="claude-sonnet-4-20250514", stream=False,
        )
        # 验证调用的是 Anthropic 的 messages API（非 chat.completions）
        mock_client.messages.create.assert_called_once()
        assert result.content == "ok"

    @patch("crew.executor._get_openai")
    def test_gpt_model_routes_to_openai(self, mock_get):
        mock_choice = MagicMock()
        mock_choice.message.content = "gpt ok"
        mock_choice.finish_reason = "stop"

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "gpt-4o"
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_get.return_value = mock_openai

        result = execute_prompt(
            system_prompt="test", api_key="key", model="gpt-4o", stream=False,
        )
        # 验证调用的是 OpenAI 的 chat.completions API
        mock_client.chat.completions.create.assert_called_once()
        # 验证没有 base_url
        client_call = mock_openai.OpenAI.call_args[1]
        assert "base_url" not in client_call
        assert result.content == "gpt ok"

    def test_unknown_model_raises_error(self):
        with pytest.raises(ValueError, match="无法识别模型"):
            execute_prompt(
                system_prompt="test", api_key="key", model="llama-3", stream=False,
            )


class TestApiKeyAutoResolve:
    """测试 API key 自动从环境变量解析."""

    @patch("crew.executor._get_anthropic")
    def test_anthropic_auto_resolve(self, mock_get):
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="ok")]
        mock_msg.model = "claude-sonnet-4-20250514"
        mock_msg.usage.input_tokens = 10
        mock_msg.usage.output_tokens = 5
        mock_msg.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_get.return_value = mock_anthropic

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-ant-key"}):
            result = execute_prompt(
                system_prompt="test", api_key=None, stream=False,
            )
        assert result.content == "ok"
        # 验证使用了 env 中的 key
        client_call = mock_anthropic.Anthropic.call_args[1]
        assert client_call["api_key"] == "env-ant-key"

    @patch("crew.executor._get_openai")
    def test_openai_auto_resolve(self, mock_get):
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_choice.finish_reason = "stop"

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "gpt-4o"
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_get.return_value = mock_openai

        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-oai-key"}):
            result = execute_prompt(
                system_prompt="test", api_key=None, model="gpt-4o", stream=False,
            )
        assert result.content == "ok"
        client_call = mock_openai.OpenAI.call_args[1]
        assert client_call["api_key"] == "env-oai-key"

    def test_missing_env_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key 未设置"):
                execute_prompt(
                    system_prompt="test", api_key=None, model="gpt-4o", stream=False,
                )


class TestAsyncOpenAI:
    """测试 aexecute_prompt（OpenAI 路径）."""

    @patch("crew.executor._get_openai")
    def test_async_non_streaming(self, mock_get):
        mock_choice = MagicMock()
        mock_choice.message.content = "async gpt"
        mock_choice.finish_reason = "stop"

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "gpt-4o"
        mock_resp.usage.prompt_tokens = 30
        mock_resp.usage.completion_tokens = 8

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client
        mock_get.return_value = mock_openai

        result = asyncio.run(aexecute_prompt(
            system_prompt="test",
            api_key="key",
            model="gpt-4o",
            stream=False,
        ))
        assert result.content == "async gpt"
        assert result.model == "gpt-4o"
        assert result.input_tokens == 30

    @patch("crew.executor._get_openai")
    def test_async_deepseek_uses_base_url(self, mock_get):
        mock_choice = MagicMock()
        mock_choice.message.content = "async ds"
        mock_choice.finish_reason = "stop"

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "deepseek-chat"
        mock_resp.usage.prompt_tokens = 20
        mock_resp.usage.completion_tokens = 5

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client
        mock_get.return_value = mock_openai

        result = asyncio.run(aexecute_prompt(
            system_prompt="test",
            api_key="key",
            model="deepseek-chat",
            stream=False,
        ))
        assert result.content == "async ds"
        client_call = mock_openai.AsyncOpenAI.call_args[1]
        assert "deepseek" in client_call["base_url"]
