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


class TestRetry:
    """测试 LLM 重试机制."""

    @patch("crew.executor.time")
    @patch("crew.executor._get_anthropic")
    def test_retry_on_rate_limit(self, mock_get, mock_time):
        """429 触发重试."""
        rate_limit_err = Exception("rate limited")
        rate_limit_err.status_code = 429

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="ok")]
        mock_msg.model = "claude-sonnet-4-20250514"
        mock_msg.usage.input_tokens = 10
        mock_msg.usage.output_tokens = 5
        mock_msg.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [rate_limit_err, mock_msg]
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_get.return_value = mock_anthropic

        result = execute_prompt(system_prompt="test", api_key="key", stream=False)
        assert result.content == "ok"
        assert mock_client.messages.create.call_count == 2
        mock_time.sleep.assert_called_once()

    @patch("crew.executor.time")
    @patch("crew.executor._get_anthropic")
    def test_retry_on_server_error(self, mock_get, mock_time):
        """5xx 触发重试."""
        server_err = Exception("internal error")
        server_err.status_code = 500

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="ok")]
        mock_msg.model = "claude-sonnet-4-20250514"
        mock_msg.usage.input_tokens = 10
        mock_msg.usage.output_tokens = 5
        mock_msg.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [server_err, mock_msg]
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_get.return_value = mock_anthropic

        result = execute_prompt(system_prompt="test", api_key="key", stream=False)
        assert result.content == "ok"

    @patch("crew.executor._get_anthropic")
    def test_no_retry_on_auth_error(self, mock_get):
        """401 不重试."""
        auth_err = Exception("unauthorized")
        auth_err.status_code = 401

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = auth_err
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_get.return_value = mock_anthropic

        with pytest.raises(Exception, match="unauthorized"):
            execute_prompt(system_prompt="test", api_key="key", stream=False)
        assert mock_client.messages.create.call_count == 1

    @patch("crew.executor.time")
    @patch("crew.executor._get_anthropic")
    def test_max_retries_exceeded(self, mock_get, mock_time):
        """超过最大重试次数."""
        rate_err = Exception("rate limited")
        rate_err.status_code = 429

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = rate_err
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_get.return_value = mock_anthropic

        with pytest.raises(Exception, match="rate limited"):
            execute_prompt(system_prompt="test", api_key="key", stream=False)
        # 1 initial + 3 retries = 4 total
        assert mock_client.messages.create.call_count == 4

    def test_is_retryable_connection_error(self):
        from crew.executor import _is_retryable
        assert _is_retryable(ConnectionError("reset"))
        assert _is_retryable(TimeoutError("timed out"))

    def test_is_retryable_status_codes(self):
        from crew.executor import _is_retryable

        err_429 = Exception()
        err_429.status_code = 429
        assert _is_retryable(err_429)

        err_500 = Exception()
        err_500.status_code = 500
        assert _is_retryable(err_500)

        err_503 = Exception()
        err_503.status_code = 503
        assert _is_retryable(err_503)

        err_400 = Exception()
        err_400.status_code = 400
        assert not _is_retryable(err_400)

        err_401 = Exception()
        err_401.status_code = 401
        assert not _is_retryable(err_401)

    def test_retry_delay(self):
        from crew.executor import _retry_delay
        for attempt in range(3):
            delay = _retry_delay(attempt)
            base = 2 ** attempt
            assert base <= delay <= base + 0.5


class TestGeminiExecute:
    """测试 Gemini 执行路径."""

    @patch("crew.executor._get_genai")
    def test_non_streaming(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "Gemini result"
        mock_response.usage_metadata.prompt_token_count = 50
        mock_response.usage_metadata.candidates_token_count = 10

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response

        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_get.return_value = mock_genai

        result = execute_prompt(
            system_prompt="test", api_key="key", model="gemini-2.0-flash", stream=False,
        )
        assert result.content == "Gemini result"
        assert result.input_tokens == 50
        mock_genai.configure.assert_called_once_with(api_key="key")

    def test_missing_sdk(self):
        with patch("crew.executor._get_genai", return_value=None):
            with pytest.raises(ImportError, match="google-generativeai"):
                execute_prompt(system_prompt="test", api_key="key", model="gemini-2.0-flash", stream=False)


class TestZhipuExecute:
    """测试 Zhipu 执行路径（OpenAI 兼容）."""

    @patch("crew.executor._get_openai")
    def test_uses_zhipu_base_url(self, mock_get):
        mock_choice = MagicMock()
        mock_choice.message.content = "Zhipu result"
        mock_choice.finish_reason = "stop"

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "glm-4-flash"
        mock_resp.usage.prompt_tokens = 30
        mock_resp.usage.completion_tokens = 10

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_get.return_value = mock_openai

        result = execute_prompt(
            system_prompt="test", api_key="key", model="glm-4-flash", stream=False,
        )
        assert result.content == "Zhipu result"
        client_call = mock_openai.OpenAI.call_args[1]
        assert "bigmodel" in client_call["base_url"]


class TestQwenExecute:
    """测试 Qwen 执行路径（OpenAI 兼容）."""

    @patch("crew.executor._get_openai")
    def test_uses_qwen_base_url(self, mock_get):
        mock_choice = MagicMock()
        mock_choice.message.content = "Qwen result"
        mock_choice.finish_reason = "stop"

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "qwen-turbo"
        mock_resp.usage.prompt_tokens = 40
        mock_resp.usage.completion_tokens = 12

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_get.return_value = mock_openai

        result = execute_prompt(
            system_prompt="test", api_key="key", model="qwen-turbo", stream=False,
        )
        assert result.content == "Qwen result"
        client_call = mock_openai.OpenAI.call_args[1]
        assert "dashscope" in client_call["base_url"]


class TestMoonshotExecute:
    """测试 Moonshot 执行路径（OpenAI 兼容）."""

    @patch("crew.executor._get_openai")
    def test_uses_moonshot_base_url(self, mock_get):
        mock_choice = MagicMock()
        mock_choice.message.content = "Moonshot result"
        mock_choice.finish_reason = "stop"

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "moonshot-v1-8k"
        mock_resp.usage.prompt_tokens = 35
        mock_resp.usage.completion_tokens = 8

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_get.return_value = mock_openai

        result = execute_prompt(
            system_prompt="test", api_key="key", model="moonshot-v1-8k", stream=False,
        )
        assert result.content == "Moonshot result"
        client_call = mock_openai.OpenAI.call_args[1]
        assert "moonshot" in client_call["base_url"]


class TestFallbackModel:
    """测试 fallback 模型."""

    @patch("crew.executor.time")
    @patch("crew.executor._get_openai")
    @patch("crew.executor._get_anthropic")
    def test_fallback_on_primary_failure(self, mock_ant_get, mock_oai_get, mock_time):
        """主模型失败后切换到 fallback."""
        # 主模型（Anthropic）失败
        rate_err = Exception("rate limited")
        rate_err.status_code = 429

        mock_ant_client = MagicMock()
        mock_ant_client.messages.create.side_effect = rate_err
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_ant_client
        mock_ant_get.return_value = mock_anthropic

        # Fallback 模型（OpenAI）成功
        mock_choice = MagicMock()
        mock_choice.message.content = "fallback ok"
        mock_choice.finish_reason = "stop"

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "gpt-4o"
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5

        mock_oai_client = MagicMock()
        mock_oai_client.chat.completions.create.return_value = mock_resp
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_oai_client
        mock_oai_get.return_value = mock_openai

        result = execute_prompt(
            system_prompt="test",
            api_key="key",
            stream=False,
            fallback_model="gpt-4o",
        )
        assert result.content == "fallback ok"

    @patch("crew.executor._get_anthropic")
    def test_no_fallback_on_non_retryable(self, mock_get):
        """非可重试错误也会尝试 fallback."""
        auth_err = Exception("unauthorized")
        auth_err.status_code = 401

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = auth_err
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_get.return_value = mock_anthropic

        # 无 fallback 时直接抛出
        with pytest.raises(Exception, match="unauthorized"):
            execute_prompt(system_prompt="test", api_key="key", stream=False)


class TestMetricsRecording:
    """测试执行器自动记录指标."""

    @patch("crew.executor._get_anthropic")
    def test_records_metrics_on_success(self, mock_get):
        from crew.metrics import get_collector

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="ok")]
        mock_msg.model = "claude-sonnet-4-20250514"
        mock_msg.usage.input_tokens = 100
        mock_msg.usage.output_tokens = 10
        mock_msg.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_get.return_value = mock_anthropic

        collector = get_collector()
        collector.reset()

        execute_prompt(system_prompt="test", api_key="key", stream=False)

        snap = collector.snapshot()
        assert snap["calls"]["total"] >= 1
        assert snap["calls"]["success"] >= 1
        assert snap["tokens"]["input"] >= 100


class _FakeAsyncStream:
    """Async iterator with a settable .result attribute for testing."""

    def __init__(self, items, result=None):
        self._items = iter(items)
        self.result = result

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


class _FakeAsyncStreamWithError:
    """Async iterator that yields items then raises an error."""

    def __init__(self, items, error, result=None):
        self._items = iter(items)
        self._error = error
        self.result = result

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise self._error


class TestStreamingMetrics:
    """测试流式模式自动记录指标."""

    @patch("crew.executor._get_anthropic")
    def test_async_streaming_records_metrics(self, mock_get):
        """流消费完毕后自动记录指标."""
        import asyncio
        from crew.metrics import get_collector

        mock_client = MagicMock()

        # 模拟 streaming response
        async def _fake_stream(**kwargs):
            async def _gen():
                yield "hello"
                yield " world"
            gen = _gen()
            gen.result = MagicMock(
                content="hello world",
                model="claude-sonnet-4-20250514",
                input_tokens=50,
                output_tokens=10,
                stop_reason="end_turn",
            )
            return gen

        mock_client.messages.stream = MagicMock()

        # 我们直接测试 _MetricsStreamWrapper
        from crew.executor import _MetricsStreamWrapper, ExecutionResult

        mock_result = ExecutionResult(
            content="hello world", model="test", input_tokens=50,
            output_tokens=10, stop_reason="end_turn",
        )

        inner = _FakeAsyncStream(["hello", " world"], result=mock_result)

        collector = get_collector()
        collector.reset()

        wrapper = _MetricsStreamWrapper(inner, "anthropic", __import__("time").monotonic())

        async def _consume():
            chunks = []
            async for chunk in wrapper:
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(_consume())
        assert chunks == ["hello", " world"]
        assert wrapper.result is mock_result

        snap = collector.snapshot()
        assert snap["calls"]["success"] >= 1
        assert snap["tokens"]["input"] >= 50

    def test_streaming_wrapper_exposes_result(self):
        """wrapper.result 透传 inner.result."""
        import asyncio
        from crew.executor import _MetricsStreamWrapper, ExecutionResult

        mock_result = ExecutionResult(
            content="ok", model="test", input_tokens=1,
            output_tokens=1, stop_reason="stop",
        )

        inner = _FakeAsyncStream(["ok"], result=mock_result)

        wrapper = _MetricsStreamWrapper(inner, "test", 0)
        assert wrapper.result is mock_result

    def test_streaming_wrapper_handles_error(self):
        """流中异常不影响 wrapper."""
        import asyncio
        from crew.executor import _MetricsStreamWrapper

        inner = _FakeAsyncStreamWithError(
            ["part"], RuntimeError("boom"), result=None,
        )
        wrapper = _MetricsStreamWrapper(inner, "test", 0)

        async def _consume():
            chunks = []
            try:
                async for chunk in wrapper:
                    chunks.append(chunk)
            except RuntimeError:
                pass
            return chunks

        chunks = asyncio.run(_consume())
        assert chunks == ["part"]


class TestFallbackModelExtended:
    """Fallback 模型扩展测试."""

    @patch("crew.executor._get_openai")
    @patch("crew.executor._get_anthropic")
    def test_both_models_fail_raises_original(self, mock_ant_get, mock_oai_get):
        """主 + fallback 都失败时抛出主异常."""
        # 主模型失败
        primary_err = Exception("primary failure")
        primary_err.status_code = 500
        mock_ant_client = MagicMock()
        mock_ant_client.messages.create.side_effect = primary_err
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_ant_client
        mock_ant_get.return_value = mock_anthropic

        # Fallback 也失败
        fallback_err = Exception("fallback failure")
        mock_oai_client = MagicMock()
        mock_oai_client.chat.completions.create.side_effect = fallback_err
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_oai_client
        mock_oai_get.return_value = mock_openai

        with pytest.raises(Exception, match="primary failure"):
            execute_prompt(
                system_prompt="test", api_key="key", stream=False,
                fallback_model="gpt-4o",
            )

    @patch("crew.executor._get_anthropic")
    def test_no_fallback_without_param(self, mock_get):
        """fallback_model=None 时不触发 fallback."""
        err = Exception("fail")
        err.status_code = 401
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = err
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_get.return_value = mock_anthropic

        with pytest.raises(Exception, match="fail"):
            execute_prompt(system_prompt="test", api_key="key", stream=False)

    @patch("crew.executor.time")
    @patch("crew.executor._get_openai")
    @patch("crew.executor._get_anthropic")
    def test_async_fallback_on_failure(self, mock_ant_get, mock_oai_get, mock_time):
        """异步版本 fallback 生效."""
        import asyncio

        # 主模型失败
        primary_err = Exception("primary fail")
        primary_err.status_code = 429
        mock_ant_aclient = MagicMock()
        mock_ant_aclient.messages.create = AsyncMock(side_effect=primary_err)
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_ant_aclient
        mock_ant_get.return_value = mock_anthropic

        # Fallback 成功
        mock_choice = MagicMock()
        mock_choice.message.content = "async fallback ok"
        mock_choice.finish_reason = "stop"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.model = "gpt-4o"
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        mock_oai_aclient = MagicMock()
        mock_oai_aclient.chat.completions.create = AsyncMock(return_value=mock_resp)
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_oai_aclient
        mock_oai_get.return_value = mock_openai

        from crew.executor import aexecute_prompt
        result = asyncio.run(aexecute_prompt(
            system_prompt="test", api_key="key", stream=False,
            fallback_model="gpt-4o",
        ))
        assert result.content == "async fallback ok"


class TestMetricsLogging:
    """指标记录失败时有日志."""

    def test_record_metrics_failure_logged(self, caplog):
        """_record_metrics 失败时记录 debug 日志."""
        import logging
        from crew.executor import _record_metrics, ExecutionResult

        result = ExecutionResult(
            content="test", model="test", input_tokens=10,
            output_tokens=5, stop_reason="stop",
        )
        with caplog.at_level(logging.DEBUG, logger="crew.executor"):
            with patch("crew.metrics.get_collector", side_effect=Exception("no collector")):
                _record_metrics("test", result, 1.0)
        assert "记录指标失败" in caplog.text

    def test_record_failure_logged(self, caplog):
        """_record_failure 失败时记录 debug 日志."""
        import logging
        from crew.executor import _record_failure

        with caplog.at_level(logging.DEBUG, logger="crew.executor"):
            with patch("crew.metrics.get_collector", side_effect=Exception("no collector")):
                _record_failure("test", RuntimeError("api error"))
        assert "记录失败指标失败" in caplog.text
