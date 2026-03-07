"""Tests for reply_postprocess — auto-push 接入记忆管线."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from crew.memory import MemoryEntry
from crew.reply_postprocess import push_if_needed, should_push


def _make_entry(**kwargs) -> MemoryEntry:
    defaults = {
        "id": "test-id",
        "employee": "测试",
        "category": "finding",
        "content": "测试内容",
    }
    defaults.update(kwargs)
    return MemoryEntry(**defaults)


class TestShouldPush:
    """should_push 行为不变（回归测试）."""

    def test_empty_reply(self):
        assert should_push("", 1) == (False, "")
        assert should_push("   ", 1) == (False, "")

    def test_decision_keyword(self):
        ok, cat = should_push("我们决定使用方案 A", 1)
        assert ok is True
        assert cat == "decision"

    def test_correction_keyword(self):
        ok, cat = should_push("其实之前搞错了", 1)
        assert ok is True
        assert cat == "correction"

    def test_long_output_enough_turns(self):
        ok, cat = should_push("a" * 300, turn_count=5)
        assert ok is True
        assert cat == "finding"

    def test_long_output_few_turns(self):
        ok, cat = should_push("a" * 300, turn_count=1)
        assert ok is False

    def test_short_reply(self):
        ok, _ = should_push("好的", 1)
        assert ok is False


class TestPushIfNeeded:
    @patch("crew.reply_postprocess.process_memory")
    @patch("crew.reply_postprocess.get_memory_store")
    def test_calls_pipeline(self, mock_get_store, mock_pipeline):
        """push_if_needed 触发时调用 process_memory 管线."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_pipeline.return_value = _make_entry()

        result = push_if_needed(
            employee="赵云帆",
            reply="我们决定使用新方案",
            turn_count=1,
            session_id="sess-1",
        )

        assert result is True

        # 等后台线程完成
        time.sleep(0.5)

        mock_pipeline.assert_called_once_with(
            raw_text="我们决定使用新方案",
            employee="赵云帆",
            store=mock_store,
            skip_reflect=False,
            source_session="sess-1",
        )

    @patch("crew.reply_postprocess.process_memory")
    @patch("crew.reply_postprocess.get_memory_store")
    def test_async_not_blocking(self, mock_get_store, mock_pipeline):
        """push_if_needed 立即返回，不等待管线完成."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        # 让 process_memory 模拟 LLM 调用需要 2 秒
        def slow_pipeline(*args, **kwargs):
            time.sleep(2)
            return _make_entry()

        mock_pipeline.side_effect = slow_pipeline

        start = time.time()
        result = push_if_needed(
            employee="赵云帆",
            reply="我们决定使用新方案",
            turn_count=1,
        )
        elapsed = time.time() - start

        assert result is True
        # 应该在 1 秒内返回（远快于 2 秒的管线执行时间）
        assert elapsed < 1.0

    @patch("crew.reply_postprocess.process_memory")
    @patch("crew.reply_postprocess.get_memory_store")
    def test_pipeline_skip_no_error(self, mock_get_store, mock_pipeline):
        """process_memory 返回 None 时不报错."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_pipeline.return_value = None

        result = push_if_needed(
            employee="赵云帆",
            reply="我们决定使用新方案",
            turn_count=1,
        )

        assert result is True

        # 等后台线程完成
        time.sleep(0.5)

        # 不应抛异常，管线被调用了
        mock_pipeline.assert_called_once()

    @patch("crew.reply_postprocess.process_memory")
    @patch("crew.reply_postprocess.get_memory_store")
    def test_pipeline_error_no_crash(self, mock_get_store, mock_pipeline):
        """process_memory 抛异常时不导致崩溃."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_pipeline.side_effect = RuntimeError("LLM 挂了")

        result = push_if_needed(
            employee="赵云帆",
            reply="我们决定使用新方案",
            turn_count=1,
        )

        assert result is True

        # 等后台线程完成
        time.sleep(0.5)

        # 不会崩溃，线程内部捕获了异常
        mock_pipeline.assert_called_once()

    def test_not_triggered_returns_false(self):
        """不满足门槛时直接返回 False，不调管线."""
        result = push_if_needed(
            employee="赵云帆",
            reply="好的",
            turn_count=1,
        )
        assert result is False

    @patch("crew.reply_postprocess.invalidate")
    @patch("crew.reply_postprocess.process_memory")
    def test_do_push_invalidate_on_success(self, mock_pipeline, mock_invalidate):
        """管线成功时调用 invalidate 缓存失效（直接调 _do_push 避免线程竞态）."""
        from crew.reply_postprocess import _do_push

        mock_store = MagicMock()
        mock_pipeline.return_value = _make_entry()

        _do_push("赵云帆", "我们决定使用新方案", "sess-1", mock_store)

        mock_invalidate.assert_called_once_with("赵云帆")

    @patch("crew.reply_postprocess.invalidate")
    @patch("crew.reply_postprocess.process_memory")
    def test_do_push_no_invalidate_on_skip(self, mock_pipeline, mock_invalidate):
        """管线决定跳过时不调用 invalidate（直接调 _do_push 避免线程竞态）."""
        from crew.reply_postprocess import _do_push

        mock_store = MagicMock()
        mock_pipeline.return_value = None

        _do_push("赵云帆", "我们决定使用新方案", "sess-1", mock_store)

        mock_invalidate.assert_not_called()

    @patch("crew.reply_postprocess.threading.Thread")
    @patch("crew.reply_postprocess.get_memory_store")
    def test_thread_is_daemon(self, mock_get_store, mock_thread_cls):
        """后台线程必须是 daemon=True."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        push_if_needed(
            employee="赵云帆",
            reply="我们决定使用新方案",
            turn_count=1,
        )

        mock_thread_cls.assert_called_once()
        call_kwargs = mock_thread_cls.call_args[1]
        assert call_kwargs["daemon"] is True
        mock_thread.start.assert_called_once()
