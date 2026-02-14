"""测试 SessionMemoryWriter."""

import json
import tempfile
from pathlib import Path

from crew.session_recorder import SessionRecorder
from crew.session_summary import SessionMemoryWriter
from crew.memory import MemoryStore


class TestSessionSummary:
    """SessionMemoryWriter 行为测试."""

    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.session_dir = self.tmpdir / "sessions"
        self.memory_dir = self.tmpdir / "memory"
        self.recorder = SessionRecorder(session_dir=self.session_dir)

    def test_capture(self):
        session_id = self.recorder.start(
            "employee",
            "code-reviewer",
            metadata={"args": {"target": "main"}, "employee": "code-reviewer"},
        )
        self.recorder.record_message(session_id, "prompt", "请审查 main 函数")
        self.recorder.finish(session_id)

        writer = SessionMemoryWriter(session_dir=self.session_dir, memory_dir=self.memory_dir)
        summary = writer.capture(employee="code-reviewer", session_id=session_id)
        assert summary is not None
        assert "main" in summary

        store = MemoryStore(memory_dir=self.memory_dir)
        entries = store.query("code-reviewer")
        assert entries
        assert "main" in entries[0].content

    def test_capture_index_rebuild_failure(self, caplog):
        """索引重建失败时仍返回摘要（不影响核心逻辑）."""
        import logging
        from unittest.mock import patch

        session_id = self.recorder.start(
            "employee",
            "code-reviewer",
            metadata={"args": {"target": "auth"}, "employee": "code-reviewer"},
        )
        self.recorder.record_message(session_id, "prompt", "请审查 auth 模块")
        self.recorder.finish(session_id)

        writer = SessionMemoryWriter(session_dir=self.session_dir, memory_dir=self.memory_dir)

        with caplog.at_level(logging.WARNING, logger="crew.session_summary"):
            with patch.object(writer.index, "rebuild", side_effect=Exception("db error")):
                summary = writer.capture(employee="code-reviewer", session_id=session_id)

        assert summary is not None
        assert "auth" in summary
        assert "重建记忆索引失败" in caplog.text
