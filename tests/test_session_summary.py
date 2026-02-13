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
