"""测试 SessionRecorder."""

import tempfile
from pathlib import Path

from crew.session_recorder import SessionRecorder


class TestSessionRecorder:
    """SessionRecorder 行为测试."""

    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.recorder = SessionRecorder(session_dir=self.tmpdir)

    def test_roundtrip(self):
        session_id = self.recorder.start(
            "employee", "code-reviewer", {"args": {"target": "main"}}
        )
        self.recorder.record_message(session_id, "prompt", "hello world")
        self.recorder.record_event(session_id, "stdout", {"chars": 11})
        self.recorder.finish(session_id, status="completed", detail="done")

        entries = self.recorder.read_session(session_id)
        assert entries[-1]["event"] == "end"
        sessions = self.recorder.list_sessions()
        assert sessions[0]["session_id"] == session_id

    def test_filters(self):
        session_a = self.recorder.start("employee", "code-reviewer")
        session_b = self.recorder.start("pipeline", "full-review")
        self.recorder.finish(session_a)
        self.recorder.finish(session_b)

        employees = self.recorder.list_sessions(session_type="employee")
        assert len(employees) == 1
        assert employees[0]["session_id"] == session_a

    def test_missing_session(self):
        assert self.recorder.read_session("missing") == []
        # list on empty dir should be []
        empty_recorder = SessionRecorder(session_dir=self.tmpdir / "nested")
        assert empty_recorder.list_sessions() == []
