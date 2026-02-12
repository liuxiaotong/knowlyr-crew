"""测试工作日志."""

import tempfile
from pathlib import Path

import pytest

from crew.log import WorkLogger


class TestWorkLogger:
    """测试 WorkLogger."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.logger = WorkLogger(log_dir=Path(self.tmpdir))

    def test_create_session(self):
        """创建 session 应返回 session_id."""
        session_id = self.logger.create_session("code-reviewer", args={"target": "main"})
        assert session_id
        assert "-" in session_id

    def test_create_session_creates_file(self):
        """创建 session 应生成 JSONL 文件."""
        session_id = self.logger.create_session("test-engineer")
        session_file = Path(self.tmpdir) / f"{session_id}.jsonl"
        assert session_file.exists()

    def test_add_entry(self):
        """追加条目应成功."""
        session_id = self.logger.create_session("code-reviewer")
        self.logger.add_entry(session_id, "review_start", "开始审查")
        entries = self.logger.get_session(session_id)
        assert len(entries) == 2  # session_start + review_start
        assert entries[1]["action"] == "review_start"

    def test_add_entry_invalid_session(self):
        """向不存在的 session 追加应报错."""
        with pytest.raises(ValueError, match="Session"):
            self.logger.add_entry("nonexistent-session", "action", "detail")

    def test_list_sessions_empty(self):
        """空目录应返回空列表."""
        sessions = self.logger.list_sessions()
        assert sessions == []

    def test_list_sessions(self):
        """应能列出 sessions."""
        self.logger.create_session("code-reviewer")
        self.logger.create_session("test-engineer")
        sessions = self.logger.list_sessions()
        assert len(sessions) == 2

    def test_list_sessions_filter_by_employee(self):
        """应能按员工过滤 sessions."""
        self.logger.create_session("code-reviewer")
        self.logger.create_session("test-engineer")
        sessions = self.logger.list_sessions(employee_name="code-reviewer")
        assert len(sessions) == 1
        assert sessions[0]["employee_name"] == "code-reviewer"

    def test_list_sessions_limit(self):
        """应能限制返回条数."""
        for _ in range(5):
            self.logger.create_session("code-reviewer")
        sessions = self.logger.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_list_sessions_counts_entries(self):
        session_id = self.logger.create_session("doc-writer")
        self.logger.add_entry(session_id, "step1", "detail1")
        sessions = self.logger.list_sessions()
        assert any(s["session_id"] == session_id and s["entries"] == 2 for s in sessions)

    def test_get_session(self):
        """应能获取 session 的所有条目."""
        session_id = self.logger.create_session("code-reviewer")
        self.logger.add_entry(session_id, "step1", "detail1")
        self.logger.add_entry(session_id, "step2", "detail2")
        entries = self.logger.get_session(session_id)
        assert len(entries) == 3

    def test_get_session_not_found(self):
        """不存在的 session 应返回空列表."""
        entries = self.logger.get_session("nonexistent")
        assert entries == []

    def test_list_sessions_nonexistent_dir(self):
        """日志目录不存在时应返回空列表."""
        logger = WorkLogger(log_dir=Path("/tmp/nonexistent-crew-test-dir"))
        sessions = logger.list_sessions()
        assert sessions == []
