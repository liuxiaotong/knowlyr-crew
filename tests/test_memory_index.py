"""测试 MemorySearchIndex."""

import json
import tempfile
from pathlib import Path

from crew.memory import MemoryStore
from crew.memory_index import MemorySearchIndex


class TestMemorySearchIndex:
    """MemorySearchIndex 行为测试."""

    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.memory_dir = self.tmpdir / "memory"
        self.session_dir = self.tmpdir / "sessions"
        self.db_path = self.tmpdir / "index.db"

    def test_rebuild_and_search(self):
        store = MemoryStore(memory_dir=self.memory_dir)
        store.add("code-reviewer", category="finding", content="修复登录bug", source_session="sess1")

        self.session_dir.mkdir(parents=True, exist_ok=True)
        session_file = self.session_dir / "20250101-aaaa.jsonl"
        start = {
            "timestamp": "2025-01-01T00:00:00",
            "event": "start",
            "session_type": "employee",
            "subject": "code-reviewer",
            "metadata": {"employee": "code-reviewer"},
        }
        message = {
            "timestamp": "2025-01-01T00:00:01",
            "event": "message",
            "role": "prompt",
            "content": "请审查认证流程",
            "metadata": {"employee": "code-reviewer"},
        }
        session_file.write_text(json.dumps(start) + "\n" + json.dumps(message) + "\n", encoding="utf-8")

        index = MemorySearchIndex(
            db_path=self.db_path,
            memory_dir=self.memory_dir,
            session_dir=self.session_dir,
        )
        stats = index.rebuild()
        assert stats.memory_entries == 1
        assert stats.session_messages == 1

        results = index.search("登录", limit=5)
        assert results
        assert any(r["kind"] == "memory" for r in results)

        results_session = index.search("认证", kind="session")
        assert len(results_session) == 1
        assert results_session[0]["kind"] == "session"
