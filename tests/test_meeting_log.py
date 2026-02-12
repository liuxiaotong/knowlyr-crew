"""会议记录模块测试."""

import json

from crew.discussion import Discussion, DiscussionParticipant
from crew.meeting_log import MeetingLogger, MeetingRecord


class TestMeetingRecord:
    """测试会议记录模型."""

    def test_basic_record(self):
        r = MeetingRecord(
            meeting_id="20260212_143000",
            name="test",
            topic="测试议题",
            participants=["code-reviewer", "test-engineer"],
            mode="discussion",
            rounds=3,
            started_at="2026-02-12T14:30:00",
        )
        assert r.meeting_id == "20260212_143000"
        assert r.mode == "discussion"
        assert len(r.participants) == 2

    def test_record_with_args(self):
        r = MeetingRecord(
            meeting_id="20260212_143000",
            name="test",
            topic="测试",
            participants=["a"],
            mode="meeting",
            rounds=1,
            started_at="2026-02-12T14:30:00",
            args={"target": "auth.py"},
        )
        assert r.args["target"] == "auth.py"


class TestMeetingLogger:
    """测试会议记录管理器."""

    def test_save_and_list(self, tmp_path):
        logger = MeetingLogger(meetings_dir=tmp_path / "meetings")
        d = Discussion(
            name="test-discuss",
            topic="测试议题",
            participants=[
                DiscussionParticipant(employee="code-reviewer"),
                DiscussionParticipant(employee="test-engineer"),
            ],
        )
        meeting_id = logger.save(d, prompt="# 测试 prompt", args={"target": "x"})
        assert meeting_id  # 非空字符串

        records = logger.list()
        assert len(records) == 1
        assert records[0].name == "test-discuss"
        assert records[0].topic == "测试议题"
        assert records[0].args == {"target": "x"}

    def test_save_creates_files(self, tmp_path):
        logger = MeetingLogger(meetings_dir=tmp_path / "meetings")
        d = Discussion(
            name="test",
            topic="测试",
            participants=[DiscussionParticipant(employee="code-reviewer")],
        )
        meeting_id = logger.save(d, prompt="# prompt content")

        # 检查 index.jsonl 存在
        index = tmp_path / "meetings" / "index.jsonl"
        assert index.exists()
        lines = index.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["meeting_id"] == meeting_id

        # 检查 prompt 文件
        prompt_file = tmp_path / "meetings" / f"{meeting_id}.md"
        assert prompt_file.exists()
        assert prompt_file.read_text() == "# prompt content"

    def test_get_meeting(self, tmp_path):
        logger = MeetingLogger(meetings_dir=tmp_path / "meetings")
        d = Discussion(
            name="test",
            topic="获取测试",
            participants=[DiscussionParticipant(employee="code-reviewer")],
        )
        meeting_id = logger.save(d, prompt="# 完整内容")

        result = logger.get(meeting_id)
        assert result is not None
        record, content = result
        assert record.topic == "获取测试"
        assert content == "# 完整内容"

    def test_get_nonexistent(self, tmp_path):
        logger = MeetingLogger(meetings_dir=tmp_path / "meetings")
        assert logger.get("nonexistent") is None

    def test_list_empty(self, tmp_path):
        logger = MeetingLogger(meetings_dir=tmp_path / "meetings")
        assert logger.list() == []

    def test_list_with_keyword(self, tmp_path):
        logger = MeetingLogger(meetings_dir=tmp_path / "meetings")

        d1 = Discussion(
            name="auth-review",
            topic="认证模块审查",
            participants=[
                DiscussionParticipant(employee="code-reviewer"),
                DiscussionParticipant(employee="test-engineer"),
            ],
        )
        d2 = Discussion(
            name="perf-discuss",
            topic="性能优化讨论",
            participants=[DiscussionParticipant(employee="code-reviewer")],
        )

        logger.save(d1, prompt="# auth")
        logger.save(d2, prompt="# perf")

        results = logger.list(keyword="认证")
        assert len(results) == 1
        assert results[0].name == "auth-review"

        results = logger.list(keyword="性能")
        assert len(results) == 1
        assert results[0].name == "perf-discuss"

    def test_list_order_newest_first(self, tmp_path):
        logger = MeetingLogger(meetings_dir=tmp_path / "meetings")

        for name in ("first", "second", "third"):
            d = Discussion(
                name=name,
                topic=f"{name} topic",
                participants=[DiscussionParticipant(employee="code-reviewer")],
            )
            logger.save(d, prompt=f"# {name}")

        records = logger.list()
        assert len(records) == 3
        assert records[0].name == "third"

    def test_list_limit(self, tmp_path):
        logger = MeetingLogger(meetings_dir=tmp_path / "meetings")

        for i in range(5):
            d = Discussion(
                name=f"meeting-{i}",
                topic=f"topic {i}",
                participants=[DiscussionParticipant(employee="code-reviewer")],
            )
            logger.save(d, prompt=f"# {i}")

        records = logger.list(limit=3)
        assert len(records) == 3

    def test_1v1_mode_recorded(self, tmp_path):
        logger = MeetingLogger(meetings_dir=tmp_path / "meetings")
        d = Discussion(
            name="1v1-test",
            topic="一对一",
            participants=[DiscussionParticipant(employee="code-reviewer")],
        )
        logger.save(d, prompt="# 1v1")

        records = logger.list()
        assert records[0].mode == "meeting"
