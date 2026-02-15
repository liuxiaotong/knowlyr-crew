"""飞书对话记忆测试."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crew.feishu_memory import FeishuChatStore, capture_feishu_memory


class TestFeishuChatStore:
    def test_append_and_get_recent(self, tmp_path: Path):
        store = FeishuChatStore(tmp_path / "chats")
        store.append("chat_001", "user", "你好")
        store.append("chat_001", "assistant", "你好，Kai！")

        msgs = store.get_recent("chat_001")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "你好"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "你好，Kai！"
        assert "ts" in msgs[0]

    def test_empty_chat(self, tmp_path: Path):
        store = FeishuChatStore(tmp_path / "chats")
        assert store.get_recent("nonexistent") == []

    def test_limit(self, tmp_path: Path):
        store = FeishuChatStore(tmp_path / "chats")
        for i in range(10):
            store.append("chat_002", "user", f"msg-{i}")
        msgs = store.get_recent("chat_002", limit=3)
        assert len(msgs) == 3
        assert msgs[0]["content"] == "msg-7"
        assert msgs[2]["content"] == "msg-9"

    def test_format_for_prompt(self, tmp_path: Path):
        store = FeishuChatStore(tmp_path / "chats")
        store.append("chat_003", "user", "审查代码")
        store.append("chat_003", "assistant", "好的，已完成审查")
        store.append("chat_003", "user", "还有其他问题吗")

        text = store.format_for_prompt("chat_003")
        assert "Kai: 审查代码" in text
        assert "你: 好的，已完成审查" in text
        assert "Kai: 还有其他问题吗" in text

    def test_format_for_prompt_empty(self, tmp_path: Path):
        store = FeishuChatStore(tmp_path / "chats")
        assert store.format_for_prompt("empty") == ""

    def test_multiple_chats_isolated(self, tmp_path: Path):
        store = FeishuChatStore(tmp_path / "chats")
        store.append("chat_a", "user", "a的消息")
        store.append("chat_b", "user", "b的消息")

        a = store.get_recent("chat_a")
        b = store.get_recent("chat_b")
        assert len(a) == 1
        assert a[0]["content"] == "a的消息"
        assert len(b) == 1
        assert b[0]["content"] == "b的消息"

    def test_chat_id_sanitization(self, tmp_path: Path):
        store = FeishuChatStore(tmp_path / "chats")
        store.append("oc_abc/def", "user", "test")
        msgs = store.get_recent("oc_abc/def")
        assert len(msgs) == 1

    def test_jsonl_format(self, tmp_path: Path):
        store = FeishuChatStore(tmp_path / "chats")
        store.append("chat_fmt", "user", "hello")
        store.append("chat_fmt", "assistant", "hi")

        content = (tmp_path / "chats" / "chat_fmt.jsonl").read_text(encoding="utf-8")
        lines = [l for l in content.splitlines() if l.strip()]
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "role" in data
            assert "content" in data
            assert "ts" in data


class TestCaptureFeishuMemory:
    def test_creates_session_and_memory(self, tmp_path: Path):
        """验证 capture_feishu_memory 记录 session."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / ".crew").mkdir()

        capture_feishu_memory(
            project_dir=project_dir,
            employee_name="ceo-assistant",
            chat_id="oc_test123",
            user_text="帮我审查代码",
            assistant_text="已完成代码审查，质量良好",
        )

        # 验证 session 文件被创建
        session_dir = project_dir / ".crew" / "sessions"
        sessions = list(session_dir.glob("*.jsonl"))
        assert len(sessions) == 1

        # 验证 session 内容
        lines = sessions[0].read_text(encoding="utf-8").splitlines()
        events = [json.loads(l) for l in lines if l.strip()]
        assert events[0]["event"] == "start"
        assert events[0]["session_type"] == "feishu"
        assert events[1]["event"] == "message"
        assert events[1]["role"] == "user"
        assert events[2]["event"] == "message"
        assert events[2]["role"] == "assistant"
        assert events[3]["event"] == "end"

    def test_capture_does_not_throw(self, tmp_path: Path):
        """即使路径不存在也不会抛异常."""
        capture_feishu_memory(
            project_dir=tmp_path / "nonexistent",
            employee_name="test",
            chat_id="test",
            user_text="hello",
            assistant_text="hi",
        )
