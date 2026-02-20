"""姜墨言能力提升测试 — 新工具 + 对话体验."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from crew.tool_schema import _TOOL_SCHEMAS, AGENT_TOOLS


def _run(coro):
    return asyncio.run(coro)


# ── tool_schema 注册测试 ──


class TestNewToolSchemas:
    def test_web_search_in_schemas(self):
        assert "web_search" in _TOOL_SCHEMAS
        schema = _TOOL_SCHEMAS["web_search"]
        assert schema["name"] == "web_search"
        assert "query" in schema["input_schema"]["properties"]
        assert schema["input_schema"]["required"] == ["query"]

    def test_create_note_in_schemas(self):
        assert "create_note" in _TOOL_SCHEMAS
        schema = _TOOL_SCHEMAS["create_note"]
        assert schema["name"] == "create_note"
        props = schema["input_schema"]["properties"]
        assert "title" in props
        assert "content" in props
        assert schema["input_schema"]["required"] == ["title", "content"]

    def test_new_tools_in_agent_tools(self):
        assert "web_search" in AGENT_TOOLS
        assert "create_note" in AGENT_TOOLS


# ── web_search handler 测试 ──


class TestWebSearchHandler:
    def test_empty_query(self):
        from crew.webhook import _tool_web_search

        result = _run(_tool_web_search({"query": ""}))
        assert "错误" in result

    @patch("httpx.AsyncClient")
    def test_search_returns_results(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.text = (
            '<li class="b_algo"><a href="https://example.com">Example Title</a>'
            '<p>This is a snippet</p></li>'
        )
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        from crew.webhook import _tool_web_search

        result = _run(_tool_web_search({"query": "test search"}))
        assert "Example Title" in result
        assert "https://example.com" in result

    @patch("httpx.AsyncClient")
    def test_search_no_results(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.text = "<html><body>No results</body></html>"
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        from crew.webhook import _tool_web_search

        result = _run(_tool_web_search({"query": "xyznonexistent"}))
        assert "未找到" in result

    def test_search_network_error(self):
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("network error")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            from crew.webhook import _tool_web_search

            result = _run(_tool_web_search({"query": "test"}))
            assert "搜索失败" in result


# ── create_note handler 测试 ──


class TestCreateNoteHandler:
    def test_creates_note_file(self, tmp_path: Path):
        ctx = MagicMock()
        ctx.project_dir = tmp_path

        from crew.webhook import _tool_create_note

        result = _run(_tool_create_note(
            {"title": "测试备忘", "content": "明天开会讨论 AI 方向", "tags": "meeting,ai"},
            ctx=ctx,
        ))
        assert "笔记已保存" in result

        notes_dir = tmp_path / ".crew" / "notes"
        notes = list(notes_dir.glob("*.md"))
        assert len(notes) == 1

        content = notes[0].read_text(encoding="utf-8")
        assert "title: 测试备忘" in content
        assert "明天开会讨论 AI 方向" in content
        assert "meeting,ai" in content

    def test_empty_content_error(self):
        from crew.webhook import _tool_create_note

        result = _run(_tool_create_note({"title": "test", "content": ""}))
        assert "错误" in result

    def test_title_sanitization(self, tmp_path: Path):
        ctx = MagicMock()
        ctx.project_dir = tmp_path

        from crew.webhook import _tool_create_note

        result = _run(_tool_create_note(
            {"title": "../../etc/passwd", "content": "test"},
            ctx=ctx,
        ))
        assert "笔记已保存" in result

        # 文件应在 notes 目录内，不应路径穿越
        notes = list((tmp_path / ".crew" / "notes").glob("*.md"))
        assert len(notes) == 1
        assert ".." not in notes[0].name


# ── feishu.py 图片消息解析测试 ──


class TestImageMessageParsing:
    def test_parse_image_message(self):
        from crew.feishu import parse_message_event

        payload = {
            "event": {
                "message": {
                    "message_id": "msg_img_001",
                    "chat_id": "oc_test",
                    "chat_type": "group",
                    "message_type": "image",
                    "content": json.dumps({"image_key": "img-key-abc"}),
                    "mentions": [],
                },
                "sender": {"sender_id": {"open_id": "ou_user1"}},
            }
        }
        event = parse_message_event(payload)
        assert event is not None
        assert event.msg_type == "image"
        assert event.image_key == "img-key-abc"
        assert event.text == ""

    def test_parse_video_still_returns_none(self):
        from crew.feishu import parse_message_event

        payload = {
            "event": {
                "message": {
                    "message_id": "msg_vid_001",
                    "chat_id": "oc_test",
                    "message_type": "video",
                    "content": "{}",
                },
                "sender": {"sender_id": {"open_id": "ou_user1"}},
            }
        }
        assert parse_message_event(payload) is None

    def test_parse_text_still_works(self):
        from crew.feishu import parse_message_event

        payload = {
            "event": {
                "message": {
                    "message_id": "msg_txt_001",
                    "chat_id": "oc_test",
                    "chat_type": "p2p",
                    "message_type": "text",
                    "content": json.dumps({"text": "hello"}),
                    "mentions": [],
                },
                "sender": {"sender_id": {"open_id": "ou_user1"}},
            }
        }
        event = parse_message_event(payload)
        assert event is not None
        assert event.msg_type == "text"
        assert event.text == "hello"
        assert event.image_key == ""


# ── send_feishu_reply 测试 ──


class TestSendFeishuReply:
    def test_reply_calls_correct_endpoint(self):
        from crew.feishu import send_feishu_reply

        mock_mgr = AsyncMock()
        mock_mgr.get_token.return_value = "test-token"

        with patch("httpx.AsyncClient") as mock_cls:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"code": 0}
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_cls.return_value = mock_client

            result = _run(send_feishu_reply(mock_mgr, "msg_123", "reply text"))
            assert result["code"] == 0

            # 验证调用了 reply endpoint
            call_args = mock_client.post.call_args
            assert "/messages/msg_123/reply" in call_args[0][0]
