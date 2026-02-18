"""测试飞书双向 Bot — 事件解析、token 管理、员工路由、消息发送、webhook 端点."""

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crew.feishu import (
    EventDeduplicator,
    FeishuConfig,
    FeishuMessageEvent,
    FeishuTokenManager,
    load_feishu_config,
    parse_message_event,
    resolve_employee_from_mention,
    send_feishu_card,
    send_feishu_message,
    send_feishu_text,
    verify_feishu_event,
)


def _run(coro):
    return asyncio.run(coro)


def _mock_httpx_client(*, status_code=200, json_data=None):
    """创建 mock httpx.AsyncClient."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data or {"code": 0, "msg": "success"}

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# ── 配置 ──


class TestFeishuConfig:
    """飞书配置模型."""

    def test_defaults(self):
        cfg = FeishuConfig()
        assert cfg.app_id == ""
        assert cfg.app_secret == ""
        assert cfg.verification_token == ""
        assert cfg.default_employee == ""

    def test_from_values(self):
        cfg = FeishuConfig(
            app_id="cli_xxx",
            app_secret="secret",
            verification_token="tok",
            default_employee="product-manager",
        )
        assert cfg.app_id == "cli_xxx"
        assert cfg.default_employee == "product-manager"


class TestLoadFeishuConfig:
    """配置加载."""

    def test_from_yaml(self, tmp_path):
        crew_dir = tmp_path / ".crew"
        crew_dir.mkdir()
        (crew_dir / "feishu.yaml").write_text(
            "app_id: cli_test\napp_secret: secret_test\nverification_token: tok_test\n",
            encoding="utf-8",
        )
        cfg = load_feishu_config(tmp_path)
        assert cfg.app_id == "cli_test"
        assert cfg.app_secret == "secret_test"
        assert cfg.verification_token == "tok_test"

    def test_from_env(self, tmp_path):
        with patch.dict(os.environ, {
            "FEISHU_APP_ID": "env_id",
            "FEISHU_APP_SECRET": "env_secret",
            "FEISHU_VERIFICATION_TOKEN": "env_tok",
        }, clear=False):
            cfg = load_feishu_config(tmp_path)
        assert cfg.app_id == "env_id"
        assert cfg.app_secret == "env_secret"

    def test_yaml_overrides_env(self, tmp_path):
        crew_dir = tmp_path / ".crew"
        crew_dir.mkdir()
        (crew_dir / "feishu.yaml").write_text(
            "app_id: yaml_id\napp_secret: yaml_secret\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {
            "FEISHU_APP_ID": "env_id",
            "FEISHU_APP_SECRET": "env_secret",
        }, clear=False):
            cfg = load_feishu_config(tmp_path)
        assert cfg.app_id == "yaml_id"  # YAML 优先

    def test_no_config(self, tmp_path):
        cfg = load_feishu_config(tmp_path)
        assert cfg.app_id == ""


# ── 事件验证 ──


class TestVerifyFeishuEvent:
    """飞书事件 token 验证."""

    def test_valid_token(self):
        assert verify_feishu_event("my-token", "my-token") is True

    def test_invalid_token(self):
        assert verify_feishu_event("my-token", "wrong") is False

    def test_empty_verification_token(self):
        assert verify_feishu_event("", "any") is False

    def test_empty_payload_token(self):
        assert verify_feishu_event("my-token", "") is False


# ── 事件解析 ──


class TestParseMessageEvent:
    """飞书消息事件解析."""

    def test_text_message(self):
        payload = {
            "event": {
                "message": {
                    "message_id": "msg_001",
                    "chat_id": "oc_xxx",
                    "chat_type": "group",
                    "message_type": "text",
                    "content": json.dumps({"text": "@_user_1 审查 auth.py"}),
                    "mentions": [{
                        "key": "@_user_1",
                        "id": {"open_id": "ou_bot"},
                        "name": "林锐",
                    }],
                },
                "sender": {"sender_id": {"open_id": "ou_sender"}},
            }
        }
        event = parse_message_event(payload)
        assert event is not None
        assert event.message_id == "msg_001"
        assert event.chat_id == "oc_xxx"
        assert event.text == "审查 auth.py"
        assert len(event.mentions) == 1
        assert event.mentions[0]["name"] == "林锐"
        assert event.sender_id == "ou_sender"

    def test_image_message_parsed(self):
        """image 消息现在能被解析（用于友好拒绝回复）."""
        payload = {
            "event": {
                "message": {
                    "message_id": "msg_002",
                    "chat_id": "oc_xxx",
                    "message_type": "image",
                    "content": json.dumps({"image_key": "img-abc"}),
                },
                "sender": {"sender_id": {"open_id": "ou_sender"}},
            }
        }
        event = parse_message_event(payload)
        assert event is not None
        assert event.msg_type == "image"
        assert event.image_key == "img-abc"

    def test_unsupported_msg_type_returns_none(self):
        """video 等不支持的消息类型仍返回 None."""
        payload = {
            "event": {
                "message": {
                    "message_id": "msg_002",
                    "chat_id": "oc_xxx",
                    "message_type": "video",
                    "content": "{}",
                },
                "sender": {"sender_id": {"open_id": "ou_sender"}},
            }
        }
        assert parse_message_event(payload) is None

    def test_no_mentions(self):
        payload = {
            "event": {
                "message": {
                    "message_id": "msg_003",
                    "chat_id": "oc_xxx",
                    "message_type": "text",
                    "content": json.dumps({"text": "hello world"}),
                },
                "sender": {"sender_id": {"open_id": "ou_sender"}},
            }
        }
        event = parse_message_event(payload)
        assert event is not None
        assert event.text == "hello world"
        assert event.mentions == []

    def test_multiple_mentions(self):
        payload = {
            "event": {
                "message": {
                    "message_id": "msg_004",
                    "chat_id": "oc_xxx",
                    "message_type": "text",
                    "content": json.dumps({"text": "@_user_1 @_user_2 审查代码"}),
                    "mentions": [
                        {"key": "@_user_1", "id": {"open_id": "ou_1"}, "name": "林锐"},
                        {"key": "@_user_2", "id": {"open_id": "ou_2"}, "name": "程薇"},
                    ],
                },
                "sender": {"sender_id": {"open_id": "ou_sender"}},
            }
        }
        event = parse_message_event(payload)
        assert event is not None
        assert event.text == "审查代码"
        assert len(event.mentions) == 2

    def test_malformed_content(self):
        payload = {
            "event": {
                "message": {
                    "message_id": "msg_005",
                    "chat_id": "oc_xxx",
                    "message_type": "text",
                    "content": "not-valid-json",
                },
                "sender": {"sender_id": {"open_id": "ou_sender"}},
            }
        }
        event = parse_message_event(payload)
        assert event is not None
        assert event.text == ""


# ── 员工路由 ──


class TestResolveEmployee:
    """从 @mention 匹配员工."""

    def _make_discovery(self):
        """创建 mock DiscoveryResult."""
        emp1 = MagicMock()
        emp1.name = "code-reviewer"
        emp1.character_name = "林锐"
        emp1.display_name = "代码审查"
        emp1.triggers = ["review", "审查"]

        emp2 = MagicMock()
        emp2.name = "test-engineer"
        emp2.character_name = "程薇"
        emp2.display_name = "测试工程师"
        emp2.triggers = ["test"]

        discovery = MagicMock()
        discovery.employees = {"code-reviewer": emp1, "test-engineer": emp2}

        def _get(name_or_trigger):
            if name_or_trigger in ("code-reviewer", "review", "审查"):
                return emp1
            if name_or_trigger in ("test-engineer", "test"):
                return emp2
            return None

        discovery.get = _get
        return discovery

    def test_match_by_character_name(self):
        d = self._make_discovery()
        mentions = [{"key": "@_user_1", "id": "ou_1", "name": "林锐"}]
        name, text = resolve_employee_from_mention(mentions, "审查 auth.py", d)
        assert name == "code-reviewer"
        assert text == "审查 auth.py"

    def test_match_by_display_name(self):
        d = self._make_discovery()
        mentions = [{"key": "@_user_1", "id": "ou_1", "name": "代码审查"}]
        name, text = resolve_employee_from_mention(mentions, "审查代码", d)
        assert name == "code-reviewer"

    def test_match_by_trigger(self):
        d = self._make_discovery()
        mentions = [{"key": "@_user_1", "id": "ou_1", "name": "review"}]
        name, text = resolve_employee_from_mention(mentions, "review auth", d)
        assert name == "code-reviewer"

    def test_no_match_uses_default(self):
        d = self._make_discovery()
        mentions = [{"key": "@_user_1", "id": "ou_1", "name": "Unknown"}]
        name, text = resolve_employee_from_mention(
            mentions, "do something", d, default_employee="product-manager",
        )
        assert name == "product-manager"
        assert text == "do something"

    def test_no_match_no_default(self):
        d = self._make_discovery()
        mentions = [{"key": "@_user_1", "id": "ou_1", "name": "Unknown"}]
        name, text = resolve_employee_from_mention(mentions, "hello", d)
        assert name is None
        assert text == "hello"

    def test_text_prefix_match(self):
        d = self._make_discovery()
        name, text = resolve_employee_from_mention(
            [], "code-reviewer 审查 auth.py", d,
        )
        assert name == "code-reviewer"
        assert text == "审查 auth.py"

    def test_empty_mentions_no_prefix(self):
        d = self._make_discovery()
        name, text = resolve_employee_from_mention([], "random text", d)
        assert name is None


# ── Token 管理 ──


class TestFeishuTokenManager:
    """飞书 token 管理器."""

    def test_initial_fetch(self):
        mock_client = _mock_httpx_client(json_data={
            "code": 0,
            "tenant_access_token": "t-fresh",
            "expire": 7200,
        })
        mgr = FeishuTokenManager("app_id", "app_secret")

        with patch("httpx.AsyncClient", return_value=mock_client):
            token = _run(mgr.get_token())

        assert token == "t-fresh"
        mock_client.post.assert_called_once()

    def test_cached_token(self):
        mock_client = _mock_httpx_client(json_data={
            "code": 0,
            "tenant_access_token": "t-cached",
            "expire": 7200,
        })
        mgr = FeishuTokenManager("app_id", "app_secret")

        with patch("httpx.AsyncClient", return_value=mock_client):
            t1 = _run(mgr.get_token())
            t2 = _run(mgr.get_token())

        assert t1 == t2 == "t-cached"
        assert mock_client.post.call_count == 1  # 只调用一次

    def test_token_refresh_on_expiry(self):
        mock_client = _mock_httpx_client(json_data={
            "code": 0,
            "tenant_access_token": "t-new",
            "expire": 7200,
        })
        mgr = FeishuTokenManager("app_id", "app_secret")
        # 手动设置已过期
        mgr._token = "t-old"
        mgr._expire_at = time.time() - 100

        with patch("httpx.AsyncClient", return_value=mock_client):
            token = _run(mgr.get_token())

        assert token == "t-new"
        mock_client.post.assert_called_once()

    def test_api_error(self):
        mock_client = _mock_httpx_client(json_data={
            "code": 10003,
            "msg": "invalid app_id",
        })
        mgr = FeishuTokenManager("bad_id", "bad_secret")

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(RuntimeError, match="invalid app_id"):
                _run(mgr.get_token())


# ── 事件去重 ──


class TestEventDeduplicator:
    """事件去重器."""

    def test_first_seen(self):
        dedup = EventDeduplicator()
        assert dedup.is_duplicate("msg_001") is False

    def test_duplicate(self):
        dedup = EventDeduplicator()
        dedup.is_duplicate("msg_001")
        assert dedup.is_duplicate("msg_001") is True

    def test_different_ids(self):
        dedup = EventDeduplicator()
        dedup.is_duplicate("msg_001")
        assert dedup.is_duplicate("msg_002") is False

    def test_cleanup_on_overflow(self):
        dedup = EventDeduplicator(ttl_seconds=0.01, max_size=2)
        dedup.is_duplicate("a")
        dedup.is_duplicate("b")
        import time
        time.sleep(0.02)
        # 第 3 个触发清理，a 和 b 已过期
        dedup.is_duplicate("c")
        assert dedup.is_duplicate("a") is False


# ── 消息发送 ──


class TestSendFeishuMessage:
    """飞书消息发送."""

    def test_send_text(self):
        mock_client = _mock_httpx_client(json_data={"code": 0, "msg": "success"})
        mgr = FeishuTokenManager("id", "secret")
        mgr._token = "t-test"
        mgr._expire_at = time.time() + 3600

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = _run(send_feishu_text(mgr, "oc_xxx", "hello"))

        assert result["code"] == 0
        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs["json"]
        assert body["receive_id"] == "oc_xxx"
        assert body["msg_type"] == "text"
        assert '"text": "hello"' in body["content"] or "hello" in body["content"]

    def test_send_card(self):
        mock_client = _mock_httpx_client(json_data={"code": 0, "msg": "success"})
        mgr = FeishuTokenManager("id", "secret")
        mgr._token = "t-test"
        mgr._expire_at = time.time() + 3600

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = _run(send_feishu_card(
                mgr, "oc_xxx",
                task_name="daily-review",
                task_result={"output": "一切正常"},
                task_error=None,
            ))

        assert result["code"] == 0
        body = mock_client.post.call_args.kwargs["json"]
        assert body["msg_type"] == "interactive"

    def test_send_message_auth_header(self):
        mock_client = _mock_httpx_client(json_data={"code": 0})
        mgr = FeishuTokenManager("id", "secret")
        mgr._token = "t-auth"
        mgr._expire_at = time.time() + 3600

        with patch("httpx.AsyncClient", return_value=mock_client):
            _run(send_feishu_message(mgr, "oc_xxx", {"text": "hi"}, "text"))

        headers = mock_client.post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer t-auth"

    def test_send_api_failure_logged(self):
        mock_client = _mock_httpx_client(json_data={"code": 99999, "msg": "permission denied"})
        mgr = FeishuTokenManager("id", "secret")
        mgr._token = "t-test"
        mgr._expire_at = time.time() + 3600

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = _run(send_feishu_text(mgr, "oc_xxx", "hello"))

        assert result["code"] == 99999  # 不抛异常，只 log warning


# ── Webhook 端点 ──


pytest.importorskip("starlette")

from starlette.testclient import TestClient  # noqa: E402

from crew.webhook import create_webhook_app  # noqa: E402
from crew.webhook_config import WebhookConfig  # noqa: E402

TOKEN = "test-token-feishu"


def _make_feishu_client(feishu_config=None, token=TOKEN):
    """创建带飞书配置的测试客户端."""
    app = create_webhook_app(
        project_dir=Path("/tmp/test"),
        token=token,
        config=WebhookConfig(),
        feishu_config=feishu_config,
    )
    return TestClient(app)


class TestFeishuEventEndpoint:
    """飞书事件回调端点."""

    def test_url_verification(self):
        """URL 验证 challenge 应原样返回."""
        client = _make_feishu_client()
        resp = client.post("/feishu/event", json={
            "type": "url_verification",
            "challenge": "test-challenge-xxx",
        })
        assert resp.status_code == 200
        assert resp.json()["challenge"] == "test-challenge-xxx"

    def test_no_auth_required(self):
        """飞书事件端点不需要 Bearer token."""
        client = _make_feishu_client()
        resp = client.post("/feishu/event", json={
            "type": "url_verification",
            "challenge": "abc",
        })
        assert resp.status_code == 200

    def test_not_configured(self):
        """未配置飞书时返回 501."""
        client = _make_feishu_client(feishu_config=None)
        resp = client.post("/feishu/event", json={
            "header": {"event_type": "im.message.receive_v1", "token": "t"},
            "event": {
                "message": {
                    "message_id": "msg_x",
                    "chat_id": "oc_x",
                    "message_type": "text",
                    "content": json.dumps({"text": "hello"}),
                },
                "sender": {"sender_id": {"open_id": "ou_x"}},
            },
        })
        assert resp.status_code == 501

    def test_invalid_token(self):
        """验证 token 不匹配返回 401."""
        config = FeishuConfig(
            app_id="id",
            app_secret="secret",
            verification_token="correct-token",
        )
        client = _make_feishu_client(feishu_config=config)
        resp = client.post("/feishu/event", json={
            "header": {
                "event_type": "im.message.receive_v1",
                "token": "wrong-token",
            },
            "event": {},
        })
        assert resp.status_code == 401

    def test_no_verification_token_warns_but_allows(self):
        """未配置 verification_token 时应跳过验证但放行."""
        config = FeishuConfig(app_id="id", app_secret="secret")
        client = _make_feishu_client(feishu_config=config)
        resp = client.post("/feishu/event", json={
            "header": {"event_type": "im.chat.member.bot.added_v1"},
            "event": {},
        })
        assert resp.status_code == 200
        assert resp.json()["message"] == "ignored"

    def test_non_message_event_ignored(self):
        """非消息事件应被忽略."""
        config = FeishuConfig(app_id="id", app_secret="secret", verification_token="tok")
        client = _make_feishu_client(feishu_config=config)
        resp = client.post("/feishu/event", json={
            "header": {"event_type": "im.chat.member.bot.added_v1", "token": "tok"},
            "event": {},
        })
        assert resp.status_code == 200
        assert resp.json()["message"] == "ignored"

    def test_unsupported_message_type(self):
        """video 等不支持的消息类型应返回 unsupported."""
        config = FeishuConfig(app_id="id", app_secret="secret", verification_token="tok")
        client = _make_feishu_client(feishu_config=config)
        resp = client.post("/feishu/event", json={
            "header": {"event_type": "im.message.receive_v1", "token": "tok"},
            "event": {
                "message": {
                    "message_id": "msg_vid",
                    "chat_id": "oc_xxx",
                    "message_type": "video",
                    "content": "{}",
                },
                "sender": {"sender_id": {"open_id": "ou_x"}},
            },
        })
        assert resp.status_code == 200
        assert resp.json()["message"] == "unsupported message type"

    @patch("crew.webhook._feishu_dispatch", new_callable=AsyncMock)
    def test_message_dispatched(self, mock_dispatch):
        """合法消息事件应触发后台处理."""
        config = FeishuConfig(app_id="id", app_secret="secret", verification_token="tok")
        client = _make_feishu_client(feishu_config=config)
        resp = client.post("/feishu/event", json={
            "header": {"event_type": "im.message.receive_v1", "token": "tok"},
            "event": {
                "message": {
                    "message_id": "msg_ok",
                    "chat_id": "oc_xxx",
                    "chat_type": "group",
                    "message_type": "text",
                    "content": json.dumps({"text": "@_user_1 审查代码"}),
                    "mentions": [{
                        "key": "@_user_1",
                        "id": {"open_id": "ou_bot"},
                        "name": "林锐",
                    }],
                },
                "sender": {"sender_id": {"open_id": "ou_user"}},
            },
        })
        assert resp.status_code == 200
        assert resp.json()["message"] == "ok"

    def test_duplicate_event(self):
        """重复 message_id 应被去重."""
        config = FeishuConfig(app_id="id", app_secret="secret", verification_token="tok")
        client = _make_feishu_client(feishu_config=config)
        event_payload = {
            "header": {"event_type": "im.message.receive_v1", "token": "tok"},
            "event": {
                "message": {
                    "message_id": "msg_dup",
                    "chat_id": "oc_xxx",
                    "message_type": "text",
                    "content": json.dumps({"text": "hello"}),
                },
                "sender": {"sender_id": {"open_id": "ou_x"}},
            },
        }
        # 第一次处理
        with patch("crew.webhook._feishu_dispatch", new_callable=AsyncMock):
            resp1 = client.post("/feishu/event", json=event_payload)
            assert resp1.status_code == 200
            assert resp1.json()["message"] == "ok"

        # 第二次去重
        resp2 = client.post("/feishu/event", json=event_payload)
        assert resp2.json()["message"] == "duplicate"


# ── 闲聊快速路径判断 ──

class TestNeedsTools:
    """测试 _needs_tools() 闲聊判断."""

    def test_casual_greetings(self):
        from crew.webhook_feishu import _needs_tools
        assert not _needs_tools("早")
        assert not _needs_tools("下午好")
        assert not _needs_tools("晚安")
        assert not _needs_tools("你好")

    def test_casual_chat(self):
        from crew.webhook_feishu import _needs_tools
        assert not _needs_tools("下午有点累，不想开会了")
        assert not _needs_tools("周末有什么好吃的")
        assert not _needs_tools("帮我想个团建活动")
        assert not _needs_tools("最近怎么样")

    def test_work_keywords_trigger(self):
        from crew.webhook_feishu import _needs_tools
        assert _needs_tools("查一下昨天的数据")
        assert _needs_tools("帮我查下今天的日程")
        assert _needs_tools("创建一个飞书文档")
        assert _needs_tools("看看GitHub上的PR")
        assert _needs_tools("帮我发送邮件给张三")
        assert _needs_tools("项目进度怎么样了")

    def test_long_text_is_work(self):
        from crew.webhook_feishu import _needs_tools
        # 超过 200 字的消息被认为是正式任务
        assert _needs_tools("x" * 201)

    def test_empty_is_work(self):
        from crew.webhook_feishu import _needs_tools
        assert _needs_tools("")

    def test_url_triggers_tools(self):
        """URL 消息应走工具路径."""
        from crew.webhook_feishu import _needs_tools
        assert _needs_tools("帮我看看 https://example.com 上的内容")
        assert _needs_tools("http://localhost:8080/status")
        assert _needs_tools("https://github.com/foo/bar/pull/123")

    def test_memory_keywords_trigger(self):
        """记忆相关关键词应走工具路径."""
        from crew.webhook_feishu import _needs_tools
        assert _needs_tools("记住这件事")
        assert _needs_tools("帮我做个笔记")
        assert _needs_tools("写入到备忘录")


# ── 飞书文本净化 ──


class TestSanitizeFeishuText:
    """测试 _sanitize_feishu_text() 防 230001."""

    def test_normal_text_unchanged(self):
        from crew.feishu import _sanitize_feishu_text
        assert _sanitize_feishu_text("你好世界") == "你好世界"

    def test_removes_null_bytes(self):
        from crew.feishu import _sanitize_feishu_text
        assert _sanitize_feishu_text("hello\x00world") == "helloworld"

    def test_removes_control_chars(self):
        from crew.feishu import _sanitize_feishu_text
        # \x01-\x08, \x0b, \x0c, \x0e-\x1f 应被移除
        text = "a\x01b\x02c\x0bd\x0ee"
        result = _sanitize_feishu_text(text)
        assert result == "abcde"

    def test_preserves_newline_tab(self):
        from crew.feishu import _sanitize_feishu_text
        text = "line1\nline2\tindented\r\nline3"
        assert _sanitize_feishu_text(text) == text

    def test_truncates_long_text(self):
        from crew.feishu import _sanitize_feishu_text, _FEISHU_TEXT_MAX_LEN
        long_text = "a" * (_FEISHU_TEXT_MAX_LEN + 500)
        result = _sanitize_feishu_text(long_text)
        assert len(result) == _FEISHU_TEXT_MAX_LEN + 3  # +3 for "..."
        assert result.endswith("...")

    def test_empty_passthrough(self):
        from crew.feishu import _sanitize_feishu_text
        assert _sanitize_feishu_text("") == ""

    def test_none_like_empty(self):
        from crew.feishu import _sanitize_feishu_text
        # 空字符串 falsy 走 early return
        assert _sanitize_feishu_text("") == ""


# ── 用户名解析 ──


class TestGetUserName:
    """测试 get_user_name() 缓存和 API 调用."""

    def test_cached_result(self):
        from crew.feishu import _USER_NAME_CACHE, get_user_name

        _USER_NAME_CACHE["ou_cached"] = "张三"
        try:
            mgr = MagicMock()
            name = _run(get_user_name(mgr, "ou_cached"))
            assert name == "张三"
        finally:
            _USER_NAME_CACHE.pop("ou_cached", None)

    def test_empty_open_id(self):
        from crew.feishu import get_user_name
        mgr = MagicMock()
        assert _run(get_user_name(mgr, "")) == ""
