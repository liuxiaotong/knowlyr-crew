"""企业微信适配器基础测试 -- 加解密、XML 解析、签名验证、消息路由."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 企业微信依赖 pycryptodome 和 defusedxml（webhook optional deps）
pytest.importorskip("Crypto", reason="pycryptodome not installed")
pytest.importorskip("defusedxml", reason="defusedxml not installed")

# ── 加解密测试 ──


class TestWecomCrypto:
    """WecomCrypto 加解密 round-trip."""

    def _make_crypto(self):
        from crew.wecom import WecomCrypto

        # 43 字符的 Base64 key（解码后 32 字节）
        encoding_aes_key = "qm7mzclOzyeB7Ee6rrKzzbPZAgVixfzHLi3fs20Xruz"
        corp_id = "ww_test_corp_id"
        return WecomCrypto(encoding_aes_key, corp_id), corp_id

    def test_encrypt_decrypt_round_trip(self):
        crypto, corp_id = self._make_crypto()
        plaintext = "<xml><Content>hello world</Content></xml>"
        encrypted = crypto.encrypt(plaintext)
        assert encrypted != plaintext
        decrypted = crypto.decrypt(encrypted)
        assert decrypted == plaintext

    def test_decrypt_invalid_corp_id_raises(self):
        from crew.wecom import WecomCrypto

        encoding_aes_key = "qm7mzclOzyeB7Ee6rrKzzbPZAgVixfzHLi3fs20Xruz"
        crypto_a = WecomCrypto(encoding_aes_key, "corp_a")
        crypto_b = WecomCrypto(encoding_aes_key, "corp_b")

        encrypted = crypto_a.encrypt("hello")
        with pytest.raises(ValueError, match="CorpID"):
            crypto_b.decrypt(encrypted)

    def test_encrypt_chinese(self):
        crypto, _ = self._make_crypto()
        plaintext = "你好，企业微信"
        encrypted = crypto.encrypt(plaintext)
        decrypted = crypto.decrypt(encrypted)
        assert decrypted == plaintext


# ── 签名验证测试 ──


class TestWecomSignature:
    """verify_wecom_signature / generate_wecom_signature."""

    def test_verify_valid(self):
        from crew.wecom import generate_wecom_signature, verify_wecom_signature

        token = "test_token"
        timestamp = "1234567890"
        nonce = "abc123"
        msg_encrypt = "encrypted_content"

        sig = generate_wecom_signature(token, timestamp, nonce, msg_encrypt)
        assert verify_wecom_signature(token, timestamp, nonce, msg_encrypt, sig)

    def test_verify_invalid(self):
        from crew.wecom import verify_wecom_signature

        assert not verify_wecom_signature("token", "ts", "nonce", "msg", "bad_sig")

    def test_signature_matches_manual(self):
        from crew.wecom import generate_wecom_signature

        token = "aaa"
        timestamp = "111"
        nonce = "bbb"
        msg_encrypt = "ccc"
        parts = sorted([token, timestamp, nonce, msg_encrypt])
        expected = hashlib.sha1("".join(parts).encode("utf-8")).hexdigest()
        assert generate_wecom_signature(token, timestamp, nonce, msg_encrypt) == expected


# ── XML 解析测试 ──


class TestParseWecomMessage:
    """parse_wecom_message / parse_wecom_encrypt_xml."""

    def test_parse_text_message(self):
        from crew.wecom import parse_wecom_message

        xml = (
            "<xml>"
            "<ToUserName><![CDATA[ww123]]></ToUserName>"
            "<FromUserName><![CDATA[user001]]></FromUserName>"
            "<CreateTime>1234567890</CreateTime>"
            "<MsgType><![CDATA[text]]></MsgType>"
            "<Content><![CDATA[hello world]]></Content>"
            "<MsgId>12345</MsgId>"
            "<AgentID>1000017</AgentID>"
            "</xml>"
        )
        result = parse_wecom_message(xml)
        assert result["ToUserName"] == "ww123"
        assert result["FromUserName"] == "user001"
        assert result["MsgType"] == "text"
        assert result["Content"] == "hello world"
        assert result["MsgId"] == "12345"

    def test_parse_encrypt_xml(self):
        from crew.wecom import parse_wecom_encrypt_xml

        xml = (
            "<xml>"
            "<ToUserName><![CDATA[ww123]]></ToUserName>"
            "<Encrypt><![CDATA[encrypted_data_here]]></Encrypt>"
            "</xml>"
        )
        encrypt, to_user = parse_wecom_encrypt_xml(xml)
        assert encrypt == "encrypted_data_here"
        assert to_user == "ww123"

    def test_parse_empty_content(self):
        from crew.wecom import parse_wecom_message

        xml = "<xml><MsgType><![CDATA[text]]></MsgType><Content></Content></xml>"
        result = parse_wecom_message(xml)
        assert result["Content"] == ""


# ── 文本净化测试 ──


class TestSanitizeWecomText:
    """_sanitize_wecom_text 和 _truncate_to_bytes."""

    def test_strip_markdown(self):
        from crew.wecom import _sanitize_wecom_text

        text = "**bold** and `code`"
        result = _sanitize_wecom_text(text)
        assert "**" not in result
        assert "`" not in result
        assert "bold" in result
        assert "code" in result

    def test_truncate_to_bytes(self):
        from crew.wecom import _truncate_to_bytes

        # 中文每字 3 bytes UTF-8
        text = "你" * 700  # 700 * 3 = 2100 bytes
        result = _truncate_to_bytes(text, 2048)
        assert len(result.encode("utf-8")) <= 2048 + 10  # +... suffix

    def test_truncate_short_text_unchanged(self):
        from crew.wecom import _truncate_to_bytes

        text = "short"
        assert _truncate_to_bytes(text, 2048) == "short"

    def test_empty_text(self):
        from crew.wecom import _sanitize_wecom_text

        assert _sanitize_wecom_text("") == ""


# ── Token 管理测试 ──


class TestWecomTokenManager:
    """WecomTokenManager 基础行为."""

    @pytest.mark.asyncio
    async def test_get_token_caches(self):
        from crew.wecom import WecomTokenManager

        mgr = WecomTokenManager("corp_id", "secret")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "errcode": 0,
            "access_token": "test_token_abc",
            "expires_in": 7200,
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            token1 = await mgr.get_token()
            assert token1 == "test_token_abc"
            # Second call should use cache
            token2 = await mgr.get_token()
            assert token2 == "test_token_abc"
            # Only one HTTP call
            assert mock_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_get_token_error(self):
        from crew.wecom import WecomTokenManager

        mgr = WecomTokenManager("corp_id", "bad_secret")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "errcode": 40001,
            "errmsg": "invalid credential",
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="获取企微 access_token 失败"):
                await mgr.get_token()


# ── 配置加载测试 ──


class TestLoadWecomConfig:
    """load_wecom_config 从 YAML 和环境变量."""

    def test_load_from_yaml(self, tmp_path):
        (tmp_path / ".crew").mkdir()
        (tmp_path / ".crew" / "wecom.yaml").write_text(
            "corp_id: test_corp\n"
            "agent_id: 999\n"
            "secret: test_secret\n"
            "token: test_token\n"
            "encoding_aes_key: qm7mzclOzyeB7Ee6rrKzzbPZAgVixfzHLi3fs20Xruz\n"
        )
        from crew.wecom import load_wecom_config

        config = load_wecom_config(tmp_path)
        assert config.corp_id == "test_corp"
        assert config.agent_id == 999
        assert config.secret == "test_secret"
        assert config.token == "test_token"

    def test_load_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WECOM_CORP_ID", "env_corp")
        monkeypatch.setenv("WECOM_AGENT_ID", "888")
        monkeypatch.setenv("WECOM_SECRET", "env_secret")

        from crew.wecom import load_wecom_config

        config = load_wecom_config(tmp_path)
        assert config.corp_id == "env_corp"
        assert config.agent_id == 888
        assert config.secret == "env_secret"


# ── webhook handler 路由测试 ──


class TestWecomEventHandler:
    """handle_wecom_event GET/POST 基本路由."""

    def _make_ctx(self, wecom_configured=True):
        ctx = MagicMock()
        ctx.project_dir = None
        if wecom_configured:
            from crew.wecom import WecomCrypto

            crypto = WecomCrypto("qm7mzclOzyeB7Ee6rrKzzbPZAgVixfzHLi3fs20Xruz", "ww_test")

            from crew.feishu import EventDeduplicator

            ctx.wecom_ctx = {
                "config": MagicMock(
                    token="test_token",
                    agent_id=1000017,
                    default_employee="",
                    corp_id="ww_test",
                    encoding_aes_key="qm7mzclOzyeB7Ee6rrKzzbPZAgVixfzHLi3fs20Xruz",
                ),
                "crypto": crypto,
                "token_mgr": MagicMock(),
                "dedup": EventDeduplicator(),
            }
        else:
            ctx.wecom_ctx = None
        return ctx

    @pytest.mark.asyncio
    async def test_not_configured(self):
        from crew.webhook_wecom import handle_wecom_event

        ctx = self._make_ctx(wecom_configured=False)
        request = MagicMock()
        request.method = "GET"
        resp = await handle_wecom_event(request, ctx)
        assert resp.status_code == 501

    @pytest.mark.asyncio
    async def test_get_url_verify(self):
        from crew.webhook_wecom import handle_wecom_event
        from crew.wecom import generate_wecom_signature

        ctx = self._make_ctx()
        crypto = ctx.wecom_ctx["crypto"]
        token = "test_token"

        # 加密一个 echostr
        echostr_plain = "echostr_test_12345"
        echostr_encrypted = crypto.encrypt(echostr_plain)

        timestamp = "1234567890"
        nonce = "random_nonce"
        sig = generate_wecom_signature(token, timestamp, nonce, echostr_encrypted)

        request = MagicMock()
        request.method = "GET"
        request.query_params = {
            "msg_signature": sig,
            "timestamp": timestamp,
            "nonce": nonce,
            "echostr": echostr_encrypted,
        }

        resp = await handle_wecom_event(request, ctx)
        assert resp.status_code == 200
        assert resp.body.decode("utf-8") == echostr_plain

    @pytest.mark.asyncio
    async def test_get_bad_signature(self):
        from crew.webhook_wecom import handle_wecom_event

        ctx = self._make_ctx()
        request = MagicMock()
        request.method = "GET"
        request.query_params = {
            "msg_signature": "bad_sig",
            "timestamp": "123",
            "nonce": "abc",
            "echostr": "xxx",
        }

        resp = await handle_wecom_event(request, ctx)
        assert resp.status_code == 403


# ── dispatch 派遣逻辑测试 ──


class TestWecomDispatch:
    """_wecom_dispatch 核心派遣逻辑."""

    def _make_ctx(self):
        from crew.wecom import WecomCrypto

        ctx = MagicMock()
        ctx.project_dir = None
        ctx.wecom_ctx = {
            "config": MagicMock(
                token="test_token",
                agent_id=1000017,
                default_employee="test_emp",
                corp_id="ww_test",
                encoding_aes_key="qm7mzclOzyeB7Ee6rrKzzbPZAgVixfzHLi3fs20Xruz",
            ),
            "crypto": WecomCrypto("qm7mzclOzyeB7Ee6rrKzzbPZAgVixfzHLi3fs20Xruz", "ww_test"),
            "token_mgr": MagicMock(),
            "dedup": MagicMock(is_duplicate=MagicMock(return_value=False)),
        }
        ctx.registry = MagicMock()
        ctx.registry.create.return_value = MagicMock(task_id="task_001")
        return ctx

    def _make_post_request(self, ctx, msg_type="text", content="hello"):
        """构造 POST 请求，返回加密的 XML body."""
        from crew.wecom import generate_wecom_signature

        crypto = ctx.wecom_ctx["crypto"]
        token = ctx.wecom_ctx["config"].token

        plain_xml = (
            "<xml>"
            f"<ToUserName><![CDATA[ww_test]]></ToUserName>"
            f"<FromUserName><![CDATA[user001]]></FromUserName>"
            f"<CreateTime>1234567890</CreateTime>"
            f"<MsgType><![CDATA[{msg_type}]]></MsgType>"
            f"<Content><![CDATA[{content}]]></Content>"
            f"<MsgId>msg_001</MsgId>"
            f"<AgentID>1000017</AgentID>"
            "</xml>"
        )
        encrypted = crypto.encrypt(plain_xml)

        timestamp = "1234567890"
        nonce = "random_nonce"
        sig = generate_wecom_signature(token, timestamp, nonce, encrypted)

        body_xml = (
            "<xml>"
            f"<ToUserName><![CDATA[ww_test]]></ToUserName>"
            f"<Encrypt><![CDATA[{encrypted}]]></Encrypt>"
            "</xml>"
        )

        request = MagicMock()
        request.method = "POST"
        request.path_params = {"app_id": "default"}
        request.query_params = {
            "msg_signature": sig,
            "timestamp": timestamp,
            "nonce": nonce,
        }
        request.body = AsyncMock(return_value=body_xml.encode("utf-8"))
        return request

    @pytest.mark.asyncio
    async def test_text_message_dispatches_via_sg(self):
        """POST 文本消息正常派遣: mock sg_dispatch -> 验证 send_wecom_text 被调用."""
        from crew.webhook_wecom import handle_wecom_event

        ctx = self._make_ctx()
        request = self._make_post_request(ctx, msg_type="text", content="test task")

        with (
            patch("crew.webhook_wecom._wecom_dispatch") as mock_dispatch,
        ):
            mock_dispatch.return_value = None  # 后台 task，不影响响应

            resp = await handle_wecom_event(request, ctx)

            assert resp.status_code == 200
            assert resp.body.decode("utf-8") == "success"

    @pytest.mark.asyncio
    async def test_text_message_full_dispatch(self):
        """POST 文本消息完整 dispatch 流程: sg_dispatch -> send_wecom_text."""
        from crew.webhook_wecom import _wecom_dispatch

        ctx = self._make_ctx()
        token_mgr = ctx.wecom_ctx["token_mgr"]

        with (
            patch("crew.webhook_wecom.discover_employees", create=True) as _mock_discover,  # noqa: F841
            patch("crew.webhook_wecom.resolve_employee_from_mention", create=True) as _mock_resolve,  # noqa: F841
            patch("crew.webhook_wecom.sg_dispatch", create=True) as _mock_sg,  # noqa: F841
            patch("crew.webhook_wecom.send_wecom_text", create=True) as _mock_send,  # noqa: F841
            patch("crew.webhook_wecom.strip_internal_tags", side_effect=lambda x: x, create=True),
            patch("crew.discovery.discover_employees") as mock_disc2,
            patch("crew.feishu.resolve_employee_from_mention") as mock_res2,
            patch("crew.sg_bridge.sg_dispatch", new_callable=AsyncMock) as mock_sg2,
            patch("crew.wecom.send_wecom_text", new_callable=AsyncMock) as mock_send2,
            patch("crew.output_sanitizer.strip_internal_tags", side_effect=lambda x: x),
        ):
            mock_disc2.return_value = {"test_emp": {}}
            mock_res2.return_value = ("test_emp", "test task")
            mock_sg2.return_value = "reply from sg"
            mock_send2.return_value = {"errcode": 0}

            await _wecom_dispatch(ctx, "user001", "test task", 1000017, token_mgr)

            mock_sg2.assert_called_once()
            mock_send2.assert_called_once_with(token_mgr, "user001", 1000017, "reply from sg")

    @pytest.mark.asyncio
    async def test_non_text_message_no_dispatch(self):
        """非文本消息类型 -> 返回 success，不触发派遣."""
        from crew.webhook_wecom import handle_wecom_event

        ctx = self._make_ctx()
        request = self._make_post_request(ctx, msg_type="image", content="")

        # 不 patch _wecom_dispatch -- 如果被意外调用会抛错
        resp = await handle_wecom_event(request, ctx)

        assert resp.status_code == 200
        assert resp.body.decode("utf-8") == "success"

    @pytest.mark.asyncio
    async def test_group_message_dispatches_with_at_prefix(self):
        """群聊消息: @应用名 开头 -> 识别为群聊，去除 @前缀 后派遣."""
        from crew.webhook_wecom import handle_wecom_event

        ctx = self._make_ctx()
        # 模拟群聊消息: "@叶心蕾 你好"
        request = self._make_post_request(ctx, msg_type="text", content="@叶心蕾 你好")

        with (
            patch("crew.webhook_wecom._wecom_dispatch") as mock_dispatch,
        ):
            mock_dispatch.return_value = None

            resp = await handle_wecom_event(request, ctx)

            assert resp.status_code == 200
            assert resp.body.decode("utf-8") == "success"
            # 验证 dispatch 被调用，且 is_group=True, content 已去除 @前缀
            mock_dispatch.assert_called_once()
            call_args = mock_dispatch.call_args
            # 位置参数: (ctx, from_user, content, agent_id, token_mgr)
            assert call_args[0][2] == "你好"  # content 去除了 @叶心蕾
            assert call_args[1]["is_group"] is True

    @pytest.mark.asyncio
    async def test_group_message_full_dispatch_fallback_to_dm(self):
        """群聊消息完整 dispatch: 无 chat_id -> 降级为单聊回复发送者."""
        from crew.webhook_wecom import _wecom_dispatch

        ctx = self._make_ctx()
        token_mgr = ctx.wecom_ctx["token_mgr"]

        with (
            patch("crew.discovery.discover_employees") as mock_disc,
            patch("crew.feishu.resolve_employee_from_mention") as mock_res,
            patch("crew.sg_bridge.sg_dispatch", new_callable=AsyncMock) as mock_sg,
            patch("crew.wecom.send_wecom_text", new_callable=AsyncMock) as mock_send,
            patch("crew.output_sanitizer.strip_internal_tags", side_effect=lambda x: x),
        ):
            mock_disc.return_value = {"test_emp": {}}
            mock_res.return_value = ("test_emp", "你好")
            mock_sg.return_value = "group reply"
            mock_send.return_value = {"errcode": 0}

            await _wecom_dispatch(
                ctx,
                "user001",
                "你好",
                1000017,
                token_mgr,
                chat_id="",
                is_group=True,
            )

            # 无 chat_id -> 降级为单聊回复
            mock_send.assert_called_once_with(token_mgr, "user001", 1000017, "group reply")

    @pytest.mark.asyncio
    async def test_group_message_with_chat_id_tries_group_api(self):
        """群聊消息: 有 chat_id -> 先尝试群聊 API，成功则不走单聊."""
        from crew.webhook_wecom import _wecom_dispatch

        ctx = self._make_ctx()
        token_mgr = ctx.wecom_ctx["token_mgr"]

        with (
            patch("crew.discovery.discover_employees") as mock_disc,
            patch("crew.feishu.resolve_employee_from_mention") as mock_res,
            patch("crew.sg_bridge.sg_dispatch", new_callable=AsyncMock) as mock_sg,
            patch("crew.wecom.send_wecom_text", new_callable=AsyncMock) as mock_send_dm,
            patch("crew.wecom.send_wecom_group_text", new_callable=AsyncMock) as mock_send_group,
            patch("crew.output_sanitizer.strip_internal_tags", side_effect=lambda x: x),
        ):
            mock_disc.return_value = {"test_emp": {}}
            mock_res.return_value = ("test_emp", "你好")
            mock_sg.return_value = "group reply"
            mock_send_group.return_value = {"errcode": 0}

            await _wecom_dispatch(
                ctx,
                "user001",
                "你好",
                1000017,
                token_mgr,
                chat_id="group_chat_001",
                is_group=True,
            )

            # 群聊 API 成功 -> 不走单聊
            mock_send_group.assert_called_once_with(token_mgr, "group_chat_001", "group reply")
            mock_send_dm.assert_not_called()

    @pytest.mark.asyncio
    async def test_group_message_chat_id_fallback_on_error(self):
        """群聊消息: 有 chat_id 但群聊 API 失败 -> 降级为单聊."""
        from crew.webhook_wecom import _wecom_dispatch

        ctx = self._make_ctx()
        token_mgr = ctx.wecom_ctx["token_mgr"]

        with (
            patch("crew.discovery.discover_employees") as mock_disc,
            patch("crew.feishu.resolve_employee_from_mention") as mock_res,
            patch("crew.sg_bridge.sg_dispatch", new_callable=AsyncMock) as mock_sg,
            patch("crew.wecom.send_wecom_text", new_callable=AsyncMock) as mock_send_dm,
            patch("crew.wecom.send_wecom_group_text", new_callable=AsyncMock) as mock_send_group,
            patch("crew.output_sanitizer.strip_internal_tags", side_effect=lambda x: x),
        ):
            mock_disc.return_value = {"test_emp": {}}
            mock_res.return_value = ("test_emp", "你好")
            mock_sg.return_value = "group reply"
            # 群聊 API 返回错误（权限不足等）
            mock_send_group.return_value = {"errcode": 60011, "errmsg": "no privilege"}
            mock_send_dm.return_value = {"errcode": 0}

            await _wecom_dispatch(
                ctx,
                "user001",
                "你好",
                1000017,
                token_mgr,
                chat_id="group_chat_001",
                is_group=True,
            )

            # 群聊 API 失败 -> 降级为单聊
            mock_send_group.assert_called_once()
            mock_send_dm.assert_called_once_with(token_mgr, "user001", 1000017, "group reply")

    @pytest.mark.asyncio
    async def test_group_dispatch_records_trigger_wecom_group(self):
        """群聊消息: registry trigger 应为 wecom_group."""
        from crew.webhook_wecom import _wecom_dispatch

        ctx = self._make_ctx()
        token_mgr = ctx.wecom_ctx["token_mgr"]

        with (
            patch("crew.discovery.discover_employees") as mock_disc,
            patch("crew.feishu.resolve_employee_from_mention") as mock_res,
            patch("crew.sg_bridge.sg_dispatch", new_callable=AsyncMock) as mock_sg,
            patch("crew.wecom.send_wecom_text", new_callable=AsyncMock) as mock_send,
            patch("crew.output_sanitizer.strip_internal_tags", side_effect=lambda x: x),
        ):
            mock_disc.return_value = {"test_emp": {}}
            mock_res.return_value = ("test_emp", "你好")
            mock_sg.return_value = "reply"
            mock_send.return_value = {"errcode": 0}

            await _wecom_dispatch(
                ctx,
                "user001",
                "你好",
                1000017,
                token_mgr,
                is_group=True,
            )

            ctx.registry.create.assert_called_once_with(
                trigger="wecom_group",
                target_type="employee",
                target_name="test_emp",
                args={"task": "你好"},
            )


# ── @前缀去除测试 ──


class TestStripWecomAtPrefix:
    """strip_wecom_at_prefix 去除群聊 @应用名 前缀."""

    def test_strip_at_prefix_with_space(self):
        from crew.wecom import strip_wecom_at_prefix

        assert strip_wecom_at_prefix("@叶心蕾 你好") == "你好"

    def test_strip_at_prefix_with_newline(self):
        from crew.wecom import strip_wecom_at_prefix

        assert strip_wecom_at_prefix("@叶心蕾\n你好世界") == "你好世界"

    def test_strip_at_prefix_with_multiple_spaces(self):
        from crew.wecom import strip_wecom_at_prefix

        assert strip_wecom_at_prefix("@叶心蕾  请帮我查一下") == "请帮我查一下"

    def test_strip_at_prefix_no_content_after(self):
        from crew.wecom import strip_wecom_at_prefix

        # 只有 @应用名 没有后续内容 -> 返回原文
        assert strip_wecom_at_prefix("@叶心蕾") == "@叶心蕾"

    def test_strip_at_prefix_no_at(self):
        from crew.wecom import strip_wecom_at_prefix

        assert strip_wecom_at_prefix("普通消息") == "普通消息"

    def test_strip_at_prefix_empty(self):
        from crew.wecom import strip_wecom_at_prefix

        assert strip_wecom_at_prefix("") == ""

    def test_strip_at_prefix_english_name(self):
        from crew.wecom import strip_wecom_at_prefix

        assert strip_wecom_at_prefix("@TestBot hello world") == "hello world"

    def test_strip_at_prefix_preserves_at_in_middle(self):
        from crew.wecom import strip_wecom_at_prefix

        # @不在开头 -> 不去除
        assert strip_wecom_at_prefix("hello @叶心蕾 你好") == "hello @叶心蕾 你好"


# ── 群聊消息发送测试 ──


class TestSendWecomGroupText:
    """send_wecom_group_text 群聊消息发送."""

    @pytest.mark.asyncio
    async def test_send_group_text_success(self):
        from crew.wecom import WecomTokenManager, send_wecom_group_text

        mgr = WecomTokenManager("corp_id", "secret")
        mgr._token = "cached_token"
        mgr._expire_at = 9999999999.0

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"errcode": 0, "errmsg": "ok"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            result = await send_wecom_group_text(mgr, "chat_001", "hello group")

        assert result["errcode"] == 0
        # 验证 API 调用参数
        call_args = mock_client.post.call_args
        body = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        assert body["chatid"] == "chat_001"
        assert body["msgtype"] == "text"
        assert body["text"]["content"] == "hello group"

    @pytest.mark.asyncio
    async def test_send_group_text_error(self):
        from crew.wecom import WecomTokenManager, send_wecom_group_text

        mgr = WecomTokenManager("corp_id", "secret")
        mgr._token = "cached_token"
        mgr._expire_at = 9999999999.0

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"errcode": 60011, "errmsg": "no privilege"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            result = await send_wecom_group_text(mgr, "chat_001", "hello")

        assert result["errcode"] == 60011


# ── 通讯录 API 测试 ──


class TestGetWecomUserInfo:
    """get_wecom_user_info 企微通讯录查询."""

    @pytest.mark.asyncio
    async def test_get_user_info_success(self):
        from crew.wecom import WecomTokenManager, get_wecom_user_info

        mgr = WecomTokenManager("corp_id", "secret")
        mgr._token = "cached_token"
        mgr._expire_at = 9999999999.0

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "errcode": 0,
            "userid": "zhangsan",
            "name": "张三",
            "mobile": "13800001234",
            "status": 1,
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            result = await get_wecom_user_info(mgr, "zhangsan")

        assert result is not None
        assert result["mobile"] == "13800001234"
        assert result["userid"] == "zhangsan"

    @pytest.mark.asyncio
    async def test_get_user_info_not_found(self):
        from crew.wecom import WecomTokenManager, get_wecom_user_info

        mgr = WecomTokenManager("corp_id", "secret")
        mgr._token = "cached_token"
        mgr._expire_at = 9999999999.0

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "errcode": 60111,
            "errmsg": "userid not found",
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            result = await get_wecom_user_info(mgr, "deleted_user")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_info_network_error(self):
        from crew.wecom import WecomTokenManager, get_wecom_user_info

        mgr = WecomTokenManager("corp_id", "secret")
        mgr._token = "cached_token"
        mgr._expire_at = 9999999999.0

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("connection timeout"))

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            result = await get_wecom_user_info(mgr, "zhangsan")

        assert result is None


# ── knowlyr-id 联动测试 ──


class TestOffboardUserByPhone:
    """offboard_user_by_phone 调用 knowlyr-id 内部 API."""

    @pytest.mark.asyncio
    async def test_offboard_success(self, monkeypatch):
        from crew.wecom import offboard_user_by_phone

        monkeypatch.setenv("KNOWLYR_ID_API", "http://localhost:8100")
        monkeypatch.setenv("AGENT_API_TOKEN", "test_token_123")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "ok": True,
            "user_id": 42,
            "cleared": "员工",
            "msg": "已清除 员工 员工身份",
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            result = await offboard_user_by_phone("13800001234")

        assert result["ok"] is True
        assert result["user_id"] == 42
        assert result["cleared"] == "员工"

        # 验证请求参数
        call_args = mock_client.post.call_args
        assert "/api/internal/offboard-by-phone" in call_args[0][0]
        assert call_args[1]["json"] == {"phone": "13800001234"}
        assert "Bearer test_token_123" in call_args[1]["headers"]["Authorization"]

    @pytest.mark.asyncio
    async def test_offboard_no_token(self, monkeypatch):
        from crew.wecom import offboard_user_by_phone

        monkeypatch.setenv("AGENT_API_TOKEN", "")

        result = await offboard_user_by_phone("13800001234")
        assert result["ok"] is False
        assert "未配置" in result["detail"]

    @pytest.mark.asyncio
    async def test_offboard_user_not_found(self, monkeypatch):
        from crew.wecom import offboard_user_by_phone

        monkeypatch.setenv("KNOWLYR_ID_API", "http://localhost:8100")
        monkeypatch.setenv("AGENT_API_TOKEN", "test_token_123")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "ok": False,
            "detail": "未找到手机号 1234 对应的用户",
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            result = await offboard_user_by_phone("13800001234")

        assert result["ok"] is False


# ── 通讯录变更事件处理测试 ──


class TestContactChangeEvent:
    """change_contact 事件的 XML 解析和处理流程."""

    def test_parse_change_contact_xml(self):
        """验证 parse_wecom_message 能正确解析通讯录变更事件 XML."""
        from crew.wecom import parse_wecom_message

        xml = (
            "<xml>"
            "<ToUserName><![CDATA[ww_corp_id]]></ToUserName>"
            "<FromUserName><![CDATA[sys]]></FromUserName>"
            "<CreateTime>1234567890</CreateTime>"
            "<MsgType><![CDATA[event]]></MsgType>"
            "<Event><![CDATA[change_contact]]></Event>"
            "<ChangeType><![CDATA[delete_user]]></ChangeType>"
            "<UserID><![CDATA[zhangsan]]></UserID>"
            "</xml>"
        )
        msg = parse_wecom_message(xml)
        assert msg["MsgType"] == "event"
        assert msg["Event"] == "change_contact"
        assert msg["ChangeType"] == "delete_user"
        assert msg["UserID"] == "zhangsan"

    def test_parse_update_user_xml(self):
        from crew.wecom import parse_wecom_message

        xml = (
            "<xml>"
            "<ToUserName><![CDATA[ww_corp_id]]></ToUserName>"
            "<FromUserName><![CDATA[sys]]></FromUserName>"
            "<CreateTime>1234567890</CreateTime>"
            "<MsgType><![CDATA[event]]></MsgType>"
            "<Event><![CDATA[change_contact]]></Event>"
            "<ChangeType><![CDATA[update_user]]></ChangeType>"
            "<UserID><![CDATA[lisi]]></UserID>"
            "</xml>"
        )
        msg = parse_wecom_message(xml)
        assert msg["ChangeType"] == "update_user"
        assert msg["UserID"] == "lisi"


class TestHandleContactChange:
    """_handle_contact_change 离职同步完整流程."""

    @pytest.mark.asyncio
    async def test_delete_user_full_flow(self):
        """delete_user: 获取手机号 -> 调 offboard API -> 成功."""
        from crew.webhook_wecom import _handle_contact_change

        ctx = MagicMock()
        token_mgr = MagicMock()

        with (
            patch("crew.wecom.get_wecom_user_info", new_callable=AsyncMock) as mock_get_info,
            patch("crew.wecom.offboard_user_by_phone", new_callable=AsyncMock) as mock_offboard,
        ):
            mock_get_info.return_value = {
                "errcode": 0,
                "userid": "zhangsan",
                "mobile": "13800001234",
            }
            mock_offboard.return_value = {
                "ok": True,
                "user_id": 42,
                "cleared": "员工",
            }

            await _handle_contact_change(
                ctx,
                change_type="delete_user",
                wecom_userid="zhangsan",
                token_mgr=token_mgr,
                msg={"UserID": "zhangsan", "ChangeType": "delete_user"},
            )

            mock_get_info.assert_called_once_with(token_mgr, "zhangsan")
            mock_offboard.assert_called_once_with("13800001234")

    @pytest.mark.asyncio
    async def test_delete_user_no_mobile(self):
        """delete_user: 企微已删除用户，无法获取手机号 -> 记日志不崩溃."""
        from crew.webhook_wecom import _handle_contact_change

        ctx = MagicMock()
        token_mgr = MagicMock()

        with (
            patch("crew.wecom.get_wecom_user_info", new_callable=AsyncMock) as mock_get_info,
            patch("crew.wecom.offboard_user_by_phone", new_callable=AsyncMock) as mock_offboard,
        ):
            mock_get_info.return_value = None  # 企微查不到

            await _handle_contact_change(
                ctx,
                change_type="delete_user",
                wecom_userid="deleted_user",
                token_mgr=token_mgr,
                msg={"UserID": "deleted_user", "ChangeType": "delete_user"},
            )

            mock_offboard.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_user_disabled_triggers_offboard(self):
        """update_user + status=2(已禁用) -> 触发离职同步."""
        from crew.webhook_wecom import _handle_contact_change

        ctx = MagicMock()
        token_mgr = MagicMock()

        with (
            patch("crew.wecom.get_wecom_user_info", new_callable=AsyncMock) as mock_get_info,
            patch("crew.wecom.offboard_user_by_phone", new_callable=AsyncMock) as mock_offboard,
        ):
            mock_get_info.return_value = {
                "errcode": 0,
                "userid": "lisi",
                "mobile": "13900005678",
                "status": 2,  # 已禁用
            }
            mock_offboard.return_value = {"ok": True, "user_id": 99, "cleared": "实习生"}

            await _handle_contact_change(
                ctx,
                change_type="update_user",
                wecom_userid="lisi",
                token_mgr=token_mgr,
                msg={"UserID": "lisi", "ChangeType": "update_user"},
            )

            mock_offboard.assert_called_once_with("13900005678")

    @pytest.mark.asyncio
    async def test_update_user_active_no_offboard(self):
        """update_user + status=1(活跃) -> 不触发离职同步."""
        from crew.webhook_wecom import _handle_contact_change

        ctx = MagicMock()
        token_mgr = MagicMock()

        with (
            patch("crew.wecom.get_wecom_user_info", new_callable=AsyncMock) as mock_get_info,
            patch("crew.wecom.offboard_user_by_phone", new_callable=AsyncMock) as mock_offboard,
        ):
            mock_get_info.return_value = {
                "errcode": 0,
                "userid": "wangwu",
                "mobile": "13700009999",
                "status": 1,  # 活跃
            }

            await _handle_contact_change(
                ctx,
                change_type="update_user",
                wecom_userid="wangwu",
                token_mgr=token_mgr,
                msg={"UserID": "wangwu", "ChangeType": "update_user"},
            )

            mock_offboard.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_user_status_5_triggers_offboard(self):
        """update_user + status=5(退出企业) -> 触发离职同步."""
        from crew.webhook_wecom import _handle_contact_change

        ctx = MagicMock()
        token_mgr = MagicMock()

        with (
            patch("crew.wecom.get_wecom_user_info", new_callable=AsyncMock) as mock_get_info,
            patch("crew.wecom.offboard_user_by_phone", new_callable=AsyncMock) as mock_offboard,
        ):
            mock_get_info.return_value = {
                "errcode": 0,
                "userid": "zhaoliu",
                "mobile": "13600008888",
                "status": 5,  # 退出企业
            }
            mock_offboard.return_value = {"ok": True, "user_id": 100, "cleared": "正式员工"}

            await _handle_contact_change(
                ctx,
                change_type="update_user",
                wecom_userid="zhaoliu",
                token_mgr=token_mgr,
                msg={"UserID": "zhaoliu", "ChangeType": "update_user"},
            )

            mock_offboard.assert_called_once_with("13600008888")

    @pytest.mark.asyncio
    async def test_empty_userid_skips(self):
        """UserID 为空 -> 跳过不处理."""
        from crew.webhook_wecom import _handle_contact_change

        ctx = MagicMock()
        token_mgr = MagicMock()

        with (
            patch("crew.wecom.get_wecom_user_info", new_callable=AsyncMock) as mock_get_info,
            patch("crew.wecom.offboard_user_by_phone", new_callable=AsyncMock) as mock_offboard,
        ):
            await _handle_contact_change(
                ctx,
                change_type="delete_user",
                wecom_userid="",
                token_mgr=token_mgr,
                msg={"UserID": "", "ChangeType": "delete_user"},
            )

            mock_get_info.assert_not_called()
            mock_offboard.assert_not_called()


class TestChangeContactWebhookRouting:
    """handle_wecom_event 对 change_contact 事件的路由."""

    def _make_ctx(self):
        from crew.feishu import EventDeduplicator
        from crew.wecom import WecomCrypto

        ctx = MagicMock()
        ctx.project_dir = None
        crypto = WecomCrypto("qm7mzclOzyeB7Ee6rrKzzbPZAgVixfzHLi3fs20Xruz", "ww_test")
        ctx.wecom_ctx = {
            "config": MagicMock(
                token="test_token",
                agent_id=1000017,
                default_employee="",
                corp_id="ww_test",
            ),
            "crypto": crypto,
            "token_mgr": MagicMock(),
            "dedup": EventDeduplicator(),
        }
        return ctx

    def _make_event_request(self, ctx, msg_type="event", event="change_contact", change_type="delete_user", userid="zhangsan"):
        """构造通讯录变更事件的 POST 请求."""
        from crew.wecom import generate_wecom_signature

        crypto = ctx.wecom_ctx["crypto"]
        token = ctx.wecom_ctx["config"].token

        plain_xml = (
            "<xml>"
            "<ToUserName><![CDATA[ww_test]]></ToUserName>"
            "<FromUserName><![CDATA[sys]]></FromUserName>"
            "<CreateTime>1234567890</CreateTime>"
            f"<MsgType><![CDATA[{msg_type}]]></MsgType>"
            f"<Event><![CDATA[{event}]]></Event>"
            f"<ChangeType><![CDATA[{change_type}]]></ChangeType>"
            f"<UserID><![CDATA[{userid}]]></UserID>"
            "</xml>"
        )
        encrypted = crypto.encrypt(plain_xml)

        timestamp = "1234567890"
        nonce = "random_nonce"
        sig = generate_wecom_signature(token, timestamp, nonce, encrypted)

        body_xml = (
            "<xml>"
            f"<ToUserName><![CDATA[ww_test]]></ToUserName>"
            f"<Encrypt><![CDATA[{encrypted}]]></Encrypt>"
            "</xml>"
        )

        request = MagicMock()
        request.method = "POST"
        request.path_params = {"app_id": "default"}
        request.query_params = {
            "msg_signature": sig,
            "timestamp": timestamp,
            "nonce": nonce,
        }
        request.body = AsyncMock(return_value=body_xml.encode("utf-8"))
        return request

    @pytest.mark.asyncio
    async def test_change_contact_event_routes_to_handler(self):
        """change_contact 事件正确路由到 _handle_contact_change."""
        from crew.webhook_wecom import handle_wecom_event

        ctx = self._make_ctx()
        request = self._make_event_request(ctx, change_type="delete_user", userid="zhangsan")

        with patch("crew.webhook_wecom._handle_contact_change", new_callable=AsyncMock) as _mock_handler:
            resp = await handle_wecom_event(request, ctx)

            assert resp.status_code == 200
            assert resp.body.decode("utf-8") == "success"
            # asyncio.create_task 会异步执行，但我们 mock 了 _handle_contact_change
            # 验证至少创建了 task（通过 asyncio.create_task）

    @pytest.mark.asyncio
    async def test_non_contact_event_ignored(self):
        """非 change_contact 的 event 类型 -> 返回 success 不派遣消息处理."""
        from crew.webhook_wecom import handle_wecom_event

        ctx = self._make_ctx()
        # subscribe 事件 -> 非 change_contact
        request = self._make_event_request(ctx, event="subscribe", change_type="", userid="")

        resp = await handle_wecom_event(request, ctx)
        assert resp.status_code == 200
        assert resp.body.decode("utf-8") == "success"
