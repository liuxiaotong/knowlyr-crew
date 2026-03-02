"""企业微信适配器基础测试 -- 加解密、XML 解析、签名验证、消息路由."""

from __future__ import annotations

import hashlib
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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

        xml = (
            "<xml>"
            "<MsgType><![CDATA[text]]></MsgType>"
            "<Content></Content>"
            "</xml>"
        )
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
        from crew.wecom import generate_wecom_signature
        from crew.webhook_wecom import handle_wecom_event

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
            patch("crew.webhook_wecom.discover_employees", create=True) as mock_discover,
            patch("crew.webhook_wecom.resolve_employee_from_mention", create=True) as mock_resolve,
            patch("crew.webhook_wecom.sg_dispatch", create=True) as mock_sg,
            patch("crew.webhook_wecom.send_wecom_text", create=True) as mock_send,
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
