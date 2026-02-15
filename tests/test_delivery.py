"""测试投递层 — Webhook + 邮件 + 飞书."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crew.delivery import (
    DeliveryResult,
    DeliveryTarget,
    FEISHU_CARD_CONTENT_MAX,
    WEBHOOK_TIMEOUT,
    SMTP_TIMEOUT,
    deliver,
    _build_feishu_card,
    _deliver_email,
    _deliver_feishu,
    _deliver_webhook,
    _feishu_sign,
)


def _run(coro):
    return asyncio.run(coro)


def _mock_httpx_client(*, status_code=200):
    """创建 mock httpx.AsyncClient 上下文管理器."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# ── Webhook 投递 ──


class TestDeliverWebhook:
    """Webhook 投递."""

    def test_success(self):
        target = DeliveryTarget(type="webhook", url="https://hooks.example.com/test")
        mock_client = _mock_httpx_client(status_code=200)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = _run(_deliver_webhook(
                target, task_name="daily-review",
                task_result={"output": "all good"}, task_error=None,
            ))

        assert result.success is True
        assert "200" in result.detail

        # 验证 POST 参数
        call_args = mock_client.post.call_args
        assert call_args.kwargs["json"]["task_name"] == "daily-review"
        assert call_args.kwargs["json"]["status"] == "completed"
        assert call_args.kwargs["json"]["result"]["output"] == "all good"

    def test_failure_status(self):
        target = DeliveryTarget(type="webhook", url="https://hooks.example.com/test")
        mock_client = _mock_httpx_client(status_code=500)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = _run(_deliver_webhook(
                target, task_name="test",
                task_result=None, task_error=None,
            ))

        assert result.success is False
        assert "500" in result.detail

    def test_empty_url(self):
        target = DeliveryTarget(type="webhook", url="")
        result = _run(_deliver_webhook(target, task_name="t", task_result=None, task_error=None))
        assert result.success is False
        assert "URL" in result.detail

    def test_custom_headers(self):
        target = DeliveryTarget(
            type="webhook",
            url="https://hooks.example.com/test",
            headers={"X-Custom": "value"},
        )
        mock_client = _mock_httpx_client(status_code=200)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = _run(_deliver_webhook(
                target, task_name="t", task_result=None, task_error=None,
            ))

        assert result.success is True
        headers = mock_client.post.call_args.kwargs["headers"]
        assert headers["X-Custom"] == "value"
        assert headers["Content-Type"] == "application/json"

    def test_error_payload(self):
        target = DeliveryTarget(type="webhook", url="https://hooks.example.com/test")
        mock_client = _mock_httpx_client(status_code=200)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = _run(_deliver_webhook(
                target, task_name="fail-task",
                task_result=None, task_error="LLM 超时",
            ))

        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["status"] == "failed"
        assert payload["error"] == "LLM 超时"

    def test_connection_error(self):
        target = DeliveryTarget(type="webhook", url="https://hooks.example.com/test")
        mock_client = AsyncMock()
        mock_client.post.side_effect = ConnectionError("网络错误")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = _run(_deliver_webhook(
                target, task_name="t", task_result=None, task_error=None,
            ))

        assert result.success is False
        assert "网络错误" in result.detail


# ── 邮件投递 ──


class TestDeliverEmail:
    """邮件投递."""

    def test_success(self):
        target = DeliveryTarget(
            type="email",
            to="team@example.com",
            subject="Review: {name}",
        )

        env = {
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "bot@example.com",
            "SMTP_PASS": "secret",
        }

        with patch.dict(os.environ, env, clear=False), \
             patch("crew.delivery._send_smtp") as mock_send:
            result = _run(_deliver_email(
                target, task_name="daily-review",
                task_result={"output": "looks good"}, task_error=None,
            ))

        assert result.success is True
        assert "team@example.com" in result.detail

        # 验证邮件内容
        msg = mock_send.call_args[0][4]
        assert msg["Subject"] == "Review: daily-review"
        assert msg["To"] == "team@example.com"

    def test_no_smtp_host(self):
        target = DeliveryTarget(type="email", to="x@example.com")

        with patch.dict(os.environ, {"SMTP_HOST": ""}, clear=False):
            result = _run(_deliver_email(target, task_name="t", task_result=None, task_error=None))

        assert result.success is False
        assert "SMTP_HOST" in result.detail

    def test_empty_to(self):
        target = DeliveryTarget(type="email", to="")
        result = _run(_deliver_email(target, task_name="t", task_result=None, task_error=None))
        assert result.success is False
        assert "收件人" in result.detail

    def test_error_email(self):
        target = DeliveryTarget(type="email", to="team@example.com")

        env = {"SMTP_HOST": "smtp.example.com", "SMTP_PORT": "587", "SMTP_USER": "", "SMTP_PASS": ""}

        with patch.dict(os.environ, env, clear=False), \
             patch("crew.delivery._send_smtp") as mock_send:
            result = _run(_deliver_email(
                target, task_name="fail-task",
                task_result=None, task_error="Pipeline 中断",
            ))

        assert result.success is True
        msg = mock_send.call_args[0][4]
        assert "Pipeline 中断" in msg.get_payload(decode=True).decode("utf-8")


# ── deliver() 批量投递 ──


class TestDeliver:
    """批量投递."""

    def test_empty_targets(self):
        results = _run(deliver([], task_name="t"))
        assert results == []

    def test_mixed_targets(self):
        targets = [
            DeliveryTarget(type="webhook", url="https://example.com/hook"),
            DeliveryTarget(type="email", to="team@example.com"),
        ]

        mock_client = _mock_httpx_client(status_code=200)
        env = {"SMTP_HOST": "smtp.example.com", "SMTP_PORT": "587", "SMTP_USER": "", "SMTP_PASS": ""}

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch.dict(os.environ, env, clear=False), \
             patch("crew.delivery._send_smtp"):
            results = _run(deliver(
                targets, task_name="mixed-test",
                task_result={"output": "ok"}, task_error=None,
            ))

        assert len(results) == 2
        assert results[0].success is True  # webhook
        assert results[1].success is True  # email

    def test_exception_handling(self):
        """单个目标异常不影响其他目标."""
        targets = [
            DeliveryTarget(type="webhook", url="https://fail.example.com"),
            DeliveryTarget(type="email", to=""),  # 空收件人 → 失败
        ]

        mock_client = _mock_httpx_client(status_code=200)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results = _run(deliver(
                targets, task_name="error-test",
                task_result=None, task_error=None,
            ))

        assert len(results) == 2
        # webhook 成功，email 失败（空收件人）
        assert results[0].success is True
        assert results[1].success is False


# ── DeliveryTarget 模型 ──


class TestDeliveryTargetModel:
    """投递目标模型."""

    def test_webhook(self):
        t = DeliveryTarget(type="webhook", url="https://example.com", headers={"X-Key": "abc"})
        assert t.type == "webhook"
        assert t.url == "https://example.com"
        assert t.headers == {"X-Key": "abc"}

    def test_email(self):
        t = DeliveryTarget(type="email", to="a@b.com", subject="Report: {name}")
        assert t.type == "email"
        assert t.to == "a@b.com"
        assert t.subject == "Report: {name}"

    def test_defaults(self):
        t = DeliveryTarget(type="webhook")
        assert t.url == ""
        assert t.headers == {}
        assert t.to == ""
        assert t.subject == ""


class TestSmtpValidation:
    """SMTP 配置校验."""

    def test_invalid_smtp_port(self):
        """非数字端口返回错误."""
        from crew.delivery import _validate_smtp_config
        ok, err = _validate_smtp_config("smtp.example.com", "abc", "user@example.com")
        assert ok is False
        assert "无效" in err

    def test_smtp_port_out_of_range(self):
        """端口 >65535 返回错误."""
        from crew.delivery import _validate_smtp_config
        ok, err = _validate_smtp_config("smtp.example.com", "99999", "user@example.com")
        assert ok is False
        assert "超出范围" in err

    def test_invalid_email_format(self):
        """无 @ 返回错误."""
        from crew.delivery import _validate_smtp_config
        ok, err = _validate_smtp_config("smtp.example.com", "587", "no-at-sign")
        assert ok is False
        assert "@" in err

    def test_valid_config(self):
        """合法配置返回 ok."""
        from crew.delivery import _validate_smtp_config
        ok, err = _validate_smtp_config("smtp.example.com", "587", "user@example.com")
        assert ok is True
        assert err == ""

    def test_smtp_timeout(self):
        """连接超时返回错误."""
        target = DeliveryTarget(type="email", to="team@example.com")
        env = {
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "bot@example.com",
            "SMTP_PASS": "secret",
        }

        with patch.dict(os.environ, env, clear=False), \
             patch("crew.delivery._send_smtp", side_effect=TimeoutError("连接超时")):
            result = _run(_deliver_email(
                target, task_name="test",
                task_result=None, task_error=None,
            ))

        assert result.success is False
        assert "超时" in result.detail


class TestTimeoutConstants:
    """超时常量定义."""

    def test_webhook_timeout(self):
        assert WEBHOOK_TIMEOUT == 30.0

    def test_smtp_timeout(self):
        assert SMTP_TIMEOUT == 10

    def test_email_body_max_length(self):
        from crew.delivery import EMAIL_BODY_MAX_LENGTH
        assert EMAIL_BODY_MAX_LENGTH == 2000

    def test_feishu_card_content_max(self):
        assert FEISHU_CARD_CONTENT_MAX == 4000


# ── 飞书签名 ──


class TestFeishuSign:
    """飞书机器人签名."""

    def test_sign_not_empty(self):
        result = _feishu_sign("test-secret", 1700000000)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_sign_deterministic(self):
        a = _feishu_sign("secret-key", 1700000000)
        b = _feishu_sign("secret-key", 1700000000)
        assert a == b

    def test_sign_changes_with_secret(self):
        a = _feishu_sign("secret-1", 1700000000)
        b = _feishu_sign("secret-2", 1700000000)
        assert a != b

    def test_sign_changes_with_timestamp(self):
        a = _feishu_sign("secret", 1700000000)
        b = _feishu_sign("secret", 1700000001)
        assert a != b


# ── 飞书卡片构建 ──


class TestBuildFeishuCard:
    """飞书交互式卡片消息构建."""

    def test_success_card(self):
        card = _build_feishu_card("daily-review", {"output": "一切正常"}, None)
        assert card["msg_type"] == "interactive"
        header = card["card"]["header"]
        assert header["template"] == "green"
        assert "daily-review" in header["title"]["content"]
        assert "✅" in header["title"]["content"]

        elements = card["card"]["elements"]
        assert any(e["tag"] == "markdown" and "一切正常" in e["content"] for e in elements)

    def test_error_card(self):
        card = _build_feishu_card("fail-task", None, "LLM 超时")
        header = card["card"]["header"]
        assert header["template"] == "red"
        assert "❌" in header["title"]["content"]
        assert "fail-task" in header["title"]["content"]

        elements = card["card"]["elements"]
        assert any(e["tag"] == "markdown" and "LLM 超时" in e["content"] for e in elements)

    def test_no_result_card(self):
        card = _build_feishu_card("some-task", None, None)
        elements = card["card"]["elements"]
        assert any(e["tag"] == "markdown" and "任务已完成" in e["content"] for e in elements)

    def test_content_truncation(self):
        long_output = "x" * (FEISHU_CARD_CONTENT_MAX + 500)
        card = _build_feishu_card("task", {"output": long_output}, None)
        elements = card["card"]["elements"]
        md = [e for e in elements if e["tag"] == "markdown"][0]
        assert len(md["content"]) <= FEISHU_CARD_CONTENT_MAX + 20  # +截断提示
        assert "已截断" in md["content"]

    def test_metadata_notes(self):
        result = {
            "output": "done",
            "employee": "code-reviewer",
            "model": "claude-sonnet",
            "duration_ms": 3500,
        }
        card = _build_feishu_card("task", result, None)
        elements = card["card"]["elements"]
        note = [e for e in elements if e["tag"] == "note"]
        assert len(note) == 1
        note_text = note[0]["elements"][0]["content"]
        assert "code-reviewer" in note_text
        assert "claude-sonnet" in note_text
        assert "3.5s" in note_text

    def test_non_string_output(self):
        result = {"output": {"key": "value"}}
        card = _build_feishu_card("task", result, None)
        elements = card["card"]["elements"]
        md = [e for e in elements if e["tag"] == "markdown"][0]
        assert "key" in md["content"]

    def test_empty_task_name(self):
        card = _build_feishu_card("", None, None)
        header = card["card"]["header"]
        assert "任务" in header["title"]["content"]


# ── 飞书投递 ──


def _mock_feishu_client(*, status_code=200, body=None):
    """创建 mock feishu httpx 客户端."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = body if body is not None else {"code": 0, "msg": "success"}

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


class TestDeliverFeishu:
    """飞书投递."""

    def test_success(self):
        target = DeliveryTarget(type="feishu", url="https://open.feishu.cn/open-apis/bot/v2/hook/xxx")
        mock_client = _mock_feishu_client()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = _run(_deliver_feishu(
                target, task_name="daily-review",
                task_result={"output": "all good"}, task_error=None,
            ))

        assert result.success is True
        assert result.target_type == "feishu"

        # 验证 POST payload 包含卡片
        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["msg_type"] == "interactive"
        assert "card" in payload

    def test_empty_url(self):
        target = DeliveryTarget(type="feishu", url="")
        result = _run(_deliver_feishu(target, task_name="t", task_result=None, task_error=None))
        assert result.success is False
        assert "URL" in result.detail

    def test_with_secret(self):
        target = DeliveryTarget(
            type="feishu",
            url="https://open.feishu.cn/open-apis/bot/v2/hook/xxx",
            secret="my-secret",
        )
        mock_client = _mock_feishu_client()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = _run(_deliver_feishu(
                target, task_name="task",
                task_result=None, task_error=None,
            ))

        assert result.success is True
        payload = mock_client.post.call_args.kwargs["json"]
        assert "timestamp" in payload
        assert "sign" in payload

    def test_without_secret_no_sign(self):
        target = DeliveryTarget(
            type="feishu",
            url="https://open.feishu.cn/open-apis/bot/v2/hook/xxx",
        )
        mock_client = _mock_feishu_client()

        with patch("httpx.AsyncClient", return_value=mock_client):
            _run(_deliver_feishu(
                target, task_name="task",
                task_result=None, task_error=None,
            ))

        payload = mock_client.post.call_args.kwargs["json"]
        assert "timestamp" not in payload
        assert "sign" not in payload

    def test_api_error_code(self):
        target = DeliveryTarget(type="feishu", url="https://open.feishu.cn/open-apis/bot/v2/hook/xxx")
        mock_client = _mock_feishu_client(body={"code": 19001, "msg": "sign match fail"})

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = _run(_deliver_feishu(
                target, task_name="task",
                task_result=None, task_error=None,
            ))

        assert result.success is False
        assert "sign match fail" in result.detail

    def test_http_error(self):
        target = DeliveryTarget(type="feishu", url="https://open.feishu.cn/open-apis/bot/v2/hook/xxx")
        mock_client = _mock_feishu_client(status_code=500, body={})

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = _run(_deliver_feishu(
                target, task_name="task",
                task_result=None, task_error=None,
            ))

        assert result.success is False

    def test_connection_error(self):
        target = DeliveryTarget(type="feishu", url="https://open.feishu.cn/open-apis/bot/v2/hook/xxx")
        mock_client = AsyncMock()
        mock_client.post.side_effect = ConnectionError("网络不可达")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = _run(_deliver_feishu(
                target, task_name="task",
                task_result=None, task_error=None,
            ))

        assert result.success is False
        assert "网络不可达" in result.detail


# ── deliver() 批量含飞书 ──


class TestDeliverWithFeishu:
    """批量投递含飞书目标."""

    def test_feishu_in_batch(self):
        targets = [
            DeliveryTarget(type="webhook", url="https://example.com/hook"),
            DeliveryTarget(type="feishu", url="https://open.feishu.cn/open-apis/bot/v2/hook/xxx"),
        ]

        mock_webhook = _mock_httpx_client(status_code=200)
        mock_feishu_resp = MagicMock()
        mock_feishu_resp.status_code = 200
        mock_feishu_resp.json.return_value = {"code": 0, "msg": "success"}
        mock_webhook.post.return_value = mock_feishu_resp

        with patch("httpx.AsyncClient", return_value=mock_webhook):
            results = _run(deliver(
                targets, task_name="batch",
                task_result={"output": "ok"}, task_error=None,
            ))

        assert len(results) == 2
        assert all(r.success for r in results)


# ── DeliveryTarget 飞书字段 ──


class TestDeliveryTargetFeishu:
    """飞书投递目标模型字段."""

    def test_feishu_target(self):
        t = DeliveryTarget(
            type="feishu",
            url="https://open.feishu.cn/open-apis/bot/v2/hook/xxx",
            secret="my-secret",
        )
        assert t.type == "feishu"
        assert t.secret == "my-secret"

    def test_secret_default_empty(self):
        t = DeliveryTarget(type="feishu", url="https://example.com")
        assert t.secret == ""
