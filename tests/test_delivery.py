"""测试投递层 — Webhook + 邮件."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crew.delivery import (
    DeliveryResult,
    DeliveryTarget,
    WEBHOOK_TIMEOUT,
    SMTP_TIMEOUT,
    deliver,
    _deliver_email,
    _deliver_webhook,
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
