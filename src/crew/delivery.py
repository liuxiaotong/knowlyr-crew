"""投递层 — 将任务结果推送到外部渠道（Webhook / 邮件）."""

from __future__ import annotations

import asyncio
import logging
import smtplib
import os
from email.mime.text import MIMEText
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DeliveryTarget(BaseModel):
    """单个投递目标."""

    type: Literal["webhook", "email"] = Field(description="投递类型")
    # webhook
    url: str = Field(default="", description="Webhook URL")
    headers: dict[str, str] = Field(default_factory=dict, description="自定义请求头")
    # email
    to: str = Field(default="", description="收件人邮箱")
    subject: str = Field(default="", description="邮件主题（支持 {name} 占位符）")


class DeliveryResult(BaseModel):
    """投递结果."""

    target_type: str
    success: bool
    detail: str = ""


async def deliver(
    targets: list[DeliveryTarget],
    *,
    task_name: str = "",
    task_result: dict[str, Any] | None = None,
    task_error: str | None = None,
) -> list[DeliveryResult]:
    """投递任务结果到所有目标.

    Args:
        targets: 投递目标列表.
        task_name: 任务名称（用于模板替换）.
        task_result: 任务结果数据.
        task_error: 任务错误信息（有值表示任务失败）.

    Returns:
        每个目标的投递结果.
    """
    if not targets:
        return []

    results = await asyncio.gather(
        *[_deliver_one(t, task_name=task_name, task_result=task_result, task_error=task_error) for t in targets],
        return_exceptions=True,
    )

    out: list[DeliveryResult] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            out.append(DeliveryResult(
                target_type=targets[i].type,
                success=False,
                detail=str(r),
            ))
        else:
            out.append(r)
    return out


async def _deliver_one(
    target: DeliveryTarget,
    *,
    task_name: str,
    task_result: dict[str, Any] | None,
    task_error: str | None,
) -> DeliveryResult:
    """投递到单个目标."""
    if target.type == "webhook":
        return await _deliver_webhook(target, task_name=task_name, task_result=task_result, task_error=task_error)
    elif target.type == "email":
        return await _deliver_email(target, task_name=task_name, task_result=task_result, task_error=task_error)
    else:
        return DeliveryResult(target_type=target.type, success=False, detail=f"未知投递类型: {target.type}")


async def _deliver_webhook(
    target: DeliveryTarget,
    *,
    task_name: str,
    task_result: dict[str, Any] | None,
    task_error: str | None,
) -> DeliveryResult:
    """POST JSON 到 webhook URL."""
    if not target.url:
        return DeliveryResult(target_type="webhook", success=False, detail="URL 为空")

    try:
        import httpx
    except ImportError:
        return DeliveryResult(target_type="webhook", success=False, detail="httpx 未安装")

    payload: dict[str, Any] = {
        "task_name": task_name,
        "status": "failed" if task_error else "completed",
    }
    if task_error:
        payload["error"] = task_error
    if task_result:
        payload["result"] = task_result

    headers = {"Content-Type": "application/json"}
    headers.update(target.headers)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(target.url, json=payload, headers=headers)
        if resp.status_code < 400:
            return DeliveryResult(target_type="webhook", success=True, detail=f"HTTP {resp.status_code}")
        else:
            return DeliveryResult(target_type="webhook", success=False, detail=f"HTTP {resp.status_code}")
    except Exception as e:
        return DeliveryResult(target_type="webhook", success=False, detail=str(e))


async def _deliver_email(
    target: DeliveryTarget,
    *,
    task_name: str,
    task_result: dict[str, Any] | None,
    task_error: str | None,
) -> DeliveryResult:
    """通过 SMTP 发送邮件."""
    if not target.to:
        return DeliveryResult(target_type="email", success=False, detail="收件人为空")

    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    smtp_from = os.environ.get("SMTP_FROM", smtp_user)

    if not smtp_host:
        return DeliveryResult(target_type="email", success=False, detail="SMTP_HOST 未配置")

    subject = target.subject.format(name=task_name) if target.subject else f"任务完成: {task_name}"

    # 构建邮件正文
    if task_error:
        body = f"任务 {task_name} 执行失败。\n\n错误: {task_error}"
    elif task_result:
        # 提取关键信息
        output = task_result.get("output", "")
        if isinstance(output, str) and len(output) > 2000:
            output = output[:2000] + "\n...(已截断)"
        body = f"任务 {task_name} 执行完成。\n\n结果:\n{output}"
    else:
        body = f"任务 {task_name} 执行完成。"

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = target.to

    try:
        # 在线程池中执行阻塞的 SMTP 操作
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _send_smtp, smtp_host, smtp_port, smtp_user, smtp_pass, msg)
        return DeliveryResult(target_type="email", success=True, detail=f"发送至 {target.to}")
    except Exception as e:
        return DeliveryResult(target_type="email", success=False, detail=str(e))


def _send_smtp(host: str, port: int, user: str, password: str, msg: MIMEText) -> None:
    """同步 SMTP 发送."""
    with smtplib.SMTP(host, port) as server:
        server.starttls()
        if user and password:
            server.login(user, password)
        server.send_message(msg)
