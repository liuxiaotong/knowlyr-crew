"""企业微信事件处理 -- 接收企微回调、消息路由、员工执行、回复."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# 后台任务引用集合 -- 防止 GC 提前回收 + 异常日志
_background_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]


def _task_done_callback(task: asyncio.Task) -> None:  # type: ignore[type-arg]
    """后台 task 完成回调：记录异常日志 + 从引用集合移除."""
    _background_tasks.discard(task)
    if not task.cancelled():
        exc = task.exception()
        if exc:
            logger.error("企微后台任务异常: %s", exc, exc_info=exc)


async def handle_wecom_event(request: Any, ctx: Any) -> Any:
    """处理企微事件回调 -- GET 验证 URL + POST 接收消息.

    GET /wecom/event/{app_id}?msg_signature=...&timestamp=...&nonce=...&echostr=...
    POST /wecom/event/{app_id} (XML body with encrypted message)
    """
    from starlette.responses import PlainTextResponse, Response

    wecom_ctx = ctx.wecom_ctx
    if wecom_ctx is None:
        return PlainTextResponse("wecom not configured", status_code=501)

    config = wecom_ctx["config"]
    crypto = wecom_ctx["crypto"]
    token_mgr = wecom_ctx["token_mgr"]
    dedup = wecom_ctx["dedup"]

    # ── GET: URL 验证 ──
    if request.method == "GET":
        msg_signature = request.query_params.get("msg_signature", "")
        timestamp = request.query_params.get("timestamp", "")
        nonce = request.query_params.get("nonce", "")
        echostr = request.query_params.get("echostr", "")

        from crew.wecom import verify_wecom_signature

        if not verify_wecom_signature(config.token, timestamp, nonce, echostr, msg_signature):
            return PlainTextResponse("signature mismatch", status_code=403)

        # 解密 echostr 返回明文
        try:
            plain_echostr = crypto.decrypt(echostr)
        except Exception as exc:
            logger.error("企微 echostr 解密失败: %s", exc)
            return PlainTextResponse("decrypt error", status_code=400)

        return PlainTextResponse(plain_echostr)

    # ── POST: 消息接收 ──
    msg_signature = request.query_params.get("msg_signature", "")
    timestamp = request.query_params.get("timestamp", "")
    nonce = request.query_params.get("nonce", "")

    body = await request.body()
    xml_text = body.decode("utf-8")

    # 从加密 XML 中提取 Encrypt 字段
    from crew.wecom import parse_wecom_encrypt_xml, verify_wecom_signature

    encrypt_content, _to_user = parse_wecom_encrypt_xml(xml_text)
    if not encrypt_content:
        return PlainTextResponse("missing Encrypt", status_code=400)

    # 验证签名
    if not verify_wecom_signature(config.token, timestamp, nonce, encrypt_content, msg_signature):
        return PlainTextResponse("signature mismatch", status_code=403)

    # 解密消息
    try:
        decrypted_xml = crypto.decrypt(encrypt_content)
    except Exception as exc:
        logger.error("企微消息解密失败: %s", exc)
        return PlainTextResponse("decrypt error", status_code=400)

    # 解析明文 XML
    from crew.wecom import parse_wecom_message

    msg = parse_wecom_message(decrypted_xml)
    msg_type = msg.get("MsgType", "")
    msg_id = msg.get("MsgId", "")
    from_user = msg.get("FromUserName", "")
    content = msg.get("Content", "").strip()

    logger.info(
        "企微消息: type=%s from=%s content=%s",
        msg_type,
        from_user,
        content[:50] if content else "(empty)",
    )

    # 初期只处理文本消息
    if msg_type != "text":
        logger.info("企微消息类型不支持: %s", msg_type)
        return PlainTextResponse("success")

    if not content:
        return PlainTextResponse("success")

    # 去重
    if msg_id and dedup.is_duplicate(msg_id):
        logger.warning("企微消息去重: %s", msg_id)
        return PlainTextResponse("success")

    # 后台处理（企微要求快速返回 "success"）
    task = asyncio.create_task(
        _wecom_dispatch(ctx, from_user, content, config.agent_id, token_mgr)
    )
    _background_tasks.add(task)
    task.add_done_callback(_task_done_callback)

    return PlainTextResponse("success")


async def _wecom_dispatch(
    ctx: Any,
    from_user: str,
    text: str,
    agent_id: int,
    token_mgr: Any,
) -> None:
    """后台处理企微消息：路由到员工 -> 执行 -> 主动推送回复."""
    from crew.discovery import discover_employees
    from crew.feishu import resolve_employee_from_mention
    from crew.wecom import send_wecom_text

    try:
        discovery = discover_employees(project_dir=ctx.project_dir)

        # 企微单聊：用 default_employee
        wecom_config = ctx.wecom_ctx["config"]
        default_emp = wecom_config.default_employee

        # 企微没有 @mention 机制，用文本前缀匹配 + default
        employee_name, task_text = resolve_employee_from_mention(
            [],  # 无 mentions
            text,
            discovery,
            default_employee=default_emp,
        )

        if employee_name is None:
            await send_wecom_text(
                token_mgr,
                from_user,
                agent_id,
                "未能识别目标员工，请用员工名开头发送消息。",
            )
            return

        emp = discovery.get(employee_name)

        # ── SG Bridge 主通道尝试 ──
        _sg_reply: str | None = None
        try:
            from crew.sg_bridge import SGBridgeError, sg_dispatch

            _sg_reply = await sg_dispatch(
                task_text,
                project_dir=ctx.project_dir,
                employee_name=employee_name,
                chat_context="这是企业微信实时聊天。像平时一样自然回复。",
                message_history=None,
                permission_callback=None,
            )
        except Exception as _sg_exc:
            logger.info("SG Bridge fallback: %s -> 走 crew 引擎", _sg_exc)
            _sg_reply = None

        if _sg_reply is not None:
            output_text = _sg_reply
        else:
            # Fallback: crew engine
            from crew.engine import CrewEngine

            engine = CrewEngine(project_dir=ctx.project_dir)
            result = await engine.chat(
                employee_id=employee_name,
                message=task_text,
                channel="wecom",
                sender_id=from_user,
                max_visibility="private",
                message_history=None,
            )
            output_text = result.get("reply") or result.get("output", "")

        # 清洗内部标签
        from crew.output_sanitizer import strip_internal_tags

        output_text = strip_internal_tags(output_text)

        # 发送回复
        await send_wecom_text(token_mgr, from_user, agent_id, output_text)

        # 记录任务
        record = ctx.registry.create(
            trigger="wecom",
            target_type="employee",
            target_name=employee_name,
            args={"task": task_text},
        )
        ctx.registry.update(record.task_id, "completed", result={"output": output_text})

    except Exception as e:
        logger.exception("企微消息处理失败: %s", e)
        try:
            await send_wecom_text(
                token_mgr,
                from_user,
                agent_id,
                "处理时出了点问题，请稍后再试。",
            )
        except Exception:
            logger.warning("企微错误回复发送失败", exc_info=True)
