"""飞书事件处理 — 接收飞书回调、消息路由、员工执行、回复."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

from crew.webhook_context import _AppContext


async def _handle_feishu_event(request: Any, ctx: _AppContext) -> Any:
    """处理飞书事件回调.

    支持:
    1. URL verification challenge
    2. im.message.receive_v1 消息事件
    """
    from starlette.responses import JSONResponse

    payload = await request.json()

    # 1. URL 验证（飞书注册回调地址时发送）
    if payload.get("type") == "url_verification":
        challenge = payload.get("challenge", "")
        return JSONResponse({"challenge": challenge})

    # 2. 检查飞书是否配置
    if ctx.feishu_config is None or ctx.feishu_token_mgr is None:
        return JSONResponse({"error": "飞书 Bot 未配置"}, status_code=501)

    # 3. 验证 event token（未配置时跳过但记录警告）
    header = payload.get("header", {})
    event_token = header.get("token", payload.get("token", ""))
    if ctx.feishu_config.verification_token:
        from crew.feishu import verify_feishu_event

        if not verify_feishu_event(ctx.feishu_config.verification_token, event_token):
            return JSONResponse({"error": "invalid token"}, status_code=401)
    else:
        logger.warning("飞书 verification_token 未配置，跳过事件验证（建议设置 FEISHU_VERIFICATION_TOKEN）")

    # 4. 只处理消息事件
    event_type = header.get("event_type", "")
    if event_type != "im.message.receive_v1":
        logger.warning("飞书事件忽略: event_type=%s", event_type)
        return JSONResponse({"message": "ignored", "event_type": event_type})

    # 5. 解析消息
    from crew.feishu import parse_message_event

    msg_event = parse_message_event(payload)
    if msg_event is None:
        msg_type = payload.get("event", {}).get("message", {}).get("message_type", "?")
        logger.warning("飞书消息类型不支持: msg_type=%s", msg_type)
        return JSONResponse({"message": "unsupported message type"})

    logger.warning(
        "飞书消息: type=%s chat=%s text=%s image_key=%s mentions=%d",
        msg_event.msg_type, msg_event.chat_type,
        msg_event.text[:50] if msg_event.text else "(empty)",
        msg_event.image_key or "-",
        len(msg_event.mentions),
    )

    # 6. 去重
    if ctx.feishu_dedup and ctx.feishu_dedup.is_duplicate(msg_event.message_id):
        logger.warning("飞书消息去重: %s", msg_event.message_id)
        return JSONResponse({"message": "duplicate"})

    # 7. 后台处理（飞书要求 3s 内响应）
    # 通过 webhook 模块查找，确保 mock patch 生效
    import crew.webhook as _wh
    asyncio.create_task(_wh._feishu_dispatch(ctx, msg_event))

    return JSONResponse({"message": "ok"})


async def _find_recent_image_in_chat(
    token_mgr: Any,
    chat_id: str,
    sender_id: str,
    max_messages: int = 5,
) -> tuple[str, str] | None:
    """在群聊历史中查找同一发送者最近发的图片，返回 (image_key, message_id)."""
    import httpx
    import json as _json

    token = await token_mgr.get_token()
    url = "https://open.feishu.cn/open-apis/im/v1/messages"
    params = {
        "container_id_type": "chat",
        "container_id": chat_id,
        "page_size": max_messages,
        "sort_type": "ByCreateTimeDesc",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            params=params,
        )
        data = resp.json()

    if data.get("code", -1) != 0:
        logger.warning("查群消息失败: %s", data.get("msg", ""))
        return None

    for item in data.get("data", {}).get("items", []):
        if item.get("msg_type") != "image":
            continue
        msg_sender = item.get("sender", {}).get("id", "")
        if msg_sender != sender_id:
            continue
        try:
            content = _json.loads(item.get("body", {}).get("content", "{}"))
        except _json.JSONDecodeError:
            continue
        img_key = content.get("image_key", "")
        msg_id = item.get("message_id", "")
        if img_key and msg_id:
            return img_key, msg_id

    return None


async def _feishu_dispatch(ctx: _AppContext, msg_event: Any) -> None:
    """后台处理飞书消息：路由到员工 → 执行 → 回复卡片."""
    from crew.discovery import discover_employees
    from crew.feishu import (
        download_feishu_image,
        resolve_employee_from_mention,
        send_feishu_card,
        send_feishu_reply,
        send_feishu_text,
    )

    assert ctx.feishu_config is not None
    assert ctx.feishu_token_mgr is not None

    try:
        discovery = discover_employees(project_dir=ctx.project_dir)

        employee_name, task_text = resolve_employee_from_mention(
            msg_event.mentions,
            msg_event.text,
            discovery,
            default_employee=ctx.feishu_config.default_employee,
        )

        if employee_name is None:
            await send_feishu_text(
                ctx.feishu_token_mgr,
                msg_event.chat_id,
                "未能识别目标员工，请 @员工名 + 任务描述。",
            )
            return

        emp = discovery.get(employee_name)

        # 图片 → 下载图片 + vision
        image_key = msg_event.image_key
        image_message_id = msg_event.message_id

        if (
            not image_key
            and msg_event.chat_type == "group"
            and msg_event.msg_type == "text"
        ):
            try:
                found = await _find_recent_image_in_chat(
                    ctx.feishu_token_mgr,
                    msg_event.chat_id,
                    msg_event.sender_id,
                )
                if found:
                    image_key, image_message_id = found
            except Exception as exc:
                logger.warning("查找群聊近期图片失败: %s", exc)

        image_data: tuple[bytes, str] | None = None
        if image_key:
            try:
                image_data = await download_feishu_image(
                    ctx.feishu_token_mgr, image_key,
                    message_id=image_message_id,
                )
            except Exception as exc:
                logger.warning("飞书图片下载失败: %s", exc)
                await send_feishu_text(
                    ctx.feishu_token_mgr,
                    msg_event.chat_id,
                    "图片没下载下来，你描述一下？",
                )
                return
            if not task_text:
                task_text = "Kai 发了一张图片，请看图片内容并回应。"

        # 执行员工 — 飞书实时聊天
        from crew.tool_schema import AGENT_TOOLS

        has_tools = any(
            t in AGENT_TOOLS
            for t in (emp.tools or [])
        ) if emp else False

        if has_tools:
            chat_context = (
                "你正在飞书上和 Kai 聊天。像平时一样自然回复。"
                "需要数据就调工具查，拿到数据用自己的话说，别搬 JSON。"
                "如果 Kai 在追问上一个话题，直接接着聊，不用重新查。"
                "只陈述工具返回的事实，没查过的事不要说，不要主动编造任何信息。"
            )
        else:
            chat_context = (
                "这是飞书实时聊天。你现在没有任何业务数据。"
                "你不知道：今天的会议安排、项目进度、客户状态、合同情况、同事的工作产出、任何具体数字。"
                "Kai 问这些就说「这个我得看看今天的数据才能说」。"
                "你可以聊：你的职能介绍、帮他想问题、帮他起草文字。"
            )

        # 加载对话历史
        message_history = None
        if ctx.feishu_chat_store:
            history_entries = ctx.feishu_chat_store.get_recent(
                msg_event.chat_id, limit=40,
            )
            if history_entries:
                message_history = [
                    {"role": e["role"], "content": e["content"]}
                    for e in history_entries
                ]

        if not has_tools and message_history:
            history_text = ctx.feishu_chat_store.format_for_prompt(
                msg_event.chat_id, limit=40,
            )
            if history_text:
                chat_context += (
                    "\n\n## 最近对话记录\n\n"
                    "以下是你和 Kai 最近的对话，用来保持上下文连贯。\n\n"
                    + history_text
                )
            message_history = None

        args = {"task": chat_context}
        if emp and emp.args:
            first_required = next((a for a in emp.args if a.required), None)
            if first_required:
                args[first_required.name] = chat_context

        # 构建 user_message（支持图片多模态）
        user_msg: str | list[dict[str, Any]] = task_text
        if image_data is not None:
            import base64 as _b64
            img_bytes, media_type = image_data
            b64 = _b64.b64encode(img_bytes).decode()
            user_msg = [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                {"type": "text", "text": task_text},
            ]

        # 通过 webhook 模块查找，确保 mock patch 生效
        import crew.webhook as _wh
        result = await _wh._execute_employee(
            ctx, employee_name, args, model=None,
            user_message=user_msg,
            message_history=message_history,
        )

        # 记录对话历史
        output_text = result.get("output", "") if isinstance(result, dict) else str(result)
        if ctx.feishu_chat_store:
            ctx.feishu_chat_store.append(msg_event.chat_id, "user", task_text)
            ctx.feishu_chat_store.append(msg_event.chat_id, "assistant", output_text)

        # 沉淀长期记忆
        if ctx.project_dir:
            try:
                from crew.feishu_memory import capture_feishu_memory

                capture_feishu_memory(
                    project_dir=ctx.project_dir,
                    employee_name=employee_name,
                    chat_id=msg_event.chat_id,
                    user_text=task_text,
                    assistant_text=output_text,
                )
            except Exception as e:
                logger.debug("飞书记忆沉淀失败: %s", e)

        # 发送回复
        if msg_event.chat_type == "group":
            await send_feishu_reply(
                ctx.feishu_token_mgr,
                msg_event.message_id,
                output_text,
            )
        else:
            await send_feishu_text(
                ctx.feishu_token_mgr,
                msg_event.chat_id,
                output_text,
            )

        # 记录任务
        record = ctx.registry.create(
            trigger="feishu",
            target_type="employee",
            target_name=employee_name,
            args=args,
        )
        ctx.registry.update(record.task_id, "completed", result=result)

    except Exception as e:
        logger.exception("飞书消息处理失败: %s", e)
        try:
            error_text = "处理时出了点问题，请稍后再试。"
            if msg_event.chat_type == "group":
                await send_feishu_reply(
                    ctx.feishu_token_mgr,
                    msg_event.message_id,
                    error_text,
                )
            else:
                await send_feishu_text(
                    ctx.feishu_token_mgr,
                    msg_event.chat_id,
                    error_text,
                )
        except Exception:
            logger.warning("飞书错误回复发送失败", exc_info=True)
