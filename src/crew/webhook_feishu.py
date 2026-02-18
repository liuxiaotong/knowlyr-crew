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


# ── 闲聊快速路径 ──

# 工作关键词 — 命中任一则走完整 agent loop（带工具）
_WORK_KEYWORDS = frozenset([
    "数据", "报表", "分析", "查一下", "查下", "帮我查", "统计",
    "日程", "日历", "会议", "审批", "待办", "任务",
    "委派", "安排", "创建", "删除", "更新", "发送", "通知",
    "项目", "进度", "上线", "部署", "发布",
    "飞书", "文档", "表格", "知识库",
    "github", "pr", "issue", "仓库",
    "邮件", "快递", "航班",
    "密码", "二维码", "短链",
])


def _needs_tools(text: str) -> bool:
    """判断消息是否需要工具（即不是纯闲聊）."""
    if not text or len(text) > 200:
        # 长消息通常是正式任务描述
        return True
    t = text.lower()
    return any(kw in t for kw in _WORK_KEYWORDS)


async def _feishu_fast_reply(
    ctx: _AppContext,
    emp: Any,
    task_text: str,
    message_history: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """闲聊快速路径 — 不加载工具，单轮回复.

    对比完整路径: ~87K tokens → ~8K tokens, 10x 节省.
    """
    from crew.engine import CrewEngine
    from crew.executor import aexecute_prompt

    engine = CrewEngine(project_dir=ctx.project_dir)
    # soul prompt（人设）+ 简短 chat context，不附加委派花名册和工具
    prompt = engine.prompt(emp, args={"task": (
        "你正在飞书上和 Kai 聊天。像平时一样自然回复。"
        "这次对话不需要查数据或调工具，纯聊天就行。"
    )})

    # 用备用模型（kimi）降低成本，没配就用主模型
    chat_model = emp.fallback_model or emp.model or "claude-sonnet-4-20250514"
    chat_api_key = emp.fallback_api_key or emp.api_key or None
    chat_base_url = emp.fallback_base_url or emp.base_url or None

    # 把对话历史嵌入 prompt（aexecute_prompt 不支持 message_history 参数）
    if message_history:
        history_lines = []
        for msg in message_history[-6:]:  # 最近 6 条足够
            role = "Kai" if msg["role"] == "user" else emp.character_name or emp.name
            history_lines.append(f"{role}: {msg['content']}")
        if history_lines:
            prompt += "\n\n## 最近对话\n\n" + "\n".join(history_lines)

    result = await aexecute_prompt(
        system_prompt=prompt,
        model=chat_model,
        api_key=chat_api_key,
        base_url=chat_base_url,
        stream=False,
        user_message=task_text,
    )

    return {
        "employee": emp.name,
        "prompt": prompt[:500],
        "output": result.content if result else "",
        "model": result.model if result else chat_model,
        "input_tokens": result.input_tokens if result else 0,
        "output_tokens": result.output_tokens if result else 0,
        "base_url": chat_base_url or "",
        "fast_path": True,
    }


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

        # 加载对话历史（15 条足够保持上下文，更早的可用 feishu_chat_history 工具回查）
        _history_limit = 15
        message_history = None
        if ctx.feishu_chat_store:
            history_entries = ctx.feishu_chat_store.get_recent(
                msg_event.chat_id, limit=_history_limit,
            )
            if history_entries:
                message_history = [
                    {"role": e["role"], "content": e["content"]}
                    for e in history_entries
                ]

        if not has_tools and message_history:
            history_text = ctx.feishu_chat_store.format_for_prompt(
                msg_event.chat_id, limit=_history_limit,
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

        # ── 闲聊快速路径 ──
        # 纯闲聊不需要 98 个工具和 agent loop，直接用 soul prompt 回复
        # 省 ~80K tokens / $0.07 per message
        use_fast_path = (
            has_tools
            and image_data is None
            and isinstance(user_msg, str)
            and not _needs_tools(task_text)
            and emp is not None
        )

        if use_fast_path:
            result = await _feishu_fast_reply(
                ctx, emp, task_text, message_history,
            )
        else:
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
