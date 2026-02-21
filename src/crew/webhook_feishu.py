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
        logger.warning(
            "飞书 verification_token 未配置，跳过事件验证（建议设置 FEISHU_VERIFICATION_TOKEN）"
        )

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
        msg_event.msg_type,
        msg_event.chat_type,
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
    import json as _json

    import httpx

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
_WORK_KEYWORDS = frozenset(
    [
        "数据",
        "报表",
        "分析",
        "查一下",
        "查下",
        "帮我查",
        "统计",
        "日程",
        "日历",
        "会议",
        "审批",
        "待办",
        "任务",
        "委派",
        "安排",
        "创建",
        "删除",
        "更新",
        "发送",
        "通知",
        "催",
        "让他",
        "让她",
        "转告",
        "转发",
        "项目",
        "进度",
        "上线",
        "部署",
        "发布",
        "飞书",
        "文档",
        "表格",
        "知识库",
        "github",
        "pr",
        "issue",
        "仓库",
        "邮件",
        "快递",
        "航班",
        "密码",
        "二维码",
        "短链",
        "记忆",
        "记住",
        "笔记",
        "写入",
        "网站",
        "网页",
        "链接",
    ]
)


def _needs_tools(text: str) -> bool:
    """判断消息是否需要工具（即不是纯闲聊）."""
    if not text or len(text) > 200:
        # 长消息通常是正式任务描述
        return True
    t = text.lower()
    # URL 通常需要工具处理（读取网页、链接预览等）
    if "http://" in t or "https://" in t:
        return True
    return any(kw in t for kw in _WORK_KEYWORDS)


async def _feishu_fast_reply(
    ctx: _AppContext,
    emp: Any,
    task_text: str,
    message_history: list[dict[str, Any]] | None,
    max_visibility: str = "open",
    extra_context: str | None = None,
    sender_name: str = "Kai",
) -> dict[str, Any]:
    """闲聊快速路径 — 不加载工具，单轮回复.

    对比完整路径: ~87K tokens → ~8K tokens, 10x 节省.
    """
    from datetime import datetime as _dt

    from crew.executor import aexecute_prompt

    _WEEKDAY_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    _now = _dt.now()
    _date_str = _now.strftime("%Y-%m-%d")
    _weekday_str = _WEEKDAY_CN[_now.weekday()]

    # 只用 soul.md（人设），不用 prompt.md（工具路由表），避免模型 hallucinate 工具调用
    soul_text = ""
    if emp.source_path and emp.source_path.is_dir():
        soul_path = emp.source_path / "soul.md"
        if soul_path.exists():
            soul_text = soul_path.read_text(encoding="utf-8")

    # 去掉 soul.md 中包含工具调用指令的段落（如"你的内心习惯"提到 read_notes/add_memory）
    import re as _re

    soul_text = _re.split(r"(?m)^## 你的内心习惯", soul_text)[0].rstrip()

    display = emp.display_name or emp.name
    char = emp.character_name or display
    prompt = (
        f"# {display}\n\n"
        f"你是{char}。{emp.description}\n\n"
        f"{soul_text}\n\n"
        "---\n\n"
        f"今天是 {_date_str}（{_weekday_str}）。\n\n"
        f"你正在和 {sender_name} 聊天。像平时一样自然回复，纯聊天，不需要查数据或调工具。\n"
        "你知道今天的日期，但没有读过笔记和日记，不知道最近发生了什么具体的事。"
        "不要编造具体事件（客户拜访、会议、调休、出差等）。"
        "不确定的事就说不确定，或者自然地聊你 soul 里写过的日常。\n"
        "不要把别人的生活细节当成自己的——同事的猫、爱好、习惯是他们的，不是你的。"
        "只聊你自己 soul 里明确写过的事，没写过的就说不清楚。\n"
        "你没有手机，不能发照片、文件，也不能主动联系任何人。别装能做到。\n"
        "回复简洁自然，3-5 句话为宜，别写太长。"
    )

    # 私聊时注入私密记忆（快速路径也能看到 Kai 的秘密）
    if max_visibility == "private":
        try:
            from crew.memory import MemoryStore

            store = MemoryStore(project_dir=ctx.project_dir)
            memory_text = store.format_for_prompt(
                emp.name,
                limit=5,
                max_visibility="private",
            )
            if memory_text:
                prompt += "\n\n## 你记得的事\n\n" + memory_text
        except Exception:
            pass

    # 注入同事花名册（name→角色名映射，让 fast path 也认识所有同事）
    try:
        from crew.discovery import discover_employees

        _all = discover_employees(project_dir=ctx.project_dir)
        _roster = []
        for _e in _all.employees.values():
            if _e.name == emp.name:
                continue
            _cname = _e.character_name or _e.display_name or _e.name
            _role = _e.display_name or _e.description or ""
            _roster.append(f"- {_cname}（{_e.name}）— {_role}")
        if _roster:
            prompt += "\n\n## 你的同事\n\n" + "\n".join(_roster)
    except Exception:
        pass

    # 注入外部上下文（站内消息等渠道传入的记忆、用户记忆、对话摘要）
    if extra_context:
        prompt += "\n\n" + extra_context

    # 用备用模型（kimi）降低成本，没配就用主模型
    # 注意：fallback 系列配置不能混用主模型的配置（key/base_url 不通用）
    if emp.fallback_model:
        chat_model = emp.fallback_model
        chat_api_key = emp.fallback_api_key or None
        chat_base_url = emp.fallback_base_url or None
    else:
        chat_model = emp.model or "claude-sonnet-4-20250514"
        chat_api_key = emp.api_key or None
        chat_base_url = emp.base_url or None

    # 把对话历史嵌入 prompt（aexecute_prompt 不支持 message_history 参数）
    if message_history:
        history_lines = []
        for msg in message_history[-6:]:  # 最近 6 条足够
            role = sender_name if msg["role"] == "user" else emp.character_name or emp.name
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
        max_tokens=500,
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


async def _feishu_approve_dispatch(
    ctx: _AppContext,
    msg_event: Any,
    task_id: str,
    action: str,
    *,
    send_feishu_text: Any,
    send_feishu_reply: Any,
) -> None:
    """飞书审批命令 — approve/reject 等待中的任务."""

    async def _reply_text(text: str) -> None:
        if msg_event.chat_type == "group":
            await send_feishu_reply(ctx.feishu_token_mgr, msg_event.message_id, text)
        else:
            await send_feishu_text(ctx.feishu_token_mgr, msg_event.chat_id, text)

    record = ctx.registry.get(task_id)
    if record is None:
        await _reply_text(f"未找到任务: {task_id}")
        return
    if record.status != "awaiting_approval":
        await _reply_text(f"任务 {task_id} 当前状态为 {record.status}，不需要审批。")
        return

    if action == "reject":
        ctx.registry.update(task_id, "failed", error="被 Kai 在飞书拒绝")
        await _reply_text(f"已拒绝任务 {task_id}")
        return

    # approve — 恢复链执行
    from crew.webhook_executor import _resume_chain

    asyncio.create_task(_resume_chain(ctx, task_id))
    await _reply_text(f"已批准任务 {task_id}，继续执行中...")


async def _feishu_route_dispatch(
    ctx: _AppContext,
    msg_event: Any,
    template_name: str,
    task: str,
    *,
    send_feishu_text: Any,
    send_feishu_reply: Any,
) -> None:
    """飞书触发路由模板 — 展开为 chain 异步执行."""
    import json as _json

    from crew.organization import load_organization

    _reply = send_feishu_reply if msg_event.chat_type == "group" else send_feishu_text

    org = load_organization(project_dir=ctx.project_dir)

    async def _reply_text(text: str) -> None:
        if msg_event.chat_type == "group":
            await send_feishu_reply(ctx.feishu_token_mgr, msg_event.message_id, text)
        else:
            await send_feishu_text(ctx.feishu_token_mgr, msg_event.chat_id, text)

    if not template_name:
        names = list(org.routing_templates.keys())
        await _reply_text(f"请指定流程名称。可用: {', '.join(names)}")
        return

    tmpl = org.routing_templates.get(template_name)
    if not tmpl:
        names = list(org.routing_templates.keys())
        await _reply_text(f"未找到流程「{template_name}」。可用: {', '.join(names)}")
        return

    if not task:
        await _reply_text(f"请提供任务描述。格式: 流程 {template_name} <任务描述>")
        return

    # 展开模板为 chain steps
    steps: list[dict[str, Any]] = []
    step_names: list[str] = []
    for step in tmpl.steps:
        if step.optional or step.human:
            continue
        emp_name = None
        if step.employee:
            emp_name = step.employee
        elif step.employees:
            emp_name = step.employees[0]
        elif step.team:
            members = org.get_team_members(step.team)
            emp_name = members[0] if members else None
        if not emp_name:
            continue
        step_task = f"[{step.role}] {task}"
        if steps:
            step_task += "\n\n上一步结果: {prev}"
        step_dict: dict[str, Any] = {"employee_name": emp_name, "task": step_task}
        if step.approval:
            step_dict["approval"] = True
        steps.append(step_dict)
        step_names.append(emp_name)

    if not steps:
        await _reply_text("模板展开后无可执行步骤。")
        return

    # 创建 chain 任务并异步执行
    chain_name = " → ".join(step_names)
    record = ctx.registry.create(
        trigger="feishu",
        target_type="chain",
        target_name=chain_name,
        args={"steps_json": _json.dumps(steps, ensure_ascii=False)},
    )
    import crew.webhook as _wh

    asyncio.create_task(_wh._execute_task(ctx, record.task_id))

    # 回复确认
    await _reply_text(
        f"已启动「{tmpl.label}」流程 ({len(steps)} 步)\n{chain_name}\n任务: {task[:100]}"
    )


async def _feishu_dispatch(ctx: _AppContext, msg_event: Any) -> None:
    """后台处理飞书消息：路由到员工 → 执行 → 回复卡片."""
    from crew.discovery import discover_employees
    from crew.feishu import (
        download_feishu_image,
        resolve_employee_from_mention,
        send_feishu_reply,
        send_feishu_text,
    )

    assert ctx.feishu_config is not None
    assert ctx.feishu_token_mgr is not None

    try:
        discovery = discover_employees(project_dir=ctx.project_dir)

        # 群聊只响应 @mention，私聊可用 default_employee
        use_default = ctx.feishu_config.default_employee if msg_event.chat_type != "group" else ""
        employee_name, task_text = resolve_employee_from_mention(
            msg_event.mentions,
            msg_event.text,
            discovery,
            default_employee=use_default,
        )

        # ── 命令拦截 ──
        _raw = (task_text or msg_event.text or "").strip()

        # 审批命令: "approve xxx" / "reject xxx" / "批准 xxx" / "拒绝 xxx"
        _raw_lower = _raw.lower()
        _approve_action = None
        _approve_task_id = None
        for _ap, _act in [
            ("approve ", "approve"),
            ("reject ", "reject"),
            ("批准 ", "approve"),
            ("拒绝 ", "reject"),
        ]:
            if _raw_lower.startswith(_ap):
                _approve_action = _act
                _approve_task_id = _raw[len(_ap) :].strip()
                break
        if _approve_action and _approve_task_id:
            await _feishu_approve_dispatch(
                ctx,
                msg_event,
                _approve_task_id,
                _approve_action,
                send_feishu_text=send_feishu_text,
                send_feishu_reply=send_feishu_reply,
            )
            return

        # 路由模板命令: "流程 code_change 任务描述"
        _route_match = None
        for _prefix in ("流程 ", "流程:", "route ", "route:"):
            if _raw.lower().startswith(_prefix):
                _route_match = _raw[len(_prefix) :].strip()
                break
        if _route_match:
            parts = _route_match.split(None, 1)
            _tmpl_name = parts[0] if parts else ""
            _tmpl_task = parts[1] if len(parts) > 1 else ""
            await _feishu_route_dispatch(
                ctx,
                msg_event,
                _tmpl_name,
                _tmpl_task,
                send_feishu_text=send_feishu_text,
                send_feishu_reply=send_feishu_reply,
            )
            return

        if employee_name is None:
            if msg_event.chat_type == "group":
                # 群聊没 @人，静默忽略
                return
            await send_feishu_text(
                ctx.feishu_token_mgr,
                msg_event.chat_id,
                "未能识别目标员工，请 @员工名 + 任务描述。",
            )
            return

        emp = discovery.get(employee_name)

        # 解析发送者姓名（群聊需要区分多人）
        sender_name = ""
        if msg_event.chat_type == "group" and msg_event.sender_id:
            try:
                from crew.feishu import get_user_name

                sender_name = await get_user_name(ctx.feishu_token_mgr, msg_event.sender_id)
            except Exception:
                pass

        # 图片 → 下载图片 + vision
        image_key = msg_event.image_key
        image_message_id = msg_event.message_id

        if not image_key and msg_event.chat_type == "group" and msg_event.msg_type == "text":
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
                    ctx.feishu_token_mgr,
                    image_key,
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

        # ── 可见性上下文 ──
        # 私聊 → private（可以看到 Kai 的秘密），群聊 → open
        _visibility = "private" if msg_event.chat_type != "group" else "open"

        # 执行员工 — 飞书实时聊天
        from crew.tool_schema import AGENT_TOOLS

        has_tools = any(t in AGENT_TOOLS for t in (emp.tools or [])) if emp else False

        from datetime import datetime as _dt

        _WEEKDAY_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        _now = _dt.now()
        _date_header = f"今天是 {_now.strftime('%Y-%m-%d')}（{_WEEKDAY_CN[_now.weekday()]}）。\n"

        _is_group = msg_event.chat_type == "group"
        _chat_partner = sender_name if (_is_group and sender_name) else "Kai"

        if has_tools:
            if _is_group:
                chat_context = (
                    _date_header + f"你在飞书群聊里。当前和你说话的人是{_chat_partner}，不是 Kai。"
                    "注意区分群里不同的人，不要把所有人都当成 Kai。\n"
                    "像平时一样自然回复。"
                    "需要数据就调工具查，拿到数据用自己的话说，别搬 JSON。"
                    "只陈述工具返回的事实，没查过的事不要说，不要主动编造任何信息。"
                )
            else:
                chat_context = (
                    _date_header + "你正在飞书上和 Kai 聊天。像平时一样自然回复。"
                    "需要数据就调工具查，拿到数据用自己的话说，别搬 JSON。"
                    "如果 Kai 在追问上一个话题，直接接着聊，不用重新查。"
                    "只陈述工具返回的事实，没查过的事不要说，不要主动编造任何信息。"
                )
        if not has_tools:
            if _is_group:
                chat_context = (
                    _date_header + f"你在飞书群聊里。当前和你说话的人是{_chat_partner}，不是 Kai。"
                    "注意区分群里不同的人，不要把所有人都当成 Kai。\n"
                    "你现在没有任何业务数据。"
                    "你可以聊：你的职能介绍、帮他想问题、帮他起草文字。"
                )
            else:
                chat_context = (
                    _date_header + "这是飞书实时聊天。你现在没有任何业务数据。"
                    "你不知道：今天的会议安排、项目进度、客户状态、合同情况、同事的工作产出、任何具体数字。"
                    "Kai 问这些就说「这个我得看看今天的数据才能说」。"
                    "你可以聊：你的职能介绍、帮他想问题、帮他起草文字。"
                )

        # 加载对话历史（15 条足够保持上下文，更早的可用 feishu_chat_history 工具回查）
        _history_limit = 15
        message_history = None
        _prev_was_full = False  # 上一轮是否走了 full path（有工具）
        if ctx.feishu_chat_store:
            history_entries = ctx.feishu_chat_store.get_recent(
                msg_event.chat_id,
                limit=_history_limit,
            )
            if history_entries:
                message_history = [
                    {"role": e["role"], "content": e["content"]} for e in history_entries
                ]
                # 检查上一轮 assistant 是否走了 full path（有工具）
                # 如果是，跟进消息也走 full path，避免"是的""好的"等确认被误判为闲聊
                for _he in reversed(history_entries):
                    if _he.get("role") == "assistant":
                        _prev_was_full = _he.get("path") == "full"
                        break

        if not has_tools and message_history:
            history_text = ctx.feishu_chat_store.format_for_prompt(
                msg_event.chat_id,
                limit=_history_limit,
            )
            if history_text:
                chat_context += (
                    "\n\n## 最近对话记录\n\n"
                    "以下是你和 Kai 最近的对话，用来保持上下文连贯。\n\n" + history_text
                )
            message_history = None

        args = {"task": chat_context, "_max_visibility": _visibility}
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
        import time as _time

        _t0 = _time.monotonic()

        use_fast_path = (
            has_tools
            and image_data is None
            and isinstance(user_msg, str)
            and not _needs_tools(task_text)
            and not _prev_was_full  # 上一轮用了工具，跟进也走 full path
            and emp is not None
        )

        if use_fast_path:
            try:
                result = await _feishu_fast_reply(
                    ctx,
                    emp,
                    task_text,
                    message_history,
                    max_visibility=_visibility,
                    sender_name=sender_name or "Kai",
                )
            except Exception as fast_exc:
                logger.warning("闲聊快速路径失败，回退完整路径: %s", fast_exc)
                use_fast_path = False
        if not use_fast_path:
            # 通过 webhook 模块查找，确保 mock patch 生效
            import crew.webhook as _wh

            result = await _wh._execute_employee(
                ctx,
                employee_name,
                args,
                model=None,
                user_message=user_msg,
                message_history=message_history,
            )

        _elapsed = _time.monotonic() - _t0
        _path_label = "fast" if use_fast_path else "full"
        _model_used = result.get("model", "?") if isinstance(result, dict) else "?"
        _in_tok = result.get("input_tokens", 0) if isinstance(result, dict) else 0
        _out_tok = result.get("output_tokens", 0) if isinstance(result, dict) else 0
        logger.warning(
            "飞书回复 [%s] %.1fs model=%s in=%d out=%d msg=%s",
            _path_label,
            _elapsed,
            _model_used,
            _in_tok,
            _out_tok,
            task_text[:40],
        )

        # 记录对话历史
        output_text = result.get("output", "") if isinstance(result, dict) else str(result)
        if ctx.feishu_chat_store:
            ctx.feishu_chat_store.append(
                msg_event.chat_id, "user", task_text, sender_name=sender_name
            )
            ctx.feishu_chat_store.append(
                msg_event.chat_id, "assistant", output_text, path=_path_label
            )

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
            send_data = await send_feishu_reply(
                ctx.feishu_token_mgr,
                msg_event.message_id,
                output_text,
            )
        else:
            send_data = await send_feishu_text(
                ctx.feishu_token_mgr,
                msg_event.chat_id,
                output_text,
            )

        # 检查发送结果
        send_code = send_data.get("code", -1) if isinstance(send_data, dict) else -1
        send_ok = send_code == 0
        if not send_ok:
            send_msg = (
                send_data.get("msg", "unknown") if isinstance(send_data, dict) else "no response"
            )
            logger.error(
                "飞书消息投递失败: code=%s msg=%s chat_id=%s employee=%s",
                send_code,
                send_msg,
                msg_event.chat_id,
                employee_name,
            )

        # 记录任务（附带投递状态）
        record = ctx.registry.create(
            trigger="feishu",
            target_type="employee",
            target_name=employee_name,
            args=args,
        )
        task_result = dict(result) if isinstance(result, dict) else {"output": str(result)}
        if send_ok:
            task_result["delivered"] = True
        else:
            task_result["delivered"] = False
            task_result["feishu_error"] = {
                "code": send_code,
                "msg": send_data.get("msg", "unknown")
                if isinstance(send_data, dict)
                else "no response",
            }
        ctx.registry.update(record.task_id, "completed", result=task_result)

        # 工作日志记录
        if isinstance(result, dict):
            try:
                from crew.id_client import alog_work

                _aid = emp.agent_id if emp else None
                if _aid:
                    await alog_work(
                        agent_id=_aid,
                        task_type=employee_name,
                        task_input=(task_text or "")[:500],
                        task_output=output_text[:2000],
                        model_used=result.get("model", ""),
                        tokens_used=_in_tok + _out_tok,
                        execution_ms=int(_elapsed * 1000),
                        crew_task_id=record.task_id,
                    )
            except Exception:
                logger.debug("飞书工作日志记录失败: %s", employee_name)

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
