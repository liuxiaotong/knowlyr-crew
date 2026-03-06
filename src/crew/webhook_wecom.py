"""企业微信事件处理 -- 接收企微回调、消息路由、员工执行、回复.

支持单聊和群聊两种模式:
- 单聊: 用户直接给应用发消息，回复走 /cgi-bin/message/send
- 群聊: 用户在群里 @应用 发消息，回复走单聊（降级方案）或群聊 API

群聊降级说明:
  企微 /cgi-bin/appchat/send 要求应用可见范围为根部门且只能发到应用自建群，
  限制过严。默认用 /cgi-bin/message/send 单聊回复发送者，确保可用性。
  如果配置了 group_reply_mode="group" 且满足权限条件，可尝试群聊回复。
"""

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
    from starlette.responses import PlainTextResponse

    # 解析路由参数 app_id（多应用预留）
    app_id = request.path_params.get("app_id", "default")
    logger.info("企微事件: app_id=%s method=%s", app_id, request.method)

    wecom_ctx = ctx.wecom_ctx
    if wecom_ctx is None:
        return PlainTextResponse("wecom not configured", status_code=501)

    config = wecom_ctx["config"]
    dedup = wecom_ctx["dedup"]

    # 根据 app_id 选择密钥（contact = 通讯录同步独立密钥）
    is_contact = app_id == "contact"
    if is_contact and "contact_crypto" in wecom_ctx:
        crypto = wecom_ctx["contact_crypto"]
        callback_token = wecom_ctx["contact_token"]
        token_mgr = wecom_ctx.get("contact_token_mgr", wecom_ctx["token_mgr"])
    else:
        crypto = wecom_ctx["crypto"]
        callback_token = config.token
        token_mgr = wecom_ctx["token_mgr"]

    # ── GET: URL 验证 ──
    if request.method == "GET":
        msg_signature = request.query_params.get("msg_signature", "")
        timestamp = request.query_params.get("timestamp", "")
        nonce = request.query_params.get("nonce", "")
        echostr = request.query_params.get("echostr", "")

        from crew.wecom import verify_wecom_signature

        if not verify_wecom_signature(callback_token, timestamp, nonce, echostr, msg_signature):
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
    if not verify_wecom_signature(callback_token, timestamp, nonce, encrypt_content, msg_signature):
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

    # ── 通讯录变更事件 ──
    if msg_type == "event" and msg.get("Event") == "change_contact":
        change_type = msg.get("ChangeType", "")
        logger.info("企微通讯录变更: change_type=%s UserID=%s", change_type, msg.get("UserID", ""))

        if change_type in ("delete_user", "update_user"):
            task = asyncio.create_task(
                _handle_contact_change(
                    ctx,
                    change_type=change_type,
                    wecom_userid=msg.get("UserID", ""),
                    token_mgr=wecom_ctx["token_mgr"],
                    msg=msg,
                )
            )
            _background_tasks.add(task)
            task.add_done_callback(_task_done_callback)

        return PlainTextResponse("success")

    # 群聊判断：企微自建应用的群聊回调 XML 中没有标准 chatid 字段，
    # 但消息内容会以 "@应用名" 开头。这里同时检查可能存在的扩展字段。
    chat_id = msg.get("ChatId", "")
    # 部分企微版本/配置下 ToUserName 为 corpid 不带群信息，
    # 判断群聊的最可靠方式是检查 Content 是否以 @应用名 开头
    is_group = bool(chat_id) or content.startswith("@")

    logger.info(
        "企微消息: type=%s from=%s group=%s chat_id=%s content=%s",
        msg_type,
        from_user,
        is_group,
        chat_id or "(none)",
        content[:50] if content else "(empty)",
    )

    # 初期只处理文本消息
    if msg_type != "text":
        logger.info("企微消息类型不支持: %s", msg_type)
        return PlainTextResponse("success")

    if not content:
        return PlainTextResponse("success")

    # 群聊消息：去除 @应用名 前缀，提取纯净文本
    if is_group:
        from crew.wecom import strip_wecom_at_prefix

        content = strip_wecom_at_prefix(content)
        if not content:
            return PlainTextResponse("success")

    # 去重
    if msg_id and dedup.is_duplicate(msg_id):
        logger.warning("企微消息去重: %s", msg_id)
        return PlainTextResponse("success")

    # 后台处理（企微要求快速返回 "success"）
    task = asyncio.create_task(
        _wecom_dispatch(
            ctx,
            from_user,
            content,
            config.agent_id,
            token_mgr,
            chat_id=chat_id if is_group else "",
            is_group=is_group,
        )
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
    *,
    chat_id: str = "",
    is_group: bool = False,
) -> None:
    """后台处理企微消息：路由到员工 -> 执行 -> 主动推送回复.

    Args:
        ctx: 应用上下文
        from_user: 发送者 userid
        text: 消息文本（群聊已去除 @前缀）
        agent_id: 应用 AgentId
        token_mgr: token 管理器
        chat_id: 群聊 ID（群聊时可能有值，自建应用通常为空）
        is_group: 是否群聊消息
    """
    from crew.discovery import discover_employees
    from crew.feishu import resolve_employee_from_mention
    from crew.wecom import send_wecom_group_text, send_wecom_text

    async def _reply(reply_text: str) -> None:
        """根据消息来源选择回复方式.

        群聊回复策略（按优先级）:
        1. 如果有 chat_id 且配置了 group_reply_mode="group"，尝试群聊 API
        2. 降级为单聊回复发送者（默认方案，最稳定）
        """
        if is_group and chat_id:
            # 有 chat_id 时尝试群聊回复
            # 注意: appchat/send 需要应用可见范围为根部门且群必须是应用创建的
            try:
                result = await send_wecom_group_text(token_mgr, chat_id, reply_text)
                if result.get("errcode", -1) == 0:
                    return
                logger.info(
                    "群聊 API 回复失败 (errcode=%d)，降级为单聊回复",
                    result.get("errcode", -1),
                )
            except Exception as exc:
                logger.info("群聊 API 回复异常: %s，降级为单聊回复", exc)

        # 默认/降级: 单聊回复发送者
        await send_wecom_text(token_mgr, from_user, agent_id, reply_text)

    try:
        wecom_config = ctx.wecom_ctx["config"]
        default_emp = wecom_config.default_employee
        # 从企微 config 读取 tenant_id（空字符串视为 None → 走 admin 租户）
        _tenant_id: str | None = getattr(wecom_config, "tenant_id", "") or None

        discovery = discover_employees(project_dir=ctx.project_dir, tenant_id=_tenant_id)

        # 文本前缀匹配 + default_employee
        employee_name, task_text = resolve_employee_from_mention(
            [],  # 企微无结构化 mentions
            text,
            discovery,
            default_employee=default_emp,
        )

        if employee_name is None:
            await _reply("未能识别目标员工，请用员工名开头发送消息。")
            return

        # ── SG Bridge 主通道尝试 ──
        chat_context = (
            "这是企业微信群聊，用户 @你 提问。像平时一样自然回复。"
            if is_group
            else "这是企业微信实时聊天。像平时一样自然回复。"
        )

        _sg_reply: str | None = None
        try:
            from crew.sg_bridge import sg_dispatch

            _sg_reply = await sg_dispatch(
                task_text,
                project_dir=ctx.project_dir,
                employee_name=employee_name,
                chat_context=chat_context,
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
        await _reply(output_text)

        # 记录任务
        trigger = "wecom_group" if is_group else "wecom"
        record = ctx.registry.create(
            trigger=trigger,
            target_type="employee",
            target_name=employee_name,
            args={"task": task_text},
        )
        ctx.registry.update(record.task_id, "completed", result={"output": output_text})

    except Exception as e:
        logger.exception("企微消息处理失败: %s", e)
        try:
            await _reply("处理时出了点问题，请稍后再试。")
        except Exception:
            logger.warning("企微错误回复发送失败", exc_info=True)


async def _handle_contact_change(
    ctx: Any,
    *,
    change_type: str,
    wecom_userid: str,
    token_mgr: Any,
    msg: dict[str, str],
) -> None:
    """处理企微通讯录变更事件 — 离职员工自动清除 staff 身份.

    企微通讯录回调 XML 格式:
      MsgType=event, Event=change_contact
      ChangeType=delete_user (离职删除) / update_user (信息变更)

    流程:
      1. 从企微 API 获取用户详情（含手机号）
      2. 通过手机号调 knowlyr-id 清除 staff_badge + about_section
      3. 官网 about 页自动更新（因为 team.js 动态读取 about_section）

    Args:
        ctx: 应用上下文
        change_type: delete_user 或 update_user
        wecom_userid: 企微 UserID
        token_mgr: WecomTokenManager
        msg: 解析后的完整 XML 字段
    """
    from crew.wecom import get_wecom_user_info, offboard_user_by_phone

    logger.info(
        "处理通讯录变更: change_type=%s userid=%s",
        change_type,
        wecom_userid,
    )

    if not wecom_userid:
        logger.warning("通讯录变更事件缺少 UserID，跳过")
        return

    # ── delete_user: 直接走离职流程 ──
    if change_type == "delete_user":
        # 离职用户可能已从企微删除，先尝试查询
        user_info = await get_wecom_user_info(token_mgr, wecom_userid)
        mobile = ""
        if user_info:
            mobile = user_info.get("mobile", "")

        if not mobile:
            # 企微已删除该用户，无法获取手机号
            # 记录日志，后续可人工处理
            logger.warning(
                "企微离职用户 %s 无法获取手机号（可能已被删除），需人工处理",
                wecom_userid,
            )
            return

        result = await offboard_user_by_phone(mobile)
        if result.get("ok"):
            logger.info(
                "离职同步完成: wecom_userid=%s phone=***%s user_id=%s cleared=%s",
                wecom_userid,
                mobile[-4:],
                result.get("user_id"),
                result.get("cleared"),
            )
        else:
            logger.warning(
                "离职同步失败: wecom_userid=%s phone=***%s detail=%s",
                wecom_userid,
                mobile[-4:],
                result.get("detail", "unknown"),
            )
        return

    # ── update_user: 检查是否禁用（status=2 表示已禁用/离职） ──
    if change_type == "update_user":
        user_info = await get_wecom_user_info(token_mgr, wecom_userid)
        if not user_info:
            logger.info("update_user: 无法获取用户 %s 信息，跳过", wecom_userid)
            return

        # 企微 status: 1=已激活, 2=已禁用, 4=未激活, 5=退出企业
        status = user_info.get("status", 1)
        if status in (2, 5):
            mobile = user_info.get("mobile", "")
            if not mobile:
                logger.warning(
                    "企微用户 %s 已禁用/退出(status=%d) 但无手机号，需人工处理",
                    wecom_userid,
                    status,
                )
                return

            result = await offboard_user_by_phone(mobile)
            if result.get("ok"):
                logger.info(
                    "用户禁用同步完成: wecom_userid=%s status=%d phone=***%s cleared=%s",
                    wecom_userid,
                    status,
                    mobile[-4:],
                    result.get("cleared"),
                )
            else:
                logger.warning(
                    "用户禁用同步失败: wecom_userid=%s detail=%s",
                    wecom_userid,
                    result.get("detail", "unknown"),
                )
        else:
            logger.debug("update_user: 用户 %s status=%d，非离职状态，跳过", wecom_userid, status)
