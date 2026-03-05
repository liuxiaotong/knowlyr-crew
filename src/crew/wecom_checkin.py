"""企业微信打卡通报 -- 每日自动获取考勤数据并推送到群聊."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# 北京时间
_CST = timezone(timedelta(hours=8))

# 上班时间 fallback（getcheckinoption 失败时回退）
_FALLBACK_WORK_SEC = 9 * 3600 + 30 * 60  # 09:30
_FALLBACK_FLEX_TIME = 0

WECOM_API_BASE = "https://qyapi.weixin.qq.com/cgi-bin"

# 打卡通报推送群
CHECKIN_GROUP_CHAT_ID = "wrlzQdBgAAjyn00rCdxEegaHe8vRY0FA"


async def get_department_users(
    token: str,
    department_id: int = 1,
    fetch_child: bool = True,
) -> list[dict[str, Any]]:
    """获取部门下的用户列表.

    调用 /cgi-bin/user/simplelist 接口。

    Args:
        token: access_token
        department_id: 部门 ID（1 = 根部门）
        fetch_child: 是否递归获取子部门

    Returns:
        用户列表，每项包含 userid, name, department 等
    """
    from crew.wecom import get_wecom_client

    client = get_wecom_client()
    url = f"{WECOM_API_BASE}/user/simplelist"
    params = {
        "access_token": token,
        "department_id": department_id,
        "fetch_child": 1 if fetch_child else 0,
    }

    resp = await client.get(url, params=params)
    data = resp.json()

    errcode = data.get("errcode", -1)
    if errcode != 0:
        errmsg = data.get("errmsg", "unknown")
        logger.error("获取用户列表失败: %s (errcode=%d)", errmsg, errcode)
        raise RuntimeError(f"获取用户列表失败: {errmsg} (errcode={errcode})")

    return data.get("userlist", [])


async def get_checkin_data(
    token: str,
    user_ids: list[str],
    start_time: int,
    end_time: int,
    checkin_type: int = 1,
) -> list[dict[str, Any]]:
    """获取打卡数据.

    调用 /cgi-bin/checkin/getcheckindata 接口。
    注意：useridlist 每次最多 100 人，超过需要分批。

    Args:
        token: access_token
        user_ids: 用户 userid 列表
        start_time: 开始时间（unix timestamp）
        end_time: 结束时间（unix timestamp）
        checkin_type: 1=上班 2=下班 3=全部

    Returns:
        打卡记录列表
    """
    from crew.wecom import get_wecom_client

    client = get_wecom_client()
    url = f"{WECOM_API_BASE}/checkin/getcheckindata"
    params = {"access_token": token}

    all_records: list[dict[str, Any]] = []

    # 企微限制每次最多 100 人
    batch_size = 100
    for i in range(0, len(user_ids), batch_size):
        batch = user_ids[i : i + batch_size]
        body = {
            "opencheckindatatype": checkin_type,
            "starttime": start_time,
            "endtime": end_time,
            "useridlist": batch,
        }

        resp = await client.post(url, json=body, params=params)
        data = resp.json()

        errcode = data.get("errcode", -1)
        if errcode != 0:
            errmsg = data.get("errmsg", "unknown")
            logger.error("获取打卡数据失败: %s (errcode=%d)", errmsg, errcode)
            raise RuntimeError(f"获取打卡数据失败: {errmsg} (errcode={errcode})")

        all_records.extend(data.get("checkindata", []))

    return all_records


async def get_checkin_rules(
    token: str,
    user_ids: list[str],
    dt: int,
) -> dict[str, dict[str, Any]]:
    """获取打卡规则（上班时间、弹性时间等）.

    调用 /cgi-bin/checkin/getcheckinoption 接口。
    没有返回的 userid 表示该员工不需要打卡。

    Args:
        token: access_token
        user_ids: 用户 userid 列表
        dt: 时间戳（用于查询当天的打卡规则）

    Returns:
        {userid: {"work_sec": int, "flex_time": int, "groupname": str}}
        API 调用失败时返回空 dict（调用方应回退到 fallback）
    """
    from crew.wecom import get_wecom_client

    client = get_wecom_client()
    url = f"{WECOM_API_BASE}/checkin/getcheckinoption"
    params = {"access_token": token}

    rules: dict[str, dict[str, Any]] = {}

    batch_size = 100
    for i in range(0, len(user_ids), batch_size):
        batch = user_ids[i : i + batch_size]
        body = {
            "datetime": dt,
            "useridlist": batch,
        }

        try:
            resp = await client.post(url, json=body, params=params)
            data = resp.json()
        except Exception:
            logger.exception("获取打卡规则请求异常")
            return {}

        errcode = data.get("errcode", -1)
        if errcode != 0:
            errmsg = data.get("errmsg", "unknown")
            logger.warning("获取打卡规则失败: %s (errcode=%d)，将使用 fallback", errmsg, errcode)
            return {}

        for info in data.get("info", []):
            uid = info.get("userid", "")
            if not uid:
                continue
            group = info.get("group", {})
            groupname = group.get("groupname", "")

            # 取第一个 checkindate（当天的规则）
            checkindate_list = group.get("checkindate", [])
            if not checkindate_list:
                continue

            checkindate = checkindate_list[0]
            checkintime_list = checkindate.get("checkintime", [])
            if not checkintime_list:
                continue

            work_sec = checkintime_list[0].get("work_sec", _FALLBACK_WORK_SEC)

            # 弹性时间
            # 企微有两种弹性模式：
            # 1. 允许晚到晚走：弹性时间在 late_rule.onwork_flex_time
            # 2. 早到早走/晚到晚走：弹性时间在 max_allow_arrive_late
            # 取两者的 max 以覆盖两种模式
            flex_time = 0
            allow_flex = checkindate.get("allow_flex", False)
            if allow_flex:
                late_rule = checkindate.get("late_rule", {})
                onwork_flex = late_rule.get("onwork_flex_time", 0)
                max_arrive_late = checkindate.get("max_allow_arrive_late", 0)
                flex_time = max(onwork_flex, max_arrive_late)

            rules[uid] = {
                "work_sec": work_sec,
                "flex_time": flex_time,
                "groupname": groupname,
            }

    return rules


async def get_leave_users(
    token: str,
    start_time: int,
    end_time: int,
) -> set[str]:
    """获取当天请假（已审批通过）的用户 userid 集合.

    依次调用 getapprovalinfo（分页获取审批单号）和 getapprovaldetail（逐个获取详情），
    筛选 sp_name == "请假" 且 sp_status == 2（已同意）的审批单。

    API 失败时不阻断流程，返回空 set 并记录 warning。

    Args:
        token: access_token
        start_time: 开始时间戳
        end_time: 结束时间戳

    Returns:
        当天请假的 userid 集合
    """
    from crew.wecom import get_wecom_client

    client = get_wecom_client()
    leave_user_ids: set[str] = set()

    # ── 第 1 步：获取审批单号列表（分页） ──
    sp_no_list: list[str] = []
    cursor = 0

    try:
        while True:
            url = f"{WECOM_API_BASE}/oa/getapprovalinfo"
            params = {"access_token": token}
            body = {
                "starttime": str(start_time),
                "endtime": str(end_time),
                "cursor": cursor,
                "size": 100,
                "filters": [{"key": "sp_status", "value": "2"}],
            }

            resp = await client.post(url, json=body, params=params)
            data = resp.json()

            errcode = data.get("errcode", -1)
            if errcode != 0:
                errmsg = data.get("errmsg", "unknown")
                logger.warning("获取审批单列表失败: %s (errcode=%d)", errmsg, errcode)
                return set()

            sp_no_list.extend(data.get("sp_no_list", []))

            new_cursor = data.get("new_next_cursor")
            if not new_cursor or new_cursor == cursor:
                break
            cursor = new_cursor

    except Exception:
        logger.exception("获取审批单列表请求异常")
        return set()

    if not sp_no_list:
        return set()

    logger.info("获取到 %d 个已审批通过的审批单", len(sp_no_list))

    # ── 第 2 步：逐个获取审批详情，筛选请假单 ──
    for sp_no in sp_no_list:
        try:
            url = f"{WECOM_API_BASE}/oa/getapprovaldetail"
            params = {"access_token": token}
            body = {"sp_no": sp_no}

            resp = await client.post(url, json=body, params=params)
            data = resp.json()

            errcode = data.get("errcode", -1)
            if errcode != 0:
                errmsg = data.get("errmsg", "unknown")
                logger.warning("获取审批详情失败 sp_no=%s: %s (errcode=%d)", sp_no, errmsg, errcode)
                continue

            info = data.get("info", {})
            sp_name = info.get("sp_name", "")
            sp_status = info.get("sp_status", 0)

            if sp_name == "请假" and sp_status == 2:
                applyer = info.get("applyer", {})
                userid = applyer.get("userid", "")
                if userid:
                    leave_user_ids.add(userid)

        except Exception:
            logger.exception("获取审批详情请求异常 sp_no=%s", sp_no)
            continue

    return leave_user_ids


def classify_checkin(
    users: list[dict[str, Any]],
    checkin_records: list[dict[str, Any]],
    rules: dict[str, dict[str, Any]] | None = None,
    leave_users: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """将打卡记录分类为正常、迟到、请假、未打卡.

    Args:
        users: 用户列表（含 userid, name）
        checkin_records: 打卡记录
        rules: 打卡规则 {userid: {"work_sec", "flex_time", ...}}。
               如果提供，每个人用自己的 work_sec + flex_time 判断迟到。
               如果为 None，回退到全局 fallback (09:30)。
        leave_users: 当天请假的 userid 集合。
                     如果为 None 或空集，不影响分类逻辑。

    Returns:
        {"normal": [...], "late": [...], "on_leave": [...], "absent": [...]}
        每项包含 name, userid, checkin_time(可选)
    """
    # 构建 userid -> 最早上班打卡时间
    # 打卡记录可能有多条（补卡等），取最早的一条
    checkin_map: dict[str, int] = {}
    for record in checkin_records:
        uid = record.get("userid", "")
        checkin_time = record.get("checkin_time", 0)
        if uid and checkin_time:
            if uid not in checkin_map or checkin_time < checkin_map[uid]:
                checkin_map[uid] = checkin_time

    normal: list[dict[str, Any]] = []
    late: list[dict[str, Any]] = []
    on_leave: list[dict[str, Any]] = []
    absent: list[dict[str, Any]] = []

    _leave_set = leave_users or set()

    for user in users:
        uid = user["userid"]
        name = user.get("name", uid)

        # 请假的人直接归入 on_leave，不再判断打卡
        if uid in _leave_set:
            on_leave.append({"name": name, "userid": uid})
            continue

        # 每个人的迟到阈值
        if rules is not None and uid in rules:
            rule = rules[uid]
            late_threshold = rule["work_sec"] + rule["flex_time"]
        else:
            late_threshold = _FALLBACK_WORK_SEC + _FALLBACK_FLEX_TIME

        if uid not in checkin_map:
            absent.append({"name": name, "userid": uid})
            continue

        ts = checkin_map[uid]
        dt = datetime.fromtimestamp(ts, tz=_CST)
        time_str = dt.strftime("%H:%M")

        # 判断迟到：比较当天时间
        day_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        if day_seconds > late_threshold:
            late.append({"name": name, "userid": uid, "checkin_time": time_str})
        else:
            normal.append({"name": name, "userid": uid, "checkin_time": time_str})

    return {"normal": normal, "late": late, "on_leave": on_leave, "absent": absent}


def format_checkin_report(
    classified: dict[str, list[dict[str, Any]]],
    report_date: datetime | None = None,
) -> str:
    """格式化打卡通报消息.

    Args:
        classified: classify_checkin 的返回值
        report_date: 报告日期（默认今天）

    Returns:
        格式化的通报文本
    """
    if report_date is None:
        report_date = datetime.now(_CST)

    date_str = f"{report_date.month}/{report_date.day}"

    normal = classified["normal"]
    late = classified["late"]
    on_leave = classified.get("on_leave", [])
    absent = classified["absent"]

    lines: list[str] = []
    lines.append(f"今日考勤通报（{date_str}）")
    lines.append("")

    # 正常打卡
    if normal:
        lines.append(f"正常打卡（{len(normal)}人）")
        # 每行最多 4 人，用 | 分隔
        items = [f"{p['name']} {p['checkin_time']}" for p in normal]
        for i in range(0, len(items), 4):
            lines.append(" | ".join(items[i : i + 4]))
    else:
        lines.append("正常打卡（0人）")

    lines.append("")

    # 迟到
    if late:
        lines.append(f"迟到（{len(late)}人）")
        items = [f"{p['name']} {p['checkin_time']}" for p in late]
        for i in range(0, len(items), 4):
            lines.append(" | ".join(items[i : i + 4]))
    else:
        lines.append("迟到（0人）")

    lines.append("")

    # 请假
    if on_leave:
        lines.append(f"请假（{len(on_leave)}人）")
        items = [p["name"] for p in on_leave]
        for i in range(0, len(items), 6):
            lines.append(" | ".join(items[i : i + 6]))
    else:
        lines.append("请假（0人）")

    lines.append("")

    # 未打卡
    if absent:
        lines.append(f"未打卡（{len(absent)}人）")
        items = [p["name"] for p in absent]
        for i in range(0, len(items), 6):
            lines.append(" | ".join(items[i : i + 6]))
    else:
        lines.append("未打卡（0人）")

    return "\n".join(lines)


async def run_checkin_report(
    corp_id: str | None = None,
    secret: str | None = None,
    chat_id: str = CHECKIN_GROUP_CHAT_ID,
    dry_run: bool = False,
) -> str:
    """执行打卡通报完整流程.

    1. 获取 token
    2. 获取全员列表
    3. 获取当天上班打卡数据
    4. 分类 + 格式化
    5. 推送到群聊（dry_run=True 时跳过推送）

    Args:
        corp_id: 企业 CorpID（None 时从配置加载）
        secret: 应用 Secret（None 时从配置加载）
        chat_id: 推送目标群聊 ID
        dry_run: True 时只生成消息不推送

    Returns:
        格式化的通报消息文本
    """
    from crew.wecom import WecomTokenManager, send_wecom_group_text

    # 加载配置
    if corp_id is None or secret is None:
        from crew.wecom import load_wecom_config

        config = load_wecom_config()
        corp_id = corp_id or config.corp_id
        secret = secret or config.secret

    if not corp_id or not secret:
        raise RuntimeError("企微 CorpID 或 Secret 未配置")

    token_mgr = WecomTokenManager(corp_id, secret)
    token = await token_mgr.get_token()

    # 获取全员列表（含 userid -> name 映射）
    logger.info("获取全员列表...")
    users = await get_department_users(token, department_id=1, fetch_child=True)
    if not users:
        logger.warning("未获取到任何用户")
        return "未获取到任何用户，请检查企微权限配置。"

    user_ids = [u["userid"] for u in users]
    logger.info("共 %d 名员工", len(user_ids))

    # 计算当天时间范围（北京时间 00:00 ~ 当前时间）
    now = datetime.now(_CST)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_ts = int(today_start.timestamp())
    end_ts = int(now.timestamp())

    # 获取打卡规则（用于判断谁需要打卡 + 个人迟到阈值）
    logger.info("获取打卡规则...")
    rules = await get_checkin_rules(token, user_ids, start_ts)

    if rules:
        # 只统计有打卡规则的人（没有规则 = 不需要打卡）
        rule_user_ids = set(rules.keys())
        filtered_users = [u for u in users if u["userid"] in rule_user_ids]
        filtered_user_ids = [u["userid"] for u in filtered_users]
        logger.info(
            "有打卡规则 %d 人（排除 %d 人）",
            len(filtered_users),
            len(users) - len(filtered_users),
        )
    else:
        # fallback：规则获取失败，用全员 + 默认阈值
        logger.warning("打卡规则获取失败，回退到全员 + 默认 09:30 阈值")
        filtered_users = users
        filtered_user_ids = user_ids
        rules = None  # 传 None 让 classify_checkin 用 fallback

    # 获取上班打卡数据
    logger.info(
        "获取打卡数据: %s ~ %s", today_start.strftime("%Y-%m-%d %H:%M"), now.strftime("%H:%M")
    )
    checkin_records = await get_checkin_data(
        token, filtered_user_ids, start_ts, end_ts, checkin_type=1
    )
    logger.info("获取到 %d 条打卡记录", len(checkin_records))

    # 获取请假人员
    logger.info("获取请假审批数据...")
    leave_users = await get_leave_users(token, start_ts, end_ts)
    logger.info("当天请假 %d 人", len(leave_users))

    # 分类
    classified = classify_checkin(
        filtered_users, checkin_records, rules=rules, leave_users=leave_users
    )
    report = format_checkin_report(classified, report_date=now)

    logger.info(
        "考勤统计: 正常=%d 迟到=%d 请假=%d 未打卡=%d",
        len(classified["normal"]),
        len(classified["late"]),
        len(classified["on_leave"]),
        len(classified["absent"]),
    )

    # 推送
    if not dry_run:
        logger.info("推送考勤通报到群: %s", chat_id)
        result = await send_wecom_group_text(token_mgr, chat_id, report)
        errcode = result.get("errcode", -1)
        if errcode != 0:
            errmsg = result.get("errmsg", "unknown")
            logger.error("考勤通报推送失败: %s (errcode=%d)", errmsg, errcode)
        else:
            logger.info("考勤通报推送成功")
    else:
        logger.info("dry_run 模式，跳过推送")

    return report


async def cron_checkin_report() -> None:
    """cron 定时任务入口 -- 由 cron_scheduler 调用."""
    try:
        report = await run_checkin_report()
        logger.info("考勤通报已完成:\n%s", report)
    except Exception:
        logger.exception("考勤通报执行失败")
