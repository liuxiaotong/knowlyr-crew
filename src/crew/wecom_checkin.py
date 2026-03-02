"""企业微信打卡通报 -- 每日自动获取考勤数据并推送到群聊."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# 北京时间
_CST = timezone(timedelta(hours=8))

# 上班时间基准（09:30 视为迟到）
_LATE_THRESHOLD_HOUR = 9
_LATE_THRESHOLD_MINUTE = 30

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


def classify_checkin(
    users: list[dict[str, Any]],
    checkin_records: list[dict[str, Any]],
    late_hour: int = _LATE_THRESHOLD_HOUR,
    late_minute: int = _LATE_THRESHOLD_MINUTE,
) -> dict[str, list[dict[str, Any]]]:
    """将打卡记录分类为正常、迟到、未打卡.

    Args:
        users: 用户列表（含 userid, name）
        checkin_records: 打卡记录
        late_hour: 迟到阈值（小时）
        late_minute: 迟到阈值（分钟）

    Returns:
        {"normal": [...], "late": [...], "absent": [...]}
        每项包含 name, userid, checkin_time(可选)
    """
    # 构建 userid -> name 映射
    user_map = {u["userid"]: u.get("name", u["userid"]) for u in users}

    # 构建 userid -> 最早上班打卡时间
    # 打卡记录可能有多条（补卡等），取最早的一条
    checkin_map: dict[str, int] = {}
    for record in checkin_records:
        uid = record.get("userid", "")
        checkin_time = record.get("checkin_time", 0)
        if uid and checkin_time:
            if uid not in checkin_map or checkin_time < checkin_map[uid]:
                checkin_map[uid] = checkin_time

    late_threshold = late_hour * 3600 + late_minute * 60  # 秒（当天零点偏移）

    normal: list[dict[str, Any]] = []
    late: list[dict[str, Any]] = []
    absent: list[dict[str, Any]] = []

    for user in users:
        uid = user["userid"]
        name = user.get("name", uid)

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

    return {"normal": normal, "late": late, "absent": absent}


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

    # 获取全员列表
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

    # 获取上班打卡数据
    logger.info("获取打卡数据: %s ~ %s", today_start.strftime("%Y-%m-%d %H:%M"), now.strftime("%H:%M"))
    checkin_records = await get_checkin_data(
        token, user_ids, start_ts, end_ts, checkin_type=1
    )
    logger.info("获取到 %d 条打卡记录", len(checkin_records))

    # 分类
    classified = classify_checkin(users, checkin_records)
    report = format_checkin_report(classified, report_date=now)

    logger.info(
        "考勤统计: 正常=%d 迟到=%d 未打卡=%d",
        len(classified["normal"]),
        len(classified["late"]),
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
