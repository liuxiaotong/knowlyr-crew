"""企业微信打卡通报测试 -- classify / format / rules / leave / run_checkin_report."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_CST = timezone(timedelta(hours=8))


# ── 分类逻辑测试 ──


class TestClassifyCheckin:
    """classify_checkin 正常 / 迟到 / 未打卡分类."""

    def _make_users(self, names: list[str]) -> list[dict]:
        return [{"userid": f"uid_{i}", "name": n} for i, n in enumerate(names)]

    def _make_records(self, data: list[tuple[str, str]]) -> list[dict]:
        """data: [(userid, "HH:MM"), ...] -> checkin records."""
        records = []
        today = datetime.now(_CST).replace(hour=0, minute=0, second=0, microsecond=0)
        for uid, time_str in data:
            h, m = map(int, time_str.split(":"))
            dt = today.replace(hour=h, minute=m)
            records.append(
                {
                    "userid": uid,
                    "checkin_time": int(dt.timestamp()),
                }
            )
        return records

    def test_all_normal_fallback(self):
        """无 rules 时回退到 09:30 阈值."""
        from crew.wecom_checkin import classify_checkin

        users = self._make_users(["Alice", "Bob"])
        records = self._make_records([("uid_0", "08:55"), ("uid_1", "09:10")])

        result = classify_checkin(users, records)
        assert len(result["normal"]) == 2
        assert len(result["late"]) == 0
        assert len(result["absent"]) == 0

    def test_late_detection_fallback(self):
        """无 rules 时用 09:30 fallback 判断迟到."""
        from crew.wecom_checkin import classify_checkin

        users = self._make_users(["Alice", "Bob"])
        records = self._make_records([("uid_0", "09:00"), ("uid_1", "09:45")])

        result = classify_checkin(users, records)
        assert len(result["normal"]) == 1
        assert result["normal"][0]["name"] == "Alice"
        assert len(result["late"]) == 1
        assert result["late"][0]["name"] == "Bob"
        assert result["late"][0]["checkin_time"] == "09:45"

    def test_absent_detection(self):
        from crew.wecom_checkin import classify_checkin

        users = self._make_users(["Alice", "Bob", "Charlie"])
        records = self._make_records([("uid_0", "09:00")])

        result = classify_checkin(users, records)
        assert len(result["normal"]) == 1
        assert len(result["absent"]) == 2
        absent_names = {p["name"] for p in result["absent"]}
        assert absent_names == {"Bob", "Charlie"}

    def test_empty_users(self):
        from crew.wecom_checkin import classify_checkin

        result = classify_checkin([], [])
        assert result == {"normal": [], "late": [], "on_leave": [], "absent": []}

    def test_with_rules_per_user_threshold(self):
        """每个人使用自己的 work_sec + flex_time 判断迟到."""
        from crew.wecom_checkin import classify_checkin

        users = self._make_users(["Alice", "Bob"])
        # Alice 09:35, Bob 09:35
        records = self._make_records([("uid_0", "09:35"), ("uid_1", "09:35")])

        rules = {
            # Alice: 09:00 上班 + 1h 弹性 = 10:00 阈值 -> 09:35 正常
            "uid_0": {"work_sec": 9 * 3600, "flex_time": 3600, "groupname": "上海"},
            # Bob: 09:30 上班 + 0 弹性 = 09:30 阈值 -> 09:35 迟到
            "uid_1": {"work_sec": 9 * 3600 + 1800, "flex_time": 0, "groupname": "北京"},
        }

        result = classify_checkin(users, records, rules=rules)
        assert len(result["normal"]) == 1
        assert result["normal"][0]["name"] == "Alice"
        assert len(result["late"]) == 1
        assert result["late"][0]["name"] == "Bob"

    def test_with_rules_flex_time(self):
        """弹性时间内不算迟到."""
        from crew.wecom_checkin import classify_checkin

        users = self._make_users(["Alice"])
        # Alice 10:25 打卡
        records = self._make_records([("uid_0", "10:25")])

        rules = {
            # 09:30 + 1h弹性 = 10:30 阈值 -> 10:25 正常
            "uid_0": {"work_sec": 34200, "flex_time": 3600, "groupname": "上海"},
        }

        result = classify_checkin(users, records, rules=rules)
        assert len(result["normal"]) == 1
        assert len(result["late"]) == 0

    def test_with_rules_beyond_flex(self):
        """超过弹性时间算迟到."""
        from crew.wecom_checkin import classify_checkin

        users = self._make_users(["Alice"])
        # Alice 10:35 打卡
        records = self._make_records([("uid_0", "10:35")])

        rules = {
            # 09:30 + 1h弹性 = 10:30 阈值 -> 10:35 迟到
            "uid_0": {"work_sec": 34200, "flex_time": 3600, "groupname": "上海"},
        }

        result = classify_checkin(users, records, rules=rules)
        assert len(result["normal"]) == 0
        assert len(result["late"]) == 1

    def test_multiple_records_takes_earliest(self):
        """同一用户多条打卡记录（补卡等），取最早的."""
        from crew.wecom_checkin import classify_checkin

        users = self._make_users(["Alice"])
        today = datetime.now(_CST).replace(hour=0, minute=0, second=0, microsecond=0)
        records = [
            {"userid": "uid_0", "checkin_time": int(today.replace(hour=10, minute=0).timestamp())},
            {"userid": "uid_0", "checkin_time": int(today.replace(hour=8, minute=30).timestamp())},
        ]

        result = classify_checkin(users, records)
        # 取最早的 08:30 -> 正常
        assert len(result["normal"]) == 1
        assert result["normal"][0]["checkin_time"] == "08:30"

    def test_on_leave_user(self):
        """请假的人归入 on_leave，不出现在 absent."""
        from crew.wecom_checkin import classify_checkin

        users = self._make_users(["Alice", "Bob", "Charlie"])
        records = self._make_records([("uid_0", "09:00")])

        leave_users = {"uid_1"}  # Bob 请假

        result = classify_checkin(users, records, leave_users=leave_users)
        assert len(result["normal"]) == 1
        assert result["normal"][0]["name"] == "Alice"
        assert len(result["on_leave"]) == 1
        assert result["on_leave"][0]["name"] == "Bob"
        assert len(result["absent"]) == 1
        assert result["absent"][0]["name"] == "Charlie"

    def test_on_leave_overrides_checkin(self):
        """即使请假的人有打卡记录，也归入 on_leave."""
        from crew.wecom_checkin import classify_checkin

        users = self._make_users(["Alice"])
        records = self._make_records([("uid_0", "09:00")])
        leave_users = {"uid_0"}

        result = classify_checkin(users, records, leave_users=leave_users)
        assert len(result["on_leave"]) == 1
        assert len(result["normal"]) == 0

    def test_no_leave_users_backward_compat(self):
        """不传 leave_users 时，行为和之前完全一致（返回含 on_leave 空列表）."""
        from crew.wecom_checkin import classify_checkin

        users = self._make_users(["Alice"])
        records = self._make_records([("uid_0", "09:00")])

        result = classify_checkin(users, records)
        assert "on_leave" in result
        assert len(result["on_leave"]) == 0
        assert len(result["normal"]) == 1

    def test_haikou_flex_0946_is_normal(self):
        """海口职场 flex_time=1800: 09:46 打卡应为正常（阈值 09:30+1800s=10:00）."""
        from crew.wecom_checkin import classify_checkin

        users = self._make_users(["吴宜莹"])
        records = self._make_records([("uid_0", "09:46")])

        rules = {
            # 海口职场：09:30 上班 + 1800s 弹性 = 10:00 阈值
            "uid_0": {"work_sec": 34200, "flex_time": 1800, "groupname": "海口职场"},
        }

        result = classify_checkin(users, records, rules=rules)
        assert len(result["normal"]) == 1
        assert result["normal"][0]["name"] == "吴宜莹"
        assert len(result["late"]) == 0


# ── 格式化测试 ──


class TestFormatCheckinReport:
    """format_checkin_report 通报消息格式."""

    def test_basic_format(self):
        from crew.wecom_checkin import format_checkin_report

        classified = {
            "normal": [
                {"name": "Alice", "userid": "u1", "checkin_time": "08:55"},
                {"name": "Bob", "userid": "u2", "checkin_time": "09:01"},
            ],
            "late": [
                {"name": "Charlie", "userid": "u3", "checkin_time": "09:45"},
            ],
            "on_leave": [],
            "absent": [
                {"name": "David", "userid": "u4"},
            ],
        }

        report_date = datetime(2026, 3, 2, 10, 30, tzinfo=_CST)
        report = format_checkin_report(classified, report_date=report_date)

        assert "3/2" in report
        assert "正常打卡（2人）" in report
        assert "Alice 08:55" in report
        assert "迟到（1人）" in report
        assert "Charlie 09:45" in report
        assert "请假（0人）" in report
        assert "未打卡（1人）" in report
        assert "David" in report

    def test_all_normal_no_late_no_absent(self):
        from crew.wecom_checkin import format_checkin_report

        classified = {
            "normal": [{"name": "Alice", "userid": "u1", "checkin_time": "09:00"}],
            "late": [],
            "absent": [],
        }

        report = format_checkin_report(classified)
        assert "正常打卡（1人）" in report
        assert "迟到（0人）" in report
        assert "未打卡（0人）" in report

    def test_empty_report(self):
        from crew.wecom_checkin import format_checkin_report

        classified = {"normal": [], "late": [], "absent": []}
        report = format_checkin_report(classified)
        assert "考勤通报" in report

    def test_chinese_names(self):
        """通报中显示中文名."""
        from crew.wecom_checkin import format_checkin_report

        classified = {
            "normal": [
                {"name": "王瑶", "userid": "WangYao", "checkin_time": "09:00"},
            ],
            "late": [
                {"name": "张三", "userid": "ZhangSan", "checkin_time": "10:05"},
            ],
            "absent": [
                {"name": "李四", "userid": "LiSi"},
            ],
        }

        report = format_checkin_report(classified)
        assert "王瑶 09:00" in report
        assert "张三 10:05" in report
        assert "李四" in report
        # 不应出现 userid
        assert "WangYao" not in report
        assert "ZhangSan" not in report
        assert "LiSi" not in report

    def test_with_on_leave(self):
        """请假区域在迟到和未打卡之间."""
        from crew.wecom_checkin import format_checkin_report

        classified = {
            "normal": [
                {"name": "Alice", "userid": "u1", "checkin_time": "09:00"},
            ],
            "late": [],
            "on_leave": [
                {"name": "王瑶", "userid": "WangYao"},
                {"name": "张三", "userid": "ZhangSan"},
            ],
            "absent": [
                {"name": "李四", "userid": "LiSi"},
            ],
        }

        report = format_checkin_report(classified)
        assert "请假（2人）" in report
        assert "王瑶" in report
        assert "张三" in report
        # 请假区域在迟到之后、未打卡之前
        late_pos = report.index("迟到")
        leave_pos = report.index("请假")
        absent_pos = report.index("未打卡")
        assert late_pos < leave_pos < absent_pos

    def test_on_leave_without_key_backward_compat(self):
        """旧格式（没有 on_leave key）也能正常渲染."""
        from crew.wecom_checkin import format_checkin_report

        classified = {
            "normal": [{"name": "Alice", "userid": "u1", "checkin_time": "09:00"}],
            "late": [],
            "absent": [],
        }

        report = format_checkin_report(classified)
        assert "请假（0人）" in report


# ── API 调用测试 ──


class TestGetDepartmentUsers:
    """get_department_users 接口调用."""

    @pytest.mark.asyncio
    async def test_success(self):
        from crew.wecom_checkin import get_department_users

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "errcode": 0,
            "errmsg": "ok",
            "userlist": [
                {"userid": "u1", "name": "Alice", "department": [1]},
                {"userid": "u2", "name": "Bob", "department": [1]},
            ],
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            users = await get_department_users("test_token")

        assert len(users) == 2
        assert users[0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_error(self):
        from crew.wecom_checkin import get_department_users

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "errcode": 60011,
            "errmsg": "no privilege to access",
        }

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="获取用户列表失败"):
                await get_department_users("bad_token")


class TestGetCheckinData:
    """get_checkin_data 接口调用."""

    @pytest.mark.asyncio
    async def test_success(self):
        from crew.wecom_checkin import get_checkin_data

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "errcode": 0,
            "errmsg": "ok",
            "checkindata": [
                {"userid": "u1", "checkin_time": 1709341200, "checkin_type": "上班打卡"},
            ],
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            records = await get_checkin_data("token", ["u1"], 1709280000, 1709366400)

        assert len(records) == 1
        assert records[0]["userid"] == "u1"

    @pytest.mark.asyncio
    async def test_batch_split(self):
        """超过 100 人时自动分批."""
        from crew.wecom_checkin import get_checkin_data

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "errcode": 0,
            "errmsg": "ok",
            "checkindata": [{"userid": "u1", "checkin_time": 1709341200}],
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        user_ids = [f"u{i}" for i in range(150)]

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            records = await get_checkin_data("token", user_ids, 1709280000, 1709366400)

        # 150 人 -> 2 批 (100 + 50)
        assert mock_client.post.call_count == 2
        # 每批返回 1 条 -> 总共 2 条
        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_error(self):
        from crew.wecom_checkin import get_checkin_data

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "errcode": 301023,
            "errmsg": "not checkin app",
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="获取打卡数据失败"):
                await get_checkin_data("token", ["u1"], 100, 200)


# ── 打卡规则测试 ──


class TestGetCheckinRules:
    """get_checkin_rules 接口调用."""

    def _mock_option_response(self, info: list[dict]) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = {"errcode": 0, "errmsg": "ok", "info": info}
        return resp

    @pytest.mark.asyncio
    async def test_success_basic(self):
        from crew.wecom_checkin import get_checkin_rules

        info = [
            {
                "userid": "WangYao",
                "group": {
                    "groupname": "上海职场",
                    "checkindate": [
                        {
                            "workdays": [1, 2, 3, 4, 5],
                            "checkintime": [{"work_sec": 34200, "off_work_sec": 66600}],
                            "late_rule": {"onwork_flex_time": 3600},
                            "allow_flex": True,
                        }
                    ],
                },
            }
        ]

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=self._mock_option_response(info))

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            rules = await get_checkin_rules("token", ["WangYao"], 1709280000)

        assert "WangYao" in rules
        assert rules["WangYao"]["work_sec"] == 34200
        assert rules["WangYao"]["flex_time"] == 3600
        assert rules["WangYao"]["groupname"] == "上海职场"

    @pytest.mark.asyncio
    async def test_no_flex(self):
        """allow_flex=False 时 flex_time 应为 0."""
        from crew.wecom_checkin import get_checkin_rules

        info = [
            {
                "userid": "u1",
                "group": {
                    "groupname": "北京",
                    "checkindate": [
                        {
                            "checkintime": [{"work_sec": 32400}],
                            "allow_flex": False,
                        }
                    ],
                },
            }
        ]

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=self._mock_option_response(info))

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            rules = await get_checkin_rules("token", ["u1"], 1709280000)

        assert rules["u1"]["flex_time"] == 0

    @pytest.mark.asyncio
    async def test_missing_user_excluded(self):
        """API 不返回的人 = 不需要打卡."""
        from crew.wecom_checkin import get_checkin_rules

        # 只返回 u1，不返回 u2
        info = [
            {
                "userid": "u1",
                "group": {
                    "groupname": "上海",
                    "checkindate": [
                        {
                            "checkintime": [{"work_sec": 34200}],
                            "allow_flex": False,
                        }
                    ],
                },
            }
        ]

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=self._mock_option_response(info))

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            rules = await get_checkin_rules("token", ["u1", "u2"], 1709280000)

        assert "u1" in rules
        assert "u2" not in rules

    @pytest.mark.asyncio
    async def test_api_error_returns_empty(self):
        """API 返回错误码时返回空 dict."""
        from crew.wecom_checkin import get_checkin_rules

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"errcode": 301023, "errmsg": "not checkin app"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            rules = await get_checkin_rules("token", ["u1"], 1709280000)

        assert rules == {}

    @pytest.mark.asyncio
    async def test_request_exception_returns_empty(self):
        """网络异常时返回空 dict."""
        from crew.wecom_checkin import get_checkin_rules

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ConnectionError("timeout"))

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            rules = await get_checkin_rules("token", ["u1"], 1709280000)

        assert rules == {}

    @pytest.mark.asyncio
    async def test_batch_split(self):
        """超过 100 人时自动分批."""
        from crew.wecom_checkin import get_checkin_rules

        info = [
            {
                "userid": "u0",
                "group": {
                    "groupname": "G",
                    "checkindate": [{"checkintime": [{"work_sec": 34200}], "allow_flex": False}],
                },
            }
        ]

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=self._mock_option_response(info))

        user_ids = [f"u{i}" for i in range(150)]

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            await get_checkin_rules("token", user_ids, 1709280000)

        # 150 人 -> 2 批
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_max_allow_arrive_late(self):
        """海口职场：onwork_flex_time=0 但 max_allow_arrive_late=1800，flex_time 应为 1800."""
        from crew.wecom_checkin import get_checkin_rules

        info = [
            {
                "userid": "WuYiYing",
                "group": {
                    "groupname": "海口职场",
                    "checkindate": [
                        {
                            "workdays": [1, 2, 3, 4, 5],
                            "checkintime": [{"work_sec": 34200, "off_work_sec": 66600}],
                            "allow_flex": True,
                            "max_allow_arrive_early": 1800,
                            "max_allow_arrive_late": 1800,
                            "late_rule": {
                                "onwork_flex_time": 0,
                                "timerules": [{"offwork_after_time": 0, "onwork_flex_time": 0}],
                            },
                        }
                    ],
                },
            }
        ]

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=self._mock_option_response(info))

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            rules = await get_checkin_rules("token", ["WuYiYing"], 1709280000)

        assert "WuYiYing" in rules
        assert rules["WuYiYing"]["flex_time"] == 1800
        assert rules["WuYiYing"]["groupname"] == "海口职场"


# ── 完整流程测试 ──


class TestRunCheckinReport:
    """run_checkin_report 端到端流程."""

    @pytest.mark.asyncio
    async def test_full_flow_with_rules(self):
        """有打卡规则时，排除无规则的人，按个人阈值判断."""
        from crew.wecom_checkin import run_checkin_report

        today = datetime.now(_CST).replace(hour=0, minute=0, second=0, microsecond=0)

        mock_token_mgr = AsyncMock()
        mock_token_mgr.get_token = AsyncMock(return_value="test_token")

        # 3 个人，u3 (LiuKai) 没有打卡规则
        users = [
            {"userid": "u1", "name": "王瑶", "department": [1]},
            {"userid": "u2", "name": "张三", "department": [1]},
            {"userid": "u3", "name": "刘凯", "department": [1]},
        ]

        # 打卡规则：只返回 u1, u2（u3 不需要打卡）
        rules = {
            "u1": {"work_sec": 34200, "flex_time": 3600, "groupname": "上海"},
            "u2": {"work_sec": 34200, "flex_time": 0, "groupname": "北京"},
        }

        checkin_records = [
            # 王瑶 09:35 -> work_sec=34200 + flex=3600 = 10:30 阈值 -> 正常
            {"userid": "u1", "checkin_time": int(today.replace(hour=9, minute=35).timestamp())},
            # 张三 09:35 -> work_sec=34200 + flex=0 = 09:30 阈值 -> 迟到
            {"userid": "u2", "checkin_time": int(today.replace(hour=9, minute=35).timestamp())},
        ]

        with (
            patch("crew.wecom.WecomTokenManager", return_value=mock_token_mgr),
            patch("crew.wecom_checkin.get_department_users", new_callable=AsyncMock) as mock_users,
            patch("crew.wecom_checkin.get_checkin_rules", new_callable=AsyncMock) as mock_rules,
            patch("crew.wecom_checkin.get_checkin_data", new_callable=AsyncMock) as mock_checkin,
            patch("crew.wecom_checkin.get_leave_users", new_callable=AsyncMock) as mock_leave,
        ):
            mock_users.return_value = users
            mock_rules.return_value = rules
            mock_checkin.return_value = checkin_records
            mock_leave.return_value = set()

            report = await run_checkin_report(
                corp_id="test_corp",
                secret="test_secret",
                dry_run=True,
            )

        # 王瑶正常（弹性范围内）
        assert "正常打卡（1人）" in report
        assert "王瑶 09:35" in report
        # 张三迟到（无弹性）
        assert "迟到（1人）" in report
        assert "张三 09:35" in report
        # 刘凯不应出现（无打卡规则）
        assert "刘凯" not in report
        # 未打卡应为 0 人
        assert "未打卡（0人）" in report

    @pytest.mark.asyncio
    async def test_fallback_when_rules_fail(self):
        """打卡规则获取失败时，回退到全员 + 09:30 阈值."""
        from crew.wecom_checkin import run_checkin_report

        today = datetime.now(_CST).replace(hour=0, minute=0, second=0, microsecond=0)

        mock_token_mgr = AsyncMock()
        mock_token_mgr.get_token = AsyncMock(return_value="test_token")

        users = [
            {"userid": "u1", "name": "Alice", "department": [1]},
            {"userid": "u2", "name": "Bob", "department": [1]},
            {"userid": "u3", "name": "Charlie", "department": [1]},
        ]

        checkin_records = [
            {"userid": "u1", "checkin_time": int(today.replace(hour=8, minute=55).timestamp())},
            {"userid": "u2", "checkin_time": int(today.replace(hour=9, minute=45).timestamp())},
            # u3 没有打卡记录
        ]

        with (
            patch("crew.wecom.WecomTokenManager", return_value=mock_token_mgr),
            patch("crew.wecom_checkin.get_department_users", new_callable=AsyncMock) as mock_users,
            patch("crew.wecom_checkin.get_checkin_rules", new_callable=AsyncMock) as mock_rules,
            patch("crew.wecom_checkin.get_checkin_data", new_callable=AsyncMock) as mock_checkin,
            patch("crew.wecom_checkin.get_leave_users", new_callable=AsyncMock) as mock_leave,
        ):
            mock_users.return_value = users
            mock_rules.return_value = {}  # 规则获取失败
            mock_checkin.return_value = checkin_records
            mock_leave.return_value = set()

            report = await run_checkin_report(
                corp_id="test_corp",
                secret="test_secret",
                dry_run=True,
            )

        # 回退到全员 + 09:30 阈值
        assert "正常打卡（1人）" in report
        assert "Alice 08:55" in report
        assert "迟到（1人）" in report
        assert "Bob 09:45" in report
        assert "未打卡（1人）" in report
        assert "Charlie" in report

    @pytest.mark.asyncio
    async def test_full_flow_sends_to_group(self):
        from crew.wecom_checkin import run_checkin_report

        today = datetime.now(_CST).replace(hour=0, minute=0, second=0, microsecond=0)

        mock_token_mgr = AsyncMock()
        mock_token_mgr.get_token = AsyncMock(return_value="test_token")

        users = [{"userid": "u1", "name": "Alice", "department": [1]}]
        rules = {"u1": {"work_sec": 34200, "flex_time": 0, "groupname": "上海"}}
        checkin_records = [
            {"userid": "u1", "checkin_time": int(today.replace(hour=9, minute=0).timestamp())},
        ]

        with (
            patch("crew.wecom.WecomTokenManager", return_value=mock_token_mgr),
            patch("crew.wecom_checkin.get_department_users", new_callable=AsyncMock) as mock_users,
            patch("crew.wecom_checkin.get_checkin_rules", new_callable=AsyncMock) as mock_rules,
            patch("crew.wecom_checkin.get_checkin_data", new_callable=AsyncMock) as mock_checkin,
            patch("crew.wecom_checkin.get_leave_users", new_callable=AsyncMock) as mock_leave,
            patch("crew.wecom.send_wecom_group_text", new_callable=AsyncMock) as mock_send,
        ):
            mock_users.return_value = users
            mock_rules.return_value = rules
            mock_checkin.return_value = checkin_records
            mock_leave.return_value = set()
            mock_send.return_value = {"errcode": 0, "errmsg": "ok"}

            await run_checkin_report(
                corp_id="test_corp",
                secret="test_secret",
                dry_run=False,
            )

        # 验证群消息发送
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert call_args[0][1] == "wrlzQdBgAAjyn00rCdxEegaHe8vRY0FA"  # chat_id

    @pytest.mark.asyncio
    async def test_no_users_returns_warning(self):
        from crew.wecom_checkin import run_checkin_report

        mock_token_mgr = AsyncMock()
        mock_token_mgr.get_token = AsyncMock(return_value="test_token")

        with (
            patch("crew.wecom.WecomTokenManager", return_value=mock_token_mgr),
            patch("crew.wecom_checkin.get_department_users", new_callable=AsyncMock) as mock_users,
        ):
            mock_users.return_value = []

            report = await run_checkin_report(
                corp_id="test_corp",
                secret="test_secret",
                dry_run=True,
            )

        assert "未获取到任何用户" in report

    @pytest.mark.asyncio
    async def test_missing_config_raises(self):
        from crew.wecom_checkin import run_checkin_report

        with patch("crew.wecom.load_wecom_config") as mock_config:
            mock_config.return_value = MagicMock(corp_id="", secret="")

            with pytest.raises(RuntimeError, match="CorpID 或 Secret 未配置"):
                await run_checkin_report()

    @pytest.mark.asyncio
    async def test_full_flow_with_leave(self):
        """有请假人员时，报告中显示请假而非未打卡."""
        from crew.wecom_checkin import run_checkin_report

        today = datetime.now(_CST).replace(hour=0, minute=0, second=0, microsecond=0)

        mock_token_mgr = AsyncMock()
        mock_token_mgr.get_token = AsyncMock(return_value="test_token")

        users = [
            {"userid": "u1", "name": "王瑶", "department": [1]},
            {"userid": "u2", "name": "张三", "department": [1]},
            {"userid": "u3", "name": "李四", "department": [1]},
        ]

        rules = {
            "u1": {"work_sec": 34200, "flex_time": 0, "groupname": "上海"},
            "u2": {"work_sec": 34200, "flex_time": 0, "groupname": "上海"},
            "u3": {"work_sec": 34200, "flex_time": 0, "groupname": "上海"},
        }

        checkin_records = [
            # 张三正常打卡
            {"userid": "u2", "checkin_time": int(today.replace(hour=9, minute=0).timestamp())},
            # 王瑶和李四没有打卡记录
        ]

        with (
            patch("crew.wecom.WecomTokenManager", return_value=mock_token_mgr),
            patch("crew.wecom_checkin.get_department_users", new_callable=AsyncMock) as mock_users,
            patch("crew.wecom_checkin.get_checkin_rules", new_callable=AsyncMock) as mock_rules,
            patch("crew.wecom_checkin.get_checkin_data", new_callable=AsyncMock) as mock_checkin,
            patch("crew.wecom_checkin.get_leave_users", new_callable=AsyncMock) as mock_leave,
        ):
            mock_users.return_value = users
            mock_rules.return_value = rules
            mock_checkin.return_value = checkin_records
            mock_leave.return_value = {"u1"}  # 王瑶请假

            report = await run_checkin_report(
                corp_id="test_corp",
                secret="test_secret",
                dry_run=True,
            )

        # 张三正常
        assert "正常打卡（1人）" in report
        assert "张三 09:00" in report
        # 王瑶请假（不是未打卡）
        assert "请假（1人）" in report
        assert "王瑶" in report
        # 李四未打卡
        assert "未打卡（1人）" in report
        assert "李四" in report


# ── 请假审批 API 测试 ──


class TestGetLeaveUsers:
    """get_leave_users 审批 API 调用."""

    @pytest.mark.asyncio
    async def test_success_with_leave(self):
        """正常获取请假人员."""
        from crew.wecom_checkin import get_leave_users

        # getapprovalinfo 返回 2 个审批单
        approval_info_resp = MagicMock()
        approval_info_resp.json.return_value = {
            "errcode": 0,
            "sp_no_list": ["202503020001", "202503020002"],
            "new_next_cursor": 0,
        }

        # getapprovaldetail 返回：第一个是请假，第二个是报销
        detail_resp_leave = MagicMock()
        detail_resp_leave.json.return_value = {
            "errcode": 0,
            "info": {
                "sp_no": "202503020001",
                "sp_name": "请假",
                "sp_status": 2,
                "applyer": {"userid": "WangYao"},
            },
        }

        detail_resp_expense = MagicMock()
        detail_resp_expense.json.return_value = {
            "errcode": 0,
            "info": {
                "sp_no": "202503020002",
                "sp_name": "报销",
                "sp_status": 2,
                "applyer": {"userid": "ZhangSan"},
            },
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=[approval_info_resp, detail_resp_leave, detail_resp_expense]
        )

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            result = await get_leave_users("token", 1709280000, 1709366400)

        assert result == {"WangYao"}

    @pytest.mark.asyncio
    async def test_pagination(self):
        """审批单分页获取."""
        from crew.wecom_checkin import get_leave_users

        # 第一页
        page1_resp = MagicMock()
        page1_resp.json.return_value = {
            "errcode": 0,
            "sp_no_list": ["sp001"],
            "new_next_cursor": 5,
        }

        # 第二页
        page2_resp = MagicMock()
        page2_resp.json.return_value = {
            "errcode": 0,
            "sp_no_list": ["sp002"],
            "new_next_cursor": 0,
        }

        # 两个审批详情都是请假
        detail1 = MagicMock()
        detail1.json.return_value = {
            "errcode": 0,
            "info": {"sp_name": "请假", "sp_status": 2, "applyer": {"userid": "u1"}},
        }
        detail2 = MagicMock()
        detail2.json.return_value = {
            "errcode": 0,
            "info": {"sp_name": "请假", "sp_status": 2, "applyer": {"userid": "u2"}},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[page1_resp, page2_resp, detail1, detail2])

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            result = await get_leave_users("token", 100, 200)

        assert result == {"u1", "u2"}
        # 4 次 POST：2 次 getapprovalinfo + 2 次 getapprovaldetail
        assert mock_client.post.call_count == 4

    @pytest.mark.asyncio
    async def test_empty_approval_list(self):
        """没有审批单时返回空 set."""
        from crew.wecom_checkin import get_leave_users

        resp = MagicMock()
        resp.json.return_value = {
            "errcode": 0,
            "sp_no_list": [],
            "new_next_cursor": 0,
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=resp)

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            result = await get_leave_users("token", 100, 200)

        assert result == set()

    @pytest.mark.asyncio
    async def test_approval_info_api_error(self):
        """getapprovalinfo API 错误时返回空 set."""
        from crew.wecom_checkin import get_leave_users

        resp = MagicMock()
        resp.json.return_value = {"errcode": 40001, "errmsg": "invalid credential"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=resp)

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            result = await get_leave_users("token", 100, 200)

        assert result == set()

    @pytest.mark.asyncio
    async def test_approval_detail_error_skips(self):
        """单个 getapprovaldetail 失败时跳过，不阻断其他."""
        from crew.wecom_checkin import get_leave_users

        info_resp = MagicMock()
        info_resp.json.return_value = {
            "errcode": 0,
            "sp_no_list": ["sp001", "sp002"],
            "new_next_cursor": 0,
        }

        # 第一个详情失败
        detail_err = MagicMock()
        detail_err.json.return_value = {"errcode": 40001, "errmsg": "invalid"}

        # 第二个详情成功（请假）
        detail_ok = MagicMock()
        detail_ok.json.return_value = {
            "errcode": 0,
            "info": {"sp_name": "请假", "sp_status": 2, "applyer": {"userid": "u2"}},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[info_resp, detail_err, detail_ok])

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            result = await get_leave_users("token", 100, 200)

        assert result == {"u2"}

    @pytest.mark.asyncio
    async def test_network_exception_returns_empty(self):
        """网络异常时返回空 set."""
        from crew.wecom_checkin import get_leave_users

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ConnectionError("timeout"))

        with patch("crew.wecom.get_wecom_client", return_value=mock_client):
            result = await get_leave_users("token", 100, 200)

        assert result == set()
