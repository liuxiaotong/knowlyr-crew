"""企业微信打卡通报测试 -- classify / format / run_checkin_report."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
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
            records.append({
                "userid": uid,
                "checkin_time": int(dt.timestamp()),
            })
        return records

    def test_all_normal(self):
        from crew.wecom_checkin import classify_checkin

        users = self._make_users(["Alice", "Bob"])
        records = self._make_records([("uid_0", "08:55"), ("uid_1", "09:10")])

        result = classify_checkin(users, records)
        assert len(result["normal"]) == 2
        assert len(result["late"]) == 0
        assert len(result["absent"]) == 0

    def test_late_detection(self):
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
        assert result == {"normal": [], "late": [], "absent": []}

    def test_custom_threshold(self):
        from crew.wecom_checkin import classify_checkin

        users = self._make_users(["Alice"])
        records = self._make_records([("uid_0", "09:05")])

        # 阈值设为 09:00 -> 09:05 算迟到
        result = classify_checkin(users, records, late_hour=9, late_minute=0)
        assert len(result["late"]) == 1

        # 阈值设为 09:10 -> 09:05 算正常
        result = classify_checkin(users, records, late_hour=9, late_minute=10)
        assert len(result["normal"]) == 1

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


# ── 完整流程测试 ──


class TestRunCheckinReport:
    """run_checkin_report 端到端流程."""

    @pytest.mark.asyncio
    async def test_full_flow_dry_run(self):
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
            patch("crew.wecom_checkin.get_checkin_data", new_callable=AsyncMock) as mock_checkin,
        ):
            mock_users.return_value = users
            mock_checkin.return_value = checkin_records

            report = await run_checkin_report(
                corp_id="test_corp",
                secret="test_secret",
                dry_run=True,
            )

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
        checkin_records = [
            {"userid": "u1", "checkin_time": int(today.replace(hour=9, minute=0).timestamp())},
        ]

        with (
            patch("crew.wecom.WecomTokenManager", return_value=mock_token_mgr),
            patch("crew.wecom_checkin.get_department_users", new_callable=AsyncMock) as mock_users,
            patch("crew.wecom_checkin.get_checkin_data", new_callable=AsyncMock) as mock_checkin,
            patch("crew.wecom.send_wecom_group_text", new_callable=AsyncMock) as mock_send,
        ):
            mock_users.return_value = users
            mock_checkin.return_value = checkin_records
            mock_send.return_value = {"errcode": 0, "errmsg": "ok"}

            report = await run_checkin_report(
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
