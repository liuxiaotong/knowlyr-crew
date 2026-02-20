"""agents freeze / unfreeze CLI 命令测试."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from crew.cli import main


@pytest.fixture()
def runner():
    return CliRunner()


def _make_emp(name: str, character_name: str, agent_id: int):
    """构造一个最小 Employee mock."""
    emp = MagicMock()
    emp.name = name
    emp.character_name = character_name
    emp.agent_id = agent_id
    return emp


def _make_discovery(*employees):
    """构造 discover_employees 返回值."""
    result = MagicMock()
    emp_map = {}
    for emp in employees:
        emp_map[emp.name] = emp
        if emp.character_name:
            emp_map[emp.character_name] = emp
    result.get = lambda name: emp_map.get(name)
    result.employees = emp_map
    return result


def _make_identity(agent_id: int, status: str = "active"):
    identity = MagicMock()
    identity.agent_id = agent_id
    identity.agent_status = status
    return identity


# ── freeze ────────────────────────────────────────────────────


class TestFreeze:
    """agents freeze 命令."""

    def test_freeze_single(self, runner):
        """冻结单个员工."""
        emp = _make_emp("refactor-guide", "顾然", 3054)
        discovery = _make_discovery(emp)

        with (
            patch("crew.discovery.discover_employees", return_value=discovery),
            patch(
                "crew.id_client.fetch_agent_identity", return_value=_make_identity(3054, "active")
            ),
            patch("crew.id_client.update_agent", return_value=True) as mock_update,
        ):
            result = runner.invoke(main, ["agents", "freeze", "顾然", "--force"])

        assert result.exit_code == 0
        assert "已冻结" in result.output
        mock_update.assert_called_once_with(3054, agent_status="frozen")

    def test_freeze_multiple(self, runner):
        """批量冻结多个员工."""
        emp1 = _make_emp("refactor-guide", "顾然", 3054)
        emp2 = _make_emp("pr-creator", "秦合", 3055)
        discovery = _make_discovery(emp1, emp2)

        with (
            patch("crew.discovery.discover_employees", return_value=discovery),
            patch(
                "crew.id_client.fetch_agent_identity",
                side_effect=[
                    _make_identity(3054, "active"),
                    _make_identity(3055, "active"),
                ],
            ),
            patch("crew.id_client.update_agent", return_value=True) as mock_update,
        ):
            result = runner.invoke(main, ["agents", "freeze", "顾然", "秦合", "--force"])

        assert result.exit_code == 0
        assert mock_update.call_count == 2

    def test_freeze_already_frozen(self, runner):
        """已冻结的员工跳过."""
        emp = _make_emp("refactor-guide", "顾然", 3054)
        discovery = _make_discovery(emp)

        with (
            patch("crew.discovery.discover_employees", return_value=discovery),
            patch(
                "crew.id_client.fetch_agent_identity", return_value=_make_identity(3054, "frozen")
            ),
            patch("crew.id_client.update_agent") as mock_update,
        ):
            result = runner.invoke(main, ["agents", "freeze", "顾然", "--force"])

        assert result.exit_code == 0
        assert "已处于冻结状态" in result.output
        mock_update.assert_not_called()

    def test_freeze_not_found(self, runner):
        """员工不存在时报错."""
        discovery = _make_discovery()

        with patch("crew.discovery.discover_employees", return_value=discovery):
            result = runner.invoke(main, ["agents", "freeze", "不存在", "--force"])

        assert result.exit_code != 0
        assert "未找到员工" in result.output

    def test_freeze_no_agent_id(self, runner):
        """员工未绑定 Agent 时报错."""
        emp = _make_emp("test", "测试", None)
        discovery = _make_discovery(emp)

        with patch("crew.discovery.discover_employees", return_value=discovery):
            result = runner.invoke(main, ["agents", "freeze", "测试", "--force"])

        assert result.exit_code != 0
        assert "未绑定 Agent" in result.output

    def test_freeze_failure(self, runner):
        """冻结 API 失败时提示."""
        emp = _make_emp("refactor-guide", "顾然", 3054)
        discovery = _make_discovery(emp)

        with (
            patch("crew.discovery.discover_employees", return_value=discovery),
            patch(
                "crew.id_client.fetch_agent_identity", return_value=_make_identity(3054, "active")
            ),
            patch("crew.id_client.update_agent", return_value=False),
        ):
            result = runner.invoke(main, ["agents", "freeze", "顾然", "--force"])

        assert result.exit_code == 0
        assert "冻结失败" in result.output


# ── unfreeze ──────────────────────────────────────────────────


class TestUnfreeze:
    """agents unfreeze 命令."""

    def test_unfreeze_single(self, runner):
        """解冻单个员工."""
        emp = _make_emp("refactor-guide", "顾然", 3054)
        discovery = _make_discovery(emp)

        with (
            patch("crew.discovery.discover_employees", return_value=discovery),
            patch(
                "crew.id_client.fetch_agent_identity", return_value=_make_identity(3054, "frozen")
            ),
            patch("crew.id_client.update_agent", return_value=True) as mock_update,
        ):
            result = runner.invoke(main, ["agents", "unfreeze", "顾然", "--force"])

        assert result.exit_code == 0
        assert "已解冻" in result.output
        mock_update.assert_called_once_with(3054, agent_status="active")

    def test_unfreeze_already_active(self, runner):
        """已活跃的员工跳过."""
        emp = _make_emp("refactor-guide", "顾然", 3054)
        discovery = _make_discovery(emp)

        with (
            patch("crew.discovery.discover_employees", return_value=discovery),
            patch(
                "crew.id_client.fetch_agent_identity", return_value=_make_identity(3054, "active")
            ),
            patch("crew.id_client.update_agent") as mock_update,
        ):
            result = runner.invoke(main, ["agents", "unfreeze", "顾然", "--force"])

        assert result.exit_code == 0
        assert "已处于活跃状态" in result.output
        mock_update.assert_not_called()


# ── sync 兼容 ─────────────────────────────────────────────────


class TestSyncFrozenCompat:
    """sync 不覆盖 frozen 状态."""

    def test_skip_frozen_on_disable(self, tmp_path):
        """sync 不会把 frozen 的员工标为 inactive."""
        from crew.sync import sync_all

        with (
            patch(
                "crew.sync.list_agents",
                return_value=[
                    {"id": 3054, "nickname": "顾然", "status": "frozen"},
                ],
            ),
            patch("crew.sync.update_agent", return_value=True) as mock_update,
        ):
            report = sync_all(tmp_path, push=True, pull=False)

        assert len(report.disabled) == 0
        mock_update.assert_not_called()
