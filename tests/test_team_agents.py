"""测试 GET /api/team/agents 端点."""

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("starlette")

from starlette.testclient import TestClient

from crew.models import Employee
from crew.webhook import create_webhook_app
from crew.webhook_config import WebhookConfig

TOKEN = "test-token-123"


def _make_employee(
    name: str,
    character_name: str = "",
    display_name: str = "",
    agent_id: int | None = None,
    agent_status: str = "active",
    tags: list[str] | None = None,
    source_path: Path | None = None,
) -> Employee:
    """构造一个最小化的 Employee 对象用于测试."""
    return Employee(
        name=name,
        character_name=character_name or name,
        display_name=display_name or name,
        description=f"{name} 的描述",
        tags=tags or [],
        agent_id=agent_id,
        agent_status=agent_status,
        body="prompt body",
        source_path=source_path,
    )


def _make_client():
    app = create_webhook_app(
        project_dir=Path("/tmp/test-team-agents"),
        token=TOKEN,
        config=WebhookConfig(),
    )
    return TestClient(app)


class TestTeamAgents:
    """GET /api/team/agents 端点测试."""

    def test_returns_only_active_agents(self):
        """只返回 agent_status=active 的员工."""
        from crew.discovery import DiscoveryResult

        employees = {
            "eng-a": _make_employee("eng-a", "张三", "工程师A", 3001, "active", ["Python"]),
            "eng-b": _make_employee("eng-b", "李四", "工程师B", 3002, "frozen", ["Go"]),
            "eng-c": _make_employee("eng-c", "王五", "工程师C", 3003, "active", ["Rust"]),
        }
        mock_result = DiscoveryResult(employees=employees, conflicts=[])

        with patch("crew.discovery.discover_employees", return_value=mock_result):
            client = _make_client()
            resp = client.get("/api/team/agents")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

        names = {a["nickname"] for a in data}
        assert names == {"张三", "王五"}
        # frozen 的不应出现
        assert "李四" not in names

    def test_response_format(self):
        """响应格式与官网模板兼容."""
        from crew.discovery import DiscoveryResult

        employees = {
            "backend-eng": _make_employee(
                "backend-eng", "赵云帆", "后端工程师", 3081, "active", ["Python", "FastAPI"]
            ),
        }
        mock_result = DiscoveryResult(employees=employees, conflicts=[])

        with patch("crew.discovery.discover_employees", return_value=mock_result):
            client = _make_client()
            resp = client.get("/api/team/agents")

        data = resp.json()
        assert len(data) == 1
        agent = data[0]

        # 校验所有必需字段
        assert agent["id"] == 3081
        assert agent["nickname"] == "赵云帆"
        assert agent["title"] == "后端工程师"
        assert agent["avatar_url"] == ""
        assert agent["is_agent"] is True
        assert agent["staff_badge"] == "集识光年"
        assert isinstance(agent["bio"], str)
        assert isinstance(agent["expertise"], list)
        assert "Python" in agent["expertise"]
        assert isinstance(agent["domains"], list)

    def test_sorted_by_agent_id(self):
        """返回结果按 agent_id 升序排列."""
        from crew.discovery import DiscoveryResult

        employees = {
            "c": _make_employee("c", agent_id=3090, agent_status="active"),
            "a": _make_employee("a", agent_id=3050, agent_status="active"),
            "b": _make_employee("b", agent_id=3070, agent_status="active"),
        }
        mock_result = DiscoveryResult(employees=employees, conflicts=[])

        with patch("crew.discovery.discover_employees", return_value=mock_result):
            client = _make_client()
            resp = client.get("/api/team/agents")

        data = resp.json()
        ids = [a["id"] for a in data]
        assert ids == [3050, 3070, 3090]

    def test_empty_when_no_active(self):
        """全部 frozen 时返回空数组."""
        from crew.discovery import DiscoveryResult

        employees = {
            "frozen-a": _make_employee("frozen-a", agent_id=3001, agent_status="frozen"),
        }
        mock_result = DiscoveryResult(employees=employees, conflicts=[])

        with patch("crew.discovery.discover_employees", return_value=mock_result):
            client = _make_client()
            resp = client.get("/api/team/agents")

        data = resp.json()
        assert data == []

    def test_bio_from_yaml(self, tmp_path):
        """bio 从 employee.yaml 读取."""
        import yaml

        from crew.discovery import DiscoveryResult

        # 创建一个带 bio 的 employee.yaml
        emp_dir = tmp_path / "test-emp"
        emp_dir.mkdir()
        yaml_data = {
            "name": "test-emp",
            "bio": "让代码更优雅",
            "domains": ["后端开发", "API设计"],
        }
        (emp_dir / "employee.yaml").write_text(
            yaml.dump(yaml_data, allow_unicode=True), encoding="utf-8"
        )

        employees = {
            "test-emp": _make_employee(
                "test-emp", "测试员", "测试工程师", 3099, "active", source_path=emp_dir
            ),
        }
        mock_result = DiscoveryResult(employees=employees, conflicts=[])

        with patch("crew.discovery.discover_employees", return_value=mock_result):
            client = _make_client()
            resp = client.get("/api/team/agents")

        data = resp.json()
        assert len(data) == 1
        assert data[0]["bio"] == "让代码更优雅"
        assert data[0]["domains"] == ["后端开发", "API设计"]

    def test_no_auth_required(self):
        """公开端点，无 token 也能访问."""
        from crew.discovery import DiscoveryResult

        mock_result = DiscoveryResult(employees={}, conflicts=[])

        with patch("crew.discovery.discover_employees", return_value=mock_result):
            client = _make_client()
            resp = client.get("/api/team/agents")

        assert resp.status_code == 200
        assert resp.json() == []
