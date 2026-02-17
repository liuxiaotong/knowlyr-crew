"""测试组织架构加载与查询."""

import tempfile
from pathlib import Path

import yaml

from crew.models import Organization, RoutingStep, RoutingTemplate
from crew.organization import invalidate_cache, load_organization


def _make_org_yaml() -> dict:
    """构建测试用组织架构数据."""
    return {
        "teams": {
            "engineering": {
                "label": "工程组",
                "members": ["code-reviewer", "test-engineer", "backend-engineer"],
            },
            "data": {
                "label": "数据组",
                "members": ["data-engineer", "dba"],
            },
        },
        "authority": {
            "A": {
                "label": "自主执行",
                "description": "做完直接交付",
                "members": ["code-reviewer", "test-engineer"],
            },
            "C": {
                "label": "看场景",
                "description": "复杂任务需确认",
                "members": ["backend-engineer", "data-engineer"],
            },
            "B": {
                "label": "需 Kai 确认",
                "description": "涉及决策",
                "members": ["dba"],
            },
        },
        "routing_templates": {
            "code_change": {
                "label": "代码变更",
                "steps": [
                    {"role": "implement", "employee": "backend-engineer"},
                    {"role": "review", "employee": "code-reviewer"},
                    {"role": "test", "employee": "test-engineer"},
                ],
            },
            "with_optional": {
                "label": "带可选步骤",
                "steps": [
                    {"role": "draft", "employee": "backend-engineer"},
                    {"role": "i18n", "employee": "dba", "optional": True},
                ],
            },
        },
    }


class TestOrganizationModel:
    """测试 Organization Pydantic 模型."""

    def test_parse_from_dict(self):
        """应能从字典解析组织架构."""
        data = _make_org_yaml()
        org = Organization(**data)
        assert len(org.teams) == 2
        assert len(org.authority) == 3
        assert len(org.routing_templates) == 2

    def test_empty_organization(self):
        """空字典应返回空 Organization."""
        org = Organization()
        assert org.teams == {}
        assert org.authority == {}
        assert org.routing_templates == {}

    def test_get_team(self):
        """应能按员工名查团队."""
        org = Organization(**_make_org_yaml())
        assert org.get_team("code-reviewer") == "engineering"
        assert org.get_team("data-engineer") == "data"
        assert org.get_team("unknown") is None

    def test_get_authority(self):
        """应能按员工名查权限级别."""
        org = Organization(**_make_org_yaml())
        assert org.get_authority("code-reviewer") == "A"
        assert org.get_authority("backend-engineer") == "C"
        assert org.get_authority("dba") == "B"
        assert org.get_authority("unknown") is None

    def test_get_team_members(self):
        """应能按团队 ID 查成员列表."""
        org = Organization(**_make_org_yaml())
        members = org.get_team_members("engineering")
        assert "code-reviewer" in members
        assert "test-engineer" in members
        assert org.get_team_members("nonexistent") == []


class TestLoadOrganization:
    """测试文件加载."""

    def setup_method(self):
        invalidate_cache()

    def teardown_method(self):
        invalidate_cache()

    def test_load_from_private(self):
        """应从 private/organization.yaml 加载."""
        with tempfile.TemporaryDirectory() as tmpdir:
            org_dir = Path(tmpdir) / "private"
            org_dir.mkdir()
            org_file = org_dir / "organization.yaml"
            org_file.write_text(
                yaml.dump(_make_org_yaml(), allow_unicode=True),
                encoding="utf-8",
            )
            org = load_organization(project_dir=Path(tmpdir))
            assert len(org.teams) == 2
            assert org.get_team("code-reviewer") == "engineering"

    def test_load_from_crew_dir(self):
        """应从 .crew/organization.yaml 加载."""
        with tempfile.TemporaryDirectory() as tmpdir:
            crew_dir = Path(tmpdir) / ".crew"
            crew_dir.mkdir()
            org_file = crew_dir / "organization.yaml"
            org_file.write_text(
                yaml.dump(_make_org_yaml(), allow_unicode=True),
                encoding="utf-8",
            )
            org = load_organization(project_dir=Path(tmpdir))
            assert len(org.teams) == 2

    def test_private_takes_priority(self):
        """private 优先于 .crew."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # .crew 版本只有 1 个团队
            crew_dir = Path(tmpdir) / ".crew"
            crew_dir.mkdir()
            crew_data = {"teams": {"only_one": {"label": "唯一", "members": ["a"]}}}
            (crew_dir / "organization.yaml").write_text(
                yaml.dump(crew_data, allow_unicode=True), encoding="utf-8",
            )
            # private 版本有 2 个团队
            priv_dir = Path(tmpdir) / "private"
            priv_dir.mkdir()
            (priv_dir / "organization.yaml").write_text(
                yaml.dump(_make_org_yaml(), allow_unicode=True), encoding="utf-8",
            )
            org = load_organization(project_dir=Path(tmpdir))
            assert len(org.teams) == 2  # 用的是 private 的

    def test_empty_fallback(self):
        """无配置文件时返回空 Organization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            org = load_organization(project_dir=Path(tmpdir))
            assert org.teams == {}
            assert org.authority == {}

    def test_none_project_dir(self):
        """project_dir 为 None 时返回空 Organization."""
        org = load_organization(project_dir=None)
        assert org.teams == {}

    def test_cache_works(self):
        """加载结果应被缓存."""
        with tempfile.TemporaryDirectory() as tmpdir:
            priv_dir = Path(tmpdir) / "private"
            priv_dir.mkdir()
            (priv_dir / "organization.yaml").write_text(
                yaml.dump(_make_org_yaml(), allow_unicode=True), encoding="utf-8",
            )
            org1 = load_organization(project_dir=Path(tmpdir))
            org2 = load_organization(project_dir=Path(tmpdir))
            assert org1 is org2


class TestRoutingTemplate:
    """测试路由模板逻辑."""

    def test_template_steps_parsed(self):
        """应正确解析路由模板步骤."""
        org = Organization(**_make_org_yaml())
        tmpl = org.routing_templates["code_change"]
        assert tmpl.label == "代码变更"
        assert len(tmpl.steps) == 3
        assert tmpl.steps[0].role == "implement"
        assert tmpl.steps[0].employee == "backend-engineer"

    def test_optional_flag(self):
        """应正确解析 optional 标记."""
        org = Organization(**_make_org_yaml())
        tmpl = org.routing_templates["with_optional"]
        assert tmpl.steps[0].optional is False
        assert tmpl.steps[1].optional is True

    def test_step_with_team(self):
        """步骤可以指定 team 而非具体员工."""
        step = RoutingStep(role="implement", team="engineering")
        assert step.employee is None
        assert step.team == "engineering"

    def test_step_with_employees_list(self):
        """步骤可以指定多个候选员工."""
        step = RoutingStep(role="test", employees=["test-engineer", "e2e-tester"])
        assert step.employees == ["test-engineer", "e2e-tester"]


class TestRealOrganization:
    """测试真实项目的 organization.yaml."""

    def setup_method(self):
        invalidate_cache()

    def teardown_method(self):
        invalidate_cache()

    def test_load_project_organization(self):
        """应能加载项目的 private/organization.yaml."""
        project_dir = Path(__file__).parent.parent
        org = load_organization(project_dir=project_dir)

        # 验证基本结构
        assert len(org.teams) == 6
        assert "engineering" in org.teams
        assert "data" in org.teams
        assert "research" in org.teams
        assert "infrastructure" in org.teams
        assert "business" in org.teams
        assert "functions" in org.teams

        # 验证权限级别
        assert len(org.authority) == 3
        assert "A" in org.authority
        assert "B" in org.authority
        assert "C" in org.authority

        # 验证路由模板
        assert len(org.routing_templates) == 6

    def test_all_employees_have_team(self):
        """所有在权限列表中的员工都应有团队归属."""
        project_dir = Path(__file__).parent.parent
        org = load_organization(project_dir=project_dir)

        all_in_authority = set()
        for auth in org.authority.values():
            all_in_authority.update(auth.members)

        for emp_name in all_in_authority:
            team = org.get_team(emp_name)
            assert team is not None, f"{emp_name} 没有团队归属"

    def test_all_employees_have_authority(self):
        """所有在团队中的员工都应有权限级别."""
        project_dir = Path(__file__).parent.parent
        org = load_organization(project_dir=project_dir)

        all_in_teams = set()
        for team in org.teams.values():
            all_in_teams.update(team.members)

        for emp_name in all_in_teams:
            auth = org.get_authority(emp_name)
            assert auth is not None, f"{emp_name} 没有权限级别"

    def test_no_duplicate_team_membership(self):
        """每个员工只应属于一个团队."""
        project_dir = Path(__file__).parent.parent
        org = load_organization(project_dir=project_dir)

        seen: dict[str, str] = {}
        for tid, team in org.teams.items():
            for member in team.members:
                assert member not in seen, (
                    f"{member} 同时在 {seen[member]} 和 {tid}"
                )
                seen[member] = tid

    def test_no_duplicate_authority(self):
        """每个员工只应有一个权限级别."""
        project_dir = Path(__file__).parent.parent
        org = load_organization(project_dir=project_dir)

        seen: dict[str, str] = {}
        for level, auth in org.authority.items():
            for member in auth.members:
                assert member not in seen, (
                    f"{member} 同时在 {seen[member]} 和 {level}"
                )
                seen[member] = level
