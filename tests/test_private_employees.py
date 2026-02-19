"""Private employees 静态检查 — 配置完整性、模板无重复、组织架构一致性."""

import yaml
from pathlib import Path

import pytest

from crew.parser import parse_employee_dir, validate_employee
from crew.organization import load_organization, invalidate_cache

PRIVATE_DIR = Path(__file__).parent.parent / "private" / "employees"
PROJECT_DIR = Path(__file__).parent.parent

# 收集所有员工目录（排除 _templates）
EMPLOYEE_DIRS = sorted([
    d for d in PRIVATE_DIR.iterdir()
    if d.is_dir() and (d / "employee.yaml").exists()
]) if PRIVATE_DIR.is_dir() else []


def _load_yaml(emp_dir: Path) -> dict:
    return yaml.safe_load((emp_dir / "employee.yaml").read_text("utf-8"))


# ── 已知例外 ──
PERMISSIONS_EXCEPTIONS = {"ceo-assistant"}  # 墨言有特殊权限策略
REQUIRED_YAML_FIELDS = {"name", "description", "model_tier", "tools"}
VALID_MODEL_TIERS = {"claude", "kimi"}

# 跳过条件: 没有 private 员工时跳过整个模块
pytestmark = pytest.mark.skipif(
    not EMPLOYEE_DIRS,
    reason="private/employees 不存在或为空",
)


class TestNoTemplateDuplication:
    """回归测试: 模板内容不应在 body 中出现两次."""

    TEMPLATES: dict[str, str] = {}

    @classmethod
    def setup_class(cls):
        templates_dir = PRIVATE_DIR / "_templates"
        if templates_dir.is_dir():
            for md in templates_dir.glob("*.md"):
                content = md.read_text("utf-8").strip()
                if len(content) > 50:
                    cls.TEMPLATES[md.name] = content

    @pytest.mark.parametrize("emp_dir", EMPLOYEE_DIRS, ids=lambda d: d.name)
    def test_no_duplicate_template_content(self, emp_dir):
        """每个员工的 body 中，模板内容不应出现两次."""
        emp = parse_employee_dir(emp_dir, source_layer="private")
        for tpl_name, tpl_content in self.TEMPLATES.items():
            fingerprint = tpl_content[:100]
            count = emp.body.count(fingerprint)
            assert count <= 1, (
                f"{emp.name} 的 body 中模板 {tpl_name} 出现了 {count} 次"
            )


class TestPermissionsCoverage:
    """每个员工 yaml 应有 permissions.roles（已知例外除外）."""

    @pytest.mark.parametrize("emp_dir", EMPLOYEE_DIRS, ids=lambda d: d.name)
    def test_has_permissions_roles(self, emp_dir):
        config = _load_yaml(emp_dir)
        emp_name = config.get("name", "")
        if emp_name in PERMISSIONS_EXCEPTIONS:
            pytest.skip(f"{emp_name} 在已知例外列表中")
        perm = config.get("permissions")
        assert perm is not None, f"{emp_name} 缺少 permissions 字段"
        assert isinstance(perm, dict), f"{emp_name} permissions 不是字典"
        roles = perm.get("roles")
        assert roles and len(roles) > 0, f"{emp_name} 缺少 permissions.roles"


class TestOrganizationCoverage:
    """每个 private employee 应在 organization.yaml 中恰好出现一次."""

    @classmethod
    def setup_class(cls):
        invalidate_cache()
        cls.org = load_organization(project_dir=PROJECT_DIR)
        cls.emp_names = set()
        for emp_dir in EMPLOYEE_DIRS:
            config = _load_yaml(emp_dir)
            cls.emp_names.add(config["name"])

    @classmethod
    def teardown_class(cls):
        invalidate_cache()

    @pytest.mark.parametrize("emp_dir", EMPLOYEE_DIRS, ids=lambda d: d.name)
    def test_in_exactly_one_team(self, emp_dir):
        config = _load_yaml(emp_dir)
        emp_name = config["name"]
        teams_found = [
            tid for tid, team in self.org.teams.items()
            if emp_name in team.members
        ]
        assert len(teams_found) == 1, (
            f"{emp_name}: 应属于 1 个团队，实际属于 {len(teams_found)} 个: {teams_found}"
        )

    @pytest.mark.parametrize("emp_dir", EMPLOYEE_DIRS, ids=lambda d: d.name)
    def test_in_exactly_one_authority(self, emp_dir):
        config = _load_yaml(emp_dir)
        emp_name = config["name"]
        auths_found = [
            level for level, auth in self.org.authority.items()
            if emp_name in auth.members
        ]
        assert len(auths_found) == 1, (
            f"{emp_name}: 应有 1 个权限级别，实际有 {len(auths_found)}: {auths_found}"
        )

    def test_org_has_no_ghost_members(self):
        """organization.yaml 不应引用不存在的员工."""
        all_org_members = set()
        for team in self.org.teams.values():
            all_org_members.update(team.members)
        for auth in self.org.authority.values():
            all_org_members.update(auth.members)
        ghosts = all_org_members - self.emp_names
        assert not ghosts, f"organization.yaml 引用了不存在的员工: {ghosts}"


class TestRequiredFields:
    """employee.yaml 必须包含关键字段."""

    @pytest.mark.parametrize("emp_dir", EMPLOYEE_DIRS, ids=lambda d: d.name)
    def test_required_fields_present(self, emp_dir):
        config = _load_yaml(emp_dir)
        for field in REQUIRED_YAML_FIELDS:
            assert field in config and config[field], (
                f"{config.get('name', emp_dir.name)} 缺少必填字段: {field}"
            )

    @pytest.mark.parametrize("emp_dir", EMPLOYEE_DIRS, ids=lambda d: d.name)
    def test_model_tier_valid(self, emp_dir):
        config = _load_yaml(emp_dir)
        tier = config.get("model_tier", "")
        assert tier in VALID_MODEL_TIERS, (
            f"{config['name']} model_tier='{tier}' 不在 {VALID_MODEL_TIERS} 中"
        )

    @pytest.mark.parametrize("emp_dir", EMPLOYEE_DIRS, ids=lambda d: d.name)
    def test_parseable_and_valid(self, emp_dir):
        """每个员工目录应能成功 parse 且通过 validate."""
        emp = parse_employee_dir(emp_dir, source_layer="private")
        errors = validate_employee(emp)
        assert errors == [], f"{emp.name} 校验失败: {'; '.join(errors)}"
