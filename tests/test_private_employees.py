"""员工静态检查 — CI 必跑 + 本地 private 员工扩展检查."""

import yaml
from pathlib import Path

import pytest

from crew.parser import parse_employee_dir, validate_employee

FIXTURES = Path(__file__).parent / "fixtures"
PRIVATE_DIR = Path(__file__).parent.parent / "private" / "employees"
PROJECT_DIR = Path(__file__).parent.parent

REQUIRED_YAML_FIELDS = {"name", "description", "tools"}
VALID_MODEL_TIERS = {"claude", "kimi", ""}  # 空字符串 = 未指定（builtin 员工）
PERMISSIONS_EXCEPTIONS = {"ceo-assistant"}


def _load_yaml(emp_dir: Path) -> dict:
    return yaml.safe_load((emp_dir / "employee.yaml").read_text("utf-8"))


# ── 收集员工目录 ──

def _collect_employee_dirs(base: Path) -> list[Path]:
    if not base.is_dir():
        return []
    return sorted([
        d for d in base.iterdir()
        if d.is_dir() and (d / "employee.yaml").exists()
    ])


FIXTURE_DIRS = _collect_employee_dirs(FIXTURES / "valid_employee_dir".replace("valid_employee_dir", ""))
# 如果 fixtures 下没有多个员工目录，至少用已有的 valid_employee_dir
if not FIXTURE_DIRS and (FIXTURES / "valid_employee_dir" / "employee.yaml").exists():
    FIXTURE_DIRS = [FIXTURES / "valid_employee_dir"]

PRIVATE_DIRS = _collect_employee_dirs(PRIVATE_DIR)


# ====================================================================
# Part 1: CI 必跑 — 用 fixture 数据验证核心解析逻辑
# ====================================================================

class TestEmployeeValidation:
    """核心验证逻辑 — 始终在 CI 中运行."""

    @pytest.mark.parametrize("emp_dir", FIXTURE_DIRS, ids=lambda d: d.name)
    def test_fixture_parseable(self, emp_dir):
        """fixture 员工应能成功 parse 且通过 validate."""
        emp = parse_employee_dir(emp_dir, source_layer="builtin")
        errors = validate_employee(emp)
        assert errors == [], f"{emp.name} 校验失败: {'; '.join(errors)}"

    def test_template_dedup_logic(self, tmp_path):
        """模板去重回归测试: 共享模板不应在 body 中出现两次."""
        # 构造: 员工目录 + _templates 目录 + workflows(空)
        templates = tmp_path / "_templates"
        templates.mkdir()
        (templates / "shared.md").write_text("## 共享模板内容\n\n这是共享模板。")

        emp_dir = tmp_path / "test-worker"
        emp_dir.mkdir()
        (emp_dir / "employee.yaml").write_text(
            "name: test-worker\ndescription: 模板去重测试\n"
        )
        (emp_dir / "prompt.md").write_text("# 测试员工\n\n你是测试员工。")
        (emp_dir / "workflows").mkdir()
        (emp_dir / "adaptors").mkdir()

        emp = parse_employee_dir(emp_dir, source_layer="project")
        count = emp.body.count("## 共享模板内容")
        assert count == 1, f"共享模板在 body 中出现了 {count} 次，应为 1 次"

    def test_required_fields_check(self, tmp_path):
        """缺少必填字段应报错."""
        emp_dir = tmp_path / "bad-worker"
        emp_dir.mkdir()
        (emp_dir / "employee.yaml").write_text("name: bad-worker\n")
        (emp_dir / "prompt.md").write_text("# Bad\n\ntest")

        with pytest.raises(Exception):
            # description 缺失应报错
            parse_employee_dir(emp_dir, source_layer="project")


# ====================================================================
# Part 2: 本地扩展 — private 员工存在时才跑
# ====================================================================

_skip_no_private = pytest.mark.skipif(
    not PRIVATE_DIRS,
    reason="private/employees 不存在（CI 环境正常）",
)


@_skip_no_private
class TestPrivateTemplateDuplication:
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

    @pytest.mark.parametrize("emp_dir", PRIVATE_DIRS, ids=lambda d: d.name)
    def test_no_duplicate_template_content(self, emp_dir):
        emp = parse_employee_dir(emp_dir, source_layer="private")
        for tpl_name, tpl_content in self.TEMPLATES.items():
            fingerprint = tpl_content[:100]
            count = emp.body.count(fingerprint)
            assert count <= 1, (
                f"{emp.name} 的 body 中模板 {tpl_name} 出现了 {count} 次"
            )


@_skip_no_private
class TestPrivatePermissions:
    """每个员工 yaml 应有 permissions.roles."""

    @pytest.mark.parametrize("emp_dir", PRIVATE_DIRS, ids=lambda d: d.name)
    def test_has_permissions_roles(self, emp_dir):
        config = _load_yaml(emp_dir)
        emp_name = config.get("name", "")
        if emp_name in PERMISSIONS_EXCEPTIONS:
            pytest.skip(f"{emp_name} 在已知例外列表中")
        perm = config.get("permissions")
        assert perm is not None, f"{emp_name} 缺少 permissions 字段"
        roles = perm.get("roles")
        assert roles and len(roles) > 0, f"{emp_name} 缺少 permissions.roles"


@_skip_no_private
class TestPrivateOrganization:
    """每个 private employee 应在 organization.yaml 中恰好出现一次."""

    @classmethod
    def setup_class(cls):
        from crew.organization import load_organization, invalidate_cache
        invalidate_cache()
        cls.org = load_organization(project_dir=PROJECT_DIR)
        cls.emp_names = set()
        for emp_dir in PRIVATE_DIRS:
            config = _load_yaml(emp_dir)
            cls.emp_names.add(config["name"])

    @classmethod
    def teardown_class(cls):
        from crew.organization import invalidate_cache
        invalidate_cache()

    @pytest.mark.parametrize("emp_dir", PRIVATE_DIRS, ids=lambda d: d.name)
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

    @pytest.mark.parametrize("emp_dir", PRIVATE_DIRS, ids=lambda d: d.name)
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
        all_org_members = set()
        for team in self.org.teams.values():
            all_org_members.update(team.members)
        for auth in self.org.authority.values():
            all_org_members.update(auth.members)
        ghosts = all_org_members - self.emp_names
        assert not ghosts, f"organization.yaml 引用了不存在的员工: {ghosts}"


@_skip_no_private
class TestPrivateRequiredFields:
    """employee.yaml 必须包含关键字段."""

    @pytest.mark.parametrize("emp_dir", PRIVATE_DIRS, ids=lambda d: d.name)
    def test_required_fields_present(self, emp_dir):
        config = _load_yaml(emp_dir)
        for field in REQUIRED_YAML_FIELDS:
            assert field in config and config[field], (
                f"{config.get('name', emp_dir.name)} 缺少必填字段: {field}"
            )

    @pytest.mark.parametrize("emp_dir", PRIVATE_DIRS, ids=lambda d: d.name)
    def test_model_tier_valid(self, emp_dir):
        config = _load_yaml(emp_dir)
        tier = config.get("model_tier", "")
        assert tier in VALID_MODEL_TIERS, (
            f"{config['name']} model_tier='{tier}' 不在 {VALID_MODEL_TIERS} 中"
        )

    @pytest.mark.parametrize("emp_dir", PRIVATE_DIRS, ids=lambda d: d.name)
    def test_parseable_and_valid(self, emp_dir):
        emp = parse_employee_dir(emp_dir, source_layer="private")
        errors = validate_employee(emp)
        assert errors == [], f"{emp.name} 校验失败: {'; '.join(errors)}"
