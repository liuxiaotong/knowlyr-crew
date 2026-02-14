"""测试自动版本管理."""

import shutil
from pathlib import Path

import yaml

from crew.versioning import _bump_patch, check_and_bump, compute_content_hash

FIXTURES = Path(__file__).parent / "fixtures"


class TestBumpPatch:
    """测试版本号 patch 递增."""

    def test_two_part_version(self):
        assert _bump_patch("3.0") == "3.0.1"

    def test_three_part_version(self):
        assert _bump_patch("3.0.1") == "3.0.2"

    def test_higher_patch(self):
        assert _bump_patch("1.2.9") == "1.2.10"

    def test_single_part(self):
        assert _bump_patch("1") == "1.1"


class TestComputeContentHash:
    """测试内容哈希计算."""

    def test_basic_hash(self):
        h = compute_content_hash(FIXTURES / "valid_employee_dir")
        assert isinstance(h, str)
        assert len(h) == 8

    def test_same_content_same_hash(self):
        h1 = compute_content_hash(FIXTURES / "valid_employee_dir")
        h2 = compute_content_hash(FIXTURES / "valid_employee_dir")
        assert h1 == h2

    def test_with_extras_different_hash(self):
        h1 = compute_content_hash(FIXTURES / "valid_employee_dir")
        h2 = compute_content_hash(FIXTURES / "employee_with_extras")
        assert h1 != h2

    def test_includes_workflows_and_adaptors(self):
        """带 workflows/adaptors 的目录哈希应包含子目录文件."""
        h = compute_content_hash(FIXTURES / "employee_with_extras")
        assert isinstance(h, str)
        assert len(h) == 8


class TestCheckAndBump:
    """测试自动版本 bump."""

    def test_first_bump_adds_hash(self, tmp_path):
        """首次运行应设置 hash 并 bump patch."""
        emp_dir = tmp_path / "worker"
        shutil.copytree(FIXTURES / "valid_employee_dir", emp_dir)

        version, bumped = check_and_bump(emp_dir)
        assert bumped is True
        assert version == "1.0.1"

        # 验证回写
        config = yaml.safe_load((emp_dir / "employee.yaml").read_text())
        assert config["version"] == "1.0.1"
        assert "_content_hash" in config

    def test_no_bump_when_hash_matches(self, tmp_path):
        """hash 匹配时不应 bump."""
        emp_dir = tmp_path / "worker"
        shutil.copytree(FIXTURES / "valid_employee_dir", emp_dir)

        # 第一次 bump
        check_and_bump(emp_dir)

        # 第二次不应 bump
        version, bumped = check_and_bump(emp_dir)
        assert bumped is False

    def test_bump_on_content_change(self, tmp_path):
        """内容变更后应 bump."""
        emp_dir = tmp_path / "worker"
        shutil.copytree(FIXTURES / "valid_employee_dir", emp_dir)

        # 第一次 bump
        check_and_bump(emp_dir)

        # 修改内容
        prompt = emp_dir / "prompt.md"
        prompt.write_text(prompt.read_text() + "\n新增内容。\n")

        # 应该再次 bump
        version, bumped = check_and_bump(emp_dir)
        assert bumped is True
        assert version == "1.0.2"

    def test_missing_config_returns_default(self, tmp_path):
        """缺少 employee.yaml 时返回默认版本."""
        version, bumped = check_and_bump(tmp_path)
        assert version == "1.0"
        assert bumped is False

    def test_empty_yaml_returns_default(self, tmp_path):
        """空 employee.yaml 应返回默认版本."""
        emp_dir = tmp_path / "worker"
        emp_dir.mkdir()
        (emp_dir / "employee.yaml").write_text("")
        version, bumped = check_and_bump(emp_dir)
        assert version == "1.0"
        assert bumped is False

    def test_non_dict_yaml_returns_default(self, tmp_path):
        """YAML 内容不是 dict 时应返回默认版本."""
        emp_dir = tmp_path / "worker"
        emp_dir.mkdir()
        (emp_dir / "employee.yaml").write_text("- item1\n- item2\n")
        version, bumped = check_and_bump(emp_dir)
        assert version == "1.0"
        assert bumped is False
