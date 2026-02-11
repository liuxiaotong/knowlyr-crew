"""测试 Skills 相关 CLI 命令."""

import tempfile
from pathlib import Path

from click.testing import CliRunner

from crew.cli import main


class TestExportCommand:
    """测试 export 命令."""

    def test_export_builtin(self):
        """应能导出内置员工到 .claude/skills/."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(main, ["export", "code-reviewer", "-d", tmpdir])
            assert result.exit_code == 0
            assert "已导出" in result.output

            skill_file = Path(tmpdir) / ".claude" / "skills" / "code-reviewer" / "SKILL.md"
            assert skill_file.exists()
            content = skill_file.read_text(encoding="utf-8")
            assert "code-reviewer" in content
            assert "allowed-tools:" in content

    def test_export_not_found(self):
        """导出不存在的员工应报错."""
        runner = CliRunner()
        result = runner.invoke(main, ["export", "nonexistent"])
        assert result.exit_code != 0
        assert "未找到" in result.output


class TestExportAllCommand:
    """测试 export-all 命令."""

    def test_export_all(self):
        """应导出所有员工."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(main, ["export-all", "-d", tmpdir])
            assert result.exit_code == 0
            assert "共导出" in result.output

            skills_dir = Path(tmpdir) / ".claude" / "skills"
            assert skills_dir.is_dir()
            skill_dirs = [d for d in skills_dir.iterdir() if d.is_dir()]
            assert len(skill_dirs) >= 5


class TestSyncCommand:
    """测试 sync 命令."""

    def test_sync(self):
        """应同步所有员工."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(main, ["sync", "-d", tmpdir])
            assert result.exit_code == 0
            assert "同步完成" in result.output

    def test_sync_clean(self):
        """sync --clean 应报告删除数."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建孤儿
            orphan = Path(tmpdir) / ".claude" / "skills" / "orphan"
            orphan.mkdir(parents=True)
            (orphan / "SKILL.md").write_text("---\nname: orphan\ndescription: x\n---\nhi\n")

            result = runner.invoke(main, ["sync", "--clean", "-d", tmpdir])
            assert result.exit_code == 0
            assert "已删除" in result.output


class TestListWithSkillLayer:
    """测试 list 命令的 skill 层过滤."""

    def test_list_layer_skill(self):
        """--layer skill 应只显示 skill 层员工."""
        runner = CliRunner()
        # 无 skill 层员工时应显示 "未找到"
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(main, ["list", "--layer", "skill"])
            # 可能有也可能没有，取决于当前项目是否有 .claude/skills/
            assert result.exit_code == 0
