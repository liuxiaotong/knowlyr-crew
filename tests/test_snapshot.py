"""数据快照管理测试 — snapshot.py."""

import json

import pytest

from crew.snapshot import SnapshotManager


@pytest.fixture
def crew_root(tmp_path):
    """创建基本目录结构."""
    return tmp_path


@pytest.fixture
def sm(crew_root):
    return SnapshotManager(crew_root)


class TestNextVersion:
    def test_first_snapshot_of_day(self, sm):
        version = sm._next_version()
        assert version.endswith("a")
        assert version.startswith("v2026")

    def test_increments_letter(self, sm):
        # 创建第一个快照目录
        v1 = sm._next_version()
        (sm.snapshots_dir / v1).mkdir(parents=True)
        v2 = sm._next_version()
        assert v2.endswith("b")

    def test_third_snapshot(self, sm):
        v1 = sm._next_version()
        (sm.snapshots_dir / v1).mkdir(parents=True)
        v2 = sm._next_version()
        (sm.snapshots_dir / v2).mkdir(parents=True)
        v3 = sm._next_version()
        assert v3.endswith("c")


class TestCreate:
    def test_creates_manifest(self, sm):
        version = sm.create("测试快照")
        manifest_path = sm.snapshots_dir / version / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text("utf-8"))
        assert manifest["version"] == version
        assert manifest["description"] == "测试快照"

    def test_empty_description(self, sm):
        version = sm.create()
        manifest_path = sm.snapshots_dir / version / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        assert manifest["description"] == ""

    def test_no_global_dir(self, sm):
        """global 目录不存在时 employees 为空."""
        version = sm.create("无员工")
        manifest_path = sm.snapshots_dir / version / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        assert manifest["employees"] == {}

    def test_with_employee(self, sm):
        """有员工目录时拷贝 prompt 文件."""
        emp_dir = sm.global_dir / "test-emp"
        emp_dir.mkdir(parents=True)
        (emp_dir / "employee.yaml").write_text(
            "name: test-emp\ncharacter_name: 测试\nversion: '2.0'\nmodel: claude-sonnet\n",
            encoding="utf-8",
        )
        (emp_dir / "prompt.md").write_text("# 系统 Prompt", encoding="utf-8")

        version = sm.create("有员工")
        manifest_path = sm.snapshots_dir / version / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        assert "test-emp" in manifest["employees"]
        info = manifest["employees"]["test-emp"]
        assert info["display_name"] == "测试"
        assert info["prompt_version"] == "2.0"

        # 检查文件被拷贝
        copied_yaml = sm.snapshots_dir / version / "prompts" / "test-emp" / "employee.yaml"
        assert copied_yaml.exists()


class TestCountData:
    def test_no_data_dirs(self, sm):
        counts = sm._count_data()
        assert counts.get("memory_entries", 0) == 0

    def test_sessions_count(self, sm):
        sm.sessions_dir.mkdir(parents=True)
        # 创建 organic session
        (sm.sessions_dir / "s1.jsonl").write_text(
            json.dumps({"metadata": {"source": "human"}}) + "\n",
            encoding="utf-8",
        )
        # 创建 synthetic session
        (sm.sessions_dir / "s2.jsonl").write_text(
            json.dumps({"metadata": {"source": "cli.generate"}}) + "\n",
            encoding="utf-8",
        )
        counts = sm._count_data()
        assert counts["sessions_total"] == 2
        assert counts["sessions_organic"] == 1
        assert counts["sessions_synthetic"] == 1

    def test_meetings_count(self, sm):
        sm.meetings_dir.mkdir(parents=True)
        (sm.meetings_dir / "m1.md").write_text("会议记录", encoding="utf-8")
        (sm.meetings_dir / "m2.md").write_text("会议记录", encoding="utf-8")
        counts = sm._count_data()
        assert counts["meetings"] == 2

    def test_memory_count(self, sm):
        mem_dir = sm.global_dir / "test-emp" / "memory"
        mem_dir.mkdir(parents=True)
        (mem_dir / "entries.jsonl").write_text(
            '{"text":"a"}\n{"text":"b"}\n',
            encoding="utf-8",
        )
        counts = sm._count_data()
        assert counts["memory_entries"] == 2


class TestListSnapshots:
    def test_empty(self, sm):
        assert sm.list_snapshots() == []

    def test_lists_created_snapshots(self, sm):
        sm.create("快照一")
        sm.create("快照二")
        snapshots = sm.list_snapshots()
        assert len(snapshots) == 2
        assert snapshots[0]["description"] == "快照一"
        assert snapshots[1]["description"] == "快照二"


class TestDiff:
    def test_missing_snapshot_raises(self, sm):
        sm.create("a")
        with pytest.raises(FileNotFoundError):
            sm.diff("v20260216a", "nonexistent")

    def test_diff_identical_snapshots(self, sm):
        v1 = sm.create("第一版")
        v2 = sm.create("第二版")
        result = sm.diff(v1, v2)
        assert result["prompt_changes"] == []
        assert result["new_employees"] == []
        assert result["removed_employees"] == []

    def test_diff_detects_new_employee(self, sm):
        v1 = sm.create("无员工")

        # 添加员工
        emp_dir = sm.global_dir / "new-emp"
        emp_dir.mkdir(parents=True)
        (emp_dir / "employee.yaml").write_text(
            "name: new-emp\nversion: '1.0'\n",
            encoding="utf-8",
        )
        v2 = sm.create("有员工")

        result = sm.diff(v1, v2)
        assert "new-emp" in result["new_employees"]

    def test_diff_detects_removed_employee(self, sm):
        import shutil

        emp_dir = sm.global_dir / "rm-emp"
        emp_dir.mkdir(parents=True)
        (emp_dir / "employee.yaml").write_text(
            "name: rm-emp\nversion: '1.0'\n",
            encoding="utf-8",
        )
        v1 = sm.create("有员工")

        shutil.rmtree(emp_dir)
        v2 = sm.create("无员工")

        result = sm.diff(v1, v2)
        assert "rm-emp" in result["removed_employees"]
