"""daily_eval 范例导出 + MemoryStore 集成测试."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# daily_eval 里用 sys.path.insert 来确保 crew 可导入，但测试环境已在 uv 管理下
# 直接导入即可
from crew.memory import MemoryStore

# 确保 scripts 目录在 path 中以便 import daily_eval
_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import daily_eval as _daily_eval_mod  # noqa: E402, I001


# ── helpers ───────────────────────────────────────────────────────────


def _make_result(
    employee: str = "backend-engineer",
    total_score: float = 0.85,
    session_id: str = "sess-001",
    task: str = "修复用户注册接口的并发问题",
    rubric_scores: dict | None = None,
) -> dict:
    """构造一条评分结果."""
    return {
        "employee": employee,
        "domain": "engineering",
        "task": task,
        "model": "test-model",
        "session_id": session_id,
        "total_score": total_score,
        "outcome_score": 0.6,
        "process_score": 0.25,
        "rubric_scores": rubric_scores
        if rubric_scores is not None
        else {"correctness": 0.9, "efficiency": 0.8},
        "engine": "fallback",
        "scored_at": "2026-02-24T00:00:00",
    }


# ── _build_exemplar_content ──────────────────────────────────────────


class TestBuildExemplarContent:
    """测试范例内容构建."""

    def test_with_rubric(self):
        """有 rubric_scores 时包含亮点."""
        result = _make_result(
            total_score=0.85,
            task="修复注册接口并发问题",
            rubric_scores={"correctness": 0.95, "efficiency": 0.80, "style": 0.70},
        )
        content = _daily_eval_mod._build_exemplar_content(result)
        assert "[高分范例 0.85分]" in content
        assert "修复注册接口并发问题" in content
        assert "correctness=0.95" in content

    def test_without_rubric(self):
        """无 rubric_scores 时不含亮点部分."""
        result = _make_result(rubric_scores={})
        content = _daily_eval_mod._build_exemplar_content(result)
        assert "[高分范例" in content
        assert "表现亮点" not in content


# ── _export_exemplars + MemoryStore ──────────────────────────────────


class TestExportExemplarsMemory:
    """测试范例导出写入 MemoryStore."""

    @pytest.fixture()
    def crew_root(self, tmp_path: Path) -> Path:
        """创建临时 crew 目录结构."""
        (tmp_path / ".crew" / "exemplars").mkdir(parents=True)
        (tmp_path / ".crew" / "memory").mkdir(parents=True)
        return tmp_path

    @pytest.fixture()
    def _patch_crew_root(self, crew_root: Path):
        """Patch CREW_ROOT 和 EXEMPLARS_DIR."""
        with (
            patch.object(_daily_eval_mod, "CREW_ROOT", crew_root),
            patch.object(_daily_eval_mod, "EXEMPLARS_DIR", crew_root / ".crew" / "exemplars"),
        ):
            yield _daily_eval_mod

    def test_exemplar_written_to_memory(self, crew_root: Path, _patch_crew_root):
        """高分范例成功写入 MemoryStore."""
        daily_eval = _patch_crew_root
        results = [_make_result(total_score=0.85, session_id="sess-100")]

        count = daily_eval._export_exemplars(results)

        assert count == 1

        # 验证 MemoryStore 中有记录
        store = MemoryStore(project_dir=crew_root)
        entries = store.query("backend-engineer", category="finding")
        exemplar_entries = [e for e in entries if "exemplar" in e.tags]
        assert len(exemplar_entries) == 1
        assert "高分范例" in exemplar_entries[0].content
        assert exemplar_entries[0].source_session == "sess-100"

    def test_exemplar_has_correct_tags(self, crew_root: Path, _patch_crew_root):
        """写入的记忆有正确的 tags."""
        daily_eval = _patch_crew_root
        results = [_make_result(session_id="sess-200")]

        daily_eval._export_exemplars(results)

        store = MemoryStore(project_dir=crew_root)
        entries = store.query("backend-engineer", category="finding")
        exemplar_entries = [e for e in entries if "exemplar" in e.tags]
        assert len(exemplar_entries) == 1
        assert "exemplar" in exemplar_entries[0].tags
        assert "high-score" in exemplar_entries[0].tags

    def test_no_duplicate_on_rerun(self, crew_root: Path, _patch_crew_root):
        """同一 session 不重复写入."""
        daily_eval = _patch_crew_root
        results = [_make_result(session_id="sess-300")]

        # 跑两次
        daily_eval._export_exemplars(results)
        daily_eval._export_exemplars(results)

        store = MemoryStore(project_dir=crew_root)
        entries = store.query("backend-engineer", category="finding")
        exemplar_entries = [e for e in entries if "exemplar" in e.tags]
        assert len(exemplar_entries) == 1

    def test_low_score_not_written(self, crew_root: Path, _patch_crew_root):
        """低分轨迹不写入 MemoryStore."""
        daily_eval = _patch_crew_root
        results = [_make_result(total_score=0.5, session_id="sess-400")]

        count = daily_eval._export_exemplars(results)

        assert count == 0
        store = MemoryStore(project_dir=crew_root)
        entries = store.query("backend-engineer", category="finding")
        exemplar_entries = [e for e in entries if "exemplar" in e.tags]
        assert len(exemplar_entries) == 0

    def test_multiple_employees(self, crew_root: Path, _patch_crew_root):
        """多员工范例各写各的记忆."""
        daily_eval = _patch_crew_root
        results = [
            _make_result(employee="backend-engineer", session_id="sess-500"),
            _make_result(employee="frontend-engineer", session_id="sess-501"),
        ]

        count = daily_eval._export_exemplars(results)
        assert count == 2

        store = MemoryStore(project_dir=crew_root)

        be_entries = store.query("backend-engineer", category="finding")
        be_exemplars = [e for e in be_entries if "exemplar" in e.tags]
        assert len(be_exemplars) == 1

        fe_entries = store.query("frontend-engineer", category="finding")
        fe_exemplars = [e for e in fe_entries if "exemplar" in e.tags]
        assert len(fe_exemplars) == 1

    def test_no_session_id_still_writes(self, crew_root: Path, _patch_crew_root):
        """无 session_id 时仍然写入（但无去重保护）."""
        daily_eval = _patch_crew_root
        results = [_make_result(session_id="")]

        count = daily_eval._export_exemplars(results)
        assert count == 1

        store = MemoryStore(project_dir=crew_root)
        entries = store.query("backend-engineer", category="finding")
        exemplar_entries = [e for e in entries if "exemplar" in e.tags]
        assert len(exemplar_entries) == 1


# ── engine.py 高分范例注入 ───────────────────────────────────────────


class TestEngineExemplarInjection:
    """测试 engine.py prompt 中高分范例段的注入."""

    def test_exemplar_section_in_prompt(self, tmp_path: Path):
        """当存在 exemplar 记忆时，prompt 中包含'高分范例'段."""
        from crew.engine import CrewEngine
        from crew.models import Employee

        # 准备 memory
        memory_dir = tmp_path / ".crew" / "memory"
        memory_dir.mkdir(parents=True)
        store = MemoryStore(memory_dir=memory_dir)
        store.add(
            employee="test-worker",
            category="finding",
            content="[高分范例 0.90分] 任务：测试任务。表现亮点：correctness=0.95",
            tags=["exemplar", "high-score"],
            origin_employee="test-worker",
        )

        # 最小 Employee
        emp = Employee(
            name="test-worker",
            description="测试用员工",
            body="请执行任务。",
        )

        engine = CrewEngine(project_dir=tmp_path)
        prompt = engine.prompt(emp)

        assert "## 高分范例" in prompt
        assert "高分范例 0.90分" in prompt
        assert "以下是你近期表现优秀的任务案例" in prompt

    def test_no_exemplar_section_when_empty(self, tmp_path: Path):
        """无 exemplar 记忆时，prompt 中不出现'高分范例'段."""
        from crew.engine import CrewEngine
        from crew.models import Employee

        memory_dir = tmp_path / ".crew" / "memory"
        memory_dir.mkdir(parents=True)

        emp = Employee(
            name="test-worker",
            description="测试用员工",
            body="请执行任务。",
        )

        engine = CrewEngine(project_dir=tmp_path)
        prompt = engine.prompt(emp)

        assert "## 高分范例" not in prompt

    def test_non_exemplar_finding_not_included(self, tmp_path: Path):
        """普通 finding 记忆（无 exemplar tag）不出现在高分范例段."""
        from crew.engine import CrewEngine
        from crew.models import Employee

        memory_dir = tmp_path / ".crew" / "memory"
        memory_dir.mkdir(parents=True)
        store = MemoryStore(memory_dir=memory_dir)
        store.add(
            employee="test-worker",
            category="finding",
            content="普通发现，不是范例",
            tags=["general"],
        )

        emp = Employee(
            name="test-worker",
            description="测试用员工",
            body="请执行任务。",
        )

        engine = CrewEngine(project_dir=tmp_path)
        prompt = engine.prompt(emp)

        assert "## 高分范例" not in prompt
