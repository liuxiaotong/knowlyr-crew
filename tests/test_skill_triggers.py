"""Crew Skills 触发率测试 — pytest 单元测试.

测试核心逻辑（数据结构、加载、报告），不依赖远程 API。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

# 让 pytest 能找到 scripts 目录下的模块
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from test_skill_triggers import (
    CaseResult,
    SkillTestResult,
    load_testcases,
    run_skill_test,
    to_json_report,
)


# ---------------------------------------------------------------------------
# CaseResult
# ---------------------------------------------------------------------------
class TestCaseResult:
    def test_correct_positive_hit(self) -> None:
        r = CaseResult(
            text="写一个 API",
            expected_trigger=True,
            actual_trigger=True,
            triggered_skills=["query-before-code"],
        )
        assert r.correct is True

    def test_incorrect_positive_miss(self) -> None:
        r = CaseResult(
            text="写一个 API",
            expected_trigger=True,
            actual_trigger=False,
            triggered_skills=[],
        )
        assert r.correct is False

    def test_correct_negative(self) -> None:
        r = CaseResult(
            text="天气怎么样",
            expected_trigger=False,
            actual_trigger=False,
            triggered_skills=[],
        )
        assert r.correct is True

    def test_false_trigger(self) -> None:
        r = CaseResult(
            text="天气怎么样",
            expected_trigger=False,
            actual_trigger=True,
            triggered_skills=["query-before-code"],
        )
        assert r.correct is False


# ---------------------------------------------------------------------------
# SkillTestResult
# ---------------------------------------------------------------------------
class TestSkillTestResult:
    @staticmethod
    def _make(
        pos_hits: int = 5,
        pos_total: int = 5,
        neg_false: int = 0,
        neg_total: int = 5,
        trigger_type: str = "keyword",
    ) -> SkillTestResult:
        r = SkillTestResult(
            employee="赵云帆",
            skill_name="query-before-code",
            trigger_type=trigger_type,
        )
        for i in range(pos_total):
            r.positive_results.append(
                CaseResult(
                    text=f"pos-{i}",
                    expected_trigger=True,
                    actual_trigger=i < pos_hits,
                    triggered_skills=["query-before-code"] if i < pos_hits else [],
                )
            )
        for i in range(neg_total):
            r.negative_results.append(
                CaseResult(
                    text=f"neg-{i}",
                    expected_trigger=False,
                    actual_trigger=i < neg_false,
                    triggered_skills=["query-before-code"] if i < neg_false else [],
                )
            )
        return r

    def test_perfect_score(self) -> None:
        r = self._make(pos_hits=5, pos_total=5, neg_false=0, neg_total=5)
        assert r.positive_hit_rate == 1.0
        assert r.false_trigger_rate == 0.0
        assert r.passed is True

    def test_below_hit_rate_threshold(self) -> None:
        r = self._make(pos_hits=3, pos_total=5, neg_false=0, neg_total=5)
        assert r.positive_hit_rate == 0.6
        assert r.passed is False

    def test_above_false_trigger_threshold(self) -> None:
        r = self._make(pos_hits=5, pos_total=5, neg_false=3, neg_total=5)
        assert r.false_trigger_rate == 0.6
        assert r.passed is False

    def test_always_type_no_negatives(self) -> None:
        r = self._make(
            pos_hits=5,
            pos_total=5,
            neg_false=0,
            neg_total=0,
            trigger_type="always",
        )
        assert r.passed is True
        assert r.false_trigger_rate == 0.0

    def test_missed_positives_list(self) -> None:
        r = self._make(pos_hits=3, pos_total=5)
        assert len(r.missed_positives) == 2

    def test_false_negatives_list(self) -> None:
        r = self._make(neg_false=2, neg_total=5)
        assert len(r.false_negatives) == 2

    def test_empty_positive_results(self) -> None:
        r = SkillTestResult(
            employee="test",
            skill_name="test",
            trigger_type="keyword",
        )
        assert r.positive_hit_rate == 1.0

    def test_empty_negative_results(self) -> None:
        r = SkillTestResult(
            employee="test",
            skill_name="test",
            trigger_type="keyword",
        )
        assert r.false_trigger_rate == 0.0


# ---------------------------------------------------------------------------
# load_testcases
# ---------------------------------------------------------------------------
class TestLoadTestcases:
    def test_load_from_dir(self, tmp_path: Path) -> None:
        tc = {
            "employee": "测试员工",
            "skills": [
                {
                    "skill": "test-skill",
                    "trigger_type": "keyword",
                    "positive_cases": ["写代码"],
                    "negative_cases": ["聊天"],
                }
            ],
        }
        (tmp_path / "test.yaml").write_text(yaml.dump(tc, allow_unicode=True), encoding="utf-8")
        loaded = load_testcases(tmp_path)
        assert len(loaded) == 1
        assert loaded[0]["employee"] == "测试员工"

    def test_empty_dir(self, tmp_path: Path) -> None:
        loaded = load_testcases(tmp_path)
        assert loaded == []

    def test_skips_invalid_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "bad.yaml").write_text("not_a_skill_file: true", encoding="utf-8")
        loaded = load_testcases(tmp_path)
        assert loaded == []


# ---------------------------------------------------------------------------
# run_skill_test（mock API）
# ---------------------------------------------------------------------------
class TestRunSkillTest:
    @staticmethod
    def _mock_check_triggers(
        employee: str,
        task: str,
        api_url: str = "",
        api_token: str = "",
    ) -> list[dict[str, Any]]:
        """模拟 API：任务包含 'API' 或 '写' 就触发 query-before-code."""
        if "API" in task or "写" in task:
            return [{"name": "query-before-code", "match_score": 0.5}]
        return []

    def test_all_positive_hit(self) -> None:
        config: dict[str, Any] = {
            "skill": "query-before-code",
            "trigger_type": "keyword",
            "positive_cases": ["写 API 接口", "新的 API endpoint"],
            "negative_cases": ["聊天", "天气"],
        }
        with patch("test_skill_triggers.check_triggers", side_effect=self._mock_check_triggers):
            result = run_skill_test("赵云帆", config)
        assert result.positive_hit_rate == 1.0
        assert result.false_trigger_rate == 0.0
        assert result.passed is True

    def test_partial_miss(self) -> None:
        config: dict[str, Any] = {
            "skill": "query-before-code",
            "trigger_type": "keyword",
            "positive_cases": ["写 API", "重构代码", "修改 schema"],
            "negative_cases": [],
        }
        with patch("test_skill_triggers.check_triggers", side_effect=self._mock_check_triggers):
            result = run_skill_test("赵云帆", config)
        # "写 API" 命中，"重构代码" 和 "修改 schema" 不命中
        assert result.positive_hit_rate == pytest.approx(1 / 3, abs=0.01)
        assert result.passed is False


# ---------------------------------------------------------------------------
# to_json_report
# ---------------------------------------------------------------------------
class TestJsonReport:
    def test_report_structure(self) -> None:
        r = SkillTestResult(
            employee="赵云帆",
            skill_name="query-before-code",
            trigger_type="keyword",
        )
        r.positive_results.append(
            CaseResult(
                text="写 API", expected_trigger=True, actual_trigger=True, triggered_skills=[]
            )
        )
        report = to_json_report([r])
        assert "passed" in report
        assert "total_skills" in report
        assert report["total_skills"] == 1
        assert len(report["skills"]) == 1

    def test_json_serializable(self) -> None:
        r = SkillTestResult(
            employee="test",
            skill_name="test",
            trigger_type="keyword",
        )
        report = to_json_report([r])
        # 确保可以 JSON 序列化
        serialized = json.dumps(report, ensure_ascii=False)
        assert isinstance(serialized, str)
