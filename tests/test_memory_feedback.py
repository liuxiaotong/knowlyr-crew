"""记忆反馈和使用分析测试."""

import json
from pathlib import Path

import pytest

from crew.memory_feedback import MemoryFeedbackManager


@pytest.fixture
def feedback_manager(tmp_path):
    """创建测试用的反馈管理器."""
    return MemoryFeedbackManager(
        feedback_dir=tmp_path / "feedback",
        stats_dir=tmp_path / "stats",
    )


class TestMemoryFeedbackManager:
    """测试记忆反馈管理器."""

    def test_submit_feedback(self, feedback_manager):
        """测试提交反馈."""
        feedback = feedback_manager.submit_feedback(
            memory_id="mem-001",
            employee="赵云帆",
            feedback_type="helpful",
            submitted_by="姜墨言",
            context="修复 API bug",
            comment="这条记忆很有帮助",
        )

        assert feedback.memory_id == "mem-001"
        assert feedback.employee == "赵云帆"
        assert feedback.feedback_type == "helpful"
        assert feedback.submitted_by == "姜墨言"
        assert feedback.context == "修复 API bug"
        assert feedback.comment == "这条记忆很有帮助"

        # 验证文件已创建
        feedback_file = feedback_manager.feedback_dir / "mem-001.jsonl"
        assert feedback_file.exists()

    def test_get_feedback(self, feedback_manager):
        """测试获取反馈."""
        # 提交多条反馈
        feedback_manager.submit_feedback(
            memory_id="mem-001",
            employee="赵云帆",
            feedback_type="helpful",
            submitted_by="姜墨言",
        )
        feedback_manager.submit_feedback(
            memory_id="mem-001",
            employee="赵云帆",
            feedback_type="not_helpful",
            submitted_by="林锐",
        )

        # 获取反馈
        feedbacks = feedback_manager.get_feedback("mem-001")

        assert len(feedbacks) == 2
        # 应该按时间倒序
        assert feedbacks[0].feedback_type == "not_helpful"
        assert feedbacks[1].feedback_type == "helpful"

    def test_get_feedback_not_found(self, feedback_manager):
        """测试获取不存在的反馈."""
        feedbacks = feedback_manager.get_feedback("nonexistent")
        assert len(feedbacks) == 0

    def test_record_usage(self, feedback_manager):
        """测试记录使用."""
        feedback_manager.record_usage(
            memory_id="mem-001",
            employee="赵云帆",
            relevance_score=0.85,
        )

        # 验证统计已创建
        stats = feedback_manager.get_stats("mem-001")
        assert stats is not None
        assert stats.memory_id == "mem-001"
        assert stats.employee == "赵云帆"
        assert stats.total_uses == 1
        assert stats.avg_relevance_score == 0.85

    def test_record_usage_multiple(self, feedback_manager):
        """测试多次记录使用."""
        feedback_manager.record_usage("mem-001", "赵云帆", 0.8)
        feedback_manager.record_usage("mem-001", "赵云帆", 0.9)
        feedback_manager.record_usage("mem-001", "赵云帆", 0.7)

        stats = feedback_manager.get_stats("mem-001")
        assert stats.total_uses == 3
        # 平均分数应该是 (0.8 + 0.9 + 0.7) / 3 = 0.8
        assert abs(stats.avg_relevance_score - 0.8) < 0.01

    def test_get_stats(self, feedback_manager):
        """测试获取统计."""
        # 提交反馈（会自动更新统计）
        feedback_manager.submit_feedback(
            memory_id="mem-001",
            employee="赵云帆",
            feedback_type="helpful",
            submitted_by="姜墨言",
        )
        feedback_manager.submit_feedback(
            memory_id="mem-001",
            employee="赵云帆",
            feedback_type="helpful",
            submitted_by="林锐",
        )
        feedback_manager.submit_feedback(
            memory_id="mem-001",
            employee="赵云帆",
            feedback_type="not_helpful",
            submitted_by="程薇",
        )

        # 获取统计
        stats = feedback_manager.get_stats("mem-001")

        assert stats is not None
        assert stats.helpful_count == 2
        assert stats.not_helpful_count == 1
        assert stats.outdated_count == 0
        assert stats.incorrect_count == 0

    def test_get_stats_not_found(self, feedback_manager):
        """测试获取不存在的统计."""
        stats = feedback_manager.get_stats("nonexistent")
        assert stats is None

    def test_get_low_quality_memories(self, feedback_manager):
        """测试获取低质量记忆."""
        # 创建一些记忆统计
        # mem-001: 10 次使用，2 次有帮助 (20%)
        for _ in range(2):
            feedback_manager.submit_feedback(
                "mem-001", "赵云帆", "helpful", "姜墨言"
            )
        for _ in range(8):
            feedback_manager.submit_feedback(
                "mem-001", "赵云帆", "not_helpful", "姜墨言"
            )

        # mem-002: 10 次使用，8 次有帮助 (80%)
        for _ in range(8):
            feedback_manager.submit_feedback(
                "mem-002", "赵云帆", "helpful", "姜墨言"
            )
        for _ in range(2):
            feedback_manager.submit_feedback(
                "mem-002", "赵云帆", "not_helpful", "姜墨言"
            )

        # mem-003: 3 次使用（不满足 min_uses=5）
        for _ in range(3):
            feedback_manager.submit_feedback(
                "mem-003", "赵云帆", "not_helpful", "姜墨言"
            )

        # 获取低质量记忆
        low_quality = feedback_manager.get_low_quality_memories(
            min_uses=5,
            max_helpful_ratio=0.3,
        )

        assert len(low_quality) == 1
        assert low_quality[0].memory_id == "mem-001"

    def test_get_popular_memories(self, feedback_manager):
        """测试获取热门记忆."""
        # mem-001: 10 次使用，8 次有帮助 (80%)
        for _ in range(8):
            feedback_manager.submit_feedback(
                "mem-001", "赵云帆", "helpful", "姜墨言"
            )
        for _ in range(2):
            feedback_manager.submit_feedback(
                "mem-001", "赵云帆", "not_helpful", "姜墨言"
            )

        # mem-002: 5 次使用，4 次有帮助 (80%)
        for _ in range(4):
            feedback_manager.submit_feedback(
                "mem-002", "赵云帆", "helpful", "姜墨言"
            )
        for _ in range(1):
            feedback_manager.submit_feedback(
                "mem-002", "赵云帆", "not_helpful", "姜墨言"
            )

        # mem-003: 10 次使用，2 次有帮助 (20%) - 不应该出现
        for _ in range(2):
            feedback_manager.submit_feedback(
                "mem-003", "赵云帆", "helpful", "姜墨言"
            )
        for _ in range(8):
            feedback_manager.submit_feedback(
                "mem-003", "赵云帆", "not_helpful", "姜墨言"
            )

        # 获取热门记忆
        popular = feedback_manager.get_popular_memories(limit=10)

        assert len(popular) == 2
        # 应该按使用次数降序
        assert popular[0].memory_id == "mem-001"
        assert popular[0].total_uses == 10
        assert popular[1].memory_id == "mem-002"
        assert popular[1].total_uses == 5

    def test_get_feedback_summary(self, feedback_manager):
        """测试获取反馈汇总."""
        # 提交各种类型的反馈
        feedback_manager.submit_feedback("mem-001", "赵云帆", "helpful", "姜墨言")
        feedback_manager.submit_feedback("mem-001", "赵云帆", "helpful", "林锐")
        feedback_manager.submit_feedback("mem-002", "赵云帆", "not_helpful", "程薇")
        feedback_manager.submit_feedback("mem-003", "卫子昂", "outdated", "姜墨言")
        feedback_manager.submit_feedback("mem-004", "卫子昂", "incorrect", "林锐")

        # 获取汇总
        summary = feedback_manager.get_feedback_summary()

        assert summary["total_memories"] == 4
        assert summary["total_feedback"] == 5
        assert summary["feedback_by_type"]["helpful"] == 2
        assert summary["feedback_by_type"]["not_helpful"] == 1
        assert summary["feedback_by_type"]["outdated"] == 1
        assert summary["feedback_by_type"]["incorrect"] == 1
        assert abs(summary["avg_feedback_per_memory"] - 1.25) < 0.01

    def test_get_feedback_summary_filter_by_employee(self, feedback_manager):
        """测试按员工过滤反馈汇总."""
        feedback_manager.submit_feedback("mem-001", "赵云帆", "helpful", "姜墨言")
        feedback_manager.submit_feedback("mem-002", "赵云帆", "helpful", "林锐")
        feedback_manager.submit_feedback("mem-003", "卫子昂", "not_helpful", "程薇")

        # 只获取赵云帆的汇总
        summary = feedback_manager.get_feedback_summary(employee="赵云帆")

        assert summary["total_memories"] == 2
        assert summary["total_feedback"] == 2
        assert summary["feedback_by_type"]["helpful"] == 2
        assert summary["feedback_by_type"]["not_helpful"] == 0

    def test_feedback_updates_stats(self, feedback_manager):
        """测试反馈自动更新统计."""
        feedback_manager.submit_feedback("mem-001", "赵云帆", "helpful", "姜墨言")

        stats = feedback_manager.get_stats("mem-001")
        assert stats.helpful_count == 1

        feedback_manager.submit_feedback("mem-001", "赵云帆", "outdated", "林锐")

        stats = feedback_manager.get_stats("mem-001")
        assert stats.helpful_count == 1
        assert stats.outdated_count == 1
