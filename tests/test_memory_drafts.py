"""记忆草稿管理测试."""

import json

import pytest

from crew.memory_drafts import MemoryDraft, MemoryDraftStore


@pytest.fixture
def drafts_dir(tmp_path):
    """临时草稿目录."""
    return tmp_path / "drafts"


@pytest.fixture
def store(drafts_dir):
    """创建测试用的草稿存储."""
    return MemoryDraftStore(drafts_dir=drafts_dir)


class TestMemoryDraft:
    """测试记忆草稿模型."""

    def test_default_fields(self):
        """测试默认字段."""
        draft = MemoryDraft(
            employee="赵云帆",
            category="finding",
            content="测试内容",
        )

        assert draft.employee == "赵云帆"
        assert draft.category == "finding"
        assert draft.content == "测试内容"
        assert draft.status == "pending"
        assert draft.confidence == 1.0
        assert draft.tags == []
        assert draft.id  # 自动生成
        assert draft.created_at  # 自动生成

    def test_custom_fields(self):
        """测试自定义字段."""
        draft = MemoryDraft(
            employee="赵云帆",
            category="correction",
            content="修正内容",
            tags=["api", "bug"],
            confidence=0.8,
            source_trajectory_id="traj-123",
        )

        assert draft.tags == ["api", "bug"]
        assert draft.confidence == 0.8
        assert draft.source_trajectory_id == "traj-123"


class TestMemoryDraftStore:
    """测试记忆草稿存储."""

    def test_init(self, store, drafts_dir):
        """测试初始化."""
        assert store.drafts_dir == drafts_dir
        assert drafts_dir.exists()

    def test_create_draft(self, store):
        """测试创建草稿."""
        draft = store.create_draft(
            employee="赵云帆",
            category="finding",
            content="发现了一个问题",
            tags=["bug", "api"],
            confidence=0.9,
            source_trajectory_id="traj-123",
        )

        assert draft.employee == "赵云帆"
        assert draft.category == "finding"
        assert draft.content == "发现了一个问题"
        assert draft.tags == ["bug", "api"]
        assert draft.confidence == 0.9
        assert draft.status == "pending"

        # 验证文件已创建
        draft_file = store.drafts_dir / f"{draft.id}.json"
        assert draft_file.exists()

        # 验证文件内容
        data = json.loads(draft_file.read_text())
        assert data["employee"] == "赵云帆"
        assert data["category"] == "finding"

    def test_get_draft(self, store):
        """测试获取草稿."""
        # 创建草稿
        created = store.create_draft(
            employee="赵云帆",
            category="decision",
            content="决策内容",
        )

        # 获取草稿
        draft = store.get_draft(created.id)

        assert draft is not None
        assert draft.id == created.id
        assert draft.employee == "赵云帆"
        assert draft.content == "决策内容"

    def test_get_draft_not_found(self, store):
        """测试获取不存在的草稿."""
        draft = store.get_draft("nonexistent")
        assert draft is None

    def test_list_drafts(self, store):
        """测试列出草稿."""
        # 创建多个草稿
        store.create_draft("赵云帆", "finding", "内容1")
        store.create_draft("卫子昂", "correction", "内容2")
        store.create_draft("赵云帆", "pattern", "内容3")

        # 列出所有草稿
        drafts = store.list_drafts()
        assert len(drafts) == 3

    def test_list_drafts_filter_by_status(self, store):
        """测试按状态过滤."""
        draft1 = store.create_draft("赵云帆", "finding", "内容1")
        draft2 = store.create_draft("赵云帆", "finding", "内容2")

        # 批准一个
        store.approve_draft(draft1.id)

        # 只列出 pending
        pending = store.list_drafts(status="pending")
        assert len(pending) == 1
        assert pending[0].id == draft2.id

        # 只列出 approved
        approved = store.list_drafts(status="approved")
        assert len(approved) == 1
        assert approved[0].id == draft1.id

    def test_list_drafts_filter_by_employee(self, store):
        """测试按员工过滤."""
        store.create_draft("赵云帆", "finding", "内容1")
        store.create_draft("卫子昂", "finding", "内容2")
        store.create_draft("赵云帆", "finding", "内容3")

        # 只列出赵云帆的
        drafts = store.list_drafts(employee="赵云帆")
        assert len(drafts) == 2
        assert all(d.employee == "赵云帆" for d in drafts)

    def test_list_drafts_limit(self, store):
        """测试限制返回数量."""
        for i in range(10):
            store.create_draft("赵云帆", "finding", f"内容{i}")

        drafts = store.list_drafts(limit=5)
        assert len(drafts) == 5

    def test_approve_draft(self, store):
        """测试批准草稿."""
        draft = store.create_draft("赵云帆", "finding", "内容")

        # 批准
        approved = store.approve_draft(draft.id, reviewed_by="姜墨言")

        assert approved is not None
        assert approved.status == "approved"
        assert approved.reviewed_by == "姜墨言"
        assert approved.reviewed_at  # 已设置审核时间

        # 验证文件已更新
        reloaded = store.get_draft(draft.id)
        assert reloaded.status == "approved"

    def test_approve_draft_not_found(self, store):
        """测试批准不存在的草稿."""
        result = store.approve_draft("nonexistent")
        assert result is None

    def test_approve_draft_already_reviewed(self, store):
        """测试重复批准."""
        draft = store.create_draft("赵云帆", "finding", "内容")
        store.approve_draft(draft.id)

        # 再次批准（应该返回已批准的草稿，但不改变状态）
        result = store.approve_draft(draft.id)
        assert result.status == "approved"

    def test_reject_draft(self, store):
        """测试拒绝草稿."""
        draft = store.create_draft("赵云帆", "finding", "内容")

        # 拒绝
        rejected = store.reject_draft(
            draft.id,
            reason="内容不够具体",
            reviewed_by="姜墨言",
        )

        assert rejected is not None
        assert rejected.status == "rejected"
        assert rejected.reject_reason == "内容不够具体"
        assert rejected.reviewed_by == "姜墨言"
        assert rejected.reviewed_at

        # 验证文件已更新
        reloaded = store.get_draft(draft.id)
        assert reloaded.status == "rejected"
        assert reloaded.reject_reason == "内容不够具体"

    def test_reject_draft_not_found(self, store):
        """测试拒绝不存在的草稿."""
        result = store.reject_draft("nonexistent")
        assert result is None

    def test_delete_draft(self, store):
        """测试删除草稿."""
        draft = store.create_draft("赵云帆", "finding", "内容")

        # 删除
        deleted = store.delete_draft(draft.id)
        assert deleted is True

        # 验证文件已删除
        draft_file = store.drafts_dir / f"{draft.id}.json"
        assert not draft_file.exists()

        # 无法再获取
        assert store.get_draft(draft.id) is None

    def test_delete_draft_not_found(self, store):
        """测试删除不存在的草稿."""
        deleted = store.delete_draft("nonexistent")
        assert deleted is False

    def test_count_by_status(self, store):
        """测试统计各状态数量."""
        # 创建草稿
        draft1 = store.create_draft("赵云帆", "finding", "内容1")
        draft2 = store.create_draft("赵云帆", "finding", "内容2")
        store.create_draft("赵云帆", "finding", "内容3")

        # 批准一个，拒绝一个
        store.approve_draft(draft1.id)
        store.reject_draft(draft2.id)

        # 统计
        counts = store.count_by_status()

        assert counts["pending"] == 1
        assert counts["approved"] == 1
        assert counts["rejected"] == 1

    def test_count_by_status_empty(self, store):
        """测试空存储的统计."""
        counts = store.count_by_status()

        assert counts["pending"] == 0
        assert counts["approved"] == 0
        assert counts["rejected"] == 0
