"""记忆管理后台测试."""

import pytest

from crew.memory import MemoryStore


@pytest.fixture
def memory_store(tmp_path):
    """创建测试用的记忆存储."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    return MemoryStore(memory_dir=memory_dir)


class TestMemoryDashboard:
    """测试记忆管理仪表板."""

    def test_dashboard_single_employee(self, memory_store):
        """测试单个员工的仪表板数据."""
        # 创建测试数据
        memory_store.add("赵云帆", "finding", "发现1", tags=["api", "bug"], confidence=0.9)
        memory_store.add("赵云帆", "correction", "纠正1", tags=["api"], confidence=0.7)
        memory_store.add("赵云帆", "finding", "发现2", tags=["test"], confidence=0.4)

        # 获取所有记忆
        entries = memory_store.query("赵云帆", limit=1000)

        # 统计
        assert len(entries) == 3

        by_category = {}
        tag_counts = {}
        quality_dist = {"high": 0, "medium": 0, "low": 0}

        for entry in entries:
            by_category[entry.category] = by_category.get(entry.category, 0) + 1

            for tag in entry.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

            if entry.confidence >= 0.8:
                quality_dist["high"] += 1
            elif entry.confidence >= 0.5:
                quality_dist["medium"] += 1
            else:
                quality_dist["low"] += 1

        assert by_category == {"finding": 2, "correction": 1}
        assert tag_counts == {"api": 2, "bug": 1, "test": 1}
        assert quality_dist == {"high": 1, "medium": 1, "low": 1}

    def test_dashboard_multiple_employees(self, memory_store):
        """测试多个员工的全局仪表板."""
        # 创建多个员工的记忆
        memory_store.add("赵云帆", "finding", "内容1", confidence=0.9)
        memory_store.add("赵云帆", "correction", "内容2", confidence=0.8)
        memory_store.add("卫子昂", "pattern", "内容3", confidence=0.7)

        # 统计
        employees = memory_store.list_employees()
        assert len(employees) == 2

        total = 0
        by_employee = {}

        for emp in employees:
            entries = memory_store.query(emp, limit=1000)
            by_employee[emp] = len(entries)
            total += len(entries)

        assert total == 3
        assert by_employee["赵云帆"] == 2
        assert by_employee["卫子昂"] == 1


class TestMemoryBatchOperations:
    """测试批量操作."""

    def test_batch_update_tags(self, memory_store):
        """测试批量更新标签."""
        import json

        # 创建记忆
        entry1 = memory_store.add("赵云帆", "finding", "内容1", tags=["old-tag"])
        entry2 = memory_store.add("赵云帆", "finding", "内容2", tags=["old-tag"])

        # 模拟批量更新（添加标签）
        from crew.memory import MemoryEntry
        from crew.paths import file_lock

        path = memory_store.employee_file("赵云帆")
        entry_ids = {entry1.id, entry2.id}
        updates = {"tags": ["new-tag"], "remove_tags": ["old-tag"]}

        with file_lock(path):
            lines = path.read_text(encoding="utf-8").splitlines()
            new_lines = []

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue

                entry = MemoryEntry(**json.loads(stripped))

                if entry.id in entry_ids:
                    if "tags" in updates:
                        entry.tags = list(set(entry.tags + updates["tags"]))

                    if "remove_tags" in updates:
                        for tag in updates["remove_tags"]:
                            if tag in entry.tags:
                                entry.tags.remove(tag)

                    new_lines.append(entry.model_dump_json())
                else:
                    new_lines.append(stripped)

            path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        # 验证更新
        updated_entries = memory_store.query("赵云帆")
        for entry in updated_entries:
            assert "new-tag" in entry.tags
            assert "old-tag" not in entry.tags

    def test_batch_update_confidence(self, memory_store):
        """测试批量更新置信度."""
        import json

        # 创建记忆
        entry1 = memory_store.add("赵云帆", "finding", "内容1", confidence=0.5)
        entry2 = memory_store.add("赵云帆", "finding", "内容2", confidence=0.6)

        # 批量更新置信度
        from crew.memory import MemoryEntry
        from crew.paths import file_lock

        path = memory_store.employee_file("赵云帆")
        entry_ids = {entry1.id, entry2.id}
        new_confidence = 0.9

        with file_lock(path):
            lines = path.read_text(encoding="utf-8").splitlines()
            new_lines = []

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue

                entry = MemoryEntry(**json.loads(stripped))

                if entry.id in entry_ids:
                    entry.confidence = new_confidence
                    new_lines.append(entry.model_dump_json())
                else:
                    new_lines.append(stripped)

            path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        # 验证更新（使用近似比较，因为浮点数精度问题）
        updated_entries = memory_store.query("赵云帆")
        for entry in updated_entries:
            assert abs(entry.confidence - 0.9) < 0.01

    def test_batch_delete(self, memory_store):
        """测试批量删除."""
        # 创建记忆
        entry1 = memory_store.add("赵云帆", "finding", "内容1")
        entry2 = memory_store.add("赵云帆", "finding", "内容2")
        entry3 = memory_store.add("赵云帆", "finding", "内容3")

        # 批量删除前两个
        deleted_count = 0
        for entry_id in [entry1.id, entry2.id]:
            if memory_store.delete(entry_id, employee="赵云帆"):
                deleted_count += 1

        assert deleted_count == 2

        # 验证只剩一个
        remaining = memory_store.query("赵云帆")
        assert len(remaining) == 1
        assert remaining[0].id == entry3.id
