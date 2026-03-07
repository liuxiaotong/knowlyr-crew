"""记忆归档测试."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from crew.memory import MemoryStore
from crew.memory_archive import MemoryArchive


@pytest.fixture
def temp_dirs(tmp_path):
    """创建临时目录."""
    memory_dir = tmp_path / "memory"
    archive_dir = tmp_path / "archive"
    memory_dir.mkdir()
    archive_dir.mkdir()
    return memory_dir, archive_dir


@pytest.fixture
def memory_store(temp_dirs):
    """创建测试用的记忆存储."""
    memory_dir, _ = temp_dirs
    return MemoryStore(memory_dir=memory_dir)


@pytest.fixture
def archive(temp_dirs, memory_store):
    """创建测试用的归档管理器."""
    _, archive_dir = temp_dirs
    return MemoryArchive(archive_dir=archive_dir, memory_store=memory_store)


class TestMemoryArchive:
    """测试记忆归档."""

    def test_archive_entry(self, archive, memory_store):
        """测试归档单条记忆."""
        # 创建记忆
        entry = memory_store.add(
            employee="赵云帆",
            category="finding",
            content="测试内容",
            ttl_days=1,
        )

        # 归档
        success = archive.archive_entry(entry)
        assert success is True

        # 验证归档文件已创建
        created = datetime.fromisoformat(entry.created_at)
        archive_path = archive._get_archive_path("赵云帆", created.year, created.month)
        assert archive_path.exists()

        # 验证归档内容
        lines = archive_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        archived_entry = json.loads(lines[0])
        assert archived_entry["id"] == entry.id
        assert archived_entry["content"] == "测试内容"

    def test_archive_expired_memories(self, archive, memory_store):
        """测试归档过期记忆."""
        # 创建过期记忆（ttl=1 天，创建时间设为 2 天前）
        old_time = (datetime.now() - timedelta(days=2)).isoformat()
        entry1 = memory_store.add(
            employee="赵云帆",
            category="finding",
            content="过期内容1",
            ttl_days=1,
        )
        # 手动修改创建时间
        entry1.created_at = old_time

        entry2 = memory_store.add(
            employee="赵云帆",
            category="finding",
            content="过期内容2",
            ttl_days=1,
        )
        entry2.created_at = old_time

        # 创建未过期记忆
        entry3 = memory_store.add(
            employee="赵云帆",
            category="finding",
            content="未过期内容",
            ttl_days=30,
        )

        # 手动更新文件（因为我们修改了 created_at）
        path = memory_store._employee_file("赵云帆")
        entries = [entry1, entry2, entry3]
        with open(path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(e.model_dump_json() + "\n")

        # 归档过期记忆
        stats = archive.archive_expired_memories("赵云帆")

        assert stats["archived"] == 2
        assert stats["failed"] == 0

        # 验证主表只剩未过期的
        remaining = memory_store.query("赵云帆", limit=100)
        assert len(remaining) == 1
        assert remaining[0].content == "未过期内容"

    def test_query_archive(self, archive, memory_store):
        """测试查询归档记忆."""
        # 创建并归档记忆
        entry1 = memory_store.add(
            employee="赵云帆",
            category="finding",
            content="归档内容1",
        )
        entry2 = memory_store.add(
            employee="赵云帆",
            category="correction",
            content="归档内容2",
        )

        archive.archive_entry(entry1)
        archive.archive_entry(entry2)

        # 查询所有归档
        results = archive.query_archive("赵云帆")
        assert len(results) == 2

        # 按类别过滤
        results = archive.query_archive("赵云帆", category="finding")
        assert len(results) == 1
        assert results[0].content == "归档内容1"

    def test_query_archive_by_date_range(self, archive, memory_store):
        """测试按日期范围查询归档."""
        # 创建不同时间的记忆
        old_time = (datetime.now() - timedelta(days=10)).isoformat()
        recent_time = (datetime.now() - timedelta(days=1)).isoformat()

        entry1 = memory_store.add(
            employee="赵云帆",
            category="finding",
            content="旧记忆",
        )
        entry1.created_at = old_time

        entry2 = memory_store.add(
            employee="赵云帆",
            category="finding",
            content="新记忆",
        )
        entry2.created_at = recent_time

        archive.archive_entry(entry1)
        archive.archive_entry(entry2)

        # 查询最近 5 天的
        start_date = datetime.now() - timedelta(days=5)
        results = archive.query_archive("赵云帆", start_date=start_date)
        assert len(results) == 1
        assert results[0].content == "新记忆"

    def test_restore_from_archive(self, archive, memory_store):
        """测试从归档恢复记忆."""
        # 创建并归档记忆
        entry1 = memory_store.add(
            employee="赵云帆",
            category="finding",
            content="归档内容1",
        )
        entry2 = memory_store.add(
            employee="赵云帆",
            category="finding",
            content="归档内容2",
        )

        archive.archive_entry(entry1)
        archive.archive_entry(entry2)

        # 从主表删除
        memory_store.delete(entry1.id, employee="赵云帆")
        memory_store.delete(entry2.id, employee="赵云帆")

        # 验证主表为空
        assert len(memory_store.query("赵云帆")) == 0

        # 恢复
        stats = archive.restore_from_archive("赵云帆", [entry1.id, entry2.id])
        assert stats["restored"] == 2
        assert stats["not_found"] == 0

        # 验证已恢复到主表
        restored = memory_store.query("赵云帆")
        assert len(restored) == 2

    def test_restore_nonexistent_entries(self, archive):
        """测试恢复不存在的记忆."""
        stats = archive.restore_from_archive("赵云帆", ["nonexistent1", "nonexistent2"])
        assert stats["restored"] == 0
        assert stats["not_found"] == 2

    def test_get_archive_stats(self, archive, memory_store):
        """测试获取归档统计."""
        # 创建并归档记忆
        for i in range(5):
            entry = memory_store.add(
                employee="赵云帆",
                category="finding",
                content=f"内容{i}",
            )
            archive.archive_entry(entry)

        # 获取统计
        stats = archive.get_archive_stats("赵云帆")
        assert stats["total"] == 5
        assert len(stats["by_year"]) > 0

    def test_get_archive_stats_empty(self, archive):
        """测试空归档的统计."""
        stats = archive.get_archive_stats("不存在的员工")
        assert stats["total"] == 0
        assert stats["by_year"] == {}

    def test_archive_path_structure(self, archive):
        """测试归档路径结构."""
        path = archive._get_archive_path("赵云帆", 2026, 3)
        assert str(path).endswith("赵云帆/2026/03.jsonl")
        assert path.parent.exists()


class TestPublicInterfaceCompat:
    """测试公开接口兼容性（确保 cron/archive 不依赖私有方法）."""

    def test_memory_store_public_load_employee_entries(self, memory_store):
        """测试 MemoryStore.load_employee_entries() 公开接口."""
        memory_store.add(employee="测试员工", category="finding", content="公开接口测试")
        entries = memory_store.load_employee_entries("测试员工")
        assert len(entries) >= 1
        assert any(e.content == "公开接口测试" for e in entries)

    def test_memory_store_public_is_expired(self, memory_store):
        """测试 MemoryStore.is_expired() 公开接口."""
        from crew.memory import MemoryEntry

        # 未过期
        entry = MemoryEntry(
            employee="测试员工",
            category="finding",
            content="未过期",
            ttl_days=30,
        )
        assert memory_store.is_expired(entry) is False

        # 无 TTL = 永不过期
        entry_no_ttl = MemoryEntry(
            employee="测试员工",
            category="finding",
            content="永不过期",
            ttl_days=0,
        )
        assert memory_store.is_expired(entry_no_ttl) is False

        # 已过期
        old_time = (datetime.now() - timedelta(days=10)).isoformat()
        entry_expired = MemoryEntry(
            employee="测试员工",
            category="finding",
            content="已过期",
            ttl_days=1,
            created_at=old_time,
        )
        assert memory_store.is_expired(entry_expired) is True

    def test_archive_uses_public_interface(self, archive, memory_store):
        """测试 MemoryArchive.archive_expired_memories 使用公开接口."""
        # 创建过期记忆
        old_time = (datetime.now() - timedelta(days=5)).isoformat()
        entry = memory_store.add(
            employee="赵云帆",
            category="finding",
            content="过期记忆-公开接口",
            ttl_days=1,
        )
        entry.created_at = old_time

        # 手动写入文件
        path = memory_store._employee_file("赵云帆")
        with open(path, "w", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

        # archive_expired_memories 内部应通过公开接口工作
        stats = archive.archive_expired_memories("赵云帆")
        assert stats["archived"] == 1
        assert stats["failed"] == 0

    def test_archive_with_mock_db_store(self, tmp_path):
        """测试 MemoryArchive 兼容类 DB 接口的 mock store."""
        from crew.memory import MemoryEntry

        old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()

        # 模拟 MemoryStoreDB 的公开接口（无 _load_employee_entries / _is_expired）
        class MockDBStore:
            def __init__(self):
                self.entries = [
                    MemoryEntry(
                        employee="mock员工",
                        category="finding",
                        content="DB过期记忆",
                        ttl_days=1,
                        created_at=old_time,
                    ),
                    MemoryEntry(
                        employee="mock员工",
                        category="finding",
                        content="DB未过期记忆",
                        ttl_days=30,
                    ),
                ]
                self.deleted_ids = []

            def load_employee_entries(self, employee):
                return [e for e in self.entries if e.employee == employee]

            def is_expired(self, entry):
                if entry.ttl_days <= 0:
                    return False
                created = datetime.fromisoformat(entry.created_at)
                now = datetime.now(timezone.utc) if created.tzinfo else datetime.now()
                age_days = (now - created).total_seconds() / 86400
                return age_days > entry.ttl_days

            def delete(self, entry_id, employee=None):
                self.deleted_ids.append(entry_id)
                return True

        mock_store = MockDBStore()
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        archive = MemoryArchive(archive_dir=archive_dir, memory_store=mock_store)

        stats = archive.archive_expired_memories("mock员工")
        assert stats["archived"] == 1
        assert stats["failed"] == 0
        assert len(mock_store.deleted_ids) == 1

    def test_cron_dry_run_uses_public_interface(self, memory_store):
        """测试 dry-run 逻辑使用公开接口（模拟 cron 脚本的核心逻辑）."""
        # 创建过期记忆
        old_time = (datetime.now() - timedelta(days=5)).isoformat()
        entry = memory_store.add(
            employee="赵云帆",
            category="finding",
            content="dry-run测试",
            ttl_days=1,
        )
        entry.created_at = old_time

        # 手动更新文件
        path = memory_store._employee_file("赵云帆")
        with open(path, "w", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

        # 模拟 cron 脚本的 dry-run 逻辑（使用公开接口）
        entries = memory_store.load_employee_entries("赵云帆")
        expired_count = sum(1 for e in entries if memory_store.is_expired(e))
        assert expired_count == 1
