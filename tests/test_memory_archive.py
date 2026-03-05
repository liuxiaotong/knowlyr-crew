"""记忆归档测试."""

import json
from datetime import datetime, timedelta

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
