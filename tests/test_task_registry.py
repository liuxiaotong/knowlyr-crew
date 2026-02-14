"""测试任务注册表."""

import asyncio
from pathlib import Path

import pytest

from crew.task_registry import TaskRecord, TaskRegistry


class TestTaskRecord:
    """TaskRecord 模型."""

    def test_basic(self):
        r = TaskRecord(
            task_id="test-001",
            trigger="direct",
            target_type="pipeline",
            target_name="full-review",
        )
        assert r.status == "pending"
        assert r.checkpoint is None

    def test_with_checkpoint(self):
        r = TaskRecord(
            task_id="test-002",
            trigger="cron",
            target_type="pipeline",
            target_name="full-review",
            checkpoint={"pipeline_name": "full-review", "next_step_i": 1},
        )
        assert r.checkpoint["pipeline_name"] == "full-review"

    def test_serialize_checkpoint(self):
        r = TaskRecord(
            task_id="test-003",
            trigger="direct",
            target_type="pipeline",
            target_name="test",
            checkpoint={"completed_steps": [{"employee": "a"}], "next_step_i": 1},
        )
        json_str = r.model_dump_json()
        assert "checkpoint" in json_str
        restored = TaskRecord.model_validate_json(json_str)
        assert restored.checkpoint["next_step_i"] == 1


class TestTaskRegistry:
    """TaskRegistry 基本操作."""

    def test_create_and_get(self):
        registry = TaskRegistry()
        record = registry.create(trigger="direct", target_type="pipeline", target_name="test")
        assert record.status == "pending"
        got = registry.get(record.task_id)
        assert got is not None
        assert got.task_id == record.task_id

    def test_update_status(self):
        registry = TaskRegistry()
        record = registry.create(trigger="direct", target_type="employee", target_name="test")
        registry.update(record.task_id, "running")
        assert registry.get(record.task_id).status == "running"
        registry.update(record.task_id, "completed", result={"output": "ok"})
        assert registry.get(record.task_id).status == "completed"
        assert registry.get(record.task_id).result == {"output": "ok"}

    def test_update_checkpoint(self):
        registry = TaskRegistry()
        record = registry.create(trigger="cron", target_type="pipeline", target_name="test")
        assert registry.get(record.task_id).checkpoint is None

        registry.update_checkpoint(record.task_id, {"next_step_i": 1})
        assert registry.get(record.task_id).checkpoint == {"next_step_i": 1}

        registry.update_checkpoint(record.task_id, {"next_step_i": 2})
        assert registry.get(record.task_id).checkpoint["next_step_i"] == 2

    def test_persist_and_load(self, tmp_path):
        path = tmp_path / "tasks.jsonl"

        registry = TaskRegistry(persist_path=path)
        record = registry.create(trigger="direct", target_type="pipeline", target_name="test")
        registry.update(record.task_id, "running")
        registry.update_checkpoint(record.task_id, {"next_step_i": 1})

        # 新实例加载
        registry2 = TaskRegistry(persist_path=path)
        loaded = registry2.get(record.task_id)
        assert loaded is not None
        # checkpoint 应该在最后一条记录中
        assert loaded.checkpoint == {"next_step_i": 1}

    def test_list_recent(self):
        registry = TaskRegistry()
        for i in range(5):
            registry.create(trigger="direct", target_type="employee", target_name=f"emp-{i}")
        recent = registry.list_recent(n=3)
        assert len(recent) == 3

    def test_nonexistent_task(self):
        registry = TaskRegistry()
        assert registry.get("no-such-id") is None
        assert registry.update("no-such-id", "running") is None

    def test_update_checkpoint_nonexistent(self):
        registry = TaskRegistry()
        # 不应报错
        registry.update_checkpoint("no-such-id", {"x": 1})


class TestTaskRegistryThreadSafety:
    """线程安全测试."""

    def test_concurrent_create(self):
        """多线程同时 create 无异常."""
        import threading
        registry = TaskRegistry()
        errors = []

        def _create(i):
            try:
                registry.create(trigger="direct", target_type="employee", target_name=f"emp-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_create, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(registry.list_recent(n=100)) == 20

    def test_concurrent_update(self):
        """多线程同时 update 无丢失."""
        import threading
        registry = TaskRegistry()
        records = [
            registry.create(trigger="direct", target_type="employee", target_name=f"e-{i}")
            for i in range(10)
        ]
        errors = []

        def _update(record):
            try:
                registry.update(record.task_id, "running")
                registry.update(record.task_id, "completed", result={"ok": True})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_update, args=(r,)) for r in records]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        for r in records:
            assert registry.get(r.task_id).status == "completed"

    def test_loaded_tasks_have_events(self, tmp_path):
        """加载后任务有 Event 可 wait."""
        path = tmp_path / "tasks.jsonl"
        reg1 = TaskRegistry(persist_path=path)
        record = reg1.create(trigger="direct", target_type="pipeline", target_name="test")
        reg1.update(record.task_id, "completed", result={"ok": True})

        # 新实例加载
        reg2 = TaskRegistry(persist_path=path)
        event = reg2._events.get(record.task_id)
        assert event is not None
        # 已完成的任务 Event 应该已 set
        assert event.is_set()

    def test_atomic_compaction(self, tmp_path):
        """压缩使用原子写入."""
        path = tmp_path / "tasks.jsonl"
        registry = TaskRegistry(persist_path=path, max_history=5)

        for i in range(10):
            registry.create(trigger="direct", target_type="employee", target_name=f"e-{i}")

        # 压缩已执行，文件应存在
        assert path.exists()
        # 重新加载应该只有 5 条
        reg2 = TaskRegistry(persist_path=path)
        assert len(reg2.list_recent(n=100)) == 5


class TestTaskRegistryPersistence:
    """持久化测试补充."""

    def test_compaction_preserves_recent(self, tmp_path):
        """压缩后保留最新 N 条."""
        path = tmp_path / "tasks.jsonl"
        registry = TaskRegistry(persist_path=path, max_history=3)

        task_ids = []
        for i in range(6):
            r = registry.create(trigger="direct", target_type="employee", target_name=f"e-{i}")
            task_ids.append(r.task_id)

        # 应只保留最新 3 条
        assert len(registry.list_recent(n=100)) == 3
        # 最后 3 个 task 应仍存在
        for tid in task_ids[-3:]:
            assert registry.get(tid) is not None

    def test_persist_creates_parent_dirs(self, tmp_path):
        """目录不存在时自动创建."""
        path = tmp_path / "sub" / "dir" / "tasks.jsonl"
        registry = TaskRegistry(persist_path=path)
        registry.create(trigger="direct", target_type="employee", target_name="test")
        assert path.exists()

    def test_update_checkpoint_persists(self, tmp_path):
        """checkpoint 数据正确持久化并可恢复."""
        path = tmp_path / "tasks.jsonl"
        reg1 = TaskRegistry(persist_path=path)
        record = reg1.create(trigger="cron", target_type="pipeline", target_name="pl")
        reg1.update_checkpoint(record.task_id, {"step": 2, "data": "abc"})

        reg2 = TaskRegistry(persist_path=path)
        loaded = reg2.get(record.task_id)
        assert loaded is not None
        assert loaded.checkpoint == {"step": 2, "data": "abc"}

    def test_compaction_cleans_events(self, tmp_path):
        """压缩后旧任务事件被清理."""
        path = tmp_path / "tasks.jsonl"
        registry = TaskRegistry(persist_path=path, max_history=2)

        r1 = registry.create(trigger="direct", target_type="employee", target_name="old")
        for i in range(3):
            registry.create(trigger="direct", target_type="employee", target_name=f"new-{i}")

        # r1 应该被压缩掉
        assert registry.get(r1.task_id) is None
        assert r1.task_id not in registry._events
