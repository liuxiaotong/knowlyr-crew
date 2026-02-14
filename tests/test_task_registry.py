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
