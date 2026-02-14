"""任务注册表 — 追踪 webhook 触发的异步任务."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskRecord(BaseModel):
    """单个任务记录."""

    task_id: str = Field(description="任务唯一标识")
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    trigger: str = Field(description="触发来源: github / openclaw / generic / direct")
    target_type: str = Field(description="目标类型: pipeline / employee")
    target_name: str = Field(description="目标名称")
    args: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class TaskRegistry:
    """内存任务注册表，支持 asyncio.Event 同步等待."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._events: dict[str, asyncio.Event] = {}

    def create(
        self,
        trigger: str,
        target_type: str,
        target_name: str,
        args: dict[str, str] | None = None,
    ) -> TaskRecord:
        """创建新任务."""
        task_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        record = TaskRecord(
            task_id=task_id,
            trigger=trigger,
            target_type=target_type,
            target_name=target_name,
            args=args or {},
        )
        self._tasks[task_id] = record
        self._events[task_id] = asyncio.Event()
        return record

    def update(
        self,
        task_id: str,
        status: Literal["running", "completed", "failed"],
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> TaskRecord | None:
        """更新任务状态."""
        record = self._tasks.get(task_id)
        if record is None:
            return None
        record.status = status
        if result is not None:
            record.result = result
        if error is not None:
            record.error = error
        if status in ("completed", "failed"):
            record.completed_at = datetime.now()
            event = self._events.get(task_id)
            if event:
                event.set()
        return record

    def get(self, task_id: str) -> TaskRecord | None:
        """查询任务."""
        return self._tasks.get(task_id)

    async def wait(self, task_id: str, timeout: float = 300) -> TaskRecord | None:
        """等待任务完成（同步模式）."""
        event = self._events.get(task_id)
        if event is None:
            return None
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        return self._tasks.get(task_id)
