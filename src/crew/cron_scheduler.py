"""Cron 调度器 — asyncio 后台定时任务."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from crew.cron_config import CronConfig, CronSchedule


class CronScheduler:
    """asyncio 后台 cron 调度器.

    使用 croniter 计算下次触发时间，每个 schedule 一个 asyncio.Task。
    """

    def __init__(
        self,
        config: CronConfig,
        execute_fn,
    ):
        self._config = config
        self._execute_fn = execute_fn  # async fn(schedule) -> None
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        """启动所有 schedule 的后台 task."""
        if not self._config.schedules:
            logger.info("无 cron 任务配置，调度器未启动")
            return

        self._running = True
        for schedule in self._config.schedules:
            task = asyncio.create_task(
                self._run_schedule(schedule),
                name=f"cron-{schedule.name}",
            )
            self._tasks.append(task)
        logger.info("Cron 调度器已启动，%d 个任务", len(self._tasks))

    async def stop(self) -> None:
        """优雅停止所有 schedule."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("Cron 调度器已停止")

    async def _run_schedule(self, schedule: CronSchedule) -> None:
        """单个 schedule 的无限循环."""
        try:
            from croniter import croniter
        except ImportError:
            logger.error("croniter 未安装，无法启动 cron 任务: %s", schedule.name)
            return

        cron = croniter(schedule.cron, datetime.now())
        logger.info(
            "Cron 任务 [%s] 已注册: %s → %s/%s",
            schedule.name, schedule.cron, schedule.target_type, schedule.target_name,
        )

        while self._running:
            try:
                next_time = cron.get_next(float)
                delay = next_time - time.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(0)  # 让出事件循环
                if not self._running:
                    break
                logger.info("Cron 触发: [%s]", schedule.name)
                await self._execute_fn(schedule)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Cron 任务执行异常: [%s]", schedule.name)
                await asyncio.sleep(0)  # 异常后也让出事件循环

    @property
    def schedules(self) -> list[CronSchedule]:
        """返回当前调度列表."""
        return list(self._config.schedules)

    @property
    def running(self) -> bool:
        """调度器是否运行中."""
        return self._running

    def get_next_runs(self) -> list[dict]:
        """获取各任务的下次触发时间."""
        result = []
        try:
            from croniter import croniter
        except ImportError:
            return result

        now = datetime.now()
        for schedule in self._config.schedules:
            try:
                cron = croniter(schedule.cron, now)
                next_dt = cron.get_next(datetime)
                result.append({
                    "name": schedule.name,
                    "cron": schedule.cron,
                    "target_type": schedule.target_type,
                    "target_name": schedule.target_name,
                    "next_run": next_dt.isoformat(),
                })
            except Exception:
                result.append({
                    "name": schedule.name,
                    "cron": schedule.cron,
                    "error": "invalid cron expression",
                })
        return result
