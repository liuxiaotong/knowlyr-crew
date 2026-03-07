"""记忆归档管理 — TTL 过期记忆的归档和恢复."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from crew.memory import MemoryEntry, MemoryStore
from crew.paths import file_lock

logger = logging.getLogger(__name__)


class MemoryArchive:
    """记忆归档管理器."""

    def __init__(
        self,
        archive_dir: Path | None = None,
        memory_store: MemoryStore | None = None,
    ):
        """初始化归档管理器.

        Args:
            archive_dir: 归档目录，默认 /data/memory_archive
            memory_store: 记忆存储实例，用于访问主表
        """
        self.archive_dir = archive_dir or Path("/data/memory_archive")
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.memory_store = memory_store

    def _get_archive_path(self, employee: str, year: int, month: int) -> Path:
        """获取归档文件路径.

        Args:
            employee: 员工名称
            year: 年份
            month: 月份

        Returns:
            归档文件路径: /data/memory_archive/{employee}/{year}/{month}.jsonl
        """
        employee_dir = self.archive_dir / employee / str(year)
        employee_dir.mkdir(parents=True, exist_ok=True)
        return employee_dir / f"{month:02d}.jsonl"

    def archive_entry(self, entry: MemoryEntry) -> bool:
        """归档单条记忆.

        Args:
            entry: 记忆条目

        Returns:
            True 如果归档成功
        """
        try:
            # 解析创建时间，确定归档位置
            created = datetime.fromisoformat(entry.created_at)
            archive_path = self._get_archive_path(
                entry.employee,
                created.year,
                created.month,
            )

            # 追加到归档文件
            with file_lock(archive_path):
                with open(archive_path, "a", encoding="utf-8") as f:
                    f.write(entry.model_dump_json() + "\n")

            logger.info(
                "归档记忆: id=%s employee=%s archive=%s",
                entry.id,
                entry.employee,
                archive_path,
            )
            return True

        except Exception as e:
            logger.error("归档记忆失败: id=%s error=%s", entry.id, e)
            return False

    def archive_expired_memories(self, employee: str) -> dict[str, int]:
        """归档指定员工的所有过期记忆.

        Args:
            employee: 员工名称

        Returns:
            统计信息: {"archived": 归档数量, "failed": 失败数量}
        """
        if self.memory_store is None:
            raise RuntimeError("memory_store 未设置")

        stats = {"archived": 0, "failed": 0}

        # 加载所有记忆（包括过期的）— 使用公开接口，兼容文件版和 DB 版
        entries = self.memory_store.load_employee_entries(employee)
        expired_entries = [e for e in entries if self.memory_store.is_expired(e)]

        if not expired_entries:
            logger.info("员工 %s 没有过期记忆", employee)
            return stats

        # 归档过期记忆
        archived_ids: list[str] = []
        for entry in expired_entries:
            if self.archive_entry(entry):
                archived_ids.append(entry.id)
                stats["archived"] += 1
            else:
                stats["failed"] += 1

        # 从主表删除已归档的记忆
        if archived_ids:
            self._remove_from_main_table(employee, archived_ids)

        logger.info(
            "归档完成: employee=%s archived=%d failed=%d",
            employee,
            stats["archived"],
            stats["failed"],
        )
        return stats

    def _remove_from_main_table(self, employee: str, entry_ids: list[str]) -> None:
        """从主表删除已归档的记忆.

        兼容文件版 MemoryStore 和数据库版 MemoryStoreDB：
        - 有 delete() 方法时逐条调用（DB 模式和文件模式都支持）
        - 回退到文件操作仅在有 _employee_file 属性时

        Args:
            employee: 员工名称
            entry_ids: 要删除的记忆 ID 列表
        """
        if self.memory_store is None:
            return

        # 统一使用 delete() 公开接口（MemoryStore 和 MemoryStoreDB 都有）
        for entry_id in entry_ids:
            try:
                self.memory_store.delete(entry_id, employee=employee)
            except Exception as e:
                logger.warning("删除已归档记忆失败: id=%s error=%s", entry_id, e)

        logger.info("从主表删除 %d 条已归档记忆: employee=%s", len(entry_ids), employee)

    def query_archive(
        self,
        employee: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        category: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """查询归档记忆.

        Args:
            employee: 员工名称
            start_date: 起始日期（包含）
            end_date: 结束日期（包含）
            category: 按类别过滤
            limit: 最大返回数量

        Returns:
            归档记忆列表（按时间倒序）
        """
        entries: list[MemoryEntry] = []

        employee_dir = self.archive_dir / employee
        if not employee_dir.exists():
            return entries

        # 遍历所有归档文件
        for year_dir in sorted(employee_dir.iterdir(), reverse=True):
            if not year_dir.is_dir():
                continue

            for month_file in sorted(year_dir.glob("*.jsonl"), reverse=True):
                try:
                    for line in month_file.read_text(encoding="utf-8").splitlines():
                        stripped = line.strip()
                        if not stripped:
                            continue

                        entry = MemoryEntry(**json.loads(stripped))

                        # 时间范围过滤
                        if start_date or end_date:
                            created = datetime.fromisoformat(entry.created_at)
                            if start_date and created < start_date:
                                continue
                            if end_date and created > end_date:
                                continue

                        # 类别过滤
                        if category and entry.category != category:
                            continue

                        entries.append(entry)

                        if len(entries) >= limit:
                            return entries

                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug("跳过损坏的归档条目: %s", e)
                    continue

        return entries

    def restore_from_archive(
        self,
        employee: str,
        entry_ids: list[str],
    ) -> dict[str, int]:
        """从归档恢复记忆到主表.

        Args:
            employee: 员工名称
            entry_ids: 要恢复的记忆 ID 列表

        Returns:
            统计信息: {"restored": 恢复数量, "not_found": 未找到数量}
        """
        if self.memory_store is None:
            raise RuntimeError("memory_store 未设置")

        stats = {"restored": 0, "not_found": 0}
        id_set = set(entry_ids)
        found_entries: list[MemoryEntry] = []

        # 在归档中查找要恢复的记忆
        employee_dir = self.archive_dir / employee
        if not employee_dir.exists():
            stats["not_found"] = len(entry_ids)
            return stats

        for year_dir in employee_dir.iterdir():
            if not year_dir.is_dir():
                continue

            for month_file in year_dir.glob("*.jsonl"):
                try:
                    for line in month_file.read_text(encoding="utf-8").splitlines():
                        stripped = line.strip()
                        if not stripped:
                            continue

                        entry = MemoryEntry(**json.loads(stripped))
                        if entry.id in id_set:
                            found_entries.append(entry)
                            id_set.remove(entry.id)

                            if not id_set:
                                break

                except (json.JSONDecodeError, ValueError):
                    continue

            if not id_set:
                break

        # 恢复到主表 — 使用 add() 公开接口，兼容文件版和 DB 版
        for entry in found_entries:
            try:
                self.memory_store.add(
                    employee=entry.employee,
                    category=entry.category,
                    content=entry.content,
                    source_session=entry.source_session,
                    confidence=entry.confidence,
                    ttl_days=entry.ttl_days,  # 保留原始 TTL
                    tags=entry.tags,
                    shared=entry.shared,
                    visibility=entry.visibility,
                    trigger_condition=entry.trigger_condition,
                    applicability=entry.applicability,
                    origin_employee=entry.origin_employee,
                )
                stats["restored"] += 1
            except Exception as e:
                logger.warning("恢复记忆失败: id=%s error=%s", entry.id, e)

        stats["not_found"] = len(id_set)

        logger.info(
            "恢复归档记忆: employee=%s restored=%d not_found=%d",
            employee,
            stats["restored"],
            stats["not_found"],
        )
        return stats

    def get_archive_stats(self, employee: str) -> dict[str, int]:
        """获取归档统计信息.

        Args:
            employee: 员工名称

        Returns:
            统计信息: {"total": 总数, "by_year": {year: count}}
        """
        stats = {"total": 0, "by_year": {}}

        employee_dir = self.archive_dir / employee
        if not employee_dir.exists():
            return stats

        for year_dir in employee_dir.iterdir():
            if not year_dir.is_dir():
                continue

            year = year_dir.name
            year_count = 0

            for month_file in year_dir.glob("*.jsonl"):
                try:
                    count = sum(
                        1
                        for line in month_file.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
                    year_count += count
                except Exception:
                    continue

            if year_count > 0:
                stats["by_year"][year] = year_count
                stats["total"] += year_count

        return stats
