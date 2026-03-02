"""定时任务：归档过期记忆."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    """主函数."""
    parser = argparse.ArgumentParser(description="归档过期记忆")
    parser.add_argument(
        "--employee",
        type=str,
        help="指定员工名称（不指定则处理所有员工）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行模式（只统计不实际归档）",
    )

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("开始归档过期记忆: employee=%s dry_run=%s", args.employee or "all", args.dry_run)

    try:
        from crew.memory import MemoryStore
        from crew.memory_archive import MemoryArchive

        # 初始化
        memory_store = MemoryStore()
        archive = MemoryArchive(memory_store=memory_store)

        # 确定要处理的员工列表
        if args.employee:
            employees = [args.employee]
        else:
            employees = memory_store.list_employees()

        if not employees:
            logger.info("没有员工需要处理")
            sys.exit(0)

        # 处理每个员工
        total_stats = {"archived": 0, "failed": 0}

        for employee in employees:
            logger.info("处理员工: %s", employee)

            if args.dry_run:
                # 试运行：只统计过期记忆数量
                entries = memory_store._load_employee_entries(employee)
                expired_count = sum(1 for e in entries if memory_store._is_expired(e))
                logger.info("员工 %s 有 %d 条过期记忆（试运行，未归档）", employee, expired_count)
                total_stats["archived"] += expired_count
            else:
                # 实际归档
                stats = archive.archive_expired_memories(employee)
                total_stats["archived"] += stats["archived"]
                total_stats["failed"] += stats["failed"]

        # 输出总统计
        logger.info(
            "归档完成: employees=%d archived=%d failed=%d",
            len(employees),
            total_stats["archived"],
            total_stats["failed"],
        )

        # 输出归档统计
        if not args.dry_run:
            for employee in employees:
                archive_stats = archive.get_archive_stats(employee)
                if archive_stats["total"] > 0:
                    logger.info(
                        "归档统计 [%s]: total=%d by_year=%s",
                        employee,
                        archive_stats["total"],
                        archive_stats["by_year"],
                    )

    except Exception as e:
        logger.exception("归档任务失败: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
