#!/usr/bin/env python3
"""Memory 数据库迁移脚本.

将 .crew/memory/*.jsonl 文件中的记忆迁移到 PostgreSQL 数据库。
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from crew.database import get_connection, is_pg
from crew.memory_store_db import init_memory_tables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def migrate_memory_files(memory_dir: Path, dry_run: bool = False) -> dict:
    """迁移记忆文件到数据库.

    Args:
        memory_dir: 记忆文件目录
        dry_run: 是否为试运行模式

    Returns:
        统计信息字典
    """
    if not is_pg():
        raise RuntimeError("迁移脚本仅支持 PostgreSQL 模式")

    if not memory_dir.is_dir():
        raise ValueError(f"记忆目录不存在: {memory_dir}")

    # 初始化表
    if not dry_run:
        logger.info("初始化 memories 表...")
        init_memory_tables()

    # 统计信息
    stats = {
        "total_files": 0,
        "total_entries": 0,
        "inserted": 0,
        "skipped": 0,
        "errors": 0,
        "employees": [],
    }

    # 扫描所有 .jsonl 文件
    jsonl_files = list(memory_dir.glob("*.jsonl"))
    stats["total_files"] = len(jsonl_files)

    if not jsonl_files:
        logger.warning("未找到任何 .jsonl 文件")
        return stats

    logger.info(f"找到 {len(jsonl_files)} 个记忆文件")

    # 逐个文件处理
    for jsonl_file in jsonl_files:
        employee_name = jsonl_file.stem
        stats["employees"].append(employee_name)
        logger.info(f"处理员工: {employee_name}")

        entries = []
        line_num = 0

        # 读取文件
        try:
            for line in jsonl_file.read_text(encoding="utf-8").splitlines():
                line_num += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    entries.append(entry)
                    stats["total_entries"] += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"  行 {line_num} JSON 解析失败: {e}")
                    stats["errors"] += 1

        except Exception as e:
            logger.error(f"  读取文件失败: {e}")
            stats["errors"] += 1
            continue

        logger.info(f"  读取到 {len(entries)} 条记忆")

        # 写入数据库
        if not dry_run and entries:
            inserted = _insert_entries(entries, employee_name)
            stats["inserted"] += inserted
            stats["skipped"] += len(entries) - inserted
        else:
            stats["skipped"] += len(entries)

    return stats


def _insert_entries(entries: list[dict], employee_name: str) -> int:
    """批量插入记忆条目.

    Args:
        entries: 记忆条目列表
        employee_name: 员工名称

    Returns:
        成功插入的条数
    """
    inserted = 0

    with get_connection() as conn:
        cur = conn.cursor()

        for entry in entries:
            try:
                # 提取字段（兼容旧格式）
                entry_id = entry.get("id", "")
                employee = entry.get("employee", employee_name)
                created_at = entry.get("created_at", "")
                category = entry.get("category", "finding")
                content = entry.get("content", "")
                source_session = entry.get("source_session", "")
                confidence = entry.get("confidence", 1.0)
                superseded_by = entry.get("superseded_by", "")
                ttl_days = entry.get("ttl_days", 0)
                importance = entry.get("importance", 3)
                last_accessed = entry.get("last_accessed") or None
                tags = entry.get("tags", [])
                shared = entry.get("shared", False)
                visibility = entry.get("visibility", "open")
                trigger_condition = entry.get("trigger_condition", "")
                applicability = entry.get("applicability", [])
                origin_employee = entry.get("origin_employee", "")
                verified_count = entry.get("verified_count", 0)

                # 插入数据库（忽略重复）
                cur.execute(
                    """
                    INSERT INTO memories (
                        id, employee, created_at, category, content,
                        source_session, confidence, superseded_by, ttl_days,
                        importance, last_accessed, tags, shared, visibility,
                        trigger_condition, applicability, origin_employee, verified_count
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        entry_id,
                        employee,
                        created_at,
                        category,
                        content,
                        source_session,
                        confidence,
                        superseded_by,
                        ttl_days,
                        importance,
                        last_accessed,
                        tags,
                        shared,
                        visibility,
                        trigger_condition,
                        applicability,
                        origin_employee,
                        verified_count,
                    ),
                )

                if cur.rowcount > 0:
                    inserted += 1

            except Exception as e:
                logger.warning(f"  插入记忆失败 (id={entry.get('id', 'unknown')}): {e}")
                continue

    return inserted


def verify_migration(memory_dir: Path) -> dict:
    """验证迁移结果.

    Args:
        memory_dir: 记忆文件目录

    Returns:
        验证结果字典
    """
    if not is_pg():
        raise RuntimeError("验证脚本仅支持 PostgreSQL 模式")

    # 统计文件中的记忆数
    file_counts = {}
    for jsonl_file in memory_dir.glob("*.jsonl"):
        employee_name = jsonl_file.stem
        count = 0
        for line in jsonl_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError:
                    pass
        file_counts[employee_name] = count

    # 统计数据库中的记忆数
    db_counts = {}
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT employee, COUNT(*) FROM memories GROUP BY employee")
        for row in cur.fetchall():
            db_counts[row[0]] = row[1]

    # 对比结果
    result = {
        "file_total": sum(file_counts.values()),
        "db_total": sum(db_counts.values()),
        "employees": {},
        "missing": [],
        "extra": [],
    }

    all_employees = set(file_counts.keys()) | set(db_counts.keys())
    for emp in all_employees:
        file_count = file_counts.get(emp, 0)
        db_count = db_counts.get(emp, 0)
        result["employees"][emp] = {
            "file": file_count,
            "db": db_count,
            "diff": db_count - file_count,
        }

        if file_count > 0 and db_count == 0:
            result["missing"].append(emp)
        elif file_count == 0 and db_count > 0:
            result["extra"].append(emp)

    return result


def main():
    parser = argparse.ArgumentParser(description="Memory 数据库迁移脚本")
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=Path.cwd() / ".crew" / "memory",
        help="记忆文件目录（默认: .crew/memory）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行模式（不写入数据库）",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="验证迁移结果",
    )

    args = parser.parse_args()

    try:
        if args.verify:
            logger.info("验证迁移结果...")
            result = verify_migration(args.memory_dir)
            logger.info(f"文件总数: {result['file_total']}")
            logger.info(f"数据库总数: {result['db_total']}")
            logger.info(f"差异: {result['db_total'] - result['file_total']}")

            if result["missing"]:
                logger.warning(f"缺失员工: {', '.join(result['missing'])}")

            if result["extra"]:
                logger.info(f"额外员工: {', '.join(result['extra'])}")

            logger.info("\n员工详情:")
            for emp, counts in sorted(result["employees"].items()):
                logger.info(
                    f"  {emp}: 文件={counts['file']}, 数据库={counts['db']}, 差异={counts['diff']}"
                )

        else:
            if args.dry_run:
                logger.info("=== 试运行模式 ===")

            logger.info(f"记忆目录: {args.memory_dir}")
            stats = migrate_memory_files(args.memory_dir, dry_run=args.dry_run)

            logger.info("\n=== 迁移统计 ===")
            logger.info(f"文件数: {stats['total_files']}")
            logger.info(f"总记忆数: {stats['total_entries']}")
            logger.info(f"成功插入: {stats['inserted']}")
            logger.info(f"跳过: {stats['skipped']}")
            logger.info(f"错误: {stats['errors']}")
            logger.info(f"员工列表: {', '.join(stats['employees'])}")

            if args.dry_run:
                logger.info("\n试运行完成，未写入数据库")
            else:
                logger.info("\n迁移完成！")
                logger.info("建议运行 --verify 验证结果")

    except Exception as e:
        logger.error(f"迁移失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
