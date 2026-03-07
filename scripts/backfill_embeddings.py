#!/usr/bin/env python3
"""回填 embedding 向量到现有记忆.

用法：
    python scripts/backfill_embeddings.py                     # 全量回填
    python scripts/backfill_embeddings.py --dry-run            # 仅统计，不写入
    python scripts/backfill_embeddings.py --employee 姜墨言    # 只回填指定员工
    python scripts/backfill_embeddings.py --batch-size 50      # 每批处理 50 条
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """回填入口."""
    parser = argparse.ArgumentParser(description="为现有记忆生成 embedding 向量")
    parser.add_argument("--dry-run", action="store_true", help="仅统计，不写入数据库")
    parser.add_argument("--batch-size", type=int, default=100, help="每批处理条数（默认 100）")
    parser.add_argument("--employee", type=str, default="", help="仅处理指定员工的记忆")
    args = parser.parse_args()

    # 延迟导入：确保项目路径正确
    try:
        from crew.database import get_connection
        from crew.embedding import build_embedding_text, get_embedding, is_available
    except ImportError as e:
        logger.error("导入失败: %s — 请确保在项目根目录下运行", e)
        sys.exit(1)

    if not is_available():
        logger.error("sentence-transformers 不可用，无法生成 embedding")
        sys.exit(1)

    # 查询缺少 embedding 的记忆
    conditions = ["embedding IS NULL"]
    params: list[str] = []

    if args.employee:
        conditions.append("employee = %s")
        params.append(args.employee)

    where_clause = " AND ".join(conditions)

    with get_connection() as conn:
        import psycopg2.extras

        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # 统计总数
        cur.execute(f"SELECT COUNT(*) AS cnt FROM memories WHERE {where_clause}", tuple(params))
        total = cur.fetchone()["cnt"]  # type: ignore[index]
        logger.info("待回填记忆: %d 条", total)

        if args.dry_run:
            logger.info("[dry-run] 不执行写入")
            return

        if total == 0:
            logger.info("无需回填")
            return

        # 分批处理
        processed = 0
        failed = 0
        offset = 0

        while offset < total:
            cur.execute(
                f"""
                SELECT id, content, keywords
                FROM memories
                WHERE {where_clause}
                ORDER BY created_at
                LIMIT %s OFFSET %s
                """,
                (*params, args.batch_size, offset),
            )
            rows = cur.fetchall()

            if not rows:
                break

            batch_start = time.monotonic()

            for row in rows:
                text = build_embedding_text(
                    row["content"],
                    list(row.get("keywords") or []),
                )
                embedding = get_embedding(text)

                if embedding is None:
                    failed += 1
                    continue

                cur.execute(
                    "UPDATE memories SET embedding = %s WHERE id = %s",
                    (str(embedding), row["id"]),
                )
                processed += 1

            conn.commit()
            batch_time = time.monotonic() - batch_start
            logger.info(
                "已处理 %d/%d（本批 %d 条，%.1f 秒）",
                processed + failed,
                total,
                len(rows),
                batch_time,
            )
            offset += args.batch_size

    logger.info("回填完成: 成功 %d，失败 %d，总计 %d", processed, failed, total)


if __name__ == "__main__":
    main()
