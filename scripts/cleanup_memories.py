"""一次性清洗脚本：清理存量垃圾记忆."""

import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def cleanup_linrui_dumps(store):
    """清理林锐的 auto-push dump."""
    entries = store.query("林锐", limit=500, include_expired=True)
    to_delete = []
    for e in entries:
        # 内容以 [任务] 开头且是 finding，大概率是 auto-push 的审查 dump
        if e.category == "finding" and e.content.startswith("[任务]"):
            to_delete.append(e.id)
        # 带 auto-push tag 的也删
        elif "auto-push" in e.tags:
            to_delete.append(e.id)

    logger.info("林锐待清理: %d / %d", len(to_delete), len(entries))
    for eid in to_delete:
        store.delete(eid, employee="林锐")
    return len(to_delete)


def cleanup_duplicates(store, employee):
    """清理重复记忆（内容前 100 字相同的，保留最新）."""
    entries = store.query(employee, limit=500, include_expired=True)
    seen = {}  # content_prefix -> entry
    to_delete = []

    # entries 按 created_at DESC，第一个是最新的
    for e in entries:
        prefix = e.content[:100].strip()
        if prefix in seen:
            to_delete.append(e.id)  # 删旧的（当前遍历到的更旧）
        else:
            seen[prefix] = e

    if to_delete:
        logger.info("%s 重复记忆: %d 条", employee, len(to_delete))
        for eid in to_delete:
            store.delete(eid, employee=employee)
    return len(to_delete)


def backfill_keywords(store, employee):
    """为没有 keywords 的记忆补充关键词."""
    from crew.memory_pipeline import reflect

    entries = store.query(employee, limit=500)
    updated = 0
    for e in entries:
        # 跳过已有 keywords 的
        if hasattr(e, "keywords") and e.keywords:
            continue

        # 用 Reflect 提取 keywords（但不存储决策，只要 keywords）
        result = reflect(e.content, e.employee)
        if result and result.keywords:
            store.update_keywords(e.id, employee, result.keywords)
            updated += 1

    logger.info("%s 补充关键词: %d 条", employee, updated)
    return updated


def main():
    """主函数."""
    from crew.memory import get_memory_store

    store = get_memory_store()
    employees = store.list_employees()

    total_deleted = 0
    total_deduped = 0
    total_keywords = 0

    # Step 1: 清理林锐 dump
    if "林锐" in employees:
        total_deleted += cleanup_linrui_dumps(store)

    # Step 2: 全员去重
    for emp in employees:
        total_deduped += cleanup_duplicates(store, emp)

    # Step 3: 补关键词（可选，跳过则注释掉）
    # 注意：这步调 LLM，有成本。每条约 $0.001，500 条约 $0.5
    # 如果不想跑，设环境变量 SKIP_KEYWORDS=1
    if not os.environ.get("SKIP_KEYWORDS"):
        for emp in employees:
            total_keywords += backfill_keywords(store, emp)

    logger.info(
        "清洗完成: deleted=%d deduped=%d keywords=%d",
        total_deleted,
        total_deduped,
        total_keywords,
    )


if __name__ == "__main__":
    main()
