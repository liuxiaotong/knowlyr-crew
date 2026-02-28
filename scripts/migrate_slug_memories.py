#!/usr/bin/env python3
"""migrate_slug_memories.py — 一次性迁移：英文 slug 残留记忆 → 中文花名.

将冻结员工的英文 slug JSONL 文件中的记忆条目迁移到对应中文花名的 JSONL 文件。

用法（在生产服务器上）:
    # 预检（dry-run，不做任何修改）
    python scripts/migrate_slug_memories.py --project-dir /home/deploy/knowlyr-crew-private

    # 执行迁移
    python scripts/migrate_slug_memories.py --project-dir /home/deploy/knowlyr-crew-private --execute

    # 迁移后重建索引
    python scripts/migrate_slug_memories.py --project-dir /home/deploy/knowlyr-crew-private --rebuild-index
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# 英文 slug → 中文花名 映射
SLUG_TO_NAME: dict[str, str] = {
    "pr-creator": "秦合",
    "doc-writer": "苏文",
    "test-engineer": "程薇",
    "debug-expert": "贺铭",
    "mlops-engineer": "郑锐航",
    "data-engineer": "罗清河",
}


def get_memory_dir(project_dir: Path) -> Path:
    return project_dir / ".crew" / "memory"


def migrate(memory_dir: Path, *, execute: bool = False) -> dict:
    """执行迁移.

    Returns:
        统计结果 dict
    """
    stats = {
        "checked": 0,
        "found": 0,
        "migrated": 0,
        "skipped": 0,
        "errors": [],
    }

    for slug, char_name in SLUG_TO_NAME.items():
        stats["checked"] += 1
        slug_file = memory_dir / f"{slug}.jsonl"

        if not slug_file.exists():
            print(f"  [SKIP] {slug}.jsonl 不存在，跳过")
            stats["skipped"] += 1
            continue

        # 读取 slug 文件内容
        lines = slug_file.read_text(encoding="utf-8").splitlines()
        entries = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"  [WARN] {slug}.jsonl 解析失败行: {e}")
                stats["errors"].append(f"{slug}: JSON parse error: {e}")

        if not entries:
            print(f"  [SKIP] {slug}.jsonl 无有效条目")
            stats["skipped"] += 1
            continue

        stats["found"] += len(entries)
        print(f"  [FOUND] {slug}.jsonl: {len(entries)} 条记忆")
        for entry in entries:
            content_preview = entry.get("content", "")[:60]
            print(f"    - [{entry.get('id', '?')}] ({entry.get('category', '?')}) {content_preview}")

        if not execute:
            print(f"    → 将迁移到 {char_name}.jsonl（dry-run，未执行）")
            continue

        # ── 执行迁移 ──

        # 1. 修改 employee 字段
        for entry in entries:
            entry["employee"] = char_name
            # 同时修改 origin_employee（如果是 slug）
            if entry.get("origin_employee") == slug:
                entry["origin_employee"] = char_name

        # 2. 追加到花名 JSONL
        char_file = memory_dir / f"{char_name}.jsonl"
        existing_ids: set[str] = set()
        if char_file.exists():
            for line in char_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    existing = json.loads(line)
                    existing_ids.add(existing.get("id", ""))
                except json.JSONDecodeError:
                    pass

        appended = 0
        with char_file.open("a", encoding="utf-8") as f:
            for entry in entries:
                entry_id = entry.get("id", "")
                if entry_id in existing_ids:
                    print(f"    [DUP] {entry_id} 已存在于 {char_name}.jsonl，跳过")
                    continue
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                appended += 1

        print(f"    → 追加 {appended} 条到 {char_name}.jsonl")

        # 3. 备份原 slug 文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak_file = memory_dir / f"{slug}.jsonl.bak.{timestamp}"
        shutil.copy2(slug_file, bak_file)
        print(f"    → 备份: {bak_file.name}")

        # 4. 删除原 slug 文件
        slug_file.unlink()
        print(f"    → 已删除 {slug}.jsonl")

        # 同时清理 lock 文件
        lock_file = memory_dir / f"{slug}.jsonl.lock"
        if lock_file.exists():
            lock_file.unlink()
            print(f"    → 已清理 {slug}.jsonl.lock")

        stats["migrated"] += appended

    return stats


def rebuild_index(project_dir: Path) -> None:
    """重建混合搜索索引."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from crew.memory_index import MemorySearchIndex

    memory_dir = get_memory_dir(project_dir)
    index = MemorySearchIndex(
        memory_dir=memory_dir,
        session_dir=project_dir / ".crew" / "sessions",
        project_dir=project_dir,
    )
    stats = index.rebuild()
    print(f"索引重建完成: 记忆 {stats.memory_entries} 条, 会话 {stats.session_messages} 条")

    # 也重建语义索引（embeddings.db）
    try:
        from crew.memory import MemoryStore
        from crew.memory_search import SemanticMemoryIndex

        store = MemoryStore(memory_dir=memory_dir, project_dir=project_dir)
        db_path = memory_dir / "embeddings.db"
        if db_path.exists():
            db_path.unlink()
            print(f"已删除旧语义索引: {db_path}")

        idx = SemanticMemoryIndex(memory_dir)
        total = 0
        for emp in store.list_employees():
            entries = store.query(emp, limit=10000)
            count = idx.reindex(emp, entries)
            total += count
            print(f"  {emp}: {count}/{len(entries)} 条已索引")
        idx.close()
        print(f"语义索引重建完成: 共 {total} 条")
    except Exception as e:
        print(f"语义索引重建跳过: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="迁移英文 slug 残留记忆到中文花名"
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="项目目录（默认当前目录）",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际执行迁移（默认 dry-run）",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="迁移后重建索引",
    )

    args = parser.parse_args()
    memory_dir = get_memory_dir(args.project_dir)

    if not memory_dir.is_dir():
        print(f"错误: 记忆目录不存在: {memory_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"记忆目录: {memory_dir}")
    print(f"模式: {'执行' if args.execute else 'DRY-RUN（预检）'}")
    print(f"待检查: {len(SLUG_TO_NAME)} 个 slug")
    print()

    # 列出当前目录下所有 JSONL 文件（便于确认）
    all_files = sorted(memory_dir.glob("*.jsonl"))
    print(f"当前 JSONL 文件 ({len(all_files)}):")
    for f in all_files:
        line_count = sum(1 for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"  {f.name}: {line_count} 行")
    print()

    stats = migrate(memory_dir, execute=args.execute)

    print()
    print("=== 统计 ===")
    print(f"  检查: {stats['checked']} 个 slug")
    print(f"  找到: {stats['found']} 条记忆")
    print(f"  迁移: {stats['migrated']} 条")
    print(f"  跳过: {stats['skipped']} 个 slug")
    if stats["errors"]:
        print(f"  错误: {len(stats['errors'])} 个")
        for err in stats["errors"]:
            print(f"    - {err}")

    if args.rebuild_index and args.execute:
        print()
        print("=== 重建索引 ===")
        rebuild_index(args.project_dir)

    if not args.execute and stats["found"] > 0:
        print()
        print("以上为预检结果。确认无误后添加 --execute 参数执行迁移。")


if __name__ == "__main__":
    main()
