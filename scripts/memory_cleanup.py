"""Deduplicate memory entries under .crew/memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def dedupe_file(path: Path, dry_run: bool) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return 0, 0
    entries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    original_count = len(entries)
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for entry in reversed(entries):
        key = (entry.get("content", ""), entry.get("category", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    deduped.reverse()
    if dry_run or len(deduped) == original_count:
        return original_count, len(deduped)
    with path.open("w", encoding="utf-8") as fh:
        for entry in deduped:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return original_count, len(deduped)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate .crew/memory entries")
    parser.add_argument(
        "--memory-dir",
        default=Path(".crew/memory"),
        type=Path,
        help="Memory directory (default: .crew/memory)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report, do not rewrite")
    args = parser.parse_args()

    if not args.memory_dir.is_dir():
        raise SystemExit(f"Memory dir not found: {args.memory_dir}")

    total_removed = 0
    for file in sorted(args.memory_dir.glob("*.jsonl")):
        before, after = dedupe_file(file, args.dry_run)
        removed = before - after
        if removed:
            total_removed += removed
            action = "would remove" if args.dry_run else "removed"
            print(f"{action} {removed} duplicate entries from {file}")
    if total_removed == 0:
        print("No duplicate memory entries found.")


if __name__ == "__main__":
    main()
