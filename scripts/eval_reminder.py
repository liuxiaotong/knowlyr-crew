"""List pending eval decisions from .crew/evaluations/decisions.jsonl."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ISO8601 = "%Y-%m-%dT%H:%M:%S"


def parse_dt(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Show pending eval decisions")
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path(".crew/evaluations/decisions.jsonl"),
        help="Path to decisions jsonl",
    )
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=None,
        help="Only show decisions older than this many hours",
    )
    args = parser.parse_args()

    if not args.decisions.exists():
        raise SystemExit(f"Decisions file not found: {args.decisions}")

    now = datetime.now(timezone.utc)
    pending = []
    for line in args.decisions.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("status") == "evaluated":
            continue
        created_at = parse_dt(data.get("created_at", ""))
        if created_at and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if args.max_age_hours and created_at:
            age_hours = (now - created_at).total_seconds() / 3600
            if age_hours < args.max_age_hours:
                continue
        pending.append((data.get("id"), data.get("employee"), data.get("category"), data.get("content"), created_at))

    if not pending:
        print("No pending eval decisions.")
        return

    print("Pending decisions:")
    for did, employee, category, content, created_at in pending:
        age = "unknown"
        if created_at:
            age_hours = (now - created_at).total_seconds() / 3600
            age = f"{age_hours:.1f}h"
        print(f"  - {did}: {employee} [{category}] {content} (age {age})")


if __name__ == "__main__":
    main()
