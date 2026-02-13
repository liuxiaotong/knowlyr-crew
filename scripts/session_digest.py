"""Summarize recent session logs from .crew/sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize_session(path: Path) -> dict[str, str | int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = {}
    messages = []
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = data.get("event")
        if event == "start":
            start = data
        elif event == "message":
            messages.append(data)
    summary = {
        "session_id": path.stem,
        "session_type": start.get("session_type", ""),
        "subject": start.get("subject", ""),
        "message_count": len(messages),
    }
    if messages:
        summary["first_message"] = messages[0].get("content", "")[:200]
        summary["last_message"] = messages[-1].get("content", "")[:200]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize recent sessions")
    parser.add_argument("--sessions", type=Path, default=Path(".crew/sessions"))
    parser.add_argument("-n", "--count", type=int, default=5)
    args = parser.parse_args()

    if not args.sessions.is_dir():
        raise SystemExit(f"Sessions dir not found: {args.sessions}")

    files = sorted(args.sessions.glob("*.jsonl"), reverse=True)
    targets = files[: args.count]
    if not targets:
        print("No sessions found.")
        return

    for path in targets:
        info = summarize_session(path)
        print(
            f"Session {info['session_id']} ({info['session_type']}:{info['subject']}) "
            f"messages={info['message_count']}"
        )
        if info.get("last_message"):
            print(f"  last: {info['last_message']}")


if __name__ == "__main__":
    main()
