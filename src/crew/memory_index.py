"""Hybrid memory/search索引 — 结合持久记忆与会话记录."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crew.memory import MemoryStore
from crew.paths import resolve_project_dir

logger = logging.getLogger(__name__)


def _prepare_index_text(text: str) -> str:
    """为 FTS 预处理文本，拆分 CJK 字符."""
    pieces: list[str] = []
    for ch in text:
        if ord(ch) > 127:
            pieces.extend([" ", ch, " "])
        else:
            pieces.append(ch)
    return "".join(pieces)


@dataclass
class IndexStats:
    memory_entries: int = 0
    session_messages: int = 0

    def total(self) -> int:
        return self.memory_entries + self.session_messages


class MemorySearchIndex:
    """基于 SQLite FTS 的检索索引."""

    def __init__(
        self,
        db_path: Path | None = None,
        memory_dir: Path | None = None,
        session_dir: Path | None = None,
        *,
        project_dir: Path | None = None,
    ) -> None:
        root = resolve_project_dir(project_dir)
        self.db_path = db_path or root / ".crew" / "memory-index.db"
        self.memory_dir = memory_dir or root / ".crew" / "memory"
        self.session_dir = session_dir or root / ".crew" / "sessions"

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _setup_tables(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                employee TEXT,
                kind TEXT,
                source TEXT,
                title TEXT,
                content TEXT,
                metadata TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts
            USING fts5(content, title, employee, tokenize='unicode61')
            """
        )

    def rebuild(self) -> IndexStats:
        """重建索引，返回计数."""
        conn = self._connect()
        try:
            self._setup_tables(conn)
            conn.execute("DELETE FROM entries")
            conn.execute("DELETE FROM entries_fts")
            stats = IndexStats()
            stats.memory_entries = self._index_memory(conn)
            stats.session_messages = self._index_sessions(conn)
            conn.commit()
            return stats
        finally:
            conn.close()

    def _insert_entry(
        self,
        conn: sqlite3.Connection,
        *,
        entry_id: str,
        employee: str,
        kind: str,
        source: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None,
        created_at: str,
    ) -> None:
        payload = (
            entry_id,
            employee,
            kind,
            source,
            title,
            content,
            json.dumps(metadata or {}, ensure_ascii=False),
            created_at,
        )
        conn.execute("INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?, ?, ?, ?, ?)", payload)
        conn.execute(
            "INSERT INTO entries_fts(rowid, content, title, employee)"
            " VALUES ((SELECT rowid FROM entries WHERE id=?), ?, ?, ?)",
            (
                entry_id,
                _prepare_index_text(content),
                _prepare_index_text(title),
                employee,
            ),
        )

    def _index_memory(self, conn: sqlite3.Connection) -> int:
        store = MemoryStore(memory_dir=self.memory_dir)
        total = 0
        for employee in store.list_employees():
            entries = store.query(employee, limit=10_000)
            for entry in entries:
                title = f"[{entry.category}] {entry.content[:40]}"
                self._insert_entry(
                    conn,
                    entry_id=f"memory:{entry.id}",
                    employee=employee,
                    kind="memory",
                    source=entry.source_session,
                    title=title,
                    content=entry.content,
                    metadata={"category": entry.category, "source_session": entry.source_session},
                    created_at=entry.created_at,
                )
                total += 1
        return total

    def _index_sessions(self, conn: sqlite3.Connection) -> int:
        if not self.session_dir.is_dir():
            return 0
        total = 0
        for path in sorted(self.session_dir.glob("*.jsonl")):
            session_id = path.stem
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as e:
                logger.debug("读取 session 文件失败 %s: %s", path.name, e)
                continue
            session_subject = ""
            start_metadata: dict[str, Any] = {}
            for idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("event") == "start" and not session_subject:
                    session_subject = data.get("subject", "")
                    start_metadata = data.get("metadata", {}) or {}
                    continue
                if data.get("event") != "message":
                    continue
                role = data.get("role", "")
                content = data.get("content", "")
                if not content:
                    continue
                metadata = data.get("metadata", {}) or {}
                employee = (
                    metadata.get("employee") or start_metadata.get("employee") or session_subject
                )
                title = f"{session_id} ({role})"
                entry_id = f"session:{session_id}:{idx}"
                self._insert_entry(
                    conn,
                    entry_id=entry_id,
                    employee=employee or "",
                    kind="session",
                    source=session_id,
                    title=title,
                    content=content,
                    metadata=metadata,
                    created_at=data.get("timestamp", ""),
                )
                total += 1
        return total

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        employee: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            self._setup_tables(conn)
            fts_query = _prepare_index_text(query).strip()
            if not fts_query:
                return []
            sql = (
                "SELECT e.id, e.employee, e.kind, e.source, e.title, e.content, "
                "e.metadata, e.created_at, snippet(entries_fts, 0, '[', ']', '…', 10) as snippet "
                "FROM entries_fts JOIN entries e ON e.rowid = entries_fts.rowid "
                "WHERE entries_fts MATCH ?"
            )
            params: list[Any] = [fts_query]
            if employee:
                sql += " AND e.employee = ?"
                params.append(employee)
            if kind:
                sql += " AND e.kind = ?"
                params.append(kind)
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            results: list[dict[str, Any]] = []
            for row in rows:
                metadata = row["metadata"]
                try:
                    metadata_obj = json.loads(metadata) if metadata else {}
                except json.JSONDecodeError:
                    metadata_obj = {}
                results.append(
                    {
                        "id": row["id"],
                        "employee": row["employee"],
                        "kind": row["kind"],
                        "source": row["source"],
                        "title": row["title"],
                        "content": row["content"],
                        "metadata": metadata_obj,
                        "created_at": row["created_at"],
                        "snippet": row["snippet"],
                    }
                )
            return results
        finally:
            conn.close()


__all__ = ["MemorySearchIndex", "IndexStats"]
