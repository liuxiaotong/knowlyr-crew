"""Tests for is_expired timezone safety."""

from datetime import datetime, timedelta, timezone

from crew.memory import MemoryEntry
from crew.memory_store_db import MemoryStoreDB


def _make_entry(created_at: str, ttl_days: int = 1) -> MemoryEntry:
    return MemoryEntry(
        employee="test",
        category="finding",
        content="test",
        created_at=created_at,
        ttl_days=ttl_days,
    )


class TestMemoryStoreDBIsExpired:
    """Tests for MemoryStoreDB.is_expired static method."""

    def test_naive_datetime_expired(self):
        """naive datetime (no tzinfo) + ttl_days=1 should be detected as expired."""
        entry = _make_entry("2026-01-01T00:00:00", ttl_days=1)
        assert MemoryStoreDB.is_expired(entry) is True

    def test_aware_datetime_expired(self):
        """aware datetime + ttl_days=1 should be detected as expired."""
        entry = _make_entry("2026-01-01T00:00:00+00:00", ttl_days=1)
        assert MemoryStoreDB.is_expired(entry) is True

    def test_ttl_zero_never_expires(self):
        """ttl_days=0 should never expire."""
        entry = _make_entry("2026-01-01T00:00:00", ttl_days=0)
        assert MemoryStoreDB.is_expired(entry) is False

    def test_future_created_at_not_expired(self):
        """Future created_at + ttl_days=30 should not be expired."""
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        entry = _make_entry(future, ttl_days=30)
        assert MemoryStoreDB.is_expired(entry) is False
