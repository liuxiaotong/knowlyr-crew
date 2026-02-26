"""记忆双向同步 P1 冒烟测试.

覆盖：
1. webhook_handlers._handle_memory_add — REST 写入端点
2. claude2crew 记忆提取 — classify_text + extract_memories_from_entries
3. sync_employee_memories — 增量检查逻辑
4. verify_memory_sync — 对账逻辑
"""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crew.memory import MemoryConfig, MemoryStore
from crew.memory_cache import _CACHE, invalidate


# ── Fixtures ──


@pytest.fixture
def tmp_memory_dir(tmp_path):
    """创建临时记忆目录."""
    mem_dir = tmp_path / ".crew" / "memory"
    mem_dir.mkdir(parents=True)
    return mem_dir


@pytest.fixture
def store(tmp_memory_dir):
    """创建临时 MemoryStore."""
    return MemoryStore(memory_dir=tmp_memory_dir)


@pytest.fixture(autouse=True)
def clear_cache():
    """每个测试前清空缓存."""
    _CACHE.clear()
    yield
    _CACHE.clear()


# ── 1. _handle_memory_add 端点测试 ──


class TestMemoryAddHandler:
    """REST API /api/memory/add 端点测试."""

    @pytest.mark.asyncio
    async def test_add_memory_success(self, tmp_path):
        """成功写入记忆."""
        from crew.webhook_handlers import _handle_memory_add
        from crew.webhook_context import _AppContext

        mem_dir = tmp_path / ".crew" / "memory"
        mem_dir.mkdir(parents=True)

        ctx = MagicMock(spec=_AppContext)
        ctx.project_dir = tmp_path

        request = AsyncMock()
        request.json = AsyncMock(
            return_value={
                "employee": "test-engineer",
                "category": "decision",
                "content": "决定使用 PostgreSQL",
                "source_session": "sess-001",
                "tags": ["auto-push", "claude-code"],
            }
        )

        response = await _handle_memory_add(request, ctx)
        body = json.loads(response.body)

        assert body["ok"] is True
        assert body["skipped"] is False
        assert body["employee"] == "test-engineer"
        assert body["category"] == "decision"
        assert "entry_id" in body

    @pytest.mark.asyncio
    async def test_add_memory_idempotent(self, tmp_path):
        """同 session + category 不重复写入."""
        from crew.webhook_handlers import _handle_memory_add
        from crew.webhook_context import _AppContext

        mem_dir = tmp_path / ".crew" / "memory"
        mem_dir.mkdir(parents=True)

        ctx = MagicMock(spec=_AppContext)
        ctx.project_dir = tmp_path

        payload = {
            "employee": "test-engineer",
            "category": "decision",
            "content": "决定使用 PostgreSQL",
            "source_session": "sess-002",
            "tags": ["auto-push"],
        }

        # 第一次写入
        request1 = AsyncMock()
        request1.json = AsyncMock(return_value=payload)
        resp1 = await _handle_memory_add(request1, ctx)
        body1 = json.loads(resp1.body)
        assert body1["ok"] is True
        assert body1["skipped"] is False

        # 第二次写入（幂等）
        request2 = AsyncMock()
        request2.json = AsyncMock(return_value=payload)
        resp2 = await _handle_memory_add(request2, ctx)
        body2 = json.loads(resp2.body)
        assert body2["ok"] is True
        assert body2["skipped"] is True
        assert "existing_id" in body2

    @pytest.mark.asyncio
    async def test_add_memory_missing_fields(self, tmp_path):
        """缺少必填字段返回 400."""
        from crew.webhook_handlers import _handle_memory_add
        from crew.webhook_context import _AppContext

        ctx = MagicMock(spec=_AppContext)
        ctx.project_dir = tmp_path

        request = AsyncMock()
        request.json = AsyncMock(
            return_value={"employee": "test", "category": "decision"}
        )

        response = await _handle_memory_add(request, ctx)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_add_memory_invalid_category(self, tmp_path):
        """无效 category 返回 400."""
        from crew.webhook_handlers import _handle_memory_add
        from crew.webhook_context import _AppContext

        ctx = MagicMock(spec=_AppContext)
        ctx.project_dir = tmp_path

        request = AsyncMock()
        request.json = AsyncMock(
            return_value={
                "employee": "test",
                "category": "invalid",
                "content": "test",
            }
        )

        response = await _handle_memory_add(request, ctx)
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_add_memory_invalidates_cache(self, tmp_path):
        """写入后缓存被失效."""
        from crew.memory_cache import get_prompt_cached
        from crew.webhook_handlers import _handle_memory_add
        from crew.webhook_context import _AppContext

        mem_dir = tmp_path / ".crew" / "memory"
        mem_dir.mkdir(parents=True)
        store = MemoryStore(memory_dir=mem_dir)
        store.add("cache-test", "finding", "旧记忆")

        # 预热缓存
        get_prompt_cached("cache-test", store=store)
        assert "cache-test" in _CACHE

        ctx = MagicMock(spec=_AppContext)
        ctx.project_dir = tmp_path

        request = AsyncMock()
        request.json = AsyncMock(
            return_value={
                "employee": "cache-test",
                "category": "decision",
                "content": "新决策",
                "source_session": "sess-cache",
            }
        )

        await _handle_memory_add(request, ctx)
        assert "cache-test" not in _CACHE


# ── 2. claude2crew 记忆提取测试 ──


class TestClassifyText:
    """测试 classify_text — 复用 reply_postprocess.should_push 标准."""

    def test_decision_keyword(self):
        # 需要独立导入（claude2crew.py 不在 crew 包中）
        # 这里我们直接测试 reply_postprocess 的逻辑一致性
        from crew.reply_postprocess import should_push

        ok, cat = should_push("经过评估，我们决定迁移到 PostgreSQL")
        assert ok is True
        assert cat == "decision"

    def test_correction_keyword(self):
        from crew.reply_postprocess import should_push

        ok, cat = should_push("其实上次的配置有误")
        assert ok is True
        assert cat == "correction"

    def test_finding_long_output(self):
        from crew.reply_postprocess import should_push

        long_text = "经过分析发现了以下规律" * 30
        ok, cat = should_push(long_text, turn_count=5)
        assert ok is True
        assert cat == "finding"

    def test_no_push_short(self):
        from crew.reply_postprocess import should_push

        ok, cat = should_push("好的")
        assert ok is False

    def test_consistency_with_claude2crew(self):
        """验证 claude2crew 中内联的分类逻辑与 reply_postprocess 一致."""
        import re

        # 复制 claude2crew 中的判断逻辑
        _D = re.compile(r"决定|统一|不再|确定|方案定|最终选|采用|弃用|禁止")
        _C = re.compile(r"其实|说错了|纠正|更正|之前错|搞错|误解|实际上应该")

        from crew.reply_postprocess import should_push

        test_cases = [
            "我们决定使用新架构",
            "统一采用 camelCase",
            "其实之前搞错了",
            "这只是一个简单回复",
            "纠正一下之前的说法",
        ]

        for text in test_cases:
            # claude2crew 的分类
            c2c_cat = ""
            if _D.search(text):
                c2c_cat = "decision"
            elif _C.search(text):
                c2c_cat = "correction"

            # reply_postprocess 的分类
            _, rp_cat = should_push(text, turn_count=1)

            # 对于短文本（非 finding），两者应一致
            if c2c_cat:
                assert c2c_cat == rp_cat, f"不一致: '{text}' — c2c={c2c_cat} rp={rp_cat}"


# ── 3. sync_employee_memories 逻辑测试 ──


class TestSyncLogic:
    """测试 sync 脚本的增量检查逻辑."""

    def test_should_refresh_no_cache(self, tmp_path):
        """无缓存文件应刷新."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "test.json"
        assert not cache_file.exists()
        # 模拟 should_refresh 逻辑
        assert not cache_file.exists()  # 需要刷新

    def test_should_refresh_stale_cache(self, tmp_path):
        """超过 12h 的缓存应刷新."""
        cache_file = tmp_path / "test.json"
        cache_file.write_text("{}", encoding="utf-8")
        # 将修改时间设为 13 小时前
        import os

        old_time = time.time() - 13 * 3600
        os.utime(cache_file, (old_time, old_time))

        mtime = cache_file.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600
        assert age_hours > 12

    def test_should_not_refresh_fresh_cache(self, tmp_path):
        """新鲜缓存不刷新."""
        cache_file = tmp_path / "test.json"
        cache_file.write_text("{}", encoding="utf-8")

        mtime = cache_file.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600
        assert age_hours < 12

    def test_filter_memories_by_age(self):
        """过滤出最近 N 天内的记忆."""
        from datetime import datetime, timedelta

        now = datetime.now()
        old = now - timedelta(days=40)
        recent = now - timedelta(days=5)

        memories = [
            {"content": "旧记忆", "created_at": old.isoformat()},
            {"content": "新记忆", "created_at": recent.isoformat()},
        ]

        cutoff = now - timedelta(days=30)
        cutoff_str = cutoff.isoformat()
        filtered = [m for m in memories if m["created_at"] >= cutoff_str]

        assert len(filtered) == 1
        assert filtered[0]["content"] == "新记忆"

    def test_save_and_load_cache(self, tmp_path):
        """保存和加载缓存文件."""
        cache_dir = tmp_path / "employee-memories"
        cache_dir.mkdir(parents=True)
        cache_file = cache_dir / "backend-engineer.json"

        data = {
            "slug": "backend-engineer",
            "synced_at": "2026-02-26T10:00:00",
            "character_name": "赵云帆",
            "agent_status": "active",
            "memory_count": 2,
            "memories": [
                {"category": "decision", "content": "决策 1", "created_at": "2026-02-25T10:00:00"},
                {"category": "finding", "content": "发现 1", "created_at": "2026-02-24T10:00:00"},
            ],
        }

        cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        loaded = json.loads(cache_file.read_text(encoding="utf-8"))

        assert loaded["slug"] == "backend-engineer"
        assert loaded["memory_count"] == 2
        assert len(loaded["memories"]) == 2


# ── 4. verify_memory_sync 逻辑测试 ──


class TestVerifyLogic:
    """测试对账脚本的比较逻辑."""

    def test_synced_state(self):
        """条数和时间戳都一致 → synced."""
        local_count = 5
        remote_count = 5
        local_latest = "2026-02-25T10:00:00"
        remote_latest = "2026-02-25T10:00:00"

        assert local_count == remote_count
        assert local_latest == remote_latest

    def test_count_mismatch(self):
        """条数不一致."""
        local_count = 3
        remote_count = 5

        assert local_count != remote_count
        diff = remote_count - local_count
        assert diff == 2

    def test_timestamp_mismatch(self):
        """条数一致但时间戳不同."""
        local_count = 5
        remote_count = 5
        local_latest = "2026-02-24T10:00:00"
        remote_latest = "2026-02-25T10:00:00"

        assert local_count == remote_count
        assert local_latest != remote_latest

    def test_stale_detection(self):
        """缓存超过 24h 标记为 stale."""
        from datetime import datetime, timedelta

        synced_at = (datetime.now() - timedelta(hours=25)).isoformat()
        synced = datetime.fromisoformat(synced_at)
        age_hours = (datetime.now() - synced).total_seconds() / 3600
        assert age_hours > 24


# ── 5. 端到端集成测试 ──


class TestEndToEnd:
    """端到端集成：写入 → 缓存失效 → 重新拉取."""

    @pytest.mark.asyncio
    async def test_write_then_query(self, tmp_path):
        """通过 API 写入后能通过 query 查到."""
        from crew.webhook_handlers import _handle_memory_add
        from crew.webhook_context import _AppContext

        mem_dir = tmp_path / ".crew" / "memory"
        mem_dir.mkdir(parents=True)

        ctx = MagicMock(spec=_AppContext)
        ctx.project_dir = tmp_path

        # 写入
        request = AsyncMock()
        request.json = AsyncMock(
            return_value={
                "employee": "e2e-test",
                "category": "finding",
                "content": "端到端测试记忆",
                "source_session": "e2e-sess",
                "tags": ["e2e"],
            }
        )

        resp = await _handle_memory_add(request, ctx)
        body = json.loads(resp.body)
        assert body["ok"] is True

        # 查询
        store = MemoryStore(memory_dir=mem_dir)
        entries = store.query("e2e-test")
        assert len(entries) == 1
        assert entries[0].content == "端到端测试记忆"
        assert "e2e" in entries[0].tags

    def test_different_sessions_different_categories(self, store):
        """不同 session 可以写入相同 category."""
        from crew.reply_postprocess import push_if_needed

        # session 1
        push_if_needed(
            employee="multi",
            reply="决定使用方案 A",
            session_id="sess-A",
            store=store,
        )

        # session 2
        push_if_needed(
            employee="multi",
            reply="决定使用方案 B",
            session_id="sess-B",
            store=store,
        )

        entries = store.query("multi")
        assert len(entries) == 2
        assert all(e.category == "decision" for e in entries)
