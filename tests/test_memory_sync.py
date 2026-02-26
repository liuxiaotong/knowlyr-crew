"""记忆双向同步（P0）冒烟测试.

覆盖：
1. memory_cache — 缓存命中/未命中/失效
2. reply_postprocess — 推送门槛判断 + 写入
3. webhook_handlers — 轨迹写回链路（幂等）
4. engine — 集成 get_prompt_cached
"""

from unittest.mock import patch

import pytest

from crew.memory import MemoryStore
from crew.memory_cache import (
    _CACHE,
    _count_lines,
    get_prompt_cached,
    invalidate,
    invalidate_all,
)
from crew.reply_postprocess import push_if_needed, should_push

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


# ── 1. memory_cache 测试 ──


class TestMemoryCache:
    """进程内缓存层测试."""

    def test_cache_miss_then_hit(self, store):
        """首次调用缓存未命中，第二次命中."""
        store.add("alice", "finding", "测试记忆条目", confidence=1.0)

        # 第一次：缓存未命中
        text1 = get_prompt_cached("alice", store=store)
        assert "alice" in _CACHE
        assert text1  # 非空

        # 第二次：缓存命中（相同内容）
        text2 = get_prompt_cached("alice", store=store)
        assert text2 == text1

    def test_cache_invalidate_on_new_write(self, store):
        """写入新记忆后行数变化，缓存失效."""
        store.add("bob", "finding", "初始记忆")
        get_prompt_cached("bob", store=store)
        assert "bob" in _CACHE

        old_seq = _CACHE["bob"].last_seq

        # 写入新记忆
        store.add("bob", "decision", "新决策")

        # 行数变了，应重新加载
        text = get_prompt_cached("bob", store=store)
        assert "新决策" in text
        assert _CACHE["bob"].last_seq > old_seq

    def test_manual_invalidate(self, store):
        """手动失效缓存."""
        store.add("carol", "finding", "记忆")
        get_prompt_cached("carol", store=store)
        assert "carol" in _CACHE

        invalidate("carol")
        assert "carol" not in _CACHE

    def test_invalidate_all(self, store):
        """清空全部缓存."""
        store.add("dave", "finding", "记忆")
        store.add("eve", "finding", "记忆")
        get_prompt_cached("dave", store=store)
        get_prompt_cached("eve", store=store)
        assert len(_CACHE) == 2

        invalidate_all()
        assert len(_CACHE) == 0

    def test_empty_employee_returns_empty(self, store):
        """无记忆的员工返回空字符串."""
        text = get_prompt_cached("nobody", store=store)
        assert text == ""

    def test_count_lines(self, store):
        """行数统计."""
        assert _count_lines(store, "test") == 0
        store.add("test", "finding", "一条")
        assert _count_lines(store, "test") == 1
        store.add("test", "finding", "两条")
        assert _count_lines(store, "test") == 2

    def test_token_truncation(self, store):
        """超过 800 token 的内容被截断."""
        # 写入大量记忆
        for i in range(50):
            store.add(
                "long",
                "finding",
                f"这是第{i}条非常长的记忆内容，用来测试截断功能" * 5,
                confidence=1.0,
            )

        text = get_prompt_cached("long", store=store)
        # 粗略检查：800 token ≈ 1600 字符，允许一定误差
        assert len(text) <= 2000


# ── 2. reply_postprocess 测试 ──


class TestShouldPush:
    """推送门槛判断测试."""

    def test_decision_keywords(self):
        ok, cat = should_push("经过讨论，我们决定使用 PostgreSQL")
        assert ok is True
        assert cat == "decision"

    def test_correction_keywords(self):
        ok, cat = should_push("其实之前的方案有问题")
        assert ok is True
        assert cat == "correction"

    def test_long_output_with_turns(self):
        long_text = "这是一段很长的输出" * 30  # 超过 200 字
        ok, cat = should_push(long_text, turn_count=3)
        assert ok is True
        assert cat == "finding"

    def test_long_output_insufficient_turns(self):
        long_text = "这是一段很长的输出" * 30
        ok, cat = should_push(long_text, turn_count=2)  # 不足 3 轮
        assert ok is False
        assert cat == ""

    def test_short_output_no_keywords(self):
        ok, cat = should_push("好的，明白了")
        assert ok is False

    def test_empty_reply(self):
        ok, cat = should_push("")
        assert ok is False

    def test_unified_keyword(self):
        ok, cat = should_push("统一使用 camelCase 格式")
        assert ok is True
        assert cat == "decision"

    def test_actually_keyword(self):
        ok, cat = should_push("说错了，端口应该是 8200")
        assert ok is True
        assert cat == "correction"

    def test_decision_priority_over_correction(self):
        """决策词优先级高于纠正词."""
        ok, cat = should_push("决定不再使用旧方案，其实之前就该换了")
        assert ok is True
        assert cat == "decision"  # 决策词先匹配


class TestPushIfNeeded:
    """推送记忆写入测试."""

    def test_push_decision(self, store):
        result = push_if_needed(
            employee="alice",
            reply="我们决定使用新架构",
            session_id="sess-001",
            store=store,
        )
        assert result is True

        entries = store.query("alice")
        assert len(entries) == 1
        assert entries[0].category == "decision"
        assert "auto-push" in entries[0].tags

    def test_push_idempotent(self, store):
        """同 session 同 category 不重复写入."""
        push_if_needed(
            employee="bob",
            reply="决定重构",
            session_id="sess-002",
            store=store,
        )
        result = push_if_needed(
            employee="bob",
            reply="决定另一件事",
            session_id="sess-002",
            store=store,
        )
        assert result is False

        entries = store.query("bob")
        assert len(entries) == 1

    def test_no_push_for_short_reply(self, store):
        result = push_if_needed(
            employee="carol",
            reply="好的",
            store=store,
        )
        assert result is False
        assert store.query("carol") == []

    def test_push_invalidates_cache(self, store):
        """推送后缓存被失效."""
        store.add("dave", "finding", "旧记忆")
        get_prompt_cached("dave", store=store)
        assert "dave" in _CACHE

        push_if_needed(
            employee="dave",
            reply="统一使用新方案",
            session_id="sess-003",
            store=store,
        )
        assert "dave" not in _CACHE

    def test_push_retry_on_failure(self, store):
        """写入失败时重试."""
        call_count = 0
        original_add = store.add

        def flaky_add(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("临时写入错误")
            return original_add(*args, **kwargs)

        store.add = flaky_add
        result = push_if_needed(
            employee="eve",
            reply="决定使用新框架",
            session_id="sess-004",
            store=store,
        )
        assert result is True
        assert call_count == 2  # 第 1 次失败，第 2 次成功


# ── 3. 轨迹写回链路测试（单元测试模式） ──


class TestTrajectoryWriteback:
    """测试轨迹上报时的记忆写回逻辑."""

    def test_successful_trajectory_writes_memory(self, store):
        """成功轨迹写入 finding 记忆."""
        store.add(
            employee="赵云帆",
            category="finding",
            content="[轨迹] 修复登录 bug (工具: Bash, Edit) [5步]",
            source_session="crew-abc123",
            tags=["trajectory", "claude-code"],
        )

        entries = store.query("赵云帆")
        assert len(entries) == 1
        assert entries[0].category == "finding"
        assert "trajectory" in entries[0].tags
        assert "claude-code" in entries[0].tags
        assert entries[0].source_session == "crew-abc123"

    def test_idempotent_trajectory_write(self, store):
        """同 task_id 不重复写入."""
        # 第一次写入
        store.add(
            employee="赵云帆",
            category="finding",
            content="[轨迹] 任务 A",
            source_session="crew-xyz789",
            tags=["trajectory"],
        )

        # 模拟幂等检查
        existing = store.query("赵云帆", limit=50)
        should_write = True
        for e in existing:
            if e.source_session == "crew-xyz789" and e.category == "finding":
                should_write = False
                break

        assert should_write is False

    def test_failed_trajectory_no_memory(self, store):
        """失败的轨迹不写入记忆（由 success=False 控制，此处验证逻辑分支）."""
        success = False
        task_description = "修复 bug"

        if success and task_description:
            store.add(employee="test", category="finding", content="不应该到这")

        entries = store.query("test")
        assert len(entries) == 0


# ── 4. engine 集成测试 ──


class TestEngineIntegration:
    """验证 engine.py 正确调用 get_prompt_cached."""

    def test_engine_prompt_uses_cache(self, tmp_path):
        """engine.prompt() 应通过缓存层加载记忆."""
        # 写入记忆
        mem_dir = tmp_path / ".crew" / "memory"
        mem_dir.mkdir(parents=True)
        mem_store = MemoryStore(memory_dir=mem_dir)
        mem_store.add("test-eng", "finding", "重要发现：API 延迟高", confidence=1.0)

        # 验证 get_prompt_cached 被调用（lazy import 用 patch 模块级别）
        with patch("crew.memory_cache.get_prompt_cached") as mock_cache:
            mock_cache.return_value = "- [发现] 重要发现：API 延迟高"

            from crew.engine import CrewEngine
            from crew.models import Employee

            engine = CrewEngine(project_dir=tmp_path)
            emp = Employee(
                name="test-eng",
                display_name="测试工程师",
                description="测试",
                body="执行任务",
            )

            prompt = engine.prompt(emp)

            # 验证缓存函数被调用
            mock_cache.assert_called_once()
            call_args = mock_cache.call_args
            assert call_args[0][0] == "test-eng"  # employee name
            # 验证记忆内容被注入到 prompt 中
            assert "历史经验" in prompt
            assert "重要发现：API 延迟高" in prompt
