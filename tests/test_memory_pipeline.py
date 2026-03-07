"""Tests for memory pipeline (Reflect -> Connect -> Store)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from crew.memory import MemoryEntry
from crew.memory_pipeline import (
    ConnectResult,
    ReflectResult,
    _keyword_overlap,
    connect,
    process_memory,
    reflect,
)

# ── Fixtures ──


def _make_entry(
    id: str = "abc123",
    employee: str = "测试员工",
    category: str = "finding",
    content: str = "测试内容",
    keywords: list[str] | None = None,
    linked_memories: list[str] | None = None,
    **kwargs,
) -> MemoryEntry:
    """创建测试用 MemoryEntry."""
    return MemoryEntry(
        id=id,
        employee=employee,
        category=category,
        content=content,
        keywords=keywords or [],
        linked_memories=linked_memories or [],
        **kwargs,
    )


def _make_mock_store() -> MagicMock:
    """创建 mock MemoryStoreDB."""
    store = MagicMock()
    store._tenant_id = "test-tenant"
    store._resolve_to_character_name = MagicMock(side_effect=lambda x: x)

    # 默认 add 返回新 entry
    def mock_add(employee, category, content, tags=None, **kw):
        return _make_entry(
            id="new123",
            employee=employee,
            category=category,
            content=content,
        )

    store.add = MagicMock(side_effect=mock_add)
    store.update = MagicMock(return_value=True)
    store.update_keywords = MagicMock(return_value=True)
    store.update_linked_memories = MagicMock(return_value=True)
    return store


# ── Test: keyword_overlap ──


class TestKeywordOverlap:
    def test_full_overlap(self):
        assert _keyword_overlap(["a", "b"], ["a", "b"]) == 1.0

    def test_no_overlap(self):
        assert _keyword_overlap(["a", "b"], ["c", "d"]) == 0.0

    def test_partial_overlap(self):
        result = _keyword_overlap(["a", "b", "c"], ["b", "c", "d"])
        assert 0.4 < result < 0.6  # 2/4 = 0.5

    def test_empty_lists(self):
        assert _keyword_overlap([], ["a"]) == 0.0
        assert _keyword_overlap(["a"], []) == 0.0
        assert _keyword_overlap([], []) == 0.0

    def test_case_insensitive(self):
        assert _keyword_overlap(["API", "Key"], ["api", "key"]) == 1.0


# ── Test: Reflect ──


class TestReflect:
    @patch("crew.memory_pipeline._call_llm")
    def test_reflect_extracts_structured_note(self, mock_llm):
        """Reflect 正确提取结构化笔记."""
        mock_llm.return_value = json.dumps(
            {
                "store": True,
                "content": "发现 API 限流阈值为 100 req/min",
                "category": "finding",
                "keywords": ["API", "限流", "阈值"],
                "tags": ["infra"],
                "context": "排查超时问题时发现",
            }
        )

        result = reflect("在排查 API 超时问题时发现限流阈值设为 100", "赵云帆")

        assert result is not None
        assert result.store is True
        assert "API" in result.content
        assert result.category == "finding"
        assert "API" in result.keywords
        assert "infra" in result.tags
        assert result.context != ""
        mock_llm.assert_called_once()

    @patch("crew.memory_pipeline._call_llm")
    def test_reflect_skip_decision(self, mock_llm):
        """LLM 返回 store=false 时，Reflect 返回 None."""
        mock_llm.return_value = json.dumps(
            {
                "store": False,
                "content": "",
                "category": "finding",
                "keywords": [],
                "tags": [],
                "context": "",
            }
        )

        result = reflect("我打开了文件看了一下", "赵云帆")
        assert result is None

    @patch("crew.memory_pipeline._call_llm")
    def test_reflect_truncates_long_input(self, mock_llm):
        """超过 2000 字的输入被截断."""
        mock_llm.return_value = json.dumps(
            {
                "store": True,
                "content": "长文本",
                "category": "finding",
                "keywords": [],
                "tags": [],
                "context": "",
            }
        )
        long_text = "x" * 3000
        reflect(long_text, "赵云帆")

        # 验证发给 LLM 的 prompt 中文本被截断到 2000 字
        call_args = mock_llm.call_args[0][0]
        assert "x" * 2000 in call_args
        assert "x" * 2001 not in call_args

    @patch("crew.memory_pipeline._call_llm")
    def test_reflect_handles_markdown_wrapped_json(self, mock_llm):
        """处理 LLM 返回 markdown 代码块包裹的 JSON."""
        mock_llm.return_value = '```json\n{"store": true, "content": "test", "category": "decision", "keywords": ["k1"], "tags": [], "context": ""}\n```'

        result = reflect("决策记录", "赵云帆")
        assert result is not None
        assert result.category == "decision"

    @patch("crew.memory_pipeline._call_llm")
    def test_reflect_handles_invalid_category(self, mock_llm):
        """无效 category 降级为 finding."""
        mock_llm.return_value = json.dumps(
            {
                "store": True,
                "content": "test",
                "category": "invalid_cat",
                "keywords": [],
                "tags": [],
                "context": "",
            }
        )

        result = reflect("某条记录", "赵云帆")
        assert result is not None
        assert result.category == "finding"

    @patch("crew.memory_pipeline._call_llm")
    def test_reflect_handles_llm_error(self, mock_llm):
        """LLM 调用失败时返回 None."""
        mock_llm.side_effect = Exception("API error")
        result = reflect("test", "赵云帆")
        assert result is None


# ── Test: Connect ──


class TestConnect:
    def test_connect_new_no_candidates(self):
        """无候选时走 new 逻辑."""
        note = ReflectResult(
            store=True,
            content="全新发现",
            category="finding",
            keywords=["新关键词"],
            tags=[],
        )
        store = _make_mock_store()

        with patch("crew.memory_pipeline._find_candidates_by_keywords", return_value=[]):
            result = connect(note, "赵云帆", store)

        assert result.action == "new"
        store.add.assert_called_once()

    def test_connect_merge_high_overlap(self):
        """高重叠 + 同 category -> merge."""
        existing = _make_entry(
            id="exist1",
            category="finding",
            content="旧发现",
            keywords=["API", "限流", "阈值"],
        )
        note = ReflectResult(
            store=True,
            content="新发现补充",
            category="finding",
            keywords=["API", "限流", "阈值"],  # 100% 重叠
            tags=[],
        )
        store = _make_mock_store()

        with patch(
            "crew.memory_pipeline._find_candidates_by_keywords",
            return_value=[existing],
        ):
            result = connect(note, "赵云帆", store)

        assert result.action == "merge"
        assert result.merged_entry_id == "exist1"
        # 验证调了 update
        store.update.assert_called_once()
        store.update_keywords.assert_called_once()
        # 未调 add（merge 不创建新条目）
        store.add.assert_not_called()

    def test_connect_link_partial_overlap(self):
        """30%-70% 重叠 -> link."""
        existing = _make_entry(
            id="exist2",
            category="finding",
            content="已有记忆",
            keywords=["API", "限流", "阈值", "超时"],  # 4 个
        )
        note = ReflectResult(
            store=True,
            content="部分相关发现",
            category="decision",  # 不同 category 也触发 link
            keywords=["API", "限流", "监控", "告警"],  # 2/6 = 33%
            tags=[],
        )
        store = _make_mock_store()

        with patch(
            "crew.memory_pipeline._find_candidates_by_keywords",
            return_value=[existing],
        ):
            result = connect(note, "赵云帆", store)

        assert result.action == "link"
        # 验证双向 linked_memories 更新
        assert store.update_linked_memories.call_count == 2
        # 验证新记忆被创建
        store.add.assert_called_once()

    def test_connect_new_low_overlap(self):
        """< 30% 重叠 -> new."""
        existing = _make_entry(
            id="exist3",
            category="finding",
            content="完全不同的记忆",
            keywords=["数据库", "迁移", "PostgreSQL", "备份"],
        )
        note = ReflectResult(
            store=True,
            content="全新话题",
            category="finding",
            keywords=["前端", "React", "组件"],  # 0% 重叠
            tags=[],
        )
        store = _make_mock_store()

        with patch(
            "crew.memory_pipeline._find_candidates_by_keywords",
            return_value=[existing],
        ):
            result = connect(note, "赵云帆", store)

        assert result.action == "new"
        store.add.assert_called_once()

    def test_connect_high_overlap_different_category_links(self):
        """高重叠但 category 不同 -> link（不是 merge）."""
        existing = _make_entry(
            id="exist4",
            category="finding",
            content="发现",
            keywords=["API", "限流"],
        )
        note = ReflectResult(
            store=True,
            content="决策",
            category="decision",  # 不同 category
            keywords=["API", "限流"],  # 100% 重叠
            tags=[],
        )
        store = _make_mock_store()

        with patch(
            "crew.memory_pipeline._find_candidates_by_keywords",
            return_value=[existing],
        ):
            result = connect(note, "赵云帆", store)

        assert result.action == "link"


# ── Test: process_memory ──


class TestProcessMemory:
    @patch("crew.memory_pipeline.reflect")
    @patch("crew.memory_pipeline.connect")
    def test_process_memory_full_pipeline(self, mock_connect, mock_reflect):
        """完整管线冒烟测试."""
        note = ReflectResult(
            store=True,
            content="提炼后的内容",
            category="finding",
            keywords=["k1"],
            tags=["t1"],
        )
        mock_reflect.return_value = note

        entry = _make_entry(content="提炼后的内容")
        mock_connect.return_value = ConnectResult(action="new", entry=entry)

        store = _make_mock_store()
        result = process_memory("原始文本", "赵云帆", store=store)

        assert result is not None
        assert result.content == "提炼后的内容"
        mock_reflect.assert_called_once_with("原始文本", "赵云帆")
        mock_connect.assert_called_once_with(note, "赵云帆", store)

    @patch("crew.memory_pipeline.reflect")
    def test_process_memory_reflect_skip(self, mock_reflect):
        """Reflect 返回 None 时，管线返回 None."""
        mock_reflect.return_value = None
        store = _make_mock_store()

        result = process_memory("不值得存的文本", "赵云帆", store=store)
        assert result is None

    @patch("crew.memory_pipeline.connect")
    def test_process_memory_skip_reflect(self, mock_connect):
        """skip_reflect=True 跳过 Reflect 阶段."""
        entry = _make_entry(content="已结构化内容")
        mock_connect.return_value = ConnectResult(action="new", entry=entry)

        store = _make_mock_store()
        result = process_memory(
            "原始文本",
            "赵云帆",
            store=store,
            skip_reflect=True,
            content="已结构化内容",
            category="decision",
            keywords=["kw1", "kw2"],
            tags=["tag1"],
        )

        assert result is not None
        # 验证 connect 被调用，且 note 中包含传入的参数
        call_args = mock_connect.call_args
        note = call_args[0][0]
        assert note.content == "已结构化内容"
        assert note.category == "decision"
        assert note.keywords == ["kw1", "kw2"]

    @patch("crew.memory_pipeline.connect")
    def test_process_memory_skip_reflect_defaults(self, mock_connect):
        """skip_reflect=True 时使用默认值."""
        entry = _make_entry()
        mock_connect.return_value = ConnectResult(action="new", entry=entry)

        store = _make_mock_store()
        process_memory(
            "原始文本",
            "赵云帆",
            store=store,
            skip_reflect=True,
        )

        call_args = mock_connect.call_args
        note = call_args[0][0]
        assert note.content == "原始文本"  # 默认用 raw_text
        assert note.category == "finding"  # 默认 category

    @patch("crew.memory_pipeline.connect")
    def test_process_memory_kwargs_passthrough(self, mock_connect):
        """store.add() 参数通过 kwargs 透传到 connect."""
        entry = _make_entry()
        mock_connect.return_value = ConnectResult(action="new", entry=entry)

        store = _make_mock_store()
        process_memory(
            "原始文本",
            "赵云帆",
            store=store,
            skip_reflect=True,
            category="decision",
            source_session="sess-42",
            ttl_days=30,
            shared=True,
            confidence=0.9,
        )

        call_kwargs = mock_connect.call_args[1]
        assert call_kwargs["source_session"] == "sess-42"
        assert call_kwargs["ttl_days"] == 30
        assert call_kwargs["shared"] is True
        assert call_kwargs["confidence"] == 0.9


# ── Test: DB fields (keywords + linked_memories) ──


class TestDBFields:
    def test_memory_entry_keywords_default(self):
        """MemoryEntry keywords 默认为空列表."""
        entry = MemoryEntry(
            employee="测试",
            category="finding",
            content="test",
        )
        assert entry.keywords == []
        assert entry.linked_memories == []

    def test_memory_entry_keywords_set(self):
        """MemoryEntry keywords 可以正常设置和读取."""
        entry = MemoryEntry(
            employee="测试",
            category="finding",
            content="test",
            keywords=["k1", "k2"],
            linked_memories=["id1", "id2"],
        )
        assert entry.keywords == ["k1", "k2"]
        assert entry.linked_memories == ["id1", "id2"]

    def test_memory_entry_json_roundtrip(self):
        """keywords 和 linked_memories 可以 JSON 序列化和反序列化."""
        entry = MemoryEntry(
            employee="测试",
            category="finding",
            content="test",
            keywords=["API", "限流"],
            linked_memories=["abc123"],
        )
        data = json.loads(entry.model_dump_json())
        restored = MemoryEntry(**data)
        assert restored.keywords == ["API", "限流"]
        assert restored.linked_memories == ["abc123"]

    def test_db_keywords_and_linked_memories(self):
        """验证 DB store 的 update_keywords / update_linked_memories mock 行为."""
        store = _make_mock_store()

        store.update_keywords("entry1", "员工A", ["kw1", "kw2"])
        store.update_keywords.assert_called_once_with("entry1", "员工A", ["kw1", "kw2"])

        store.update_linked_memories("entry1", "员工A", ["linked1"])
        store.update_linked_memories.assert_called_once_with("entry1", "员工A", ["linked1"])
