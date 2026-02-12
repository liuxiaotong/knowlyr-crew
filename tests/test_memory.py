"""持久化记忆模块测试."""

from crew.memory import MemoryEntry, MemoryStore


class TestMemoryEntry:
    """测试记忆条目模型."""

    def test_default_fields(self):
        entry = MemoryEntry(employee="code-reviewer", category="finding", content="test")
        assert entry.employee == "code-reviewer"
        assert entry.category == "finding"
        assert entry.confidence == 1.0
        assert entry.superseded_by == ""
        assert entry.id  # 自动生成

    def test_custom_fields(self):
        entry = MemoryEntry(
            employee="test-engineer",
            category="estimate",
            content="CSS 拆分需要 2 天",
            confidence=0.8,
        )
        assert entry.confidence == 0.8


class TestMemoryStore:
    """测试记忆存储."""

    def test_add_and_query(self, tmp_path):
        store = MemoryStore(memory_dir=tmp_path / "memory")
        store.add("code-reviewer", "finding", "发现 main.css 有 2057 行")
        store.add("code-reviewer", "decision", "建议拆分 CSS")

        entries = store.query("code-reviewer")
        assert len(entries) == 2
        # 最新在前
        assert entries[0].category == "decision"
        assert entries[1].category == "finding"

    def test_query_by_category(self, tmp_path):
        store = MemoryStore(memory_dir=tmp_path / "memory")
        store.add("code-reviewer", "finding", "发现1")
        store.add("code-reviewer", "decision", "决策1")
        store.add("code-reviewer", "finding", "发现2")

        findings = store.query("code-reviewer", category="finding")
        assert len(findings) == 2
        assert all(e.category == "finding" for e in findings)

    def test_query_empty(self, tmp_path):
        store = MemoryStore(memory_dir=tmp_path / "memory")
        entries = store.query("nonexistent")
        assert entries == []

    def test_query_limit(self, tmp_path):
        store = MemoryStore(memory_dir=tmp_path / "memory")
        for i in range(10):
            store.add("code-reviewer", "finding", f"发现 {i}")

        entries = store.query("code-reviewer", limit=3)
        assert len(entries) == 3

    def test_correct(self, tmp_path):
        store = MemoryStore(memory_dir=tmp_path / "memory")
        original = store.add("refactor-guide", "estimate", "CSS 拆分 2 天")

        correction = store.correct(
            "refactor-guide",
            old_id=original.id,
            new_content="CSS 拆分实际花了 5 天，未来类似任务 ×2.5",
        )
        assert correction is not None
        assert correction.category == "correction"
        assert "5 天" in correction.content

        # 旧记忆被标记为 superseded，不出现在查询结果中
        entries = store.query("refactor-guide")
        assert len(entries) == 1
        assert entries[0].id == correction.id

    def test_correct_nonexistent(self, tmp_path):
        store = MemoryStore(memory_dir=tmp_path / "memory")
        result = store.correct("code-reviewer", "nonexistent-id", "new content")
        assert result is None

    def test_format_for_prompt(self, tmp_path):
        store = MemoryStore(memory_dir=tmp_path / "memory")
        store.add("code-reviewer", "finding", "main.css 有 2057 行")
        store.add("code-reviewer", "decision", "应该拆分 CSS")

        text = store.format_for_prompt("code-reviewer")
        assert "发现" in text or "finding" in text
        assert "决策" in text or "decision" in text
        assert "2057" in text

    def test_format_for_prompt_empty(self, tmp_path):
        store = MemoryStore(memory_dir=tmp_path / "memory")
        text = store.format_for_prompt("nonexistent")
        assert text == ""

    def test_list_employees(self, tmp_path):
        store = MemoryStore(memory_dir=tmp_path / "memory")
        store.add("code-reviewer", "finding", "test1")
        store.add("test-engineer", "finding", "test2")

        employees = store.list_employees()
        assert "code-reviewer" in employees
        assert "test-engineer" in employees

    def test_list_employees_empty(self, tmp_path):
        store = MemoryStore(memory_dir=tmp_path / "memory")
        assert store.list_employees() == []

    def test_min_confidence_filter(self, tmp_path):
        store = MemoryStore(memory_dir=tmp_path / "memory")
        store.add("code-reviewer", "finding", "高置信度", confidence=1.0)
        store.add("code-reviewer", "finding", "低置信度", confidence=0.3)

        high_conf = store.query("code-reviewer", min_confidence=0.5)
        assert len(high_conf) == 1
        assert high_conf[0].content == "高置信度"
