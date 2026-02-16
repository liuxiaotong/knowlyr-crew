"""持久化记忆模块测试."""

import json
from datetime import datetime, timedelta

from crew.memory import MemoryConfig, MemoryEntry, MemoryStore


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

    def test_new_fields_defaults(self):
        """新增字段有合理默认值，向后兼容."""
        entry = MemoryEntry(employee="pm", category="decision", content="test")
        assert entry.ttl_days == 0
        assert entry.tags == []
        assert entry.shared is False

    def test_new_fields_custom(self):
        entry = MemoryEntry(
            employee="pm", category="decision", content="test",
            ttl_days=30, tags=["architecture", "api"], shared=True,
        )
        assert entry.ttl_days == 30
        assert entry.tags == ["architecture", "api"]
        assert entry.shared is True

    def test_backward_compat_json(self):
        """旧格式 JSON（无新字段）可正常解析."""
        old_json = '{"id":"abc","employee":"pm","category":"finding","content":"old","confidence":1.0,"superseded_by":""}'
        entry = MemoryEntry(**json.loads(old_json))
        assert entry.ttl_days == 0
        assert entry.tags == []
        assert entry.shared is False


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

    def test_query_skips_corrupted_entries(self, tmp_path):
        """损坏的 JSON 行被跳过，不影响其它条目."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        store.add("code-reviewer", "finding", "正常条目")

        # 追加损坏行
        path = tmp_path / "memory" / "code-reviewer.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write("{broken json\n")

        entries = store.query("code-reviewer")
        assert len(entries) == 1
        assert entries[0].content == "正常条目"

    def test_correct_skips_corrupted_entries(self, tmp_path):
        """correct() 中损坏行被保留原文."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        entry = store.add("code-reviewer", "finding", "待纠正")

        # 追加损坏行
        path = tmp_path / "memory" / "code-reviewer.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write("not-json-line\n")

        result = store.correct("code-reviewer", entry.id, "已纠正")
        assert result is not None
        assert result.content == "已纠正"

        # 验证损坏行保留
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert any("not-json-line" in l for l in lines)

    def test_format_for_prompt_semantic_fallback(self, tmp_path):
        """语义搜索失败时降级到普通查询."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        store.add("code-reviewer", "finding", "安全审查经验")

        # 语义搜索不可用时应降级
        text = store.format_for_prompt("code-reviewer", query="安全")
        assert "安全审查经验" in text

    def test_add_with_new_fields(self, tmp_path):
        """add() 接受新参数 ttl_days/tags/shared."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        entry = store.add(
            "pm", "decision", "使用 REST API",
            ttl_days=30, tags=["architecture", "api"], shared=True,
        )
        assert entry.ttl_days == 30
        assert entry.tags == ["architecture", "api"]
        assert entry.shared is True

        # 查询回来也有新字段
        entries = store.query("pm")
        assert entries[0].tags == ["architecture", "api"]
        assert entries[0].shared is True

    def test_add_default_ttl_from_config(self, tmp_path):
        """config.default_ttl_days 在 ttl_days=0 时生效."""
        config = MemoryConfig(default_ttl_days=60)
        store = MemoryStore(memory_dir=tmp_path / "memory", config=config)
        entry = store.add("pm", "finding", "test")
        assert entry.ttl_days == 60

    def test_add_explicit_ttl_overrides_config(self, tmp_path):
        """显式 ttl_days 覆盖 config 默认值."""
        config = MemoryConfig(default_ttl_days=60)
        store = MemoryStore(memory_dir=tmp_path / "memory", config=config)
        entry = store.add("pm", "finding", "test", ttl_days=7)
        assert entry.ttl_days == 7


class TestMemoryConfig:
    """测试记忆配置."""

    def test_default_config(self):
        config = MemoryConfig()
        assert config.default_ttl_days == 0
        assert config.max_entries_per_employee == 500
        assert config.confidence_half_life_days == 90.0
        assert config.auto_index is True

    def test_load_config_from_file(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        config_path = memory_dir / "config.json"
        config_path.write_text(json.dumps({
            "default_ttl_days": 30,
            "max_entries_per_employee": 100,
        }), encoding="utf-8")

        store = MemoryStore(memory_dir=memory_dir)
        assert store.config.default_ttl_days == 30
        assert store.config.max_entries_per_employee == 100
        # 未指定的保持默认
        assert store.config.confidence_half_life_days == 90.0

    def test_config_missing_file_uses_defaults(self, tmp_path):
        store = MemoryStore(memory_dir=tmp_path / "memory")
        assert store.config.default_ttl_days == 0

    def test_config_corrupted_file_uses_defaults(self, tmp_path):
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "config.json").write_text("{broken", encoding="utf-8")

        store = MemoryStore(memory_dir=memory_dir)
        assert store.config.default_ttl_days == 0


class TestMemoryDecay:
    """测试记忆衰减与 TTL."""

    def _make_old_entry(self, store, employee, content, days_ago, **kwargs):
        """写入一条带指定创建时间的记忆."""
        entry = store.add(employee, "finding", content, **kwargs)
        # 手动修改创建时间
        path = store._employee_file(employee)
        lines = path.read_text(encoding="utf-8").splitlines()
        new_lines = []
        old_time = (datetime.now() - timedelta(days=days_ago)).isoformat()
        for line in lines:
            if entry.id in line:
                data = json.loads(line)
                data["created_at"] = old_time
                new_lines.append(json.dumps(data, ensure_ascii=False))
            else:
                new_lines.append(line)
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return entry

    def test_ttl_expired_entries_skipped(self, tmp_path):
        """过期条目不出现在查询结果中."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        self._make_old_entry(store, "pm", "旧记忆", days_ago=31, ttl_days=30)
        store.add("pm", "finding", "新记忆")

        entries = store.query("pm")
        assert len(entries) == 1
        assert entries[0].content == "新记忆"

    def test_ttl_zero_never_expires(self, tmp_path):
        """ttl_days=0 的记忆永不过期."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        self._make_old_entry(store, "pm", "永久记忆", days_ago=365)

        entries = store.query("pm")
        assert len(entries) == 1

    def test_include_expired_flag(self, tmp_path):
        """include_expired=True 返回过期条目."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        self._make_old_entry(store, "pm", "过期记忆", days_ago=31, ttl_days=30)

        entries = store.query("pm", include_expired=True)
        assert len(entries) == 1

    def test_confidence_decay_by_age(self, tmp_path):
        """置信度随时间衰减."""
        config = MemoryConfig(confidence_half_life_days=90.0)
        store = MemoryStore(memory_dir=tmp_path / "memory", config=config)

        self._make_old_entry(store, "pm", "90天前的记忆", days_ago=90)

        entries = store.query("pm")
        assert len(entries) == 1
        # 90 天 = 一个半衰期，置信度应约 0.5
        assert 0.4 < entries[0].confidence < 0.6

    def test_decay_half_life_configurable(self, tmp_path):
        """半衰期可配置."""
        config = MemoryConfig(confidence_half_life_days=30.0)
        store = MemoryStore(memory_dir=tmp_path / "memory", config=config)

        self._make_old_entry(store, "pm", "30天前", days_ago=30)

        entries = store.query("pm")
        # 30 天 = 一个半衰期
        assert 0.4 < entries[0].confidence < 0.6

    def test_min_confidence_with_decay(self, tmp_path):
        """衰减后低于 min_confidence 的被过滤."""
        config = MemoryConfig(confidence_half_life_days=30.0)
        store = MemoryStore(memory_dir=tmp_path / "memory", config=config)

        # 60 天 = 两个半衰期 → confidence ≈ 0.25
        self._make_old_entry(store, "pm", "很旧的记忆", days_ago=60)
        store.add("pm", "finding", "新记忆")

        entries = store.query("pm", min_confidence=0.3)
        assert len(entries) == 1
        assert entries[0].content == "新记忆"


class TestCapacityControl:
    """测试容量控制."""

    def test_enforce_capacity_prunes_oldest(self, tmp_path):
        """超出限额时裁剪低置信度条目."""
        config = MemoryConfig(max_entries_per_employee=5)
        store = MemoryStore(memory_dir=tmp_path / "memory", config=config)

        for i in range(7):
            store.add("pm", "finding", f"记忆 {i}")

        entries = store.query("pm")
        assert len(entries) == 5

    def test_capacity_preserves_high_confidence(self, tmp_path):
        """高置信度条目优先保留."""
        config = MemoryConfig(max_entries_per_employee=3, confidence_half_life_days=9999)
        store = MemoryStore(memory_dir=tmp_path / "memory", config=config)

        store.add("pm", "finding", "低置信", confidence=0.1)
        store.add("pm", "finding", "高置信1", confidence=1.0)
        store.add("pm", "finding", "高置信2", confidence=0.9)
        store.add("pm", "finding", "高置信3", confidence=0.8)

        entries = store.query("pm")
        assert len(entries) == 3
        contents = {e.content for e in entries}
        assert "低置信" not in contents
        assert "高置信1" in contents

    def test_capacity_zero_means_unlimited(self, tmp_path):
        """max_entries=0 不限制."""
        config = MemoryConfig(max_entries_per_employee=0)
        store = MemoryStore(memory_dir=tmp_path / "memory", config=config)

        for i in range(20):
            store.add("pm", "finding", f"记忆 {i}")

        entries = store.query("pm")
        assert len(entries) == 20

    def test_prune_preserves_superseded(self, tmp_path):
        """裁剪不影响 superseded 条目（它们已被隐藏）."""
        config = MemoryConfig(max_entries_per_employee=2)
        store = MemoryStore(memory_dir=tmp_path / "memory", config=config)

        e1 = store.add("pm", "finding", "原始")
        store.correct("pm", e1.id, "纠正后")
        store.add("pm", "finding", "第二条")
        store.add("pm", "finding", "第三条")

        entries = store.query("pm")
        assert len(entries) == 2
