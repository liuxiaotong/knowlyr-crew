"""跨员工记忆共享测试."""

import json
from datetime import datetime, timedelta

from crew.memory import MemoryConfig, MemoryStore


class TestQueryShared:
    """测试共享记忆查询."""

    def test_shared_visible_to_others(self, tmp_path):
        """shared=True 的记忆对其他员工可见."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        store.add("pm", "decision", "使用 REST API", shared=True)
        store.add("pm", "finding", "私有记忆")  # shared=False

        shared = store.query_shared(exclude_employee="code-reviewer")
        assert len(shared) == 1
        assert shared[0].content == "使用 REST API"

    def test_non_shared_invisible(self, tmp_path):
        """shared=False 的记忆不出现在共享池."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        store.add("pm", "decision", "内部决策")

        shared = store.query_shared()
        assert len(shared) == 0

    def test_exclude_self(self, tmp_path):
        """排除自身员工的共享记忆."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        store.add("pm", "decision", "PM 决策", shared=True)
        store.add("code-reviewer", "finding", "CR 发现", shared=True)

        shared = store.query_shared(exclude_employee="pm")
        assert len(shared) == 1
        assert shared[0].employee == "code-reviewer"

    def test_filter_by_tags(self, tmp_path):
        """按标签过滤共享记忆."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        store.add("pm", "decision", "API 架构决策",
                   tags=["architecture", "api"], shared=True)
        store.add("pm", "decision", "安全策略",
                   tags=["security"], shared=True)

        # 按 architecture 标签过滤
        shared = store.query_shared(tags=["architecture"])
        assert len(shared) == 1
        assert shared[0].content == "API 架构决策"

    def test_tag_any_match(self, tmp_path):
        """标签取交集（任一匹配）."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        store.add("pm", "decision", "API 决策",
                   tags=["api"], shared=True)
        store.add("pm", "decision", "安全决策",
                   tags=["security"], shared=True)
        store.add("pm", "decision", "UI 决策",
                   tags=["frontend"], shared=True)

        shared = store.query_shared(tags=["api", "security"])
        assert len(shared) == 2
        contents = {e.content for e in shared}
        assert "API 决策" in contents
        assert "安全决策" in contents

    def test_shared_respects_ttl(self, tmp_path):
        """共享记忆也受 TTL 限制."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        entry = store.add("pm", "decision", "过期决策", ttl_days=1, shared=True)

        # 手动修改创建时间为 2 天前
        path = store._employee_file("pm")
        lines = path.read_text(encoding="utf-8").splitlines()
        new_lines = []
        old_time = (datetime.now() - timedelta(days=2)).isoformat()
        for line in lines:
            if entry.id in line:
                data = json.loads(line)
                data["created_at"] = old_time
                new_lines.append(json.dumps(data, ensure_ascii=False))
            else:
                new_lines.append(line)
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        shared = store.query_shared()
        assert len(shared) == 0

    def test_shared_respects_confidence_decay(self, tmp_path):
        """共享记忆受衰减影响，低于阈值时不返回."""
        config = MemoryConfig(confidence_half_life_days=30.0)
        store = MemoryStore(memory_dir=tmp_path / "memory", config=config)
        entry = store.add("pm", "decision", "旧决策", shared=True)

        # 60 天前 → confidence ≈ 0.25
        path = store._employee_file("pm")
        lines = path.read_text(encoding="utf-8").splitlines()
        new_lines = []
        old_time = (datetime.now() - timedelta(days=60)).isoformat()
        for line in lines:
            if entry.id in line:
                data = json.loads(line)
                data["created_at"] = old_time
                new_lines.append(json.dumps(data, ensure_ascii=False))
            else:
                new_lines.append(line)
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        # min_confidence=0.3 应排除
        shared = store.query_shared(min_confidence=0.3)
        assert len(shared) == 0

    def test_shared_skips_superseded(self, tmp_path):
        """被覆盖的共享记忆不出现."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        e = store.add("pm", "decision", "旧决策", shared=True)
        store.correct("pm", e.id, "新决策")

        shared = store.query_shared()
        assert len(shared) == 0  # 旧条目被 superseded，新条目 shared=False

    def test_query_shared_limit(self, tmp_path):
        """共享查询支持 limit."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        for i in range(5):
            store.add("pm", "finding", f"共享 {i}", shared=True)

        shared = store.query_shared(limit=2)
        assert len(shared) == 2

    def test_query_shared_empty_dir(self, tmp_path):
        """空目录返回空列表."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        shared = store.query_shared()
        assert shared == []


class TestFormatForPromptWithSharing:
    """测试 format_for_prompt 包含共享记忆."""

    def test_includes_shared_section(self, tmp_path):
        """有共享记忆时，prompt 包含团队共享经验段落."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        store.add("pm", "decision", "使用微服务架构",
                   tags=["architecture"], shared=True)
        store.add("code-reviewer", "finding", "CR 自己的经验")

        text = store.format_for_prompt("code-reviewer", employee_tags=["architecture"])
        assert "CR 自己的经验" in text
        assert "团队共享经验" in text
        assert "使用微服务架构" in text

    def test_no_shared_section_when_empty(self, tmp_path):
        """无共享记忆时不出现团队共享经验段落."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        store.add("code-reviewer", "finding", "自己的经验")

        text = store.format_for_prompt("code-reviewer")
        assert "团队共享经验" not in text
        assert "自己的经验" in text

    def test_shared_excludes_self(self, tmp_path):
        """共享段落不包含自己的记忆."""
        store = MemoryStore(memory_dir=tmp_path / "memory")
        store.add("pm", "decision", "PM 决策", shared=True)
        store.add("pm", "finding", "PM 自己的经验")

        text = store.format_for_prompt("pm")
        # 应该有自己的经验
        assert "PM 自己的经验" in text or "PM 决策" in text
        # 共享段落不应该有（因为排除了自身）
        if "团队共享经验" in text:
            # 如果有共享段落，里面不应该有 pm 的内容
            shared_section = text.split("团队共享经验")[1]
            assert "(pm)" not in shared_section
