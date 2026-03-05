#!/usr/bin/env python3
"""测试 Skills API P0 功能."""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crew.memory import MemoryStore
from crew.skills import Skill, SkillAction, SkillMetadata, SkillStore, SkillTrigger
from crew.skills_engine import SkillsEngine


def test_skill_crud():
    """测试 Skill CRUD 操作."""
    print("=== 测试 Skill CRUD ===")

    # 创建临时目录
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_store = SkillStore(skills_dir=Path(tmpdir) / "skills")

        # 1. 创建 Skill
        skill = Skill(
            name="query-before-code",
            version="0.1.0",
            employee="赵云帆",
            description="写代码前自动触发，查询历史事故",
            trigger=SkillTrigger(type="keyword", keywords=["写代码", "实现 API", "修改 schema"]),
            actions=[
                SkillAction(
                    type="query_memory",
                    params={"query": "{{task_keywords}} + correction", "limit": 10},
                )
            ],
            metadata=SkillMetadata(category="safety", priority="high"),
        )

        created = skill_store.create_skill(skill)
        print(f"✓ 创建 Skill: {created.name} (ID: {created.skill_id})")

        # 2. 获取 Skill
        retrieved = skill_store.get_skill("赵云帆", "query-before-code")
        assert retrieved is not None
        assert retrieved.name == "query-before-code"
        print(f"✓ 获取 Skill: {retrieved.name}")

        # 3. 列出 Skills
        skills = skill_store.list_skills("赵云帆")
        assert len(skills) == 1
        print(f"✓ 列出 Skills: {len(skills)} 个")

        # 4. 更新 Skill
        updated = skill_store.update_skill(
            "赵云帆", "query-before-code", {"version": "0.2.0", "enabled": False}
        )
        assert updated.version == "0.2.0"
        assert updated.enabled is False
        print(f"✓ 更新 Skill: version={updated.version}, enabled={updated.enabled}")

        # 5. 删除 Skill
        deleted = skill_store.delete_skill("赵云帆", "query-before-code")
        assert deleted is True
        print("✓ 删除 Skill")

        # 验证已删除
        retrieved = skill_store.get_skill("赵云帆", "query-before-code")
        assert retrieved is None
        print("✓ 验证删除成功")


def test_keyword_trigger():
    """测试关键词触发."""
    print("\n=== 测试关键词触发 ===")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_store = SkillStore(skills_dir=Path(tmpdir) / "skills")
        memory_store = MemoryStore(memory_dir=Path(tmpdir) / "memory")
        engine = SkillsEngine(skill_store, memory_store)

        # 创建测试 Skill
        skill = Skill(
            name="query-before-code",
            employee="赵云帆",
            description="写代码前触发",
            trigger=SkillTrigger(type="keyword", keywords=["写代码", "实现 API", "修改 schema"]),
            actions=[SkillAction(type="query_memory", params={"query": "correction", "limit": 5})],
        )
        skill_store.create_skill(skill)

        # 测试触发
        test_cases = [
            ("帮我写代码实现一个新功能", True, "应该触发（包含'写代码'）"),
            ("需要实现 API endpoint", True, "应该触发（包含'实现 API'）"),
            ("修改 schema 添加字段", True, "应该触发（包含'修改 schema'）"),
            ("帮我调试这个 bug", False, "不应该触发（无关键词）"),
            ("查看一下日志", False, "不应该触发（无关键词）"),
        ]

        for task, should_trigger, reason in test_cases:
            triggered = engine.check_triggers("赵云帆", task)
            is_triggered = len(triggered) > 0

            if is_triggered == should_trigger:
                print(f"✓ {reason}: '{task}'")
            else:
                print(f"✗ {reason}: '{task}' (期望={should_trigger}, 实际={is_triggered})")


def test_skill_execution():
    """测试 Skill 执行."""
    print("\n=== 测试 Skill 执行 ===")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_store = SkillStore(skills_dir=Path(tmpdir) / "skills")
        memory_store = MemoryStore(memory_dir=Path(tmpdir) / "memory")
        engine = SkillsEngine(skill_store, memory_store)

        # 添加测试记忆
        memory_store.add(
            employee="赵云帆",
            category="correction",
            content="新增 API handler 必须同步更新 webhook.py 的导入列表",
        )

        # 创建 Skill
        skill = Skill(
            name="query-before-code",
            employee="赵云帆",
            description="写代码前触发",
            trigger=SkillTrigger(type="keyword", keywords=["写代码"]),
            actions=[
                SkillAction(type="query_memory", params={"category": "correction", "limit": 5})
            ],
        )
        skill_store.create_skill(skill)

        # 执行 Skill
        result = engine.execute_skill(skill, "赵云帆", {"task": "写一个新的 API"})

        print("✓ 执行 Skill")
        print(f"  - 执行动作数: {len(result['executed_actions'])}")
        print(f"  - 执行时间: {result['execution_time_ms']}ms")
        print(f"  - 加载记忆数: {len(result['enhanced_context'].get('memories', []))}")

        # 验证记忆被加载
        memories = result["enhanced_context"].get("memories", [])
        assert len(memories) > 0
        print(f"✓ 成功加载 {len(memories)} 条记忆")


def test_trigger_history():
    """测试触发历史记录."""
    print("\n=== 测试触发历史 ===")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_store = SkillStore(skills_dir=Path(tmpdir) / "skills")
        memory_store = MemoryStore(memory_dir=Path(tmpdir) / "memory")
        engine = SkillsEngine(skill_store, memory_store)

        # 创建 Skill
        skill = Skill(
            name="test-skill",
            employee="赵云帆",
            description="测试",
            trigger=SkillTrigger(type="always"),
            actions=[],
        )
        skill_store.create_skill(skill)

        # 执行并记录
        result = engine.execute_skill(skill, "赵云帆", {"task": "测试任务"})
        engine.record_trigger(
            skill=skill,
            employee="赵云帆",
            task="测试任务",
            match_score=1.0,
            execution_result=result,
        )

        print("✓ 记录触发历史")

        # 查询历史
        history = skill_store.get_trigger_history(employee="赵云帆", limit=10)
        assert len(history) > 0
        print(f"✓ 查询到 {len(history)} 条触发记录")

        record = history[0]
        print(f"  - Skill ID: {record.skill_id}")
        print(f"  - 任务: {record.task}")
        print(f"  - 匹配分数: {record.match_score}")
        print(f"  - 执行状态: {record.execution_status}")


def test_stats():
    """测试统计功能."""
    print("\n=== 测试统计功能 ===")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        skill_store = SkillStore(skills_dir=Path(tmpdir) / "skills")

        # 创建多个 Skills
        employees = ["赵云帆", "卫子昂", "林锐"]
        for emp in employees:
            skill = Skill(
                name=f"test-skill-{emp}",
                employee=emp,
                description="测试",
                trigger=SkillTrigger(type="always"),
                actions=[],
                metadata=SkillMetadata(category="safety", priority="high"),
            )
            skill_store.create_skill(skill)

        # 获取统计
        stats = skill_store.get_stats()

        print("✓ 获取统计信息")
        print(f"  - 总 Skills 数: {stats['total_skills']}")
        print(f"  - 按员工: {stats['by_employee']}")
        print(f"  - 按分类: {stats['by_category']}")
        print(f"  - 按优先级: {stats['by_priority']}")
        print(f"  - 启用数量: {stats['enabled_count']}")

        assert stats["total_skills"] == 3
        assert len(stats["by_employee"]) == 3


if __name__ == "__main__":
    try:
        test_skill_crud()
        test_keyword_trigger()
        test_skill_execution()
        test_trigger_history()
        test_stats()

        print("\n" + "=" * 50)
        print("✓ 所有测试通过！")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
