#!/usr/bin/env python3
"""测试 Skills 集成到员工执行流程."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crew.memory import MemoryStore
from crew.skills import SkillStore
from crew.skills_engine import SkillsEngine


def test_integration():
    """测试集成效果."""
    print("=== 测试 Skills 集成 ===\n")

    project_dir = Path(__file__).parent.parent
    skill_store = SkillStore(project_dir=project_dir)
    memory_store = MemoryStore(project_dir=project_dir)
    engine = SkillsEngine(skill_store, memory_store)

    # 添加测试记忆
    print("1. 添加测试记忆")
    memory_store.add(
        employee="赵云帆",
        category="correction",
        content="新增 API handler 必须同步更新 webhook.py 的导入列表，否则会 404",
    )
    memory_store.add(
        employee="赵云帆",
        category="correction",
        content="Schema 改动必须先本地 uvicorn 验证，再推生产，避免服务崩溃",
    )
    print("  ✓ 已添加 2 条 correction 记忆\n")

    # 模拟任务
    test_cases = [
        {
            "employee": "赵云帆",
            "task": "帮我写一个新的 API endpoint，处理用户上传头像",
            "expected_memories": True,
        },
        {
            "employee": "卫子昂",
            "task": "重构 Wiki 页面的三栏布局",
            "expected_memories": False,  # 卫子昂没有 correction 记忆
        },
        {
            "employee": "赵云帆",
            "task": "帮我调试这个 bug",
            "expected_memories": False,  # 不触发 query-before-code
        },
    ]

    print("2. 测试任务触发\n")
    for i, test_case in enumerate(test_cases, 1):
        employee = test_case["employee"]
        task = test_case["task"]
        expected = test_case["expected_memories"]

        print(f"测试 {i}: {employee}")
        print(f"  任务: {task}")

        # 检查触发
        triggered = engine.check_triggers(employee, task, {})
        print(f"  触发的 skills: {[s.name for s, _ in triggered]}")

        if triggered:
            # 执行第一个 skill
            skill, score = triggered[0]
            result = engine.execute_skill(skill, employee, {"task": task})

            memories = result.get("enhanced_context", {}).get("memories", [])
            print(f"  加载的记忆数: {len(memories)}")

            if memories:
                print("  记忆内容:")
                for m in memories[:2]:
                    print(f"    - [{m['category']}] {m['content'][:60]}...")

            has_memories = len(memories) > 0
            if has_memories == expected:
                print("  ✓ 通过\n")
            else:
                print(f"  ✗ 失败（期望记忆={expected}, 实际={has_memories}）\n")
        else:
            if not expected:
                print("  ✓ 通过（未触发）\n")
            else:
                print("  ✗ 失败（期望触发但未触发）\n")

    print("=" * 50)
    print("测试完成")
    print("=" * 50)


if __name__ == "__main__":
    try:
        test_integration()
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
