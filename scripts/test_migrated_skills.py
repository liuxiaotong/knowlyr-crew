#!/usr/bin/env python3
"""测试迁移后的 Skills 是否正常工作."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crew.memory import MemoryStore
from crew.skills import SkillStore
from crew.skills_engine import SkillsEngine


def test_migrated_skills():
    """测试迁移后的 Skills."""
    print("=== 测试迁移后的 Skills ===\n")

    project_dir = Path(__file__).parent.parent
    skill_store = SkillStore(project_dir=project_dir)
    memory_store = MemoryStore(project_dir=project_dir)
    engine = SkillsEngine(skill_store, memory_store)

    # 测试用例
    test_cases = [
        {
            "employee": "赵云帆",
            "task": "帮我写一个新的 API endpoint，处理用户上传头像",
            "expected_skills": ["query-before-code"],
        },
        {
            "employee": "卫子昂",
            "task": "重构 Wiki 页面的三栏布局，改成单页应用",
            "expected_skills": ["query-before-refactor"],
        },
        {
            "employee": "林锐",
            "task": "帮我审查这个 PR #123，主要改了 webhook_handlers.py",
            "expected_skills": ["query-review-history"],
        },
        {
            "employee": "程薇",
            "task": "帮我写一个测试，验证用户登录后能正确获取 token",
            "expected_skills": ["query-test-patterns"],
        },
        {
            "employee": "姜墨言",
            "task": "我们需要决定一下：antgather 的文件上传功能，应该用 OSS 直传还是通过后端中转？",
            "expected_skills": ["query-before-decision"],
        },
        {
            "employee": "姜墨言",
            "task": "派赵云帆来处理这个后端问题",
            "expected_skills": ["query-before-dispatch"],
        },
        {
            "employee": "姜墨言",
            "task": "这个方案也不行，我已经提了三个方案了",
            "expected_skills": ["query-on-failure"],
        },
    ]

    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        employee = test_case["employee"]
        task = test_case["task"]
        expected = test_case["expected_skills"]

        print(f"测试 {i}: {employee}")
        print(f"  任务: {task}")

        # 检查触发
        triggered = engine.check_triggers(employee, task)
        triggered_names = [skill.name for skill, score in triggered]

        print(f"  期望触发: {expected}")
        print(f"  实际触发: {triggered_names}")

        # 验证
        if set(expected).issubset(set(triggered_names)):
            print("  ✓ 通过\n")
            passed += 1
        else:
            print("  ✗ 失败\n")
            failed += 1

    # 总结
    print("=" * 50)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 50)

    return failed == 0


def test_moyan_start():
    """测试 moyan-start (always trigger)."""
    print("\n=== 测试 moyan-start (always trigger) ===\n")

    project_dir = Path(__file__).parent.parent
    skill_store = SkillStore(project_dir=project_dir)
    memory_store = MemoryStore(project_dir=project_dir)
    engine = SkillsEngine(skill_store, memory_store)

    # moyan-start 应该总是触发
    triggered = engine.check_triggers("姜墨言", "任意任务")
    triggered_names = [skill.name for skill, score in triggered]

    print("任务: 任意任务")
    print(f"触发的 skills: {triggered_names}")

    if "moyan-start" in triggered_names:
        print("✓ moyan-start 正确触发（always trigger）\n")
        return True
    else:
        print("✗ moyan-start 未触发\n")
        return False


def test_skill_execution():
    """测试 Skill 执行."""
    print("=== 测试 Skill 执行 ===\n")

    project_dir = Path(__file__).parent.parent
    skill_store = SkillStore(project_dir=project_dir)
    memory_store = MemoryStore(project_dir=project_dir)
    engine = SkillsEngine(skill_store, memory_store)

    # 获取赵云帆的 query-before-code skill
    skill = skill_store.get_skill("赵云帆", "query-before-code")
    if not skill:
        print("✗ 未找到 query-before-code skill\n")
        return False

    print(f"执行 skill: {skill.name}")

    # 执行
    result = engine.execute_skill(skill, "赵云帆", {"task": "写一个新的 API"})

    print(f"  - 执行动作数: {len(result['executed_actions'])}")
    print(f"  - 执行时间: {result['execution_time_ms']}ms")
    print(f"  - 执行状态: {result['executed_actions'][0]['status']}")

    if result["executed_actions"][0]["status"] == "success":
        print("✓ Skill 执行成功\n")
        return True
    else:
        print("✗ Skill 执行失败\n")
        return False


if __name__ == "__main__":
    try:
        result1 = test_migrated_skills()
        result2 = test_moyan_start()
        result3 = test_skill_execution()

        if result1 and result2 and result3:
            print("\n" + "=" * 50)
            print("✓ 所有测试通过！迁移成功！")
            print("=" * 50)
            sys.exit(0)
        else:
            print("\n" + "=" * 50)
            print("✗ 部分测试失败")
            print("=" * 50)
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
