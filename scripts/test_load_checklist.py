#!/usr/bin/env python3
"""测试 load_checklist Action."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crew.memory import MemoryStore
from crew.skills import Skill, SkillAction, SkillStore, SkillTrigger
from crew.skills_engine import SkillsEngine


def test_load_checklist():
    """测试从 soul 中加载检查清单."""
    print("=== 测试 load_checklist Action ===\n")

    project_dir = Path(__file__).parent.parent
    skill_store = SkillStore(project_dir=project_dir)
    memory_store = MemoryStore(project_dir=project_dir)
    engine = SkillsEngine(skill_store, memory_store)

    # 创建一个使用 load_checklist 的 skill
    skill = Skill(
        name="test-checklist",
        employee="赵云帆",
        description="测试检查清单加载",
        trigger=SkillTrigger(type="always"),
        actions=[
            SkillAction(
                type="load_checklist",
                params={"section": "工作检查清单"},
            )
        ],
    )

    print("1. 执行 load_checklist")
    result = engine.execute_skill(skill, "赵云帆", {"task": "写一个新的 API"})

    checklist_items = result.get("enhanced_context", {}).get("items", [])
    print(f"  加载的检查项数: {len(checklist_items)}\n")

    if checklist_items:
        print("2. 检查清单内容:")
        for i, item in enumerate(checklist_items[:10], 1):
            print(f"  {i}. {item}")

        if len(checklist_items) > 10:
            print(f"  ... 还有 {len(checklist_items) - 10} 项")

        print("\n✓ load_checklist 功能正常工作")
    else:
        print("✗ 未能加载检查清单")
        return False

    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)
    return True


if __name__ == "__main__":
    try:
        success = test_load_checklist()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
