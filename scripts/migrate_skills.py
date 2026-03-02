#!/usr/bin/env python3
"""迁移本地 Skills 到服务端."""

import json
import re
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crew.skills import Skill, SkillAction, SkillMetadata, SkillStore, SkillTrigger


def parse_skill_md(skill_path: Path) -> dict | None:
    """解析 SKILL.md 文件."""
    if not skill_path.exists():
        return None

    content = skill_path.read_text(encoding="utf-8")

    # 提取 YAML frontmatter
    yaml_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if not yaml_match:
        print(f"  ⚠️  未找到 YAML frontmatter: {skill_path}")
        return None

    yaml_content = yaml_match.group(1)

    # 简单解析 YAML（只提取需要的字段）
    skill_data = {}
    for line in yaml_content.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            skill_data[key.strip()] = value.strip()

    return skill_data


def extract_keywords_from_description(description: str) -> list[str]:
    """从 description 中提取关键词."""
    # 提取引号中的内容
    keywords = re.findall(r'"([^"]+)"', description)

    # 添加中文关键词映射
    chinese_keywords = {
        "write code": "写代码",
        "implement": "实现",
        "add a new API endpoint": "新的 API",
        "API endpoint": "API",
        "modify the schema": "修改 schema",
        "create a migration": "创建 migration",
        "refactor": "重构",
        "restructure": "重构",
        "review": "审查",
        "check this code": "审查代码",
        "write tests": "写测试",
        "write a test": "写一个测试",
        "add test": "写测试",
        "decide": "决定",
        "make a decision": "做决策",
        "dispatch": "派",
        "delegate": "派",
        "ask": "派",
        "failed": "失败",
        "not working": "不行",
        "didn't work": "不行",
        "problem": "问题",
        "bug": "bug",
        "error": "错误",
    }

    # 添加中文翻译
    all_keywords = keywords.copy()
    for eng_kw in keywords:
        eng_lower = eng_kw.lower()
        for eng, chn in chinese_keywords.items():
            if eng in eng_lower and chn not in all_keywords:
                all_keywords.append(chn)

    return all_keywords[:20]  # 最多 20 个关键词


def map_skill_to_employee(skill_name: str) -> str | None:
    """根据 skill 名称映射到员工."""
    mapping = {
        "moyan-start": "姜墨言",
        "query-on-problem": "姜墨言",
        "query-before-solution": "姜墨言",
        "query-before-decision": "姜墨言",
        "query-before-dispatch": "姜墨言",
        "query-on-failure": "姜墨言",
        "query-before-code": "赵云帆",
        "query-before-refactor": "卫子昂",
        "query-review-history": "林锐",
        "query-test-patterns": "程薇",
    }
    return mapping.get(skill_name)


def determine_trigger_type(skill_name: str) -> str:
    """确定触发类型."""
    if skill_name == "moyan-start":
        return "always"
    else:
        return "keyword"


def determine_category(skill_name: str) -> str:
    """确定分类."""
    if "failure" in skill_name or "problem" in skill_name:
        return "safety"
    elif "review" in skill_name or "test" in skill_name:
        return "quality"
    else:
        return "efficiency"


def determine_priority(skill_name: str) -> str:
    """确定优先级."""
    if skill_name in ["query-on-failure", "moyan-start"]:
        return "critical"
    elif skill_name in ["query-before-code", "query-review-history"]:
        return "high"
    else:
        return "medium"


def create_actions(skill_name: str) -> list[SkillAction]:
    """创建 Actions."""
    # 大部分 skills 都是查询记忆
    if "review" in skill_name:
        category = "finding"
    elif "failure" in skill_name or "problem" in skill_name:
        category = "correction"
    elif "decision" in skill_name:
        category = "decision"
    elif "test" in skill_name:
        category = "pattern"
    else:
        category = None  # 不指定 category，查询所有

    return [
        SkillAction(
            type="query_memory",
            params={"category": category, "limit": 10} if category else {"limit": 10},
        )
    ]


def migrate_skills():
    """迁移所有 Skills."""
    print("=== 迁移本地 Skills 到服务端 ===\n")

    # 本地 skills 目录
    local_skills_dir = Path.home() / ".claude" / "skills"
    if not local_skills_dir.exists():
        print(f"❌ 本地 skills 目录不存在: {local_skills_dir}")
        return

    # 服务端 skills 存储
    project_dir = Path(__file__).parent.parent
    skill_store = SkillStore(project_dir=project_dir)

    # 遍历所有 skill 目录
    migrated = 0
    skipped = 0
    failed = 0

    for skill_dir in sorted(local_skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue

        skill_name = skill_dir.name
        skill_md = skill_dir / "SKILL.md"

        print(f"处理: {skill_name}")

        # 解析 SKILL.md
        skill_data = parse_skill_md(skill_md)
        if not skill_data:
            print(f"  ⚠️  跳过（无法解析）\n")
            skipped += 1
            continue

        # 映射到员工
        employee = map_skill_to_employee(skill_name)
        if not employee:
            print(f"  ⚠️  跳过（无法映射到员工）\n")
            skipped += 1
            continue

        # 检查是否已存在
        existing = skill_store.get_skill(employee, skill_name)
        if existing:
            print(f"  ℹ️  已存在，跳过\n")
            skipped += 1
            continue

        try:
            # 提取关键词
            description = skill_data.get("description", "")
            keywords = extract_keywords_from_description(description)

            # 创建 Skill
            skill = Skill(
                name=skill_name,
                version=skill_data.get("version", "0.1.0"),
                employee=employee,
                description=description,
                trigger=SkillTrigger(
                    type=determine_trigger_type(skill_name),
                    keywords=keywords,
                ),
                actions=create_actions(skill_name),
                metadata=SkillMetadata(
                    category=determine_category(skill_name),
                    priority=determine_priority(skill_name),
                ),
            )

            # 保存
            skill_store.create_skill(skill)

            print(f"  ✓ 迁移成功")
            print(f"    - 员工: {employee}")
            print(f"    - 触发类型: {skill.trigger.type}")
            print(f"    - 关键词数: {len(keywords)}")
            print(f"    - 优先级: {skill.metadata.priority}\n")

            migrated += 1

        except Exception as e:
            print(f"  ✗ 迁移失败: {e}\n")
            failed += 1

    # 总结
    print("=" * 50)
    print(f"迁移完成:")
    print(f"  - 成功: {migrated}")
    print(f"  - 跳过: {skipped}")
    print(f"  - 失败: {failed}")
    print("=" * 50)

    # 显示统计
    if migrated > 0:
        print("\n服务端 Skills 统计:")
        stats = skill_store.get_stats()
        print(f"  - 总数: {stats['total_skills']}")
        print(f"  - 按员工: {stats['by_employee']}")
        print(f"  - 按分类: {stats['by_category']}")
        print(f"  - 按优先级: {stats['by_priority']}")


if __name__ == "__main__":
    try:
        migrate_skills()
    except Exception as e:
        print(f"\n❌ 迁移失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
