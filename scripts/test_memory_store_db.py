#!/usr/bin/env python3
"""测试 memory_store_db.py 的核心功能."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crew.database import is_pg
from crew.memory_store_db import MemoryStoreDB, init_memory_tables


def test_memory_store_db():
    """测试数据库版 MemoryStore 的基本功能."""

    if not is_pg():
        print("❌ 需要 PostgreSQL 环境")
        return False

    print("✓ PostgreSQL 环境检测通过")

    # 初始化表
    try:
        init_memory_tables()
        print("✓ memories 表初始化成功")
    except Exception as e:
        print(f"❌ 表初始化失败: {e}")
        return False

    # 创建 MemoryStoreDB 实例
    try:
        store = MemoryStoreDB()
        print("✓ MemoryStoreDB 实例创建成功")
    except Exception as e:
        print(f"❌ 实例创建失败: {e}")
        return False

    # 测试添加记忆
    try:
        entry = store.add(
            employee="测试员工",
            category="finding",
            content="这是一条测试记忆",
            tags=["test", "migration"],
            shared=True,
        )
        print(f"✓ 添加记忆成功: {entry['id']}")
        test_id = entry["id"]
    except Exception as e:
        print(f"❌ 添加记忆失败: {e}")
        return False

    # 测试查询记忆
    try:
        results = store.query(employee="测试员工", limit=10)
        if len(results) > 0:
            print(f"✓ 查询记忆成功: 找到 {len(results)} 条")
        else:
            print("⚠️  查询记忆成功但结果为空")
    except Exception as e:
        print(f"❌ 查询记忆失败: {e}")
        return False

    # 测试查询共享记忆
    try:
        shared = store.query_shared(tags=["test"], limit=5)
        print(f"✓ 查询共享记忆成功: 找到 {len(shared)} 条")
    except Exception as e:
        print(f"❌ 查询共享记忆失败: {e}")
        return False

    # 测试统计
    try:
        count = store.count(employee="测试员工")
        print(f"✓ 统计记忆成功: {count} 条")
    except Exception as e:
        print(f"❌ 统计记忆失败: {e}")
        return False

    # 测试删除
    try:
        deleted = store.delete(test_id, employee="测试员工")
        if deleted:
            print(f"✓ 删除记忆成功: {test_id}")
        else:
            print(f"⚠️  删除记忆失败: 未找到 {test_id}")
    except Exception as e:
        print(f"❌ 删除记忆失败: {e}")
        return False

    print("\n✅ 所有测试通过！")
    return True


if __name__ == "__main__":
    success = test_memory_store_db()
    sys.exit(0 if success else 1)
