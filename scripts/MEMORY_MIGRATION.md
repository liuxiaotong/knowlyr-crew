# Memory 数据库迁移指南

## 概述

将 Memory 系统从本地 JSONL 文件迁移到 PostgreSQL 数据库，与 Soul 保持统一的存储方式。

## 迁移内容

- **源**: `.crew/memory/*.jsonl` 文件
- **目标**: PostgreSQL `memories` 表
- **数据量**: 约 369 条记忆（9 个员工）

## 实施步骤

### 1. 备份现有数据

```bash
# 备份记忆文件
tar -czf memory_backup_$(date +%Y%m%d_%H%M%S).tar.gz .crew/memory/
```

### 2. 试运行迁移

```bash
# 验证数据可以正确读取
python3 scripts/migrate_memory_to_db.py --dry-run
```

### 3. 执行迁移

```bash
# 正式迁移到数据库
python3 scripts/migrate_memory_to_db.py
```

### 4. 验证结果

```bash
# 对比文件和数据库中的记录数
python3 scripts/migrate_memory_to_db.py --verify
```

### 5. 测试数据库版本

```bash
# 测试 MemoryStoreDB 的核心功能
python3 scripts/test_memory_store_db.py
```

## 数据库 Schema

```sql
CREATE TABLE memories (
    id VARCHAR(12) PRIMARY KEY,
    employee VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    category VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    source_session VARCHAR(255) DEFAULT '',
    confidence FLOAT DEFAULT 1.0,
    superseded_by VARCHAR(12) DEFAULT '',
    ttl_days INTEGER DEFAULT 0,
    importance INTEGER DEFAULT 3,
    last_accessed TIMESTAMP,
    tags TEXT[],
    shared BOOLEAN DEFAULT FALSE,
    visibility VARCHAR(20) DEFAULT 'open',
    trigger_condition TEXT DEFAULT '',
    applicability TEXT[],
    origin_employee VARCHAR(255) DEFAULT '',
    verified_count INTEGER DEFAULT 0
);

-- 索引
CREATE INDEX idx_memories_employee ON memories(employee);
CREATE INDEX idx_memories_category ON memories(category);
CREATE INDEX idx_memories_created_at ON memories(created_at);
CREATE INDEX idx_memories_shared ON memories(shared);
CREATE INDEX idx_memories_tags ON memories USING GIN(tags);
```

## 新增文件

1. **src/crew/memory_store_db.py** - 数据库版 MemoryStore 实现
2. **scripts/migrate_memory_to_db.py** - 迁移脚本
3. **scripts/test_memory_store_db.py** - 测试脚本

## 修改文件

1. **src/crew/config_store.py** - 添加 memories 表创建逻辑

## 兼容性

- 文件版本（memory.py）保持不变，作为 fallback
- 数据库版本（memory_store_db.py）提供相同接口
- 未来可在 memory.py 中添加自动切换逻辑

## 回滚方案

如果迁移后发现问题：

1. 停止使用数据库版本
2. 恢复备份的 JSONL 文件
3. 继续使用文件版本

## 注意事项

- 迁移脚本使用 `ON CONFLICT DO NOTHING`，重复运行不会覆盖数据
- PostgreSQL 数组字段（tags, applicability）使用 `TEXT[]` 类型
- 时间字段使用 `TIMESTAMP`，Python 端使用 UTC 时区
- 迁移过程不会删除原始 JSONL 文件

## 性能优化

- 使用 GIN 索引加速标签查询
- 批量插入提高迁移速度
- 过期记忆通过 SQL 过滤，避免加载到内存

## 下一步

1. 在生产环境执行迁移
2. 监控数据库查询性能
3. 根据需要调整索引策略
4. 考虑在 memory.py 中添加数据库支持的兼容层
