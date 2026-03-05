# 记忆去重和合并机制

## 概述

记忆去重系统在写入新记忆时自动检测相似的已有记忆，避免重复记录相同或高度相似的经验。

## 工作原理

### 1. 相似度检测

当调用 `POST /api/memory/add` 添加新记忆时，系统会：

1. **质量检查**：首先检查记忆质量（分数 >= 0.6）
2. **相似度检测**：查找该员工最近 100 条同类记忆，计算相似度
3. **返回警告**：如果发现相似度 >= 阈值的记忆，返回警告而非直接写入

### 2. 相似度计算方法

**主要方式：OpenAI Embedding（推荐）**
- 使用 `text-embedding-3-small` 模型计算语义向量
- 计算余弦相似度
- 阈值：0.85（高度相似）
- 优点：语义理解准确，可识别同义表达

**降级方式：关键词匹配**
- 当 OpenAI API 不可用时自动降级
- 提取高频关键词，计算 Jaccard 相似度
- 阈值：0.7（或用户指定阈值 - 0.15）
- 优点：无需外部 API，成本为零

### 3. 缓存机制

- Embedding 向量缓存到 `.crew/memory/.embeddings/{employee}.json`
- 避免重复计算，提升性能
- 写入时自动更新缓存

## API 使用

### 添加记忆（带去重检测）

```bash
curl -X POST https://crew.knowlyr.com/api/memory/add \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "employee": "赵云帆",
    "category": "finding",
    "content": "修复了数据库连接池的内存泄漏问题，原因是连接未正确关闭"
  }'
```

**成功响应（无相似记忆）：**
```json
{
  "ok": true,
  "skipped": false,
  "entry_id": "abc123",
  "employee": "赵云帆",
  "category": "finding",
  "suggested_tags": ["database", "memory-leak"]
}
```

**警告响应（发现相似记忆）：**
```json
{
  "ok": false,
  "warning": "similar_memories_found",
  "similar_memories": [
    {
      "id": "xyz789",
      "content": "解决了数据库连接池内存泄漏的bug，问题是连接没有正确关闭",
      "similarity": 0.87,
      "created_at": "2026-03-01T10:30:00",
      "category": "finding"
    }
  ],
  "suggestions": [
    "如果是相同内容，考虑更新已有记忆而非新增",
    "如果是补充信息，可以在原记忆基础上扩展",
    "如果确实是新的独立经验，添加 ?force=true 参数重新提交"
  ]
}
```

### 强制写入（绕过去重检测）

如果确认是新的独立经验，可以使用 `force=true` 参数：

```bash
curl -X POST "https://crew.knowlyr.com/api/memory/add?force=true" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "employee": "赵云帆",
    "category": "finding",
    "content": "修复了数据库连接池的内存泄漏问题，原因是连接未正确关闭"
  }'
```

### 更新已有记忆

如果发现相似记忆，可以选择更新而非新增：

```bash
curl -X PUT https://crew.knowlyr.com/api/memory/update \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "entry_id": "xyz789",
    "employee": "赵云帆",
    "content": "修复了数据库连接池的内存泄漏问题，根因是连接未正确关闭。解决方案：在 finally 块中显式调用 close()，并添加连接池监控",
    "tags": ["database", "memory-leak", "monitoring"],
    "updated_by": "姜墨言"
  }'
```

**响应：**
```json
{
  "ok": true,
  "entry_id": "xyz789",
  "updated": true
}
```

更新后的记忆会自动添加以下标签：
- `updated-by:姜墨言`
- `updated-at:2026-03-02`

## 配置

### OpenAI API Key

在 `.env` 文件中配置：

```bash
OPENAI_API_KEY=sk-your-api-key
```

如果未配置，系统会自动降级到关键词匹配。

### 相似度阈值

默认阈值：
- Embedding 方式：0.85
- 关键词匹配：0.7

可以在调用 `find_similar_memories` 时自定义阈值。

## 性能和成本

### Embedding 方式
- **成本**：$0.00002 / 1K tokens
- **延迟**：100-300ms（首次计算）
- **缓存后**：< 10ms（直接读取缓存）

### 关键词匹配
- **成本**：零
- **延迟**：< 5ms
- **准确度**：中等（无法识别同义表达）

## 最佳实践

1. **优先使用 Embedding**：语义理解更准确，成本可控
2. **定期清理缓存**：如果记忆内容更新频繁，定期清理 `.embeddings/` 目录
3. **合理设置阈值**：
   - 0.9+：几乎完全相同
   - 0.85-0.9：高度相似（推荐）
   - 0.7-0.85：中等相似
   - < 0.7：不太相似
4. **更新而非新增**：发现相似记忆时，优先考虑更新已有记忆
5. **记录更新历史**：使用 `updated_by` 参数记录谁更新了记忆

## 限制

1. **只检查同类记忆**：不同 category 的记忆不会互相比较
2. **只检查最近 100 条**：避免全量扫描影响性能
3. **不支持跨员工去重**：每个员工的记忆独立检测
4. **需要质量合格**：记忆质量分数 < 0.6 会被拒绝，不会进入去重检测

## 故障排查

### 问题：相似记忆未被检测到

**可能原因：**
1. 相似度低于阈值（0.85 或 0.7）
2. OpenAI API 失败且关键词匹配相似度不足
3. 记忆类别不同

**解决方案：**
- 检查日志中的相似度分数
- 降低阈值（如果合理）
- 确保记忆类别一致

### 问题：Embedding 计算失败

**可能原因：**
1. OpenAI API Key 未配置或无效
2. 网络连接问题
3. API 配额用尽

**解决方案：**
- 检查 `.env` 中的 `OPENAI_API_KEY`
- 查看日志中的错误信息
- 系统会自动降级到关键词匹配

## 技术细节

### 文件结构

```
.crew/memory/
├── 赵云帆.jsonl              # 记忆数据
├── .embeddings/
│   └── 赵云帆.json           # Embedding 缓存
└── config.json               # 配置文件
```

### 代码模块

- `src/crew/memory_similarity.py`：相似度检测核心逻辑
- `src/crew/webhook_handlers.py`：API 端点实现
- `tests/test_memory_similarity.py`：单元测试
- `tests/test_memory_deduplication.py`：集成测试

## 更新日志

**2026-03-02**
- ✅ 实现基于 Embedding 的相似度检测
- ✅ 实现关键词匹配降级方案
- ✅ 新增更新记忆 API
- ✅ 添加 Embedding 缓存机制
- ✅ 完整的测试覆盖
