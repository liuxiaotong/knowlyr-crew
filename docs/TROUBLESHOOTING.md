# 故障排查 / Troubleshooting

## 常见错误

### API Key 未设置

```
ProviderError: anthropic API key 未设置。请设置环境变量 ANTHROPIC_API_KEY 或通过参数传递 api_key。
```

**解决**: 设置对应提供商的环境变量：

| 提供商 | 环境变量 |
|--------|----------|
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Moonshot | `MOONSHOT_API_KEY` |
| Gemini | `GOOGLE_API_KEY` |
| 智谱 | `ZHIPUAI_API_KEY` |
| 通义千问 | `DASHSCOPE_API_KEY` |

### SDK 未安装

```
ImportError: anthropic SDK 未安装。请运行: pip install knowlyr-crew[execute]
```

**解决**: 安装对应的 extra：
```bash
pip install knowlyr-crew[execute]   # Anthropic
pip install knowlyr-crew[openai]    # OpenAI / DeepSeek / Moonshot / 智谱 / 通义
pip install knowlyr-crew[gemini]    # Gemini
pip install knowlyr-crew[webhook]   # Webhook 服务器
pip install knowlyr-crew[mcp]       # MCP 服务器
```

### 员工未找到

```
未找到员工: code-reviwer
类似的名称: code-reviewer
```

**解决**: 检查拼写，CLI 会自动提示相似名称。使用 `knowlyr-crew list` 查看所有可用员工。

### Pipeline 断点恢复失败

```
未找到 pipeline: full-review
```

**解决**: 确保 `.crew/pipelines/` 目录下有对应的 YAML 文件。使用 `knowlyr-crew pipeline list` 查看可用流水线。

### 记忆索引损坏

**症状**: 搜索结果异常或报 SQLite 错误。

**解决**:
```bash
knowlyr-crew memory index --repair
```
这会删除旧的 `embeddings.db` 并重建向量索引。

## 调试

### 详细日志

使用 `-v` 启用 DEBUG 级别日志：
```bash
knowlyr-crew -v run code-reviewer main
```

### 检测项目上下文

查看自动检测的项目信息：
```bash
knowlyr-crew run code-reviewer main --debug-context
```

### 检查服务健康状态

```bash
# Webhook 服务器
curl http://localhost:8765/health

# 运行时指标
curl http://localhost:8765/metrics
```

## 速率限制

当看到 HTTP 429 错误时：

1. **LLM API 限流**: 执行器会自动重试（最多 3 次，指数退避 + 随机抖动）
2. **Webhook 速率限制**: 默认 60 请求/分钟/IP，`/health` 和 `/metrics` 不受限制
3. **knowlyr-id 断路器**: 连续 3 次失败后暂停请求 30 秒，自动恢复

## 获取帮助

- GitHub Issues: https://github.com/liuxiaotong/knowlyr-crew/issues
- `knowlyr-crew --help`: 查看所有可用命令
