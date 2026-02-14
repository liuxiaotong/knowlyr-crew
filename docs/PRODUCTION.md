# 生产部署清单 / Production Checklist

## 环境变量

### 必需

| 变量 | 用途 |
|------|------|
| `ANTHROPIC_API_KEY` 或其他 LLM API Key | LLM 调用 |
| `CREW_API_TOKEN` | Webhook/MCP Bearer 认证 |

### 推荐

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `CREW_CORS_ORIGINS` | CORS 允许的来源（逗号分隔） | 无（不启用 CORS） |
| `KNOWLYR_ID_URL` | knowlyr-id 服务地址 | `https://knowlyr.com` |
| `KNOWLYR_ID_TOKEN` | knowlyr-id API token | - |

### 邮件投递（可选）

| 变量 | 用途 |
|------|------|
| `SMTP_HOST` | SMTP 服务器 |
| `SMTP_PORT` | 端口（默认 587） |
| `SMTP_USER` | 用户名 |
| `SMTP_PASS` | 密码 |

## 安全配置

### Bearer Token 认证

```bash
knowlyr-crew serve --token "$CREW_API_TOKEN"
```

- 使用 `hmac.compare_digest` 做时序安全比较
- `/health` 和 `/webhook/github` 跳过认证
- GitHub Webhook 用 HMAC-SHA256 签名验证

### 请求限制

- **大小限制**: 默认 1MB，拒绝超大请求（HTTP 413）
- **速率限制**: 启用认证时自动生效，默认 60 请求/分钟/IP
  - `/health`、`/metrics`、`/webhook/github` 不受限制

### CORS

```bash
knowlyr-crew serve --cors-origins "https://app.example.com,https://admin.example.com"
```

## 部署方式

### systemd

```ini
[Unit]
Description=knowlyr-crew Webhook Server
After=network.target

[Service]
Type=simple
User=crew
WorkingDirectory=/opt/crew
Environment=ANTHROPIC_API_KEY=sk-...
Environment=CREW_API_TOKEN=your-secret-token
ExecStart=/opt/crew/.venv/bin/knowlyr-crew serve --host 0.0.0.0 --port 8765 --token $CREW_API_TOKEN
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Docker

```dockerfile
FROM python:3.12-slim
RUN pip install knowlyr-crew[webhook,execute,openai,id]
WORKDIR /app
COPY .crew/ .crew/
CMD ["knowlyr-crew", "serve", "--host", "0.0.0.0"]
```

## 监控

### 健康检查

```bash
curl http://localhost:8765/health
# {"status": "ok", "service": "crew-webhook"}
```

### 运行时指标

```bash
curl http://localhost:8765/metrics
```

返回：
- `calls`: 总调用次数、成功/失败
- `tokens`: 输入/输出 token 用量
- `latency`: 延迟统计（p50/p95/max）
- `by_employee`: 按员工统计
- `by_provider`: 按提供商统计（含错误分类）

### Cron 状态

```bash
curl http://localhost:8765/cron/status
```

## 备份

关键数据目录：
- `.crew/tasks.jsonl` — 任务历史（含断点数据）
- `.crew/memory/` — 员工持久记忆
- `.crew/sessions/` — 会话记录
- `.crew/pipelines/` — 流水线定义
- `.crew/cron.yaml` — 定时任务配置

建议定期备份 `.crew/` 目录。

## 扩展建议

- 使用反向代理（Nginx）处理 TLS 和负载均衡
- 对大型部署，考虑将 TaskRegistry 迁移到 Redis/PostgreSQL
- 监控 `/metrics` 端点的延迟 p95 值，设置告警阈值
