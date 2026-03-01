# knowlyr-crew 部署详情

## 架构

**CREW 服务（数据库 + API）是唯一真理源 (Single Source of Truth)**

| 数据 | 存储位置 | 管理方式 |
|------|---------|---------|
| 引擎代码 | GitHub `knowlyr-crew` 仓库 | git push → CI 部署 |
| 员工配置（soul / prompt / 参数） | CREW PostgreSQL (`employee_souls` 表) | MCP 工具 / API |
| 讨论会定义 | CREW PostgreSQL (`discussions` 表) | MCP 工具 / API |
| 流水线定义 | CREW PostgreSQL (`pipelines` 表) | MCP 工具 / API |
| 组织架构 | CREW PostgreSQL + 服务器文件 | API |
| 记忆 | CREW PostgreSQL (`entries` 表) | MCP 工具 (`add_memory` / `query_memory`) |

> ⚠️ 旧版 `knowlyr-crew-private` 仓库已废弃。员工配置不再通过 git 管理，
> 全部通过 CREW API 在线管理。服务器上 `/opt/knowlyr-crew/private-repo/` 是历史遗留。

## 部署路径

| 场景 | 触发 | 流程 |
|------|------|------|
| 代码变更 | `make deploy`（= git pull + push） | CI: pip install → restart → health check → audit |
| 员工配置变更 | CREW MCP 工具 / API | 直接写数据库，实时生效 |
| 紧急旁路 | `make emergency-engine` | SSH: pip install → restart |

## CI 流程 (`deploy.yml`)

```
git push main
  → pip install knowlyr-crew 从 GitHub
  → systemctl restart knowlyr-crew
  → health check
  → 部署 scripts/ 和 static/avatars/
  → 员工配置审计
```

## 员工管理闭环（不再需要 git）

```
创建员工:  MCP run_employee / CREW API → 写入 employee_souls 表
修改配置:  MCP get_employee → 修改 → update → 写入 employee_souls 表（自动版本管理）
注册到 id: make register NAME=xxx 或 CREW API
查看状态:  MCP get_employee / list_employees
```

## 服务器信息

- 主机: `knowlyr-web-1` (8.159.150.234)
- 路径: `/opt/knowlyr-crew/`
- venv: `/opt/knowlyr-crew/venv/`
- 端口: `8765`
- 服务: `knowlyr-crew.service`
- 数据库: PostgreSQL (`knowlyr_crew`)

## Makefile 快捷命令

| 命令 | 用途 |
|------|------|
| `make deploy` | 推送代码（自动 pull + push → CI 部署） |
| `make status` | 查看服务器状态和版本 |
| `make emergency-engine` | 紧急：跳过 CI 直接更新引擎 |
| `make emergency-restart` | 紧急：只重启服务 |
| `make register NAME=xxx` | 注册员工到 knowlyr-id |
| `make test-employee NAME=xxx` | 测试员工（真实 webhook 链路） |

## 禁止操作（硬规则）

- ❌ **禁止直接 SSH 改服务器 venv 代码** — 下次 CI 部署会被覆盖
- ❌ **禁止本地 push 不先 pull** — SG 上的 Claude Code 可能已推新 commit
- ❌ **禁止通过文件修改员工配置** — CREW 数据库是唯一真理源，改文件下次重启会被覆盖
- ❌ **禁止 rsync 本地文件到服务器** — 本地可能落后远程

## CI 所需 GitHub Secrets

| Secret | 说明 |
|--------|------|
| `DEPLOY_SSH_KEY` | 服务器 SSH 私钥 |
| `DEPLOY_HOST` | 服务器地址 |
| `WEBSITE_DISPATCH_TOKEN` | 触发官网重建 |
