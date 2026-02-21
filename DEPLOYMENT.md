# knowlyr-crew 部署详情

**所有涉及 crew 的部署必须走以下路径，禁止其他方式。**

## 部署路径

| 场景 | 触发 | 流程 |
|------|------|------|
| 代码变更 | `git push main` (public) | CI: pip install → restart → 配置同步 → audit |
| 员工配置变更 | `git push main` (private) | CI: SSH → git pull → sync project-dir → restart → sync knowlyr-id |
| 紧急旁路 | `make push` (rsync) | 仅限紧急情况，事后必须补 git commit 到 private 仓库 |

## 仓库结构

| 仓库 | 类型 | 内容 |
|------|------|------|
| `knowlyr-crew` | Public | 框架代码、builtin 员工、测试 |
| `knowlyr-crew-private` | Private | 员工配置 (employee.yaml, prompt.md, soul.md)、组织架构 (organization.yaml) |

## 数据流

```
private repo (GitHub)
  ↓ CI: git pull --ff-only
服务器 /opt/knowlyr-crew/private-repo/ (源头)
  ↓ rsync
服务器 /opt/knowlyr-crew/project/private/ (副本)
  ↓ sync-all
knowlyr-id (员工注册信息)
```

**所有写操作先到 private 仓库，CI 自动同步到服务器。不直接写副本。**

## 服务器信息

- 主机: `knowlyr-web-1` (8.159.150.234)
- 路径: `/opt/knowlyr-crew/`
- Private 仓库 clone: `/opt/knowlyr-crew/private-repo/`
- Project 目录: `/opt/knowlyr-crew/project/`
- venv: `/opt/knowlyr-crew/venv/`
- 端口: `8765`
- 服务: `knowlyr-crew.service`

## 服务器端初始化（一次性）

```bash
# 1. 创建 deploy key
ssh-keygen -t ed25519 -f /opt/knowlyr-crew/.deploy-key -N "" -C "knowlyr-crew-private-deploy"
# 将公钥添加到 GitHub 仓库 Settings → Deploy keys (只读)

# 2. Clone private 仓库
GIT_SSH_COMMAND="ssh -i /opt/knowlyr-crew/.deploy-key" \
  git clone git@github.com:liuxiaotong/knowlyr-crew-private.git /opt/knowlyr-crew/private-repo

# 3. 配置 git 使用 deploy key
cd /opt/knowlyr-crew/private-repo
git config core.sshCommand "ssh -i /opt/knowlyr-crew/.deploy-key"
```

## 本地开发环境

```bash
# private/ 是 symlink，指向 private 仓库的本地 clone
# 开发体验不变，代码直接读 private/employees/
ls -la private/  # → ../knowlyr-crew-private

# 安装 git hooks（防止 private/ 文件误提交到 public 仓库）
make install-hooks
```

## 禁止操作（硬规则）

- ❌ **禁止直接 SSH 到服务器改 project-dir** — project/private/ 是副本，手动改了下次部署会被覆盖
- ❌ **禁止手动 pip install / systemctl restart** — 绕过流程会跳过配置同步和审计
- ❌ **禁止 rsync --delete 从本地到服务器** — 会删除服务器独有文件（data/、运行时缓存）
- ❌ **禁止 rsync 不排除 .env** — 服务器 .env 有完整凭据，本地只有 1 个变量
- ❌ **禁止 private/ 进 public git** — 有 pre-commit hook 硬拦，`make install-hooks` 安装
- ❌ **紧急旁路不补 commit** — 每次 `make push` 后必须在 private 仓库补一个 git commit

## CI 所需 GitHub Secrets（private 仓库）

| Secret | 说明 |
|--------|------|
| `SSH_PRIVATE_KEY` | 服务器 SSH 私钥（用于 CI 部署） |
| `SSH_HOST` | 服务器地址 |
| `SSH_USER` | SSH 用户名 |
