#!/usr/bin/env bash
# knowlyr-crew 唯一正确的部署脚本
# 用法: bash scripts/deploy.sh [--skip-test]
#   --skip-test  跳过本地测试（仅配置变更时用）
#
# 部署流程 6 步:
#   1. 预检 — 本地测试 + 服务器 .env 完整性
#   2. 同步代码 — rsync（不用 --delete，排除 .env/data/.git）
#   3. 同步配置 — organization.yaml + employees/ → project-dir
#   4. 重装包 — 自动检测 src/ 有变更则 pip install -e
#   5. 重启服务 — systemctl restart
#   6. 验证 — API 健康检查 + .env 复查
#
# ⚠️ 安全规则:
#   - rsync 不用 --delete（防止删除服务器独有文件）
#   - 始终排除 .env（服务器 .env 有完整凭据，本地只有 1 个）
#   - private/ 不进 git，只通过此脚本同步到服务器

set -euo pipefail

SERVER="knowlyr-web-1"
REMOTE_DIR="/opt/knowlyr-crew"
PROJECT_DIR="$REMOTE_DIR/project"
VENV="$REMOTE_DIR/venv"
LOCAL_DIR="/Users/liukai/knowlyr-crew"
SERVICE="knowlyr-crew"
PORT=8765
MIN_ENV_VARS=8

SKIP_TEST=false
[[ "${1:-}" == "--skip-test" ]] && SKIP_TEST=true

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; exit 1; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }

echo "═══ knowlyr-crew 部署 ═══"
echo ""

# ── 1. 预检 ──
echo "1/6 预检"

if $SKIP_TEST; then
    warn "跳过本地测试（--skip-test）"
else
    echo "  运行测试..."
    if cd "$LOCAL_DIR" && uv run --extra dev pytest tests/ -q --tb=no -x > /tmp/crew-test-output 2>&1; then
        TESTS=$(tail -1 /tmp/crew-test-output)
        pass "测试通过 ($TESTS)"
    else
        tail -5 /tmp/crew-test-output
        fail "测试未通过，中止部署"
    fi
fi

# 服务器 .env 预检
ENV_COUNT=$(ssh "$SERVER" "wc -l < $REMOTE_DIR/.env" 2>/dev/null || echo 0)
if (( ENV_COUNT >= MIN_ENV_VARS )); then
    pass "服务器 .env 完整 (${ENV_COUNT} 行)"
else
    fail "服务器 .env 只有 ${ENV_COUNT} 行（需要 >= ${MIN_ENV_VARS}），请先修复"
fi

echo ""

# ── 2. 同步代码 ──
echo "2/6 同步代码"

# ⚠️ 不用 --delete，防止删除服务器独有文件（如 data/、运行时缓存）
rsync -avz \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='data' \
    --exclude='__pycache__' \
    --exclude='.claude' \
    --exclude='*.pyc' \
    --exclude='.env' \
    "$LOCAL_DIR/" "$SERVER:$REMOTE_DIR/" \
    > /tmp/crew-rsync-output 2>&1

CHANGED=$(grep -c '^>' /tmp/crew-rsync-output 2>/dev/null || echo 0)
pass "同步完成 (${CHANGED} 个文件变更)"
echo ""

# ── 3. 同步配置到 project-dir ──
# ⚠️ 必须做！crew 服务运行时从 project-dir 读取配置，
#    rsync 只同步到 /opt/knowlyr-crew/，不会自动同步到 project/ 子目录
echo "3/6 同步配置"

ssh "$SERVER" "mkdir -p $PROJECT_DIR/private/employees"
ssh "$SERVER" "cp $REMOTE_DIR/private/organization.yaml $PROJECT_DIR/private/organization.yaml"
pass "organization.yaml → project/"

ssh "$SERVER" "rsync -av $REMOTE_DIR/private/employees/ $PROJECT_DIR/private/employees/" > /dev/null 2>&1
pass "employees/ → project/"
echo ""

# ── 4. 重装包 ──
# 自动检测：对比本地 src/ 和服务器已安装版本的 mtime
echo "4/6 重装包"

# 检查 src/ 下有没有比上次部署更新的文件
NEED_REINSTALL=false
SRC_CHANGED=$(rsync -avzn \
    --include='src/***' \
    --exclude='*' \
    "$LOCAL_DIR/" "$SERVER:$REMOTE_DIR/" 2>/dev/null | grep -c '^>' || echo 0)

if (( SRC_CHANGED > 0 )) || [[ "${1:-}" == "--reinstall" ]]; then
    NEED_REINSTALL=true
fi

if $NEED_REINSTALL; then
    ssh "$SERVER" "$VENV/bin/pip install -e $REMOTE_DIR" > /dev/null 2>&1
    pass "pip install -e 完成（src/ 有变更）"
else
    pass "跳过（src/ 无变更）"
fi
echo ""

# ── 5. 重启服务 ──
echo "5/6 重启服务"
ssh "$SERVER" "sudo systemctl restart $SERVICE"
sleep 3
STATUS=$(ssh "$SERVER" "systemctl is-active $SERVICE" 2>/dev/null || echo "failed")
if [[ "$STATUS" == "active" ]]; then
    pass "服务已重启 (active)"
else
    # 打印最后几行日志帮助排查
    ssh "$SERVER" "journalctl -u $SERVICE --no-pager -n 10" 2>/dev/null || true
    fail "服务启动失败: $STATUS"
fi
echo ""

# ── 6. 验证 ──
echo "6/6 验证"

# API 健康检查
HTTP_CODE=$(ssh "$SERVER" "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health" 2>/dev/null)
if [[ "$HTTP_CODE" == "200" ]]; then
    pass "health 返回 200"
else
    fail "health 返回 $HTTP_CODE"
fi

# 带鉴权的 API 检查
API_TOKEN=$(ssh "$SERVER" "grep CREW_API_TOKEN $REMOTE_DIR/.env | cut -d= -f2")
HTTP_CODE=$(ssh "$SERVER" "curl -s -o /dev/null -w '%{http_code}' -H 'Authorization: Bearer $API_TOKEN' http://127.0.0.1:$PORT/api/project/status" 2>/dev/null)
if [[ "$HTTP_CODE" == "200" ]]; then
    pass "API 鉴权正常"
else
    warn "API 鉴权返回 $HTTP_CODE"
fi

# .env 完整性复查（防止 rsync 意外覆盖）
POST_ENV=$(ssh "$SERVER" "wc -l < $REMOTE_DIR/.env" 2>/dev/null || echo 0)
if (( POST_ENV >= MIN_ENV_VARS )); then
    pass ".env 完整 (${POST_ENV} 行，未被覆盖)"
else
    fail ".env 被覆盖！只剩 ${POST_ENV} 行"
fi

echo ""
echo -e "${GREEN}═══ 部署完成 ═══${NC}"
