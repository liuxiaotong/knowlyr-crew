#!/usr/bin/env bash
# knowlyr-crew 部署脚本
# 用法: bash scripts/deploy.sh [--reinstall]
#   --reinstall  强制重新 pip install -e（src/ 有代码变更时用）

set -euo pipefail

SERVER="knowlyr-web-1"
REMOTE_DIR="/opt/knowlyr-crew"
PROJECT_DIR="$REMOTE_DIR/project"
VENV="$REMOTE_DIR/venv"
LOCAL_DIR="/Users/liukai/knowlyr-crew"
SERVICE="knowlyr-crew"
MIN_ENV_VARS=8

REINSTALL=false
[[ "${1:-}" == "--reinstall" ]] && REINSTALL=true

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

# 本地测试
echo "  运行测试..."
if cd "$LOCAL_DIR" && uv run --extra dev pytest tests/ -q --tb=no -x > /tmp/crew-test-output 2>&1; then
    TESTS=$(tail -1 /tmp/crew-test-output)
    pass "测试通过 ($TESTS)"
else
    cat /tmp/crew-test-output | tail -5
    fail "测试未通过，中止部署"
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
rsync -avz --delete \
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

# ── 3. 同步配置 ──
echo "3/6 同步配置"
ssh "$SERVER" "mkdir -p $PROJECT_DIR/private && cp $REMOTE_DIR/private/organization.yaml $PROJECT_DIR/private/organization.yaml"
pass "organization.yaml → project/"
echo ""

# ── 4. 重装包 ──
echo "4/6 重装包"
if $REINSTALL; then
    ssh "$SERVER" "$VENV/bin/pip install -e $REMOTE_DIR" > /dev/null 2>&1
    pass "pip install -e 完成"
else
    warn "跳过（无 --reinstall 参数）"
fi
echo ""

# ── 5. 重启服务 ──
echo "5/6 重启服务"
ssh "$SERVER" "sudo systemctl restart $SERVICE"
sleep 2
STATUS=$(ssh "$SERVER" "systemctl is-active $SERVICE" 2>/dev/null || echo "failed")
if [[ "$STATUS" == "active" ]]; then
    pass "服务已重启 (active)"
else
    fail "服务启动失败: $STATUS"
fi
echo ""

# ── 6. 验证 ──
echo "6/6 验证"

# API 健康检查
API_TOKEN=$(ssh "$SERVER" "grep CREW_API_TOKEN $REMOTE_DIR/.env | cut -d= -f2")
HTTP_CODE=$(ssh "$SERVER" "curl -s -o /dev/null -w '%{http_code}' -H 'Authorization: Bearer $API_TOKEN' http://127.0.0.1:8765/api/project/status" 2>/dev/null)
if [[ "$HTTP_CODE" == "200" ]]; then
    pass "API 返回 200"
else
    fail "API 返回 $HTTP_CODE"
fi

# Moonshot 计费
BILLING=$(ssh "$SERVER" "curl -s -H 'Authorization: Bearer $API_TOKEN' http://127.0.0.1:8765/api/project/status | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get(\"moonshot_billing\",{}).get(\"balance\",{}).get(\"balance_cny\",\"null\"))'" 2>/dev/null)
if [[ "$BILLING" != "null" && -n "$BILLING" ]]; then
    pass "Moonshot 计费正常 (余额 ¥$BILLING)"
else
    warn "Moonshot 计费无数据"
fi

# .env 完整性复查
POST_ENV=$(ssh "$SERVER" "wc -l < $REMOTE_DIR/.env" 2>/dev/null || echo 0)
if (( POST_ENV >= MIN_ENV_VARS )); then
    pass ".env 完整 (${POST_ENV} 行，未被覆盖)"
else
    fail ".env 被覆盖！只剩 ${POST_ENV} 行"
fi

echo ""
echo -e "${GREEN}═══ 部署完成 ═══${NC}"
