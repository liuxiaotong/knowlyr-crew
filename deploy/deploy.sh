#!/usr/bin/env bash
# knowlyr-crew 部署脚本
# 用法（从项目根目录执行）:
#   bash deploy/deploy.sh              # 同步数据 + 重启 + ID同步
#   bash deploy/deploy.sh sync         # 同步私有数据 + 重启 + ID同步
#   bash deploy/deploy.sh engine       # 更新引擎 + 重启
#   bash deploy/deploy.sh restart      # 只重启服务
#   bash deploy/deploy.sh id-sync      # 只同步到 knowlyr-id
#   bash deploy/deploy.sh all          # 引擎 + 数据 + 重启 + ID同步
set -euo pipefail

SERVER="knowlyr-web-1"
REMOTE_DIR="/opt/knowlyr-crew"
VENV="$REMOTE_DIR/venv"
PROJECT="$REMOTE_DIR/project"
ENGINE_REPO="git+https://github.com/liuxiaotong/knowlyr-crew.git"

# 本地私有数据目录（相对于项目根）
LOCAL_CREW=".crew"

ACTION="${1:-sync}"

sync_data() {
    echo "=== 同步私有数据 ==="

    # 员工: .crew/global/ → server:private/employees/
    echo "  同步员工..."
    rsync -av --delete \
        "$LOCAL_CREW/global/" \
        "$SERVER:$PROJECT/private/employees/"

    # 讨论会
    echo "  同步讨论会..."
    rsync -av --delete \
        --include="*.yaml" --exclude="*" \
        "$LOCAL_CREW/discussions/" \
        "$SERVER:$PROJECT/.crew/discussions/"

    # 流水线
    echo "  同步流水线..."
    rsync -av --delete \
        --include="*.yaml" --exclude="*" \
        "$LOCAL_CREW/pipelines/" \
        "$SERVER:$PROJECT/.crew/pipelines/"

    # 服务器端 git 提交
    echo "  记录版本..."
    ssh "$SERVER" "cd $PROJECT && git add -A && \
        git diff --cached --quiet || git commit -m 'sync: $(date +%Y%m%d-%H%M%S)'"

    echo "    数据已同步"
}

sync_id() {
    echo "=== 同步到 knowlyr-id ==="
    ssh "$SERVER" "set -a && source $REMOTE_DIR/.env && set +a && \
        $VENV/bin/knowlyr-crew agents sync-all \
        --dir $PROJECT/private/employees/ --force"
    echo "    ID 同步完成"
}

deploy_engine() {
    echo "=== 更新引擎 ==="
    ssh "$SERVER" "$VENV/bin/pip install --force-reinstall --no-deps \
        'knowlyr-crew[webhook,execute,openai,id] @ $ENGINE_REPO'"
    echo "    引擎已更新"
}

restart_service() {
    echo "=== 重启服务 ==="
    ssh "$SERVER" "systemctl restart knowlyr-crew"
    sleep 2
    local status
    status=$(ssh "$SERVER" "systemctl is-active knowlyr-crew")
    if [ "$status" = "active" ]; then
        echo "    服务已启动"
        ssh "$SERVER" "curl -s http://localhost:8765/health"
        echo ""
    else
        echo "    启动失败! 查看日志:"
        echo "      ssh $SERVER journalctl -u knowlyr-crew -n 20"
        exit 1
    fi
}

case "$ACTION" in
    sync)
        sync_data
        restart_service
        sync_id
        ;;
    engine)
        deploy_engine
        restart_service
        ;;
    restart)
        restart_service
        ;;
    id-sync)
        sync_id
        ;;
    all)
        deploy_engine
        sync_data
        restart_service
        sync_id
        ;;
    *)
        echo "用法: bash deploy/deploy.sh [sync|engine|restart|id-sync|all]"
        exit 1
        ;;
esac

echo "=== 部署完成 ==="
