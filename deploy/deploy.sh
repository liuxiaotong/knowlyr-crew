#!/usr/bin/env bash
# knowlyr-crew 紧急运维脚本（非常规部署路径）
#
# 常规部署：git push main → GitHub Actions 自动执行
# 本脚本仅用于 CI 不可用时的紧急操作
#
# 用法:
#   bash deploy/deploy.sh engine       # 更新引擎代码 + 重启
#   bash deploy/deploy.sh restart      # 只重启服务
#   bash deploy/deploy.sh id-sync      # 同步员工到 knowlyr-id
#   bash deploy/deploy.sh trajectory   # 升级轨迹组件 + 重启
#   bash deploy/deploy.sh status       # 查看服务状态
set -euo pipefail

SERVER="knowlyr-web-1"
REMOTE_DIR="/opt/knowlyr-crew"
VENV="$REMOTE_DIR/venv"
PROJECT="$REMOTE_DIR/project"
ENGINE_REPO="git+https://github.com/liuxiaotong/knowlyr-crew.git"

ACTION="${1:-status}"

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
        'knowlyr-crew[webhook,execute,openai,id,trajectory] @ $ENGINE_REPO'"
    echo "    引擎已更新"
}

upgrade_trajectory() {
    echo "=== 更新轨迹组件 ==="
    ssh "$SERVER" "$VENV/bin/pip install --upgrade \
        --index-url https://pypi.org/simple/ \
        knowlyr-core knowlyr-sandbox knowlyr-recorder knowlyr-reward knowlyr-hub"
    echo "    轨迹组件已更新"
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

show_status() {
    echo "=== Crew 服务状态 ==="
    ssh "$SERVER" "systemctl is-active knowlyr-crew && curl -s http://localhost:8765/health && echo ''"
    echo "=== 引擎版本 ==="
    ssh "$SERVER" "$VENV/bin/pip show knowlyr-crew 2>/dev/null | grep Version"
    echo "=== 最近日志 ==="
    ssh "$SERVER" "journalctl -u knowlyr-crew --no-pager -n 5"
}

case "$ACTION" in
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
    trajectory)
        upgrade_trajectory
        restart_service
        ;;
    status)
        show_status
        ;;
    *)
        echo "用法: bash deploy/deploy.sh [engine|restart|id-sync|trajectory|status]"
        echo ""
        echo "  engine      更新引擎代码 + 重启（从 GitHub 安装最新）"
        echo "  restart     只重启服务"
        echo "  id-sync     同步员工信息到 knowlyr-id"
        echo "  trajectory  升级轨迹组件 + 重启"
        echo "  status      查看服务状态和版本"
        echo ""
        echo "注意: 员工配置通过 CREW API 在线管理（crew.knowlyr.com），不再需要 git 操作。"
        exit 1
        ;;
esac

echo "=== 完成 ==="
