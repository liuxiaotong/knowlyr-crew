#!/usr/bin/env bash
# Crew 服务器一键部署脚本
# 用法: bash deploy/setup.sh
set -euo pipefail

DEPLOY_USER="deploy"
DEPLOY_HOME="/home/$DEPLOY_USER"
PROJECT_DIR="$DEPLOY_HOME/knowlyr-crew-private"

echo "=== 1. 安装 knowlyr-crew ==="
pip install --upgrade "knowlyr-crew[webhook,execute,openai,id]"

echo "=== 2. 初始化私有项目目录 ==="
mkdir -p "$PROJECT_DIR/.crew"

if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$(dirname "$0")/../.env.example" "$PROJECT_DIR/.env"
    echo ">>> 请编辑 $PROJECT_DIR/.env 填写实际的 API Key 和 Token"
fi

echo "=== 3. 安装 systemd service ==="
sudo cp "$(dirname "$0")/knowlyr-crew.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable knowlyr-crew

echo "=== 4. 安装 nginx 配置 ==="
sudo cp "$(dirname "$0")/nginx-crew.conf" /etc/nginx/sites-available/crew.knowlyr.com
sudo ln -sf /etc/nginx/sites-available/crew.knowlyr.com /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

echo ""
echo "=== 部署完成 ==="
echo ""
echo "下一步:"
echo "  1. 编辑 $PROJECT_DIR/.env 填写 API Key"
echo "  2. 配置 SSL 证书到 /etc/nginx/ssl/crew.knowlyr.com.{pem,key}"
echo "  3. 添加 DNS: crew.knowlyr.com → 服务器 IP"
echo "  4. 启动: sudo systemctl start knowlyr-crew"
echo "  5. 验证: curl https://crew.knowlyr.com/health"
