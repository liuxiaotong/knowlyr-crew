#!/bin/bash
# 系统健康巡检 — 纯 shell，不消耗 LLM token
# 通过系统 crontab 每 10 分钟运行
# 正常时静默，异常时调 crew API 派马骁通知
set -uo pipefail

CREW_API="http://127.0.0.1:8765"
CREW_TOKEN="X52I08vGWptmvtZxCMzX500odojsdv30k-gEq0G4sp8"
LOG="/var/log/knowlyr-health-check.log"

# 服务端点列表: name url
ENDPOINTS=(
  "knowlyr-id|http://127.0.0.1:8100/login"
  "knowlyr-ledger|http://127.0.0.1:8201/health"
  "knowlyr-crew|http://127.0.0.1:8765/health"
  "antgather|http://127.0.0.1:3100/projects"
  "knowlyr-website|http://127.0.0.1:80/"
)

FAILURES=""
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# 检查各服务端点
for entry in "${ENDPOINTS[@]}"; do
  NAME="${entry%%|*}"
  URL="${entry##*|}"
  HTTP_CODE=$(curl -sf -o /dev/null -w '%{http_code}' --max-time 10 "$URL" 2>/dev/null || echo "000")
  if [ "$HTTP_CODE" != "200" ]; then
    FAILURES="${FAILURES}${NAME}(HTTP ${HTTP_CODE}) "
    echo "$TIMESTAMP FAIL $NAME HTTP=$HTTP_CODE" >> "$LOG"
  fi
done

# 检查磁盘
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | tr -d '%')
if [ "$DISK_USAGE" -ge 85 ]; then
  FAILURES="${FAILURES}disk(${DISK_USAGE}%) "
  echo "$TIMESTAMP FAIL disk usage=${DISK_USAGE}%" >> "$LOG"
fi

# 全部正常 → 静默退出
if [ -z "$FAILURES" ]; then
  exit 0
fi

# 有异常 → 调 crew API 派马骁通知
ALERT_MSG="系统巡检异常: ${FAILURES}— 请立即排查。时间: ${TIMESTAMP}"

curl -s -X POST "${CREW_API}/api/employees/devops-engineer/run" \
  -H "Authorization: Bearer ${CREW_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"task\": \"紧急：${ALERT_MSG}\n\n请用 send_feishu_dm 通知 Kai，然后用 bash 检查相关服务日志（journalctl -u <service> -n 30）并给出初步判断。\", \"format\": \"memo\"}" \
  >> "$LOG" 2>&1

echo "$TIMESTAMP ALERT sent to devops-engineer: $FAILURES" >> "$LOG"
