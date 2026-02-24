#!/bin/bash
# 每日评估 wrapper — 按人并行跑，3 并发
# cron: 0 23 * * * /opt/knowlyr-crew/project/scripts/run_daily_eval.sh >> /var/log/knowlyr-crew-eval.log 2>&1

set -a; source /opt/knowlyr-crew/.env; set +a
cd /opt/knowlyr-crew/project

if [ -z "$AIBERM_API_KEY" ]; then
    echo "[$(date)] ERROR: AIBERM_API_KEY 未设置，跳过评估"
    exit 1
fi

PYTHON=/opt/knowlyr-crew/venv/bin/python
SCRIPT=scripts/daily_eval.py
MODEL=claude-opus-4-6
PROVIDER=openai
BASE_URL=https://aiberm.com/v1
MAX_PARALLEL=3
DATE=${1:-$(date +%Y%m%d)}  # 可选参数：指定日期 YYYYMMDD，不传则评估当天

# 获取活跃员工列表
EMPLOYEES=$($PYTHON -c "
import json
from pathlib import Path
traj = Path('.crew/trajectories/trajectories.jsonl')
if not traj.exists():
    exit()
employees = set()
for line in traj.read_text().splitlines():
    try:
        d = json.loads(line)
        emp = d.get('metadata',{}).get('employee','') or d.get('employee_name','')
        if emp and emp not in {'unknown','unknown-agent',''}:
            employees.add(emp)
    except:
        pass
print(' '.join(sorted(employees)))
" 2>/dev/null)

if [ -z "$EMPLOYEES" ]; then
    echo "[$(date)] 无活跃员工数据，跳过"
    exit 0
fi

echo "[$(date)] 开始每日评估，员工: $EMPLOYEES"

DATE_ARG=""
[ -n "$DATE" ] && DATE_ARG="--date $DATE"

running=0
for emp in $EMPLOYEES; do
    echo "[$(date +%H:%M:%S)] 启动: $emp"
    $PYTHON $SCRIPT \
        $DATE_ARG --force --with-judge \
        --model $MODEL --provider $PROVIDER \
        --base-url $BASE_URL --api-key "$AIBERM_API_KEY" \
        --employee "$emp" &
    running=$((running + 1))
    if [ $running -ge $MAX_PARALLEL ]; then
        wait -n
        running=$((running - 1))
    fi
done
wait

echo "[$(date)] 每日评估完成"
