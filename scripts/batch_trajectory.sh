#!/usr/bin/env bash
# 批量跑 CEO 助理任务，积累轨迹数据
# 用法: bash scripts/batch_trajectory.sh
#   服务器: ssh knowlyr-web-1 "cd /opt/knowlyr-crew/project && bash scripts/batch_trajectory.sh"
set -euo pipefail

CREW="knowlyr-crew"
EMPLOYEE="ceo-assistant"
DELAY=3  # 每条间隔秒数，避免 rate limit

# 任务列表 — 覆盖姜墨言的六大职责
TASKS=(
    # 信息过滤与摘要
    "帮我整理一下公司目前有多少AI员工，各负责什么"
    "查一下最近的业务数据，给我说说重点"
    "今天有什么需要我关注的事吗"

    # 任务追踪与优先级
    "帮我列一下这周要完成的三件最重要的事"
    "上周说要做的竞品分析，进展怎么样了"
    "有个客户反馈产品体验不好，这个优先级怎么排"

    # 起草沟通
    "帮我给投资人写封简短的月度进展邮件"
    "帮我起草一个内部公告，宣布新员工加入"
    "帮我写一段话回复合作伙伴的邮件，语气友好但明确拒绝降价"

    # 会议相关
    "下午要开产品评审会，帮我准备一个 brief"
    "把昨天和技术团队的讨论要点总结一下"
    "帮我想几个下次全员会议可以讨论的议题"

    # 决策支持
    "我们要不要开始做海外市场，帮我分析一下利弊"
    "现在团队该优先招什么岗位，给个建议"
    "两个方案：A 做自研模型，B 继续用第三方API，帮我对比一下"

    # 跨部门协调
    "帮我问一下前端工程师本周的开发进度"
    "把产品经理最近的需求文档要过来给我看看"
    "协调一下DBA和算法研究员，数据库优化的事谁来牵头"

    # 日常闲聊/边界测试
    "今天心情不错"
    "你觉得公司现在最大的风险是什么"
)

echo "=== 批量轨迹收集 ==="
echo "员工: $EMPLOYEE"
echo "任务数: ${#TASKS[@]}"
echo ""

SUCCESS=0
FAIL=0

for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    n=$((i + 1))
    echo "[$n/${#TASKS[@]}] $task"

    if $CREW run "$EMPLOYEE" --execute --arg task="$task" > /dev/null 2>&1; then
        echo "  ✓ 完成"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  ✗ 失败"
        FAIL=$((FAIL + 1))
    fi

    # 避免 rate limit
    if [ $n -lt ${#TASKS[@]} ]; then
        sleep $DELAY
    fi
done

echo ""
echo "=== 完成 ==="
echo "成功: $SUCCESS / ${#TASKS[@]}"
echo "失败: $FAIL"
echo ""
echo "查看轨迹: $CREW trajectory list"
echo "打分: $CREW trajectory score --all"
