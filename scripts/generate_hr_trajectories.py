#!/usr/bin/env python3
"""批量执行心蕾（hr-manager）HR 训练场景，生成轨迹数据.

服务器端的 TrajectoryCollector 会自动录制每次调用的轨迹。
本脚本只负责按顺序发起请求并汇总结果。

用法:
    # 默认跑前 5 个场景（验证用）
    python scripts/generate_hr_trajectories.py

    # 跑第 10 到第 20 个场景
    python scripts/generate_hr_trajectories.py --start 10 --end 20

    # 跑全部 63 个场景
    python scripts/generate_hr_trajectories.py --start 1 --end 63

    # 自定义间隔（默认 2 秒）
    python scripts/generate_hr_trajectories.py --delay 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# ── 常量 ──

API_URL = "https://crew.knowlyr.com/run/employee/hr-manager"
API_TOKEN = "X52I08vGWptmvtZxCMzX500odojsdv30k-gEq0G4sp8"

# 场景标题（用于输出摘要）
SCENARIO_TITLES = [
    "新岗位需求评估",           # 01
    "Soul 编写指导",            # 02
    "Prompt 设计最佳实践",      # 03
    "跨领域岗位设计",           # 04
    "批量创建同类型员工",       # 05
    "整体绩效盘点",             # 06
    "单个员工深度评估",         # 07
    "员工能力边界测试",         # 08
    "Soul 调优建议",            # 09
    "员工性格一致性问题",       # 10
    "编制充足性评估",           # 11
    "冗余检查",                 # 12
    "组织架构建议",             # 13
    "岗位优先级排序",           # 14
    "阶段性编制规划",           # 15
    "模型分级策略",             # 16
    "模型升级评估",             # 17
    "特定任务的模型选型",       # 18
    "工作流串联",               # 19
    "最佳搭档组合",             # 20
    "跨组协作机制",             # 21
    "新员工融入流程",           # 22
    "冻结标准制定",             # 23
    "具体淘汰建议",             # 24
    "员工转型/重定义",          # 25
    "完整入职 SOP",             # 26
    "入职验收标准",             # 27
    "Token 消耗分析",           # 28
    "ROI 评估框架",             # 29
    "成本优化方案",             # 30
    "招聘需求分析",             # 31
    "JD 撰写",                  # 32
    "面试流程设计",             # 33
    "候选人评估",               # 34
    "薪酬体系搭建",             # 35
    "期权方案",                 # 36
    "绩效考核体系",             # 37
    "绩效问题处理",             # 38
    "人类与 AI 员工绩效对比",   # 39
    "文化定义",                 # 40
    "团队凝聚力",               # 41
    "新人融入",                 # 42
    "人机协作文化",             # 43
    "劳动合同规范",             # 44
    "远程工作政策",             # 45
    "社保与合规",               # 46
    "年度人力规划",             # 47
    "组织健康度诊断",           # 48
    "离职处理",                 # 49
    "培训体系",                 # 50
    "加薪谈判话术",             # 51
    "绩效不达标沟通话术",       # 52
    "劝退谈话话术",             # 53
    "候选人 offer 谈判话术",    # 54
    "拒绝候选人的回复",         # 55
    "试用期评估表",             # 56
    "离职交接清单",             # 57
    "年度调薪方案",             # 58
    "员工之间的矛盾调解",       # 59
    "核心人才被挖",             # 60
    "辞退的法律风险评估",       # 61
    "病假/产假处理",            # 62
    "人员流失率分析",           # 63
]

# 63 个场景的用户提问
SCENARIOS = [
    # 01: 新岗位需求评估
    "心蕾，我想加一个 AI 员工专门做竞品分析，你觉得有没有必要？现在的人能不能覆盖？",
    # 02: Soul 编写指导
    "新员工的 soul 怎么写才能让他真的像那个角色？你给个框架。",
    # 03: Prompt 设计最佳实践
    "林锐的代码审查 prompt 总是抓不住重点，要么太啰嗦要么漏东西。怎么改？",
    # 04: 跨领域岗位设计
    "我需要一个既懂数据又懂业务的人，能帮客户做数据质量诊断还能给方案。现在林晓桐管数据质量，黄维达管解决方案，要不要合并？还是再加一个？",
    # 05: 批量创建同类型员工
    "RLHF 标注量上来了，一个林晓桐不够用。我想再加两三个数据质量方向的 AI 员工，怎么做差异化？",
    # 06: 整体绩效盘点
    "心蕾，33 个 AI 员工，你觉得谁表现最好？谁最拉垮？给我一个大概的印象。",
    # 07: 单个员工深度评估
    "苏文最近写的文档质量怎么样？我感觉有时候太模板化了。",
    # 08: 员工能力边界测试
    "方逸凡能不能处理多模态的算法研究？还是说他只能搞 NLP 方向的？",
    # 09: Soul 调优建议
    "姜墨言有时候太主动了，会自己做决策而不是先问我。怎么在 soul 里调？",
    # 10: 员工性格一致性问题
    "我发现有些员工回复的风格很不稳定，同一个人有时候很正式有时候很随意。怎么解决？",
    # 11: 编制充足性评估
    "33 个 AI 员工够不够？我感觉有些事情还是没人管。",
    # 12: 冗余检查
    "有没有职责重叠的员工？我不想养闲人。",
    # 13: 组织架构建议
    "33 个人全是扁平的，都向墨言或我汇报。要不要分组？怎么分？",
    # 14: 岗位优先级排序
    "如果预算只够再加 3 个 AI 员工，你建议加什么岗位？",
    # 15: 阶段性编制规划
    "从天使轮到 A 轮，AI 员工团队应该怎么演进？给我一个路线图。",
    # 16: 模型分级策略
    "不是每个员工都需要用最好的模型吧？哪些员工可以用便宜点的模型？",
    # 17: 模型升级评估
    "新的 Claude 模型出了，要不要全员升级？还是只升部分人？",
    # 18: 特定任务的模型选型
    "我们要做一个自动化的标注质量审核流程，用哪个模型跑比较合适？",
    # 19: 工作流串联
    "从客户提需求到最终交付数据，中间应该经过哪些 AI 员工？串一个流程出来。",
    # 20: 最佳搭档组合
    "哪些员工天然适合搭配在一起？哪些放在一起容易出问题？",
    # 21: 跨组协作机制
    "研发组和数据组经常需要配合，但总感觉信息不通畅。怎么改善？",
    # 22: 新员工融入流程
    "新加的 AI 员工怎么让他快速融入团队？其他员工怎么知道有新人来了？",
    # 23: 冻结标准制定
    "什么情况下应该冻结一个 AI 员工？给我一个判断标准。",
    # 24: 具体淘汰建议
    "你觉得现在 33 个人里有没有可以砍掉的？",
    # 25: 员工转型/重定义
    "傅语桥现在国际化的活不多，但我不想直接砍掉。能不能让他转做别的？",
    # 26: 完整入职 SOP
    "从决定要一个新 AI 员工到他能正常工作，完整流程是什么？帮我列一个 SOP。",
    # 27: 入职验收标准
    "新员工上线前要通过什么测试？谁来验收？",
    # 28: Token 消耗分析
    "每个月 AI 员工的 token 消耗大概是什么情况？哪些员工最费钱？",
    # 29: ROI 评估框架
    "怎么衡量一个 AI 员工值不值？给我一个 ROI 的算法。",
    # 30: 成本优化方案
    "这个月 AI 的费用超了，怎么砍下来但不影响业务？",
    # 31: 招聘需求分析
    "心蕾，我们现在最缺什么人？下一个要招的人类员工应该是什么岗位？",
    # 32: JD 撰写
    "帮我写一个数据标注项目经理的 JD，要能吸引好的人，但不要写得太高大上吓跑人。",
    # 33: 面试流程设计
    "我们面试流程太随意了，有时候我聊两句就定了。帮我设计一个正式点的流程。",
    # 34: 候选人评估
    "这个人简历看着不错但面试表现一般，要不要？",
    # 35: 薪酬体系搭建
    "我们现在发工资比较随意，每个人谈的都不一样。要不要搞一个体系？",
    # 36: 期权方案
    "期权池怎么设？给多少合适？vest 周期怎么定？",
    # 37: 绩效考核体系
    "20 多个人，每个月怎么看谁干得好谁干得不好？搞个绩效考核方案。",
    # 38: 绩效问题处理
    "有个员工最近状态不好，产出明显下降。怎么处理？",
    # 39: 人类与 AI 员工绩效对比
    "有些活 AI 做得比人还好。我怎么跟人类员工解释这件事，不让他们觉得被威胁？",
    # 40: 文化定义
    "集识光年的企业文化应该是什么？帮我提炼。",
    # 41: 团队凝聚力
    "远程团队怎么搞团建？人和 AI 都算上。",
    # 42: 新人融入
    "新人来了前三天应该做什么？现在的 onboarding 太乱了。",
    # 43: 人机协作文化
    "怎么让人类员工自然地把 AI 员工当同事用起来，而不是当工具？",
    # 44: 劳动合同规范
    "我们的劳动合同是不是该规范一下？创业公司有什么容易踩的坑？",
    # 45: 远程工作政策
    "我们要不要允许全远程？怎么管？",
    # 46: 社保与合规
    "标注员是按兼职还是全职签？有没有合规风险？",
    # 47: 年度人力规划
    "今年的人力预算怎么做？人类员工和 AI 员工一起算。",
    # 48: 组织健康度诊断
    "你觉得我们团队现在最大的问题是什么？",
    # 49: 离职处理
    "有个核心员工要走，怎么办？",
    # 50: 培训体系
    "我们需不需要搞培训？都培训什么？",
    # 51: 加薪谈判话术
    "心蕾，我下周要跟数据标注组的组长谈加薪，他来了一年半，表现不错但公司现在现金流紧。帮我拟个话术，既要肯定他又不能承诺太多。",
    # 52: 绩效不达标沟通话术
    "有个标注员连续两个月质量分倒数，我得找他谈，但怕他觉得被针对。帮我想想怎么开口。",
    # 53: 劝退谈话话术
    "心蕾，有个员工试用期表现确实不行，决定不留了。这种谈话我没经验，帮我准备一下，别搞出劳动仲裁。",
    # 54: 候选人 offer 谈判话术
    "有个很想要的算法工程师，但他要的薪资比我们预算高 30%。帮我想想怎么谈，既有诚意又不超预算。",
    # 55: 拒绝候选人的回复
    "面了三轮的那个产品经理最终没过，但人家也花了不少时间，帮我写个拒绝邮件，别太冷冰冰的。",
    # 56: 试用期评估表
    "下个月有三个人到试用期了，我们还没有正式的评估表。帮我出一个，适合我们这种小公司的，别太复杂。",
    # 57: 离职交接清单
    "有人要走了，之前离职交接都是口头说说，这次帮我做个正式的清单模板。",
    # 58: 年度调薪方案
    "快到年底了，得做明年的调薪方案。公司 20 多人，天使轮，帮我理一下思路和框架。",
    # 59: 员工之间的矛盾调解
    "标注组两个组长闹矛盾，一个觉得另一个抢了他的优质任务，影响到组员了。我该怎么介入？",
    # 60: 核心人才被挖
    "心蕾，听说竞对在挖我们的首席算法工程师，他还没提离职但感觉有动摇。怎么办？",
    # 61: 辞退的法律风险评估
    "有个员工表现一般，不算差但也不好，合同还有一年。如果想让他走，法律上有什么风险？帮我分析一下。",
    # 62: 病假/产假处理
    "有个员工说她怀孕了，产假怎么算？期间工资怎么发？我不太懂上海这边的规定。",
    # 63: 人员流失率分析
    "心蕾，今年走了好几个人，帮我分析一下我们的流失情况，看看有没有什么规律。",
]


def call_hr_api(user_message: str, timeout: int = 120) -> dict:
    """调用 /run/employee/hr-manager API，返回 JSON 响应."""
    payload = json.dumps(
        {"user_message": user_message, "channel": "training"},
        ensure_ascii=False,
    ).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量执行心蕾 HR 训练场景，生成轨迹数据",
    )
    parser.add_argument(
        "--start", type=int, default=1,
        help="起始场景编号（从 1 开始，默认 1）",
    )
    parser.add_argument(
        "--end", type=int, default=5,
        help="结束场景编号（含，默认 5）",
    )
    parser.add_argument(
        "--delay", type=float, default=2,
        help="每个场景之间的间隔秒数（默认 2）",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="单次 API 调用超时秒数（默认 120）",
    )
    args = parser.parse_args()

    total = len(SCENARIOS)
    start = max(1, args.start)
    end = min(total, args.end)

    if start > end:
        print(f"错误: --start ({start}) 大于 --end ({end})")
        sys.exit(1)

    count = end - start + 1
    print(f"=== HR 轨迹生成 ===")
    print(f"员工: hr-manager (心蕾)")
    print(f"范围: #{start:02d} ~ #{end:02d} (共 {count} 个场景)")
    print(f"间隔: {args.delay}s")
    print()

    success_count = 0
    fail_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for i in range(start - 1, end):
        num = i + 1
        title = SCENARIO_TITLES[i] if i < len(SCENARIO_TITLES) else f"场景 {num}"
        message = SCENARIOS[i]

        try:
            result = call_hr_api(message, timeout=args.timeout)
            in_tok = result.get("input_tokens", 0)
            out_tok = result.get("output_tokens", 0)
            tokens = in_tok + out_tok
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            success_count += 1
            print(f"[OK]   #{num:02d} | tokens: {tokens:>6} (in:{in_tok} out:{out_tok}) | {title}")
        except urllib.error.HTTPError as e:
            fail_count += 1
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                pass
            print(f"[FAIL] #{num:02d} | HTTP {e.code}: {body} | {title}")
        except urllib.error.URLError as e:
            fail_count += 1
            print(f"[FAIL] #{num:02d} | 连接错误: {e.reason} | {title}")
        except Exception as e:
            fail_count += 1
            print(f"[FAIL] #{num:02d} | {e} | {title}")

        # 间隔等待（最后一个不等）
        if num < end:
            time.sleep(args.delay)

    # 汇总
    total_tokens = total_input_tokens + total_output_tokens
    print()
    print(f"=== 完成 ===")
    print(f"成功: {success_count} / {count}")
    print(f"失败: {fail_count} / {count}")
    print(f"Token 消耗: {total_tokens} (input: {total_input_tokens}, output: {total_output_tokens})")


if __name__ == "__main__":
    main()
