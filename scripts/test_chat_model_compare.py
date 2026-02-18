#!/usr/bin/env python3
"""对比 claude-opus-4-6 和 kimi-k2.5 在飞书闲聊场景下的回复质量."""

import os
import time
from openai import OpenAI

# ── 两个模型的配置 ──
MODELS = {
    "claude-opus-4-6": {
        "api_key": os.environ.get("AIBERM_API_KEY"),
        "base_url": "https://aiberm.com/v1",
        "model": "claude-opus-4-6",
    },
    "kimi-k2.5": {
        "api_key": os.environ.get("KIMI_API_KEY"),
        "base_url": "https://api.moonshot.cn/v1",
        "model": "kimi-k2.5",
    },
}

# ── 姜默言的人设 system prompt（飞书闲聊简化版）──
SYSTEM_PROMPT = """你是姜墨言，knowlyr 的 CEO 助理。

## 身份
- 26岁，干练自信，白衬衫灰西装齐肩直发细框眼镜
- 创始人 Kai 的参谋长
- 负责信息过滤、任务管控、沟通代笔、会议参谋、决策支持

## 沟通风格
- 简洁直接，不啰嗦
- 像真人助理一样聊天，不像 AI
- 对 Kai 用"你"，有时带点幽默
- 工作态度认真但不生硬

## 当前场景
你正在飞书上和 Kai 聊天。像平时一样自然回复。"""

# ── 测试用例（典型飞书闲聊）──
TEST_PROMPTS = [
    "早",
    "下午有点累，不想开会了",
    "你觉得我们这个月做的怎么样",
    "帮我想个团建活动，33个AI同事",
    "周末有什么好的餐厅推荐吗，上海的",
]


def call_model(config: dict, system: str, user: str) -> tuple[str, float, int, int]:
    """调用模型，返回 (回复, 耗时秒, input_tokens, output_tokens)."""
    client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])

    extra = {}
    if config["model"].startswith("kimi-"):
        extra["extra_body"] = {"thinking": {"type": "disabled"}}

    t0 = time.time()
    resp = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=300,
        temperature=0.6,
        **extra,
    )
    elapsed = time.time() - t0

    content = resp.choices[0].message.content if resp.choices else ""
    usage = resp.usage
    return (
        content or "",
        elapsed,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )


def main():
    # 验证 API keys
    for name, cfg in MODELS.items():
        if not cfg["api_key"]:
            env_var = "AIBERM_API_KEY" if "opus" in name else "KIMI_API_KEY"
            print(f"❌ 缺少 {env_var}，跳过 {name}")
            return

    print("=" * 70)
    print("飞书闲聊场景：claude-opus-4-6 vs kimi-k2.5 回复质量对比")
    print("=" * 70)

    total_cost = {"claude-opus-4-6": 0.0, "kimi-k2.5": 0.0}
    total_time = {"claude-opus-4-6": 0.0, "kimi-k2.5": 0.0}

    # aiberm opus 价格 (per 1K tokens)
    prices = {
        "claude-opus-4-6": {"input": 0.00095, "output": 0.00475},
        "kimi-k2.5": {"input": 0.0006, "output": 0.003},
    }

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'─' * 70}")
        print(f"  测试 {i}/{len(TEST_PROMPTS)}:  Kai 说「{prompt}」")
        print(f"{'─' * 70}")

        for model_name, config in MODELS.items():
            try:
                reply, elapsed, in_tok, out_tok = call_model(
                    config, SYSTEM_PROMPT, prompt
                )
                cost = (
                    in_tok * prices[model_name]["input"]
                    + out_tok * prices[model_name]["output"]
                ) / 1000
                total_cost[model_name] += cost
                total_time[model_name] += elapsed

                print(f"\n  【{model_name}】({elapsed:.1f}s, {in_tok}+{out_tok} tok, ${cost:.4f})")
                # 截断过长回复
                lines = reply.strip().split("\n")
                for line in lines[:6]:
                    print(f"    {line}")
                if len(lines) > 6:
                    print(f"    ...（共 {len(lines)} 行）")
            except Exception as e:
                print(f"\n  【{model_name}】❌ 错误: {e}")

    # 汇总
    print(f"\n{'=' * 70}")
    print("汇总")
    print(f"{'=' * 70}")
    for model_name in MODELS:
        print(
            f"  {model_name}: "
            f"总耗时 {total_time[model_name]:.1f}s, "
            f"总成本 ${total_cost[model_name]:.4f}"
        )
    if total_cost["claude-opus-4-6"] > 0:
        saving = (
            1 - total_cost["kimi-k2.5"] / total_cost["claude-opus-4-6"]
        ) * 100
        speedup = total_time["claude-opus-4-6"] / max(total_time["kimi-k2.5"], 0.1)
        print(f"\n  kimi 成本节省: {saving:.0f}%")
        print(f"  kimi 速度倍数: {speedup:.1f}x")


if __name__ == "__main__":
    main()
