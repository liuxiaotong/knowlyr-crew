"""讨论会引擎 — 多员工多轮讨论."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from crew.context_detector import detect_project
from crew.discovery import discover_employees
from crew.engine import CrewEngine


# ── 数据模型 ──


class DiscussionParticipant(BaseModel):
    """讨论会参与者."""

    employee: str = Field(description="员工名称")
    role: Literal["moderator", "speaker", "recorder"] = Field(
        default="speaker", description="会议角色"
    )
    focus: str = Field(default="", description="本次讨论的关注重点")


class DiscussionRound(BaseModel):
    """讨论轮次配置."""

    name: str = Field(default="", description="轮次名称")
    instruction: str = Field(default="", description="该轮特殊指令")


class Discussion(BaseModel):
    """讨论会定义."""

    name: str = Field(description="讨论会名称")
    description: str = Field(default="", description="描述")
    topic: str = Field(description="议题（支持 $variable）")
    goal: str = Field(default="", description="讨论目标")
    participants: list[DiscussionParticipant] = Field(description="参与者列表")
    rounds: int | list[DiscussionRound] = Field(default=3, description="讨论轮次")
    output_format: Literal["decision", "transcript", "summary"] = Field(
        default="decision", description="输出格式"
    )


# ── 加载 / 校验 / 渲染 / 发现 ──


def load_discussion(path: Path) -> Discussion:
    """从 YAML 文件加载讨论会定义."""
    content = path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)

    # rounds 可以是 int 或 list[dict]，需要转换
    if isinstance(data.get("rounds"), list):
        data["rounds"] = [
            DiscussionRound(**r) if isinstance(r, dict) else r for r in data["rounds"]
        ]

    return Discussion(**data)


def validate_discussion(
    discussion: Discussion, project_dir: Path | None = None
) -> list[str]:
    """校验讨论会定义，返回错误列表."""
    errors: list[str] = []

    if len(discussion.participants) < 2:
        errors.append("讨论会至少需要 2 个参与者")
        return errors

    rounds_count = (
        discussion.rounds if isinstance(discussion.rounds, int) else len(discussion.rounds)
    )
    if rounds_count < 1:
        errors.append("讨论会至少需要 1 轮")

    result = discover_employees(project_dir=project_dir)
    for p in discussion.participants:
        emp = result.get(p.employee)
        if emp is None:
            errors.append(f"未找到员工: '{p.employee}'")

    return errors


def render_discussion(
    discussion: Discussion,
    initial_args: dict[str, str] | None = None,
    project_dir: Path | None = None,
    agent_id: int | None = None,
    smart_context: bool = True,
) -> str:
    """渲染讨论会，生成完整的讨论指令 prompt."""
    initial_args = initial_args or {}
    result = discover_employees(project_dir=project_dir)
    engine = CrewEngine()

    project_info = detect_project(project_dir) if smart_context else None

    # 获取 agent 身份（可选）
    agent_identity = None
    if agent_id is not None:
        try:
            from crew.id_client import fetch_agent_identity

            agent_identity = fetch_agent_identity(agent_id)
        except ImportError:
            pass

    # 变量替换（topic, goal）
    topic = discussion.topic
    goal = discussion.goal
    for k, v in initial_args.items():
        topic = topic.replace(f"${k}", v)
        goal = goal.replace(f"${k}", v)

    # 解析参与者
    participants_info = []
    for p in discussion.participants:
        emp = result.get(p.employee)
        participants_info.append({"participant": p, "employee": emp})

    role_labels = {"moderator": "主持人", "speaker": "发言人", "recorder": "记录员"}
    parts: list[str] = []

    # ── 头部 ──
    parts.append(f"# 讨论会：{discussion.description or discussion.name}")
    parts.append("")
    parts.append(f"**议题**: {topic}")
    if goal:
        parts.append(f"**目标**: {goal}")

    if project_info and project_info.project_type != "unknown":
        parts.append(f"**项目类型**: {project_info.display_label}")
    parts.append("")

    # ── 参会者 ──
    parts.append("---")
    parts.append("")
    parts.append("## 参会者")
    parts.append("")

    for info in participants_info:
        p = info["participant"]
        emp = info["employee"]

        if emp is None:
            parts.append(f"### {p.employee}（{role_labels[p.role]}）— 未找到")
            parts.append("")
            continue

        parts.append(f"### {emp.effective_display_name}（{role_labels[p.role]}）")
        parts.append(f"**描述**: {emp.description}")
        if p.focus:
            parts.append(f"**本次关注**: {p.focus}")
        if emp.tags:
            parts.append(f"**标签**: {', '.join(emp.tags)}")
        parts.append("")

        # 注入员工专业背景（body），通过引擎渲染变量
        rendered_body = engine.render(emp, args=dict(initial_args))
        parts.append(f"<专业背景>\n{rendered_body}\n</专业背景>")
        parts.append("")

    # ── 讨论规则 ──
    parts.append("---")
    parts.append("")
    parts.append("## 讨论规则")
    parts.append("")
    parts.append("1. 每轮讨论中，每位参会者必须以自己的专业视角发言")
    parts.append("2. 发言时标注角色，格式：**【角色名】**: 发言内容")
    parts.append("3. 后续轮次中，每位参会者应回应前轮他人的观点")
    parts.append("4. 主持人负责引导方向、总结争议、推动共识")
    parts.append("5. 记录员在最后一轮整理结构化的会议记录")
    parts.append("6. 鼓励建设性的分歧——不同观点有助于全面分析")
    parts.append("")

    # ── 轮次安排 ──
    parts.append("## 轮次安排")
    parts.append("")

    if isinstance(discussion.rounds, int):
        for i in range(1, discussion.rounds + 1):
            if i == 1:
                parts.append(f"### 第 {i} 轮：开场")
                parts.append("主持人介绍议题，每位参会者从自身专业角度给出初步观点。")
            elif i == discussion.rounds:
                parts.append(f"### 第 {i} 轮：总结与决议")
                parts.append("主持人总结各方观点，达成共识，形成明确的决议和行动项。")
            else:
                parts.append(f"### 第 {i} 轮：深入讨论")
                parts.append("回应前轮观点，深入探讨分歧点，提出具体方案。")
            parts.append("")
    else:
        for i, rnd in enumerate(discussion.rounds, 1):
            title = rnd.name or f"第 {i} 轮"
            parts.append(f"### {title}")
            if rnd.instruction:
                parts.append(rnd.instruction)
            parts.append("")

    # ── 输出格式 ──
    parts.append("---")
    parts.append("")
    parts.append("## 输出格式")
    parts.append("")
    parts.append(_OUTPUT_TEMPLATES[discussion.output_format])

    return "\n".join(parts)


# ── 发现 ──

DISCUSSIONS_DIR_NAME = "discussions"


def discover_discussions(project_dir: Path | None = None) -> dict[str, Path]:
    """发现所有可用讨论会.

    搜索顺序：
    1. 内置（src/crew/employees/discussions/）
    2. 项目（.crew/discussions/）— 同名覆盖内置
    """
    discussions: dict[str, Path] = {}

    # 内置讨论会
    builtin_dir = Path(__file__).parent / "employees" / DISCUSSIONS_DIR_NAME
    if builtin_dir.is_dir():
        for f in sorted(builtin_dir.glob("*.yaml")):
            discussions[f.stem] = f

    # 项目讨论会（覆盖同名内置）
    root = Path(project_dir) if project_dir else Path.cwd()
    project_dir_path = root / ".crew" / DISCUSSIONS_DIR_NAME
    if project_dir_path.is_dir():
        for f in sorted(project_dir_path.glob("*.yaml")):
            discussions[f.stem] = f

    return discussions


# ── 输出格式模板 ──

_OUTPUT_TEMPLATES = {
    "decision": """\
请按以下格式输出讨论结果：

# 讨论会记录

## 参会者
| 角色 | 姓名 | 关注方向 |
|------|------|---------|
（列出所有参会者）

## 讨论过程
（每轮每人的发言，格式：**【角色名】**: 发言内容）

## 决议

### 达成共识
1. ...

### 待解决分歧
1. ...（如有）

### 行动项
| 序号 | 事项 | 建议负责角色 | 优先级 |
|------|------|-------------|--------|
| 1 | ... | ... | P0-P3 |

### 风险清单
| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| ... | 高/中/低 | ... |""",
    "transcript": """\
请按以下格式输出完整的讨论记录：

# 讨论会完整记录

## 参会者
（列出所有参会者及角色）

## 第 N 轮
**【角色名】**: 完整发言内容...
（保留所有讨论细节，不做压缩）""",
    "summary": """\
请按以下格式输出讨论总结：

# 讨论会总结

## 议题
（一句话概括）

## 主要观点
- **角色A**: 核心观点...
- **角色B**: 核心观点...

## 共识
1. ...

## 后续行动
1. ...""",
}
