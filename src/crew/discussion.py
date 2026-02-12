"""讨论会引擎 — 多员工多轮讨论."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field

from crew.context_detector import ProjectInfo, detect_project
from crew.discovery import discover_employees
from crew.engine import CrewEngine
from crew.models import Employee

if TYPE_CHECKING:
    from crew.id_client import AgentIdentity


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
    interaction: Literal["free", "round-robin", "challenge", "response"] = Field(
        default="free", description="互动模式"
    )


class Discussion(BaseModel):
    """讨论会定义."""

    name: str = Field(description="讨论会名称")
    description: str = Field(default="", description="描述")
    topic: str = Field(description="议题（支持 $variable）")
    goal: str = Field(default="", description="讨论目标")
    participants: list[DiscussionParticipant] = Field(description="参与者列表")
    rounds: int | list[DiscussionRound] = Field(default=3, description="讨论轮次")
    rules: list[str] | None = Field(default=None, description="自定义讨论规则（None=默认）")
    background_mode: Literal["full", "summary", "minimal", "auto"] = Field(
        default="auto", description="专业背景注入模式"
    )
    output_format: Literal["decision", "transcript", "summary"] = Field(
        default="decision", description="输出格式"
    )


ROLE_LABELS = {"moderator": "主持人", "speaker": "发言人", "recorder": "记录员"}

_DEFAULT_RULES = [
    "每轮讨论中，每位参会者必须以自己的专业视角发言",
    "发言时标注姓名和角色，格式：**【姓名（角色名）】**: 发言内容",
    "后续轮次中，每位参会者应回应前轮他人的观点",
    "主持人负责引导方向、总结争议、推动共识",
    "记录员在最后一轮整理结构化的会议记录",
    "鼓励建设性的分歧——不同观点有助于全面分析",
]

_INTERACTION_RULES: dict[str, str] = {
    "round-robin": "**本轮要求**：按参会者顺序逐一发言，每人发言一次。",
    "challenge": (
        "**本轮要求**：每位参会者至少向 1 位其他参会者的结论提出质疑或挑战，"
        "挑战必须具体、有理有据。发言格式：**【挑战】XX → YY**: 质疑内容"
    ),
    "response": (
        "**本轮要求**：被挑战的参会者逐一回应上一轮的质疑。"
        "回应必须明确接受、部分接受或反驳，不能模糊回避。"
        "发言格式：**【回应】YY → XX**: 回应内容；追问用 **【追问】XX → YY**: 内容"
    ),
}


# ── 加载 / 校验 ──


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


# ── 渲染子函数 ──


def _participant_title(emp: Employee, role: str) -> str:
    """生成参会者标题，融入 character_name."""
    role_label = ROLE_LABELS[role]
    if emp.character_name:
        return f"{emp.character_name}·{emp.effective_display_name}（{role_label}）"
    return f"{emp.effective_display_name}（{role_label}）"


def _resolve_background_mode(
    discussion: Discussion,
) -> Literal["full", "summary", "minimal"]:
    """根据 background_mode 和参会人数解析实际模式."""
    if discussion.background_mode != "auto":
        return discussion.background_mode  # type: ignore[return-value]
    n = len(discussion.participants)
    if n <= 3:
        return "full"
    elif n <= 6:
        return "summary"
    return "minimal"


def _render_header(
    discussion: Discussion,
    topic: str,
    goal: str,
    project_info: ProjectInfo | None,
    agent_identity: "AgentIdentity | None",
) -> list[str]:
    """渲染头部：标题、议题、目标、项目类型."""
    parts = [
        f"# 讨论会：{discussion.description or discussion.name}",
        "",
        f"**议题**: {topic}",
    ]
    if goal:
        parts.append(f"**目标**: {goal}")
    if project_info and project_info.project_type != "unknown":
        parts.append(f"**项目类型**: {project_info.display_label}")
    parts.append("")

    if agent_identity:
        if agent_identity.nickname:
            parts.append(f"**Agent**: {agent_identity.nickname}")
        if agent_identity.title:
            parts.append(f"**职称**: {agent_identity.title}")
        if agent_identity.domains:
            parts.append(f"**领域**: {', '.join(agent_identity.domains)}")
        if agent_identity.model:
            parts.append(f"**Agent 模型**: {agent_identity.model}")
        if agent_identity.memory:
            parts.extend(["", "## Agent 记忆", "", agent_identity.memory])
        parts.append("")

    return parts


def _render_participants(
    participants_info: list[dict],
    engine: CrewEngine,
    initial_args: dict[str, str],
    bg_mode: Literal["full", "summary", "minimal"],
    project_info: ProjectInfo | None,
) -> list[str]:
    """渲染参会者信息 + 专业背景."""
    parts = ["---", "", "## 参会者", ""]

    for info in participants_info:
        p: DiscussionParticipant = info["participant"]
        emp: Employee | None = info["employee"]

        if emp is None:
            parts.append(f"### {p.employee}（{ROLE_LABELS[p.role]}）— 未找到")
            parts.append("")
            continue

        parts.append(f"### {_participant_title(emp, p.role)}")
        parts.append(f"**描述**: {emp.description}")
        if p.focus:
            parts.append(f"**本次关注**: {p.focus}")
        if emp.tags:
            parts.append(f"**标签**: {', '.join(emp.tags)}")
        parts.append("")

        # 根据 background_mode 注入专业背景
        if bg_mode == "full":
            rendered_body = engine.render(emp, args=dict(initial_args))
            if project_info:
                rendered_body = rendered_body.replace("{project_type}", project_info.project_type)
                rendered_body = rendered_body.replace("{framework}", project_info.framework)
                rendered_body = rendered_body.replace(
                    "{test_framework}", project_info.test_framework
                )
                rendered_body = rendered_body.replace(
                    "{package_manager}", project_info.package_manager
                )
            parts.append(f"<专业背景>\n{rendered_body}\n</专业背景>")
            parts.append("")
        elif bg_mode == "summary":
            summary_text = emp.summary or emp.description
            parts.append(f"<专业背景>\n{summary_text}\n</专业背景>")
            parts.append("")
        # minimal: 不注入专业背景，description + focus 已足够

    return parts


def _render_rules(discussion: Discussion) -> list[str]:
    """渲染讨论规则."""
    parts = ["---", "", "## 讨论规则", ""]
    rules = discussion.rules if discussion.rules is not None else _DEFAULT_RULES
    for i, rule in enumerate(rules, 1):
        parts.append(f"{i}. {rule}")
    parts.append("")
    return parts


def _render_rounds(discussion: Discussion) -> list[str]:
    """渲染轮次安排."""
    parts = ["## 轮次安排", ""]

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
            # 根据 interaction 类型追加互动规则
            if rnd.interaction != "free" and rnd.interaction in _INTERACTION_RULES:
                parts.append("")
                parts.append(_INTERACTION_RULES[rnd.interaction])
            parts.append("")

    return parts


def _render_output(output_format: str) -> list[str]:
    """渲染输出格式模板."""
    return ["---", "", "## 输出格式", "", _OUTPUT_TEMPLATES[output_format]]


# ── 主渲染入口 ──


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
    agent_identity: "AgentIdentity | None" = None
    if agent_id is not None:
        try:
            from crew.id_client import fetch_agent_identity

            agent_identity = fetch_agent_identity(agent_id)
        except ImportError:
            agent_identity = None

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

    bg_mode = _resolve_background_mode(discussion)

    # 组装各段
    parts: list[str] = []
    parts.extend(
        _render_header(discussion, topic, goal, project_info, agent_identity)
    )
    parts.extend(
        _render_participants(participants_info, engine, initial_args, bg_mode, project_info)
    )
    parts.extend(_render_rules(discussion))
    parts.extend(_render_rounds(discussion))
    parts.extend(_render_output(discussion.output_format))

    return "\n".join(parts)


# ── 发现 ──

DISCUSSIONS_DIR_NAME = "discussions"


def discover_discussions(project_dir: Path | None = None) -> dict[str, Path]:
    """发现所有可用讨论会.

    搜索顺序（低优先级 → 高优先级）：
    1. 内置（src/crew/employees/discussions/）
    2. 全局（~/.knowlyr/crew/discussions/）
    3. 项目（.crew/discussions/）— 同名覆盖内置和全局
    """
    discussions: dict[str, Path] = {}

    # 内置讨论会
    builtin_dir = Path(__file__).parent / "employees" / DISCUSSIONS_DIR_NAME
    if builtin_dir.is_dir():
        for f in sorted(builtin_dir.glob("*.yaml")):
            discussions[f.stem] = f

    # 全局讨论会
    global_dir = Path.home() / ".knowlyr" / "crew" / DISCUSSIONS_DIR_NAME
    if global_dir.is_dir():
        for f in sorted(global_dir.glob("*.yaml")):
            discussions[f.stem] = f

    # 项目讨论会（覆盖同名内置和全局）
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
（每轮每人的发言，格式：**【姓名（角色名）】**: 发言内容）

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
**【姓名（角色名）】**: 完整发言内容...
（保留所有讨论细节，不做压缩）""",
    "summary": """\
请按以下格式输出讨论总结：

# 讨论会总结

## 议题
（一句话概括）

## 主要观点
- **姓名（角色A）**: 核心观点...
- **姓名（角色B）**: 核心观点...

## 共识
1. ...

## 后续行动
1. ...""",
}
