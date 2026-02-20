"""讨论会引擎 — 多员工多轮讨论."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml
from pydantic import BaseModel, Field

from crew.context_detector import ProjectInfo, detect_project
from crew.discovery import discover_employees
from crew.engine import CrewEngine
from crew.models import DiscussionPlan, Employee, EmployeeOutput, ParticipantPrompt, RoundPlan
from crew.paths import get_global_discussions_dir

if TYPE_CHECKING:
    from crew.id_client import AgentIdentity

logger = logging.getLogger(__name__)

_TOPIC_FILENAME_MAX_LENGTH = 60


# ── 数据模型 ──


class DiscussionParticipant(BaseModel):
    """讨论会参与者."""

    employee: str = Field(description="员工名称")
    role: Literal["moderator", "speaker", "recorder"] = Field(
        default="speaker", description="会议角色"
    )
    focus: str = Field(default="", description="本次讨论的关注重点")
    stance: str = Field(default="", description="预设立场/倾向（如 '偏保守'、'偏激进'）")
    must_challenge: list[str] = Field(
        default_factory=list,
        description="本参会者必须质疑的其他参会者列表（employee name）",
    )
    max_agree_ratio: float = Field(
        default=1.0,
        description="最大同意比例（0.0~1.0），低于 1.0 时强制产生分歧",
    )
    execution_role: Literal["executor", "reviewer", "monitor", "idle"] = Field(
        default="idle",
        description="讨论后执行阶段的角色",
    )


class DiscussionRound(BaseModel):
    """讨论轮次配置."""

    name: str = Field(default="", description="轮次名称")
    instruction: str = Field(default="", description="该轮特殊指令")
    interaction: Literal[
        "free",
        "round-robin",
        "challenge",
        "response",
        "brainstorm",
        "vote",
        "debate",
        "cross-examine",
        "steelman-then-attack",
    ] = Field(default="free", description="互动模式")
    min_disagreements: int = Field(
        default=0,
        description="本轮最少产生的分歧数（0=不限制）",
    )
    max_words_per_turn: int = Field(
        default=0,
        description="每人每轮最大字数（0=不限制），防止独白",
    )
    require_direct_reply: bool = Field(
        default=False,
        description="是否要求每人必须引用并回应至少一位他人的具体观点",
    )


class Discussion(BaseModel):
    """讨论会定义."""

    name: str = Field(description="讨论会名称")
    description: str = Field(default="", description="描述")
    topic: str = Field(description="议题（支持 $variable）")
    goal: str = Field(default="", description="讨论目标")
    participants: list[DiscussionParticipant] = Field(description="参与者列表")
    rounds: int | list[DiscussionRound] = Field(default=3, description="讨论轮次")
    round_template: str | None = Field(default=None, description="轮次模板名称（优先于 rounds）")
    mode: Literal["auto", "discussion", "meeting"] = Field(
        default="auto", description="auto=按参与者数自动判断"
    )
    rules: list[str] | None = Field(default=None, description="自定义讨论规则（None=默认）")
    background_mode: Literal["full", "summary", "minimal", "auto"] = Field(
        default="auto", description="专业背景注入模式"
    )
    output_format: Literal["decision", "transcript", "summary"] = Field(
        default="decision", description="输出格式"
    )
    output: EmployeeOutput | None = Field(
        default=None, description="自动保存配置（filename/dir），为 None 时不自动保存"
    )
    tension_seeds: list[str] = Field(
        default_factory=list,
        description="预设的争议点列表，注入到讨论中强制触发分歧",
    )
    action_output: bool = Field(
        default=False,
        description="讨论结束后是否自动生成 pipeline 可执行的行动计划",
    )
    post_pipeline: str = Field(
        default="",
        description="讨论结束后自动触发的 pipeline 名称",
    )

    @property
    def effective_mode(self) -> str:
        """根据参与者数自动判断模式."""
        if self.mode != "auto":
            return self.mode
        return "meeting" if len(self.participants) == 1 else "discussion"


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
    "brainstorm": (
        "**本轮要求**：发散思维阶段。每位参会者至少提出 2 个新想法或方案，"
        "不评判他人的想法，鼓励天马行空。"
        "发言格式：**【创意】姓名**: 想法内容"
    ),
    "vote": (
        "**本轮要求**：投票决策。每位参会者对前面提出的方案/选项进行投票，"
        "必须给出明确的支持/反对/弃权，并附简要理由。"
        "发言格式：**【投票】姓名 → 方案X**: 支持/反对/弃权 — 理由"
    ),
    "debate": (
        "**本轮要求**：结构化辩论。参会者分为正方和反方，"
        "正方先陈述论点，反方逐一反驳。每方发言需引用具体事实和数据。"
        "发言格式：**【正方/反方】姓名**: 论点内容"
    ),
    "cross-examine": (
        "**本轮要求**：交叉盘问模式。每位参会者选择一位他人，"
        "针对其上一轮的关键论点提出 3 个具体问题。问题必须是：\n"
        "1）基于事实的挑战（'你说X，但数据显示Y'）；\n"
        "2）逻辑推演（'如果按你的方案，那么Z怎么办？'）；\n"
        "3）替代方案（'为什么不用W方案？'）。\n"
        "被盘问者必须逐条回答，不可回避。\n"
        "发言格式：**【盘问】XX → YY**: 问题内容"
    ),
    "steelman-then-attack": (
        "**本轮要求**：先强化再攻击。每位参会者必须：\n"
        "1）先用 2-3 句话尽可能强化另一位参会者的论点（steelman）；\n"
        "2）然后再指出该论点即使在最强形式下仍然存在的弱点。\n"
        "格式：**【强化】XX → YY 的观点**: 最强版本...\n"
        "**【弱点】XX → YY 的观点**: 即使如此，仍有..."
    ),
}

_1V1_RULES = [
    "这是一场一对一的专业咨询/讨论",
    "员工以自己的专业视角提供意见和建议",
    "对话应保持互动性，主动询问用户的想法和补充信息",
    "在讨论结束时，总结要点和行动建议",
]

_ROUND_TEMPLATES: dict[str, list[DiscussionRound]] = {
    "standard": [
        DiscussionRound(name="开场", instruction="主持人介绍议题，各方给出初步观点"),
        DiscussionRound(name="深入讨论", instruction="回应前轮观点，深入分歧点"),
        DiscussionRound(name="总结决议", instruction="达成共识，形成行动项"),
    ],
    "brainstorm-to-decision": [
        DiscussionRound(name="发散", instruction="自由提出创意和方案", interaction="brainstorm"),
        DiscussionRound(
            name="筛选讨论", instruction="评估各方案的可行性", interaction="round-robin"
        ),
        DiscussionRound(name="投票决策", instruction="对最终方案投票", interaction="vote"),
    ],
    "adversarial": [
        DiscussionRound(
            name="各抒己见", instruction="每位参会者表明立场", interaction="round-robin"
        ),
        DiscussionRound(name="质疑挑战", instruction="互相挑战观点", interaction="challenge"),
        DiscussionRound(name="回应辩护", instruction="回应质疑", interaction="response"),
        DiscussionRound(name="共识决议", instruction="达成共识，形成决议"),
    ],
    "deep-adversarial": [
        DiscussionRound(
            name="各抒己见",
            instruction="每位参会者表明立场，给出核心论点和关键依据",
            interaction="round-robin",
            max_words_per_turn=300,
        ),
        DiscussionRound(
            name="交叉盘问",
            instruction="针对他人观点提出具体问题和挑战",
            interaction="cross-examine",
            require_direct_reply=True,
            min_disagreements=2,
        ),
        DiscussionRound(
            name="强化后攻击",
            instruction="先尽可能强化对方论点，再找弱点",
            interaction="steelman-then-attack",
            require_direct_reply=True,
        ),
        DiscussionRound(
            name="收敛方案",
            instruction="基于前面的分歧，提出折中或创新的综合方案",
            interaction="free",
            max_words_per_turn=400,
        ),
        DiscussionRound(
            name="决议与分工",
            instruction="形成决议、分配任务、标记未解决分歧",
            interaction="vote",
        ),
    ],
    "discuss-then-execute": [
        DiscussionRound(
            name="问题定义",
            instruction="主持人定义问题，各方从自身角度补充问题边界",
            interaction="round-robin",
            max_words_per_turn=200,
        ),
        DiscussionRound(
            name="方案提出",
            instruction="每人提出至少一个完整方案，包含具体实现步骤",
            interaction="brainstorm",
        ),
        DiscussionRound(
            name="方案对质",
            instruction="对每个方案进行可行性挑战",
            interaction="challenge",
            require_direct_reply=True,
            min_disagreements=1,
        ),
        DiscussionRound(
            name="方案选择",
            instruction="投票选择最终方案，标记需要妥协的部分",
            interaction="vote",
        ),
        DiscussionRound(
            name="任务拆解",
            instruction="将方案拆解为具体任务，指定负责角色和依赖关系",
            interaction="round-robin",
        ),
    ],
}


# ── 轮次解析 ──


def _resolve_rounds(discussion: Discussion) -> int | list[DiscussionRound]:
    """解析轮次：round_template > rounds list > rounds int."""
    if discussion.round_template and discussion.round_template in _ROUND_TEMPLATES:
        return _ROUND_TEMPLATES[discussion.round_template]
    return discussion.rounds


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


def validate_discussion(discussion: Discussion, project_dir: Path | None = None) -> list[str]:
    """校验讨论会定义，返回错误列表."""
    errors: list[str] = []

    if len(discussion.participants) < 1:
        errors.append("讨论会至少需要 1 个参与者")
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
    agent_identity: AgentIdentity | None,
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
    rounds = _resolve_rounds(discussion)

    if isinstance(rounds, int):
        for i in range(1, rounds + 1):
            if i == 1:
                parts.append(f"### 第 {i} 轮：开场")
                parts.append("主持人介绍议题，每位参会者从自身专业角度给出初步观点。")
            elif i == rounds:
                parts.append(f"### 第 {i} 轮：总结与决议")
                parts.append("主持人总结各方观点，达成共识，形成明确的决议和行动项。")
            else:
                parts.append(f"### 第 {i} 轮：深入讨论")
                parts.append("回应前轮观点，深入探讨分歧点，提出具体方案。")
            parts.append("")
    else:
        for i, rnd in enumerate(rounds, 1):
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


def _render_auto_save(
    output: EmployeeOutput, topic: str, initial_args: dict[str, str]
) -> list[str]:
    """渲染自动保存指令."""
    from datetime import datetime

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")

    # 替换 filename 模板中的变量
    filename = output.filename
    filename = filename.replace("{date}", date_str)
    filename = filename.replace("{datetime}", now.strftime("%Y-%m-%d_%H%M%S"))

    # 替换 $variable 参数（优先用原始参数值，更短更准确）
    for k, v in initial_args.items():
        filename = filename.replace(f"${k}", v)

    # $topic 兜底：用完整 topic（截断后）
    if "$topic" in filename:
        safe_topic = topic.replace(" ", "-").replace("/", "-").replace("\\", "-")
        if len(safe_topic) > _TOPIC_FILENAME_MAX_LENGTH:
            safe_topic = safe_topic[:_TOPIC_FILENAME_MAX_LENGTH]
        filename = filename.replace("$topic", safe_topic)

    # 展开 ~ 为绝对路径
    save_dir = output.dir
    if save_dir.startswith("~"):
        save_dir = str(Path.home() / save_dir[2:])

    parts = [
        "",
        "---",
        "",
        "## 自动保存",
        "",
        "**重要**：讨论结束后，你必须将完整的会议纪要保存为 Markdown 文件。",
        "",
        f"- 保存目录：`{save_dir}`（如目录不存在则自动创建）",
        f"- 文件名：`{filename}`",
        f"- 完整路径：`{save_dir}/{filename}`",
        "",
        "保存完成后，在回复中告知用户文件路径。",
    ]
    return parts


def _render_1v1_meeting(
    discussion: Discussion,
    topic: str,
    goal: str,
    participants_info: list[dict],
    engine: CrewEngine,
    initial_args: dict[str, str],
    project_info: ProjectInfo | None,
    agent_identity: AgentIdentity | None,
) -> str:
    """渲染 1v1 会议 prompt — 会话式而非多轮结构."""
    info = participants_info[0]
    p: DiscussionParticipant = info["participant"]
    emp: Employee | None = info["employee"]

    parts: list[str] = [
        f"# 会议：{discussion.description or topic}",
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

    # 员工信息
    parts.extend(["---", "", "## 参会者", ""])
    if emp is None:
        parts.append(f"### {p.employee} — 未找到")
        parts.append("")
    else:
        title = _participant_title(emp, p.role)
        parts.append(f"### {title}")
        parts.append(f"**描述**: {emp.description}")
        if p.focus:
            parts.append(f"**本次关注**: {p.focus}")
        parts.append("")
        # 1v1 始终注入完整背景
        rendered_body = engine.render(emp, args=dict(initial_args))
        if project_info:
            rendered_body = rendered_body.replace("{project_type}", project_info.project_type)
            rendered_body = rendered_body.replace("{framework}", project_info.framework)
            rendered_body = rendered_body.replace("{test_framework}", project_info.test_framework)
            rendered_body = rendered_body.replace("{package_manager}", project_info.package_manager)
        parts.append(f"<专业背景>\n{rendered_body}\n</专业背景>")
        parts.append("")

    # 会议规则
    rules = discussion.rules if discussion.rules is not None else _1V1_RULES
    parts.extend(["---", "", "## 会议规则", ""])
    for i, rule in enumerate(rules, 1):
        parts.append(f"{i}. {rule}")
    parts.append("")

    # 输出格式
    parts.extend(["---", "", "## 输出格式", "", _1V1_OUTPUT_TEMPLATES[discussion.output_format]])

    # 自动保存
    if discussion.output is not None and discussion.output.filename:
        parts.extend(_render_auto_save(discussion.output, topic, initial_args))

    return "\n".join(parts)


def _log_meeting(
    discussion: Discussion,
    prompt: str,
    initial_args: dict[str, str],
    project_dir: Path | None = None,
) -> None:
    """尝试记录会议日志（静默失败）."""
    try:
        from crew.meeting_log import MeetingLogger

        ml = MeetingLogger(project_dir=project_dir)
        ml.save(discussion, prompt, initial_args)
    except Exception as e:
        logger.debug("记录会议日志失败: %s", e)


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
    engine = CrewEngine(project_dir=project_dir)

    project_info = detect_project(project_dir) if smart_context else None

    # 获取 agent 身份（可选）
    agent_identity: AgentIdentity | None = None
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

    # 1v1 会议走单独渲染路径
    if discussion.effective_mode == "meeting":
        prompt = _render_1v1_meeting(
            discussion,
            topic,
            goal,
            participants_info,
            engine,
            initial_args,
            project_info,
            agent_identity,
        )
        _log_meeting(discussion, prompt, initial_args, project_dir=project_dir)
        return prompt

    bg_mode = _resolve_background_mode(discussion)

    # 组装各段
    parts: list[str] = []
    parts.extend(_render_header(discussion, topic, goal, project_info, agent_identity))
    parts.extend(
        _render_participants(participants_info, engine, initial_args, bg_mode, project_info)
    )
    parts.extend(_render_rules(discussion))
    parts.extend(_render_rounds(discussion))
    parts.extend(_render_output(discussion.output_format))

    # 自动保存指令
    if discussion.output is not None and discussion.output.filename:
        parts.extend(_render_auto_save(discussion.output, topic, initial_args))

    prompt = "\n".join(parts)
    _log_meeting(discussion, prompt, initial_args, project_dir=project_dir)
    return prompt


# ── 编排式讨论（独立 session）──


def _render_participant_prompt(
    emp: Employee,
    participant: DiscussionParticipant,
    discussion: Discussion,
    topic: str,
    goal: str,
    round_info: DiscussionRound | dict,
    round_number: int,
    total_rounds: int,
    other_participants: list[dict],
    engine: CrewEngine,
    initial_args: dict[str, str],
    project_info: ProjectInfo | None,
    is_first_round: bool,
    project_dir: Path | None = None,
) -> str:
    """为单个参会者生成独立的 prompt."""
    role_label = ROLE_LABELS.get(participant.role, participant.role)

    # 轮次信息
    if isinstance(round_info, DiscussionRound):
        round_name = round_info.name or f"第 {round_number} 轮"
        round_instruction = round_info.instruction
        interaction = round_info.interaction
    else:
        round_name = f"第 {round_number} 轮"
        round_instruction = round_info.get("instruction", "")
        interaction = round_info.get("interaction", "free")

    parts: list[str] = [
        f"# 讨论会：{discussion.description or discussion.name}",
        "",
        f"**议题**: {topic}",
    ]
    if goal:
        parts.append(f"**目标**: {goal}")
    if project_info and project_info.project_type != "unknown":
        parts.append(f"**项目类型**: {project_info.display_label}")
    parts.append(f"**当前轮次**: {round_name}（第 {round_number}/{total_rounds} 轮）")
    parts.append("")

    # 自己的身份和背景
    parts.extend(["---", "", "## 你的身份", ""])
    display = emp.character_name or emp.effective_display_name
    parts.append(f"你是 **{display}**（{role_label}），{emp.description}。")
    if participant.focus:
        parts.append(f"本次讨论你重点关注：**{participant.focus}**")
    parts.append("")

    rendered_body = engine.render(emp, args=dict(initial_args))
    if project_info:
        rendered_body = rendered_body.replace("{project_type}", project_info.project_type)
        rendered_body = rendered_body.replace("{framework}", project_info.framework)
        rendered_body = rendered_body.replace("{test_framework}", project_info.test_framework)
        rendered_body = rendered_body.replace("{package_manager}", project_info.package_manager)
    parts.append(f"<专业背景>\n{rendered_body}\n</专业背景>")
    parts.append("")

    # 其他参会者（摘要）
    if other_participants:
        parts.extend(["---", "", "## 其他参会者", ""])
        for other in other_participants:
            other_emp: Employee | None = other["employee"]
            other_p: DiscussionParticipant = other["participant"]
            other_role = ROLE_LABELS.get(other_p.role, other_p.role)
            if other_emp:
                other_display = other_emp.character_name or other_emp.effective_display_name
                summary = other_emp.summary or other_emp.description
                parts.append(f"- **{other_display}**（{other_role}）: {summary}")
            else:
                parts.append(f"- {other_p.employee}（{other_role}）: 未找到")
        parts.append("")

    # 讨论规则
    rules = discussion.rules if discussion.rules is not None else _DEFAULT_RULES
    parts.extend(["---", "", "## 讨论规则", ""])
    for i, rule in enumerate(rules, 1):
        parts.append(f"{i}. {rule}")
    parts.append("")

    # 本轮要求
    parts.extend(["## 本轮要求", ""])
    if round_instruction:
        parts.append(round_instruction)
    if interaction != "free" and interaction in _INTERACTION_RULES:
        parts.append("")
        parts.append(_INTERACTION_RULES[interaction])
    parts.append("")

    # 历史经验（从持久化记忆注入）
    try:
        from crew.memory import MemoryStore

        memory_store = MemoryStore(project_dir=project_dir)
        memory_text = memory_store.format_for_prompt(emp.name, limit=5)
        if memory_text:
            parts.extend(["---", "", "## 你的历史经验", "", memory_text, ""])
    except Exception as e:
        logger.debug("加载员工 %s 历史经验失败: %s", emp.name, e)

    # 预研发现（如果有 research round，第一轮注入）
    if is_first_round and (emp.research_instructions or emp.tools):
        parts.extend(["---", "", "## 你的预研发现", ""])
        parts.append("{research_findings}")
        parts.append("")
        parts.append("**重要**：你的发言必须引用上述预研发现中的数据，不得编造。")
        parts.append("")

    # 前序讨论记录（占位符）
    parts.extend(["---", "", "## 前序讨论记录", ""])
    if is_first_round:
        parts.append("（这是第一轮讨论，尚无前序记录。请直接给出你的观点。）")
    else:
        parts.append("{previous_rounds}")
    parts.append("")

    # 输出要求
    parts.extend(["---", "", "## 输出要求", ""])
    parts.append(
        f"以 **【{display}（{role_label}）】** 的身份发言。"
        "发言应基于你的专业背景，针对议题给出具体、有依据的观点。"
    )
    if not is_first_round:
        parts.append("你必须回应前序讨论中其他参会者的观点，不能忽略他们的发言。")
    if interaction == "challenge":
        parts.append("你必须对至少一位其他参会者的观点提出质疑或挑战。")
    elif interaction == "response":
        parts.append("你必须回应上一轮中对你的质疑。如果没有人直接质疑你，也要回应他人的论点。")
    parts.append("")

    # ── 对抗性约束 ──

    if participant.stance:
        parts.extend(
            [
                "## 你的立场",
                "",
                f"**预设立场**: {participant.stance}",
                "你的发言必须体现这一立场倾向，即使其他参会者不同意。",
                "",
            ]
        )

    if participant.must_challenge:
        names = ", ".join(participant.must_challenge)
        parts.extend(
            [
                "## 必须质疑",
                "",
                f"你必须对 **{names}** 的观点提出至少一个实质性挑战。",
                "不可以只说「我同意」或「补充一下」——你必须找到不同意的地方。",
                "",
            ]
        )

    if participant.max_agree_ratio < 1.0:
        pct = int(participant.max_agree_ratio * 100)
        parts.extend(
            [
                "## 分歧配额",
                "",
                f"对于他人提出的观点，你最多只能完全同意 {pct}%。",
                "其余必须提出替代方案、补充条件或反对意见。",
                "",
            ]
        )

    # 争议种子注入（首轮）
    if discussion.tension_seeds and is_first_round:
        parts.extend(["## 需要讨论的争议点", ""])
        for i, seed in enumerate(discussion.tension_seeds, 1):
            parts.append(f"{i}. {seed}")
        parts.extend(
            [
                "",
                "你必须对上述至少一个争议点表明明确立场。",
                "",
            ]
        )

    # 反独白约束
    if isinstance(round_info, DiscussionRound) and round_info.max_words_per_turn > 0:
        parts.extend(
            [
                "## 字数限制",
                "",
                f"本轮发言不超过 **{round_info.max_words_per_turn}** 字。精练表达，不要堆砌长段落。",
                "",
            ]
        )

    if isinstance(round_info, DiscussionRound) and round_info.require_direct_reply:
        parts.extend(
            [
                "## 直接回应要求",
                "",
                "你必须引用（用 > 引用标记）至少一位参会者上一轮发言中的具体句子，",
                "然后表明同意、反对或修改，不可笼统评论。",
                "",
            ]
        )

    if isinstance(round_info, DiscussionRound) and round_info.min_disagreements > 0:
        parts.extend(
            [
                "## 最低分歧要求",
                "",
                f"本轮你必须至少提出 **{round_info.min_disagreements}** 个实质性分歧点。",
                "",
            ]
        )

    # ── 去重约束（非首轮）──

    if not is_first_round:
        parts.extend(
            [
                "## 发言约束",
                "",
                "- **禁止重复**: 不要重述你在前几轮已经说过的观点。如果立场没变，用一句话确认即可。",
                "- **增量贡献**: 每次发言必须包含至少一个新信息、新论点或新方案。",
                "- **引用回应**: 回应他人时，必须引用其具体的原文片段（用 > 引用标记）。",
                "- **标记状态**: 对每个讨论点标注 [已共识] [有分歧] [待决]。",
                "",
            ]
        )

    return "\n".join(parts)


def _render_synthesis_prompt(
    discussion: Discussion,
    topic: str,
    goal: str,
    participants_info: list[dict],
) -> str:
    """生成最终汇总 prompt."""
    parts: list[str] = [
        f"# 讨论会汇总：{discussion.description or discussion.name}",
        "",
        f"**议题**: {topic}",
    ]
    if goal:
        parts.append(f"**目标**: {goal}")
    parts.append("")

    parts.extend(["## 参会者", ""])
    for info in participants_info:
        p: DiscussionParticipant = info["participant"]
        emp: Employee | None = info["employee"]
        role_label = ROLE_LABELS.get(p.role, p.role)
        if emp:
            display = emp.character_name or emp.effective_display_name
            parts.append(f"- **{display}**（{role_label}）")
        else:
            parts.append(f"- {p.employee}（{role_label}）")
    parts.append("")

    parts.extend(
        [
            "## 所有讨论记录",
            "",
            "{all_rounds}",
            "",
            "---",
            "",
        ]
    )

    parts.append(_OUTPUT_TEMPLATES[discussion.output_format])

    # ActionPlan 输出
    if discussion.action_output:
        parts.extend(
            [
                "",
                "---",
                "",
                "## 行动计划（结构化 JSON）",
                "",
                "在会议纪要之后，你还 **必须** 输出一份结构化的行动计划（JSON 格式）：",
                "",
                "```json",
                "{",
                '  "decisions": ["决策1", "决策2"],',
                '  "unresolved": ["未解决的分歧1"],',
                '  "actions": [',
                "    {",
                '      "id": "A1",',
                '      "description": "具体任务描述",',
                '      "assignee_role": "executor",',
                '      "assignee_employee": "员工名称（可选）",',
                '      "depends_on": [],',
                '      "priority": "P0",',
                '      "verification": "怎样算完成",',
                '      "phase": "implement"',
                "    }",
                "  ],",
                '  "review_criteria": ["验收标准1", "验收标准2"]',
                "}",
                "```",
                "",
                "**重要**：",
                "- actions 中每个任务必须可独立执行，不可笼统",
                "- depends_on 用于标记任务间依赖（如类型变更必须先于组件开发）",
                "- phase 用于区分 research / implement / review / deploy 阶段",
                "- 每个 action 必须指定 verification（验证方式）",
                "",
            ]
        )

    # 自动保存
    if discussion.output is not None and discussion.output.filename:
        parts.extend(_render_auto_save(discussion.output, topic, {}))

    return "\n".join(parts)


def render_discussion_plan(
    discussion: Discussion,
    initial_args: dict[str, str] | None = None,
    project_dir: Path | None = None,
    agent_id: int | None = None,
    smart_context: bool = True,
) -> DiscussionPlan:
    """渲染编排式讨论计划 — 每个参会者独立 prompt.

    与 render_discussion() 的区别:
    - render_discussion() 生成单个 prompt，所有角色在同一次推理中
    - render_discussion_plan() 生成结构化计划，每个角色独立推理

    Args:
        discussion: 讨论会定义
        initial_args: 参数替换
        project_dir: 项目目录
        agent_id: Agent ID
        smart_context: 自动检测项目类型

    Returns:
        DiscussionPlan，包含多轮多人独立 prompt
    """
    initial_args = initial_args or {}
    result = discover_employees(project_dir=project_dir)
    engine = CrewEngine(project_dir=project_dir)

    project_info = detect_project(project_dir) if smart_context else None

    # 变量替换
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

    # 解析轮次
    rounds = _resolve_rounds(discussion)
    if isinstance(rounds, int):
        round_list: list[DiscussionRound | dict] = []
        for i in range(1, rounds + 1):
            if i == 1:
                round_list.append(
                    {
                        "name": "开场",
                        "instruction": "每位参会者从自身专业角度给出初步观点。",
                        "interaction": "round-robin",
                    }
                )
            elif i == rounds:
                round_list.append(
                    {
                        "name": "总结与决议",
                        "instruction": "基于前几轮讨论，达成共识，形成明确的决议和行动项。",
                        "interaction": "free",
                    }
                )
            else:
                round_list.append(
                    {
                        "name": "深入讨论",
                        "instruction": "回应前轮观点，深入探讨分歧点，提出具体方案。",
                        "interaction": "free",
                    }
                )
    else:
        round_list = list(rounds)

    # ── Research round (round 0) ──
    # 当任意参会者有 research_instructions 或 tools 时，自动插入预研轮
    has_research = any(
        info["employee"] is not None
        and (info["employee"].research_instructions or info["employee"].tools)
        for info in participants_info
    )

    plan_rounds: list[RoundPlan] = []

    if has_research:
        research_prompts: list[ParticipantPrompt] = []
        for info in participants_info:
            emp: Employee | None = info["employee"]
            p: DiscussionParticipant = info["participant"]
            if emp is None:
                continue

            role_label = ROLE_LABELS.get(p.role, p.role)
            display = emp.character_name or emp.effective_display_name
            tools_str = ", ".join(emp.tools) if emp.tools else "无特定工具"
            context_str = ", ".join(emp.context) if emp.context else "无指定上下文"

            r_parts: list[str] = [
                f"# 预研：{discussion.description or discussion.name}",
                "",
                f"**议题**: {topic}",
                f"**你的身份**: {display}（{role_label}），{emp.description}",
                f"**可用工具**: {tools_str}",
                f"**预读上下文**: {context_str}",
                "",
                "---",
                "",
                "## 任务",
                "",
                "你即将参加一场讨论会。在发言之前，请先使用你的工具收集真实数据。",
                "**禁止编造任何数据或估算值**——所有信息必须来自工具调用的实际结果。",
                "",
            ]

            if emp.research_instructions:
                r_parts.extend(
                    [
                        "## 预研指令",
                        "",
                        emp.research_instructions,
                        "",
                    ]
                )
            else:
                # 根据工具自动生成通用预研指令
                r_parts.extend(
                    [
                        "## 预研指令",
                        "",
                        f"请根据议题「{topic}」，使用你的工具收集相关信息。",
                    ]
                )
                if emp.context:
                    r_parts.append(f"建议先读取以下文件：{', '.join(emp.context)}")
                r_parts.append("")

            r_parts.extend(
                [
                    "## 输出格式",
                    "",
                    "以 JSON 格式输出你的发现：",
                    "",
                    "```json",
                    "{",
                    '  "findings": [',
                    '    {"id": "F1", "content": "发现内容", "source": "工具/文件来源"},',
                    '    {"id": "F2", "content": "发现内容", "source": "工具/文件来源"}',
                    "  ]",
                    "}",
                    "```",
                    "",
                    "这些发现将注入你在讨论中的 prompt，作为你发言的依据。",
                ]
            )

            research_prompts.append(
                ParticipantPrompt(
                    employee_name=emp.name,
                    character_name=emp.character_name,
                    role=p.role,
                    prompt="\n".join(r_parts),
                )
            )

        if research_prompts:
            plan_rounds.append(
                RoundPlan(
                    round_number=0,
                    name="预研",
                    instruction="各参会者使用工具收集真实数据",
                    interaction="free",
                    participant_prompts=research_prompts,
                )
            )

    total_rounds = len(round_list)

    for idx, round_info in enumerate(round_list):
        round_number = idx + 1
        is_first_round = idx == 0

        # 轮次元数据
        if isinstance(round_info, DiscussionRound):
            rnd_name = round_info.name or f"第 {round_number} 轮"
            rnd_instruction = round_info.instruction
            rnd_interaction = round_info.interaction
        else:
            rnd_name = round_info.get("name", f"第 {round_number} 轮")
            rnd_instruction = round_info.get("instruction", "")
            rnd_interaction = round_info.get("interaction", "free")

        participant_prompts: list[ParticipantPrompt] = []

        for i, info in enumerate(participants_info):
            p: DiscussionParticipant = info["participant"]
            emp: Employee | None = info["employee"]

            if emp is None:
                participant_prompts.append(
                    ParticipantPrompt(
                        employee_name=p.employee,
                        role=p.role,
                        prompt=f"# 错误\n\n未找到员工: {p.employee}",
                    )
                )
                continue

            # 其他参会者
            others = [x for j, x in enumerate(participants_info) if j != i]

            prompt_text = _render_participant_prompt(
                emp=emp,
                participant=p,
                discussion=discussion,
                topic=topic,
                goal=goal,
                round_info=round_info,
                round_number=round_number,
                total_rounds=total_rounds,
                other_participants=others,
                engine=engine,
                initial_args=initial_args,
                project_info=project_info,
                is_first_round=is_first_round,
                project_dir=project_dir,
            )

            participant_prompts.append(
                ParticipantPrompt(
                    employee_name=emp.name,
                    character_name=emp.character_name,
                    role=p.role,
                    prompt=prompt_text,
                )
            )

        plan_rounds.append(
            RoundPlan(
                round_number=round_number,
                name=rnd_name,
                instruction=rnd_instruction,
                interaction=rnd_interaction,
                participant_prompts=participant_prompts,
            )
        )

    # 汇总 prompt
    synthesis = _render_synthesis_prompt(discussion, topic, goal, participants_info)

    plan = DiscussionPlan(
        discussion_name=discussion.name,
        topic=topic,
        goal=goal,
        rounds=plan_rounds,
        synthesis_prompt=synthesis,
    )

    # 记录会议
    _log_meeting(
        discussion,
        f"[orchestrated plan] {len(plan_rounds)} rounds",
        initial_args,
        project_dir=project_dir,
    )

    return plan


# ── 发现 ──

DISCUSSIONS_DIR_NAME = "discussions"


def discover_discussions(project_dir: Path | None = None) -> dict[str, Path]:
    """发现所有可用讨论会.

    搜索顺序（低优先级 → 高优先级）：
    1. 内置（src/crew/employees/discussions/）
    2. 全局（默认 .crew/global/discussions/，可用 KNOWLYR_CREW_GLOBAL_DIR 覆盖）
    3. 项目（.crew/discussions/）— 同名覆盖内置和全局
    """
    discussions: dict[str, Path] = {}

    # 内置讨论会
    builtin_dir = Path(__file__).parent / "employees" / DISCUSSIONS_DIR_NAME
    if builtin_dir.is_dir():
        for f in sorted(builtin_dir.glob("*.yaml")):
            discussions[f.stem] = f

    # 全局讨论会
    global_dir = get_global_discussions_dir(project_dir=project_dir)
    if global_dir.is_dir():
        for f in sorted(global_dir.glob("*.yaml")):
            discussions[f.stem] = f

    # 项目讨论会（覆盖同名内置和全局）
    from crew.paths import resolve_project_dir

    root = resolve_project_dir(project_dir)
    project_dir_path = root / ".crew" / DISCUSSIONS_DIR_NAME
    if project_dir_path.is_dir():
        for f in sorted(project_dir_path.glob("*.yaml")):
            discussions[f.stem] = f

    return discussions


# ── 即席讨论 ──


def create_adhoc_discussion(
    employees: list[str],
    topic: str,
    goal: str = "",
    rounds: int = 2,
    output_format: str = "summary",
    round_template: str | None = None,
) -> Discussion:
    """创建即席讨论会（无需 YAML）.

    Args:
        employees: 员工名称列表（1+ 个）
        topic: 议题
        goal: 目标
        rounds: 轮次数（默认 2）
        output_format: 输出格式（默认 summary）
        round_template: 轮次模板名称
    """
    participants = []
    for i, emp_name in enumerate(employees):
        if i == 0 and len(employees) > 1:
            role = "moderator"
        else:
            role = "speaker"
        participants.append(DiscussionParticipant(employee=emp_name, role=role))

    name = f"adhoc-{'-'.join(employees[:3])}"

    return Discussion(
        name=name,
        description=f"即席讨论：{topic[:50]}",
        topic=topic,
        goal=goal,
        participants=participants,
        rounds=rounds,
        output_format=output_format,
        round_template=round_template,
    )


# ── 输出格式模板 ──

_1V1_OUTPUT_TEMPLATES = {
    "decision": """\
请按以下格式输出会议结果：

# 会议纪要

## 议题
（一句话概括）

## 讨论要点
1. ...
2. ...

## 建议与决议
1. ...

## 行动项
| 序号 | 事项 | 优先级 |
|------|------|--------|
| 1 | ... | P0-P3 |""",
    "transcript": """\
请按以下格式输出完整的会议记录：

# 会议完整记录

## 议题
（一句话概括）

## 对话过程
（保留完整的对话内容，不做压缩）""",
    "summary": """\
请按以下格式输出会议总结：

# 会议总结

## 议题
（一句话概括）

## 核心观点
- ...

## 后续行动
1. ...""",
}

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
