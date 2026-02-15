"""数据模型 — Employee / WorkLog / DiscoveryResult / Pipeline / AgentExecution."""

from dataclasses import dataclass, field as dc_field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

# ── Crew tools <-> Claude Code Skills allowed-tools 映射 ──

TOOL_TO_SKILL: dict[str, str] = {
    "file_read": "Read",
    "file_write": "Write",
    "git": "Bash(git:*)",
    "bash": "Bash",
    "grep": "Grep",
    "glob": "Glob",
}

SKILL_TO_TOOL: dict[str, str] = {v: k for k, v in TOOL_TO_SKILL.items()}


class EmployeeArg(BaseModel):
    """员工参数定义."""

    name: str = Field(description="参数名")
    description: str = Field(default="", description="参数说明")
    required: bool = Field(default=False, description="是否必填")
    default: str | None = Field(default=None, description="默认值")


class EmployeeOutput(BaseModel):
    """员工输出配置."""

    format: Literal["markdown", "json", "text"] = Field(
        default="markdown", description="输出格式"
    )
    filename: str = Field(default="", description="输出文件名模板")
    dir: str = Field(default=".crew/logs", description="输出目录")


class Employee(BaseModel):
    """数字员工定义 — 对应一个 EMPLOYEE.md 文件."""

    name: str = Field(description="唯一标识符，仅 [a-z0-9-]")
    display_name: str = Field(default="", description="中文显示名")
    character_name: str = Field(default="", description="角色姓名")
    summary: str = Field(default="", description="一段话摘要（用于讨论会 summary 模式）")
    version: str = Field(default="1.0", description="版本号")
    description: str = Field(description="一句话描述")
    tags: list[str] = Field(default_factory=list, description="分类标签")
    author: str = Field(default="", description="作者")
    triggers: list[str] = Field(default_factory=list, description="触发别名")
    args: list[EmployeeArg] = Field(default_factory=list, description="参数定义")
    output: EmployeeOutput = Field(default_factory=EmployeeOutput, description="输出配置")
    tools: list[str] = Field(default_factory=list, description="需要的工具声明")
    context: list[str] = Field(default_factory=list, description="需要预读的文件/模式")
    model: str = Field(default="", description="推荐使用的模型 ID（如 claude-opus-4-6）")
    api_key: str = Field(default="", description="专属 API key（留空则使用环境变量）")
    base_url: str = Field(default="", description="专属 API base URL（留空则按 provider 自动推断）")
    fallback_model: str = Field(default="", description="备用模型（主模型失败时自动切换）")
    fallback_api_key: str = Field(default="", description="备用模型的 API key")
    fallback_base_url: str = Field(default="", description="备用模型的 base URL")
    agent_id: int | None = Field(default=None, description="绑定的 knowlyr-id Agent ID")
    avatar_prompt: str = Field(default="", description="头像生成 prompt（留空则自动推断）")
    research_instructions: str = Field(
        default="", description="讨论前预研指令（编排式讨论的 round 0）"
    )
    body: str = Field(description="Markdown 正文（自然语言指令）")
    source_path: Path | None = Field(default=None, description="来源文件路径")
    source_layer: Literal["builtin", "global", "skill", "project", "private"] = Field(
        default="builtin", description="来源层"
    )

    @property
    def effective_display_name(self) -> str:
        """有效显示名."""
        return self.display_name or self.name


class WorkLogEntry(BaseModel):
    """单条工作日志."""

    timestamp: datetime = Field(default_factory=datetime.now)
    employee_name: str
    action: str = Field(description="动作描述")
    detail: str = Field(default="", description="详细信息")
    args: dict[str, str] = Field(default_factory=dict, description="调用参数")
    agent_id: int | None = Field(default=None, description="关联的 knowlyr-id Agent ID")
    severity: str = Field(default="info", description="info / warning / critical")
    metrics: dict[str, float] = Field(default_factory=dict, description="可选指标（数值类型）")
    links: list[str] = Field(default_factory=list, description="关联链接或引用")


class ParticipantPrompt(BaseModel):
    """编排式讨论中单个参会者的独立 prompt."""

    employee_name: str = Field(description="员工标识符")
    character_name: str = Field(default="", description="角色姓名")
    role: str = Field(default="speaker", description="会议角色")
    prompt: str = Field(description="完整的独立 prompt")


class RoundPlan(BaseModel):
    """编排式讨论中单轮的计划."""

    round_number: int = Field(description="轮次序号（从 1 开始）")
    name: str = Field(default="", description="轮次名称")
    instruction: str = Field(default="", description="轮次指令")
    interaction: str = Field(default="free", description="互动模式")
    participant_prompts: list[ParticipantPrompt] = Field(
        default_factory=list, description="各参会者的独立 prompt"
    )


class DiscussionPlan(BaseModel):
    """编排式讨论的完整计划 — 多轮多人独立 prompt.

    使用方式:
    1. 依次执行每轮的 participant_prompts（同一轮内可并行）
    2. 收集每轮所有参会者的输出
    3. 将输出填入下一轮 prompt 中的 {previous_rounds} 占位符
    4. 最后一轮结束后，用 synthesis_prompt 生成最终汇总
    """

    discussion_name: str = Field(description="讨论会名称")
    topic: str = Field(description="议题")
    goal: str = Field(default="", description="目标")
    rounds: list[RoundPlan] = Field(default_factory=list, description="各轮计划")
    synthesis_prompt: str = Field(
        default="", description="最终汇总 prompt（在所有轮次结束后执行）"
    )


# ── ActionPlan 数据模型（讨论→执行衔接）──


class ActionItem(BaseModel):
    """讨论产出的行动项."""

    id: str = Field(description="行动项 ID，如 A1, A2")
    description: str = Field(description="行动描述")
    assignee_role: str = Field(description="建议负责的角色（executor/reviewer/monitor）")
    assignee_employee: str = Field(default="", description="建议负责的员工名称")
    depends_on: list[str] = Field(default_factory=list, description="依赖的行动项 ID")
    priority: Literal["P0", "P1", "P2", "P3"] = Field(default="P2")
    verification: str = Field(default="", description="验证标准")
    phase: Literal["research", "implement", "review", "deploy"] = Field(
        default="implement", description="执行阶段"
    )


class DiscussionActionPlan(BaseModel):
    """讨论产出的可执行行动计划."""

    discussion_name: str = Field(default="", description="讨论会名称")
    topic: str = Field(default="", description="议题")
    decisions: list[str] = Field(default_factory=list, description="达成的共识决策")
    unresolved: list[str] = Field(default_factory=list, description="未解决分歧")
    actions: list[ActionItem] = Field(default_factory=list, description="行动项列表")
    review_criteria: list[str] = Field(
        default_factory=list, description="验收标准（用于后续 review pipeline）"
    )


class DiscoveryResult(BaseModel):
    """发现结果 — 包含发现的员工和冲突信息."""

    employees: dict[str, Employee] = Field(
        default_factory=dict, description="name -> Employee 映射"
    )
    conflicts: list[dict[str, Any]] = Field(
        default_factory=list, description="同名冲突记录"
    )

    def get(self, name_or_trigger: str) -> Employee | None:
        """按名称或触发别名查找员工."""
        if name_or_trigger in self.employees:
            return self.employees[name_or_trigger]
        for emp in self.employees.values():
            if name_or_trigger in emp.triggers:
                return emp
        return None


# ── Pipeline 数据模型 ──


class PipelineStep(BaseModel):
    """流水线步骤."""

    employee: str = Field(description="员工名称")
    id: str = Field(default="", description="步骤标识符（用于输出引用）")
    args: dict[str, str] = Field(default_factory=dict, description="参数")


class ParallelGroup(BaseModel):
    """并行步骤组."""

    parallel: list[PipelineStep] = Field(description="并行执行的步骤列表")


class StepResult(BaseModel):
    """单步执行结果."""

    employee: str = Field(description="员工名称")
    step_id: str = Field(default="", description="步骤 ID")
    step_index: int = Field(description="步骤序号（flat index, 从 0 开始）")
    args: dict[str, str] = Field(default_factory=dict, description="解析后参数")
    prompt: str = Field(description="生成的完整 prompt")
    output: str = Field(default="", description="LLM 输出（仅 execute 模式）")
    error: bool = Field(default=False, description="是否出错")
    error_message: str = Field(default="", description="错误信息")
    model: str = Field(default="", description="使用的模型")
    input_tokens: int = Field(default=0, description="输入 token 数")
    output_tokens: int = Field(default=0, description="输出 token 数")
    duration_ms: int = Field(default=0, description="执行耗时 (ms)")


class PipelineResult(BaseModel):
    """流水线执行结果."""

    pipeline_name: str = Field(description="流水线名称")
    mode: Literal["prompt", "execute"] = Field(description="执行模式")
    steps: list[StepResult | list[StepResult]] = Field(
        description="步骤结果（list[StepResult] 表示并行组）"
    )
    total_duration_ms: int = Field(default=0, description="总耗时 (ms)")
    total_input_tokens: int = Field(default=0, description="总输入 token")
    total_output_tokens: int = Field(default=0, description="总输出 token")


# ── Agent 执行数据模型 ──


@dataclass
class ToolCall:
    """LLM 返回的工具调用."""

    id: str
    name: str
    arguments: dict[str, Any] = dc_field(default_factory=dict)


@dataclass
class ToolExecutionResult:
    """带工具调用的 LLM 执行结果."""

    content: str
    tool_calls: list[ToolCall] = dc_field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""

    @property
    def has_tool_calls(self) -> bool:
        """是否包含工具调用."""
        return len(self.tool_calls) > 0
