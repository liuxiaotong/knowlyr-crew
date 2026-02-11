"""数据模型 — Employee / WorkLog / DiscoveryResult."""

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
    body: str = Field(description="Markdown 正文（自然语言指令）")
    source_path: Path | None = Field(default=None, description="来源文件路径")
    source_layer: Literal["builtin", "global", "skill", "project"] = Field(
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
