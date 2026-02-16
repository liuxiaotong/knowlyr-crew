"""Crew — 数字员工管理框架

用 Markdown 定义「数字员工」，在 Claude Code 等 AI 编程工具中
加载不同员工，按预设流程自动完成工作。
"""

from pathlib import Path
import re

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
except ImportError:  # pragma: no cover
    PackageNotFoundError = Exception  # type: ignore


def _load_local_version() -> str:
    """从 pyproject.toml 读取版本，供本地开发环境使用."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists():
        return "0.0.0"
    try:
        match = re.search(r"^version\s*=\s*\"([^\"]+)\"", pyproject.read_text(encoding="utf-8"), re.MULTILINE)
        if match:
            return match.group(1)
    except Exception:
        pass
    return "0.0.0"


try:
    __version__ = _pkg_version("knowlyr-crew")
except PackageNotFoundError:  # pragma: no cover - editable install
    __version__ = _load_local_version()

from crew.models import (
    Condition,
    ConditionalBody,
    ConditionalStep,
    DiscoveryResult,
    DiscussionPlan,
    Employee,
    EmployeeArg,
    EmployeeOutput,
    LoopBody,
    LoopStep,
    ParallelGroup,
    ParticipantPrompt,
    PipelineResult,
    PipelineStep,
    RoundPlan,
    SKILL_TO_TOOL,
    StepResult,
    TOOL_TO_SKILL,
    WorkLogEntry,
)

__all__ = [
    "Condition",
    "ConditionalBody",
    "ConditionalStep",
    "DiscoveryResult",
    "DiscussionPlan",
    "Employee",
    "EmployeeArg",
    "EmployeeOutput",
    "LoopBody",
    "LoopStep",
    "ParallelGroup",
    "ParticipantPrompt",
    "PipelineResult",
    "PipelineStep",
    "RoundPlan",
    "SKILL_TO_TOOL",
    "StepResult",
    "TOOL_TO_SKILL",
    "WorkLogEntry",
    "__version__",
]
