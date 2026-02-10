"""Crew — 数字员工管理框架

用 Markdown 定义「数字员工」，在 Claude Code 等 AI 编程工具中
加载不同员工，按预设流程自动完成工作。
"""

__version__ = "0.1.0"

from crew.models import (
    DiscoveryResult,
    Employee,
    EmployeeArg,
    EmployeeOutput,
    WorkLogEntry,
)

__all__ = [
    "DiscoveryResult",
    "Employee",
    "EmployeeArg",
    "EmployeeOutput",
    "WorkLogEntry",
    "__version__",
]
