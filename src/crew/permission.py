"""权限守卫 — 运行时工具调用拦截."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crew.models import Employee

logger = logging.getLogger(__name__)


class PermissionDenied(Exception):
    """工具调用被权限系统拒绝."""

    def __init__(self, employee_name: str, tool_name: str):
        self.employee_name = employee_name
        self.tool_name = tool_name
        super().__init__(f"员工 '{employee_name}' 无权调用工具 '{tool_name}'")


class PermissionGuard:
    """工具调用权限守卫.

    用法::

        guard = PermissionGuard(employee)
        msg = guard.check_soft("bash")  # None 表示允许，否则返回拒绝消息
    """

    # 始终允许的工具（任务完成 / 工具加载）
    _ALWAYS_ALLOWED = {"submit", "finish", "load_tools"}

    def __init__(self, employee: Employee) -> None:
        from crew.tool_schema import resolve_effective_tools

        self.employee_name = employee.name
        self.allowed = resolve_effective_tools(employee) | self._ALWAYS_ALLOWED

    def check(self, tool_name: str) -> None:
        """检查工具调用是否允许，不允许则抛 PermissionDenied."""
        if tool_name not in self.allowed:
            logger.warning(
                "权限拒绝: %s 尝试调用 %s", self.employee_name, tool_name,
            )
            raise PermissionDenied(self.employee_name, tool_name)

    def check_soft(self, tool_name: str) -> str | None:
        """软检查 — 返回 None（允许）或拒绝消息."""
        if tool_name not in self.allowed:
            return (
                f"工具 '{tool_name}' 不在允许列表中。"
                f"可用工具: {', '.join(sorted(self.allowed - self._ALWAYS_ALLOWED))}"
            )
        return None
