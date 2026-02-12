"""执行引擎 — 变量替换 + prompt 生成."""

import re
import subprocess
from datetime import datetime
from pathlib import Path

from crew.models import Employee, EmployeeOutput


def _get_git_branch() -> str:
    """获取当前 git 分支名，失败返回空."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


class CrewEngine:
    """数字员工执行引擎.

    核心功能:
    1. render() — 变量替换，生成最终指令
    2. prompt() — 生成完整的 system prompt（供 LLM 使用）
    3. validate_args() — 校验参数
    """

    def validate_args(
        self,
        employee: Employee,
        args: dict[str, str] | None = None,
    ) -> list[str]:
        """校验参数完整性，返回错误列表."""
        errors = []
        args = args or {}
        for arg_def in employee.args:
            if arg_def.required and arg_def.name not in args:
                if arg_def.default is None:
                    errors.append(f"缺少必填参数: {arg_def.name}")
        return errors

    def render(
        self,
        employee: Employee,
        args: dict[str, str] | None = None,
        positional: list[str] | None = None,
    ) -> str:
        """变量替换后的完整指令.

        替换规则:
        - $name 类: 按 args.name 匹配
        - $1, $2: 按位置参数
        - $ARGUMENTS: 所有参数空格拼接
        - {date}, {datetime}, {cwd}, {git_branch}, {name}: 环境变量
        """
        args = args or {}
        positional = positional or []
        text = employee.body

        # 1. 填充默认值
        effective_args: dict[str, str] = {}
        for arg_def in employee.args:
            if arg_def.name in args:
                effective_args[arg_def.name] = args[arg_def.name]
            elif arg_def.default is not None:
                effective_args[arg_def.name] = arg_def.default

        # 2. 按 args.name 替换 $name（长名优先，避免 $target 被 $t 部分匹配）
        for name in sorted(effective_args.keys(), key=len, reverse=True):
            text = text.replace(f"${name}", effective_args[name])

        # 3. 位置参数替换 $1, $2, ...
        for i in range(len(positional), 0, -1):
            text = text.replace(f"${i}", positional[i - 1])

        # 4. $ARGUMENTS 和 $@
        all_args_str = " ".join(positional) if positional else " ".join(effective_args.values())
        text = text.replace("$ARGUMENTS", all_args_str)
        text = text.replace("$@", all_args_str)

        # 5. 环境变量
        now = datetime.now()
        env_vars = {
            "{date}": now.strftime("%Y-%m-%d"),
            "{datetime}": now.strftime("%Y-%m-%d %H:%M:%S"),
            "{cwd}": str(Path.cwd()),
            "{git_branch}": _get_git_branch(),
            "{name}": employee.name,
        }
        for placeholder, value in env_vars.items():
            text = text.replace(placeholder, value)

        return text

    def prompt(
        self,
        employee: Employee,
        args: dict[str, str] | None = None,
        positional: list[str] | None = None,
        agent_identity: "AgentIdentity | None" = None,
        project_info: "ProjectInfo | None" = None,
    ) -> str:
        """生成完整的 system prompt.

        包含角色前言 + 渲染后正文 + 输出约束。

        Args:
            agent_identity: 可选的 knowlyr-id Agent 身份（注入 prompt header）
            project_info: 可选的项目类型检测结果（注入 prompt header + 环境变量）
        """
        rendered = self.render(employee, args=args, positional=positional)

        # 项目类型环境变量替换（在渲染后的 body 中）
        if project_info:
            rendered = rendered.replace("{project_type}", project_info.project_type)
            rendered = rendered.replace("{framework}", project_info.framework)
            rendered = rendered.replace("{test_framework}", project_info.test_framework)
            rendered = rendered.replace("{package_manager}", project_info.package_manager)

        display = employee.effective_display_name

        parts = [
            f"# {display}",
            "",
            f"**角色**: {display}",
        ]
        if employee.character_name:
            parts.append(f"**姓名**: {employee.character_name}")
        parts.append(f"**描述**: {employee.description}")

        # 注入 Agent 身份信息
        if agent_identity:
            if agent_identity.nickname:
                parts.append(f"**Agent**: {agent_identity.nickname}")
            if agent_identity.title:
                parts.append(f"**职称**: {agent_identity.title}")
            if agent_identity.domains:
                parts.append(f"**领域**: {', '.join(agent_identity.domains)}")

        if employee.model:
            parts.append(f"**模型**: {employee.model}")
        if employee.tags:
            parts.append(f"**标签**: {', '.join(employee.tags)}")
        if employee.tools:
            parts.append(f"**需要工具**: {', '.join(employee.tools)}")
        if employee.context:
            parts.append(f"**预读上下文**: {', '.join(employee.context)}")

        # 注入项目类型信息
        if project_info and project_info.project_type != "unknown":
            parts.append(f"**项目类型**: {project_info.display_label}")
            if project_info.test_framework:
                parts.append(f"**测试框架**: {project_info.test_framework}")
            if project_info.lint_tools:
                parts.append(f"**Lint**: {', '.join(project_info.lint_tools)}")
            if project_info.package_manager:
                parts.append(f"**包管理**: {project_info.package_manager}")

        # Agent memory（持久记忆/上下文）
        if agent_identity and agent_identity.memory:
            parts.extend(["", "---", "", "## Agent 记忆", "", agent_identity.memory])

        parts.extend(["", "---", "", rendered])

        # 输出约束
        default_output = EmployeeOutput()
        needs_output_section = (
            employee.output.format != default_output.format
            or bool(employee.output.filename)
            or employee.output.dir != default_output.dir
        )

        if needs_output_section:
            parts.extend(["", "---", "", "## 输出约束"])
            parts.append(f"- 输出格式: {employee.output.format}")
            if employee.output.filename:
                parts.append(f"- 文件名: {employee.output.filename}")
            if employee.output.dir:
                parts.append(f"- 输出目录: {employee.output.dir}")

        return "\n".join(parts)
