"""轨迹录制 — 将 crew 员工执行数据写为 agentrecorder 标准 Trajectory 格式.

通过 contextvars 实现零侵入式录制：调用方用 TrajectoryCollector 上下文管理器
包裹执行流程，executor / agent_bridge 自动检测并记录每一步。

用法::

    with TrajectoryCollector("code-reviewer", "审查代码", model="claude-sonnet") as tc:
        result = execute_prompt(system_prompt=prompt, ...)
    traj = tc.finish()  # 写入 .crew/trajectories/trajectories.jsonl

依赖 agentrecorder 时输出标准 Trajectory JSONL；未安装时 fallback 为纯 JSON。
"""

from __future__ import annotations

import contextvars
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_collector_var: contextvars.ContextVar[TrajectoryCollector | None] = contextvars.ContextVar(
    "trajectory_collector", default=None
)


def _try_import_recorder():
    """尝试导入 agentrecorder，不可用时返回 None."""
    try:
        from agentrecorder.schema import (
            Outcome,
            Step,
            Trajectory,
        )
        from agentrecorder.schema import ToolCall as RecToolCall
        from knowlyrcore import TaskInfo
        from knowlyrcore import ToolResult as RecToolResult

        return Trajectory, Step, RecToolCall, RecToolResult, Outcome, TaskInfo
    except ImportError:
        return None


class TrajectoryCollector:
    """收集 LLM 调用数据，输出 agentrecorder 标准 Trajectory.

    使用 contextvars 实现跨函数传递，executor.py / agent_bridge.py 自动检测并记录。
    同时支持 sync 和 async 上下文。
    """

    def __init__(
        self,
        employee_name: str,
        task_description: str,
        model: str = "",
        *,
        output_dir: Path | None = None,
    ):
        self.employee_name = employee_name
        self.task_description = task_description
        self.model = model
        self.output_dir = output_dir or Path(".crew/trajectories")
        self._steps: list[dict[str, Any]] = []
        self._step_count = 0
        self._token: contextvars.Token | None = None
        self._pending: dict[str, Any] | None = None

    def __enter__(self) -> TrajectoryCollector:
        self._token = _collector_var.set(self)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._token is not None:
            _collector_var.reset(self._token)
            self._token = None

    @staticmethod
    def current() -> TrajectoryCollector | None:
        """获取当前上下文中的 collector（如果有）."""
        return _collector_var.get(None)

    # ── 单轮录制（execute_prompt） ──

    def add_prompt_step(
        self,
        content: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """记录 execute_prompt() 的单轮执行结果."""
        self._step_count += 1
        if not self.model:
            self.model = model
        self._steps.append(
            {
                "step_id": self._step_count,
                "thought": content,
                "tool_name": "respond",
                "tool_params": {},
                "tool_output": content,
                "tool_exit_code": 0,
                "timestamp": datetime.now().isoformat(),
                "token_count": input_tokens + output_tokens,
            }
        )

    # ── 多轮录制（execute_with_tools，分两步） ──

    def begin_tool_step(
        self,
        thought: str,
        tool_name: str,
        tool_params: dict[str, Any],
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "",
    ) -> None:
        """开始记录一个 tool-use 步骤（等待 complete_tool_step 提供工具执行结果）."""
        # 如果有上一个未完成的 pending step，先保存
        if self._pending is not None:
            self._pending.setdefault("tool_output", "")
            self._pending.setdefault("tool_exit_code", -1)
            self._steps.append(self._pending)
        self._step_count += 1
        if model and not self.model:
            self.model = model
        self._pending = {
            "step_id": self._step_count,
            "thought": thought,
            "tool_name": tool_name,
            "tool_params": tool_params,
            "timestamp": datetime.now().isoformat(),
            "token_count": input_tokens + output_tokens,
        }

    def complete_tool_step(self, tool_output: str, tool_exit_code: int = 0) -> None:
        """补全 pending 步骤的工具执行结果."""
        if self._pending is None:
            return
        self._pending["tool_output"] = tool_output
        self._pending["tool_exit_code"] = tool_exit_code
        self._steps.append(self._pending)
        self._pending = None

    # ── 完整步骤一次性录制 ──

    def add_tool_step(
        self,
        thought: str,
        tool_name: str,
        tool_params: dict[str, Any],
        tool_output: str,
        tool_exit_code: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """记录一个已完成的 tool-use 步骤（含工具执行结果）."""
        self._step_count += 1
        self._steps.append(
            {
                "step_id": self._step_count,
                "thought": thought,
                "tool_name": tool_name,
                "tool_params": tool_params,
                "tool_output": tool_output,
                "tool_exit_code": tool_exit_code,
                "timestamp": datetime.now().isoformat(),
                "token_count": input_tokens + output_tokens,
            }
        )

    # ── 输出 ──

    def finish(self, *, success: bool = True, score: float = 0.0) -> Any:
        """完成录制，写入 Trajectory JSONL.

        Returns:
            agentrecorder Trajectory 对象（已安装时），否则返回 dict。
        """
        # 保存未完成的 pending step
        if self._pending is not None:
            self._pending.setdefault("tool_output", "")
            self._pending.setdefault("tool_exit_code", -1)
            self._steps.append(self._pending)
            self._pending = None

        if not self._steps:
            logger.debug("无步骤数据，跳过轨迹录制")
            return None

        total_tokens = sum(s.get("token_count", 0) or 0 for s in self._steps)

        imports = _try_import_recorder()
        if imports is not None:
            Trajectory, Step, RecToolCall, RecToolResult, Outcome, TaskInfo = imports
            steps = [
                Step(
                    step_id=s["step_id"],
                    thought=s["thought"],
                    tool_call=RecToolCall(
                        name=s["tool_name"],
                        parameters=s["tool_params"],
                    ),
                    tool_result=RecToolResult(
                        output=s["tool_output"],
                        exit_code=s["tool_exit_code"],
                    ),
                    timestamp=s["timestamp"],
                    token_count=s.get("token_count"),
                )
                for s in self._steps
            ]
            traj = Trajectory(
                task=TaskInfo(
                    task_id=f"crew-{uuid.uuid4().hex[:8]}",
                    description=self.task_description,
                    domain="crew",
                ),
                agent=f"crew/{self.employee_name}",
                model=self.model,
                steps=steps,
                outcome=Outcome(
                    success=success,
                    score=score,
                    total_steps=len(steps),
                    total_tokens=total_tokens,
                ),
                metadata={"employee": self.employee_name},
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / "trajectories.jsonl"
            traj.to_jsonl(output_path)
            logger.info("轨迹已录制: %s (%d 步)", output_path, len(steps))
            return traj

        # fallback: agentrecorder 未安装，写纯 JSON
        data = {
            "task_id": f"crew-{uuid.uuid4().hex[:8]}",
            "employee": self.employee_name,
            "task": self.task_description,
            "model": self.model,
            "steps": self._steps,
            "success": success,
            "total_steps": len(self._steps),
            "total_tokens": total_tokens,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "trajectories.jsonl"
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        logger.info("轨迹已录制 (fallback JSON): %s (%d 步)", output_path, len(self._steps))
        return data
