"""流水线引擎 — 多员工顺序执行."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from crew.context_detector import ProjectInfo, detect_project
from crew.discovery import discover_employees
from crew.engine import CrewEngine


class PipelineStep(BaseModel):
    """流水线步骤."""

    employee: str = Field(description="员工名称")
    args: dict[str, str] = Field(default_factory=dict, description="参数")


class Pipeline(BaseModel):
    """流水线定义."""

    name: str = Field(description="流水线名称")
    description: str = Field(default="", description="描述")
    steps: list[PipelineStep] = Field(description="步骤列表")


def load_pipeline(path: Path) -> Pipeline:
    """从 YAML 文件加载流水线定义."""
    content = path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    return Pipeline(**data)


def validate_pipeline(pipeline: Pipeline, project_dir: Path | None = None) -> list[str]:
    """校验流水线定义，返回错误列表."""
    errors = []
    if not pipeline.steps:
        errors.append("流水线至少需要一个步骤")
        return errors

    result = discover_employees(project_dir=project_dir)
    for i, step in enumerate(pipeline.steps):
        emp = result.get(step.employee)
        if emp is None:
            errors.append(f"步骤 {i + 1}: 未找到员工 '{step.employee}'")
    return errors


def run_pipeline(
    pipeline: Pipeline,
    initial_args: dict[str, str] | None = None,
    project_dir: Path | None = None,
    agent_id: int | None = None,
    smart_context: bool = True,
) -> list[dict]:
    """顺序执行流水线，返回每步的结果.

    Returns:
        [{employee, args, prompt}]
    """
    initial_args = initial_args or {}
    result = discover_employees(project_dir=project_dir)
    engine = CrewEngine()

    # 检测项目类型
    project_info = detect_project(project_dir) if smart_context else None

    # 获取 agent 身份（可选）
    agent_identity = None
    if agent_id is not None:
        try:
            from crew.id_client import fetch_agent_identity
            agent_identity = fetch_agent_identity(agent_id)
        except ImportError:
            pass

    outputs = []
    for step in pipeline.steps:
        emp = result.get(step.employee)
        if emp is None:
            outputs.append({
                "employee": step.employee,
                "args": step.args,
                "prompt": f"[错误] 未找到员工: {step.employee}",
                "error": True,
            })
            continue

        # 合并参数：step.args 中的 $xxx 引用替换为 initial_args
        resolved_args = {}
        for k, v in step.args.items():
            if v.startswith("$") and v[1:] in initial_args:
                resolved_args[k] = initial_args[v[1:]]
            else:
                resolved_args[k] = v

        # 生成 prompt
        prompt = engine.prompt(
            emp,
            args=resolved_args,
            agent_identity=agent_identity,
            project_info=project_info,
        )

        outputs.append({
            "employee": step.employee,
            "args": resolved_args,
            "prompt": prompt,
        })

    return outputs


# ── 内置流水线发现 ──

PIPELINES_DIR_NAME = "pipelines"


def discover_pipelines(project_dir: Path | None = None) -> dict[str, Path]:
    """发现所有可用流水线.

    搜索顺序：
    1. 内置（src/crew/employees/pipelines/）
    2. 项目（.crew/pipelines/）
    """
    pipelines: dict[str, Path] = {}

    # 内置流水线
    builtin_dir = Path(__file__).parent / "employees" / PIPELINES_DIR_NAME
    if builtin_dir.is_dir():
        for f in sorted(builtin_dir.glob("*.yaml")):
            pipelines[f.stem] = f

    # 项目流水线（覆盖同名内置）
    root = Path(project_dir) if project_dir else Path.cwd()
    project_pipeline_dir = root / ".crew" / PIPELINES_DIR_NAME
    if project_pipeline_dir.is_dir():
        for f in sorted(project_pipeline_dir.glob("*.yaml")):
            pipelines[f.stem] = f

    return pipelines
