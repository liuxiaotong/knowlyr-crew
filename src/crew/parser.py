"""EMPLOYEE.md 解析器 — YAML frontmatter + Markdown 正文."""

import re
from pathlib import Path

import yaml

from crew.models import Employee, EmployeeArg, EmployeeOutput

# name 格式：仅小写字母、数字、连字符，不以连字符开头或结尾
NAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


def _split_frontmatter(content: str) -> tuple[dict, str]:
    """分离 YAML frontmatter 和 Markdown 正文.

    Returns:
        (frontmatter_dict, body_text)
    """
    content = content.strip()
    if not content.startswith("---"):
        return {}, content

    # 查找第二个 ---
    end = content.find("---", 3)
    if end == -1:
        return {}, content

    yaml_str = content[3:end].strip()
    body = content[end + 3:].strip()

    try:
        frontmatter = yaml.safe_load(yaml_str)
    except yaml.YAMLError:
        return {}, content

    if not isinstance(frontmatter, dict):
        return {}, content

    return frontmatter, body


def parse_employee(path: Path) -> Employee:
    """解析 EMPLOYEE.md 文件，返回 Employee 对象.

    Args:
        path: .md 文件路径

    Raises:
        ValueError: 解析失败或必填字段缺失
    """
    content = path.read_text(encoding="utf-8")
    return parse_employee_string(content, source_path=path)


def parse_employee_string(
    content: str,
    source_path: Path | None = None,
    source_layer: str = "builtin",
) -> Employee:
    """从字符串解析员工定义.

    Args:
        content: EMPLOYEE.md 文件内容
        source_path: 来源文件路径
        source_layer: 来源层（builtin / global / project）

    Raises:
        ValueError: 解析失败或必填字段缺失
    """
    frontmatter, body = _split_frontmatter(content)

    if not frontmatter:
        raise ValueError("缺少 YAML frontmatter（文件需以 --- 开头）")

    name = frontmatter.get("name")
    if not name:
        raise ValueError("缺少必填字段: name")

    description = frontmatter.get("description")
    if not description:
        raise ValueError("缺少必填字段: description")

    if not body:
        raise ValueError("Markdown 正文不能为空")

    # 解析 args
    raw_args = frontmatter.get("args", [])
    args = []
    if isinstance(raw_args, list):
        for item in raw_args:
            if isinstance(item, dict):
                args.append(EmployeeArg(**item))

    # 解析 output
    raw_output = frontmatter.get("output", {})
    output = EmployeeOutput(**raw_output) if isinstance(raw_output, dict) else EmployeeOutput()

    return Employee(
        name=name,
        display_name=frontmatter.get("display_name", ""),
        version=str(frontmatter.get("version", "1.0")),
        description=description,
        tags=frontmatter.get("tags", []),
        author=frontmatter.get("author", ""),
        triggers=frontmatter.get("triggers", []),
        args=args,
        output=output,
        body=body,
        source_path=source_path,
        source_layer=source_layer,
    )


def validate_employee(employee: Employee) -> list[str]:
    """校验员工定义，返回错误信息列表.

    Returns:
        错误信息列表，为空表示校验通过
    """
    errors = []

    # name 格式
    if not NAME_PATTERN.match(employee.name):
        errors.append(
            f"name '{employee.name}' 格式无效（仅允许小写字母、数字、连字符，"
            "不能以连字符开头或结尾）"
        )

    # name 长度
    if len(employee.name) > 64:
        errors.append(f"name 长度 {len(employee.name)} 超过 64 字符限制")

    # description 长度
    if len(employee.description) > 1024:
        errors.append(f"description 长度 {len(employee.description)} 超过 1024 字符限制")

    # body 非空
    if not employee.body.strip():
        errors.append("Markdown 正文不能为空")

    # args name 唯一性
    arg_names = [a.name for a in employee.args]
    if len(arg_names) != len(set(arg_names)):
        errors.append("args 中存在重复的参数名")

    # triggers 格式
    for trigger in employee.triggers:
        if not NAME_PATTERN.match(trigger):
            errors.append(f"trigger '{trigger}' 格式无效")

    return errors
