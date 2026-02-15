#!/usr/bin/env python3
"""从员工 prompt.md 中抽取角色定义到 soul.md，并添加通用沟通规则。

用法: python scripts/split_soul.py [--dry-run]
"""
import sys
from pathlib import Path

EMPLOYEES_DIR = Path("private/employees")
SKIP = {"姜墨言-3073"}  # 已手动处理

# 通用沟通规则 — 从姜墨言的铁律 8 抽象而来
COMMUNICATION_RULES = """\
## 沟通原则

1. 每条回复必须有信息量。"好的""收到""我来看看"这类纯意图表达不是信息，不要发。直接做事，做完说结果。
2. 能一句话说清的不要说三句。结论先行，细节按需展开。
3. 不确定的事直接说不确定，不要编。
4. 做完就停，不要在末尾追问"还需要我做什么吗""要不要我继续"。用户需要的时候自己会说。
"""


def extract_role_section(text: str) -> tuple[str, str]:
    """从 prompt.md 中提取 '# 角色定义' 段落。

    Returns:
        (role_section, remaining_text)
    """
    lines = text.split("\n")
    role_start = -1
    role_end = len(lines)

    for i, line in enumerate(lines):
        if line.strip() == "# 角色定义":
            role_start = i
        elif role_start >= 0 and line.startswith("# ") and i > role_start:
            role_end = i
            break

    if role_start < 0:
        return "", text

    role_lines = lines[role_start + 1 : role_end]
    remaining_lines = lines[:role_start] + lines[role_end:]

    role_text = "\n".join(role_lines).strip()
    remaining_text = "\n".join(remaining_lines).strip()

    return role_text, remaining_text


def process_employee(emp_dir: Path, dry_run: bool = False) -> str:
    """处理单个员工目录。返回状态描述。"""
    prompt_path = emp_dir / "prompt.md"
    soul_path = emp_dir / "soul.md"
    wf_dir = emp_dir / "workflows"
    comm_path = wf_dir / "communication.md"

    if not prompt_path.exists():
        return "skip (no prompt.md)"

    if soul_path.exists():
        return "skip (soul.md exists)"

    text = prompt_path.read_text(encoding="utf-8")
    role_text, remaining_text = extract_role_section(text)

    if not role_text or len(role_text.split("\n")) < 3:
        return "skip (role section too short)"

    if dry_run:
        return f"would create soul.md ({len(role_text.split(chr(10)))} lines) + communication.md"

    # 写 soul.md
    soul_path.write_text(role_text + "\n", encoding="utf-8")

    # 更新 prompt.md（去掉角色定义段）
    prompt_path.write_text(remaining_text + "\n", encoding="utf-8")

    # 写通用沟通规则
    wf_dir.mkdir(exist_ok=True)
    if not comm_path.exists():
        comm_path.write_text(COMMUNICATION_RULES, encoding="utf-8")

    return f"done (soul: {len(role_text.split(chr(10)))} lines)"


def main():
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("=== DRY RUN ===\n")

    results = {"done": 0, "skip": 0}

    for emp_dir in sorted(EMPLOYEES_DIR.iterdir()):
        if not emp_dir.is_dir():
            continue
        name = emp_dir.name
        if name in SKIP:
            continue

        status = process_employee(emp_dir, dry_run=dry_run)
        print(f"  {name}: {status}")

        if status.startswith("done") or status.startswith("would"):
            results["done"] += 1
        else:
            results["skip"] += 1

    print(f"\n处理: {results['done']}  跳过: {results['skip']}")


if __name__ == "__main__":
    main()
