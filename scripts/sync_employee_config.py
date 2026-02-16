#!/usr/bin/env python3
"""同步姜墨言的工具列表和模型配置到其他所有员工的 employee.yaml."""

import yaml
from pathlib import Path

EMPLOYEES_DIR = Path(__file__).parent.parent / "private" / "employees"
SOURCE_DIR = EMPLOYEES_DIR / "姜墨言-3073"

# 要同步的字段
SYNC_TOOLS = True
SYNC_MODEL_CONFIG = True

# 模型配置字段
MODEL_CONFIG_FIELDS = [
    "model", "api_key", "base_url",
    "fallback_model", "fallback_api_key", "temperature",
]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def main():
    source = load_yaml(SOURCE_DIR / "employee.yaml")
    source_tools = source.get("tools", [])
    source_model_config = {k: source[k] for k in MODEL_CONFIG_FIELDS if k in source}

    print(f"源: 姜墨言 — {len(source_tools)} 个工具")
    print(f"模型配置: {list(source_model_config.keys())}")
    print()

    updated = 0
    for emp_dir in sorted(EMPLOYEES_DIR.iterdir()):
        if not emp_dir.is_dir() or emp_dir.name == "姜墨言-3073":
            continue

        yaml_path = emp_dir / "employee.yaml"
        if not yaml_path.exists():
            print(f"  跳过 {emp_dir.name}: 无 employee.yaml")
            continue

        data = load_yaml(yaml_path)
        old_tools = data.get("tools", [])

        if SYNC_TOOLS:
            data["tools"] = list(source_tools)

        if SYNC_MODEL_CONFIG:
            for k, v in source_model_config.items():
                data[k] = v

        save_yaml(yaml_path, data)
        updated += 1
        print(f"  ✓ {emp_dir.name} ({data.get('character_name', '?')}): {len(old_tools)} → {len(source_tools)} 工具")

    print(f"\n完成: 更新 {updated} 个员工")


if __name__ == "__main__":
    main()
