#!/usr/bin/env python3
"""通过 API 批量创建 skills 到服务端."""

import json
import sys
from pathlib import Path

import requests

# API 配置
API_BASE = "https://crew.knowlyr.com"
API_TOKEN = "X52I08vGWptmvtZxCMzX500odojsdv30k-gEq0G4sp8"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
}


def upload_skill(employee: str, skill_file: Path) -> bool:
    """上传单个 skill."""
    try:
        # 读取 skill 数据
        with skill_file.open("r", encoding="utf-8") as f:
            skill_data = json.load(f)

        skill_name = skill_data.get("name")
        print(f"上传: {employee}/{skill_name}")

        # 调用 API 创建 skill
        url = f"{API_BASE}/api/employees/{employee}/skills"
        response = requests.post(url, headers=HEADERS, json=skill_data, timeout=10)

        if response.status_code == 200:
            print("  ✓ 成功")
            return True
        elif response.status_code == 400 and "already exists" in response.text:
            print("  ℹ️  已存在，跳过")
            return True
        else:
            print(f"  ✗ 失败: HTTP {response.status_code}")
            print(f"    {response.text[:200]}")
            return False

    except Exception as e:
        print(f"  ✗ 异常: {e}")
        return False


def main():
    """批量上传所有 skills."""
    print("=== 通过 API 批量创建 Skills ===\n")

    skills_dir = Path(".crew/skills")
    if not skills_dir.exists():
        print(f"❌ Skills 目录不存在: {skills_dir}")
        return 1

    success_count = 0
    failed_count = 0

    # 遍历所有员工目录
    for employee_dir in sorted(skills_dir.iterdir()):
        if not employee_dir.is_dir() or employee_dir.name == "triggers":
            continue

        employee = employee_dir.name
        print(f"\n处理员工: {employee}")

        # 遍历该员工的所有 skills
        for skill_file in sorted(employee_dir.glob("*.json")):
            result = upload_skill(employee, skill_file)
            if result:
                success_count += 1
            else:
                failed_count += 1

    # 总结
    print("\n" + "=" * 50)
    print("上传完成:")
    print(f"  - 成功: {success_count}")
    print(f"  - 失败: {failed_count}")
    print("=" * 50)

    # 验证
    print("\n验证服务端 Skills 统计:")
    try:
        response = requests.get(f"{API_BASE}/api/skills/stats", headers=HEADERS, timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print(f"  - 总数: {stats.get('total_skills', 0)}")
            print(f"  - 按员工: {stats.get('by_employee', {})}")
            print(f"  - 按分类: {stats.get('by_category', {})}")
        else:
            print(f"  ✗ 获取统计失败: HTTP {response.status_code}")
    except Exception as e:
        print(f"  ✗ 验证失败: {e}")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
