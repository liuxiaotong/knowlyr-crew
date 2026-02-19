#!/usr/bin/env python3
"""员工权限审计脚本 — 一键检查所有私有员工的配置一致性.

检查项:
  1. 必填字段 (name, description, tools)
  2. permissions.roles 非空
  3. 团队归属（恰好 1 个团队）
  4. Profile 对齐（roles 包含团队对应的 profile）
  5. 工具冲突（声明的工具被 profile 排除的）
  6. 权限级别（恰好 1 个 authority level）
  7. 自检覆盖（body 中含 ## 完成后自检）

Usage:
  uv run python scripts/audit_employees.py [--json]
"""

import json
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from crew.discovery import discover_employees
from crew.organization import load_organization
from crew.tool_schema import resolve_effective_tools, validate_permissions

# 团队 → Profile 映射（与 tests/test_private_employees.py 一致）
TEAM_PROFILE_MAP = {
    "engineering": "profile-engineer",
    "data": "profile-data",
    "research": "profile-researcher",
    "infrastructure": "profile-infra",
    "business": "profile-business",
    "functions": "profile-functions",
}
PROFILE_OVERRIDES = {"security-auditor": "profile-security"}
PERMISSIONS_EXCEPTIONS = {"ceo-assistant"}


def audit_employee(emp, org):
    """审计单个员工，返回 (issues, infos) 列表."""
    issues = []  # 错误/警告
    infos = []   # 信息

    name = emp.name

    # 1. 必填字段
    if not emp.description:
        issues.append("缺少 description")
    if not emp.tools:
        issues.append("tools 为空")

    # 跳过例外员工的权限检查
    if name in PERMISSIONS_EXCEPTIONS:
        infos.append("权限例外，跳过 profile 检查")
        return issues, infos

    # 2. permissions.roles
    if emp.permissions is None:
        issues.append("缺少 permissions")
    elif not emp.permissions.roles:
        issues.append("permissions.roles 为空")

    # 3. 团队归属
    teams = [tid for tid, team in org.teams.items() if name in team.members]
    if len(teams) == 0:
        issues.append("不在任何团队中")
    elif len(teams) > 1:
        issues.append(f"属于多个团队: {teams}")

    # 4. Profile 对齐
    team_id = teams[0] if teams else None
    if team_id and emp.permissions and emp.permissions.roles:
        expected = PROFILE_OVERRIDES.get(name, TEAM_PROFILE_MAP.get(team_id))
        if expected and expected not in emp.permissions.roles:
            issues.append(f"期望 {expected}，实际 roles={emp.permissions.roles}")

    # 5. 工具冲突
    if emp.permissions:
        effective = resolve_effective_tools(emp)
        excluded = set(emp.tools) - effective
        if excluded:
            infos.append(f"工具被排除: {', '.join(sorted(excluded))}")

    # 6. 权限级别
    auths = [level for level, auth in org.authority.items() if name in auth.members]
    if len(auths) == 0:
        issues.append("无权限级别")
    elif len(auths) > 1:
        issues.append(f"多个权限级别: {auths}")

    # 7. 自检覆盖
    if hasattr(emp, "body") and emp.body:
        if "完成后自检" not in emp.body:
            issues.append("body 中缺少 ## 完成后自检")
    else:
        issues.append("body 为空，无法检查自检覆盖")

    # validate_permissions 额外警告（去重：工具排除已在上面检查过）
    warnings = validate_permissions(emp)
    for w in warnings:
        if "已声明工具被权限排除" not in w and w not in issues:
            infos.append(w)

    return issues, infos


def main():
    json_output = "--json" in sys.argv

    discovery = discover_employees(project_dir=PROJECT_DIR, cache_ttl=0)
    org = load_organization(project_dir=PROJECT_DIR)

    # 只审计 private 层员工
    private_dir = PROJECT_DIR / "private" / "employees"
    private_names = set()
    if private_dir.is_dir():
        for d in private_dir.iterdir():
            if d.is_dir() and (d / "employee.yaml").exists():
                import yaml
                config = yaml.safe_load((d / "employee.yaml").read_text("utf-8"))
                private_names.add(config.get("name", d.name))

    if not private_names:
        print("未找到 private 员工，跳过审计。")
        return

    results = []
    total_pass = 0
    total_warn = 0
    total_fail = 0

    for name in sorted(private_names):
        emp = discovery.get(name)
        if emp is None:
            results.append({"name": name, "status": "MISS", "issues": ["未被发现"], "infos": []})
            total_fail += 1
            continue

        issues, infos = audit_employee(emp, org)

        team_id = org.get_team(name) or "-"
        auth = org.get_authority(name) or "-"

        if issues:
            status = "FAIL"
            total_fail += 1
        elif infos:
            status = "INFO"
            total_warn += 1
        else:
            status = "PASS"
            total_pass += 1

        results.append({
            "name": name,
            "status": status,
            "team": team_id,
            "authority": auth,
            "issues": issues,
            "infos": infos,
        })

    if json_output:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    # Rich 输出
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"员工权限审计 ({len(private_names)} 人)")
        table.add_column("员工", style="cyan", width=24)
        table.add_column("状态", width=6)
        table.add_column("团队", width=16)
        table.add_column("权限", width=4)
        table.add_column("问题 / 信息", ratio=1)

        for r in results:
            status_style = {"PASS": "[green]PASS[/]", "INFO": "[yellow]INFO[/]",
                            "FAIL": "[red]FAIL[/]", "MISS": "[red]MISS[/]"}
            details = []
            for i in r["issues"]:
                details.append(f"[red]{i}[/]")
            for i in r.get("infos", []):
                details.append(f"[dim]{i}[/]")
            table.add_row(
                r["name"],
                status_style.get(r["status"], r["status"]),
                r.get("team", "-"),
                r.get("authority", "-"),
                "\n".join(details) if details else "[green]OK[/]",
            )

        console.print(table)
        console.print(
            f"\n总计: [green]{total_pass} PASS[/]  "
            f"[yellow]{total_warn} INFO[/]  "
            f"[red]{total_fail} FAIL[/]"
        )
    except ImportError:
        # 无 Rich 时 fallback 到纯文本
        print(f"{'员工':<24} {'状态':<6} {'团队':<16} {'权限':<4} 详情")
        print("-" * 80)
        for r in results:
            details = "; ".join(r["issues"] + r.get("infos", [])) or "OK"
            print(f"{r['name']:<24} {r['status']:<6} {r.get('team', '-'):<16} "
                  f"{r.get('authority', '-'):<4} {details}")
        print(f"\n总计: {total_pass} PASS  {total_warn} INFO  {total_fail} FAIL")

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
