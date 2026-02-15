#!/usr/bin/env python3
"""按角色给员工分配 API 工具（GitHub/Notion/飞书/搜索等）。

用法: python scripts/sync_api_tools.py [--dry-run]
"""
import sys
from pathlib import Path

import yaml

EMPLOYEES_DIR = Path("private/employees")
SKIP = {"姜墨言-3073"}

# 按角色分配的 API 工具
ROLE_TOOLS: dict[str, list[str]] = {
    # 工程类 — GitHub + web_search
    "code-reviewer": ["github_prs", "github_issues", "github_repo_activity", "web_search"],
    "test-engineer": ["github_prs", "github_issues", "github_repo_activity"],
    "refactor-guide": ["github_prs", "github_issues", "github_repo_activity"],
    "pr-creator": ["github_prs", "github_issues", "github_repo_activity"],
    "backend-engineer": ["github_prs", "github_issues", "github_repo_activity", "web_search"],
    "frontend-engineer": ["github_prs", "github_issues", "github_repo_activity", "web_search"],
    "devops-engineer": ["github_prs", "github_issues", "github_repo_activity", "web_search"],
    "e2e-tester": ["github_prs", "github_issues", "github_repo_activity"],
    "debug-expert": ["github_prs", "github_issues", "github_repo_activity"],
    "performance-optimizer": ["github_prs", "github_issues", "github_repo_activity"],
    "security-auditor": ["github_prs", "github_issues", "github_repo_activity", "web_search"],
    "api-designer": ["github_prs", "github_issues", "github_repo_activity", "web_search"],
    "dba": ["github_prs", "github_issues", "github_repo_activity"],
    "data-engineer": ["github_prs", "github_issues", "github_repo_activity"],
    "mlops-engineer": ["github_prs", "github_issues", "github_repo_activity"],
    "solutions-architect": ["github_prs", "github_issues", "github_repo_activity", "web_search"],
    # 产品/文档类 — GitHub + Notion + 飞书文档
    "product-manager": [
        "github_prs", "github_issues", "github_repo_activity", "web_search",
        "notion_search", "notion_read", "notion_create",
        "search_feishu_docs", "read_feishu_doc", "create_feishu_doc",
    ],
    "doc-writer": [
        "github_prs", "github_issues", "web_search",
        "notion_search", "notion_read", "notion_create",
        "search_feishu_docs", "read_feishu_doc", "create_feishu_doc",
    ],
    "ux-designer": ["github_issues", "web_search", "notion_search", "notion_read"],
    # 研究类 — web_search + 信息采集
    "algorithm-researcher": ["web_search", "read_url", "rss_read", "github_repo_activity"],
    "nlp-researcher": ["web_search", "read_url", "rss_read", "github_repo_activity"],
    "sociology-researcher": ["web_search", "read_url", "rss_read"],
    "economics-researcher": ["web_search", "read_url", "rss_read"],
    "benchmark-specialist": ["web_search", "read_url", "github_repo_activity"],
    "data-quality-expert": ["web_search", "github_issues", "github_repo_activity"],
    # 运营/业务类
    "i18n-expert": ["web_search", "github_prs", "github_issues"],
    "community-operator": ["web_search", "read_url", "rss_read", "notion_search", "notion_read"],
    "bd-manager": ["web_search", "read_url", "notion_search", "notion_read", "notion_create"],
    "customer-success": ["web_search", "notion_search", "notion_read"],
    "hr-manager": ["web_search", "notion_search", "notion_read", "notion_create"],
    "finance-expert": ["web_search", "notion_search", "notion_read"],
    "legal-counsel": ["web_search", "read_url"],
}


def main():
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("=== DRY RUN ===\n")

    updated = 0
    for d in sorted(EMPLOYEES_DIR.iterdir()):
        if not d.is_dir() or d.name in SKIP:
            continue
        yaml_path = d / "employee.yaml"
        if not yaml_path.exists():
            continue

        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        emp_name = config.get("name", "")

        new_tools = ROLE_TOOLS.get(emp_name, [])
        if not new_tools:
            print(f"  {d.name} ({emp_name}): skip (no mapping)")
            continue

        existing = config.get("tools", [])
        added = [t for t in new_tools if t not in existing]
        if not added:
            print(f"  {d.name} ({emp_name}): skip (already has all tools)")
            continue

        if dry_run:
            joined = ", ".join(added)
            print(f"  {d.name} ({emp_name}): would add +{len(added)} ({joined})")
        else:
            config["tools"] = existing + added
            yaml_path.write_text(
                yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
            joined = ", ".join(added)
            print(f"  {d.name} ({emp_name}): +{len(added)} ({joined})")
        updated += 1

    print(f"\n{'将更新' if dry_run else '已更新'}: {updated}")


if __name__ == "__main__":
    main()
