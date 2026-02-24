"""GitHub 工具函数 — PR、Issue、Repo 活动查询."""

from __future__ import annotations

from crew.webhook_context import _GITHUB_API_BASE, _GITHUB_REPO_RE, _GITHUB_TOKEN


def _is_valid_github_repo(repo: str) -> bool:
    """验证 GitHub repo 格式，防止路径穿越."""
    return bool(_GITHUB_REPO_RE.match(repo))


async def _tool_github_prs(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,  # noqa: F821
) -> str:
    """查看 GitHub 仓库 PR 列表."""
    import httpx

    if not _GITHUB_TOKEN:
        return "GitHub 未配置。请设置 GITHUB_TOKEN 环境变量。"

    repo = (args.get("repo") or "").strip()
    state = args.get("state", "open")
    limit = min(args.get("limit", 10), 30)
    if not repo or "/" not in repo:
        return "repo 格式错误，应为 owner/repo。"
    if not _is_valid_github_repo(repo):
        return "repo 格式错误，应为 owner/repo（仅允许字母数字、-、_、.）。"

    headers = {"Authorization": f"token {_GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_GITHUB_API_BASE}/repos/{repo}/pulls",
            params={"state": state, "per_page": limit},
            headers=headers,
        )
    if resp.status_code != 200:
        return f"GitHub API 错误 {resp.status_code}: {resp.text[:500]}"

    prs = resp.json()
    if not prs:
        return f"{repo} 没有 {state} 状态的 PR。"

    lines = []
    for pr in prs[:limit]:
        labels = ", ".join(l["name"] for l in pr.get("labels", []))
        label_str = f" [{labels}]" if labels else ""
        lines.append(
            f"#{pr['number']} {pr['title']}{label_str} — {pr['user']['login']} ({pr['state']})\n{pr['html_url']}"
        )
    return "\n---\n".join(lines)


async def _tool_github_issues(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,  # noqa: F821
) -> str:
    """查看 GitHub 仓库 Issue 列表."""
    import httpx

    if not _GITHUB_TOKEN:
        return "GitHub 未配置。请设置 GITHUB_TOKEN 环境变量。"

    repo = (args.get("repo") or "").strip()
    state = args.get("state", "open")
    labels = args.get("labels", "")
    limit = min(args.get("limit", 10), 30)
    if not repo or "/" not in repo:
        return "repo 格式错误，应为 owner/repo。"
    if not _is_valid_github_repo(repo):
        return "repo 格式错误，应为 owner/repo（仅允许字母数字、-、_、.）。"

    params: dict = {"state": state, "per_page": limit}
    if labels:
        params["labels"] = labels

    headers = {"Authorization": f"token {_GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_GITHUB_API_BASE}/repos/{repo}/issues",
            params=params,
            headers=headers,
        )
    if resp.status_code != 200:
        return f"GitHub API 错误 {resp.status_code}: {resp.text[:500]}"

    # GitHub Issues API 也返回 PR，需要过滤
    issues = [i for i in resp.json() if "pull_request" not in i]
    if not issues:
        return f"{repo} 没有 {state} 状态的 Issue。"

    lines = []
    for issue in issues[:limit]:
        labels_str = ", ".join(l["name"] for l in issue.get("labels", []))
        label_part = f" [{labels_str}]" if labels_str else ""
        assignee = issue.get("assignee", {})
        assignee_str = f" → {assignee['login']}" if assignee else ""
        lines.append(
            f"#{issue['number']} {issue['title']}{label_part}{assignee_str}\n{issue['html_url']}"
        )
    return "\n---\n".join(lines)


async def _tool_github_repo_activity(
    args: dict,
    *,
    agent_id: int | None = None,
    ctx: _AppContext | None = None,  # noqa: F821
) -> str:
    """查看 GitHub 仓库最近活动."""
    from datetime import datetime, timedelta, timezone

    import httpx

    if not _GITHUB_TOKEN:
        return "GitHub 未配置。请设置 GITHUB_TOKEN 环境变量。"

    repo = (args.get("repo") or "").strip()
    days = args.get("days", 7)
    if not repo or "/" not in repo:
        return "repo 格式错误，应为 owner/repo。"
    if not _is_valid_github_repo(repo):
        return "repo 格式错误，应为 owner/repo（仅允许字母数字、-、_、.）。"

    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    headers = {"Authorization": f"token {_GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_GITHUB_API_BASE}/repos/{repo}/commits",
            params={"per_page": 30, "since": since},
            headers=headers,
        )
    if resp.status_code != 200:
        return f"GitHub API 错误 {resp.status_code}: {resp.text[:500]}"

    commits = resp.json()
    if not commits:
        return f"{repo} 最近 {days} 天没有提交。"

    authors: dict[str, int] = {}
    lines = []
    for c in commits[:20]:
        sha = c["sha"][:7]
        msg = (c["commit"]["message"].split("\n")[0])[:80]
        author = c["commit"]["author"]["name"]
        date = c["commit"]["author"]["date"][:10]
        authors[author] = authors.get(author, 0) + 1
        lines.append(f"{sha} {msg} — {author} ({date})")

    summary = f"最近 {days} 天：{len(commits)} 次提交，{len(authors)} 位贡献者"
    top_authors = ", ".join(
        f"{k}({v})" for k, v in sorted(authors.items(), key=lambda x: -x[1])[:5]
    )
    return f"{summary}\n贡献者: {top_authors}\n\n" + "\n".join(lines)


# ── Notion 工具 ──


HANDLERS: dict[str, object] = {
    "github_prs": _tool_github_prs,
    "github_issues": _tool_github_issues,
    "github_repo_activity": _tool_github_repo_activity,
}
