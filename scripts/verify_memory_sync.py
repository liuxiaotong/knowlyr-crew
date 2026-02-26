#!/usr/bin/env python3
"""verify_memory_sync.py — 对账脚本：对比本地缓存 vs crew API 的记忆一致性.

用法:
    # 检查指定员工
    python scripts/verify_memory_sync.py backend-engineer

    # 检查所有有缓存的员工
    python scripts/verify_memory_sync.py --all

    # JSON 输出（适合 cron 日志）
    python scripts/verify_memory_sync.py --all --json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

CREW_API_URL = os.environ.get("CREW_REMOTE_URL", "https://crew.knowlyr.com")
CREW_API_TOKEN = os.environ.get("CREW_API_TOKEN", "")
LOCAL_CACHE_DIR = Path.home() / ".knowlyr" / "meta" / "employee-memories"


def load_local_cache(slug: str) -> dict | None:
    """加载本地缓存文件."""
    cache_file = LOCAL_CACHE_DIR / f"{slug}.json"
    if not cache_file.exists():
        return None
    try:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def fetch_remote_state(slug: str) -> dict | None:
    """从 crew API 获取员工当前状态."""
    url = (
        f"{CREW_API_URL.rstrip('/')}/api/employees/{slug}/state"
        f"?memory_limit=50&min_importance=0"
    )
    headers = {"Authorization": f"Bearer {CREW_API_TOKEN}"}
    req = Request(url, headers=headers)

    try:
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, Exception) as e:
        print(f"  [ERROR] 获取 {slug} 远端状态失败: {e}", file=sys.stderr)
        return None


def compare_employee(slug: str) -> dict:
    """对比单个员工的本地缓存与远端状态."""
    result = {
        "slug": slug,
        "status": "unknown",
        "local_count": 0,
        "remote_count": 0,
        "local_latest": "",
        "remote_latest": "",
        "synced_at": "",
        "issues": [],
    }

    local = load_local_cache(slug)
    if local is None:
        result["status"] = "no_local_cache"
        result["issues"].append("本地缓存文件不存在")
        # 还是获取远端数据看看
        remote = fetch_remote_state(slug)
        if remote:
            remote_memories = remote.get("memories", [])
            result["remote_count"] = len(remote_memories)
            if remote_memories:
                result["remote_latest"] = remote_memories[0].get("created_at", "")
        return result

    result["synced_at"] = local.get("synced_at", "")
    local_memories = local.get("memories", [])
    result["local_count"] = len(local_memories)
    if local_memories:
        # 按 created_at 排序取最新
        latest = max(local_memories, key=lambda m: m.get("created_at", ""))
        result["local_latest"] = latest.get("created_at", "")

    remote = fetch_remote_state(slug)
    if remote is None:
        result["status"] = "remote_unavailable"
        result["issues"].append("无法连接远端 API")
        return result

    remote_memories = remote.get("memories", [])
    result["remote_count"] = len(remote_memories)
    if remote_memories:
        latest = max(remote_memories, key=lambda m: m.get("created_at", ""))
        result["remote_latest"] = latest.get("created_at", "")

    # 对比
    if result["local_count"] == result["remote_count"]:
        if result["local_latest"] == result["remote_latest"]:
            result["status"] = "synced"
        else:
            result["status"] = "timestamp_mismatch"
            result["issues"].append(
                f"最新时间戳不一致: 本地={result['local_latest']} 远端={result['remote_latest']}"
            )
    else:
        result["status"] = "count_mismatch"
        diff = result["remote_count"] - result["local_count"]
        result["issues"].append(
            f"记忆条数不一致: 本地={result['local_count']} 远端={result['remote_count']} (差{diff:+d})"
        )
        # 条数不同时也检查时间戳
        if result["local_latest"] != result["remote_latest"]:
            result["issues"].append(
                f"最新时间戳也不一致: 本地={result['local_latest']} 远端={result['remote_latest']}"
            )

    # 检查同步时间是否过旧
    if result["synced_at"]:
        try:
            synced = datetime.fromisoformat(result["synced_at"])
            age_hours = (datetime.now() - synced).total_seconds() / 3600
            if age_hours > 24:
                result["issues"].append(f"缓存已过期: 同步于 {age_hours:.0f} 小时前")
                if result["status"] == "synced":
                    result["status"] = "stale"
        except ValueError:
            pass

    return result


def get_cached_slugs() -> list[str]:
    """列出所有有本地缓存的员工 slug."""
    if not LOCAL_CACHE_DIR.is_dir():
        return []
    return sorted(f.stem for f in LOCAL_CACHE_DIR.glob("*.json"))


def main():
    parser = argparse.ArgumentParser(description="对账：本地缓存 vs crew API 记忆一致性")
    parser.add_argument("slugs", nargs="*", help="员工 slug")
    parser.add_argument("--all", action="store_true", help="检查所有有缓存的员工")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")

    args = parser.parse_args()

    if not CREW_API_TOKEN:
        print("错误: 未设置 CREW_API_TOKEN 环境变量", file=sys.stderr)
        sys.exit(1)

    if args.all:
        slugs = get_cached_slugs()
        if not slugs:
            print("没有本地缓存文件", file=sys.stderr)
            sys.exit(0)
    elif args.slugs:
        slugs = args.slugs
    else:
        parser.print_help()
        sys.exit(1)

    results = []
    for slug in slugs:
        result = compare_employee(slug)
        results.append(result)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        # 人类可读输出
        synced = 0
        issues_total = 0

        for r in results:
            status_icon = {
                "synced": "[OK]",
                "stale": "[STALE]",
                "count_mismatch": "[DIFF]",
                "timestamp_mismatch": "[DIFF]",
                "no_local_cache": "[MISS]",
                "remote_unavailable": "[ERR]",
            }.get(r["status"], "[???]")

            print(
                f"  {status_icon} {r['slug']}: "
                f"本地={r['local_count']}条 远端={r['remote_count']}条 "
                f"同步于={r['synced_at'][:16] if r['synced_at'] else 'N/A'}"
            )

            if r["issues"]:
                for issue in r["issues"]:
                    print(f"        -> {issue}")
                issues_total += len(r["issues"])
            else:
                synced += 1

        print(
            f"\n=== 对账完成 ===\n"
            f"  检查: {len(results)} 个员工\n"
            f"  一致: {synced}\n"
            f"  问题: {issues_total} 项"
        )

    # 有问题时退出码 1（方便 cron 检测）
    has_issues = any(r["status"] not in ("synced",) for r in results)
    sys.exit(1 if has_issues else 0)


if __name__ == "__main__":
    main()
