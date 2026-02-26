#!/usr/bin/env python3
"""sync_employee_memories.py — 从 crew API 拉取员工记忆到本地缓存.

用法:
    # 拉取指定员工
    python scripts/sync_employee_memories.py backend-engineer

    # 拉取所有活跃员工
    python scripts/sync_employee_memories.py --all

    # 强制刷新（忽略 12h 增量检查）
    python scripts/sync_employee_memories.py backend-engineer --force

    # 自定义参数
    python scripts/sync_employee_memories.py backend-engineer --min-importance 2 --days 60
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CREW_API_URL = os.environ.get("CREW_REMOTE_URL", "https://crew.knowlyr.com")
CREW_API_TOKEN = os.environ.get("CREW_API_TOKEN", "")
LOCAL_CACHE_DIR = Path.home() / ".knowlyr" / "meta" / "employee-memories"

# 增量模式：文件修改时间在此时间内则跳过
INCREMENTAL_THRESHOLD_HOURS = 12

# 默认参数
DEFAULT_MEMORY_LIMIT = 9999  # 全量同步（API 不支持 0=无限，用大数兜底）
DEFAULT_MIN_IMPORTANCE = 0  # 不过滤重要度，拉取全部
DEFAULT_DAYS = 30


def fetch_employee_state(
    slug: str,
    memory_limit: int = DEFAULT_MEMORY_LIMIT,
    min_importance: int = DEFAULT_MIN_IMPORTANCE,
) -> dict | None:
    """调用 crew API 获取员工状态（含记忆）."""
    url = (
        f"{CREW_API_URL.rstrip('/')}/api/employees/{slug}/state"
        f"?memory_limit={memory_limit}&min_importance={min_importance}"
        f"&sort_by=created_at"
    )

    headers = {"Authorization": f"Bearer {CREW_API_TOKEN}"}
    req = Request(url, headers=headers)

    try:
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        if e.code == 404:
            print(f"  [SKIP] {slug}: 员工不存在 (404)", file=sys.stderr)
        else:
            print(f"  [ERROR] {slug}: HTTP {e.code} — {e.reason}", file=sys.stderr)
        return None
    except URLError as e:
        print(f"  [ERROR] {slug}: 网络错误 — {e.reason}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [ERROR] {slug}: {e}", file=sys.stderr)
        return None


def filter_memories_by_age(memories: list[dict], max_days: int) -> list[dict]:
    """过滤出最近 N 天内的记忆."""
    cutoff = datetime.now() - timedelta(days=max_days)
    cutoff_str = cutoff.isoformat()

    result = []
    for m in memories:
        created = m.get("created_at", "")
        if created >= cutoff_str:
            result.append(m)
    return result


def should_refresh(slug: str, force: bool = False) -> bool:
    """判断是否需要刷新（增量模式）."""
    if force:
        return True

    cache_file = LOCAL_CACHE_DIR / f"{slug}.json"
    if not cache_file.exists():
        return True

    mtime = cache_file.stat().st_mtime
    age_hours = (time.time() - mtime) / 3600
    if age_hours > INCREMENTAL_THRESHOLD_HOURS:
        return True

    return False


def save_cache(slug: str, data: dict) -> Path:
    """保存到本地缓存文件."""
    LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = LOCAL_CACHE_DIR / f"{slug}.json"

    cache_data = {
        "slug": slug,
        "synced_at": datetime.now().isoformat(),
        "character_name": data.get("character_name", ""),
        "agent_status": data.get("agent_status", ""),
        "memory_count": len(data.get("memories", [])),
        "memories": data.get("memories", []),
    }

    cache_file.write_text(
        json.dumps(cache_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return cache_file


def sync_employee(
    slug: str,
    force: bool = False,
    memory_limit: int = DEFAULT_MEMORY_LIMIT,
    min_importance: int = DEFAULT_MIN_IMPORTANCE,
    max_days: int = DEFAULT_DAYS,
) -> bool:
    """同步单个员工的记忆，返回是否成功."""
    if not should_refresh(slug, force):
        print(f"  [SKIP] {slug}: 缓存新鲜 (<{INCREMENTAL_THRESHOLD_HOURS}h)", file=sys.stderr)
        return True

    state = fetch_employee_state(slug, memory_limit, min_importance)
    if state is None:
        return False

    memories = state.get("memories", [])
    memories = filter_memories_by_age(memories, max_days)

    cache_path = save_cache(slug, {**state, "memories": memories})
    print(
        f"  [OK] {slug}: {len(memories)} 条记忆 → {cache_path}",
        file=sys.stderr,
    )
    return True


def fetch_active_employees() -> list[str]:
    """获取所有活跃员工的 slug 列表."""
    url = f"{CREW_API_URL.rstrip('/')}/api/employees"
    headers = {"Authorization": f"Bearer {CREW_API_TOKEN}"}
    req = Request(url, headers=headers)

    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        employees = data if isinstance(data, list) else data.get("items", data.get("employees", []))
        slugs = []
        for emp in employees:
            if isinstance(emp, dict):
                status = emp.get("agent_status", "active")
                name = emp.get("name", "")
                if status == "active" and name:
                    slugs.append(name)
        return slugs
    except Exception as e:
        print(f"  [ERROR] 获取员工列表失败: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description="从 crew API 同步员工记忆到本地")
    parser.add_argument("slugs", nargs="*", help="员工 slug（如 backend-engineer）")
    parser.add_argument("--all", action="store_true", help="同步所有活跃员工")
    parser.add_argument("--force", action="store_true", help="强制刷新（忽略增量检查）")
    parser.add_argument(
        "--memory-limit", type=int, default=DEFAULT_MEMORY_LIMIT, help="最大记忆条数"
    )
    parser.add_argument(
        "--min-importance", type=int, default=DEFAULT_MIN_IMPORTANCE, help="最低重要性"
    )
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="最近 N 天的记忆")

    args = parser.parse_args()

    if not CREW_API_TOKEN:
        print("错误: 未设置 CREW_API_TOKEN 环境变量", file=sys.stderr)
        sys.exit(1)

    if args.all:
        slugs = fetch_active_employees()
        if not slugs:
            print("未找到活跃员工", file=sys.stderr)
            sys.exit(1)
        print(f"同步 {len(slugs)} 个活跃员工...", file=sys.stderr)
    elif args.slugs:
        slugs = args.slugs
    else:
        parser.print_help()
        sys.exit(1)

    success = 0
    failed = 0
    skipped = 0

    for slug in slugs:
        result = sync_employee(
            slug,
            force=args.force,
            memory_limit=args.memory_limit,
            min_importance=args.min_importance,
            max_days=args.days,
        )
        if result:
            success += 1
        else:
            failed += 1

    print(
        f"\n=== 同步完成 ===\n  成功: {success}\n  失败: {failed}\n  缓存目录: {LOCAL_CACHE_DIR}",
        file=sys.stderr,
    )

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
