"""记忆管理命令 — memory group."""

import json
import sys

import click


@click.group()
def memory():
    """员工记忆管理 — 持久化经验存储."""
    pass


@memory.command("list")
def memory_list():
    """列出有记忆的员工."""
    from crew.memory import MemoryStore

    store = MemoryStore()
    employees = store.list_employees()
    if not employees:
        click.echo("暂无记忆数据。")
        return
    for emp in employees:
        entries = store.query(emp)
        click.echo(f"  {emp}: {len(entries)} 条记忆")


@memory.command("show")
@click.argument("employee")
@click.option(
    "--category",
    type=click.Choice(["decision", "estimate", "finding", "correction", "pattern"]),
    default=None,
    help="按类别过滤",
)
@click.option("-n", "--limit", type=int, default=20, help="返回条数")
@click.option("--include-expired", is_flag=True, help="包含已过期记忆")
def memory_show(employee: str, category: str | None, limit: int, include_expired: bool):
    """查看员工记忆."""
    from crew.memory import MemoryStore

    store = MemoryStore()
    entries = store.query(employee, category=category, limit=limit, include_expired=include_expired)
    if not entries:
        click.echo(f"员工 '{employee}' 暂无记忆。")
        return

    for entry in entries:
        conf = f" [{entry.confidence:.0%}]" if entry.confidence < 1.0 else ""
        tags = f" {entry.tags}" if entry.tags else ""
        shared = " [共享]" if entry.shared else ""
        ttl = f" (TTL:{entry.ttl_days}d)" if entry.ttl_days > 0 else ""
        click.echo(f"  [{entry.id}] ({entry.category}){conf}{tags}{shared}{ttl} {entry.content}")


@memory.command("add")
@click.argument("employee")
@click.option(
    "--category",
    "-c",
    required=True,
    type=click.Choice(["decision", "estimate", "finding", "correction", "pattern"]),
    help="记忆类别",
)
@click.option("--content", "-m", required=True, help="记忆内容")
@click.option("--ttl", type=int, default=0, help="生存期天数 (0=永不过期)")
@click.option("--tags", type=str, default="", help="逗号分隔的语义标签")
@click.option("--shared", is_flag=True, help="加入共享记忆池")
@click.option("--trigger", type=str, default="", help="触发条件（仅 pattern 类型）")
@click.option("--applicability", type=str, default="", help="适用范围，逗号分隔（仅 pattern 类型）")
def memory_add(
    employee: str,
    category: str,
    content: str,
    ttl: int,
    tags: str,
    shared: bool,
    trigger: str,
    applicability: str,
):
    """手动添加员工记忆."""
    from crew.memory import MemoryStore

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    app_list = [a.strip() for a in applicability.split(",") if a.strip()] if applicability else []
    store = MemoryStore()
    entry = store.add(
        employee=employee,
        category=category,
        content=content,
        ttl_days=ttl,
        tags=tag_list,
        shared=shared,
        trigger_condition=trigger,
        applicability=app_list,
    )
    click.echo(f"已添加: [{entry.id}] ({entry.category}) {entry.content}")


@memory.command("correct")
@click.argument("employee")
@click.argument("old_id")
@click.option("--content", "-m", required=True, help="纠正后的内容")
def memory_correct(employee: str, old_id: str, content: str):
    """纠正一条记忆."""
    from crew.memory import MemoryStore

    store = MemoryStore()
    new_entry = store.correct(employee=employee, old_id=old_id, new_content=content)
    if new_entry is None:
        click.echo(f"未找到记忆: {old_id}", err=True)
        sys.exit(1)
    click.echo(f"已纠正: [{new_entry.id}] {new_entry.content}")


@memory.command("index")
@click.option("--repair", is_flag=True, help="修复模式 — 删除并重建 embeddings.db")
def memory_index_cmd(repair: bool):
    """重建混合搜索索引."""
    if repair:
        from crew.memory import MemoryStore
        from crew.memory_search import SemanticMemoryIndex

        store = MemoryStore()
        memory_dir = store.memory_dir
        db_path = memory_dir / "embeddings.db"
        if db_path.exists():
            db_path.unlink()
            click.echo(f"已删除旧索引: {db_path}")

        idx = SemanticMemoryIndex(memory_dir)
        total = 0
        for emp in store.list_employees():
            entries = store.query(emp, limit=1000)
            count = idx.reindex(emp, entries)
            total += count
            click.echo(f"  {emp}: {count}/{len(entries)} 条已索引")
        idx.close()
        click.echo(f"修复完成: 共 {total} 条")
        return

    from crew.memory_index import MemorySearchIndex

    index = MemorySearchIndex()
    stats = index.rebuild()
    click.echo(f"索引完成: 记忆 {stats.memory_entries} 条, 会话 {stats.session_messages} 条")


@memory.command("search")
@click.argument("query")
@click.option("--employee", type=str, default=None, help="按员工过滤")
@click.option("--kind", type=click.Choice(["memory", "session"]), default=None, help="数据类型过滤")
@click.option("-n", "--limit", type=int, default=5, help="返回条数")
@click.option("--json", "json_output", is_flag=True, help="JSON 输出")
def memory_search(
    query: str, employee: str | None, kind: str | None, limit: int, json_output: bool
):
    """搜索持久记忆 + 会话记录."""
    from crew.memory_index import MemorySearchIndex

    index = MemorySearchIndex()
    results = index.search(query, limit=limit, employee=employee, kind=kind)

    if not results:
        click.echo("未找到匹配项。")
        return

    if json_output:
        click.echo(json.dumps(results, ensure_ascii=False, indent=2))
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="记忆搜索结果")
        table.add_column("类型", style="cyan")
        table.add_column("员工")
        table.add_column("标题")
        table.add_column("摘要")

        for item in results:
            snippet = item.get("snippet") or item.get("content", "")
            table.add_row(
                item.get("kind", ""),
                item.get("employee", ""),
                item.get("title", ""),
                snippet[:120],
            )

        console.print(table)
    except ImportError:
        for item in results:
            snippet = item.get("snippet") or item.get("content", "")
            click.echo(
                f"[{item.get('kind', '')}] {item.get('employee', '')} - {item.get('title', '')}\n  {snippet}"
            )


@memory.command("shared")
@click.option("--tags", type=str, default=None, help="按标签过滤（逗号分隔）")
@click.option("-n", "--limit", type=int, default=10, help="返回条数")
def memory_shared(tags: str | None, limit: int):
    """查看共享记忆池."""
    from crew.memory import MemoryStore

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    store = MemoryStore()
    entries = store.query_shared(tags=tag_list, limit=limit)
    if not entries:
        click.echo("共享记忆池为空。")
        return

    for entry in entries:
        conf = f" [{entry.confidence:.0%}]" if entry.confidence < 1.0 else ""
        tag_str = f" {entry.tags}" if entry.tags else ""
        click.echo(
            f"  [{entry.id}] ({entry.employee}/{entry.category}){conf}{tag_str} {entry.content}"
        )


@memory.command("config")
@click.option("--ttl", type=int, default=None, help="默认 TTL 天数")
@click.option("--max-entries", type=int, default=None, help="每员工最大条数")
@click.option("--half-life", type=float, default=None, help="置信度衰减半衰期（天）")
@click.option("--show", is_flag=True, help="显示当前配置")
def memory_config(ttl: int | None, max_entries: int | None, half_life: float | None, show: bool):
    """查看或设置记忆系统配置."""
    from crew.memory import MemoryStore

    store = MemoryStore()
    config_path = store.memory_dir / "config.json"

    if show or (ttl is None and max_entries is None and half_life is None):
        click.echo(f"配置文件: {config_path}")
        click.echo(f"  default_ttl_days:          {store.config.default_ttl_days}")
        click.echo(f"  max_entries_per_employee:   {store.config.max_entries_per_employee}")
        click.echo(f"  confidence_half_life_days:  {store.config.confidence_half_life_days}")
        click.echo(f"  auto_index:                {store.config.auto_index}")
        return

    data = store.config.model_dump()
    if ttl is not None:
        data["default_ttl_days"] = ttl
    if max_entries is not None:
        data["max_entries_per_employee"] = max_entries
    if half_life is not None:
        data["confidence_half_life_days"] = half_life

    store.memory_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    click.echo(f"配置已保存到 {config_path}")
