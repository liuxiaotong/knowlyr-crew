#!/usr/bin/env python3
"""AI 员工批量评估与 Prompt 优化工具.

用法:
    python scripts/crew_eval.py snapshot --description "基线"
    python scripts/crew_eval.py snapshot --list
    python scripts/crew_eval.py snapshot --diff v20260216a v20260217a
    python scripts/crew_eval.py convert --employee ceo-assistant
    python scripts/crew_eval.py score --rule-only
    python scripts/crew_eval.py score --sample 30 --model kimi-k2.5
    python scripts/crew_eval.py report
    python scripts/crew_eval.py compare <run1> <run2>
    python scripts/crew_eval.py run --employee ceo-assistant --sample 30 --snapshot
    python scripts/crew_eval.py archive --dry-run
"""

from __future__ import annotations

import json
import logging
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

import click

# 项目路径
CREW_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = CREW_ROOT / ".crew" / "sessions"
ARCHIVE_DIR = CREW_ROOT / ".crew" / "archive"
EVALS_DIR = CREW_ROOT / ".crew" / "evals"
TRAJECTORIES_FILE = EVALS_DIR / "trajectories.jsonl"
RUNS_DIR = EVALS_DIR / "runs"
REPORTS_DIR = EVALS_DIR / "reports"
BASELINES_DIR = EVALS_DIR / "baselines"

# 确保 crew 模块可导入
sys.path.insert(0, str(CREW_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── 员工-评估域映射 ──────────────────────────────────────────────────
# 33 个员工按角色类型映射到评估域。评估域决定使用哪套 rubric + judge prompt。

EMPLOYEE_DOMAIN_MAP: dict[str, str] = {
    # conversation — 对话/协调/沟通类
    "ceo-assistant": "conversation",
    "customer-success": "conversation",
    "community-operator": "conversation",

    # engineering — 工程/技术类
    "code-reviewer": "engineering",
    "backend-engineer": "engineering",
    "frontend-engineer": "engineering",
    "test-engineer": "engineering",
    "e2e-tester": "engineering",
    "devops-engineer": "engineering",
    "dba": "engineering",
    "security-auditor": "engineering",
    "debug-expert": "engineering",
    "performance-optimizer": "engineering",
    "refactor-guide": "engineering",
    "pr-creator": "engineering",
    "doc-writer": "engineering",
    "data-engineer": "engineering",
    "mlops-engineer": "engineering",
    "i18n-expert": "engineering",

    # advisory — 分析/顾问/研究类
    "product-manager": "advisory",
    "hr-manager": "advisory",
    "finance-expert": "advisory",
    "legal-counsel": "advisory",
    "bd-manager": "advisory",
    "ux-designer": "advisory",
    "api-designer": "advisory",
    "solutions-architect": "advisory",
    "algorithm-researcher": "advisory",
    "nlp-researcher": "advisory",
    "sociology-researcher": "advisory",
    "economics-researcher": "advisory",
    "data-quality-expert": "advisory",
    "benchmark-specialist": "advisory",
}

DEFAULT_DOMAIN = "conversation"


def _get_domain(employee: str) -> str:
    """获取员工对应的评估域."""
    return EMPLOYEE_DOMAIN_MAP.get(employee, DEFAULT_DOMAIN)


# ── 评分维度优化建议 ──────────────────────────────────────────────────

DIMENSION_SUGGESTIONS = {
    # conversation 域
    "relevance": [
        "在 prompt 中增加「先确认理解用户意图，再回答」的步骤",
        "添加「不确定时追问而非猜测」的指令",
    ],
    "completeness": [
        "增加「回答前先列出问题涉及的所有方面」的 checklist 思维",
        "添加「如果信息不足，主动列出需要补充的信息」的指令",
    ],
    "clarity": [
        "增加结构化输出的要求（分点、编号）",
        "限制回复长度范围，避免过长或过短",
    ],
    "actionability": [
        "在 prompt 中增加「回复必须包含至少一个具体的下一步行动」",
        "添加约束: 每个建议必须包含 who/what/when",
    ],
    "tone_fit": [
        "在 prompt 中增加更多语气示例（好的回复 vs 不好的回复）",
        "明确禁止的表达模式（如过度礼貌、generic AI 语气）",
    ],
    "non_redundancy": [
        "减少重复性表述",
        "避免对同一信息的多次阐述",
    ],
    # engineering 域
    "technical_accuracy": [
        "在 prompt 中增加「先验证再输出」的要求",
        "添加领域知识引用的要求",
    ],
    "tool_usage": [
        "明确工具选择优先级和参数规范",
        "添加「用最少的工具调用完成任务」的约束",
    ],
    "thoroughness": [
        "在 prompt 中增加「检查清单」式的审查流程",
        "要求输出前先列出需要检查的方面",
    ],
    # advisory 域
    "analysis_depth": [
        "在 prompt 中增加「必须追问到根因」的要求",
        "添加「5 Why 分析法」或类似深度分析框架",
    ],
    "recommendation_quality": [
        "要求每个建议包含具体的 who/what/when",
        "添加「建议必须可操作」的约束",
    ],
    "evidence_based": [
        "要求引用数据或案例来支撑结论",
        "添加「不能空泛断言」的约束",
    ],
    "risk_awareness": [
        "在 prompt 中增加「识别风险和不确定性」的步骤",
    ],
    "stakeholder_consideration": [
        "要求考虑不同利益方的视角",
    ],
    # discussion 域
    "substantive_contribution": [
        "在 prompt 中要求「提供新信息而非重复已知」",
    ],
    "engagement_quality": [
        "要求「直接回应上一位发言者的观点」",
    ],
    "professional_depth": [
        "要求「体现本职专业知识」",
    ],
    "constructive_challenge": [
        "鼓励「提出补充或质疑，不要一味附和」",
    ],
}


@click.group()
def cli():
    """AI 员工评估工具."""
    pass


# ── snapshot 命令 ─────────────────────────────────────────────────────

@cli.command()
@click.option("--description", "-d", default="", help="快照说明")
@click.option("--list", "list_all", is_flag=True, help="列出所有快照")
@click.option("--diff", "diff_versions", nargs=2, default=None, help="对比两个快照版本")
def snapshot(description, list_all, diff_versions):
    """管理数据快照 — 记录 prompt 状态和数据清单."""
    from crew.snapshot import SnapshotManager

    sm = SnapshotManager(CREW_ROOT)

    if list_all:
        snaps = sm.list_snapshots()
        if not snaps:
            click.echo("还没有任何快照。用 snapshot -d '说明' 创建第一个。")
            return
        click.echo(f"共 {len(snaps)} 个快照:\n")
        for s in snaps:
            emp_count = len(s.get("employees", {}))
            organic = s.get("data_counts", {}).get("sessions_organic", "?")
            click.echo(
                f"  {s['version']}  {s['created_at'][:16]}  "
                f"{emp_count} 个员工  {organic} organic sessions"
            )
            if s.get("description"):
                click.echo(f"    └ {s['description']}")
        return

    if diff_versions:
        v_a, v_b = diff_versions
        try:
            result = sm.diff(v_a, v_b)
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

        click.echo(f"对比: {v_a} → {v_b}\n")

        # prompt 变更
        changes = result["prompt_changes"]
        if changes:
            click.echo(f"Prompt 变更 ({len(changes)} 个员工):")
            click.echo("| 员工 | 版本 | Hash |")
            click.echo("|------|------|------|")
            for c in changes:
                click.echo(
                    f"| {c['employee']} ({c['display_name']}) "
                    f"| {c['version_a']} → {c['version_b']} "
                    f"| {c['hash_a'][:6]} → {c['hash_b'][:6]} |"
                )
        else:
            click.echo("Prompt 无变更。")

        click.echo("")

        # 数据变化
        data_changes = result["data_changes"]
        if data_changes:
            click.echo("数据变化:")
            for key, (va, vb) in data_changes.items():
                diff_val = vb - va
                click.echo(f"  {key}: {va} → {vb} ({diff_val:+d})")

        # 新增/删除员工
        if result["new_employees"]:
            click.echo(f"\n新增员工: {', '.join(result['new_employees'])}")
        if result["removed_employees"]:
            click.echo(f"\n删除员工: {', '.join(result['removed_employees'])}")
        return

    # 默认: 创建快照
    version = sm.create(description)
    snap = sm._load_manifest(version)
    emp_count = len(snap.get("employees", {}))
    organic = snap.get("data_counts", {}).get("sessions_organic", 0)
    click.echo(f"快照已创建: {version}")
    click.echo(f"  {emp_count} 个员工 prompt 已保存")
    click.echo(f"  {organic} organic sessions 已记录")
    if description:
        click.echo(f"  说明: {description}")


# ── convert 命令 ──────────────────────────────────────────────────────

@cli.command()
@click.option("--employee", help="只转换特定员工 (如 ceo-assistant)")
@click.option("--since", help="只转换该日期之后的 session (如 20260215)")
@click.option("--limit", type=int, help="最多转换 N 个")
@click.option(
    "--origin", default="organic",
    help="来源过滤: organic=真实对话(默认), synthetic=程序生成, all=不过滤",
)
@click.option("--output", default=str(TRAJECTORIES_FILE), help="输出文件路径")
def convert(employee, since, limit, origin, output):
    """Session → Trajectory 批量转换."""
    from crew.session_converter import convert_sessions_batch

    trajectories = convert_sessions_batch(
        SESSIONS_DIR,
        employee=employee,
        since=since,
        limit=limit,
        origin=origin,
    )

    if not trajectories:
        logger.warning("没有可转换的 session")
        return

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    # 统计
    employees = {}
    for t in trajectories:
        emp = t.get("metadata", {}).get("employee", "unknown")
        employees[emp] = employees.get(emp, 0) + 1

    click.echo(f"转换完成: {len(trajectories)} 条 → {output_path}")
    for emp, n in sorted(employees.items(), key=lambda x: -x[1]):
        domain = _get_domain(emp)
        click.echo(f"  {emp}: {n} 条 (domain={domain})")


# ── score 命令 ────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_file", default=str(TRAJECTORIES_FILE), help="输入 trajectory 文件")
@click.option("--rule-only", is_flag=True, help="仅用规则层评分 (0 API 成本)")
@click.option("--sample", type=int, help="随机采样 N 条评分")
@click.option("--model", "model_name", default="kimi-k2.5", help="LLM judge 模型")
@click.option("--provider", default="openai", help="LLM 提供商 (openai/anthropic)")
@click.option("--base-url", help="OpenAI 兼容 API 的 base_url")
@click.option("--api-key", help="LLM API 密钥 (也可用环境变量)")
@click.option("--output", "output_file", help="输出文件路径 (默认自动生成)")
@click.option("--snapshot-version", help="关联的快照版本 (自动填入结果中)")
def score(input_file, rule_only, sample, model_name, provider, base_url, api_key, output_file,
          snapshot_version):
    """批量评分 — 调用 RewardEngine."""
    try:
        from agentreward.config import RewardConfig
        from agentreward.reward import RewardEngine
    except ImportError:
        logger.error("需要安装 knowlyr-reward: pip install -e packages/reward")
        sys.exit(1)

    # 自动检测快照
    if not snapshot_version:
        try:
            from crew.snapshot import SnapshotManager
            sm = SnapshotManager(CREW_ROOT)
            snapshot_version = sm.find_matching_snapshot()
            if snapshot_version:
                click.echo(f"自动关联快照: {snapshot_version}")
        except Exception:
            pass

    # 加载 trajectories
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error("文件不存在: %s (先运行 convert 命令)", input_path)
        sys.exit(1)

    trajectories = []
    for line in input_path.read_text("utf-8").strip().split("\n"):
        if line.strip():
            trajectories.append(json.loads(line))

    if not trajectories:
        logger.warning("没有 trajectory 数据")
        return

    # 采样
    if sample and sample < len(trajectories):
        trajectories = random.sample(trajectories, sample)
        click.echo(f"采样 {sample} 条进行评分")

    # 按员工分组评分 — 每个员工使用对应的 domain
    results = []
    for i, traj in enumerate(trajectories):
        emp = traj.get("metadata", {}).get("employee", "")
        domain = _get_domain(emp)

        click.echo(f"评分 [{i+1}/{len(trajectories)}] ({domain}): {traj['task'][:50]}...")

        if rule_only:
            config = RewardConfig(
                rule_weight=1.0,
                model_weight=0.0,
                domain=domain,
            )
        else:
            config = RewardConfig(
                rule_weight=0.3,
                model_weight=0.7,
                domain=domain,
                model_name=model_name,
                provider=provider,
                base_url=base_url,
                api_key=api_key,
            )

        engine = RewardEngine(config=config)

        try:
            reward = engine.score(traj)
        except Exception as e:
            logger.warning("评分失败: %s — %s", traj["task"][:30], e)
            continue

        # 提取维度分数
        rubric_scores = {}
        if reward.step_rewards:
            rubric_scores = reward.step_rewards[0].rubric_scores

        result = {
            "session_id": traj.get("metadata", {}).get("session_id", ""),
            "employee": emp,
            "domain": domain,
            "model": traj.get("metadata", {}).get("model", ""),
            "task": traj["task"],
            "total_score": round(reward.total_score, 4),
            "outcome_score": round(reward.outcome_score, 4),
            "process_score": round(reward.process_score, 4),
            "rubric_scores": {k: round(v, 4) for k, v in rubric_scores.items()},
            "response_preview": traj["steps"][0]["output"][:200] if traj["steps"] else "",
            "scored_at": datetime.now().isoformat(),
        }
        if snapshot_version:
            result["snapshot_version"] = snapshot_version
        results.append(result)

    # 输出
    if not output_file:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = str(RUNS_DIR / f"{timestamp}.jsonl")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    click.echo(f"\n评分完成: {len(results)} 条 → {output_path}")
    if snapshot_version:
        click.echo(f"快照版本: {snapshot_version}")

    # 摘要
    if results:
        scores = [r["total_score"] for r in results]
        click.echo(f"总分: min={min(scores):.4f}, max={max(scores):.4f}, avg={statistics.mean(scores):.4f}")


# ── report 命令 ───────────────────────────────────────────────────────

@cli.command()
@click.option("--run", "run_file", help="评分结果文件路径 (默认最新)")
@click.option("--output", "output_file", help="报告输出路径")
def report(run_file, output_file):
    """生成评估报告."""
    # 找到评分文件
    if run_file:
        run_path = Path(run_file)
    else:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        runs = sorted(RUNS_DIR.glob("*.jsonl"))
        if not runs:
            logger.error("没有评分结果 (先运行 score 命令)")
            sys.exit(1)
        run_path = runs[-1]

    results = []
    for line in run_path.read_text("utf-8").strip().split("\n"):
        if line.strip():
            results.append(json.loads(line))

    if not results:
        logger.warning("评分结果为空")
        return

    # 提取快照版本（如果有）
    snapshot_ver = None
    for r in results:
        if r.get("snapshot_version"):
            snapshot_ver = r["snapshot_version"]
            break

    # 按员工分组
    by_employee: dict[str, list[dict]] = {}
    for r in results:
        emp = r.get("employee", "unknown")
        by_employee.setdefault(emp, []).append(r)

    # 生成报告
    report_lines = [
        "# AI 员工评估报告",
        "",
        f"评分文件: `{run_path.name}`",
        f"评估总数: {len(results)} 条",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]
    if snapshot_ver:
        report_lines.append(f"快照版本: {snapshot_ver}")
    report_lines.append("")

    for emp, emp_results in sorted(by_employee.items()):
        domain = emp_results[0].get("domain", _get_domain(emp))
        report_lines.extend(_generate_employee_report(emp, emp_results, domain))

    report_text = "\n".join(report_lines)

    # 输出
    if output_file:
        output_path = Path(output_file)
    else:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = REPORTS_DIR / f"{timestamp}.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")

    click.echo(report_text)
    click.echo(f"\n报告已保存: {output_path}")


def _generate_employee_report(employee: str, results: list[dict], domain: str) -> list[str]:
    """为单个员工生成报告段落."""
    lines = []
    total_scores = [r["total_score"] for r in results]
    avg_total = statistics.mean(total_scores) if total_scores else 0

    lines.append(f"## {employee} ({domain})")
    lines.append(f"")
    lines.append(f"评估: {len(results)} 条 | 平均总分: {avg_total:.2f}")
    lines.append(f"")

    # 汇总各维度
    all_dims: dict[str, list[float]] = {}
    for r in results:
        for dim, s in r.get("rubric_scores", {}).items():
            all_dims.setdefault(dim, []).append(s)

    if all_dims:
        lines.append("| 维度 | 平均 | 最低 | 最高 | 判定 |")
        lines.append("|------|------|------|------|------|")

        weak_dims = []
        for dim in sorted(all_dims.keys()):
            scores = all_dims[dim]
            avg = statistics.mean(scores)
            lo = min(scores)
            hi = max(scores)
            verdict = "弱" if avg < 0.6 else ("一般" if avg < 0.75 else "OK")
            lines.append(f"| {dim} | {avg:.2f} | {lo:.2f} | {hi:.2f} | {verdict} |")
            if avg < 0.7:
                weak_dims.append((dim, avg))

        lines.append("")

        # 弱点建议
        if weak_dims:
            lines.append("### 优化建议")
            lines.append("")
            for dim, avg in sorted(weak_dims, key=lambda x: x[1]):
                suggestions = DIMENSION_SUGGESTIONS.get(dim, [])
                lines.append(f"**{dim}** (avg={avg:.2f}):")
                for s_text in suggestions:
                    lines.append(f"- {s_text}")
                if not suggestions:
                    lines.append(f"- 需要根据 {domain} 域的特点针对性优化")
                lines.append("")

    # 低分样本
    sorted_results = sorted(results, key=lambda r: r["total_score"])
    bottom = sorted_results[:3]
    if bottom:
        lines.append("### 低分样本")
        lines.append("")
        lines.append("| Task | 总分 | 弱维度 |")
        lines.append("|------|------|--------|")
        for r in bottom:
            task = r["task"][:40]
            weak = ", ".join(
                f"{k}={v:.2f}"
                for k, v in sorted(r.get("rubric_scores", {}).items(), key=lambda x: x[1])[:2]
            )
            lines.append(f"| {task} | {r['total_score']:.2f} | {weak} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    return lines


# ── compare 命令 ──────────────────────────────────────────────────────

@cli.command()
@click.argument("run_before")
@click.argument("run_after")
@click.option("--employee", help="只对比特定员工")
def compare(run_before, run_after, employee):
    """对比两次评分运行的结果."""
    def load_run(path_str):
        p = Path(path_str)
        if not p.exists():
            p = RUNS_DIR / (path_str + ".jsonl")
        if not p.exists():
            p = RUNS_DIR / path_str
        if not p.exists():
            logger.error("找不到评分文件: %s", path_str)
            sys.exit(1)
        data = []
        for line in p.read_text("utf-8").strip().split("\n"):
            if line.strip():
                data.append(json.loads(line))
        return data, p.name

    before, name_before = load_run(run_before)
    after, name_after = load_run(run_after)

    if employee:
        before = [r for r in before if r.get("employee") == employee]
        after = [r for r in after if r.get("employee") == employee]

    # 提取快照版本
    snap_before = next((r.get("snapshot_version", "") for r in before if r.get("snapshot_version")), "")
    snap_after = next((r.get("snapshot_version", "") for r in after if r.get("snapshot_version")), "")

    click.echo(f"对比: {name_before} ({len(before)}条) vs {name_after} ({len(after)}条)")
    if snap_before or snap_after:
        click.echo(f"快照: {snap_before or '?'} → {snap_after or '?'}")
    click.echo("")

    # 按维度聚合
    def aggregate_dims(data):
        dims = {}
        for r in data:
            for dim, s in r.get("rubric_scores", {}).items():
                dims.setdefault(dim, []).append(s)
        return {dim: statistics.mean(scores) for dim, scores in dims.items()}

    dims_before = aggregate_dims(before)
    dims_after = aggregate_dims(after)

    all_dims = set(dims_before.keys()) | set(dims_after.keys())

    click.echo("| 维度 | 优化前 | 优化后 | 变化 |")
    click.echo("|------|--------|--------|------|")

    for dim in sorted(all_dims):
        b = dims_before.get(dim, 0)
        a = dims_after.get(dim, 0)
        diff = a - b
        arrow = "↑" if diff > 0.01 else ("↓" if diff < -0.01 else "→")
        if abs(diff) > 0.1:
            arrow += arrow[0]
        click.echo(f"| {dim} | {b:.2f} | {a:.2f} | {diff:+.2f} {arrow} |")

    # 总分对比
    avg_before = statistics.mean([r["total_score"] for r in before]) if before else 0
    avg_after = statistics.mean([r["total_score"] for r in after]) if after else 0
    diff = avg_after - avg_before
    click.echo(f"")
    click.echo(f"总分变化: {avg_before:.2f} → {avg_after:.2f} ({diff:+.2f})")


# ── archive 命令 ──────────────────────────────────────────────────────

@cli.command()
@click.option("--dry-run", is_flag=True, help="只统计不移动，先看看会动多少文件")
def archive(dry_run):
    """归档合成数据 — 把训练/流水线/讨论数据从 sessions/ 移到 archive/.

    归档结构:
        .crew/archive/training/     ← 批量训练数据 (employee + cli.run)
        .crew/archive/pipelines/    ← 流水线执行记录
        .crew/archive/discussions/  ← AI 间讨论

    真实用户对话 (organic) 不动，留在 sessions/ 里。
    """
    from crew.session_converter import archive_sessions

    if dry_run:
        click.echo("预览模式 — 不会移动文件\n")

    counts = archive_sessions(SESSIONS_DIR, ARCHIVE_DIR, dry_run=dry_run)

    total = sum(counts.values())
    if total == 0:
        click.echo("没有需要归档的合成数据。")
        return

    click.echo(f"{'将归档' if dry_run else '已归档'}:")
    for category, n in sorted(counts.items(), key=lambda x: -x[1]):
        if n > 0:
            dest = ARCHIVE_DIR / category
            click.echo(f"  {category}: {n} 个 → {dest}/")

    click.echo(f"\n总计: {total} 个文件")
    if dry_run:
        click.echo("\n确认无误后去掉 --dry-run 执行归档。")


# ── run 命令（完整流水线）────────────────────────────────────────────

@cli.command()
@click.option("--employee", help="只评估特定员工")
@click.option("--sample", type=int, default=30, help="采样数量")
@click.option("--since", help="只取该日期之后的 session (如 20260216)")
@click.option("--rule-only", is_flag=True, help="仅用规则层评分")
@click.option("--model", "model_name", default="kimi-k2.5", help="LLM judge 模型")
@click.option("--provider", default="openai", help="LLM 提供商")
@click.option("--base-url", help="API base_url")
@click.option("--api-key", help="API 密钥")
@click.option(
    "--origin", default="organic",
    help="来源过滤: organic=真实对话(默认), synthetic=程序生成, all=不过滤",
)
@click.option("--snapshot/--no-snapshot", "do_snapshot", default=False,
              help="运行前自动打快照")
def run(employee, sample, since, rule_only, model_name, provider, base_url, api_key, origin,
        do_snapshot):
    """完整流水线: [snapshot →] convert → score → report."""
    from click.testing import CliRunner

    runner = CliRunner()

    # Step 0: 可选快照
    if do_snapshot:
        click.echo("=== Step 0: 创建快照 ===")
        result = runner.invoke(snapshot, ["-d", f"auto-eval {datetime.now().strftime('%Y%m%d')}"])
        click.echo(result.output)

    # Step 1: convert
    click.echo("=== Step 1: 转换 sessions ===")
    convert_args = ["--origin", origin]
    if employee:
        convert_args.extend(["--employee", employee])
    if since:
        convert_args.extend(["--since", since])
    result = runner.invoke(convert, convert_args)
    click.echo(result.output)
    if result.exit_code != 0:
        return

    # Step 2: score
    click.echo("=== Step 2: 评分 ===")
    score_args = []
    if rule_only:
        score_args.append("--rule-only")
    else:
        score_args.extend(["--model", model_name, "--provider", provider])
        if base_url:
            score_args.extend(["--base-url", base_url])
        if api_key:
            score_args.extend(["--api-key", api_key])
    if sample:
        score_args.extend(["--sample", str(sample)])
    result = runner.invoke(score, score_args)
    click.echo(result.output)
    if result.exit_code != 0:
        return

    # Step 3: report
    click.echo("=== Step 3: 生成报告 ===")
    result = runner.invoke(report, [])
    click.echo(result.output)


if __name__ == "__main__":
    cli()
