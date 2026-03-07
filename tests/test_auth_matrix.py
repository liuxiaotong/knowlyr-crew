"""Auth Matrix 守门测试 — 确保每个 API 端点的鉴权级别符合预期.

纯静态分析：读取 webhook.py 路由表 + webhook_handlers.py / webhook_skills.py handler 源码，
比对实际鉴权模式与期望矩阵，CI 自动检测新增端点是否配了正确的鉴权。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# 1. 路由提取：从 webhook.py 解析所有 Route 定义
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "crew"
_WEBHOOK_PY = _SRC_DIR / "webhook.py"
_HANDLERS_PY = _SRC_DIR / "webhook_handlers.py"
_SKILLS_PY = _SRC_DIR / "webhook_skills.py"


def _extract_routes() -> list[dict[str, Any]]:
    """从 webhook.py 提取所有 Route(path, endpoint=..., methods=[...]) 定义.

    返回 [{"path": str, "methods": list[str], "handler": str}, ...]
    """
    source = _WEBHOOK_PY.read_text(encoding="utf-8")
    routes: list[dict[str, Any]] = []

    # 分步提取：先找每个 Route( 块，再从块内提取 path/handler/methods
    # Route 块可能跨多行，用括号平衡来定位完整块
    route_starts = [m.start() for m in re.finditer(r"Route\(", source)]

    for start in route_starts:
        # 找到匹配的闭括号
        depth = 0
        end = start
        for i in range(start, len(source)):
            if source[i] == "(":
                depth += 1
            elif source[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        block = source[start:end]

        # 提取 path（第一个字符串参数）
        path_m = re.search(r'Route\(\s*"([^"]+)"', block)
        if not path_m:
            path_m = re.search(r'Route\(\s*f"([^"]+)"', block)
        if not path_m:
            continue
        path = path_m.group(1)

        # 提取 handler 名
        handler_m = re.search(
            r"endpoint\s*=\s*(?:_make_handler\s*\(\s*ctx\s*,\s*)?([\w]+)",
            block,
        )
        if not handler_m:
            continue
        handler = handler_m.group(1)

        # 提取 methods
        methods_m = re.search(r"methods\s*=\s*\[([^\]]+)\]", block)
        if not methods_m:
            continue
        methods_str = methods_m.group(1)
        methods = [s.strip().strip('"').strip("'") for s in methods_str.split(",")]

        for method in methods:
            method = method.strip()
            if method:
                routes.append({"path": path, "method": method, "handler": handler})

    return routes


# ---------------------------------------------------------------------------
# 2. Handler 鉴权检测：从源码判断每个 handler 的鉴权模式
# ---------------------------------------------------------------------------


def _detect_auth_level(handler_name: str) -> str:
    """检测 handler 的鉴权模式.

    返回值:
        "admin_required"    — _require_admin_token 且立即 return 403
        "admin_conditional" — _require_admin_token 但不立即 return（条件判断）
        "tenant_explicit"   — request.state.tenant 显式检查
        "skip_auth"         — 不需要 Bearer token（health, webhook 等）
        "bearer_only"       — 其他（仅依赖中间件的 Bearer token）
    """
    # 读取 handler 源码
    source = _get_handler_source(handler_name)
    if source is None:
        return "bearer_only"  # 找不到源码，保守标记

    # 检测 _require_admin_token 调用
    if "_require_admin_token" in source:
        # 模式 1: `is_admin = _require_admin_token(request) is None` → 仅字段过滤，不门控
        # 这种用法是 bearer_only（admin 信息仅用于决定返回字段范围）
        _has_gate = False
        has_field_filter_only = True

        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "_require_admin_token" not in line:
                continue

            # 字段过滤模式：`is_admin = _require_admin_token(...) is None`
            if "is None" in line or "is not None" in line:
                continue  # 这行不算门控

            has_field_filter_only = False

            # 查找后续 5 行是否有直接 return 403 模式
            context_after = "\n".join(lines[i : i + 5])
            if re.search(r"if\s+admin_err\s*:\s*\n\s*return\s+", context_after):
                _has_gate = True
                return "admin_required"
            else:
                return "admin_conditional"

        if has_field_filter_only:
            pass  # 继续往下检测其他模式，最终返回 bearer_only

    # 检测 _require_admin_tenant（租户管理 CRUD 用的独立模式）
    if "_require_admin_tenant" in source:
        return "bearer_only"  # 走 Bearer 中间件 + 自行检查 is_admin

    # 检测 request.state.tenant 显式检查
    if "request.state" in source and "tenant" in source:
        # 排除 get_current_tenant（这是通用辅助函数）
        if re.search(r"request\s*\.\s*state\s*\.\s*tenant", source):
            # 但如果只是 getattr(request, "state", None) 也算
            return "tenant_explicit"

    return "bearer_only"


def _get_handler_source(handler_name: str) -> str | None:
    """从 webhook_handlers.py 或 webhook_skills.py 读取指定 handler 的源码."""
    for filepath in [_HANDLERS_PY, _SKILLS_PY]:
        if not filepath.exists():
            continue
        source = filepath.read_text(encoding="utf-8")

        # 查找函数定义
        pattern = re.compile(
            rf"^(async\s+)?def\s+{re.escape(handler_name)}\s*\(",
            re.MULTILINE,
        )
        match = pattern.search(source)
        if not match:
            continue

        # 提取函数体：从 def 行开始，到下一个同缩进的 def 或文件结尾
        start = match.start()
        # 查找下一个顶层函数定义
        next_func = re.search(r"\n(?:async\s+)?def\s+\w+\s*\(", source[match.end() :])
        if next_func:
            end = match.end() + next_func.start()
        else:
            end = len(source)

        return source[start:end]

    return None


# ---------------------------------------------------------------------------
# 3. 期望鉴权矩阵（硬编码 — 每次新增端点必须在此注册）
# ---------------------------------------------------------------------------

# key = (path, method), value = expected auth level
EXPECTED_AUTH_MATRIX: dict[tuple[str, str], str] = {
    # ── skip_auth（middleware skip_paths，无需 Bearer）──
    ("/health", "GET"): "skip_auth",
    ("/metrics", "GET"): "skip_auth",
    ("/webhook/github", "POST"): "skip_auth",
    ("/webhook/openclaw", "POST"): "skip_auth",
    ("/webhook", "POST"): "skip_auth",
    ("/feishu/event", "POST"): "skip_auth",
    ("/wecom/event/{app_id}", "GET"): "skip_auth",
    ("/wecom/event/{app_id}", "POST"): "skip_auth",
    # ── admin_required（L2 硬门控）──
    ("/api/employees", "POST"): "admin_required",
    ("/api/employees/{identifier}", "PUT"): "admin_required",
    ("/api/employees/{identifier}", "DELETE"): "admin_required",
    ("/api/employees/{identifier}/prompt", "GET"): "admin_required",
    ("/api/employees/{identifier}/authority/restore", "POST"): "admin_required",
    ("/api/memory/{entry_id}", "DELETE"): "admin_required",
    ("/api/memory/search", "GET"): "admin_required",
    ("/api/memory/org", "GET"): "admin_required",
    ("/api/memory/batch/update", "POST"): "admin_required",
    ("/api/memory/batch/delete", "POST"): "admin_required",
    ("/api/memory/drafts", "GET"): "admin_required",
    ("/api/memory/drafts/{draft_id}", "GET"): "admin_required",
    ("/api/memory/drafts/{draft_id}/approve", "POST"): "admin_required",
    ("/api/memory/drafts/{draft_id}/reject", "POST"): "admin_required",
    ("/api/memory/shared/stats", "GET"): "admin_required",
    ("/api/memory/dashboard", "GET"): "admin_required",
    ("/api/knowledge/dashboard", "GET"): "admin_required",
    ("/api/memory/recall-feedback", "POST"): "bearer_only",
    ("/api/memory/consolidate", "POST"): "admin_required",
    ("/api/soul/review", "POST"): "admin_required",
    ("/api/soul/candidates", "GET"): "admin_required",
    ("/api/soul/approve", "POST"): "admin_required",
    ("/api/soul/reject", "POST"): "admin_required",
    ("/api/memory/feedback/{memory_id}", "GET"): "admin_required",
    ("/api/memory/feedback/summary", "GET"): "admin_required",
    ("/api/memory/usage/stats/{memory_id}", "GET"): "admin_required",
    ("/api/memory/usage/low-quality", "GET"): "admin_required",
    ("/api/memory/usage/popular", "GET"): "admin_required",
    ("/api/trajectory/annotations", "GET"): "admin_required",
    ("/api/trajectory/annotations", "POST"): "admin_required",
    ("/api/trajectory/export", "POST"): "admin_required",
    ("/api/cost/summary", "GET"): "admin_required",
    ("/api/project/status", "GET"): "admin_required",
    ("/api/permissions", "GET"): "admin_required",
    ("/api/permissions/respond", "POST"): "admin_required",
    ("/api/permission-matrix", "GET"): "admin_required",
    ("/api/work-log", "GET"): "admin_required",
    ("/api/souls", "GET"): "admin_required",
    ("/api/souls/{employee_name}", "GET"): "admin_required",
    ("/api/souls/{employee_name}", "PUT"): "admin_required",
    ("/api/config/discussions", "POST"): "admin_required",
    ("/api/config/discussions/{name}", "PUT"): "admin_required",
    ("/api/config/pipelines", "GET"): "admin_required",
    ("/api/config/pipelines", "POST"): "admin_required",
    ("/api/config/pipelines/{name}", "GET"): "admin_required",
    ("/api/config/pipelines/{name}", "PUT"): "admin_required",
    ("/api/discussions", "GET"): "admin_required",
    ("/api/discussions/{name}/plan", "GET"): "admin_required",
    ("/api/discussions/{name}/prompt", "GET"): "admin_required",
    ("/api/meetings", "GET"): "admin_required",
    ("/api/meetings/{meeting_id}", "GET"): "admin_required",
    ("/api/decisions/track", "POST"): "admin_required",
    ("/api/decisions/{decision_id}/evaluate", "POST"): "admin_required",
    ("/api/evaluate/scan", "POST"): "admin_required",
    ("/api/pipelines", "GET"): "admin_required",
    ("/api/kv/", "GET"): "admin_required",
    ("/api/kv/{key:path}", "GET"): "admin_required",
    ("/api/kv/{key:path}", "PUT"): "admin_required",
    ("/api/wiki/files/{file_id:int}", "DELETE"): "admin_required",
    ("/run/pipeline/{name}", "POST"): "admin_required",
    ("/run/route/{name}", "POST"): "admin_required",
    ("/agent/run/{name}", "POST"): "admin_required",
    # ── admin_conditional（任务端点 — owner 存在时检查 owner，否则要求 admin）──
    ("/tasks/{task_id}", "GET"): "admin_conditional",
    ("/tasks/{task_id}/replay", "POST"): "admin_conditional",
    ("/api/tasks/{task_id}/approve", "POST"): "admin_conditional",
    # ── tenant_explicit（L1 显式检查）──
    ("/api/trajectory/report", "POST"): "tenant_explicit",
    # ── bearer_only（L0 Bearer token）──
    ("/api/team/agents", "GET"): "bearer_only",
    ("/api/employees", "GET"): "bearer_only",
    ("/api/employees/copy", "POST"): "bearer_only",
    ("/api/employees/{identifier}", "GET"): "bearer_only",
    ("/api/employees/{identifier}/state", "GET"): "bearer_only",
    ("/api/memory/add", "POST"): "bearer_only",
    ("/api/memory/query", "GET"): "bearer_only",
    ("/api/memory/update", "PUT"): "bearer_only",
    ("/api/memory/ingest", "POST"): "bearer_only",
    ("/api/memory/tags", "GET"): "bearer_only",
    ("/api/memory/tags/suggest", "GET"): "bearer_only",
    ("/api/memory/tags/search", "GET"): "bearer_only",
    ("/api/memory/archive", "GET"): "bearer_only",
    ("/api/memory/archive/restore", "POST"): "admin_required",
    ("/api/memory/archive/stats", "GET"): "bearer_only",
    ("/api/memory/shared", "GET"): "bearer_only",
    ("/api/memory/shared/usage", "POST"): "admin_required",
    ("/api/memory/semantic/search", "POST"): "bearer_only",
    ("/api/memory/semantic/recommend", "POST"): "bearer_only",
    ("/api/memory/semantic/similar/{memory_id}", "GET"): "bearer_only",
    ("/api/memory/feedback", "POST"): "admin_required",
    ("/api/memory/usage/record", "POST"): "admin_required",
    ("/api/chat", "POST"): "bearer_only",
    ("/api/model-tiers", "GET"): "bearer_only",
    ("/api/audit/trends", "GET"): "bearer_only",
    ("/run/employee/{name}", "POST"): "bearer_only",
    ("/cron/status", "GET"): "bearer_only",
    ("/api/config/discussions", "GET"): "bearer_only",
    ("/api/config/discussions/{name}", "GET"): "bearer_only",
    ("/api/tenants", "POST"): "bearer_only",
    ("/api/tenants", "GET"): "bearer_only",
    ("/api/tenants/{tenant_id}", "GET"): "bearer_only",
    ("/api/tenants/{tenant_id}", "DELETE"): "bearer_only",
    ("/api/tenants/{tenant_id}", "PATCH"): "bearer_only",
    ("/api/wiki/spaces", "GET"): "bearer_only",
    # Skills API 端点
    ("/api/employees/{employee_name}/skills", "POST"): "bearer_only",
    ("/api/employees/{employee_name}/skills", "GET"): "bearer_only",
    ("/api/employees/{employee_name}/skills/{skill_name}", "GET"): "bearer_only",
    ("/api/employees/{employee_name}/skills/{skill_name}", "PUT"): "bearer_only",
    ("/api/employees/{employee_name}/skills/{skill_name}", "DELETE"): "bearer_only",
    ("/api/skills/check-triggers", "POST"): "bearer_only",
    ("/api/skills/execute", "POST"): "bearer_only",
    ("/api/skills/stats", "GET"): "bearer_only",
    ("/api/skills/trigger-history", "GET"): "bearer_only",
}


# ---------------------------------------------------------------------------
# 4. 测试
# ---------------------------------------------------------------------------


class TestAuthMatrix:
    """Auth Matrix 自动化守门测试."""

    def _get_routes(self) -> list[dict[str, Any]]:
        """提取路由表（缓存避免重复读文件）."""
        return _extract_routes()

    def test_route_extraction_not_empty(self):
        """路由表应该能提取出端点."""
        routes = self._get_routes()
        assert len(routes) > 50, f"只提取到 {len(routes)} 条路由，预期 > 50"

    def test_auth_levels_match_expectations(self):
        """每个端点的实际鉴权级别应与期望矩阵一致."""
        routes = self._get_routes()
        mismatches: list[str] = []

        for route in routes:
            key = (route["path"], route["method"])
            if key not in EXPECTED_AUTH_MATRIX:
                continue  # 新增端点检测在另一个测试中处理

            expected = EXPECTED_AUTH_MATRIX[key]
            handler_name = route["handler"]

            # skip_auth 端点不需要检测 handler 内部鉴权
            if expected == "skip_auth":
                continue

            actual = _detect_auth_level(handler_name)
            if actual != expected:
                mismatches.append(
                    f"  {route['method']:6s} {route['path']}\n"
                    f"    handler: {handler_name}\n"
                    f"    expected: {expected}\n"
                    f"    actual:   {actual}"
                )

        if mismatches:
            detail = "\n".join(mismatches)
            pytest.fail(
                f"Auth 级别不匹配 ({len(mismatches)} 个端点):\n{detail}\n\n"
                "如果鉴权级别已有意变更，请同步更新 EXPECTED_AUTH_MATRIX。"
            )

    def test_no_unregistered_routes(self):
        """路由表中的每个端点都必须注册到期望矩阵中.

        新增端点时 CI 会在此失败，提醒开发者补充鉴权矩阵。
        """
        routes = self._get_routes()
        unregistered: list[str] = []

        for route in routes:
            key = (route["path"], route["method"])
            # 跳过动态飞书 bot 路由（/feishu/event/{bot_id}）
            if route["path"].startswith("/feishu/event/") and route["path"] != "/feishu/event":
                continue
            if key not in EXPECTED_AUTH_MATRIX:
                actual = _detect_auth_level(route["handler"])
                unregistered.append(
                    f"  {route['method']:6s} {route['path']}  "
                    f"(handler: {route['handler']}, detected: {actual})"
                )

        if unregistered:
            detail = "\n".join(unregistered)
            pytest.fail(
                f"发现 {len(unregistered)} 个未注册到 EXPECTED_AUTH_MATRIX 的端点:\n{detail}\n\n"
                "请在 tests/test_auth_matrix.py 的 EXPECTED_AUTH_MATRIX 中注册这些端点，"
                "并确认其鉴权级别正确。"
            )

    def test_no_stale_matrix_entries(self):
        """矩阵中不应有已从路由表删除的端点（warn 级别）."""
        routes = self._get_routes()
        route_keys = set()
        for route in routes:
            route_keys.add((route["path"], route["method"]))

        stale: list[str] = []
        for key in EXPECTED_AUTH_MATRIX:
            # 动态飞书路由不在静态路由表中，跳过
            if key not in route_keys:
                stale.append(f"  {key[1]:6s} {key[0]}")

        if stale:
            import warnings

            detail = "\n".join(stale)
            warnings.warn(
                f"EXPECTED_AUTH_MATRIX 中有 {len(stale)} 个端点在路由表中不存在 "
                f"（可能已删除）:\n{detail}",
                stacklevel=1,
            )

    def test_admin_required_handlers_have_403(self):
        """所有标记为 admin_required 的端点，handler 中必须包含 403 返回."""
        routes = self._get_routes()
        missing_403: list[str] = []

        for route in routes:
            key = (route["path"], route["method"])
            expected = EXPECTED_AUTH_MATRIX.get(key)
            if expected != "admin_required":
                continue

            source = _get_handler_source(route["handler"])
            if source is None:
                missing_403.append(f"  {route['method']:6s} {route['path']} — 找不到 handler 源码")
                continue

            if "403" not in source:
                missing_403.append(
                    f"  {route['method']:6s} {route['path']} — "
                    f"handler {route['handler']} 中未找到 403"
                )

        if missing_403:
            detail = "\n".join(missing_403)
            pytest.fail(f"admin_required 端点中有 {len(missing_403)} 个缺少 403 返回:\n{detail}")

    def test_bearer_only_handlers_no_admin_gate(self):
        """bearer_only 端点的 handler 不应有 _require_admin_token 硬门控.

        注意：有些 bearer_only handler 用 _require_admin_token 做"字段过滤"
        （如 employee_list 根据 admin 返回不同字段），这种用法不算硬门控。
        检测方式：如果 handler 用了 _require_admin_token 且后面紧跟 if ... return 403，
        则说明是硬门控，应标记为 admin_required。
        """
        routes = self._get_routes()
        false_bearer: list[str] = []

        for route in routes:
            key = (route["path"], route["method"])
            expected = EXPECTED_AUTH_MATRIX.get(key)
            if expected != "bearer_only":
                continue

            actual = _detect_auth_level(route["handler"])
            if actual == "admin_required":
                false_bearer.append(
                    f"  {route['method']:6s} {route['path']} — "
                    f"handler {route['handler']} 检测到 admin_required 硬门控"
                )

        if false_bearer:
            detail = "\n".join(false_bearer)
            pytest.fail(
                f"bearer_only 端点中有 {len(false_bearer)} 个实际含 admin 硬门控:\n{detail}\n\n"
                "如果鉴权级别已升级，请更新 EXPECTED_AUTH_MATRIX。"
            )

    def test_matrix_coverage_stats(self):
        """输出 auth matrix 覆盖统计."""
        routes = self._get_routes()
        total = len(routes)
        registered = sum(1 for r in routes if (r["path"], r["method"]) in EXPECTED_AUTH_MATRIX)

        level_counts: dict[str, int] = {}
        for level in EXPECTED_AUTH_MATRIX.values():
            level_counts[level] = level_counts.get(level, 0) + 1

        print("\n--- Auth Matrix 覆盖统计 ---")
        print(f"路由总数: {total}")
        print(f"矩阵注册: {registered}")
        for level, count in sorted(level_counts.items()):
            print(f"  {level}: {count}")

        # 不是硬失败，仅输出信息
        assert registered > 0, "矩阵中没有任何注册端点"
