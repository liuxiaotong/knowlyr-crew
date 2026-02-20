"""员工权限系统测试."""

from crew.models import Employee, PermissionPolicy
from crew.permission import PermissionDenied, PermissionGuard
from crew.tool_schema import (
    TOOL_ROLE_PRESETS,
    resolve_effective_tools,
    validate_permissions,
)


def _make_employee(
    name: str = "test-emp",
    tools: list[str] | None = None,
    permissions: PermissionPolicy | None = None,
) -> Employee:
    """测试用员工工厂."""
    return Employee(
        name=name,
        description="测试员工",
        body="测试正文",
        tools=tools or [],
        permissions=permissions,
    )


# ── resolve_effective_tools ──


class TestResolveEffectiveTools:
    def test_no_permissions_uses_tools(self):
        """无 permissions 时使用 tools 原样."""
        emp = _make_employee(tools=["file_read", "bash", "git"])
        result = resolve_effective_tools(emp)
        assert result == {"file_read", "bash", "git"}

    def test_role_expansion(self):
        """角色正确展开."""
        emp = _make_employee(
            tools=["file_read", "grep", "glob", "bash"],
            permissions=PermissionPolicy(roles=["readonly"]),
        )
        result = resolve_effective_tools(emp)
        # readonly = {file_read, grep, glob}，与 tools 取交集
        assert result == {"file_read", "grep", "glob"}

    def test_deny_overrides_role(self):
        """deny 优先级最高."""
        emp = _make_employee(
            tools=["file_read", "grep", "glob", "git"],
            permissions=PermissionPolicy(roles=["readonly"], deny=["grep"]),
        )
        result = resolve_effective_tools(emp)
        assert "grep" not in result
        assert result == {"file_read", "glob"}

    def test_allow_adds_extra(self):
        """allow 追加工具."""
        emp = _make_employee(
            tools=["file_read", "bash"],
            permissions=PermissionPolicy(roles=["readonly"], allow=["bash"]),
        )
        result = resolve_effective_tools(emp)
        assert "bash" in result
        assert result == {"file_read", "bash"}

    def test_tools_intersection(self):
        """与 tools 声明取交集 — 角色展开后超出 tools 的不生效."""
        emp = _make_employee(
            tools=["file_read"],
            permissions=PermissionPolicy(roles=["developer"]),
        )
        # developer 包含 file_read, file_write, bash, git, grep, glob
        # 但 tools 只声明了 file_read
        result = resolve_effective_tools(emp)
        assert result == {"file_read"}

    def test_empty_tools_returns_all_from_roles(self):
        """tools 为空时，角色展开结果全部生效."""
        emp = _make_employee(
            tools=[],
            permissions=PermissionPolicy(roles=["readonly"]),
        )
        result = resolve_effective_tools(emp)
        assert result == {"file_read", "grep", "glob"}

    def test_multiple_roles_union(self):
        """多个角色取并集."""
        emp = _make_employee(
            tools=["file_read", "grep", "glob", "github_prs", "github_issues"],
            permissions=PermissionPolicy(roles=["readonly", "github"]),
        )
        result = resolve_effective_tools(emp)
        assert result == {"file_read", "grep", "glob", "github_prs", "github_issues"}

    def test_unknown_role_ignored(self):
        """未知角色安全忽略."""
        emp = _make_employee(
            tools=["file_read", "bash"],
            permissions=PermissionPolicy(roles=["nonexistent", "readonly"]),
        )
        result = resolve_effective_tools(emp)
        # nonexistent 被忽略，readonly 正常展开
        assert "file_read" in result

    def test_deny_allow_combined(self):
        """deny 从 allow 中移除."""
        emp = _make_employee(
            tools=["file_read", "bash", "git"],
            permissions=PermissionPolicy(
                allow=["file_read", "bash", "git"],
                deny=["git"],
            ),
        )
        result = resolve_effective_tools(emp)
        assert result == {"file_read", "bash"}


# ── validate_permissions ──


class TestValidatePermissions:
    def test_no_permissions_no_warnings(self):
        emp = _make_employee()
        assert validate_permissions(emp) == []

    def test_warns_unknown_role(self):
        emp = _make_employee(
            permissions=PermissionPolicy(roles=["nonexistent"]),
        )
        warnings = validate_permissions(emp)
        assert any("未知角色" in w and "nonexistent" in w for w in warnings)

    def test_warns_unknown_tool_in_allow(self):
        emp = _make_employee(
            permissions=PermissionPolicy(allow=["totally_fake_tool"]),
        )
        warnings = validate_permissions(emp)
        assert any("未知工具" in w and "totally_fake_tool" in w for w in warnings)

    def test_warns_unknown_tool_in_deny(self):
        emp = _make_employee(
            permissions=PermissionPolicy(deny=["totally_fake_tool"]),
        )
        warnings = validate_permissions(emp)
        assert any("未知工具" in w and "totally_fake_tool" in w for w in warnings)

    def test_warns_denied_from_tools(self):
        """声明在 tools 中但被 deny 排除的工具产生警告."""
        emp = _make_employee(
            tools=["file_read", "git"],
            permissions=PermissionPolicy(roles=["readonly"], deny=["git"]),
        )
        warnings = validate_permissions(emp)
        assert any("git" in w and "排除" in w for w in warnings)

    def test_valid_config_no_extra_warnings(self):
        """合法配置不产生额外警告."""
        emp = _make_employee(
            tools=["file_read", "grep", "glob"],
            permissions=PermissionPolicy(roles=["readonly"]),
        )
        warnings = validate_permissions(emp)
        # 没有 unknown role/tool 警告
        assert not any("未知" in w for w in warnings)


# ── PermissionGuard ──


class TestPermissionGuard:
    def test_allows_declared_tools(self):
        emp = _make_employee(tools=["file_read", "bash"])
        guard = PermissionGuard(emp)
        assert guard.check_soft("file_read") is None
        assert guard.check_soft("bash") is None

    def test_blocks_undeclared_tools(self):
        emp = _make_employee(tools=["file_read"])
        guard = PermissionGuard(emp)
        msg = guard.check_soft("bash")
        assert msg is not None
        assert "bash" in msg

    def test_submit_always_allowed(self):
        """submit/finish/load_tools 始终允许."""
        emp = _make_employee(tools=["file_read"])
        guard = PermissionGuard(emp)
        for tool in ("submit", "finish", "load_tools"):
            assert guard.check_soft(tool) is None

    def test_check_raises_on_denied(self):
        emp = _make_employee(tools=["file_read"])
        guard = PermissionGuard(emp)
        try:
            guard.check("bash")
            assert False, "应抛出 PermissionDenied"
        except PermissionDenied as e:
            assert e.tool_name == "bash"
            assert e.employee_name == "test-emp"

    def test_guard_with_permissions_policy(self):
        """permissions 策略限制后，guard 只允许有效工具."""
        emp = _make_employee(
            tools=["file_read", "bash", "git"],
            permissions=PermissionPolicy(roles=["readonly"], deny=["git"]),
        )
        guard = PermissionGuard(emp)
        assert guard.check_soft("file_read") is None
        # bash 不在 readonly 角色中
        assert guard.check_soft("bash") is not None
        # git 被 deny
        assert guard.check_soft("git") is not None

    def test_guard_no_permissions_allows_all_tools(self):
        """无 permissions 时，guard 允许所有声明的 tools."""
        emp = _make_employee(tools=["file_read", "bash", "git"])
        guard = PermissionGuard(emp)
        for tool in ["file_read", "bash", "git"]:
            assert guard.check_soft(tool) is None


# ── Parser integration ──


class TestParserPermissions:
    def test_parse_employee_string_with_permissions(self):
        """parse_employee_string 正确解析 permissions 字段."""
        from crew.parser import parse_employee_string

        md = """\
---
name: test-perm
description: 测试权限
tools: [file_read, bash, git]
permissions:
  roles: [readonly]
  deny: [git]
---

测试正文
"""
        emp = parse_employee_string(md)
        assert emp.permissions is not None
        assert emp.permissions.roles == ["readonly"]
        assert emp.permissions.deny == ["git"]

    def test_parse_employee_string_without_permissions(self):
        """无 permissions 字段时为 None."""
        from crew.parser import parse_employee_string

        md = """\
---
name: test-no-perm
description: 无权限
tools: [file_read]
---

正文
"""
        emp = parse_employee_string(md)
        assert emp.permissions is None

    def test_parse_employee_dir_with_permissions(self, tmp_path):
        """parse_employee_dir 正确解析 permissions."""
        from crew.parser import parse_employee_dir

        emp_dir = tmp_path / "test-dir-perm"
        emp_dir.mkdir()
        (emp_dir / "employee.yaml").write_text("""\
name: test-dir-perm
description: 目录格式权限
tools: [file_read, grep]
permissions:
  roles: [readonly]
  allow: [bash]
""")
        (emp_dir / "prompt.md").write_text("目录格式正文")

        emp = parse_employee_dir(emp_dir)
        assert emp.permissions is not None
        assert emp.permissions.roles == ["readonly"]
        assert emp.permissions.allow == ["bash"]


# ── TOOL_ROLE_PRESETS sanity checks ──


class TestToolRolePresets:
    def test_readonly_preset(self):
        assert TOOL_ROLE_PRESETS["readonly"] == {"file_read", "grep", "glob"}

    def test_developer_preset(self):
        assert "bash" in TOOL_ROLE_PRESETS["developer"]
        assert "file_write" in TOOL_ROLE_PRESETS["developer"]

    def test_feishu_admin_is_union(self):
        admin = TOOL_ROLE_PRESETS["feishu-admin"]
        read = TOOL_ROLE_PRESETS["feishu-read"]
        write = TOOL_ROLE_PRESETS["feishu-write"]
        assert admin == read | write

    def test_all_preset_includes_sandbox_tools(self):
        all_preset = TOOL_ROLE_PRESETS["all"]
        assert "file_read" in all_preset
        assert "bash" in all_preset

    def test_profile_engineer_excludes_life_tools(self):
        """profile-engineer 应排除生活工具."""
        preset = TOOL_ROLE_PRESETS["profile-engineer"]
        for tool in (
            "weather",
            "exchange_rate",
            "stock_price",
            "flight_info",
            "aqi",
            "express_track",
        ):
            assert tool not in preset, f"profile-engineer 不应包含 {tool}"
        assert "github_prs" in preset  # 应保留代码相关

    def test_profile_researcher_excludes_admin_tools(self):
        """profile-researcher 应排除管理写操作."""
        preset = TOOL_ROLE_PRESETS["profile-researcher"]
        for tool in ("update_agent", "delegate_async", "delegate_chain", "route"):
            assert tool not in preset
        assert "web_search" in preset  # 应保留搜索

    def test_profile_security_excludes_write_ops(self):
        """profile-security 应排除影响审计独立性的写操作."""
        preset = TOOL_ROLE_PRESETS["profile-security"]
        for tool in (
            "update_agent",
            "delegate",
            "delegate_async",
            "delegate_chain",
            "route",
            "send_feishu_dm",
        ):
            assert tool not in preset

    def test_all_profiles_subset_of_base(self):
        """所有 profile-* 都应是 profile-base 的子集."""
        base = TOOL_ROLE_PRESETS["profile-base"]
        for name, preset in TOOL_ROLE_PRESETS.items():
            if name.startswith("profile-") and name != "profile-base":
                assert preset <= base, f"{name} 包含了 profile-base 中不存在的工具"
