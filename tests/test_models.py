"""数据模型测试 — models.py 核心类与映射."""


import pytest

from crew.models import (
    SKILL_TO_TOOL,
    TOOL_TO_SKILL,
    ActionItem,
    Condition,
    ConditionalBody,
    DiscoveryResult,
    Employee,
    EmployeeArg,
    EmployeeOutput,
    LoopBody,
    PermissionPolicy,
    PipelineStep,
    StepResult,
    ToolCall,
    ToolExecutionResult,
    WorkLogEntry,
)

# ── TOOL_TO_SKILL / SKILL_TO_TOOL 映射 ──


class TestToolMappings:
    def test_tool_to_skill_keys(self):
        """TOOL_TO_SKILL 包含基本工具."""
        assert "file_read" in TOOL_TO_SKILL
        assert "file_write" in TOOL_TO_SKILL
        assert "bash" in TOOL_TO_SKILL
        assert "git" in TOOL_TO_SKILL

    def test_skill_to_tool_reverse(self):
        """SKILL_TO_TOOL 是 TOOL_TO_SKILL 的反向映射."""
        for tool, skill in TOOL_TO_SKILL.items():
            assert SKILL_TO_TOOL[skill] == tool

    def test_mapping_roundtrip(self):
        """tool -> skill -> tool 往返一致."""
        for tool in TOOL_TO_SKILL:
            assert SKILL_TO_TOOL[TOOL_TO_SKILL[tool]] == tool


# ── EmployeeArg / EmployeeOutput ──


class TestEmployeeArg:
    def test_defaults(self):
        arg = EmployeeArg(name="input")
        assert arg.description == ""
        assert arg.required is False
        assert arg.default is None

    def test_full(self):
        arg = EmployeeArg(name="query", description="搜索词", required=True, default="hello")
        assert arg.name == "query"
        assert arg.required is True
        assert arg.default == "hello"


class TestEmployeeOutput:
    def test_defaults(self):
        out = EmployeeOutput()
        assert out.format == "markdown"
        assert out.dir == ".crew/logs"
        assert out.filename == ""

    def test_custom(self):
        out = EmployeeOutput(format="json", filename="report-{date}.json")
        assert out.format == "json"
        assert out.filename == "report-{date}.json"


# ── Employee ──


class TestEmployee:
    def _make(self, **kwargs):
        defaults = {
            "name": "test",
            "description": "测试",
            "body": "正文",
        }
        defaults.update(kwargs)
        return Employee(**defaults)

    def test_minimal(self):
        emp = self._make()
        assert emp.name == "test"
        assert emp.display_name == ""
        assert emp.version == "1.0"
        assert emp.tools == []
        assert emp.permissions is None
        assert emp.source_layer == "builtin"

    def test_effective_display_name_with_display_name(self):
        emp = self._make(display_name="测试员工")
        assert emp.effective_display_name == "测试员工"

    def test_effective_display_name_fallback_to_name(self):
        emp = self._make()
        assert emp.effective_display_name == "test"

    def test_resolve_env_vars_api_key(self, monkeypatch):
        """api_key 的 ${ENV_VAR} 被解析."""
        monkeypatch.setenv("MY_API_KEY", "sk-test-123")
        emp = self._make(api_key="${MY_API_KEY}")
        assert emp.api_key == "sk-test-123"

    def test_resolve_env_vars_fallback_api_key(self, monkeypatch):
        """fallback_api_key 的 ${ENV_VAR} 被解析."""
        monkeypatch.setenv("FALLBACK_KEY", "sk-fallback-456")
        emp = self._make(fallback_api_key="${FALLBACK_KEY}")
        assert emp.fallback_api_key == "sk-fallback-456"

    def test_resolve_env_vars_missing(self, monkeypatch):
        """环境变量不存在时解析为空字符串."""
        monkeypatch.delenv("NONEXISTENT_KEY_12345", raising=False)
        emp = self._make(api_key="${NONEXISTENT_KEY_12345}")
        assert emp.api_key == ""

    def test_resolve_env_vars_not_pattern(self):
        """非 ${} 格式的值不被替换."""
        emp = self._make(api_key="sk-literal-key")
        assert emp.api_key == "sk-literal-key"

    def test_resolve_env_vars_partial_pattern(self):
        """部分匹配（如 prefix + ${VAR}）不替换."""
        emp = self._make(api_key="prefix-${VAR}")
        assert emp.api_key == "prefix-${VAR}"

    def test_permissions_field(self):
        policy = PermissionPolicy(roles=["readonly"], deny=["bash"])
        emp = self._make(permissions=policy)
        assert emp.permissions is not None
        assert emp.permissions.roles == ["readonly"]


# ── PermissionPolicy ──


class TestPermissionPolicy:
    def test_defaults(self):
        pp = PermissionPolicy()
        assert pp.roles == []
        assert pp.allow == []
        assert pp.deny == []

    def test_full(self):
        pp = PermissionPolicy(roles=["developer"], allow=["custom"], deny=["bash"])
        assert pp.roles == ["developer"]
        assert pp.allow == ["custom"]
        assert pp.deny == ["bash"]


# ── Condition ──


class TestCondition:
    def test_contains_match(self):
        c = Condition(check="{prev}", contains="PASS")
        assert c.evaluate("TEST PASSED") is True

    def test_contains_no_match(self):
        c = Condition(check="{prev}", contains="FAIL")
        assert c.evaluate("TEST PASSED") is False

    def test_matches_regex(self):
        c = Condition(check="{prev}", matches=r"\d{3}")
        assert c.evaluate("code 200 ok") is True

    def test_matches_regex_no_match(self):
        c = Condition(check="{prev}", matches=r"\d{5}")
        assert c.evaluate("code 200 ok") is False

    def test_both_set_raises(self):
        """contains 和 matches 不可同时设置."""
        with pytest.raises(ValueError, match="之一"):
            Condition(check="{prev}", contains="x", matches="y")

    def test_both_empty_raises(self):
        """contains 和 matches 不可同时为空."""
        with pytest.raises(ValueError, match="之一"):
            Condition(check="{prev}")

    def test_matches_too_long_raises(self):
        with pytest.raises(ValueError, match="256"):
            Condition(check="{prev}", matches="a" * 257)

    def test_invalid_regex_raises(self):
        with pytest.raises(Exception):
            Condition(check="{prev}", matches="[invalid")


# ── ConditionalBody ──


class TestConditionalBody:
    def test_evaluate_contains(self):
        cb = ConditionalBody(
            check="{prev}",
            contains="ok",
            then=[PipelineStep(employee="a")],
        )
        assert cb.evaluate("all ok") is True
        assert cb.evaluate("failed") is False

    def test_evaluate_matches(self):
        cb = ConditionalBody(
            check="{prev}",
            matches=r"^ok$",
            then=[PipelineStep(employee="a")],
        )
        assert cb.evaluate("ok") is True
        assert cb.evaluate("not ok") is False

    def test_else_alias(self):
        """else_ 字段可通过 alias 'else' 序列化."""
        cb = ConditionalBody(
            check="{prev}",
            contains="x",
            then=[PipelineStep(employee="a")],
            **{"else": [PipelineStep(employee="b")]},
        )
        assert len(cb.else_) == 1
        assert cb.else_[0].employee == "b"


# ── DiscoveryResult ──


class TestDiscoveryResult:
    def _make_emp(self, name, triggers=None):
        return Employee(
            name=name,
            description="测试",
            body="正文",
            triggers=triggers or [],
        )

    def test_get_by_name(self):
        emp = self._make_emp("reviewer")
        dr = DiscoveryResult(employees={"reviewer": emp})
        assert dr.get("reviewer") is emp

    def test_get_by_trigger(self):
        emp = self._make_emp("reviewer", triggers=["cr", "review"])
        dr = DiscoveryResult(employees={"reviewer": emp})
        assert dr.get("cr") is emp
        assert dr.get("review") is emp

    def test_get_not_found(self):
        dr = DiscoveryResult(employees={})
        assert dr.get("nonexistent") is None

    def test_get_name_takes_priority(self):
        """名称匹配优先于触发别名."""
        emp1 = self._make_emp("alpha", triggers=["beta"])
        emp2 = self._make_emp("beta", triggers=[])
        dr = DiscoveryResult(employees={"alpha": emp1, "beta": emp2})
        assert dr.get("beta") is emp2


# ── ToolCall / ToolExecutionResult ──


class TestToolCallDataclass:
    def test_defaults(self):
        tc = ToolCall(id="t1", name="bash")
        assert tc.arguments == {}

    def test_with_args(self):
        tc = ToolCall(id="t2", name="file_read", arguments={"path": "/tmp"})
        assert tc.arguments["path"] == "/tmp"


class TestToolExecutionResult:
    def test_no_tool_calls(self):
        r = ToolExecutionResult(content="hello")
        assert r.has_tool_calls is False
        assert r.tool_calls == []

    def test_with_tool_calls(self):
        tc = ToolCall(id="t1", name="bash")
        r = ToolExecutionResult(content="", tool_calls=[tc])
        assert r.has_tool_calls is True


# ── WorkLogEntry ──


class TestWorkLogEntry:
    def test_defaults(self):
        entry = WorkLogEntry(employee_name="test", action="执行")
        assert entry.severity == "info"
        assert entry.detail == ""
        assert entry.metrics == {}

    def test_full(self):
        entry = WorkLogEntry(
            employee_name="test",
            action="审查",
            detail="发现 3 个问题",
            severity="warning",
            metrics={"issues": 3.0},
        )
        assert entry.severity == "warning"
        assert entry.metrics["issues"] == 3.0


# ── ActionItem ──


class TestActionItem:
    def test_defaults(self):
        a = ActionItem(id="A1", description="修复 bug", assignee_role="executor")
        assert a.priority == "P2"
        assert a.phase == "implement"
        assert a.depends_on == []

    def test_full(self):
        a = ActionItem(
            id="A2",
            description="部署",
            assignee_role="monitor",
            depends_on=["A1"],
            priority="P0",
            phase="deploy",
        )
        assert a.depends_on == ["A1"]
        assert a.priority == "P0"


# ── StepResult ──


class TestStepResult:
    def test_defaults(self):
        sr = StepResult(employee="test", step_index=0, prompt="hello")
        assert sr.error is False
        assert sr.branch == ""
        assert sr.duration_ms == 0

    def test_error_step(self):
        sr = StepResult(
            employee="test",
            step_index=1,
            prompt="hello",
            error=True,
            error_message="timeout",
        )
        assert sr.error is True
        assert sr.error_message == "timeout"


# ── LoopBody ──


class TestLoopBody:
    def test_defaults(self):
        lb = LoopBody(
            steps=[PipelineStep(employee="a")],
            until=Condition(check="{prev}", contains="done"),
        )
        assert lb.max_iterations == 5

    def test_max_iterations_bounds(self):
        with pytest.raises(Exception):
            LoopBody(
                steps=[PipelineStep(employee="a")],
                until=Condition(check="{prev}", contains="done"),
                max_iterations=0,
            )
        with pytest.raises(Exception):
            LoopBody(
                steps=[PipelineStep(employee="a")],
                until=Condition(check="{prev}", contains="done"),
                max_iterations=51,
            )
