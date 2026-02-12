"""讨论会模块测试."""

from pathlib import Path

import yaml

from crew.discussion import (
    Discussion,
    DiscussionParticipant,
    DiscussionRound,
    _ROUND_TEMPLATES,
    _resolve_rounds,
    create_adhoc_discussion,
    discover_discussions,
    load_discussion,
    render_discussion,
    validate_discussion,
)


# ── 数据模型测试 ──


class TestDiscussionModels:
    """测试讨论会数据模型."""

    def test_participant_default_role(self):
        p = DiscussionParticipant(employee="code-reviewer")
        assert p.role == "speaker"
        assert p.focus == ""

    def test_participant_with_role(self):
        p = DiscussionParticipant(employee="product-manager", role="moderator", focus="需求")
        assert p.role == "moderator"
        assert p.focus == "需求"

    def test_round_defaults(self):
        r = DiscussionRound()
        assert r.name == ""
        assert r.instruction == ""

    def test_round_with_values(self):
        r = DiscussionRound(name="需求对齐", instruction="讨论需求")
        assert r.name == "需求对齐"

    def test_round_new_interaction_modes(self):
        for mode in ("brainstorm", "vote", "debate"):
            r = DiscussionRound(interaction=mode)
            assert r.interaction == mode

    def test_discussion_rounds_int(self):
        d = Discussion(
            name="test",
            topic="测试议题",
            participants=[
                DiscussionParticipant(employee="code-reviewer"),
                DiscussionParticipant(employee="test-engineer"),
            ],
            rounds=3,
        )
        assert d.rounds == 3
        assert d.output_format == "decision"

    def test_discussion_rounds_list(self):
        d = Discussion(
            name="test",
            topic="测试议题",
            participants=[
                DiscussionParticipant(employee="code-reviewer"),
                DiscussionParticipant(employee="test-engineer"),
            ],
            rounds=[
                DiscussionRound(name="第一轮", instruction="开场"),
                DiscussionRound(name="第二轮", instruction="深入"),
            ],
        )
        assert len(d.rounds) == 2

    def test_discussion_output_formats(self):
        for fmt in ("decision", "transcript", "summary"):
            d = Discussion(
                name="test",
                topic="测试",
                participants=[
                    DiscussionParticipant(employee="a"),
                    DiscussionParticipant(employee="b"),
                ],
                output_format=fmt,
            )
            assert d.output_format == fmt

    def test_mode_defaults_to_auto(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="a"),
                DiscussionParticipant(employee="b"),
            ],
        )
        assert d.mode == "auto"

    def test_effective_mode_auto_discussion(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="a"),
                DiscussionParticipant(employee="b"),
            ],
        )
        assert d.effective_mode == "discussion"

    def test_effective_mode_auto_meeting(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[DiscussionParticipant(employee="a")],
        )
        assert d.effective_mode == "meeting"

    def test_effective_mode_explicit(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="a"),
                DiscussionParticipant(employee="b"),
            ],
            mode="meeting",
        )
        assert d.effective_mode == "meeting"

    def test_round_template_field(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="a"),
                DiscussionParticipant(employee="b"),
            ],
            round_template="standard",
        )
        assert d.round_template == "standard"


# ── 轮次模板测试 ──


class TestRoundTemplates:
    """测试轮次模板."""

    def test_resolve_rounds_no_template(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="a"),
                DiscussionParticipant(employee="b"),
            ],
            rounds=3,
        )
        assert _resolve_rounds(d) == 3

    def test_resolve_rounds_with_template(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="a"),
                DiscussionParticipant(employee="b"),
            ],
            round_template="standard",
        )
        rounds = _resolve_rounds(d)
        assert isinstance(rounds, list)
        assert len(rounds) == len(_ROUND_TEMPLATES["standard"])

    def test_resolve_rounds_unknown_template_fallback(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="a"),
                DiscussionParticipant(employee="b"),
            ],
            rounds=2,
            round_template="nonexistent",
        )
        assert _resolve_rounds(d) == 2

    def test_all_templates_exist(self):
        for name in ("standard", "brainstorm-to-decision", "adversarial"):
            assert name in _ROUND_TEMPLATES
            assert len(_ROUND_TEMPLATES[name]) >= 3


# ── 加载测试 ──


class TestLoadDiscussion:
    """测试加载讨论会."""

    def test_load_valid_yaml(self, tmp_path):
        data = {
            "name": "test-discuss",
            "topic": "测试议题",
            "goal": "达成共识",
            "participants": [
                {"employee": "code-reviewer", "role": "moderator"},
                {"employee": "test-engineer", "role": "speaker"},
            ],
            "rounds": 2,
        }
        f = tmp_path / "test.yaml"
        f.write_text(yaml.dump(data, allow_unicode=True))
        d = load_discussion(f)
        assert d.name == "test-discuss"
        assert d.goal == "达成共识"
        assert len(d.participants) == 2
        assert d.participants[0].role == "moderator"
        assert d.rounds == 2

    def test_load_custom_rounds(self, tmp_path):
        data = {
            "name": "test",
            "topic": "测试",
            "participants": [
                {"employee": "a"},
                {"employee": "b"},
            ],
            "rounds": [
                {"name": "需求对齐", "instruction": "讨论需求"},
                {"name": "方案讨论", "instruction": "讨论方案"},
            ],
        }
        f = tmp_path / "test.yaml"
        f.write_text(yaml.dump(data, allow_unicode=True))
        d = load_discussion(f)
        assert isinstance(d.rounds, list)
        assert len(d.rounds) == 2
        assert d.rounds[0].name == "需求对齐"

    def test_load_builtin_architecture_review(self):
        builtin = (
            Path(__file__).parent.parent
            / "src"
            / "crew"
            / "employees"
            / "discussions"
            / "architecture-review.yaml"
        )
        d = load_discussion(builtin)
        assert d.name == "architecture-review"
        assert len(d.participants) >= 3

    def test_load_builtin_feature_design(self):
        builtin = (
            Path(__file__).parent.parent
            / "src"
            / "crew"
            / "employees"
            / "discussions"
            / "feature-design.yaml"
        )
        d = load_discussion(builtin)
        assert d.name == "feature-design"
        assert isinstance(d.rounds, list)

    def test_load_yaml_with_mode(self, tmp_path):
        data = {
            "name": "test-1v1",
            "topic": "一对一讨论",
            "mode": "meeting",
            "participants": [{"employee": "code-reviewer"}],
        }
        f = tmp_path / "test.yaml"
        f.write_text(yaml.dump(data, allow_unicode=True))
        d = load_discussion(f)
        assert d.mode == "meeting"


# ── 校验测试 ──


class TestValidateDiscussion:
    """测试讨论会校验."""

    def test_zero_participants(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[],
        )
        errors = validate_discussion(d)
        assert any("至少需要 1 个参与者" in e for e in errors)

    def test_single_participant_valid(self):
        """1 个参与者应通过校验（1v1 会议）."""
        d = Discussion(
            name="test",
            topic="测试",
            participants=[DiscussionParticipant(employee="code-reviewer")],
        )
        errors = validate_discussion(d)
        assert not any("参与者" in e for e in errors)

    def test_unknown_employee(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="nonexistent-a"),
                DiscussionParticipant(employee="nonexistent-b"),
            ],
        )
        errors = validate_discussion(d)
        assert any("nonexistent-a" in e for e in errors)
        assert any("nonexistent-b" in e for e in errors)

    def test_valid_discussion(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="code-reviewer", role="moderator"),
                DiscussionParticipant(employee="test-engineer"),
            ],
        )
        errors = validate_discussion(d)
        assert errors == []


# ── 渲染测试 ──


class TestRenderDiscussion:
    """测试讨论会渲染."""

    def test_render_contains_all_participants(self):
        d = Discussion(
            name="test",
            topic="讨论 $target",
            participants=[
                DiscussionParticipant(employee="code-reviewer", role="moderator"),
                DiscussionParticipant(employee="test-engineer"),
            ],
            rounds=2,
        )
        prompt = render_discussion(d, initial_args={"target": "main"}, smart_context=False)
        assert "代码审查员" in prompt
        assert "测试工程师" in prompt
        assert "讨论 main" in prompt
        assert "主持人" in prompt
        assert "发言人" in prompt

    def test_render_injects_employee_body(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="code-reviewer"),
                DiscussionParticipant(employee="test-engineer"),
            ],
        )
        prompt = render_discussion(d, smart_context=False)
        assert "<专业背景>" in prompt
        assert "</专业背景>" in prompt
        # body 中应包含员工的角色描述
        assert "审查" in prompt.lower() or "review" in prompt.lower()

    def test_render_custom_rounds(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="code-reviewer"),
                DiscussionParticipant(employee="test-engineer"),
            ],
            rounds=[
                DiscussionRound(name="需求对齐", instruction="讨论需求"),
                DiscussionRound(name="方案讨论", instruction="讨论方案"),
            ],
        )
        prompt = render_discussion(d, smart_context=False)
        assert "需求对齐" in prompt
        assert "方案讨论" in prompt
        assert "讨论需求" in prompt

    def test_render_int_rounds(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="code-reviewer"),
                DiscussionParticipant(employee="test-engineer"),
            ],
            rounds=3,
        )
        prompt = render_discussion(d, smart_context=False)
        assert "第 1 轮" in prompt
        assert "第 2 轮" in prompt
        assert "第 3 轮" in prompt
        assert "开场" in prompt
        assert "总结与决议" in prompt

    def test_render_variable_substitution(self):
        d = Discussion(
            name="test",
            topic="评审 $target 的设计",
            goal="形成 $target 的改进方案",
            participants=[
                DiscussionParticipant(employee="code-reviewer"),
                DiscussionParticipant(employee="test-engineer"),
            ],
        )
        prompt = render_discussion(
            d, initial_args={"target": "auth.py"}, smart_context=False
        )
        assert "评审 auth.py 的设计" in prompt
        assert "形成 auth.py 的改进方案" in prompt

    def test_render_with_focus(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="code-reviewer", focus="安全性"),
                DiscussionParticipant(employee="test-engineer", focus="覆盖率"),
            ],
        )
        prompt = render_discussion(d, smart_context=False)
        assert "安全性" in prompt
        assert "覆盖率" in prompt

    def test_render_decision_format(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="code-reviewer"),
                DiscussionParticipant(employee="test-engineer"),
            ],
            output_format="decision",
        )
        prompt = render_discussion(d, smart_context=False)
        assert "决议" in prompt
        assert "行动项" in prompt

    def test_render_summary_format(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="code-reviewer"),
                DiscussionParticipant(employee="test-engineer"),
            ],
            output_format="summary",
        )
        prompt = render_discussion(d, smart_context=False)
        assert "总结" in prompt
        assert "共识" in prompt

    def test_render_unknown_employee(self):
        """未找到的员工不会导致崩溃."""
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="code-reviewer"),
                DiscussionParticipant(employee="nonexistent"),
            ],
        )
        prompt = render_discussion(d, smart_context=False)
        assert "未找到" in prompt
        assert "代码审查员" in prompt

    def test_render_with_round_template(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="code-reviewer"),
                DiscussionParticipant(employee="test-engineer"),
            ],
            round_template="brainstorm-to-decision",
        )
        prompt = render_discussion(d, smart_context=False)
        assert "发散" in prompt
        assert "投票" in prompt

    def test_render_brainstorm_interaction(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="code-reviewer"),
                DiscussionParticipant(employee="test-engineer"),
            ],
            rounds=[
                DiscussionRound(name="头脑风暴", interaction="brainstorm"),
            ],
        )
        prompt = render_discussion(d, smart_context=False)
        assert "创意" in prompt

    def test_render_vote_interaction(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="code-reviewer"),
                DiscussionParticipant(employee="test-engineer"),
            ],
            rounds=[
                DiscussionRound(name="投票", interaction="vote"),
            ],
        )
        prompt = render_discussion(d, smart_context=False)
        assert "投票" in prompt

    def test_render_debate_interaction(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="code-reviewer"),
                DiscussionParticipant(employee="test-engineer"),
            ],
            rounds=[
                DiscussionRound(name="辩论", interaction="debate"),
            ],
        )
        prompt = render_discussion(d, smart_context=False)
        assert "正方" in prompt or "反方" in prompt


# ── 1v1 会议测试 ──


class TestOneOnOneMeeting:
    """测试 1v1 会议."""

    def test_1v1_renders_meeting_header(self):
        d = Discussion(
            name="test",
            topic="招聘方案讨论",
            participants=[DiscussionParticipant(employee="code-reviewer")],
        )
        prompt = render_discussion(d, smart_context=False)
        assert "会议" in prompt
        assert "招聘方案讨论" in prompt

    def test_1v1_injects_full_background(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[DiscussionParticipant(employee="code-reviewer")],
        )
        prompt = render_discussion(d, smart_context=False)
        assert "<专业背景>" in prompt

    def test_1v1_uses_meeting_rules(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[DiscussionParticipant(employee="code-reviewer")],
        )
        prompt = render_discussion(d, smart_context=False)
        assert "一对一" in prompt
        assert "会议规则" in prompt

    def test_1v1_no_multi_person_structure(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[DiscussionParticipant(employee="code-reviewer")],
        )
        prompt = render_discussion(d, smart_context=False)
        assert "轮次安排" not in prompt
        assert "讨论规则" not in prompt

    def test_1v1_with_explicit_mode(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[
                DiscussionParticipant(employee="code-reviewer"),
                DiscussionParticipant(employee="test-engineer"),
            ],
            mode="meeting",
        )
        prompt = render_discussion(d, smart_context=False)
        # 即使有 2 人，mode=meeting 也走 1v1 路径
        assert "会议" in prompt

    def test_1v1_decision_format(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[DiscussionParticipant(employee="code-reviewer")],
            output_format="decision",
        )
        prompt = render_discussion(d, smart_context=False)
        assert "会议纪要" in prompt
        assert "行动项" in prompt

    def test_1v1_summary_format(self):
        d = Discussion(
            name="test",
            topic="测试",
            participants=[DiscussionParticipant(employee="code-reviewer")],
            output_format="summary",
        )
        prompt = render_discussion(d, smart_context=False)
        assert "会议总结" in prompt


# ── 即席讨论测试 ──


class TestAdhocDiscussion:
    """测试即席讨论."""

    def test_create_adhoc_basic(self):
        d = create_adhoc_discussion(
            employees=["code-reviewer", "test-engineer"],
            topic="测试议题",
        )
        assert d.name.startswith("adhoc-")
        assert d.topic == "测试议题"
        assert len(d.participants) == 2
        assert d.participants[0].role == "moderator"
        assert d.participants[1].role == "speaker"

    def test_create_adhoc_single_employee(self):
        d = create_adhoc_discussion(
            employees=["code-reviewer"],
            topic="代码审查",
        )
        assert len(d.participants) == 1
        assert d.participants[0].role == "speaker"
        assert d.effective_mode == "meeting"

    def test_create_adhoc_with_goal(self):
        d = create_adhoc_discussion(
            employees=["code-reviewer", "test-engineer"],
            topic="测试",
            goal="达成共识",
        )
        assert d.goal == "达成共识"

    def test_create_adhoc_with_round_template(self):
        d = create_adhoc_discussion(
            employees=["code-reviewer", "test-engineer"],
            topic="测试",
            round_template="adversarial",
        )
        assert d.round_template == "adversarial"

    def test_create_adhoc_defaults(self):
        d = create_adhoc_discussion(
            employees=["code-reviewer", "test-engineer"],
            topic="测试",
        )
        assert d.rounds == 2
        assert d.output_format == "summary"

    def test_adhoc_render(self):
        d = create_adhoc_discussion(
            employees=["code-reviewer", "test-engineer"],
            topic="auth 模块代码质量",
        )
        prompt = render_discussion(d, smart_context=False)
        assert "auth 模块代码质量" in prompt
        assert "代码审查员" in prompt


# ── 发现测试 ──


class TestDiscoverDiscussions:
    """测试讨论会发现."""

    def test_discover_builtin(self):
        discussions = discover_discussions()
        assert "architecture-review" in discussions
        assert "feature-design" in discussions

    def test_discover_project(self, tmp_path):
        d_dir = tmp_path / ".crew" / "discussions"
        d_dir.mkdir(parents=True)
        data = {
            "name": "custom",
            "topic": "自定义议题",
            "participants": [
                {"employee": "code-reviewer"},
                {"employee": "test-engineer"},
            ],
        }
        (d_dir / "custom.yaml").write_text(yaml.dump(data, allow_unicode=True))
        discussions = discover_discussions(project_dir=tmp_path)
        assert "custom" in discussions

    def test_project_overrides_builtin(self, tmp_path):
        d_dir = tmp_path / ".crew" / "discussions"
        d_dir.mkdir(parents=True)
        data = {
            "name": "architecture-review",
            "description": "自定义版",
            "topic": "自定义议题",
            "participants": [
                {"employee": "code-reviewer"},
                {"employee": "test-engineer"},
            ],
        }
        (d_dir / "architecture-review.yaml").write_text(
            yaml.dump(data, allow_unicode=True)
        )
        discussions = discover_discussions(project_dir=tmp_path)
        d = load_discussion(discussions["architecture-review"])
        assert d.description == "自定义版"
