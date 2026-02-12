"""模板管理器测试."""

from pathlib import Path

from crew.template_manager import discover_templates, render_template


def test_discover_templates_has_builtin():
    templates = discover_templates()
    assert "advanced-employee" in templates
    record = templates["advanced-employee"]
    assert record.path.exists()


def test_render_template_replaces_variables(tmp_path):
    template = "Hello {{name}} from {{project}}!"
    rendered = render_template(template, {"name": "Crew", "project": "Knowlyr"})
    assert rendered == "Hello Crew from Knowlyr!"
    rendered_missing = render_template("Hi {{missing}}", {})
    assert "{{missing}}" in rendered_missing


def test_render_template_supports_unicode_variables():
    template = "你好，{{角色}}"
    rendered = render_template(template, {"角色": "架构师"})
    assert rendered == "你好，架构师"
