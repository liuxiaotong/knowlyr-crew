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
