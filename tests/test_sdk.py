"""Tests for crew.sdk helpers."""

from crew import sdk


class TestSDK:
    """SDK level tests."""

    def test_list_employees(self):
        employees = sdk.list_employees()
        assert any(emp.name == "code-reviewer" for emp in employees)

    def test_generate_prompt_by_name(self):
        text = sdk.generate_prompt_by_name(
            "code-reviewer", args={"target": "main"}, smart_context=False
        )
        assert "代码审查员" in text

    def test_generate_prompt_raw(self):
        text = sdk.generate_prompt_by_name(
            "code-reviewer", args={"target": "main"}, raw=True
        )
        assert "# 代码审查员" not in text
