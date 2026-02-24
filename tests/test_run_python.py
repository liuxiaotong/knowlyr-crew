"""run_python 工具测试."""

from __future__ import annotations

import asyncio

from crew.webhook_tools.engineering import _tool_run_python, _validate_python_code


def _run(coro):
    return asyncio.run(coro)


# ── AST 验证 ──


class TestValidatePythonCode:
    def test_valid_code(self):
        assert _validate_python_code("print(1+1)") is None

    def test_valid_import_json(self):
        assert _validate_python_code("import json\nprint(json.dumps({'a':1}))") is None

    def test_blocked_import_httpx(self):
        result = _validate_python_code("import httpx")
        assert result is not None
        assert "不允许" in result

    def test_blocked_import_bs4(self):
        result = _validate_python_code("from bs4 import BeautifulSoup")
        assert result is not None
        assert "不允许" in result

    def test_blocked_os_system_via_attribute(self):
        result = _validate_python_code("import json\njson.system()")
        assert result is not None
        assert "不允许" in result

    def test_blocked_subprocess_popen_via_attribute(self):
        result = _validate_python_code("x.Popen(['ls'])")
        assert result is not None
        assert "不允许" in result

    def test_valid_import_from(self):
        assert _validate_python_code("from datetime import datetime") is None

    def test_blocked_import_os(self):
        result = _validate_python_code("import os")
        assert result is not None
        assert "不允许" in result

    def test_blocked_import_subprocess(self):
        result = _validate_python_code("import subprocess")
        assert result is not None
        assert "不允许" in result

    def test_blocked_import_sys(self):
        result = _validate_python_code("import sys")
        assert result is not None
        assert "不允许" in result

    def test_blocked_import_shutil(self):
        result = _validate_python_code("import shutil")
        assert result is not None

    def test_blocked_from_os(self):
        result = _validate_python_code("from os import system")
        assert result is not None

    def test_blocked_exec(self):
        result = _validate_python_code("exec('print(1)')")
        assert result is not None
        assert "不允许" in result

    def test_blocked_eval(self):
        result = _validate_python_code("eval('1+1')")
        assert result is not None

    def test_blocked_compile(self):
        result = _validate_python_code("compile('1', '', 'exec')")
        assert result is not None

    def test_blocked_getattr(self):
        result = _validate_python_code("getattr([], 'append')")
        assert result is not None

    def test_blocked_dunder_import(self):
        result = _validate_python_code("__import__('os')")
        assert result is not None

    def test_open_read_allowed(self):
        # open 读模式应该通过
        assert _validate_python_code("open('/tmp/test.txt', 'r')") is None

    def test_open_write_blocked(self):
        result = _validate_python_code("open('/tmp/test.txt', 'w')")
        assert result is not None
        assert "只允许读" in result

    def test_open_write_mode_keyword(self):
        result = _validate_python_code("open('/tmp/test.txt', mode='w')")
        assert result is not None

    def test_syntax_error(self):
        result = _validate_python_code("def")
        assert result is not None
        assert "语法错误" in result

    def test_multiline_valid(self):
        code = "import json\nimport re\ndata = {'a': 1}\nprint(json.dumps(data))"
        assert _validate_python_code(code) is None


# ── 工具执行 ──


class TestToolRunPython:
    def test_empty_code(self):
        result = _run(_tool_run_python({"code": ""}))
        assert "需要" in result

    def test_basic_execution(self):
        result = _run(_tool_run_python({"code": "print(1+1)"}))
        assert "2" in result

    def test_multiline(self):
        code = "x = 10\ny = 20\nprint(x + y)"
        result = _run(_tool_run_python({"code": code}))
        assert "30" in result

    def test_import_json(self):
        code = "import json\nprint(json.dumps({'hello': 'world'}))"
        result = _run(_tool_run_python({"code": code}))
        assert "hello" in result

    def test_import_re(self):
        code = "import re\nprint(re.findall(r'\\d+', 'abc123def456'))"
        result = _run(_tool_run_python({"code": code}))
        assert "123" in result

    def test_import_datetime(self):
        code = "from datetime import datetime\nprint(datetime.now().year)"
        result = _run(_tool_run_python({"code": code}))
        assert "202" in result  # 2025 or 2026

    def test_blocked_import_rejected(self):
        result = _run(_tool_run_python({"code": "import os\nos.listdir('.')"}))
        assert "检查失败" in result

    def test_exec_rejected(self):
        result = _run(_tool_run_python({"code": "exec('print(1)')"}))
        assert "检查失败" in result

    def test_syntax_error(self):
        result = _run(_tool_run_python({"code": "def"}))
        assert "语法错误" in result or "检查失败" in result

    def test_runtime_error(self):
        result = _run(_tool_run_python({"code": "print(1/0)"}))
        assert "ZeroDivision" in result or "stderr" in result.lower() or "除" in result

    def test_no_output(self):
        result = _run(_tool_run_python({"code": "x = 42"}))
        assert "无输出" in result

    def test_timeout(self):
        code = "import time\ntime.sleep(100)"
        result = _run(_tool_run_python({"code": code, "timeout": 5}))
        assert "超时" in result

    def test_output_truncation(self):
        code = "print('x' * 50000)"
        result = _run(_tool_run_python({"code": code}))
        assert len(result) <= 12_000  # 10000 + truncation message + margin

    def test_stderr_captured(self):
        code = "import sys\nprint('error msg', file=sys.stderr)"
        # sys is blocked by AST check, so this will fail at validation
        result = _run(_tool_run_python({"code": code}))
        assert "检查失败" in result

    def test_timeout_clamp_max(self):
        """timeout > 60 should be clamped to 60."""
        code = "print('ok')"
        result = _run(_tool_run_python({"code": code, "timeout": 999}))
        assert "ok" in result

    def test_timeout_clamp_min(self):
        """timeout < 5 should be clamped to 5."""
        code = "print('ok')"
        result = _run(_tool_run_python({"code": code, "timeout": 1}))
        assert "ok" in result


class TestToolRunPythonInToolSchema:
    """确认 run_python 在 schema 系统中正确注册."""

    def test_in_agent_tools(self):
        from crew.tool_schema import AGENT_TOOLS

        assert "run_python" in AGENT_TOOLS

    def test_in_tool_schemas(self):
        from crew.tool_schema import _TOOL_SCHEMAS

        assert "run_python" in _TOOL_SCHEMAS
        schema = _TOOL_SCHEMAS["run_python"]
        assert schema["name"] == "run_python"
        assert "code" in schema["input_schema"]["properties"]

    def test_in_handlers(self):
        from crew.webhook_tools.engineering import HANDLERS

        assert "run_python" in HANDLERS

    def test_in_dev_tools_preset(self):
        from crew.tool_schema import TOOL_ROLE_PRESETS

        assert "run_python" in TOOL_ROLE_PRESETS["dev-tools"]
