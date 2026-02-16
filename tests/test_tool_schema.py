"""tool_schema.py 测试."""

from crew.tool_schema import (
    CREW_TO_SANDBOX,
    employee_tools_to_schemas,
    is_finish_tool,
    map_tool_call,
)


class TestEmployeeToolsToSchemas:
    def test_basic_tools(self):
        schemas, _ = employee_tools_to_schemas(["file_read", "bash"])
        names = {s["name"] for s in schemas}
        assert "file_read" in names
        assert "bash" in names
        assert "submit" in names  # 始终包含

    def test_always_includes_submit(self):
        schemas, _ = employee_tools_to_schemas([])
        assert len(schemas) == 1
        assert schemas[0]["name"] == "submit"

    def test_no_duplicates(self):
        schemas, _ = employee_tools_to_schemas(["file_read", "file_read", "bash"])
        names = [s["name"] for s in schemas]
        assert names.count("file_read") == 1

    def test_all_tools(self):
        all_tools = ["file_read", "file_write", "bash", "git", "grep", "glob"]
        schemas, _ = employee_tools_to_schemas(all_tools)
        # 6 tools + submit
        assert len(schemas) == 7

    def test_schema_format(self):
        schemas, _ = employee_tools_to_schemas(["file_read"])
        schema = [s for s in schemas if s["name"] == "file_read"][0]
        assert "description" in schema
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"
        assert "path" in schema["input_schema"]["properties"]

    def test_unknown_tool_skipped(self):
        schemas, _ = employee_tools_to_schemas(["file_read", "nonexistent"])
        names = {s["name"] for s in schemas}
        assert "nonexistent" not in names
        assert "file_read" in names


class TestMapToolCall:
    def test_direct_mapping(self):
        result = map_tool_call("file_read", {"path": "/tmp/foo.py"})
        assert result == {"tool": "file_read", "params": {"path": "/tmp/foo.py"}}

    def test_bash_to_shell(self):
        result = map_tool_call("bash", {"command": "ls -la"})
        assert result["tool"] == "shell"
        assert result["params"]["command"] == "ls -la"

    def test_grep_to_search(self):
        result = map_tool_call("grep", {"pattern": "def foo", "path": "src/"})
        assert result["tool"] == "search"

    def test_glob_to_search(self):
        result = map_tool_call("glob", {"pattern": "*.py", "path": "."})
        assert result["tool"] == "search"
        assert result["params"]["file_pattern"] == "*.py"

    def test_git_passthrough(self):
        result = map_tool_call("git", {"subcommand": "diff", "args": "HEAD~1"})
        assert result["tool"] == "git"

    def test_unknown_tool_passthrough(self):
        result = map_tool_call("submit", {"result": "done"})
        assert result["tool"] == "submit"


class TestIsFinishTool:
    def test_submit(self):
        assert is_finish_tool("submit") is True

    def test_finish(self):
        assert is_finish_tool("finish") is True

    def test_not_finish(self):
        assert is_finish_tool("file_read") is False
        assert is_finish_tool("bash") is False


class TestCrewToSandbox:
    def test_mapping_keys(self):
        assert "file_read" in CREW_TO_SANDBOX
        assert "bash" in CREW_TO_SANDBOX
        assert CREW_TO_SANDBOX["bash"] == "shell"
        assert CREW_TO_SANDBOX["grep"] == "search"
