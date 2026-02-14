"""测试 MCP Server."""

import asyncio
import json

import pytest

mcp = pytest.importorskip("mcp")

from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    GetPromptRequest,
    GetPromptRequestParams,
    ListPromptsRequest,
    ListResourcesRequest,
    ListToolsRequest,
    ReadResourceRequest,
    ReadResourceRequestParams,
)

from crew.exceptions import EmployeeNotFoundError
from crew.mcp_server import create_server


def _run(coro):
    """同步运行 async 函数."""
    return asyncio.run(coro)


class TestMCPServerCreation:
    """测试 MCP Server 创建."""

    def test_create_server(self):
        """应能创建 MCP Server."""
        server = create_server()
        assert server is not None

    def test_tools_registered(self):
        """应注册 tools handler."""
        server = create_server()
        assert ListToolsRequest in server.request_handlers

    def test_prompts_registered(self):
        """应注册 prompts handler."""
        server = create_server()
        assert ListPromptsRequest in server.request_handlers
        assert GetPromptRequest in server.request_handlers

    def test_resources_registered(self):
        """应注册 resources handler."""
        server = create_server()
        assert ListResourcesRequest in server.request_handlers
        assert ReadResourceRequest in server.request_handlers


class TestMCPTools:
    """测试 MCP 工具调用."""

    def setup_method(self):
        self.server = create_server()

    def test_list_employees(self):
        """list_employees 应返回员工列表."""
        handler = self.server.request_handlers[CallToolRequest]
        result = _run(handler(CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name="list_employees", arguments={}),
        )))
        data = json.loads(result.root.content[0].text)
        assert len(data) >= 5
        names = [e["name"] for e in data]
        assert "code-reviewer" in names

    def test_run_employee(self):
        """run_employee 应返回渲染后的 prompt."""
        handler = self.server.request_handlers[CallToolRequest]
        result = _run(handler(CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(
                name="run_employee",
                arguments={"name": "code-reviewer", "args": {"target": "main"}},
            ),
        )))
        text = result.root.content[0].text
        assert "代码审查员" in text
        assert "main" in text


class TestMCPPrompts:
    """测试 MCP Prompts."""

    def setup_method(self):
        self.server = create_server()

    def test_list_prompts(self):
        """list_prompts 应返回所有员工."""
        handler = self.server.request_handlers[ListPromptsRequest]
        result = _run(handler(ListPromptsRequest(method="prompts/list")))
        prompts = result.root.prompts
        assert len(prompts) >= 5
        names = [p.name for p in prompts]
        assert "code-reviewer" in names
        assert "test-engineer" in names

    def test_list_prompts_has_arguments(self):
        """每个 prompt 应包含正确的参数定义."""
        handler = self.server.request_handlers[ListPromptsRequest]
        result = _run(handler(ListPromptsRequest(method="prompts/list")))
        cr = next(p for p in result.root.prompts if p.name == "code-reviewer")
        arg_names = [a.name for a in cr.arguments]
        assert "target" in arg_names
        assert "focus" in arg_names

    def test_get_prompt(self):
        """get_prompt 应返回渲染后的 prompt 消息."""
        handler = self.server.request_handlers[GetPromptRequest]
        result = _run(handler(GetPromptRequest(
            method="prompts/get",
            params=GetPromptRequestParams(
                name="code-reviewer",
                arguments={"target": "main"},
            ),
        )))
        prompt_result = result.root
        assert prompt_result.description
        assert len(prompt_result.messages) == 1
        assert "main" in prompt_result.messages[0].content.text

    def test_get_prompt_not_found(self):
        """不存在的 prompt 应报错."""
        handler = self.server.request_handlers[GetPromptRequest]
        with pytest.raises(EmployeeNotFoundError, match="未找到"):
            _run(handler(GetPromptRequest(
                method="prompts/get",
                params=GetPromptRequestParams(name="nonexistent"),
            )))


class TestMCPResources:
    """测试 MCP Resources."""

    def setup_method(self):
        self.server = create_server()

    def test_list_resources(self):
        """list_resources 应返回所有员工定义."""
        handler = self.server.request_handlers[ListResourcesRequest]
        result = _run(handler(ListResourcesRequest(method="resources/list")))
        resources = result.root.resources
        assert len(resources) >= 5
        uris = [str(r.uri) for r in resources]
        assert "crew://employee/code-reviewer" in uris

    def test_read_resource(self):
        """read_resource 应返回 Markdown 内容."""
        handler = self.server.request_handlers[ReadResourceRequest]
        result = _run(handler(ReadResourceRequest(
            method="resources/read",
            params=ReadResourceRequestParams(uri="crew://employee/code-reviewer"),
        )))
        contents = result.root.contents
        assert len(contents) == 1
        assert "角色定义" in contents[0].text
        assert "text/markdown" in contents[0].mimeType

    def test_read_resource_not_found(self):
        """不存在的资源应报错."""
        handler = self.server.request_handlers[ReadResourceRequest]
        with pytest.raises(EmployeeNotFoundError, match="未找到"):
            _run(handler(ReadResourceRequest(
                method="resources/read",
                params=ReadResourceRequestParams(uri="crew://employee/nonexistent"),
            )))
