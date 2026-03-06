"""测试 employees 统一表的 CRUD 操作和 DB 发现模式."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crew.discovery import _db_row_to_employee
from crew.models import Employee


class TestDbRowToEmployee:
    """测试 DB 行到 Employee 对象的转换."""

    def test_basic_conversion(self):
        """基本字段转换."""
        row = {
            "name": "test-worker",
            "character_name": "测试员",
            "display_name": "测试显示名",
            "description": "测试描述",
            "summary": "摘要",
            "version": "2.0",
            "tags": ["backend", "python"],
            "author": "kai",
            "triggers": ["tw", "测试"],
            "model": "claude-sonnet-4-6",
            "model_tier": "claude",
            "agent_id": "AI1234",
            "agent_status": "active",
            "avatar_prompt": "a robot",
            "auto_memory": True,
            "kpi": ["code quality"],
            "tools": ["bash", "file_read"],
            "context": [],
            "permissions_json": None,
            "api_key": "",
            "base_url": "",
            "fallback_model": "",
            "fallback_api_key": "",
            "fallback_base_url": "",
            "research_instructions": "",
            "body": "# Test Worker\nDo stuff.",
            "source_layer": "db",
        }
        emp = _db_row_to_employee(row)
        assert isinstance(emp, Employee)
        assert emp.name == "test-worker"
        assert emp.character_name == "测试员"
        assert emp.tags == ["backend", "python"]
        assert emp.agent_id == "AI1234"
        assert emp.auto_memory is True
        assert emp.source_path is None
        assert emp.source_layer == "db"
        assert emp.body == "# Test Worker\nDo stuff."

    def test_with_permissions(self):
        """带权限策略的转换."""
        row = {
            "name": "perm-worker",
            "character_name": "",
            "display_name": "",
            "description": "has perms",
            "summary": "",
            "version": "1.0",
            "tags": [],
            "author": "",
            "triggers": [],
            "model": "",
            "model_tier": "",
            "agent_id": None,
            "agent_status": "active",
            "avatar_prompt": "",
            "auto_memory": False,
            "kpi": [],
            "tools": [],
            "context": [],
            "permissions_json": json.dumps({"roles": ["memory"], "allow": ["bash"], "deny": []}),
            "api_key": "",
            "base_url": "",
            "fallback_model": "",
            "fallback_api_key": "",
            "fallback_base_url": "",
            "research_instructions": "",
            "body": "body",
            "source_layer": "builtin",
        }
        emp = _db_row_to_employee(row)
        assert emp.permissions is not None
        assert emp.permissions.roles == ["memory"]
        assert emp.permissions.allow == ["bash"]

    def test_minimal_row(self):
        """最小数据转换（所有字段取默认值）."""
        row = {
            "name": "minimal",
            "description": "minimal",
            "body": "",
        }
        emp = _db_row_to_employee(row)
        assert emp.name == "minimal"
        assert emp.tags == []
        assert emp.source_path is None

    def test_none_tags_becomes_empty_list(self):
        """tags 为 None 时应转为空列表."""
        row = {
            "name": "none-tags",
            "description": "test",
            "body": "",
            "tags": None,
            "triggers": None,
            "tools": None,
            "context": None,
            "kpi": None,
        }
        emp = _db_row_to_employee(row)
        assert emp.tags == []
        assert emp.triggers == []
        assert emp.tools == []
        assert emp.kpi == []


class TestDiscoverEmployeesFeatureFlag:
    """测试 CREW_EMPLOYEE_SOURCE feature flag."""

    def test_filesystem_mode_uses_uncached(self):
        """filesystem 模式应调用文件系统扫描."""
        with patch.dict(os.environ, {"CREW_EMPLOYEE_SOURCE": "filesystem"}):
            from crew.discovery import discover_employees

            result = discover_employees(project_dir=Path("/nonexistent"), cache_ttl=0)
            # 应返回内置员工
            assert len(result.employees) >= 5
            assert "code-reviewer" in result.employees

    def test_db_mode_falls_back_when_not_pg(self):
        """db 模式在非 PG 环境下应回退到文件系统."""
        with patch.dict(os.environ, {"CREW_EMPLOYEE_SOURCE": "db", "CREW_USE_SQLITE": "1"}):
            from crew.discovery import discover_employees

            result = discover_employees(project_dir=Path("/nonexistent"), cache_ttl=0)
            # 应回退到文件系统，返回内置员工
            assert len(result.employees) >= 5


class TestEmployeeColumnsConsistency:
    """测试 _EMPLOYEE_COLUMNS 与 employees 表 schema 一致性."""

    def test_columns_tuple_length(self):
        """_EMPLOYEE_COLUMNS 应包含所有表列."""
        from crew.config_store import _EMPLOYEE_COLUMNS

        # 39 列（包含 PK tenant_id + name 和所有数据列）
        assert len(_EMPLOYEE_COLUMNS) == 39

    def test_columns_include_primary_keys(self):
        """应包含主键字段."""
        from crew.config_store import _EMPLOYEE_COLUMNS

        assert "tenant_id" in _EMPLOYEE_COLUMNS
        assert "name" in _EMPLOYEE_COLUMNS

    def test_columns_include_soul_fields(self):
        """应包含 soul 相关字段."""
        from crew.config_store import _EMPLOYEE_COLUMNS

        assert "soul_content" in _EMPLOYEE_COLUMNS
        assert "soul_version" in _EMPLOYEE_COLUMNS
        assert "soul_updated_at" in _EMPLOYEE_COLUMNS
        assert "soul_updated_by" in _EMPLOYEE_COLUMNS

    def test_columns_include_extra_fields(self):
        """应包含 bio/temperature/max_tokens/domains 等额外字段."""
        from crew.config_store import _EMPLOYEE_COLUMNS

        assert "bio" in _EMPLOYEE_COLUMNS
        assert "temperature" in _EMPLOYEE_COLUMNS
        assert "max_tokens" in _EMPLOYEE_COLUMNS
        assert "domains" in _EMPLOYEE_COLUMNS


class TestEmployeeRowToDict:
    """测试行到字典的转换."""

    def test_basic_row_to_dict(self):
        from crew.config_store import _EMPLOYEE_COLUMNS, _employee_row_to_dict

        # 构造一个模拟行（元组形式）
        values = [f"val_{i}" for i in range(len(_EMPLOYEE_COLUMNS))]
        values[0] = "tenant_admin"  # tenant_id
        values[1] = "test-emp"  # name
        result = _employee_row_to_dict(tuple(values))
        assert result["tenant_id"] == "tenant_admin"
        assert result["name"] == "test-emp"
