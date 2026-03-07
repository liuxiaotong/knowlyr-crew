"""测试员工复制 API — copy_employee_to_tenant + _handle_employee_copy."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── config_store.copy_employee_to_tenant 单元测试 ──


class TestCopyEmployeeToTenant:
    """测试 copy_employee_to_tenant 业务逻辑."""

    def _make_source_row(self, **overrides):
        """构造模拟的源员工数据字典."""
        base = {
            "tenant_id": "tenant_admin",
            "name": "backend-engineer",
            "character_name": "后端工程师",
            "display_name": "BE",
            "description": "写后端代码",
            "summary": "",
            "version": "1.0",
            "tags": ["backend", "python"],
            "author": "kai",
            "triggers": ["be"],
            "model": "claude-sonnet-4-6",
            "model_tier": "claude",
            "agent_id": "AI1234",
            "agent_status": "active",
            "avatar_prompt": "a coder",
            "auto_memory": True,
            "kpi": [],
            "bio": "资深后端",
            "domains": ["backend"],
            "temperature": 0.7,
            "max_tokens": 4096,
            "tools": ["bash"],
            "context": [],
            "permissions_json": None,
            "api_key": "",
            "base_url": "",
            "fallback_model": "",
            "fallback_api_key": "",
            "fallback_base_url": "",
            "research_instructions": "",
            "body": "# Backend Engineer",
            "soul_content": "你是后端工程师",
            "soul_version": 5,
            "soul_updated_at": datetime.now(timezone.utc).isoformat(),
            "soul_updated_by": "kai",
            "source_layer": "db",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": "{}",
        }
        base.update(overrides)
        return base

    @patch("crew.config_store.is_pg", return_value=True)
    @patch("crew.config_store.upsert_employee_to_db")
    @patch("crew.config_store.get_employee_from_db")
    @patch("crew.config_store._generate_unique_agent_id", return_value="AI9999")
    def test_basic_copy(self, mock_gen_id, mock_get, mock_upsert, mock_pg):
        """正常复制：从 admin 复制到新租户."""
        from crew.config_store import copy_employee_to_tenant

        source = self._make_source_row()
        # get_employee_from_db: 第一次调用返回 source，第二次返回 None（目标不存在）
        mock_get.side_effect = [source, None]
        mock_upsert.return_value = {
            "name": "backend-engineer",
            "tenant_id": "tenant_abc",
            "character_name": "后端工程师",
            "agent_id": "AI9999",
            "metadata": json.dumps({"source_copied_from": "tenant_admin/backend-engineer"}),
        }

        result = copy_employee_to_tenant(
            source_name="backend-engineer",
            target_tenant_id="tenant_abc",
        )

        assert result["agent_id"] == "AI9999"
        assert result["name"] == "backend-engineer"
        # 验证 upsert 被调用，且 tenant_id 是目标租户
        mock_upsert.assert_called_once()
        call_args = mock_upsert.call_args
        assert call_args[1]["tenant_id"] == "tenant_abc"
        # 验证复制数据中 soul_version 重置为 1
        copy_data = call_args[0][0]
        assert copy_data["soul_version"] == 1
        assert copy_data["source_layer"] == "db"
        assert copy_data["metadata"]["source_copied_from"] == "tenant_admin/backend-engineer"

    @patch("crew.config_store.is_pg", return_value=True)
    @patch("crew.config_store.get_employee_from_db")
    def test_source_not_found(self, mock_get, mock_pg):
        """来源员工不存在时抛出 ValueError."""
        from crew.config_store import copy_employee_to_tenant

        mock_get.return_value = None

        with pytest.raises(ValueError, match="source employee not found"):
            copy_employee_to_tenant(
                source_name="nonexistent",
                target_tenant_id="tenant_abc",
            )

    @patch("crew.config_store.is_pg", return_value=True)
    @patch("crew.config_store.get_employee_from_db")
    def test_name_conflict(self, mock_get, mock_pg):
        """目标租户内名字冲突时抛出 ValueError."""
        from crew.config_store import copy_employee_to_tenant

        source = self._make_source_row()
        existing = {"name": "backend-engineer", "tenant_id": "tenant_abc"}
        mock_get.side_effect = [source, existing]

        with pytest.raises(ValueError, match="already exists"):
            copy_employee_to_tenant(
                source_name="backend-engineer",
                target_tenant_id="tenant_abc",
            )

    @patch("crew.config_store.is_pg", return_value=True)
    @patch("crew.config_store.upsert_employee_to_db")
    @patch("crew.config_store.get_employee_from_db")
    @patch("crew.config_store._generate_unique_agent_id", return_value="AI8888")
    def test_custom_name_and_overrides(self, mock_gen, mock_get, mock_upsert, mock_pg):
        """自定义名字和覆盖字段."""
        from crew.config_store import copy_employee_to_tenant

        source = self._make_source_row()
        mock_get.side_effect = [source, None]
        mock_upsert.return_value = {"name": "my-be", "agent_id": "AI8888"}

        _result = copy_employee_to_tenant(
            source_name="backend-engineer",
            target_tenant_id="tenant_abc",
            new_name="my-be",
            new_character_name="我的后端",
            customizations={
                "description": "自定义描述",
                "model": "gpt-4o",
                "tags": ["custom"],
            },
        )

        call_data = mock_upsert.call_args[0][0]
        assert call_data["name"] == "my-be"
        assert call_data["character_name"] == "我的后端"
        assert call_data["description"] == "自定义描述"
        assert call_data["model"] == "gpt-4o"
        assert call_data["tags"] == ["custom"]
        # 未覆盖的字段保留源值
        assert call_data["bio"] == "资深后端"
        assert call_data["body"] == "# Backend Engineer"

    @patch("crew.config_store.is_pg", return_value=False)
    def test_not_pg_raises(self, mock_pg):
        """非 PG 模式抛出 RuntimeError."""
        from crew.config_store import copy_employee_to_tenant

        with pytest.raises(RuntimeError, match="PG"):
            copy_employee_to_tenant(
                source_name="be",
                target_tenant_id="t",
            )


# ── _handle_employee_copy handler 测试 ──


class TestHandleEmployeeCopy:
    """测试 HTTP handler 层."""

    def _make_request(self, body: dict, tenant_id="tenant_abc", is_admin=False):
        """构造模拟请求对象."""
        from crew.tenant import TenantContext

        req = AsyncMock()
        req.json = AsyncMock(return_value=body)
        req.headers = {}
        req.state = MagicMock()
        req.state.tenant = TenantContext(
            tenant_id=tenant_id,
            tenant_name="test",
            is_admin=is_admin,
        )
        return req

    @pytest.mark.asyncio
    @patch("crew.config_store.copy_employee_to_tenant")
    async def test_successful_copy(self, mock_copy):
        """成功复制返回 201."""
        from crew.webhook_handlers import _handle_employee_copy

        mock_copy.return_value = {
            "name": "my-be",
            "tenant_id": "tenant_abc",
            "character_name": "我的后端",
            "agent_id": "AI9999",
            "metadata": json.dumps({"source_copied_from": "tenant_admin/backend-engineer"}),
        }

        req = self._make_request({"source_name": "backend-engineer", "new_name": "my-be"})
        ctx = MagicMock()
        resp = await _handle_employee_copy(req, ctx)

        assert resp.status_code == 201
        data = json.loads(resp.body)
        assert data["ok"] is True
        assert data["employee"]["agent_id"] == "AI9999"
        assert data["employee"]["source_copied_from"] == "tenant_admin/backend-engineer"

    @pytest.mark.asyncio
    async def test_missing_source_name(self):
        """缺少 source_name 返回 400."""
        from crew.webhook_handlers import _handle_employee_copy

        req = self._make_request({})
        ctx = MagicMock()
        resp = await _handle_employee_copy(req, ctx)

        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_new_name(self):
        """new_name 格式不合法返回 400."""
        from crew.webhook_handlers import _handle_employee_copy

        req = self._make_request({"source_name": "be", "new_name": "Bad Name!"})
        ctx = MagicMock()
        resp = await _handle_employee_copy(req, ctx)

        assert resp.status_code == 400
        data = json.loads(resp.body)
        assert "new_name" in data["error"]

    @pytest.mark.asyncio
    async def test_non_admin_cross_tenant_forbidden(self):
        """非 admin 租户尝试从非 admin 租户复制，返回 403."""
        from crew.webhook_handlers import _handle_employee_copy

        req = self._make_request(
            {"source_name": "be", "source_tenant_id": "tenant_other"},
            is_admin=False,
        )
        ctx = MagicMock()
        resp = await _handle_employee_copy(req, ctx)

        assert resp.status_code == 403

    @pytest.mark.asyncio
    @patch("crew.config_store.copy_employee_to_tenant")
    async def test_source_not_found_returns_404(self, mock_copy):
        """源员工不存在返回 404."""
        from crew.webhook_handlers import _handle_employee_copy

        mock_copy.side_effect = ValueError("source employee not found: tenant_admin/nope")

        req = self._make_request({"source_name": "nope"})
        ctx = MagicMock()
        resp = await _handle_employee_copy(req, ctx)

        assert resp.status_code == 404

    @pytest.mark.asyncio
    @patch("crew.config_store.copy_employee_to_tenant")
    async def test_name_conflict_returns_409(self, mock_copy):
        """名字冲突返回 409."""
        from crew.webhook_handlers import _handle_employee_copy

        mock_copy.side_effect = ValueError("employee already exists in target tenant: t/be")

        req = self._make_request({"source_name": "be"})
        ctx = MagicMock()
        resp = await _handle_employee_copy(req, ctx)

        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_admin_can_copy_from_any_tenant(self):
        """admin 租户可以从任何租户复制（权限检查通过）."""
        from crew.webhook_handlers import _handle_employee_copy

        req = self._make_request(
            {"source_name": "be", "source_tenant_id": "tenant_other"},
            tenant_id="tenant_admin",
            is_admin=True,
        )
        ctx = MagicMock()

        # 会到达 copy 调用，需要 mock
        with patch("crew.config_store.copy_employee_to_tenant") as mock_copy:
            mock_copy.return_value = {
                "name": "be",
                "tenant_id": "tenant_admin",
                "character_name": "BE",
                "agent_id": "AI1111",
                "metadata": json.dumps({"source_copied_from": "tenant_other/be"}),
            }
            resp = await _handle_employee_copy(req, ctx)

        assert resp.status_code == 201
