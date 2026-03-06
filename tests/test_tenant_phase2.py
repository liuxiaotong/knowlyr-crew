"""多租户 Phase 2 测试 — webhook_handlers 租户过滤注入.

测试覆盖:
1. _tenant_id_for_store 辅助函数
2. _tenant_id_for_config 辅助函数
3. get_memory_store 的 tenant_id 传递
"""

from crew.tenant import (
    DEFAULT_ADMIN_TENANT_ID,
    TenantContext,
)
from crew.webhook_handlers import _tenant_id_for_config, _tenant_id_for_store


# ── 辅助函数测试 ──


class _FakeState:
    def __init__(self, tenant):
        self.tenant = tenant


class _FakeRequest:
    def __init__(self, tenant=None):
        if tenant is not None:
            self.state = _FakeState(tenant)


class TestTenantIdForStore:
    """_tenant_id_for_store 测试."""

    def test_admin_returns_none(self):
        """admin 租户返回 None（让 store 用默认值）."""
        req = _FakeRequest(
            TenantContext(tenant_id=DEFAULT_ADMIN_TENANT_ID, tenant_name="admin", is_admin=True)
        )
        assert _tenant_id_for_store(req) is None

    def test_regular_tenant_returns_id(self):
        """非 admin 租户返回具体 tenant_id."""
        req = _FakeRequest(
            TenantContext(tenant_id="tenant_abc123", tenant_name="acme")
        )
        assert _tenant_id_for_store(req) == "tenant_abc123"

    def test_no_state_returns_none(self):
        """无 state 时 get_current_tenant 返回 admin → None."""
        req = _FakeRequest()
        assert _tenant_id_for_store(req) is None


class TestTenantIdForConfig:
    """_tenant_id_for_config 测试."""

    def test_admin_returns_admin_id(self):
        """admin 租户返回 DEFAULT_ADMIN_TENANT_ID."""
        req = _FakeRequest(
            TenantContext(tenant_id=DEFAULT_ADMIN_TENANT_ID, tenant_name="admin", is_admin=True)
        )
        assert _tenant_id_for_config(req) == DEFAULT_ADMIN_TENANT_ID

    def test_regular_tenant_returns_id(self):
        """非 admin 租户返回具体 tenant_id."""
        req = _FakeRequest(
            TenantContext(tenant_id="tenant_xyz", tenant_name="corp")
        )
        assert _tenant_id_for_config(req) == "tenant_xyz"

    def test_no_state_returns_admin_id(self):
        """无 state 时返回 admin 的 tenant_id."""
        req = _FakeRequest()
        assert _tenant_id_for_config(req) == DEFAULT_ADMIN_TENANT_ID


# ── get_memory_store tenant_id 传递测试 ──


class TestGetMemoryStoreTenantId:
    """验证 get_memory_store 正确传递 tenant_id."""

    def test_accepts_tenant_id_param(self):
        """get_memory_store 接受 tenant_id 参数."""
        from crew.memory import get_memory_store

        # 文件版不使用 tenant_id 但接受参数不报错
        store = get_memory_store(tenant_id="tenant_test")
        assert store is not None

    def test_none_tenant_id_uses_default(self):
        """tenant_id=None 使用默认值."""
        from crew.memory import get_memory_store

        store = get_memory_store(tenant_id=None)
        assert store is not None
