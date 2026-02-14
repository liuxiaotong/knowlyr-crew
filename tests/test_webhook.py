"""测试 Webhook 服务器."""

import hashlib
import hmac
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

pytest.importorskip("starlette")

from starlette.testclient import TestClient

from crew.task_registry import TaskRecord, TaskRegistry
from crew.webhook import create_webhook_app
from crew.webhook_config import (
    RouteTarget,
    WebhookConfig,
    WebhookRoute,
    load_webhook_config,
    match_route,
    resolve_target_args,
    resolve_template,
    verify_github_signature,
)


# ── WebhookConfig 测试 ──


class TestVerifyGithubSignature:
    """GitHub HMAC-SHA256 签名验证."""

    def test_valid_signature(self):
        secret = "mysecret"
        body = b'{"action": "opened"}'
        sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        assert verify_github_signature(body, sig, secret) is True

    def test_invalid_signature(self):
        body = b'{"action": "opened"}'
        assert verify_github_signature(body, "sha256=bad", "mysecret") is False

    def test_missing_signature(self):
        assert verify_github_signature(b"body", None, "mysecret") is False

    def test_empty_secret(self):
        assert verify_github_signature(b"body", "sha256=abc", "") is False

    def test_wrong_prefix(self):
        assert verify_github_signature(b"body", "md5=abc", "secret") is False


class TestResolveTemplate:
    """模板解析."""

    def test_simple_key(self):
        assert resolve_template("{{ref}}", {"ref": "refs/heads/main"}) == "refs/heads/main"

    def test_nested_key(self):
        payload = {"pull_request": {"head": {"ref": "feat/x"}}}
        assert resolve_template("{{pull_request.head.ref}}", payload) == "feat/x"

    def test_missing_key_preserved(self):
        assert resolve_template("{{missing}}", {}) == "{{missing}}"

    def test_mixed_text(self):
        result = resolve_template("branch: {{ref}}", {"ref": "main"})
        assert result == "branch: main"

    def test_multiple_templates(self):
        payload = {"ref": "main", "action": "push"}
        result = resolve_template("{{ref}} - {{action}}", payload)
        assert result == "main - push"

    def test_no_templates(self):
        assert resolve_template("plain text", {}) == "plain text"


class TestMatchRoute:
    """路由匹配."""

    def _config(self, routes):
        return WebhookConfig(routes=[WebhookRoute(**r) for r in routes])

    def test_exact_match(self):
        config = self._config([
            {"event": "push", "target": {"type": "pipeline", "name": "review"}},
        ])
        route = match_route("push", config)
        assert route is not None
        assert route.target.name == "review"

    def test_wildcard_match(self):
        config = self._config([
            {"event": "*", "target": {"type": "employee", "name": "code-reviewer"}},
        ])
        route = match_route("anything", config)
        assert route is not None
        assert route.target.name == "code-reviewer"

    def test_no_match(self):
        config = self._config([
            {"event": "push", "target": {"type": "pipeline", "name": "review"}},
        ])
        assert match_route("pull_request", config) is None

    def test_first_match_wins(self):
        config = self._config([
            {"event": "push", "target": {"type": "pipeline", "name": "first"}},
            {"event": "push", "target": {"type": "pipeline", "name": "second"}},
        ])
        route = match_route("push", config)
        assert route.target.name == "first"


class TestResolveTargetArgs:
    """路由目标参数模板解析."""

    def test_resolve_args(self):
        target = RouteTarget(type="pipeline", name="review", args={"target": "{{ref}}"})
        result = resolve_target_args(target, {"ref": "main"})
        assert result == {"target": "main"}

    def test_no_args(self):
        target = RouteTarget(type="pipeline", name="review")
        result = resolve_target_args(target, {"ref": "main"})
        assert result == {}


class TestLoadWebhookConfig:
    """加载 webhook 配置."""

    def test_no_config_file(self, tmp_path):
        config = load_webhook_config(tmp_path)
        assert config.routes == []
        assert config.github_secret == ""

    def test_valid_config(self, tmp_path):
        crew_dir = tmp_path / ".crew"
        crew_dir.mkdir()
        data = {
            "github_secret": "secret123",
            "routes": [
                {
                    "event": "push",
                    "target": {"type": "pipeline", "name": "full-review", "args": {"target": "{{ref}}"}},
                },
            ],
        }
        (crew_dir / "webhook.yaml").write_text(yaml.dump(data, allow_unicode=True))
        config = load_webhook_config(tmp_path)
        assert config.github_secret == "secret123"
        assert len(config.routes) == 1
        assert config.routes[0].target.name == "full-review"


# ── TaskRegistry 测试 ──


class TestTaskRegistry:
    """任务注册表."""

    def test_create(self):
        reg = TaskRegistry()
        record = reg.create("github", "pipeline", "full-review", {"target": "main"})
        assert record.status == "pending"
        assert record.trigger == "github"
        assert record.target_name == "full-review"

    def test_update(self):
        reg = TaskRegistry()
        record = reg.create("direct", "employee", "code-reviewer")
        updated = reg.update(record.task_id, "running")
        assert updated.status == "running"

    def test_update_completed(self):
        reg = TaskRegistry()
        record = reg.create("direct", "employee", "code-reviewer")
        reg.update(record.task_id, "completed", result={"output": "done"})
        final = reg.get(record.task_id)
        assert final.status == "completed"
        assert final.completed_at is not None
        assert final.result == {"output": "done"}

    def test_update_failed(self):
        reg = TaskRegistry()
        record = reg.create("direct", "employee", "code-reviewer")
        reg.update(record.task_id, "failed", error="boom")
        final = reg.get(record.task_id)
        assert final.status == "failed"
        assert final.error == "boom"

    def test_get_missing(self):
        reg = TaskRegistry()
        assert reg.get("nonexistent") is None

    def test_update_missing(self):
        reg = TaskRegistry()
        assert reg.update("nonexistent", "running") is None


# ── Webhook App 端点测试 ──


TOKEN = "test-token-123"


def _make_client(config=None, token=TOKEN):
    """创建测试客户端."""
    app = create_webhook_app(
        project_dir=Path("/tmp/test"),
        token=token,
        config=config or WebhookConfig(),
    )
    return TestClient(app)


class TestHealthEndpoint:
    """健康检查端点."""

    def test_health_no_auth(self):
        client = _make_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_with_auth(self):
        client = _make_client()
        resp = client.get("/health", headers={"Authorization": f"Bearer {TOKEN}"})
        assert resp.status_code == 200


class TestGithubWebhook:
    """GitHub webhook 端点."""

    def _github_headers(self, body: bytes, secret: str, event: str = "push"):
        sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        return {
            "X-GitHub-Event": event,
            "X-Hub-Signature-256": sig,
            "Content-Type": "application/json",
        }

    def test_no_matching_route(self):
        config = WebhookConfig(github_secret="secret")
        client = _make_client(config=config)
        body = json.dumps({"ref": "refs/heads/main"}).encode()
        headers = self._github_headers(body, "secret", "push")
        resp = client.post("/webhook/github", content=body, headers=headers)
        assert resp.status_code == 200
        assert resp.json()["message"] == "no matching route"

    def test_invalid_signature(self):
        config = WebhookConfig(github_secret="secret")
        client = _make_client(config=config)
        body = b'{"ref": "main"}'
        headers = {
            "X-GitHub-Event": "push",
            "X-Hub-Signature-256": "sha256=invalid",
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook/github", content=body, headers=headers)
        assert resp.status_code == 401

    def test_missing_event_header(self):
        config = WebhookConfig()
        client = _make_client(config=config)
        resp = client.post(
            "/webhook/github",
            json={"ref": "main"},
        )
        assert resp.status_code == 400

    @patch("crew.webhook._execute_task", new_callable=AsyncMock)
    def test_matching_route_dispatches(self, mock_execute):
        config = WebhookConfig(
            github_secret="secret",
            routes=[
                WebhookRoute(
                    event="push",
                    target=RouteTarget(type="pipeline", name="review", args={"target": "{{ref}}"}),
                ),
            ],
        )
        client = _make_client(config=config)
        body = json.dumps({"ref": "refs/heads/main"}).encode()
        headers = self._github_headers(body, "secret", "push")
        resp = client.post("/webhook/github", content=body, headers=headers)
        assert resp.status_code == 202
        assert "task_id" in resp.json()


class TestGenericWebhook:
    """通用 webhook 端点."""

    def test_missing_target_name(self):
        client = _make_client()
        resp = client.post(
            "/webhook",
            json={"target_type": "pipeline"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 400

    @patch("crew.webhook._execute_task", new_callable=AsyncMock)
    def test_async_dispatch(self, mock_execute):
        client = _make_client()
        resp = client.post(
            "/webhook",
            json={"target_type": "pipeline", "target_name": "full-review", "args": {"target": "main"}},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "pending"

    def test_requires_auth(self):
        client = _make_client()
        resp = client.post(
            "/webhook",
            json={"target_type": "pipeline", "target_name": "full-review"},
        )
        assert resp.status_code == 401


class TestOpenClawWebhook:
    """OpenClaw webhook 端点."""

    def test_missing_target_name(self):
        client = _make_client()
        resp = client.post(
            "/webhook/openclaw",
            json={"target_type": "employee"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 400

    @patch("crew.webhook._execute_task", new_callable=AsyncMock)
    def test_dispatch(self, mock_execute):
        client = _make_client()
        resp = client.post(
            "/webhook/openclaw",
            json={"target_type": "employee", "target_name": "code-reviewer", "args": {"target": "main"}},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 202


class TestRunPipeline:
    """直接触发 pipeline."""

    @patch("crew.webhook._execute_task", new_callable=AsyncMock)
    def test_async_run(self, mock_execute):
        client = _make_client()
        resp = client.post(
            "/run/pipeline/full-review",
            json={"args": {"target": "main"}},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 202
        assert "task_id" in resp.json()

    def test_requires_auth(self):
        client = _make_client()
        resp = client.post(
            "/run/pipeline/full-review",
            json={"args": {"target": "main"}},
        )
        assert resp.status_code == 401


class TestRunEmployee:
    """直接触发员工."""

    @patch("crew.webhook._execute_task", new_callable=AsyncMock)
    def test_async_run(self, mock_execute):
        client = _make_client()
        resp = client.post(
            "/run/employee/code-reviewer",
            json={"args": {"target": "main"}},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 202
        assert "task_id" in resp.json()


class TestTaskStatus:
    """任务状态查询."""

    @patch("crew.webhook._execute_task", new_callable=AsyncMock)
    def test_query_existing_task(self, mock_execute):
        client = _make_client()
        # 先创建一个任务
        resp = client.post(
            "/run/pipeline/test",
            json={"args": {}},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        task_id = resp.json()["task_id"]

        # 查询任务
        resp = client.get(
            f"/tasks/{task_id}",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == task_id

    def test_query_missing_task(self):
        client = _make_client()
        resp = client.get(
            "/tasks/nonexistent",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 404


class TestSyncExecution:
    """同步执行模式."""

    @patch("crew.webhook._execute_pipeline", new_callable=AsyncMock)
    def test_sync_pipeline(self, mock_pipeline):
        mock_pipeline.return_value = {"output": "review done"}
        client = _make_client()
        resp = client.post(
            "/run/pipeline/full-review",
            json={"args": {"target": "main"}, "sync": True},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["result"] == {"output": "review done"}

    @patch("crew.webhook._execute_employee", new_callable=AsyncMock)
    def test_sync_employee(self, mock_employee):
        mock_employee.return_value = {"employee": "code-reviewer", "output": "lgtm"}
        client = _make_client()
        resp = client.post(
            "/run/employee/code-reviewer",
            json={"args": {"target": "main"}, "sync": True},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"

    @patch("crew.webhook._execute_employee", new_callable=AsyncMock)
    def test_sync_failure(self, mock_employee):
        mock_employee.side_effect = ValueError("员工不存在")
        client = _make_client()
        resp = client.post(
            "/run/employee/bad",
            json={"args": {}, "sync": True},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "员工不存在" in data["error"]


class TestNoAuth:
    """不启用认证."""

    @patch("crew.webhook._execute_task", new_callable=AsyncMock)
    def test_no_token_allows_all(self, mock_execute):
        client = _make_client(token=None)
        resp = client.post(
            "/webhook",
            json={"target_type": "pipeline", "target_name": "test"},
        )
        assert resp.status_code == 202
