"""测试 Webhook 服务器."""

import asyncio
import hashlib
import hmac
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

pytest.importorskip("starlette")

from starlette.testclient import TestClient

from crew.task_registry import TaskRegistry
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
        config = self._config(
            [
                {"event": "push", "target": {"type": "pipeline", "name": "review"}},
            ]
        )
        route = match_route("push", config)
        assert route is not None
        assert route.target.name == "review"

    def test_wildcard_match(self):
        config = self._config(
            [
                {"event": "*", "target": {"type": "employee", "name": "code-reviewer"}},
            ]
        )
        route = match_route("anything", config)
        assert route is not None
        assert route.target.name == "code-reviewer"

    def test_no_match(self):
        config = self._config(
            [
                {"event": "push", "target": {"type": "pipeline", "name": "review"}},
            ]
        )
        assert match_route("pull_request", config) is None

    def test_first_match_wins(self):
        config = self._config(
            [
                {"event": "push", "target": {"type": "pipeline", "name": "first"}},
                {"event": "push", "target": {"type": "pipeline", "name": "second"}},
            ]
        )
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
                    "target": {
                        "type": "pipeline",
                        "name": "full-review",
                        "args": {"target": "{{ref}}"},
                    },
                },
            ],
        }
        (crew_dir / "webhook.yaml").write_text(yaml.dump(data, allow_unicode=True))
        config = load_webhook_config(tmp_path)
        assert config.github_secret == "secret123"
        assert len(config.routes) == 1
        assert config.routes[0].target.name == "full-review"

    def test_empty_yaml_returns_default(self, tmp_path):
        """空 YAML 文件应返回默认配置."""
        crew_dir = tmp_path / ".crew"
        crew_dir.mkdir()
        (crew_dir / "webhook.yaml").write_text("")
        config = load_webhook_config(tmp_path)
        assert config.routes == []

    def test_non_dict_yaml_returns_default(self, tmp_path):
        """YAML 内容为列表时应返回默认配置."""
        crew_dir = tmp_path / ".crew"
        crew_dir.mkdir()
        (crew_dir / "webhook.yaml").write_text("- item1\n- item2\n")
        config = load_webhook_config(tmp_path)
        assert config.routes == []
        assert config.github_secret == ""


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


class TestTaskPersistence:
    """任务持久化到 JSONL."""

    def test_persist_and_recover(self, tmp_path):
        """创建任务后重启应恢复."""
        jsonl = tmp_path / "tasks.jsonl"
        reg = TaskRegistry(persist_path=jsonl)
        r1 = reg.create("direct", "employee", "test-emp")
        reg.update(r1.task_id, "completed", result={"output": "ok"})
        r2 = reg.create("cron", "pipeline", "daily-review")

        # 模拟重启
        reg2 = TaskRegistry(persist_path=jsonl)
        assert reg2.get(r1.task_id) is not None
        assert reg2.get(r1.task_id).status == "completed"
        assert reg2.get(r2.task_id) is not None
        assert reg2.get(r2.task_id).status == "pending"

    def test_no_persist_without_path(self):
        """不传 persist_path 时不写文件."""
        reg = TaskRegistry()
        reg.create("direct", "employee", "test")
        # 不报错即可

    def test_compact_history(self, tmp_path):
        """超过 max_history 应截断旧记录."""
        jsonl = tmp_path / "tasks.jsonl"
        reg = TaskRegistry(persist_path=jsonl, max_history=5)
        ids = []
        for i in range(8):
            r = reg.create("direct", "employee", f"emp-{i}")
            ids.append(r.task_id)

        # 内存中应只保留最新 5 条
        assert len(reg._tasks) == 5
        # 旧的 3 条被清理
        assert reg.get(ids[0]) is None
        assert reg.get(ids[1]) is None
        assert reg.get(ids[2]) is None
        # 新的 5 条存在
        assert reg.get(ids[7]) is not None

    def test_list_recent(self):
        """list_recent 返回最近的任务."""
        reg = TaskRegistry()
        for i in range(5):
            reg.create("direct", "employee", f"emp-{i}")
        recent = reg.list_recent(3)
        assert len(recent) == 3

    def test_corrupt_jsonl_recovery(self, tmp_path):
        """损坏的 JSONL 行应被跳过."""
        jsonl = tmp_path / "tasks.jsonl"
        jsonl.write_text('{"invalid": "json"}\n{"also": "bad\n', encoding="utf-8")
        reg = TaskRegistry(persist_path=jsonl)
        assert len(reg._tasks) == 0  # 损坏行全部跳过，不崩溃

    def test_corrupt_jsonl_logs_warning(self, tmp_path, caplog):
        """损坏的 JSONL 记录应产生 warning 日志."""
        import logging

        path = tmp_path / "tasks.jsonl"
        path.write_text("not-valid-json\n")
        with caplog.at_level(logging.WARNING, logger="crew.task_registry"):
            registry = TaskRegistry(persist_path=path)
        assert "跳过无效" in caplog.text


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


class TestCORS:
    """CORS 中间件测试."""

    def test_cors_preflight(self):
        """OPTIONS 预检请求应返回 CORS 头."""
        app = create_webhook_app(
            project_dir=Path("/tmp/test"),
            token=TOKEN,
            cors_origins=["https://antgather.knowlyr.com"],
        )
        client = TestClient(app)
        resp = client.options(
            "/run/employee/test",
            headers={
                "Origin": "https://antgather.knowlyr.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization,Content-Type",
            },
        )
        assert resp.status_code == 200
        assert "https://antgather.knowlyr.com" in resp.headers.get(
            "access-control-allow-origin", ""
        )
        assert "POST" in resp.headers.get("access-control-allow-methods", "")

    def test_cors_actual_request(self):
        """带 Origin 的实际请求应包含 CORS 头."""
        app = create_webhook_app(
            project_dir=Path("/tmp/test"),
            token=TOKEN,
            cors_origins=["https://antgather.knowlyr.com"],
        )
        client = TestClient(app)
        resp = client.get(
            "/health",
            headers={"Origin": "https://antgather.knowlyr.com"},
        )
        assert resp.status_code == 200
        assert "https://antgather.knowlyr.com" in resp.headers.get(
            "access-control-allow-origin", ""
        )

    def test_cors_disallowed_origin(self):
        """不在白名单中的 origin 不应返回 CORS 头."""
        app = create_webhook_app(
            project_dir=Path("/tmp/test"),
            token=TOKEN,
            cors_origins=["https://antgather.knowlyr.com"],
        )
        client = TestClient(app)
        resp = client.get(
            "/health",
            headers={"Origin": "https://evil.com"},
        )
        assert resp.status_code == 200
        assert "evil.com" not in resp.headers.get("access-control-allow-origin", "")

    def test_no_cors_by_default(self):
        """不配置 cors_origins 时不启用 CORS."""
        client = _make_client()
        resp = client.get(
            "/health",
            headers={"Origin": "https://antgather.knowlyr.com"},
        )
        assert resp.status_code == 200
        assert "access-control-allow-origin" not in resp.headers

    def test_multiple_origins(self):
        """支持多个 origin."""
        app = create_webhook_app(
            project_dir=Path("/tmp/test"),
            token=TOKEN,
            cors_origins=["https://antgather.knowlyr.com", "http://localhost:3000"],
        )
        client = TestClient(app)
        resp = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )
        assert "localhost:3000" in resp.headers.get("access-control-allow-origin", "")


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

    def test_no_secret_rejects(self):
        """未配置 github_secret 时应拒绝请求."""
        config = WebhookConfig()
        client = _make_client(config=config)
        resp = client.post(
            "/webhook/github",
            json={"ref": "main"},
        )
        assert resp.status_code == 403

    def test_missing_event_header(self):
        config = WebhookConfig(github_secret="test-secret")
        client = _make_client(config=config)
        body = b'{"ref": "main"}'
        sig = "sha256=" + hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
        resp = client.post(
            "/webhook/github",
            content=body,
            headers={"x-hub-signature-256": sig, "content-type": "application/json"},
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
            json={
                "target_type": "pipeline",
                "target_name": "full-review",
                "args": {"target": "main"},
            },
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
            json={
                "target_type": "employee",
                "target_name": "code-reviewer",
                "args": {"target": "main"},
            },
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


class TestStreamEmployee:
    """SSE 流式执行员工."""

    @patch("crew.discovery.discover_employees")
    @patch("crew.engine.CrewEngine")
    @patch("crew.executor.aexecute_prompt", new_callable=AsyncMock)
    def test_stream_returns_sse(self, mock_exec, mock_engine_cls, mock_discover):
        """stream=true 应返回 text/event-stream."""
        from crew.models import DiscoveryResult, Employee

        emp = Employee(name="test-emp", display_name="Test", description="", body="")
        mock_discover.return_value = DiscoveryResult(employees={"test-emp": emp})
        mock_engine = MagicMock()
        mock_engine.prompt.return_value = "test prompt"
        mock_engine_cls.return_value = mock_engine

        # 模拟 async iterator
        async def _fake_stream():
            yield "Hello"
            yield " world"

        _fake_stream.result = MagicMock(
            model="gpt-4o",
            input_tokens=10,
            output_tokens=5,
        )
        mock_exec.return_value = _fake_stream()

        client = _make_client()
        resp = client.post(
            "/run/employee/test-emp",
            json={"args": {}, "stream": True},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        body = resp.text
        assert 'data: {"token": "Hello"}' in body
        assert 'data: {"token": " world"}' in body
        assert "event: done" in body

    @patch("crew.discovery.discover_employees")
    def test_stream_unknown_employee(self, mock_discover):
        """流式请求未知员工应返回 error 事件."""
        from crew.models import DiscoveryResult

        mock_discover.return_value = DiscoveryResult(employees={})

        client = _make_client()
        resp = client.post(
            "/run/employee/nonexistent",
            json={"args": {}, "stream": True},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert "event: error" in resp.text

    @patch("crew.webhook._execute_task", new_callable=AsyncMock)
    def test_no_stream_fallback(self, mock_execute):
        """stream=false 应走原有异步模式."""
        client = _make_client()
        resp = client.post(
            "/run/employee/code-reviewer",
            json={"args": {"target": "main"}, "stream": False},
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


class TestIdentityPassthrough:
    """agent_id 身份透传."""

    @patch("crew.webhook._execute_task", new_callable=AsyncMock)
    def test_employee_passes_agent_id(self, mock_execute):
        """run/employee 应传递 agent_id 到 _execute_task."""
        client = _make_client()
        resp = client.post(
            "/run/employee/code-reviewer",
            json={"args": {"target": "main"}, "agent_id": 42},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 202
        # _execute_task 应收到 agent_id
        mock_execute.assert_called_once()
        call_kwargs = mock_execute.call_args
        assert call_kwargs.kwargs.get("agent_id") == 42

    @patch("crew.webhook._execute_task", new_callable=AsyncMock)
    def test_pipeline_passes_agent_id(self, mock_execute):
        """run/pipeline 应传递 agent_id 到 _execute_task."""
        client = _make_client()
        resp = client.post(
            "/run/pipeline/full-review",
            json={"args": {"target": "main"}, "agent_id": 99},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 202
        mock_execute.assert_called_once()
        call_kwargs = mock_execute.call_args
        assert call_kwargs.kwargs.get("agent_id") == 99

    @patch("crew.webhook._execute_task", new_callable=AsyncMock)
    def test_no_agent_id_defaults_none(self, mock_execute):
        """不传 agent_id 时默认 None."""
        client = _make_client()
        resp = client.post(
            "/run/employee/code-reviewer",
            json={"args": {}},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 202
        mock_execute.assert_called_once()
        call_kwargs = mock_execute.call_args
        assert call_kwargs.kwargs.get("agent_id") is None

class TestTaskReplay:
    """任务重放."""

    @patch("crew.webhook._execute_task", new_callable=AsyncMock)
    def test_replay_completed_task(self, mock_execute):
        client = _make_client()
        # 创建一个任务并通过 execute mock 让它保持 pending
        resp = client.post(
            "/run/pipeline/test",
            json={"args": {"target": "main"}, "sync": True},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        task_id = resp.json()["task_id"]
        # _execute_task 是 mock 的，任务实际状态仍为 completed (sync mode 会更新)
        # 直接 mock registry.get 返回 completed 记录
        mock_record = MagicMock()
        mock_record.status = "completed"
        mock_record.target_type = "pipeline"
        mock_record.target_name = "test-pipe"
        mock_record.args = {"target": "main"}

        with patch.object(TaskRegistry, "get", return_value=mock_record):
            resp = client.post(
                f"/tasks/{task_id}/replay",
                headers={"Authorization": f"Bearer {TOKEN}"},
            )
        assert resp.status_code == 202

    def test_replay_not_found(self):
        client = _make_client()
        resp = client.post(
            "/tasks/nonexistent/replay",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 404

    @patch("crew.webhook._execute_task", new_callable=AsyncMock)
    def test_replay_running_task_rejected(self, mock_execute):
        client = _make_client()
        # 创建一个任务（状态 pending）
        resp = client.post(
            "/run/pipeline/test",
            json={"args": {}},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        task_id = resp.json()["task_id"]

        # 尝试重放 pending 任务 → 应该 400
        resp = client.post(
            f"/tasks/{task_id}/replay",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 400
        assert "只能重放" in resp.json()["error"]


class TestMetricsEndpoint:
    """指标端点."""

    def test_metrics_returns_json(self):
        client = _make_client(token=None)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "calls" in data
        assert "tokens" in data
        assert "uptime_seconds" in data


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


class TestStreamErrorHandling:
    """SSE 流式错误处理."""

    @patch("crew.discovery.discover_employees")
    @patch("crew.engine.CrewEngine")
    @patch("crew.executor.aexecute_prompt", new_callable=AsyncMock)
    def test_stream_timeout(self, mock_exec, mock_engine_cls, mock_discover):
        """aexecute_prompt 超时应返回 error event."""
        from crew.models import DiscoveryResult, Employee

        emp = Employee(name="test-emp", display_name="Test", description="", body="")
        mock_discover.return_value = DiscoveryResult(employees={"test-emp": emp})
        mock_engine = MagicMock()
        mock_engine.prompt.return_value = "test prompt"
        mock_engine_cls.return_value = mock_engine

        # 模拟超时
        mock_exec.side_effect = asyncio.TimeoutError()

        client = _make_client()
        resp = client.post(
            "/run/employee/test-emp",
            json={"args": {}, "stream": True},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        assert "event: error" in resp.text
        assert "timeout" in resp.text.lower()

    @patch("crew.discovery.discover_employees")
    @patch("crew.engine.CrewEngine")
    @patch("crew.executor.aexecute_prompt", new_callable=AsyncMock)
    def test_stream_mid_error(self, mock_exec, mock_engine_cls, mock_discover):
        """流中异常应返回 error event."""
        from crew.models import DiscoveryResult, Employee

        emp = Employee(name="test-emp", display_name="Test", description="", body="")
        mock_discover.return_value = DiscoveryResult(employees={"test-emp": emp})
        mock_engine = MagicMock()
        mock_engine.prompt.return_value = "test prompt"
        mock_engine_cls.return_value = mock_engine

        # 模拟流中途异常
        async def _failing_stream():
            yield "part"
            raise RuntimeError("LLM connection lost")

        mock_exec.return_value = _failing_stream()

        client = _make_client()
        resp = client.post(
            "/run/employee/test-emp",
            json={"args": {}, "stream": True},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        body = resp.text
        # 应有部分 token 和 error event
        assert "event: error" in body

    @patch("crew.discovery.discover_employees")
    @patch("crew.engine.CrewEngine")
    @patch("crew.executor.aexecute_prompt", new_callable=AsyncMock)
    def test_stream_cleanup_on_disconnect(self, mock_exec, mock_engine_cls, mock_discover):
        """流正常完成后有 done event."""
        from crew.models import DiscoveryResult, Employee

        emp = Employee(name="test-emp", display_name="Test", description="", body="")
        mock_discover.return_value = DiscoveryResult(employees={"test-emp": emp})
        mock_engine = MagicMock()
        mock_engine.prompt.return_value = "test prompt"
        mock_engine_cls.return_value = mock_engine

        async def _normal_stream():
            yield "Hello"

        _normal_stream.result = MagicMock(
            model="gpt-4o",
            input_tokens=10,
            output_tokens=5,
        )
        mock_exec.return_value = _normal_stream()

        client = _make_client()
        resp = client.post(
            "/run/employee/test-emp",
            json={"args": {}, "stream": True},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        assert "event: done" in resp.text


class TestEmployeeUpdatePUT:
    """PUT /api/employees/{identifier} — employee.yaml 唯一真相源."""

    def _make_emp(self, tmp_path, name="test-emp", agent_id=3080):
        """创建带 source_path 的 Employee + 真实 employee.yaml."""
        from crew.models import Employee

        emp_dir = tmp_path / f"{name}-{agent_id}"
        emp_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "name": name,
            "description": f"{name} desc",
            "agent_id": agent_id,
            "model": "claude-sonnet-4-20250514",
        }
        (emp_dir / "employee.yaml").write_text(
            yaml.dump(config, allow_unicode=True),
            encoding="utf-8",
        )
        emp = Employee(
            name=name,
            description=f"{name} desc",
            body="test body",
            agent_id=agent_id,
            model="claude-sonnet-4-20250514",
            source_path=emp_dir,
        )
        return emp, emp_dir

    @patch("crew.discovery.discover_employees")
    @patch("crew.webhook_handlers._write_yaml_field")
    def test_update_model(self, mock_write, mock_discover, tmp_path):
        """PUT 应更新 model 到 employee.yaml."""
        from crew.models import DiscoveryResult

        emp, emp_dir = self._make_emp(tmp_path)
        mock_discover.return_value = DiscoveryResult(employees={emp.name: emp})

        client = _make_client(config=None, token=TOKEN)
        resp = client.put(
            f"/api/employees/{emp.name}",
            json={"model": "claude-opus-4-6"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["updated"]["model"] == "claude-opus-4-6"
        mock_write.assert_called_once_with(emp_dir, {"model": "claude-opus-4-6"})

    @patch("crew.discovery.discover_employees")
    @patch("crew.webhook_handlers._write_yaml_field")
    def test_update_by_agent_id(self, mock_write, mock_discover, tmp_path):
        """PUT 可通过 agent_id 查找员工."""
        from crew.models import DiscoveryResult

        emp, emp_dir = self._make_emp(tmp_path, agent_id=3081)
        mock_discover.return_value = DiscoveryResult(employees={emp.name: emp})

        client = _make_client()
        resp = client.put(
            "/api/employees/3081",
            json={"temperature": 0.8},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["updated"]["temperature"] == 0.8

    @patch("crew.discovery.discover_employees")
    def test_update_not_found(self, mock_discover):
        """不存在的员工应返回 404."""
        from crew.models import DiscoveryResult

        mock_discover.return_value = DiscoveryResult(employees={})

        client = _make_client()
        resp = client.put(
            "/api/employees/nonexistent",
            json={"model": "gpt-4o"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 404

    @patch("crew.discovery.discover_employees")
    def test_update_invalid_field(self, mock_discover, tmp_path):
        """不在白名单的字段应被拒绝."""
        from crew.models import DiscoveryResult, Employee

        emp = Employee(
            name="emp",
            description="d",
            body="b",
            agent_id=999,
            source_path=tmp_path,
        )
        mock_discover.return_value = DiscoveryResult(employees={"emp": emp})

        client = _make_client()
        resp = client.put(
            "/api/employees/emp",
            json={"name": "hacked", "description": "pwned"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 400
        assert "No updatable fields" in resp.json()["error"]

    def test_update_no_auth(self):
        """PUT 无认证应返回 401."""
        client = _make_client()
        resp = client.put(
            "/api/employees/some-emp",
            json={"model": "gpt-4o"},
        )
        assert resp.status_code == 401

class TestModelTiersEndpoint:
    """GET /api/model-tiers — 模型档位列表."""

    @patch("crew.organization.load_organization")
    def test_returns_tiers(self, mock_load_org):
        """应返回所有 tier 的 model 和 fallback_model."""
        from crew.models import ModelTierConfig, Organization

        org = Organization(
            model_defaults={
                "claude": ModelTierConfig(
                    model="claude-sonnet-4-5-20250929",
                    api_key="sk-secret",
                    base_url="https://aiberm.com/v1",
                    fallback_model="claude-sonnet-4-20250514",
                    fallback_api_key="sk-fallback-secret",
                ),
                "kimi": ModelTierConfig(
                    model="kimi-k2.5",
                    api_key="km-secret",
                    base_url="https://api.moonshot.cn/v1",
                    fallback_model="kimi-k2.5",
                    fallback_api_key="km-fallback-secret",
                ),
            }
        )
        mock_load_org.return_value = org

        client = _make_client()
        resp = client.get(
            "/api/model-tiers",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "tiers" in data
        assert "claude" in data["tiers"]
        assert "kimi" in data["tiers"]
        assert data["tiers"]["claude"]["model"] == "claude-sonnet-4-5-20250929"
        assert data["tiers"]["claude"]["fallback_model"] == "claude-sonnet-4-20250514"
        assert data["tiers"]["kimi"]["model"] == "kimi-k2.5"
        assert data["tiers"]["kimi"]["fallback_model"] == "kimi-k2.5"

    @patch("crew.organization.load_organization")
    def test_no_sensitive_fields(self, mock_load_org):
        """不应返回 api_key、base_url 等敏感字段."""
        from crew.models import ModelTierConfig, Organization

        org = Organization(
            model_defaults={
                "claude": ModelTierConfig(
                    model="claude-sonnet-4-5-20250929",
                    api_key="sk-super-secret-key",
                    base_url="https://aiberm.com/v1",
                    fallback_model="claude-sonnet-4-20250514",
                    fallback_api_key="sk-fallback-secret",
                    fallback_base_url="https://fallback.aiberm.com/v1",
                ),
            }
        )
        mock_load_org.return_value = org

        client = _make_client()
        resp = client.get(
            "/api/model-tiers",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        tier = data["tiers"]["claude"]
        # 只有 model 和 fallback_model，不应有其他字段
        assert set(tier.keys()) == {"model", "fallback_model"}
        assert "sk-super-secret-key" not in json.dumps(data)
        assert "aiberm.com" not in json.dumps(data)
        assert "sk-fallback-secret" not in json.dumps(data)
        assert "fallback.aiberm.com" not in json.dumps(data)

    @patch("crew.organization.load_organization")
    def test_empty_model_defaults(self, mock_load_org):
        """没有 model_defaults 时应返回空 tiers."""
        from crew.models import Organization

        mock_load_org.return_value = Organization()

        client = _make_client()
        resp = client.get(
            "/api/model-tiers",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"tiers": {}}

    def test_no_auth_rejected(self):
        """无认证应返回 401."""
        client = _make_client()
        resp = client.get("/api/model-tiers")
        assert resp.status_code == 401


class TestModelTierUpdatable:
    """model_tier 应在员工可更新字段白名单中."""

    @patch("crew.discovery.discover_employees")
    @patch("crew.webhook_handlers._write_yaml_field")
    def test_update_model_tier(self, mock_write, mock_discover, tmp_path):
        """PUT 应允许更新 model_tier."""
        from crew.models import DiscoveryResult, Employee

        emp_dir = tmp_path / "test-emp-9000"
        emp_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "name": "test-emp",
            "description": "test desc",
            "agent_id": 9000,
            "model_tier": "claude",
        }
        (emp_dir / "employee.yaml").write_text(
            yaml.dump(config, allow_unicode=True),
            encoding="utf-8",
        )
        emp = Employee(
            name="test-emp",
            description="test desc",
            body="test body",
            agent_id=9000,
            model_tier="claude",
            source_path=emp_dir,
        )
        mock_discover.return_value = DiscoveryResult(employees={"test-emp": emp})

        client = _make_client()
        resp = client.put(
            "/api/employees/test-emp",
            json={"model_tier": "kimi"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["updated"]["model_tier"] == "kimi"
        mock_write.assert_called_once_with(emp_dir, {"model_tier": "kimi"})

    def test_model_tier_in_whitelist(self):
        """model_tier 应在 _EMPLOYEE_UPDATABLE_FIELDS 白名单中."""
        from crew.webhook_context import _EMPLOYEE_UPDATABLE_FIELDS

        assert "model_tier" in _EMPLOYEE_UPDATABLE_FIELDS


class TestCostSummaryEndpoint:
    """GET /api/cost/summary — 成本汇总."""

    def test_cost_summary_returns_json(self):
        """成本汇总应返回有效 JSON."""
        client = _make_client()
        resp = client.get(
            "/api/cost/summary",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "total_cost_usd" in data
        assert "by_employee" in data
        assert "by_model" in data
        assert "period_days" in data

    def test_cost_summary_with_days(self):
        """应支持 days 参数."""
        client = _make_client()
        resp = client.get(
            "/api/cost/summary?days=30",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        assert resp.json()["period_days"] == 30

    def test_cost_summary_requires_auth(self):
        """成本汇总需要认证."""
        client = _make_client()
        resp = client.get("/api/cost/summary")
        assert resp.status_code == 401


class TestProjectStatusEndpoint:
    """GET /api/project/status — 项目状态概览."""

    def test_project_status_returns_json(self):
        """项目状态应返回有效 JSON."""
        client = _make_client()
        resp = client.get(
            "/api/project/status",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "total_employees" in data
        assert "teams" in data
        assert "employees" in data
        assert "cost_7d" in data

    @patch("crew.organization.get_effective_authority", return_value="L2")
    @patch("crew.organization.load_organization")
    @patch("crew.cost.query_cost_summary", return_value={})
    @patch("crew.discovery.discover_employees")
    def test_employees_contain_agent_status(
        self, mock_discover, mock_cost, mock_org, mock_authority, tmp_path
    ):
        """employees 列表中每个员工都应包含 agent_status 字段."""
        from crew.models import DiscoveryResult, Employee, Organization

        emp_a = Employee(
            name="emp-a",
            display_name="A",
            character_name="甲",
            description="员工A",
            body="body a",
            agent_id=1001,
            agent_status="active",
            source_path=tmp_path,
        )
        emp_b = Employee(
            name="emp-b",
            display_name="B",
            character_name="乙",
            description="员工B",
            body="body b",
            agent_id=1002,
            agent_status="frozen",
            source_path=tmp_path,
        )
        mock_discover.return_value = DiscoveryResult(employees={"emp-a": emp_a, "emp-b": emp_b})
        mock_org.return_value = Organization()

        client = _make_client()
        resp = client.get(
            "/api/project/status",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        employees = data["employees"]
        assert len(employees) == 2
        for emp in employees:
            assert "agent_status" in emp, f"{emp['name']} 缺少 agent_status"
        # 验证具体值
        by_name = {e["name"]: e for e in employees}
        assert by_name["emp-a"]["agent_status"] == "active"
        assert by_name["emp-b"]["agent_status"] == "frozen"

    def test_project_status_requires_auth(self):
        """项目状态需要认证."""
        client = _make_client()
        resp = client.get("/api/project/status")
        assert resp.status_code == 401


class TestAuthorityRestoreEndpoint:
    """POST /api/employees/{id}/authority/restore — 权限恢复."""

    @patch("crew.discovery.discover_employees")
    def test_restore_no_override(self, mock_discover):
        """无覆盖记录时返回当前权限."""
        from crew.models import DiscoveryResult, Employee

        emp = Employee(name="test-emp", display_name="Test", description="d", body="b")
        mock_discover.return_value = DiscoveryResult(employees={"test-emp": emp})

        client = _make_client()
        resp = client.post(
            "/api/employees/test-emp/authority/restore",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "无覆盖记录" in data["message"]

    @patch("crew.discovery.discover_employees")
    def test_restore_not_found(self, mock_discover):
        """不存在的员工应返回 404."""
        from crew.models import DiscoveryResult

        mock_discover.return_value = DiscoveryResult(employees={})

        client = _make_client()
        resp = client.post(
            "/api/employees/nonexistent/authority/restore",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 404

    def test_restore_requires_auth(self):
        """权限恢复需要认证."""
        client = _make_client()
        resp = client.post("/api/employees/test-emp/authority/restore")
        assert resp.status_code == 401


class TestEmployeeStateEndpoint:
    """GET /api/employees/{slug}/state — 员工运行时状态."""

    @patch("crew.discovery.discover_employees")
    @patch("crew.memory.MemoryStore")
    def test_state_includes_agent_status(self, mock_store_cls, mock_discover, tmp_path):
        """/api/employees/{slug}/state 返回中包含 agent_status 字段."""
        from crew.models import DiscoveryResult, Employee

        emp_dir = tmp_path / "test-emp-3090"
        emp_dir.mkdir(parents=True, exist_ok=True)

        emp = Employee(
            name="test-emp",
            display_name="Test",
            character_name="测试",
            description="测试员工",
            body="test body",
            agent_id=3090,
            agent_status="frozen",
            source_path=emp_dir,
        )
        mock_discover.return_value = DiscoveryResult(employees={"test-emp": emp})

        mock_store = MagicMock()
        mock_store.query.return_value = []
        mock_store_cls.return_value = mock_store

        client = _make_client()
        resp = client.get(
            "/api/employees/test-emp/state",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "agent_status" in data
        assert data["agent_status"] == "frozen"

    @patch("crew.discovery.discover_employees")
    @patch("crew.memory.MemoryStore")
    def test_state_default_agent_status_active(self, mock_store_cls, mock_discover, tmp_path):
        """默认 agent_status 为 'active'."""
        from crew.models import DiscoveryResult, Employee

        emp_dir = tmp_path / "default-emp"
        emp_dir.mkdir(parents=True, exist_ok=True)

        # 不显式设置 agent_status，应走默认值 "active"
        emp = Employee(
            name="default-emp",
            display_name="Default",
            character_name="默认",
            description="默认状态测试",
            body="test body",
            source_path=emp_dir,
        )
        mock_discover.return_value = DiscoveryResult(employees={"default-emp": emp})

        mock_store = MagicMock()
        mock_store.query.return_value = []
        mock_store_cls.return_value = mock_store

        client = _make_client()
        resp = client.get(
            "/api/employees/default-emp/state",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "agent_status" in data
        assert data["agent_status"] == "active"

    @patch("crew.discovery.discover_employees")
    def test_state_not_found(self, mock_discover):
        """不存在的员工应返回 404."""
        from crew.models import DiscoveryResult

        mock_discover.return_value = DiscoveryResult(employees={})

        client = _make_client()
        resp = client.get(
            "/api/employees/nonexistent/state",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 404

    def test_state_requires_auth(self):
        """state 端点需要认证."""
        client = _make_client()
        resp = client.get("/api/employees/test-emp/state")
        assert resp.status_code == 401
