"""测试独立轨迹存储系统 — /api/trajectory/report."""

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest


class TestTrajectoryArchive:
    """测试轨迹归档存储功能."""

    @pytest.mark.asyncio
    async def test_trajectory_report_basic(self, tmp_path, monkeypatch):
        """基本轨迹上报功能."""

        from crew.webhook_handlers import _handle_trajectory_report

        # Mock 请求
        class MockRequest:
            def __init__(self, payload):
                self._payload = payload
                self.query_params = {}
                self.headers = {}
                self.state = SimpleNamespace(tenant=SimpleNamespace(tenant_id="admin", is_admin=True))

            async def body(self):
                return json.dumps(self._payload).encode("utf-8")

            async def json(self):
                return self._payload

        # Mock 上下文
        class MockContext:
            def __init__(self):
                self.project_dir = tmp_path

        payload = {
            "employee_name": "赵云帆",
            "task_description": "实现轨迹存储",
            "model": "claude-sonnet-4-6",
            "channel": "pull",
            "success": True,
            "steps": [
                {
                    "step_id": 1,
                    "thought": "先查看现有代码",
                    "tool_name": "Read",
                    "tool_params": {"file_path": "/root/test.py"},
                    "tool_output": "文件内容...",
                    "tool_exit_code": 0,
                    "timestamp": "2026-03-02T17:00:00Z",
                },
                {
                    "step_id": 2,
                    "thought": "修改代码",
                    "tool_name": "Edit",
                    "tool_params": {
                        "file_path": "/root/test.py",
                        "old_string": "old",
                        "new_string": "new",
                    },
                    "tool_output": "修改成功",
                    "tool_exit_code": 0,
                    "timestamp": "2026-03-02T17:01:00Z",
                },
            ],
        }

        # Mock _tenant_base_dir 返回 tmp_path（使 trajectory_archive 写到 tmp_path 下）
        monkeypatch.setattr(
            "crew.webhook_handlers._tenant_base_dir",
            lambda request: tmp_path,
        )

        request = MockRequest(payload)
        ctx = MockContext()

        # 调用 API
        response = await _handle_trajectory_report(request, ctx)

        # 验证响应
        assert response.status_code == 200
        resp_data = json.loads(response.body.decode("utf-8"))
        assert resp_data["ok"] is True
        assert resp_data["total_steps"] == 2
        assert "trajectory_id" in resp_data
        assert "file_path" in resp_data

    @pytest.mark.asyncio
    async def test_trajectory_file_structure(self, tmp_path, monkeypatch):
        """验证轨迹文件存储结构."""
        from crew.webhook_handlers import _handle_trajectory_report

        class MockRequest:
            def __init__(self, payload):
                self._payload = payload
                self.query_params = {}
                self.headers = {}
                self.state = SimpleNamespace(tenant=SimpleNamespace(tenant_id="admin", is_admin=True))

            async def body(self):
                return json.dumps(self._payload).encode("utf-8")

            async def json(self):
                return self._payload

        class MockContext:
            def __init__(self):
                self.project_dir = tmp_path

        payload = {
            "employee_name": "backend-engineer",
            "task_description": "测试任务",
            "model": "claude-sonnet-4",
            "channel": "test",
            "success": True,
            "steps": [
                {
                    "step_id": 1,
                    "thought": "测试",
                    "tool_name": "Bash",
                    "tool_params": {"command": "echo test"},
                    "tool_output": "test",
                    "tool_exit_code": 0,
                },
            ],
        }

        # 使用真实路径
        date_str = date.today().isoformat()

        # Mock _tenant_base_dir 返回 tmp_path（使 trajectory_archive 写到 tmp_path 下）
        monkeypatch.setattr(
            "crew.webhook_handlers._tenant_base_dir",
            lambda request: tmp_path,
        )

        request = MockRequest(payload)
        ctx = MockContext()

        response = await _handle_trajectory_report(request, ctx)

        # 验证文件已创建
        assert response.status_code == 200
        resp_data = json.loads(response.body.decode("utf-8"))

        stored_path = Path(resp_data["file_path"])
        assert stored_path.exists()
        assert stored_path.parent.name == date_str
        assert stored_path.suffix == ".jsonl"

        # 验证文件内容（单个 JSON 对象包含完整轨迹）
        content = stored_path.read_text(encoding="utf-8").strip()
        traj_data = json.loads(content)
        assert "trajectory" in traj_data
        assert len(traj_data["trajectory"]) == 1
        step_data = traj_data["trajectory"][0]
        assert step_data["action"]["tool"] == "Bash"
        assert step_data["result"] == "test"

    @pytest.mark.asyncio
    async def test_trajectory_index_updated(self, tmp_path, monkeypatch):
        """验证元数据索引正确更新."""
        from crew.webhook_handlers import _handle_trajectory_report

        class MockRequest:
            def __init__(self, payload):
                self._payload = payload
                self.query_params = {}
                self.headers = {}
                self.state = SimpleNamespace(tenant=SimpleNamespace(tenant_id="admin", is_admin=True))

            async def body(self):
                return json.dumps(self._payload).encode("utf-8")

            async def json(self):
                return self._payload

        class MockContext:
            def __init__(self):
                self.project_dir = tmp_path

        payload = {
            "employee_name": "姜墨言",
            "task_description": "测试索引",
            "model": "claude-opus-4",
            "channel": "feishu",
            "success": True,
            "steps": [
                {
                    "step_id": 1,
                    "thought": "测试",
                    "tool_name": "Test",
                    "tool_params": {},
                    "tool_output": "ok",
                    "tool_exit_code": 0,
                },
            ],
        }

        # Mock _tenant_base_dir 返回 tmp_path（使 trajectory_archive 写到 tmp_path 下）
        monkeypatch.setattr(
            "crew.webhook_handlers._tenant_base_dir",
            lambda request: tmp_path,
        )

        request = MockRequest(payload)
        ctx = MockContext()

        response = await _handle_trajectory_report(request, ctx)

        # 验证索引文件
        archive_dir = tmp_path / "trajectory_archive"
        index_file = archive_dir / "index.json"
        assert index_file.exists()

        index_data = json.loads(index_file.read_text(encoding="utf-8"))
        assert len(index_data) > 0

        # 找到刚创建的轨迹
        resp_data = json.loads(response.body.decode("utf-8"))
        trajectory_id = resp_data["trajectory_id"]
        assert trajectory_id in index_data

        entry = index_data[trajectory_id]
        assert entry["employee"] == "姜墨言"
        assert entry["task"] == "测试索引"
        assert entry["model"] == "claude-opus-4"
        assert entry["channel"] == "feishu"
        assert entry["success"] is True
        assert entry["total_steps"] == 1

    @pytest.mark.asyncio
    async def test_trajectory_no_memory_pollution(self, tmp_path, monkeypatch):
        """验证轨迹不写入永久记忆."""
        from crew.memory import MemoryStore
        from crew.webhook_handlers import _handle_trajectory_report

        class MockRequest:
            def __init__(self, payload):
                self._payload = payload
                self.query_params = {}
                self.headers = {}
                self.state = SimpleNamespace(tenant=SimpleNamespace(tenant_id="admin", is_admin=True))

            async def body(self):
                return json.dumps(self._payload).encode("utf-8")

            async def json(self):
                return self._payload

        class MockContext:
            def __init__(self):
                self.project_dir = tmp_path

        # 初始化记忆存储
        store = MemoryStore(project_dir=tmp_path)

        payload = {
            "employee_name": "test-employee",
            "task_description": "测试记忆隔离",
            "model": "claude-sonnet-4",
            "channel": "test",
            "success": True,
            "steps": [
                {
                    "step_id": 1,
                    "thought": "测试",
                    "tool_name": "Test",
                    "tool_params": {},
                    "tool_output": "ok",
                    "tool_exit_code": 0,
                },
            ],
        }

        # Mock _tenant_base_dir 返回 tmp_path（使 trajectory_archive 写到 tmp_path 下）
        monkeypatch.setattr(
            "crew.webhook_handlers._tenant_base_dir",
            lambda request: tmp_path,
        )

        request = MockRequest(payload)
        ctx = MockContext()

        # 记录上报前的记忆数量
        before_count = len(store.query("test-employee", limit=100))

        response = await _handle_trajectory_report(request, ctx)
        assert response.status_code == 200

        # 验证记忆数量没有增加
        after_count = len(store.query("test-employee", limit=100))
        assert after_count == before_count, "轨迹不应写入永久记忆"

    @pytest.mark.asyncio
    async def test_trajectory_missing_fields(self, tmp_path):
        """验证缺少必填字段时返回错误."""
        from crew.webhook_handlers import _handle_trajectory_report

        class MockRequest:
            def __init__(self, payload):
                self._payload = payload
                self.query_params = {}
                self.headers = {}
                self.state = SimpleNamespace(tenant=SimpleNamespace(tenant_id="admin", is_admin=True))

            async def body(self):
                return json.dumps(self._payload).encode("utf-8")

            async def json(self):
                return self._payload

        class MockContext:
            def __init__(self):
                self.project_dir = tmp_path

        # 缺少 steps
        payload = {
            "employee_name": "test",
            "task_description": "测试",
        }

        request = MockRequest(payload)
        ctx = MockContext()

        response = await _handle_trajectory_report(request, ctx)
        assert response.status_code == 400
        resp_data = json.loads(response.body.decode("utf-8"))
        assert "error" in resp_data

    @pytest.mark.asyncio
    async def test_trajectory_large_payload(self, tmp_path):
        """验证超大 payload 被拒绝."""
        from crew.webhook_handlers import _handle_trajectory_report

        class MockRequest:
            def __init__(self, size):
                self._size = size
                self.query_params = {}
                self.headers = {}
                self.state = SimpleNamespace(tenant=SimpleNamespace(tenant_id="admin", is_admin=True))

            async def body(self):
                return b"x" * self._size

            async def json(self):
                raise ValueError("Should not reach here")

        class MockContext:
            def __init__(self):
                self.project_dir = tmp_path

        # 3MB payload
        request = MockRequest(3 * 1024 * 1024)
        ctx = MockContext()

        response = await _handle_trajectory_report(request, ctx)
        assert response.status_code == 413
        resp_data = json.loads(response.body.decode("utf-8"))
        assert "too large" in resp_data["error"]
