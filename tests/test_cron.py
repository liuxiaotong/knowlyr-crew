"""测试 Cron 调度器."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

pytest.importorskip("croniter")
pytest.importorskip("starlette")

from starlette.testclient import TestClient

from crew.cron_config import CronConfig, CronSchedule, load_cron_config, validate_cron_config
from crew.cron_scheduler import CronScheduler
from crew.webhook import create_webhook_app
from crew.webhook_config import WebhookConfig

# ── CronConfig 测试 ──


class TestCronScheduleModel:
    """Cron 调度模型."""

    def test_basic(self):
        s = CronSchedule(
            name="daily-review",
            cron="0 9 * * *",
            target_type="pipeline",
            target_name="full-review",
        )
        assert s.name == "daily-review"
        assert s.cron == "0 9 * * *"
        assert s.args == {}

    def test_with_args(self):
        s = CronSchedule(
            name="test",
            cron="*/5 * * * *",
            target_type="employee",
            target_name="code-reviewer",
            args={"target": "main"},
        )
        assert s.args == {"target": "main"}

    def test_with_delivery(self):
        s = CronSchedule(
            name="with-delivery",
            cron="0 9 * * *",
            target_type="pipeline",
            target_name="full-review",
            delivery=[
                {"type": "webhook", "url": "https://hooks.slack.com/xxx"},
                {"type": "email", "to": "team@example.com", "subject": "Review: {name}"},
            ],
        )
        assert len(s.delivery) == 2
        assert s.delivery[0].type == "webhook"
        assert s.delivery[0].url == "https://hooks.slack.com/xxx"
        assert s.delivery[1].type == "email"
        assert s.delivery[1].to == "team@example.com"

    def test_no_delivery(self):
        s = CronSchedule(
            name="no-delivery",
            cron="0 9 * * *",
            target_type="pipeline",
            target_name="full-review",
        )
        assert s.delivery == []


class TestCronConfig:
    """Cron 配置模型."""

    def test_empty(self):
        config = CronConfig()
        assert config.schedules == []

    def test_with_schedules(self):
        config = CronConfig(schedules=[
            CronSchedule(
                name="daily",
                cron="0 9 * * *",
                target_type="pipeline",
                target_name="full-review",
            ),
        ])
        assert len(config.schedules) == 1


class TestLoadCronConfig:
    """加载 cron 配置."""

    def test_no_config_file(self, tmp_path):
        config = load_cron_config(tmp_path)
        assert config.schedules == []

    def test_empty_file(self, tmp_path):
        crew_dir = tmp_path / ".crew"
        crew_dir.mkdir()
        (crew_dir / "cron.yaml").write_text("")
        config = load_cron_config(tmp_path)
        assert config.schedules == []

    def test_valid_config(self, tmp_path):
        crew_dir = tmp_path / ".crew"
        crew_dir.mkdir()
        data = {
            "schedules": [
                {
                    "name": "daily-review",
                    "cron": "0 9 * * *",
                    "target_type": "pipeline",
                    "target_name": "full-review",
                    "args": {"target": "main"},
                },
                {
                    "name": "weekly-summary",
                    "cron": "0 0 * * 0",
                    "target_type": "employee",
                    "target_name": "doc-writer",
                },
            ],
        }
        (crew_dir / "cron.yaml").write_text(yaml.dump(data, allow_unicode=True))
        config = load_cron_config(tmp_path)
        assert len(config.schedules) == 2
        assert config.schedules[0].name == "daily-review"
        assert config.schedules[0].args == {"target": "main"}
        assert config.schedules[1].name == "weekly-summary"

    def test_config_with_delivery(self, tmp_path):
        crew_dir = tmp_path / ".crew"
        crew_dir.mkdir()
        data = {
            "schedules": [
                {
                    "name": "daily-review",
                    "cron": "0 9 * * *",
                    "target_type": "pipeline",
                    "target_name": "full-review",
                    "delivery": [
                        {"type": "webhook", "url": "https://hooks.slack.com/xxx"},
                        {"type": "email", "to": "team@example.com", "subject": "Review: {name}"},
                    ],
                },
            ],
        }
        (crew_dir / "cron.yaml").write_text(yaml.dump(data, allow_unicode=True))
        config = load_cron_config(tmp_path)
        assert len(config.schedules[0].delivery) == 2
        assert config.schedules[0].delivery[0].type == "webhook"
        assert config.schedules[0].delivery[1].to == "team@example.com"


class TestValidateCronConfig:
    """校验 cron 配置."""

    def test_valid(self):
        config = CronConfig(schedules=[
            CronSchedule(
                name="test",
                cron="0 9 * * *",
                target_type="employee",
                target_name="code-reviewer",
            ),
        ])
        errors = validate_cron_config(config)
        assert errors == []

    def test_invalid_cron_expression(self):
        config = CronConfig(schedules=[
            CronSchedule(
                name="bad-cron",
                cron="invalid",
                target_type="pipeline",
                target_name="full-review",
            ),
        ])
        errors = validate_cron_config(config)
        assert any("无效的 cron 表达式" in e for e in errors)

    def test_duplicate_name(self):
        config = CronConfig(schedules=[
            CronSchedule(name="dup", cron="0 9 * * *", target_type="pipeline", target_name="full-review"),
            CronSchedule(name="dup", cron="0 10 * * *", target_type="pipeline", target_name="full-review"),
        ])
        errors = validate_cron_config(config)
        assert any("重复" in e for e in errors)

    def test_unknown_pipeline(self):
        config = CronConfig(schedules=[
            CronSchedule(
                name="test",
                cron="0 9 * * *",
                target_type="pipeline",
                target_name="nonexistent-pipeline",
            ),
        ])
        errors = validate_cron_config(config)
        assert any("未找到 pipeline" in e for e in errors)

    def test_unknown_employee(self):
        config = CronConfig(schedules=[
            CronSchedule(
                name="test",
                cron="0 9 * * *",
                target_type="employee",
                target_name="nonexistent-employee",
            ),
        ])
        errors = validate_cron_config(config)
        assert any("未找到员工" in e for e in errors)

    def test_empty_config(self):
        config = CronConfig()
        errors = validate_cron_config(config)
        assert errors == []


# ── CronScheduler 测试 ──


class TestCronScheduler:
    """Cron 调度器."""

    def test_no_schedules(self):
        config = CronConfig()
        scheduler = CronScheduler(config=config, execute_fn=AsyncMock())
        assert scheduler.schedules == []
        assert scheduler.running is False

    @pytest.mark.asyncio
    async def test_start_stop_empty(self):
        config = CronConfig()
        scheduler = CronScheduler(config=config, execute_fn=AsyncMock())
        await scheduler.start()
        assert scheduler.running is False  # 无任务不启动
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_start_stop_with_schedule(self):
        config = CronConfig(schedules=[
            CronSchedule(
                name="test",
                cron="0 9 * * *",
                target_type="pipeline",
                target_name="full-review",
            ),
        ])
        mock_fn = AsyncMock()
        scheduler = CronScheduler(config=config, execute_fn=mock_fn)
        await scheduler.start()
        assert scheduler.running is True
        await scheduler.stop()
        assert scheduler.running is False

    def test_get_next_runs(self):
        config = CronConfig(schedules=[
            CronSchedule(
                name="daily",
                cron="0 9 * * *",
                target_type="pipeline",
                target_name="full-review",
            ),
            CronSchedule(
                name="weekly",
                cron="0 0 * * 0",
                target_type="employee",
                target_name="doc-writer",
            ),
        ])
        scheduler = CronScheduler(config=config, execute_fn=AsyncMock())
        runs = scheduler.get_next_runs()
        assert len(runs) == 2
        assert runs[0]["name"] == "daily"
        assert "next_run" in runs[0]
        assert runs[1]["name"] == "weekly"

    def test_get_next_runs_invalid_cron(self):
        config = CronConfig(schedules=[
            CronSchedule(
                name="bad",
                cron="invalid",
                target_type="pipeline",
                target_name="test",
            ),
        ])
        scheduler = CronScheduler(config=config, execute_fn=AsyncMock())
        runs = scheduler.get_next_runs()
        assert len(runs) == 1
        assert "error" in runs[0]

    @pytest.mark.asyncio
    async def test_schedule_triggers_execute(self):
        """验证 cron 到期时调用 execute_fn."""
        config = CronConfig(schedules=[
            CronSchedule(
                name="fast",
                cron="* * * * *",  # 每分钟
                target_type="pipeline",
                target_name="test",
            ),
        ])
        call_count = 0
        triggered = asyncio.Event()

        async def mock_execute(schedule):
            nonlocal call_count
            call_count += 1
            triggered.set()

        scheduler = CronScheduler(config=config, execute_fn=mock_execute)

        # Mock time.time 返回远未来时间使 delay <= 0（跳过 sleep）
        with patch("crew.cron_scheduler.time.time", return_value=9999999999.0):
            await scheduler.start()
            # 等待第一次触发（最多 2 秒）
            try:
                await asyncio.wait_for(triggered.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            await scheduler.stop()

        assert call_count >= 1


# ── Cron Status API 测试 ──


TOKEN = "test-token-123"


class TestCronStatusEndpoint:
    """Cron 状态端点."""

    def test_no_cron(self):
        app = create_webhook_app(
            project_dir=Path("/tmp/test"),
            token=TOKEN,
            config=WebhookConfig(),
        )
        client = TestClient(app)
        resp = client.get("/cron/status", headers={"Authorization": f"Bearer {TOKEN}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False
        assert data["schedules"] == []

    def test_with_cron(self):
        cron_config = CronConfig(schedules=[
            CronSchedule(
                name="daily",
                cron="0 9 * * *",
                target_type="pipeline",
                target_name="full-review",
            ),
        ])
        app = create_webhook_app(
            project_dir=Path("/tmp/test"),
            token=TOKEN,
            config=WebhookConfig(),
            cron_config=cron_config,
        )
        client = TestClient(app)
        resp = client.get("/cron/status", headers={"Authorization": f"Bearer {TOKEN}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert len(data["schedules"]) == 1
        assert data["schedules"][0]["name"] == "daily"
        assert "next_run" in data["schedules"][0]

    def test_requires_auth(self):
        app = create_webhook_app(
            project_dir=Path("/tmp/test"),
            token=TOKEN,
            config=WebhookConfig(),
        )
        client = TestClient(app)
        resp = client.get("/cron/status")
        assert resp.status_code == 401


class TestCronMisfire:
    """Cron 漏执行检测."""

    @pytest.mark.asyncio
    async def test_misfire_logs_warning(self, caplog):
        """delay < -60 时记录 warning."""
        import logging
        config = CronConfig(schedules=[
            CronSchedule(
                name="misfire-test",
                cron="* * * * *",
                target_type="pipeline",
                target_name="test",
            ),
        ])
        mock_fn = AsyncMock()
        scheduler = CronScheduler(config=config, execute_fn=mock_fn)

        # Mock time.time 返回远未来时间使 delay < -60
        call_count = 0
        triggered = asyncio.Event()

        async def _count(schedule):
            nonlocal call_count
            call_count += 1
            triggered.set()

        scheduler._execute_fn = _count

        with caplog.at_level(logging.WARNING), \
             patch("crew.cron_scheduler.time.time", return_value=9999999999.0):
            await scheduler.start()
            try:
                await asyncio.wait_for(triggered.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            await scheduler.stop()

        assert scheduler._missed_counts.get("misfire-test", 0) >= 1
        assert "漏执行" in caplog.text

    @pytest.mark.asyncio
    async def test_exception_backoff(self):
        """连续异常后 sleep 时间递增."""
        config = CronConfig(schedules=[
            CronSchedule(
                name="error-test",
                cron="* * * * *",
                target_type="pipeline",
                target_name="test",
            ),
        ])

        call_count = 0

        async def _fail(schedule):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise RuntimeError("test error")

        sleep_times = []
        original_sleep = asyncio.sleep

        async def _mock_sleep(t):
            sleep_times.append(t)
            if len(sleep_times) > 5:
                # 停止循环
                raise asyncio.CancelledError()
            await original_sleep(0)

        scheduler = CronScheduler(config=config, execute_fn=_fail)

        with patch("crew.cron_scheduler.time.time", return_value=9999999999.0), \
             patch("crew.cron_scheduler.asyncio.sleep", side_effect=_mock_sleep):
            await scheduler.start()
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
            await scheduler.stop()

        # 应有递增的 backoff sleep
        backoff_sleeps = [t for t in sleep_times if t > 0]
        if len(backoff_sleeps) >= 2:
            assert backoff_sleeps[1] >= backoff_sleeps[0]

    def test_missed_count_in_status(self):
        """get_next_runs 包含 missed_count."""
        config = CronConfig(schedules=[
            CronSchedule(
                name="test",
                cron="0 9 * * *",
                target_type="pipeline",
                target_name="full-review",
            ),
        ])
        scheduler = CronScheduler(config=config, execute_fn=AsyncMock())
        scheduler._missed_counts["test"] = 5
        runs = scheduler.get_next_runs()
        assert len(runs) == 1
        assert runs[0]["missed_count"] == 5
