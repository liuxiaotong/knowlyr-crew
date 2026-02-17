"""Cron 配置测试 — cron_config.py."""

import json

import pytest

from crew.cron_config import (
    CronConfig,
    CronSchedule,
    DeliveryTarget,
    load_cron_config,
)


# ── 模型测试 ──


class TestDeliveryTarget:
    def test_webhook_type(self):
        dt = DeliveryTarget(type="webhook", url="https://example.com/hook")
        assert dt.type == "webhook"
        assert dt.url == "https://example.com/hook"
        assert dt.headers == {}

    def test_feishu_type(self):
        dt = DeliveryTarget(
            type="feishu",
            url="https://open.feishu.cn/hook/xxx",
            secret="mysecret",
        )
        assert dt.secret == "mysecret"

    def test_email_type(self):
        dt = DeliveryTarget(type="email", to="user@example.com", subject="报告 {name}")
        assert dt.to == "user@example.com"
        assert "{name}" in dt.subject


class TestCronSchedule:
    def test_minimal(self):
        cs = CronSchedule(
            name="daily-review",
            cron="0 9 * * *",
            target_type="employee",
            target_name="code-reviewer",
        )
        assert cs.args == {}
        assert cs.delivery == []

    def test_with_delivery(self):
        cs = CronSchedule(
            name="weekly",
            cron="0 9 * * 1",
            target_type="pipeline",
            target_name="weekly-report",
            delivery=[DeliveryTarget(type="webhook", url="https://hook.example.com")],
        )
        assert len(cs.delivery) == 1


class TestCronConfig:
    def test_empty(self):
        cc = CronConfig()
        assert cc.schedules == []

    def test_with_schedules(self):
        cc = CronConfig(
            schedules=[
                CronSchedule(
                    name="t1",
                    cron="0 9 * * *",
                    target_type="employee",
                    target_name="emp",
                ),
            ]
        )
        assert len(cc.schedules) == 1


# ── load_cron_config ──


class TestLoadCronConfig:
    def test_no_file_returns_empty(self, tmp_path):
        config = load_cron_config(project_dir=tmp_path)
        assert config.schedules == []

    def test_empty_yaml_returns_empty(self, tmp_path):
        crew_dir = tmp_path / ".crew"
        crew_dir.mkdir()
        (crew_dir / "cron.yaml").write_text("", encoding="utf-8")
        config = load_cron_config(project_dir=tmp_path)
        assert config.schedules == []

    def test_valid_yaml(self, tmp_path):
        crew_dir = tmp_path / ".crew"
        crew_dir.mkdir()
        (crew_dir / "cron.yaml").write_text(
            """\
schedules:
  - name: daily-check
    cron: "0 9 * * *"
    target_type: employee
    target_name: code-reviewer
    args:
      scope: "src/"
""",
            encoding="utf-8",
        )
        config = load_cron_config(project_dir=tmp_path)
        assert len(config.schedules) == 1
        assert config.schedules[0].name == "daily-check"
        assert config.schedules[0].args["scope"] == "src/"

    def test_multiple_schedules(self, tmp_path):
        crew_dir = tmp_path / ".crew"
        crew_dir.mkdir()
        (crew_dir / "cron.yaml").write_text(
            """\
schedules:
  - name: task-a
    cron: "0 9 * * *"
    target_type: employee
    target_name: emp-a
  - name: task-b
    cron: "0 18 * * 5"
    target_type: pipeline
    target_name: pipe-b
""",
            encoding="utf-8",
        )
        config = load_cron_config(project_dir=tmp_path)
        assert len(config.schedules) == 2
        assert config.schedules[1].target_type == "pipeline"
