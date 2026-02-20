"""测试 crew ↔ knowlyr-id 双向同步."""

from pathlib import Path
from unittest.mock import patch

import yaml

from crew.sync import sync_all


def _make_employee_dir(
    base: Path, name: str, *, agent_id: int | None = None, bio: str = ""
) -> Path:
    """在 base 下创建一个最小化的员工目录."""
    dir_name = f"{name}-{agent_id}" if agent_id else name
    emp_dir = base / dir_name
    emp_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "name": name,
        "display_name": f"{name}显示名",
        "character_name": f"{name}姓名",
        "description": f"{name}描述",
        "version": "1.0",
        "tags": ["testing"],
        "_content_hash": "aabbccdd",
    }
    if agent_id is not None:
        config["agent_id"] = agent_id
    if bio:
        config["bio"] = bio

    (emp_dir / "employee.yaml").write_text(
        yaml.dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (emp_dir / "prompt.md").write_text(
        f"# {name}\n\n你是 {name}，请完成 $goal。\n",
        encoding="utf-8",
    )
    return emp_dir


class TestSyncPush:
    """推送本地数据到 knowlyr-id."""

    def test_push_metadata(self, tmp_path):
        """验证 push 正确映射字段."""
        _make_employee_dir(tmp_path, "test-worker", agent_id=3050, bio="测试简介")

        update_calls = []

        def mock_update(agent_id, **kwargs):
            update_calls.append((agent_id, kwargs))
            return True

        def mock_list():
            return [{"id": 3050, "nickname": "old", "status": "active"}]

        def mock_fetch(aid):
            from crew.id_client import AgentIdentity

            return AgentIdentity(agent_id=aid, nickname="old")

        with (
            patch("crew.sync.list_agents", mock_list),
            patch("crew.sync.update_agent", mock_update),
            patch("crew.sync.fetch_agent_identity", mock_fetch),
        ):
            report = sync_all(tmp_path, push=True, pull=True, force=True)

        assert len(report.pushed) == 1
        assert "test-worker" in report.pushed[0]

        # 验证 update_agent 调用参数
        assert len(update_calls) == 1
        aid, kwargs = update_calls[0]
        assert aid == 3050
        assert kwargs["nickname"] == "test-worker姓名"
        assert kwargs["title"] == "test-worker显示名"
        assert kwargs["capabilities"] == "test-worker描述"
        assert kwargs["domains"] == ["testing"]
        assert kwargs["system_prompt"] is not None
        assert len(kwargs["system_prompt"]) > 0

    def test_push_with_avatar(self, tmp_path):
        """有 avatar.webp 时推送头像."""
        emp_dir = _make_employee_dir(tmp_path, "avatar-worker", agent_id=3051)
        (emp_dir / "avatar.webp").write_bytes(b"fake-webp-data")

        update_calls = []

        def mock_update(agent_id, **kwargs):
            update_calls.append((agent_id, kwargs))
            return True

        with (
            patch("crew.sync.list_agents", return_value=[{"id": 3051, "status": "active"}]),
            patch("crew.sync.update_agent", mock_update),
            patch("crew.sync.fetch_agent_identity", return_value=None),
        ):
            report = sync_all(tmp_path, push=True, pull=True, force=True)

        assert len(update_calls) == 1
        assert update_calls[0][1]["avatar_base64"] is not None

    def test_push_includes_model(self, tmp_path):
        """push 时应包含 model 字段（crew 是 model 唯一真相源）."""
        emp_dir = _make_employee_dir(tmp_path, "model-worker", agent_id=3052)
        # 写入 model 到 employee.yaml
        config_path = emp_dir / "employee.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["model"] = "claude-opus-4-6"
        config_path.write_text(
            yaml.dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

        update_calls = []

        def mock_update(agent_id, **kwargs):
            update_calls.append((agent_id, kwargs))
            return True

        def mock_fetch(aid):
            from crew.id_client import AgentIdentity

            return AgentIdentity(agent_id=aid, nickname="m")

        with (
            patch("crew.sync.list_agents", return_value=[{"id": 3052, "status": "active"}]),
            patch("crew.sync.update_agent", mock_update),
            patch("crew.sync.fetch_agent_identity", mock_fetch),
        ):
            report = sync_all(tmp_path, push=True, pull=True, force=True)

        assert len(update_calls) == 1
        assert update_calls[0][1]["model"] == "claude-opus-4-6"


class TestSyncPull:
    """从 knowlyr-id 拉取运行时数据."""

    def test_pull_memory(self, tmp_path):
        """验证 memory 写入 memory-id.md."""
        _make_employee_dir(tmp_path, "mem-worker", agent_id=3060)

        def mock_fetch(aid):
            from crew.id_client import AgentIdentity

            return AgentIdentity(
                agent_id=aid,
                nickname="mem",
                memory="用户偏好暗色主题\n\n上次讨论了 API 设计",
            )

        with (
            patch("crew.sync.list_agents", return_value=[{"id": 3060, "status": "active"}]),
            patch("crew.sync.update_agent", return_value=True),
            patch("crew.sync.fetch_agent_identity", mock_fetch),
        ):
            report = sync_all(tmp_path, push=True, pull=True, force=True)

        assert len(report.pulled) == 1

        # 检查文件
        mem_path = tmp_path / "mem-worker-3060" / "memory-id.md"
        assert mem_path.exists()
        content = mem_path.read_text(encoding="utf-8")
        assert "暗色主题" in content

    def test_pull_temperature_not_model(self, tmp_path):
        """验证 temperature 回写但 model 不从 id 拉取（crew 是 model 唯一真相源）."""
        emp_dir = _make_employee_dir(tmp_path, "model-worker", agent_id=3061)

        def mock_fetch(aid):
            from crew.id_client import AgentIdentity

            return AgentIdentity(
                agent_id=aid,
                nickname="model",
                model="gpt-4o",
                temperature=0.7,
            )

        with (
            patch("crew.sync.list_agents", return_value=[{"id": 3061, "status": "active"}]),
            patch("crew.sync.update_agent", return_value=True),
            patch("crew.sync.fetch_agent_identity", mock_fetch),
        ):
            report = sync_all(tmp_path, push=True, pull=True, force=True)

        assert len(report.pulled) == 1

        # 检查 yaml — temperature 回写，model 不变
        config = yaml.safe_load((emp_dir / "employee.yaml").read_text(encoding="utf-8"))
        assert "model" not in config or config.get("model") != "gpt-4o"
        assert config["temperature"] == 0.7

    def test_pull_no_change(self, tmp_path):
        """memory 和 model 无变化时不报告 pull."""
        emp_dir = _make_employee_dir(tmp_path, "same-worker", agent_id=3062)

        def mock_fetch(aid):
            from crew.id_client import AgentIdentity

            return AgentIdentity(agent_id=aid, nickname="same")

        with (
            patch("crew.sync.list_agents", return_value=[{"id": 3062, "status": "active"}]),
            patch("crew.sync.update_agent", return_value=True),
            patch("crew.sync.fetch_agent_identity", mock_fetch),
        ):
            report = sync_all(tmp_path, push=True, pull=True, force=True)

        assert len(report.pulled) == 0


class TestSyncPullAgentStatus:
    """pull 时同步 agent_status 字段."""

    def test_pull_frozen_status_writes_yaml(self, tmp_path):
        """_pull_employee 从 id 拉到 frozen 状态时，写入 employee.yaml."""
        emp_dir = _make_employee_dir(tmp_path, "status-worker", agent_id=3070)

        def mock_fetch(aid):
            from crew.id_client import AgentIdentity

            return AgentIdentity(
                agent_id=aid,
                nickname="status",
                agent_status="frozen",
            )

        with (
            patch("crew.sync.list_agents", return_value=[{"id": 3070, "status": "active"}]),
            patch("crew.sync.update_agent", return_value=True),
            patch("crew.sync.fetch_agent_identity", mock_fetch),
        ):
            report = sync_all(tmp_path, push=True, pull=True, force=True)

        # 应有 pull 记录
        assert len(report.pulled) == 1

        # 检查 yaml — agent_status 应被写入
        config = yaml.safe_load((emp_dir / "employee.yaml").read_text(encoding="utf-8"))
        assert config.get("agent_status") == "frozen"

    def test_pull_same_status_no_write(self, tmp_path):
        """状态没变时不触发写入（mock _write_yaml_field 检查未被调用）."""
        emp_dir = _make_employee_dir(tmp_path, "same-status", agent_id=3071)

        # 先在 yaml 中写入 agent_status = active（与远程一致）
        config_path = emp_dir / "employee.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["agent_status"] = "active"
        config_path.write_text(
            yaml.dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

        def mock_fetch(aid):
            from crew.id_client import AgentIdentity

            return AgentIdentity(
                agent_id=aid,
                nickname="same",
                agent_status="active",
            )

        with (
            patch("crew.sync.list_agents", return_value=[{"id": 3071, "status": "active"}]),
            patch("crew.sync.update_agent", return_value=True),
            patch("crew.sync.fetch_agent_identity", mock_fetch),
            patch("crew.sync._write_yaml_field") as mock_write,
        ):
            report = sync_all(tmp_path, push=True, pull=True, force=True)

        # 状态未变，不应有 pull（memory 也无变化）
        assert len(report.pulled) == 0
        # _write_yaml_field 不应被调用（由 _pull_employee 触发的写入）
        mock_write.assert_not_called()

    def test_pull_id_unreachable_no_change(self, tmp_path):
        """id 不可达时不影响现有字段."""
        emp_dir = _make_employee_dir(tmp_path, "unreachable-worker", agent_id=3072)

        # 先设置一个已有的 agent_status
        config_path = emp_dir / "employee.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["agent_status"] = "active"
        config_path.write_text(
            yaml.dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

        # fetch_agent_identity 返回 None（不可达）
        with (
            patch("crew.sync.list_agents", return_value=[{"id": 3072, "status": "active"}]),
            patch("crew.sync.update_agent", return_value=True),
            patch("crew.sync.fetch_agent_identity", return_value=None),
        ):
            report = sync_all(tmp_path, push=True, pull=True, force=True)

        # 应有 error 记录（无法获取身份）
        assert len(report.errors) == 1
        assert "无法获取" in report.errors[0][1]

        # 本地 yaml 不应被修改
        config_after = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config_after.get("agent_status") == "active"


class TestSyncRegister:
    """新员工注册."""

    def test_register_new(self, tmp_path):
        """无 agent_id 时自动注册."""
        _make_employee_dir(tmp_path, "new-worker")

        register_calls = []

        def mock_register(**kwargs):
            register_calls.append(kwargs)
            return 9999

        with (
            patch("crew.sync.list_agents", return_value=[]),
            patch("crew.sync.register_agent", mock_register),
        ):
            report = sync_all(tmp_path, push=True, pull=False)

        assert len(report.registered) == 1
        assert "new-worker" in report.registered[0]
        assert "#9999" in report.registered[0]

        # 检查 agent_id 回写
        config = yaml.safe_load(
            (tmp_path / "new-worker" / "employee.yaml").read_text(encoding="utf-8")
        )
        assert config["agent_id"] == 9999

    def test_register_failure(self, tmp_path):
        """注册失败记录错误."""
        _make_employee_dir(tmp_path, "fail-worker")

        with (
            patch("crew.sync.list_agents", return_value=[]),
            patch("crew.sync.register_agent", return_value=None),
        ):
            report = sync_all(tmp_path, push=True, pull=False)

        assert len(report.errors) == 1
        assert "注册失败" in report.errors[0][1]


class TestSyncDisable:
    """禁用已删除员工."""

    def test_disable_removed(self, tmp_path):
        """本地不存在的 agent 设 inactive."""
        # 本地无员工，但远程有 agent #3070
        update_calls = []

        def mock_update(agent_id, **kwargs):
            update_calls.append((agent_id, kwargs))
            return True

        with (
            patch(
                "crew.sync.list_agents",
                return_value=[
                    {"id": 3070, "nickname": "ghost", "status": "active"},
                ],
            ),
            patch("crew.sync.update_agent", mock_update),
        ):
            report = sync_all(tmp_path, push=True, pull=False)

        assert len(report.disabled) == 1
        assert "#3070" in report.disabled[0]
        assert len(update_calls) == 1
        assert update_calls[0][1]["agent_status"] == "inactive"

    def test_skip_already_inactive(self, tmp_path):
        """已经 inactive 的不重复禁用."""
        with (
            patch(
                "crew.sync.list_agents",
                return_value=[
                    {"id": 3070, "nickname": "ghost", "status": "inactive"},
                ],
            ),
            patch("crew.sync.update_agent", return_value=True) as mock_upd,
        ):
            report = sync_all(tmp_path, push=True, pull=False)

        assert len(report.disabled) == 0


class TestSyncDryRun:
    """dry-run 模式."""

    def test_dry_run_no_writes(self, tmp_path):
        """dry-run 不执行任何写操作."""
        _make_employee_dir(tmp_path, "dry-worker", agent_id=3080)

        update_calls = []

        def mock_fetch(aid):
            from crew.id_client import AgentIdentity

            return AgentIdentity(
                agent_id=aid,
                nickname="dry",
                memory="some memory",
                model="gpt-4o",
            )

        with (
            patch("crew.sync.list_agents", return_value=[{"id": 3080, "status": "active"}]),
            patch(
                "crew.sync.update_agent", side_effect=lambda **kw: update_calls.append(kw) or True
            ),
            patch("crew.sync.fetch_agent_identity", mock_fetch),
        ):
            report = sync_all(tmp_path, dry_run=True, push=True, pull=True, force=True)

        # dry-run 标记
        assert any("dry-run" in p for p in report.pushed)
        assert any("dry-run" in p for p in report.pulled)

        # 不应调用 update_agent
        assert len(update_calls) == 0

        # 不应写入文件
        mem_path = tmp_path / "dry-worker-3080" / "memory-id.md"
        assert not mem_path.exists()


class TestSyncReport:
    """SyncReport 统计."""

    def test_empty_dir(self, tmp_path):
        """空目录同步."""
        with patch("crew.sync.list_agents", return_value=[]):
            report = sync_all(tmp_path)

        assert report.pushed == []
        assert report.pulled == []
        assert report.registered == []
        assert report.disabled == []
        assert report.errors == []

    def test_nonexistent_dir(self, tmp_path):
        """目录不存在."""
        report = sync_all(tmp_path / "nonexistent")
        assert len(report.errors) == 1
        assert "不存在" in report.errors[0][1]

    def test_list_agents_failure(self, tmp_path):
        """list_agents 返回 None."""
        _make_employee_dir(tmp_path, "some-worker", agent_id=3090)

        with patch("crew.sync.list_agents", return_value=None):
            report = sync_all(tmp_path)

        assert len(report.errors) == 1
        assert "Agent 列表" in report.errors[0][1]

    def test_push_only(self, tmp_path):
        """push-only 模式不拉取."""
        _make_employee_dir(tmp_path, "push-worker", agent_id=3091)

        fetch_calls = []

        def mock_fetch(aid):
            fetch_calls.append(aid)
            from crew.id_client import AgentIdentity

            return AgentIdentity(agent_id=aid, nickname="x", memory="data")

        with (
            patch("crew.sync.list_agents", return_value=[{"id": 3091, "status": "active"}]),
            patch("crew.sync.update_agent", return_value=True),
            patch("crew.sync.fetch_agent_identity", mock_fetch),
        ):
            report = sync_all(tmp_path, push=True, pull=False, force=True)

        assert len(report.pushed) == 1
        assert len(report.pulled) == 0
        assert len(fetch_calls) == 0

    def test_pull_only(self, tmp_path):
        """pull-only 模式不推送."""
        _make_employee_dir(tmp_path, "pull-worker", agent_id=3092)

        update_calls = []

        def mock_fetch(aid):
            from crew.id_client import AgentIdentity

            return AgentIdentity(agent_id=aid, nickname="x", memory="data", model="gpt-4o")

        with (
            patch("crew.sync.list_agents", return_value=[{"id": 3092, "status": "active"}]),
            patch(
                "crew.sync.update_agent", side_effect=lambda **kw: update_calls.append(kw) or True
            ),
            patch("crew.sync.fetch_agent_identity", mock_fetch),
        ):
            report = sync_all(tmp_path, push=False, pull=True)

        assert len(report.pushed) == 0
        assert len(report.pulled) == 1
        assert len(update_calls) == 0


class TestSyncForcePromptChanged:
    """force=True 时 prompt_changed 正确初始化."""

    def test_force_push_renders_prompt(self, tmp_path):
        """force=True 时始终渲染 prompt（验证 prompt_changed 初始化）."""
        _make_employee_dir(tmp_path, "force-worker", agent_id=3095)

        update_calls = []

        def mock_update(agent_id, **kwargs):
            update_calls.append((agent_id, kwargs))
            return True

        with (
            patch("crew.sync.list_agents", return_value=[{"id": 3095, "status": "active"}]),
            patch("crew.sync.update_agent", mock_update),
            patch("crew.sync.fetch_agent_identity", return_value=None),
        ):
            report = sync_all(tmp_path, push=True, pull=False, force=True)

        assert len(report.pushed) == 1
        assert len(update_calls) == 1
        # force=True 应该渲染 prompt（system_prompt 非 None）
        assert update_calls[0][1]["system_prompt"] is not None
