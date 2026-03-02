"""集成测试：记忆去重和更新 API."""

import json
from pathlib import Path

import pytest

starlette = pytest.importorskip("starlette")

from starlette.testclient import TestClient

from crew.webhook import create_webhook_app
from crew.webhook_config import WebhookConfig

TOKEN = "test-dedup-token"


def _make_client(project_dir: Path):
    """创建测试客户端."""
    app = create_webhook_app(
        project_dir=project_dir,
        token=TOKEN,
        config=WebhookConfig(),
    )
    return TestClient(app)


def test_memory_deduplication_workflow(tmp_path):
    """测试完整的记忆去重工作流程."""
    client = _make_client(tmp_path)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    # 1. 添加第一条记忆
    resp1 = client.post(
        "/api/memory/add",
        json={
            "employee": "测试员工",
            "category": "finding",
            "content": "修复了数据库连接池的内存泄漏问题，原因是连接未正确关闭，需要在 finally 块中显式调用 close()",
            "tags": ["database", "memory-leak"],
        },
        headers=headers,
    )

    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["ok"] is True
    assert data1["skipped"] is False
    entry_id_1 = data1["entry_id"]

    # 2. 尝试添加相似的记忆（应该被拦截）
    resp2 = client.post(
        "/api/memory/add",
        json={
            "employee": "测试员工",
            "category": "finding",
            "content": "修复了数据库连接池的内存泄漏问题，原因是连接未正确关闭，需要在 finally 块中显式调用 close() 方法释放资源",
            "tags": ["database"],
        },
        headers=headers,
    )

    assert resp2.status_code == 200
    data2 = resp2.json()

    # 应该返回警告
    assert data2["ok"] is False
    assert data2["warning"] == "similar_memories_found"
    assert len(data2["similar_memories"]) > 0

    # 检查相似记忆的信息
    similar = data2["similar_memories"][0]
    assert similar["id"] == entry_id_1
    assert "数据库" in similar["content"]
    assert similar["similarity"] > 0.1  # 关键词匹配相似度

    # 3. 强制添加（使用 force=true）
    resp3 = client.post(
        "/api/memory/add?force=true",
        json={
            "employee": "测试员工",
            "category": "finding",
            "content": "修复了数据库连接池的内存泄漏问题，原因是连接未正确关闭，需要在 finally 块中显式调用 close() 方法释放资源",
            "tags": ["database"],
        },
        headers=headers,
    )

    assert resp3.status_code == 200
    data3 = resp3.json()
    assert data3["ok"] is True
    assert data3["skipped"] is False

    # 4. 更新已有记忆
    resp4 = client.put(
        "/api/memory/update",
        json={
            "entry_id": entry_id_1,
            "employee": "测试员工",
            "content": "修复了数据库连接池的内存泄漏问题，根因是连接未正确关闭。解决方案：在 finally 块中显式调用 close()，并添加连接池监控",
            "tags": ["database", "memory-leak", "monitoring"],
            "updated_by": "姜墨言",
        },
        headers=headers,
    )

    assert resp4.status_code == 200
    data4 = resp4.json()
    assert data4["ok"] is True
    assert data4["updated"] is True
    assert data4["entry_id"] == entry_id_1

    # 5. 验证更新后的记忆
    from crew.memory import MemoryStore

    store = MemoryStore(project_dir=tmp_path)
    entries = store.query("测试员工", limit=10)

    updated_entry = None
    for entry in entries:
        if entry.id == entry_id_1:
            updated_entry = entry
            break

    assert updated_entry is not None
    assert "监控" in updated_entry.content
    assert "monitoring" in updated_entry.tags
    assert any("updated-by:姜墨言" in tag for tag in updated_entry.tags)


def test_memory_update_nonexistent(tmp_path):
    """测试更新不存在的记忆."""
    client = _make_client(tmp_path)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    # 先添加一条记忆，确保员工存在
    client.post(
        "/api/memory/add",
        json={
            "employee": "测试员工",
            "category": "finding",
            "content": "这是一条测试记忆，用于确保员工文件存在，避免测试时返回 Employee not found 错误",
        },
        headers=headers,
    )

    # 尝试更新不存在的记忆
    resp = client.put(
        "/api/memory/update",
        json={
            "entry_id": "nonexistent",
            "employee": "测试员工",
            "content": "更新的内容",
        },
        headers=headers,
    )

    assert resp.status_code == 404
    data = resp.json()
    assert data["error"] == "Memory entry not found"


def test_memory_deduplication_different_categories(tmp_path):
    """测试不同类别的记忆不会被去重."""
    client = _make_client(tmp_path)
    headers = {"Authorization": f"Bearer {TOKEN}"}

    # 添加 finding 类别的记忆
    resp1 = client.post(
        "/api/memory/add",
        json={
            "employee": "测试员工",
            "category": "finding",
            "content": "发现数据库连接池存在内存泄漏问题，经过排查发现是连接未正确关闭导致的，需要在代码中添加 finally 块确保连接释放",
        },
        headers=headers,
    )

    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["ok"] is True

    # 添加 decision 类别的相似内容（不应该被拦截）
    resp2 = client.post(
        "/api/memory/add",
        json={
            "employee": "测试员工",
            "category": "decision",
            "content": "决定修复数据库连接池的内存泄漏问题，采用在 finally 块中显式关闭连接的方案，预计需要 2 天完成",
        },
        headers=headers,
    )

    assert resp2.status_code == 200
    data2 = resp2.json()

    # 不同类别，应该正常添加
    assert data2["ok"] is True
    assert data2["skipped"] is False
