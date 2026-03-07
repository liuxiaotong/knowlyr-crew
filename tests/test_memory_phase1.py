"""记忆系统 Phase 1 改进 - 冒烟测试."""

import types
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_memory_entry(**kwargs):
    defaults = {
        "id": "mem_001",
        "employee": "test-employee",
        "category": "finding",
        "content": "测试记忆内容",
        "source_session": "",
        "superseded_by": None,
        "classification": "internal",
        "tags": [],
        "created_at": datetime.now().isoformat(),
    }
    defaults.update(kwargs)
    entry = types.SimpleNamespace(**defaults)
    entry.model_dump = lambda: dict(defaults.items())
    return entry


def _make_mock_store(entries=None):
    entries = entries or []
    store = MagicMock()
    store.memory_dir = Path("/tmp/test-memory")
    store.query.return_value = entries
    store._load_employee_entries.return_value = entries
    store._is_expired.return_value = False
    store._apply_decay.side_effect = lambda e: e
    return store


class TestSemanticMemorySearchWebhook:
    def test_fallback_to_keyword_when_no_index(self):
        from crew.webhook_handlers import _semantic_memory_search

        entries = [
            _make_memory_entry(id="m1", content="API 接口设计规范"),
            _make_memory_entry(id="m2", content="数据库迁移步骤"),
            _make_memory_entry(id="m3", content="API 认证方案"),
        ]
        store = _make_mock_store(entries)

        with patch(
            "crew.memory_search.SemanticMemoryIndex",
            side_effect=ImportError("no module"),
        ):
            result = _semantic_memory_search(store, "test-employee", "API", None, 10)

        assert len(result) >= 1

    def test_with_semantic_index(self):
        from crew.webhook_handlers import _semantic_memory_search

        entries = [
            _make_memory_entry(id="m1", content="API 接口设计规范"),
            _make_memory_entry(id="m2", content="数据库迁移步骤"),
        ]
        store = _make_mock_store(entries)

        mock_index = MagicMock()
        mock_index.has_index.return_value = True
        mock_index.search.return_value = [("m1", "API 接口设计规范", 0.9)]
        mock_index.__enter__ = MagicMock(return_value=mock_index)
        mock_index.__exit__ = MagicMock(return_value=False)

        with patch(
            "crew.memory_search.SemanticMemoryIndex",
            MagicMock(return_value=mock_index),
        ):
            result = _semantic_memory_search(store, "test-employee", "API 设计", None, 10)

        assert len(result) == 1
        assert result[0].id == "m1"

    def test_category_filter_with_index(self):
        from crew.webhook_handlers import _semantic_memory_search

        entries = [
            _make_memory_entry(id="m1", content="API 设计", category="finding"),
            _make_memory_entry(id="m2", content="API 改进", category="correction"),
        ]
        store = _make_mock_store(entries)

        mock_index = MagicMock()
        mock_index.has_index.return_value = True
        mock_index.search.return_value = [
            ("m1", "API 设计", 0.9),
            ("m2", "API 改进", 0.8),
        ]
        mock_index.__enter__ = MagicMock(return_value=mock_index)
        mock_index.__exit__ = MagicMock(return_value=False)

        with patch(
            "crew.memory_search.SemanticMemoryIndex",
            MagicMock(return_value=mock_index),
        ):
            result = _semantic_memory_search(
                store, "test-employee", "API", "finding", 10
            )

        assert len(result) == 1
        assert result[0].category == "finding"

    def test_no_index_fallback(self):
        from crew.webhook_handlers import _semantic_memory_search

        entries = [_make_memory_entry(id="m1", content="部署流程说明")]
        store = _make_mock_store(entries)

        mock_index = MagicMock()
        mock_index.has_index.return_value = False
        mock_index.__enter__ = MagicMock(return_value=mock_index)
        mock_index.__exit__ = MagicMock(return_value=False)

        with patch(
            "crew.memory_search.SemanticMemoryIndex",
            MagicMock(return_value=mock_index),
        ):
            result = _semantic_memory_search(store, "test-employee", "部署", None, 10)

        assert len(result) == 1


class TestCronEvaluateDedup:
    def test_found_existing_source(self):
        from crew.cron_evaluate import _memory_exists_by_source

        entries = [_make_memory_entry(source_session="cron:overdue:D001")]
        store = _make_mock_store(entries)
        assert _memory_exists_by_source(store, "test-employee", "cron:overdue:D001") is True

    def test_not_found(self):
        from crew.cron_evaluate import _memory_exists_by_source

        store = _make_mock_store([_make_memory_entry(source_session="other")])
        assert (
            _memory_exists_by_source(store, "test-employee", "cron:overdue:D999") is False
        )

    def test_empty_store(self):
        from crew.cron_evaluate import _memory_exists_by_source

        store = _make_mock_store([])
        assert _memory_exists_by_source(store, "test-employee", "anything") is False


class TestEvaluationCorrectionDedup:
    def test_evaluate_writes_correction_once(self, tmp_path):
        from crew.evaluation import EvaluationEngine

        engine = EvaluationEngine(eval_dir=tmp_path)
        decision = engine.track(
            employee="test-employee",
            category="estimate",
            content="估算开发时间为 3 天",
            expected_outcome="3 天完成",
        )

        mock_store = _make_mock_store([])
        mock_store.add = MagicMock()

        with patch("crew.memory.get_memory_store", return_value=mock_store):
            result1 = engine.evaluate(decision.id, "实际用了 5 天", "低估了复杂度")
            assert result1 is not None
            assert mock_store.add.call_count == 1

            mock_store.query.return_value = [
                _make_memory_entry(
                    source_session=f"eval:{decision.id}", category="correction"
                )
            ]
            mock_store.add.reset_mock()

            d2 = engine.track(
                employee="test-employee", category="estimate", content="另一个估算"
            )
            mock_store.query.return_value = [
                _make_memory_entry(
                    source_session=f"eval:{d2.id}", category="correction"
                )
            ]
            result2 = engine.evaluate(d2.id, "实际结果", "评估内容")
            assert result2 is not None
            assert mock_store.add.call_count == 0


class TestReadWikiAction:
    def test_no_wiki_config(self):
        from crew.skills import SkillAction
        from crew.skills_engine import SkillsEngine

        engine = SkillsEngine(MagicMock(), MagicMock())
        action = SkillAction(type="read_wiki", params={"doc_id": "doc123"})
        with patch.dict("os.environ", {}, clear=True):
            result = engine._execute_read_wiki(action, "test-employee", {})
        assert result["content"] == ""
        assert "error" in result

    def test_success_with_doc_id(self):
        from crew.skills import SkillAction
        from crew.skills_engine import SkillsEngine

        engine = SkillsEngine(MagicMock(), MagicMock())
        action = SkillAction(type="read_wiki", params={"doc_id": "doc123"})

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": "# 文档内容", "title": "测试文档"}
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        env = {"WIKI_API_URL": "https://wiki.example.com", "WIKI_ADMIN_TOKEN": "tok"}
        with patch.dict("os.environ", env, clear=True):
            with patch("httpx.Client", return_value=mock_client):
                result = engine._execute_read_wiki(action, "test-employee", {})
        assert result["content"] == "# 文档内容"
        assert result["doc_id"] == "doc123"

    def test_missing_params(self):
        from crew.skills import SkillAction
        from crew.skills_engine import SkillsEngine

        engine = SkillsEngine(MagicMock(), MagicMock())
        action = SkillAction(type="read_wiki", params={})
        env = {"WIKI_API_URL": "https://wiki.example.com", "WIKI_ADMIN_TOKEN": "tok"}
        with patch.dict("os.environ", env, clear=True):
            result = engine._execute_read_wiki(action, "test-employee", {})
        assert result["content"] == ""
        assert "error" in result

    def test_enhanced_context_wiki_docs(self):
        from crew.skills import Skill, SkillAction, SkillMetadata, SkillTrigger
        from crew.skills_engine import SkillsEngine

        engine = SkillsEngine(MagicMock(), MagicMock())
        skill = Skill(
            name="test-wiki-skill",
            version="0.1.0",
            employee="test-employee",
            description="test",
            trigger=SkillTrigger(type="always"),
            actions=[SkillAction(type="read_wiki", params={"doc_id": "doc1"})],
            metadata=SkillMetadata(),
        )
        with patch.object(
            engine,
            "_execute_read_wiki",
            return_value={
                "content": "Wiki 文档内容",
                "title": "参考文档",
                "doc_id": "doc1",
            },
        ):
            result = engine.execute_skill(skill, "test-employee")
        assert "wiki_docs" in result["enhanced_context"]
        assert result["enhanced_context"]["wiki_docs"][0]["title"] == "参考文档"
