"""测试记忆质量控制."""

from crew.memory_quality import check_memory_quality


class TestMemoryQuality:
    """记忆质量检查测试."""

    def test_high_quality_correction(self):
        """高质量 correction 记忆应该通过."""
        content = "教训：alembic migration 必须先检查字段是否存在再 add_column，否则重复执行会报错。正确做法是先查 information_schema.columns。"
        result = check_memory_quality("correction", content)
        assert result["score"] >= 0.6, f"高质量记忆被误判：{result}"
        assert len(result["issues"]) == 0

    def test_high_quality_pattern(self):
        """高质量 pattern 记忆应该通过."""
        content = "部署流程：git push main 触发 GitHub Actions 自动部署。紧急情况可用 make push 旁路。原则是禁止直接 SSH 改服务器文件，避免状态不一致。"
        result = check_memory_quality("pattern", content)
        assert result["score"] >= 0.6, f"高质量记忆被误判：{result}"
        assert len(result["issues"]) == 0

    def test_high_quality_finding(self):
        """高质量 finding 记忆应该通过（不强制关键词）."""
        content = "发现 trajectory 标签的记忆会污染正常记忆，因为轨迹是机械记录，不是人工提炼的经验。需要在 API 层拦截。"
        result = check_memory_quality("finding", content)
        assert result["score"] >= 0.6, f"高质量记忆被误判：{result}"

    def test_too_short_content(self):
        """内容过短应该被拒绝."""
        content = "修复了 bug"
        result = check_memory_quality("correction", content)
        assert result["score"] < 0.6
        assert any("过短" in issue for issue in result["issues"])
        assert any("补充更多细节" in s for s in result["suggestions"])

    def test_too_long_content(self):
        """内容过长应该扣分."""
        content = "A" * 600  # 超过 500 字符
        result = check_memory_quality("correction", content)
        assert result["score"] < 0.6
        assert any("过长" in issue for issue in result["issues"])
        assert any("提炼" in s or "拆分" in s for s in result["suggestions"])

    def test_missing_keywords_correction(self):
        """correction 缺少关键词应该扣分."""
        content = (
            "今天做了一个功能，把 API 改了一下，然后测试通过了，就提交了代码。" * 2
        )  # 凑够长度
        result = check_memory_quality("correction", content)
        assert result["score"] < 0.6
        assert any("缺少关键词" in issue for issue in result["issues"])

    def test_missing_keywords_pattern(self):
        """pattern 缺少关键词应该扣分."""
        content = "今天写了一些代码，改了几个文件，然后提交了。过程比较顺利，没遇到什么问题。"
        result = check_memory_quality("pattern", content)
        assert result["score"] < 0.6
        assert any("缺少关键词" in issue for issue in result["issues"])

    def test_trajectory_prefix(self):
        """以 [轨迹] 开头应该被拒绝."""
        content = (
            "[轨迹] 修复了 webhook_handlers.py 的 bug，添加了质量检查逻辑，测试通过后提交代码。"
        )
        result = check_memory_quality("correction", content)
        assert result["score"] < 0.6
        assert any("[轨迹]" in issue for issue in result["issues"])
        assert any("移除" in s for s in result["suggestions"])

    def test_missing_depth(self):
        """缺少深度内容（为什么、如何）应该扣分."""
        content = "修复了 API 的问题。更新了代码。提交了 commit。部署到服务器。验证功能正常。"
        result = check_memory_quality("correction", content)
        assert result["score"] < 0.6
        assert any("深度" in issue for issue in result["issues"])

    def test_trivial_pattern_detection(self):
        """流水账检测：开头是"修复了"等词，且没有深度内容."""
        content = "修复了数据库连接问题，更新了配置文件，重启了服务，问题解决了。"
        result = check_memory_quality("correction", content)
        assert result["score"] < 0.6
        assert any("流水账" in issue for issue in result["issues"])

    def test_boundary_min_length(self):
        """边界测试：刚好 50 字符."""
        content = "教训：部署前必须先 git pull 同步远端，避免基于过时代码工作导致冲突问题出现。"  # 刚好 50 字符
        result = check_memory_quality("correction", content)
        # 应该通过长度检查，但可能因为其他原因扣分
        assert result["score"] >= 0.3  # 至少拿到长度分

    def test_boundary_max_length_correction(self):
        """边界测试：刚好 500 字符（correction）."""
        content = "教训：" + "A" * 493  # 总共 500 字符
        result = check_memory_quality("correction", content)
        # 应该通过长度检查
        assert result["score"] >= 0.3

    def test_boundary_max_length_finding(self):
        """边界测试：刚好 1000 字符（finding）."""
        content = "发现：" + "A" * 993  # 总共 1000 字符
        result = check_memory_quality("finding", content)
        # 应该通过长度检查
        assert result["score"] >= 0.3

    def test_decision_no_keyword_requirement(self):
        """decision 类型不强制关键词."""
        content = "决定使用 PostgreSQL 作为主数据库，因为需要 JSONB 和全文搜索功能，MySQL 不够用。"
        result = check_memory_quality("decision", content)
        # 应该拿到关键词分（0.4），即使没有 correction/pattern 的关键词
        assert result["score"] >= 0.6

    def test_estimate_no_keyword_requirement(self):
        """estimate 类型不强制关键词."""
        content = "预估这个功能需要 3 天完成，包括 1 天设计、1 天开发、1 天测试。风险点是第三方 API 可能不稳定。"
        result = check_memory_quality("estimate", content)
        assert result["score"] >= 0.6

    def test_good_correction_with_all_elements(self):
        """完美的 correction：有关键词、有深度、长度合适."""
        content = """教训：后端 schema 改动必须本地验证再推生产。

        根因：直接改生产 schema 导致 API 启动失败，因为 Pydantic model 和数据库不一致。

        正确做法：
        1. 本地先 alembic upgrade head
        2. uvicorn 启动验证无报错
        3. 再推到生产环境

        这次事故影响了 10 分钟服务，教训深刻。"""

        result = check_memory_quality("correction", content)
        assert result["score"] >= 0.8, f"完美记忆得分过低：{result}"
        assert len(result["issues"]) == 0

    def test_good_pattern_with_trigger_condition(self):
        """完美的 pattern：有方法、有步骤、有原则."""
        content = """部署策略：使用 git push 触发 CI/CD 自动部署。

        流程：
        1. git push main 推送到 GitHub
        2. GitHub Actions 自动构建和部署
        3. 健康检查通过后切换流量

        原则：禁止直接 SSH 改服务器文件，避免状态不一致。
        紧急情况可用 make push 旁路，但事后必须补 commit。"""

        result = check_memory_quality("pattern", content)
        assert result["score"] >= 0.8, f"完美记忆得分过低：{result}"
        assert len(result["issues"]) == 0

    def test_score_range(self):
        """验证分数在 0-1 范围内."""
        test_cases = [
            ("correction", "A"),  # 极短
            ("correction", "A" * 1000),  # 极长
            (
                "correction",
                "教训：这是一个很好的经验，说明了为什么要这样做，以及如何避免问题。",
            ),  # 完美
        ]

        for category, content in test_cases:
            result = check_memory_quality(category, content)
            assert 0 <= result["score"] <= 1, f"分数超出范围：{result['score']}"

    def test_suggestions_always_present_when_low_score(self):
        """低分记忆必须有改进建议."""
        content = "修复了 bug"
        result = check_memory_quality("correction", content)

        if result["score"] < 0.6:
            assert len(result["suggestions"]) > 0, "低分记忆必须提供改进建议"
