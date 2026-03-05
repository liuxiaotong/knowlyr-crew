"""测试记忆标签规范和词典."""

from crew.memory_tags import (
    get_all_predefined_tags,
    normalize_tag,
    normalize_tags,
    search_tags,
    suggest_tags,
    validate_tags,
)


class TestNormalizeTag:
    """测试标签规范化."""

    def test_lowercase(self):
        assert normalize_tag("Memory_System") == "memory-system"
        assert normalize_tag("API") == "api"

    def test_underscore_to_hyphen(self):
        assert normalize_tag("memory_system") == "memory-system"
        assert normalize_tag("api_gotcha") == "api-gotcha"

    def test_space_to_hyphen(self):
        assert normalize_tag("API Gotcha") == "api-gotcha"
        assert normalize_tag("best practice") == "best-practice"

    def test_strip_whitespace(self):
        assert normalize_tag("  backend  ") == "backend"
        assert normalize_tag(" api ") == "api"

    def test_remove_special_chars(self):
        assert normalize_tag("api@gotcha") == "apigotcha"
        assert normalize_tag("test#tag") == "testtag"

    def test_consecutive_hyphens(self):
        assert normalize_tag("api---gotcha") == "api-gotcha"
        assert normalize_tag("test--tag") == "test-tag"

    def test_strip_leading_trailing_hyphens(self):
        assert normalize_tag("-api-") == "api"
        assert normalize_tag("--backend--") == "backend"

    def test_empty_input(self):
        assert normalize_tag("") == ""
        assert normalize_tag("   ") == ""

    def test_chinese_tags(self):
        assert normalize_tag("后端开发") == "后端开发"
        assert normalize_tag("API_接口") == "api-接口"


class TestNormalizeTags:
    """测试批量标签规范化."""

    def test_batch_normalize(self):
        tags = ["Memory_System", "API Gotcha", "backend"]
        result = normalize_tags(tags)
        assert result == ["api-gotcha", "backend", "memory-system"]

    def test_deduplication(self):
        tags = ["backend", "Backend", "BACKEND"]
        result = normalize_tags(tags)
        assert result == ["backend"]

    def test_remove_empty(self):
        tags = ["backend", "", "  ", "api"]
        result = normalize_tags(tags)
        assert result == ["api", "backend"]

    def test_sorted_output(self):
        tags = ["zebra", "apple", "banana"]
        result = normalize_tags(tags)
        assert result == ["apple", "banana", "zebra"]


class TestSuggestTags:
    """测试标签建议."""

    def test_suggest_tech_tags(self):
        content = "修复了 API 的数据库查询性能问题"
        suggestions = suggest_tags("correction", content)
        assert "api" in suggestions
        assert "database" in suggestions

    def test_suggest_project_tags(self):
        content = "在 knowlyr-crew 项目中实现了新功能"
        suggestions = suggest_tags("finding", content)
        assert "knowlyr-crew" in suggestions

    def test_suggest_service_tags(self):
        content = "企微 API 返回字段有陷阱"
        suggestions = suggest_tags("correction", content)
        assert "wecom" in suggestions

    def test_suggest_by_category_correction(self):
        content = "教训：必须先验证再部署，避免生产事故"
        suggestions = suggest_tags("correction", content)
        assert "gotcha" in suggestions or "lesson-learned" in suggestions

    def test_suggest_by_category_pattern(self):
        content = "最佳实践：代码审查流程的标准步骤"
        suggestions = suggest_tags("pattern", content)
        assert "best-practice" in suggestions or "workflow" in suggestions

    def test_suggest_by_category_finding_bug(self):
        content = "修复了登录页面的 bug"
        suggestions = suggest_tags("finding", content)
        assert "bug-fix" in suggestions

    def test_suggest_by_category_finding_feature(self):
        content = "实现了新的用户管理功能"
        suggestions = suggest_tags("finding", content)
        assert "feature" in suggestions

    def test_suggest_api_gotcha(self):
        content = "API 字段映射陷阱：管理后台和接口返回不一致"
        suggestions = suggest_tags("correction", content)
        assert "api-gotcha" in suggestions

    def test_suggest_code_review(self):
        content = "代码审查发现了安全漏洞"
        suggestions = suggest_tags("finding", content)
        assert "code-review" in suggestions

    def test_suggest_incident(self):
        content = "生产环境崩溃事故分析"
        suggestions = suggest_tags("finding", content)
        assert "incident" in suggestions

    def test_exclude_existing_tags(self):
        content = "API 数据库性能优化"
        existing = ["api"]
        suggestions = suggest_tags("correction", content, existing)
        assert "api" not in suggestions
        assert "database" in suggestions

    def test_limit_suggestions(self):
        content = "API 数据库性能优化 backend frontend deploy"
        suggestions = suggest_tags("correction", content)
        assert len(suggestions) <= 5


class TestValidateTags:
    """测试标签验证."""

    def test_valid_tags(self):
        tags = ["backend", "api", "database"]
        result = validate_tags(tags)
        assert result["valid"] is True
        assert result["issues"] == []
        assert result["normalized"] == ["backend", "api", "database"]

    def test_normalize_during_validation(self):
        tags = ["Backend", "API_Gotcha", "data base"]
        result = validate_tags(tags)
        assert result["valid"] is True
        assert result["normalized"] == ["backend", "api-gotcha", "data-base"]

    def test_empty_tag(self):
        tags = ["backend", "", "api"]
        result = validate_tags(tags)
        assert result["valid"] is False
        assert any("空" in issue for issue in result["issues"])
        assert result["normalized"] == ["backend", "api"]

    def test_too_long_tag(self):
        tags = ["a" * 60]
        result = validate_tags(tags)
        assert result["valid"] is False
        assert any("过长" in issue for issue in result["issues"])

    def test_reserved_tag(self):
        tags = ["backend", "trajectory"]
        result = validate_tags(tags)
        assert result["valid"] is False
        assert any("保留" in issue for issue in result["issues"])
        assert result["normalized"] == ["backend"]

    def test_deduplication_in_validation(self):
        tags = ["backend", "Backend", "BACKEND"]
        result = validate_tags(tags)
        assert result["valid"] is True
        assert result["normalized"] == ["backend"]

    def test_invalid_type(self):
        tags = ["backend", 123, "api"]
        result = validate_tags(tags)
        assert result["valid"] is False
        assert any("字符串" in issue for issue in result["issues"])


class TestGetAllPredefinedTags:
    """测试获取所有预定义标签."""

    def test_returns_all_categories(self):
        tags = get_all_predefined_tags()
        assert "tech" in tags
        assert "projects" in tags
        assert "types" in tags
        assert "status" in tags
        assert "services" in tags
        assert "reserved" in tags

    def test_tech_tags_structure(self):
        tags = get_all_predefined_tags()
        assert isinstance(tags["tech"], dict)
        assert "backend" in tags["tech"]
        assert "api" in tags["tech"]

    def test_project_tags_list(self):
        tags = get_all_predefined_tags()
        assert isinstance(tags["projects"], list)
        assert "knowlyr-crew" in tags["projects"]


class TestSearchTags:
    """测试标签搜索."""

    def test_exact_match(self):
        results = search_tags("backend")
        assert "backend" in results
        assert results[0] == "backend"  # 完全匹配排第一

    def test_partial_match(self):
        results = search_tags("api")
        assert "api" in results
        assert "api-gotcha" in results

    def test_case_insensitive(self):
        results = search_tags("API")
        assert "api" in results

    def test_limit_results(self):
        results = search_tags("a", limit=5)
        assert len(results) <= 5

    def test_empty_query(self):
        results = search_tags("")
        assert results == []

    def test_no_match(self):
        results = search_tags("nonexistent-tag-xyz")
        assert results == []
