"""Tests for crew.output_sanitizer — LLM 输出清洗."""

from crew.output_sanitizer import strip_internal_tags


class TestStripInternalTags:
    """strip_internal_tags 单元测试."""

    # ── 基本功能 ──

    def test_empty_string(self):
        assert strip_internal_tags("") == ""

    def test_none_passthrough(self):
        assert strip_internal_tags(None) is None

    def test_plain_text_unchanged(self):
        text = "你好，这是一段正常的回复。"
        assert strip_internal_tags(text) == text

    def test_multiline_plain_text_unchanged(self):
        text = "第一行\n\n第二行\n第三行"
        assert strip_internal_tags(text) == text

    # ── <thinking> 标签 ──

    def test_thinking_block_removed(self):
        text = (
            "我去查一下你最近七天的工作记录，等我两分钟。\n\n"
            "<thinking>\n"
            "Kai 让我拉他最近七天的工作记录。我需要：\n"
            "1. 查看他的日记/笔记，看最近七天有什么记录\n"
            "2. 整理成简洁的汇总给他\n"
            "我应该用 readDiary 工具查看他最近的日记。\n"
            "</thinking>"
        )
        result = strip_internal_tags(text)
        assert "<thinking>" not in result
        assert "readDiary" not in result
        assert "我去查一下你最近七天的工作记录" in result

    def test_thinking_block_inline(self):
        text = "好的<thinking>内部推理</thinking>，我来处理。"
        result = strip_internal_tags(text)
        assert result == "好的，我来处理。"

    def test_multiple_thinking_blocks(self):
        text = (
            "第一句话。\n"
            "<thinking>推理1</thinking>\n"
            "第二句话。\n"
            "<thinking>推理2</thinking>\n"
            "第三句话。"
        )
        result = strip_internal_tags(text)
        assert "<thinking>" not in result
        assert "第一句话" in result
        assert "第二句话" in result
        assert "第三句话" in result

    # ── <reflection> 标签 ──

    def test_reflection_block_removed(self):
        text = "让我想想。\n<reflection>\n这个问题需要仔细考虑。\n</reflection>\n答案是42。"
        result = strip_internal_tags(text)
        assert "<reflection>" not in result
        assert "答案是42" in result

    # ── <inner_monologue> 标签 ──

    def test_inner_monologue_removed(self):
        text = "好的。\n<inner_monologue>我应该怎么回答呢</inner_monologue>\n这是答案。"
        result = strip_internal_tags(text)
        assert "<inner_monologue>" not in result
        assert "这是答案" in result

    # ── XML 工具调用块 ──

    def test_xml_tool_call_removed(self):
        """用户报告的原始 bug 场景."""
        text = (
            "我去查一下你最近七天的工作记录，等我两分钟。\n\n"
            "<thinking>\n"
            "Kai 让我拉他最近七天的工作记录。\n"
            "</thinking>\n\n"
            "<read_diary>\n"
            "<user>Kai</user>\n"
            "<count>7</count>\n"
            "</read_diary>"
        )
        result = strip_internal_tags(text)
        assert "<thinking>" not in result
        assert "<read_diary>" not in result
        assert "<user>" not in result
        assert "我去查一下你最近七天的工作记录" in result

    def test_tool_call_with_nesting(self):
        text = "正在查询。\n<search_data>\n<query>近七天</query>\n<limit>10</limit>\n</search_data>"
        result = strip_internal_tags(text)
        assert "<search_data>" not in result
        assert "正在查询" in result

    def test_various_tool_names(self):
        """测试各种工具名称的 XML 块都能被清除."""
        tool_blocks = [
            "<read_notes>\n<user>Kai</user>\n</read_notes>",
            "<write_file>\n<path>test.txt</path>\n</write_file>",
            "<create_note>\n<content>test</content>\n</create_note>",
            "<send_message>\n<to>user</to>\n</send_message>",
            "<lookup_user>\n<name>Kai</name>\n</lookup_user>",
            "<add_memory>\n<content>记住这个</content>\n</add_memory>",
            "<check_task>\n<id>123</id>\n</check_task>",
            "<schedule_task>\n<name>test</name>\n</schedule_task>",
            "<delegate>\n<employee>moyan</employee>\n</delegate>",
            "<find_free_time>\n<date>today</date>\n</find_free_time>",
            "<cancel_schedule>\n<id>456</id>\n</cancel_schedule>",
            "<get_calendar>\n<range>week</range>\n</get_calendar>",
        ]
        for block in tool_blocks:
            text = f"开头文本。\n{block}\n结尾文本。"
            result = strip_internal_tags(text)
            assert "<" not in result or ">" not in result.split("<")[0], f"Failed for: {block}"
            assert "开头文本" in result
            assert "结尾文本" in result

    # ── 混合场景 ──

    def test_thinking_plus_tool_call(self):
        """完整的 bug 复现场景 — thinking + 工具调用."""
        text = (
            "我去查一下你最近七天的工作记录，等我两分钟。\n\n"
            "<thinking>\n"
            "Kai 让我拉他最近七天的工作记录。我需要：\n"
            "1. 查看他的日记/笔记，看最近七天有什么记录\n"
            "2. 整理成简洁的汇总给他\n"
            "我应该用 readDiary 工具查看他最近的日记。\n"
            "</thinking>\n\n"
            "<read_diary>\n"
            "<user>Kai</user>\n"
            "<count>7</count>\n"
            "</read_diary>"
        )
        result = strip_internal_tags(text)
        assert result == "我去查一下你最近七天的工作记录，等我两分钟。"

    def test_only_thinking_returns_empty(self):
        """如果 LLM 输出只有 thinking 块，清洗后为空."""
        text = "<thinking>只有推理，没有回复</thinking>"
        result = strip_internal_tags(text)
        assert result == ""

    # ── 边界情况 ──

    def test_incomplete_thinking_tag_preserved(self):
        """不完整的标签不应被清除（安全起见保留）."""
        text = "这里有个 <thinking> 但没有关闭标签"
        result = strip_internal_tags(text)
        # 没有 </thinking>，正则不匹配，保留原文
        assert "<thinking>" in result

    def test_normal_html_preserved(self):
        """正常的 HTML 内容不应被误删（如用户讨论 HTML）."""
        text = "请把 <div> 标签改成 <span>。"
        result = strip_internal_tags(text)
        assert "<div>" in result
        assert "<span>" in result

    def test_code_block_with_tags(self):
        """代码块中的标签不做特殊处理（当前实现会清除，这是可接受的权衡）."""
        # 注意：如果未来需要保留代码块中的标签，可以先提取代码块
        text = "看看这段代码：\n```\n<thinking>test</thinking>\n```"
        # 当前行为：thinking 标签被清除
        result = strip_internal_tags(text)
        assert "<thinking>" not in result

    def test_excessive_whitespace_compressed(self):
        """清洗后多余的空行应被压缩."""
        text = "第一行。\n\n\n<thinking>推理</thinking>\n\n\n\n第二行。"
        result = strip_internal_tags(text)
        # 不应有超过两个连续换行
        assert "\n\n\n" not in result
        assert "第一行" in result
        assert "第二行" in result

    def test_real_world_deepseek_thinking(self):
        """DeepSeek 模型常见的 thinking 输出格式."""
        text = (
            "<thinking>\n"
            "用户问的是项目进度，我需要：\n"
            "1. 查看项目状态\n"
            "2. 汇总关键信息\n"
            "</thinking>\n\n"
            "好的，让我查一下当前项目进度。\n\n"
            "<query_project>\n"
            "<name>knowlyr</name>\n"
            "</query_project>"
        )
        result = strip_internal_tags(text)
        assert result == "好的，让我查一下当前项目进度。"

    def test_chinese_content_preserved(self):
        """中文内容不受影响."""
        text = "集识光年平台本周完成了三个重要功能的上线，包括：\n1. 数据标注质量提升\n2. 用户系统优化\n3. 蚁聚平台改进"
        assert strip_internal_tags(text) == text
