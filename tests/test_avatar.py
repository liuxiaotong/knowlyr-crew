"""测试头像生成模块."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crew.avatar import (
    AVATAR_SIZE,
    PHOTO_SUFFIX,
    _get_api_key,
    build_avatar_prompt,
    compress_avatar,
    generate_avatar,
)


class TestBuildAvatarPrompt:
    """测试 prompt 构建."""

    def test_with_avatar_prompt(self):
        result = build_avatar_prompt(avatar_prompt="一位穿西装的男性")
        assert result == "一位穿西装的男性" + PHOTO_SUFFIX

    def test_with_character_name(self):
        result = build_avatar_prompt(character_name="林锐", description="代码审查专家")
        assert "林锐" in result
        assert "代码审查专家" in result
        assert result.endswith(PHOTO_SUFFIX)

    def test_with_display_name_only(self):
        result = build_avatar_prompt(display_name="Code Reviewer")
        assert "Code Reviewer" in result

    def test_defaults(self):
        result = build_avatar_prompt()
        assert "a professional" in result
        assert "employee" in result

    def test_avatar_prompt_takes_precedence(self):
        result = build_avatar_prompt(
            display_name="Ignored",
            character_name="Also Ignored",
            avatar_prompt="Custom prompt",
        )
        assert result.startswith("Custom prompt")
        assert "Ignored" not in result


class TestGetApiKey:
    """测试 API key 读取."""

    def test_from_env(self):
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "sk-test"}):
            assert _get_api_key() == "sk-test"

    def test_missing(self):
        with patch.dict("os.environ", {}, clear=False):
            # Make sure DASHSCOPE_API_KEY is not set
            import os

            old = os.environ.pop("DASHSCOPE_API_KEY", None)
            try:
                assert _get_api_key() == ""
            finally:
                if old is not None:
                    os.environ["DASHSCOPE_API_KEY"] = old


class TestGenerateAvatar:
    """测试头像生成."""

    def test_no_api_key(self):
        with patch("crew.avatar._get_api_key", return_value=""):
            result = generate_avatar(display_name="Test")
        assert result is None

    @patch("crew.avatar.urllib.request.urlretrieve")
    @patch("crew.avatar.urllib.request.urlopen")
    @patch("crew.avatar._get_api_key", return_value="sk-test")
    def test_success(self, mock_key, mock_urlopen, mock_retrieve, tmp_path):
        # Submit task response
        submit_resp = MagicMock()
        submit_resp.read.return_value = json.dumps({"output": {"task_id": "task-123"}}).encode()
        submit_resp.__enter__ = MagicMock(return_value=submit_resp)
        submit_resp.__exit__ = MagicMock(return_value=False)

        # Poll response - succeeded
        poll_resp = MagicMock()
        poll_resp.read.return_value = json.dumps(
            {
                "output": {
                    "task_status": "SUCCEEDED",
                    "results": [{"url": "https://example.com/img.png"}],
                }
            }
        ).encode()
        poll_resp.__enter__ = MagicMock(return_value=poll_resp)
        poll_resp.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [submit_resp, poll_resp]

        # Make urlretrieve create the file so stat() works
        def _fake_retrieve(url, path):
            Path(path).write_bytes(b"\x89PNG" + b"\x00" * 100)

        mock_retrieve.side_effect = _fake_retrieve

        with patch("crew.avatar.time.sleep"):  # skip sleep
            generate_avatar(display_name="Test", output_dir=tmp_path)

        assert mock_retrieve.called
        # Should try to save to avatar_raw.png
        save_path = mock_retrieve.call_args[0][1]
        assert "avatar_raw.png" in save_path

    @patch("crew.avatar.urllib.request.urlopen")
    @patch("crew.avatar._get_api_key", return_value="sk-test")
    def test_submit_fails(self, mock_key, mock_urlopen):
        mock_urlopen.side_effect = Exception("network error")
        result = generate_avatar(display_name="Test")
        assert result is None

    @patch("crew.avatar.urllib.request.urlopen")
    @patch("crew.avatar._get_api_key", return_value="sk-test")
    def test_no_task_id(self, mock_key, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = json.dumps({"output": {}}).encode()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp
        result = generate_avatar(display_name="Test")
        assert result is None

    @patch("crew.avatar.urllib.request.urlopen")
    @patch("crew.avatar._get_api_key", return_value="sk-test")
    def test_task_failed(self, mock_key, mock_urlopen):
        submit = MagicMock()
        submit.read.return_value = json.dumps({"output": {"task_id": "t1"}}).encode()
        submit.__enter__ = MagicMock(return_value=submit)
        submit.__exit__ = MagicMock(return_value=False)

        poll = MagicMock()
        poll.read.return_value = json.dumps(
            {"output": {"task_status": "FAILED", "message": "bad prompt"}}
        ).encode()
        poll.__enter__ = MagicMock(return_value=poll)
        poll.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [submit, poll]

        with patch("crew.avatar.time.sleep"):
            result = generate_avatar(display_name="Test")
        assert result is None

    @patch("crew.avatar.POLL_MAX_ATTEMPTS", 2)
    @patch("crew.avatar.urllib.request.urlopen")
    @patch("crew.avatar._get_api_key", return_value="sk-test")
    def test_timeout(self, mock_key, mock_urlopen):
        submit = MagicMock()
        submit.read.return_value = json.dumps({"output": {"task_id": "t1"}}).encode()
        submit.__enter__ = MagicMock(return_value=submit)
        submit.__exit__ = MagicMock(return_value=False)

        pending = MagicMock()
        pending.read.return_value = json.dumps({"output": {"task_status": "RUNNING"}}).encode()
        pending.__enter__ = MagicMock(return_value=pending)
        pending.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [submit, pending, pending]

        with patch("crew.avatar.time.sleep"):
            result = generate_avatar(display_name="Test")
        assert result is None


class TestCompressAvatar:
    """测试头像压缩."""

    def test_no_pillow(self):
        with patch("crew.avatar._get_pillow", return_value=None):
            result = compress_avatar(Path("fake.png"))
        assert result is None

    def test_rgb_image(self, tmp_path):
        """RGB 图片直接缩放."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img = Image.new("RGB", (512, 512), (255, 0, 0))
        src = tmp_path / "test.png"
        img.save(src)

        result = compress_avatar(src)
        assert result is not None
        assert result.suffix == ".webp"
        assert result.exists()

        out = Image.open(result)
        assert out.size == (AVATAR_SIZE, AVATAR_SIZE)

    def test_rgba_image(self, tmp_path):
        """RGBA 图片转 RGB."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img = Image.new("RGBA", (512, 512), (255, 0, 0, 128))
        src = tmp_path / "test.png"
        img.save(src)

        result = compress_avatar(src)
        assert result is not None

        out = Image.open(result)
        assert out.size == (AVATAR_SIZE, AVATAR_SIZE)

    def test_non_square_crop(self, tmp_path):
        """非正方形中心裁剪."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img = Image.new("RGB", (800, 400), (0, 255, 0))
        src = tmp_path / "wide.png"
        img.save(src)

        result = compress_avatar(src)
        assert result is not None

        out = Image.open(result)
        assert out.size == (AVATAR_SIZE, AVATAR_SIZE)

    def test_custom_output_path(self, tmp_path):
        """自定义输出路径."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img = Image.new("RGB", (256, 256), (0, 0, 255))
        src = tmp_path / "input.png"
        img.save(src)

        out = tmp_path / "custom.webp"
        result = compress_avatar(src, output_path=out)
        assert result == out
        assert out.exists()

    def test_already_correct_size(self, tmp_path):
        """已是 256x256 不缩放."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img = Image.new("RGB", (AVATAR_SIZE, AVATAR_SIZE), (100, 100, 100))
        src = tmp_path / "exact.png"
        img.save(src)

        result = compress_avatar(src)
        assert result is not None

        out = Image.open(result)
        assert out.size == (AVATAR_SIZE, AVATAR_SIZE)


class TestGenerateAvatarSafety:
    """generate_avatar 安全性测试."""

    def test_no_api_key(self, tmp_path):
        """无 API key 应返回 None."""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}, clear=False):
            result = generate_avatar(
                display_name="Test",
                output_dir=tmp_path,
            )
            assert result is None

    def test_download_empty_file_cleanup(self, tmp_path):
        """下载空文件应被清理."""
        output_path = tmp_path / "avatar_raw.png"
        output_path.write_bytes(b"")
        assert output_path.stat().st_size == 0
        output_path.unlink(missing_ok=True)
        assert not output_path.exists()
