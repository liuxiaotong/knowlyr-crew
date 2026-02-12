"""头像生成模块 — 调用 Gemini CLI 生成 + Pillow 压缩."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

AVATAR_SIZE = 256
AVATAR_QUALITY = 85


def _get_pillow():
    """延迟导入 Pillow，未安装时返回 None."""
    try:
        from PIL import Image
        return Image
    except ImportError:
        return None


def _find_gemini() -> str | None:
    """查找 gemini CLI 可执行文件路径."""
    return shutil.which("gemini")


def build_avatar_prompt(
    display_name: str = "",
    character_name: str = "",
    description: str = "",
    avatar_prompt: str = "",
) -> str:
    """构建头像生成 prompt.

    方案 D：有 avatar_prompt 就用，没有则自动推断。
    """
    if avatar_prompt:
        return avatar_prompt

    # 自动推断
    name = character_name or display_name or "a professional"
    role = description or "employee"

    return (
        f"Generate a professional headshot photo of {name}, "
        f"who works as {role}. "
        "Neutral light grey background, soft studio lighting, "
        "head and shoulders framing, looking at camera with a natural expression. "
        "High quality, photorealistic, corporate profile photo style."
    )


def generate_avatar(
    display_name: str = "",
    character_name: str = "",
    description: str = "",
    avatar_prompt: str = "",
    output_dir: Path | None = None,
) -> Path | None:
    """调用 Gemini CLI 生成头像.

    Args:
        display_name: 员工显示名
        character_name: 角色姓名
        description: 一句话描述
        avatar_prompt: 自定义头像 prompt（优先使用）
        output_dir: 输出目录

    Returns:
        生成的图片路径（avatar.png），失败返回 None
    """
    gemini = _find_gemini()
    if not gemini:
        logger.error("未找到 gemini CLI，请先安装: npm install -g @anthropic-ai/gemini")
        return None

    if output_dir is None:
        output_dir = Path.cwd()

    output_path = output_dir / "avatar.png"

    prompt = build_avatar_prompt(
        display_name=display_name,
        character_name=character_name,
        description=description,
        avatar_prompt=avatar_prompt,
    )

    # 完整 prompt：生成图片 + 保存到指定路径
    full_prompt = (
        f"{prompt}\n\n"
        f"Save the generated image to: {output_path}"
    )

    try:
        result = subprocess.run(
            [gemini, "-p", full_prompt, "--sandbox", "false"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(output_dir),
        )

        if result.returncode != 0:
            logger.warning("Gemini CLI 非零退出: %s\nstderr: %s", result.returncode, result.stderr)

        # 检查是否生成了文件
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path

        # 也检查 webp 格式（Gemini 可能直接输出 webp）
        webp_path = output_dir / "avatar.webp"
        if webp_path.exists() and webp_path.stat().st_size > 0:
            return webp_path

        logger.warning("Gemini CLI 执行完成但未生成头像文件")
        return None

    except subprocess.TimeoutExpired:
        logger.error("Gemini CLI 执行超时（120s）")
        return None
    except Exception as e:
        logger.error("Gemini CLI 执行异常: %s", e)
        return None


def compress_avatar(input_path: Path, output_path: Path | None = None) -> Path | None:
    """压缩头像到 web 规格（256x256 webp）.

    Args:
        input_path: 原始图片路径
        output_path: 输出路径（默认同目录 avatar.webp）

    Returns:
        压缩后的图片路径，失败返回 None
    """
    Image = _get_pillow()
    if Image is None:
        logger.error("Pillow 未安装，请执行: pip install Pillow")
        return None

    if output_path is None:
        output_path = input_path.parent / "avatar.webp"

    try:
        img = Image.open(input_path)

        # RGBA/P → RGB
        if img.mode in ("RGBA", "LA", "P"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[-1] if "A" in img.mode else None)
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # 中心裁剪为正方形
        w, h = img.size
        if w != h:
            size = min(w, h)
            left = (w - size) // 2
            top = (h - size) // 2
            img = img.crop((left, top, left + size, top + size))

        # 缩放到目标尺寸
        if img.size[0] != AVATAR_SIZE:
            img = img.resize((AVATAR_SIZE, AVATAR_SIZE), Image.LANCZOS)

        # 保存为 webp
        img.save(output_path, "WEBP", quality=AVATAR_QUALITY, method=4)

        # 删除原图（如果不同文件）
        if input_path != output_path and input_path.suffix != ".webp":
            input_path.unlink(missing_ok=True)

        return output_path

    except Exception as e:
        logger.error("头像压缩失败: %s", e)
        return None
