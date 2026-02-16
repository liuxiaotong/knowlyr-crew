"""自动版本管理 — 内容哈希 + patch 自动递增."""

import hashlib
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def compute_content_hash(dir_path: Path) -> str:
    """计算目录内所有内容文件的组合 SHA256.

    扫描 prompt.md + workflows/*.md + adaptors/*.md，按固定顺序拼接后取哈希。

    Args:
        dir_path: 员工目录路径

    Returns:
        SHA256 前 8 位十六进制字符串
    """
    hasher = hashlib.sha256()

    # 按固定顺序收集内容文件
    content_files: list[Path] = []

    prompt_path = dir_path / "prompt.md"
    if prompt_path.exists():
        content_files.append(prompt_path)

    for subdir in ("workflows", "adaptors"):
        sub_path = dir_path / subdir
        if sub_path.is_dir():
            content_files.extend(sorted(sub_path.glob("*.md")))

    for f in content_files:
        hasher.update(f.read_bytes())

    return hasher.hexdigest()[:8]


def _bump_patch(version: str) -> str:
    """将版本号的 patch 部分 +1.

    支持 "3.0" → "3.0.1" 和 "3.0.1" → "3.0.2" 格式。

    Args:
        version: 当前版本号

    Returns:
        递增后的版本号
    """
    parts = version.split(".")
    if len(parts) < 3:
        parts.append("1")
    else:
        parts[2] = str(int(parts[2]) + 1)
    return ".".join(parts)


def check_and_bump(dir_path: Path) -> tuple[str, bool]:
    """检查内容是否变更，若变更则 bump patch 并回写 employee.yaml.

    Args:
        dir_path: 员工目录路径

    Returns:
        (version, bumped) — 当前版本号和是否发生了 bump
    """
    config_path = dir_path / "employee.yaml"
    if not config_path.exists():
        return ("1.0", False)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        return ("1.0", False)
    version = str(config.get("version", "1.0"))
    stored_hash = config.get("_content_hash", "")

    current_hash = compute_content_hash(dir_path)

    if current_hash == stored_hash:
        return (version, False)

    # 内容变更 → bump patch
    new_version = _bump_patch(version)
    config["version"] = new_version
    config["_content_hash"] = current_hash

    try:
        import os
        import tempfile
        content = yaml.dump(config, allow_unicode=True, sort_keys=False, default_flow_style=False)
        fd, tmp = tempfile.mkstemp(dir=config_path.parent, suffix=".tmp")
        fd_closed = False
        try:
            os.write(fd, content.encode("utf-8"))
            os.close(fd)
            fd_closed = True
            os.replace(tmp, config_path)
        except Exception:
            if not fd_closed:
                os.close(fd)
            Path(tmp).unlink(missing_ok=True)
            raise
        logger.info("版本 bump: %s → %s (hash: %s)", version, new_version, current_hash)
    except OSError as e:
        logger.warning("无法回写 %s: %s", config_path, e)
        return (version, False)

    return (new_version, True)
