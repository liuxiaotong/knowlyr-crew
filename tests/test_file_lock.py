"""file_lock 并发安全测试."""

import threading
from pathlib import Path

from crew.paths import file_lock


def test_file_lock_basic(tmp_path: Path):
    """基本锁定/释放：lock 文件被创建."""
    target = tmp_path / "data.jsonl"
    target.write_text("line1\n")

    with file_lock(target):
        # 在锁内可以正常读写
        content = target.read_text()
        target.write_text(content + "line2\n")

    assert "line2" in target.read_text()
    assert (tmp_path / "data.jsonl.lock").exists()


def test_file_lock_protects_read_modify_write(tmp_path: Path):
    """模拟并发 read-modify-write，验证文件锁防止数据丢失."""
    target = tmp_path / "counter.txt"
    target.write_text("0\n")
    iterations = 50
    errors: list[str] = []

    def increment():
        for _ in range(iterations):
            try:
                with file_lock(target):
                    val = int(target.read_text().strip())
                    target.write_text(f"{val + 1}\n")
            except Exception as e:
                errors.append(str(e))

    threads = [threading.Thread(target=increment) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors: {errors}"
    final = int(target.read_text().strip())
    assert final == iterations * 4


def test_file_lock_on_nonexistent_parent(tmp_path: Path):
    """目标文件父目录不存在时自动创建."""
    target = tmp_path / "sub" / "dir" / "data.jsonl"
    with file_lock(target):
        pass  # lock 文件的父目录应被创建
    assert (tmp_path / "sub" / "dir" / "data.jsonl.lock").exists()
