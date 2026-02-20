"""测试 LaneLock."""

import os
import tempfile
from pathlib import Path

from crew.lanes import LaneLock, lane_lock


class TestLanes:
    """Lane 调度基础测试."""

    def test_lane_lock_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with lane_lock("employee:test", enabled=True):
                    lock_file = Path(".crew/lanes/employee_test.lock")
                    assert lock_file.exists()
            finally:
                os.chdir(old_cwd)

    def test_lane_lock_manual(self):
        lane = LaneLock("pipeline:demo", root=Path(tempfile.mkdtemp()))
        lane.acquire()
        assert lane.path.exists()
        lane.release()

    def test_acquire_closes_fh_on_error(self):
        """acquire() 在写入失败时关闭文件句柄."""
        from unittest.mock import MagicMock, patch

        root = Path(tempfile.mkdtemp())
        lane = LaneLock("error:test", root=root)
        lane.path.parent.mkdir(parents=True, exist_ok=True)

        mock_fh = MagicMock()
        mock_fh.write.side_effect = OSError("disk full")
        mock_fh.fileno.return_value = 3

        with patch("builtins.open", return_value=mock_fh), patch("crew.lanes.fcntl", None):
            try:
                lane.acquire()
            except OSError:
                pass

        mock_fh.close.assert_called_once()
