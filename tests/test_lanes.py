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
