"""测试断路器."""

import time

from crew.id_client import _CircuitBreaker


class TestCircuitBreaker:
    """_CircuitBreaker 单元测试."""

    def test_closed_initially(self):
        cb = _CircuitBreaker()
        assert not cb.is_open()

    def test_stays_closed_below_threshold(self):
        cb = _CircuitBreaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open()

    def test_opens_at_threshold(self):
        cb = _CircuitBreaker(threshold=2, cooldown=10)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open()

    def test_closes_after_cooldown(self):
        cb = _CircuitBreaker(threshold=1, cooldown=0.05)
        cb.record_failure()
        assert cb.is_open()
        time.sleep(0.06)
        assert not cb.is_open()

    def test_success_resets_failures(self):
        cb = _CircuitBreaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open()  # Reset by success, only 2 after reset

    def test_open_blocks_then_halfopen(self):
        cb = _CircuitBreaker(threshold=1, cooldown=0.05)
        cb.record_failure()
        assert cb.is_open()
        time.sleep(0.06)
        # Half-open: is_open returns False (allows one try)
        assert not cb.is_open()
        # Failures counter was reset by is_open
        assert cb._failures == 0
