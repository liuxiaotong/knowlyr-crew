"""运行时指标收集器 — 记录 LLM 调用统计."""

from __future__ import annotations

import threading
import time


class MetricsCollector:
    """线程安全的运行时指标收集器.

    记录 LLM 调用次数、token 用量、错误数等统计信息。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._total_calls = 0
        self._success_calls = 0
        self._failed_calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._by_employee: dict[str, dict[str, int]] = {}
        self._by_provider: dict[str, dict[str, int]] = {}

    def record_call(
        self,
        *,
        employee: str = "",
        provider: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        success: bool = True,
    ) -> None:
        """记录一次 LLM 调用."""
        with self._lock:
            self._total_calls += 1
            if success:
                self._success_calls += 1
            else:
                self._failed_calls += 1
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens

            if employee:
                if employee not in self._by_employee:
                    self._by_employee[employee] = {"calls": 0, "tokens": 0}
                self._by_employee[employee]["calls"] += 1
                self._by_employee[employee]["tokens"] += input_tokens + output_tokens

            if provider:
                if provider not in self._by_provider:
                    self._by_provider[provider] = {"calls": 0, "success": 0, "failed": 0}
                self._by_provider[provider]["calls"] += 1
                if success:
                    self._by_provider[provider]["success"] += 1
                else:
                    self._by_provider[provider]["failed"] += 1

    def snapshot(self) -> dict:
        """返回当前指标快照."""
        with self._lock:
            return {
                "uptime_seconds": round(time.monotonic() - self._start_time),
                "calls": {
                    "total": self._total_calls,
                    "success": self._success_calls,
                    "failed": self._failed_calls,
                },
                "tokens": {
                    "input": self._input_tokens,
                    "output": self._output_tokens,
                },
                "by_employee": dict(self._by_employee),
                "by_provider": dict(self._by_provider),
            }

    def reset(self) -> None:
        """重置所有指标."""
        with self._lock:
            self._start_time = time.monotonic()
            self._total_calls = 0
            self._success_calls = 0
            self._failed_calls = 0
            self._input_tokens = 0
            self._output_tokens = 0
            self._by_employee.clear()
            self._by_provider.clear()


# 全局单例
_collector = MetricsCollector()


def get_collector() -> MetricsCollector:
    """获取全局指标收集器."""
    return _collector
