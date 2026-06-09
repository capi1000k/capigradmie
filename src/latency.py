# src/latency.py
from __future__ import annotations

import threading
from collections import deque

import numpy as np

from src.logger import get_logger

_WARN_MS     = 500.0
_CRITICAL_MS = 1000.0

_EMPTY: dict = {
    "p50": 0.0, "p95": 0.0, "p99": 0.0,
    "avg": 0.0, "max": 0.0, "count": 0,
}


class LatencyMonitor:
    def __init__(self, name: str = "latency", maxlen: int = 5000) -> None:
        self._name = name
        self._buf: deque[float] = deque(maxlen=maxlen)
        self._lock = threading.RLock()
        self._log  = get_logger(f"latency.{name}")

    def update(self, latency_ms: float) -> None:
        if latency_ms >= _CRITICAL_MS:
            self._log.critical("[%s] CRITICAL spike: %.1f ms", self._name, latency_ms)
        elif latency_ms >= _WARN_MS:
            self._log.warning("[%s] spike: %.1f ms", self._name, latency_ms)
        with self._lock:
            self._buf.append(float(latency_ms))

    def stats(self) -> dict:
        with self._lock:
            if not self._buf:
                return dict(_EMPTY)
            arr = np.array(self._buf, dtype=np.float64)
        return {
            "p50":   float(np.percentile(arr, 50)),
            "p95":   float(np.percentile(arr, 95)),
            "p99":   float(np.percentile(arr, 99)),
            "avg":   float(arr.mean()),
            "max":   float(arr.max()),
            "count": len(arr),
        }

    def reset(self) -> None:
        with self._lock:
            self._buf.clear()