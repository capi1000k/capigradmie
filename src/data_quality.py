# src/data_quality.py
"""
DATA QUALITY MONITOR — Market Observatory Stage 1.1C
─────────────────────────────────────────────────────
Metrics:
  - Stream freshness  (trade / orderbook / cvd)
  - Gap & resync counts
  - Message rate      (trades/s, ob_updates/s)
  - Integrity score   (0-100)
  - Alert engine
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

from src.logger import get_logger
from src.time_utils import utc_now, fmt_hms

log = get_logger("data_quality")

# ── Thresholds ────────────────────────────────────────────────────────────────
_STALE_S         = 3.0    # freshness alert threshold (seconds)
_LATENCY_WARN_MS = 500.0  # latency alert threshold (ms)
_RATE_WINDOW_S   = 10.0   # sliding window for msg rate calculation
_SPREAD_MULT     = 5.0    # spread anomaly: current > median * N

# ── Score weights ─────────────────────────────────────────────────────────────
_W_TRADE_STALE  = 30
_W_OB_STALE     = 30
_W_LAT_HIGH     = 20
_W_GAP          = 10
_W_SPREAD       = 10


class DataQualityMonitor:
    """
    Thread-safe data quality monitor.

    Usage
    ─────
    dq = DataQualityMonitor()

    # in stream_trades():
    dq.record_trade()

    # in orderbook_sync.run():
    dq.record_ob_update()
    dq.record_gap()      # when gap detected
    dq.record_resync()   # when _fetch_snapshot called after gap

    # in CVDTracker.update():
    dq.record_cvd()

    # dash callback reads:
    shared_state["dq"]
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        # ── Freshness ─────────────────────────────────────────────────────────
        self._last_trade_ts: Optional[float] = None   # time.monotonic()
        self._last_ob_ts:    Optional[float] = None
        self._last_cvd_ts:   Optional[float] = None

        # ── Gap / Resync ──────────────────────────────────────────────────────
        self._gap_count:    int           = 0
        self._resync_count: int           = 0
        self._last_gap_time: Optional[str] = None   # HH:MM:SS UTC string

        # ── Message rate (sliding window) ─────────────────────────────────────
        # Store monotonic arrival times for last _RATE_WINDOW_S seconds
        self._trade_times: deque[float] = deque()
        self._ob_times:    deque[float] = deque()

        # ── Spread history (anomaly detection) ───────────────────────────────
        self._spread_buf: deque[float] = deque(maxlen=200)
        self._last_alert_time: float = 0.0

    # ── Record events ─────────────────────────────────────────────────────────

    def record_trade(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._last_trade_ts = now
            self._trade_times.append(now)

    def record_ob_update(self, spread: float = 0.0) -> None:
        now = time.monotonic()
        with self._lock:
            self._last_ob_ts = now
            self._ob_times.append(now)
            if spread > 0:
                self._spread_buf.append(spread)

    def record_cvd(self) -> None:
        with self._lock:
            self._last_cvd_ts = time.monotonic()

    def record_gap(self) -> None:
        with self._lock:
            self._gap_count += 1
            self._last_gap_time = fmt_hms(utc_now())
        log.warning("[dq] gap detected — total gaps: %d", self._gap_count)

    def record_resync(self) -> None:
        with self._lock:
            self._resync_count += 1
        log.info("[dq] resync #%d", self._resync_count)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _freshness(self, last_ts: Optional[float]) -> float:
        """Seconds since last event. Returns 9999 if never seen."""
        if last_ts is None:
            return 9999.0
        return time.monotonic() - last_ts

    def _rate(self, times: deque) -> float:
        """Events per second over sliding window. Prunes old entries."""
        now = time.monotonic()
        cutoff = now - _RATE_WINDOW_S
        while times and times[0] < cutoff:
            times.popleft()
        return len(times) / _RATE_WINDOW_S

    def _spread_anomaly(self, current_spread: float) -> bool:
        """True if current spread > median * _SPREAD_MULT."""
        if len(self._spread_buf) < 20 or current_spread <= 0:
            return False
        import numpy as np
        median = float(np.median(self._spread_buf))
        return median > 0 and current_spread > median * _SPREAD_MULT

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self, shared_state: dict) -> dict:
        """
        Compute all metrics, write to shared_state["dq"], return dict.
        Never raises.
        """
        try:
            with self._lock:
                trade_fresh  = self._freshness(self._last_trade_ts)
                ob_fresh     = self._freshness(self._last_ob_ts)
                cvd_fresh    = self._freshness(self._last_cvd_ts)
                gap_count    = self._gap_count
                resync_count = self._resync_count
                last_gap     = self._last_gap_time or "—"
                trades_per_s = self._rate(self._trade_times)
                ob_per_s     = self._rate(self._ob_times)
                spread_list  = list(self._spread_buf)

            # current spread from shared_state (latest OB snapshot)
            ob_live = shared_state.get("orderbook_live") or {}
            current_spread = float(ob_live.get("spread", 0) or 0)
            spread_anom = self._spread_anomaly(current_spread)

            # latency from shared_state
            lat_stats = shared_state.get("latency_stats") or {}
            lat_avg   = float(lat_stats.get("avg", 0) or 0)

            # ── Integrity score ───────────────────────────────────────────────
            score = 100.0
            alerts: list[str] = []

            if trade_fresh > _STALE_S:
                score -= _W_TRADE_STALE
                alerts.append(f"⚠ Trade stale {trade_fresh:.1f}s")

            if ob_fresh > _STALE_S:
                score -= _W_OB_STALE
                alerts.append(f"⚠ OB stale {ob_fresh:.1f}s")

            if lat_avg > _LATENCY_WARN_MS:
                score -= _W_LAT_HIGH
                alerts.append(f"⚠ Latency {lat_avg:.0f}ms")

            if gap_count > 0:
                score -= _W_GAP
                alerts.append(f"⚠ Gaps: {gap_count}")

            if spread_anom:
                score -= _W_SPREAD
                alerts.append(f"⚠ Spread anomaly {current_spread:.3f}")

            score = max(0.0, score)

            # log alerts (max 1 per 60s)
            now_mono = time.monotonic()
            if alerts and now_mono - self._last_alert_time > 60.0:
                for alert in alerts:
                    log.warning("[dq] %s", alert)
                self._last_alert_time = now_mono

            dq = {
                "trade_fresh_s": round(trade_fresh, 1),
                "ob_fresh_s":    round(ob_fresh, 1),
                "cvd_fresh_s":   round(cvd_fresh, 1),
                "gap_count":     gap_count,
                "resync_count":  resync_count,
                "last_gap_time": last_gap,
                "trades_per_s":  round(trades_per_s, 1),
                "ob_per_s":      round(ob_per_s, 1),
                "score":         round(score, 1),
                "alerts":        alerts,
            }

            shared_state["dq"] = dq
            return dq

        except Exception:
            import traceback
            log.error("[dq] snapshot error:\n%s", traceback.format_exc())
            return {}