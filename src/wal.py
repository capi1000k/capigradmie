# src/wal.py
"""
WAL — Write-Ahead Log Buffer  (v2 — non-blocking flush)
──────────────────────────────────────────────────────
v1 muammosi:
  append() threshold to'lganda _flush() ni event loop thread'ining
  O'ZIDA sinxron chaqirar edi → DuckDB I/O event loop'ni bloklardi.
  Bir nechta jadval (trades/orderbook/cvd/liquidations/mark_price)
  bir vaqtda flush qilishga uringanda DuckDB lock kutish natijasida
  BUTUN asyncio event loop bir necha yuz ms ga muzlab qolardi —
  socket'da xabarlar to'planib, keyin birdan "burst" bo'lib kelardi
  (soxta latency spike, real tarmoq kechikishi emas).

v2 yechimi:
  _flush() endi faqat xotiradagi buferni nusxalab tozalaydi (tez,
  I/O yo'q) va DataFrame'ni umumiy yagona writer-thread navbatiga
  (src.db.enqueue_write) topshiradi. Haqiqiy DuckDB yozuvi faqat
  o'sha bitta background thread'da sodir bo'ladi — hech qachon
  event loop yoki boshqa thread bloklanmaydi.
"""

import time
import threading
from typing import Callable

import pandas as pd

from src.config import WAL_FLUSH_SECONDS, WAL_FLUSH_ROWS
from src.logger import get_logger
from src import db

log = get_logger("wal")


class WALBuffer:
    """
    Generic WAL buffer for any table.

    Usage:
        wal = WALBuffer("trades", db.insert_trades)
        wal.append(row_dict)       # from websocket — never blocks on DB I/O
        wal.start_flush_thread()   # background timer-based flush
    """

    def __init__(self, name: str, flush_fn: Callable[[pd.DataFrame], int]):
        self.name         = name
        self.flush_fn     = flush_fn
        self._buf: list[dict] = []
        self._lock        = threading.Lock()
        self._last_flush  = time.monotonic()
        self._thread: threading.Thread | None = None
        self._running     = False

    def append(self, row: dict) -> None:
        """Called from websocket handlers (event loop thread). Must stay
        non-blocking — only in-memory list append + occasional buffer
        copy/clear, never direct DB I/O."""
        with self._lock:
            self._buf.append(row)
            hit_threshold = len(self._buf) >= WAL_FLUSH_ROWS
        if hit_threshold:
            self._flush()

    def _flush(self) -> None:
        """Copy + clear the in-memory buffer (fast, lock-only) and hand the
        DataFrame off to the single shared writer thread. No DuckDB I/O
        happens here, so this never blocks the caller (event loop safe)."""
        with self._lock:
            if not self._buf:
                return
            batch = self._buf.copy()
            self._buf.clear()
            self._last_flush = time.monotonic()

        df = pd.DataFrame(batch)
        db.enqueue_write(self.name, self.flush_fn, df)

    def _flush_loop(self) -> None:
        while self._running:
            time.sleep(WAL_FLUSH_SECONDS)
            elapsed = time.monotonic() - self._last_flush
            if elapsed >= WAL_FLUSH_SECONDS and self._buf:
                self._flush()

    def start_flush_thread(self) -> None:
        self._running = True
        self._thread  = threading.Thread(
            target=self._flush_loop,
            name=f"wal-{self.name}",
            daemon=True,
        )
        self._thread.start()
        log.info(f"[wal/{self.name}] flush thread started "
                 f"(every {WAL_FLUSH_SECONDS}s or {WAL_FLUSH_ROWS} rows, "
                 f"async writer queue)")

    def stop(self) -> None:
        self._running = False
        self._flush()                       # final buffer → queue
        db.wait_for_writer_drain(timeout=5)  # queue to'liq yozilishini kut
        log.info(f"[wal/{self.name}] stopped, final flush queued & drained")