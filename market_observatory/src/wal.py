# src/wal.py
"""
WAL — Write-Ahead Log Buffer
──────────────────────────────
Crash-safe flush strategy:
  - Accumulate rows in memory
  - Flush to DuckDB every WAL_FLUSH_SECONDS OR WAL_FLUSH_ROWS
  - On crash: at most WAL_FLUSH_ROWS * WAL_FLUSH_SECONDS rows lost
    (default: 100 rows or 1 second — minimal loss)

This replaces the old "buffer 500 trades → flush" pattern.
"""

import time
import threading
from typing import Callable

import pandas as pd

from src.config import WAL_FLUSH_SECONDS, WAL_FLUSH_ROWS
from src.logger import get_logger

log = get_logger("wal")


class WALBuffer:
    """
    Generic WAL buffer for any table.

    Usage:
        wal = WALBuffer("trades", db.insert_trades)
        wal.append(row_dict)       # from websocket
        wal.start_flush_thread()   # background auto-flush
    """

    def __init__(self, name: str, flush_fn: Callable[[pd.DataFrame], int]):
        self.name      = name
        self.flush_fn  = flush_fn
        self._buf: list[dict] = []
        self._lock     = threading.Lock()
        self._last_flush = time.monotonic()
        self._thread: threading.Thread | None = None
        self._running  = False

    def append(self, row: dict) -> None:
        with self._lock:
            self._buf.append(row)
        # immediate flush if row threshold hit
        if len(self._buf) >= WAL_FLUSH_ROWS:
            self._flush()

    def _flush(self) -> None:
        with self._lock:
            if not self._buf:
                return
            batch = self._buf.copy()
            self._buf.clear()
            self._last_flush = time.monotonic()

        df = pd.DataFrame(batch)
        n  = self.flush_fn(df)
        if n:
            log.debug(f"[wal/{self.name}] flushed {n} rows")

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
                 f"(every {WAL_FLUSH_SECONDS}s or {WAL_FLUSH_ROWS} rows)")

    def stop(self) -> None:
        self._running = False
        self._flush()   # final flush on shutdown
        log.info(f"[wal/{self.name}] stopped, final flush done")
