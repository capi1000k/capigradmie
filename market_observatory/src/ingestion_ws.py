# src/ingestion_ws.py
"""
DATA INGESTION — WEBSOCKET STREAMS v2
"""

import asyncio
import json
from collections import deque

import pandas as pd

from src.config import (
    BINANCE_WS_BASE, SYMBOL_LOWER,
    WS_RECONNECT_DELAY, WS_MAX_RETRIES,
)
from src.wal import WALBuffer
from src.db import insert_trades, insert_cvd, insert_orderbook
from src.orderbook_sync import OrderBookSyncer
from src.logger import get_logger

log = get_logger("websocket")


class CVDTracker:
    """
    CVD = CVD_prev + (BuyVol - SellVol)
    Rising CVD + flat price  → absorption
    Falling CVD + flat price → distribution
    """
    def __init__(self):
        self._cvd = 0.0
        self._wal = WALBuffer("cvd", insert_cvd)
        self._wal.start_flush_thread()

    def update(self, quantity: float, is_buyer_mm: bool) -> float:
        delta = -quantity if is_buyer_mm else quantity
        self._cvd += delta
        self._wal.append({
            "timestamp": pd.Timestamp.utcnow(),
            "cvd":       self._cvd,
            "delta":     delta,
        })
        return self._cvd


async def stream_trades(shared_state: dict) -> None:
    import websockets
    url     = f"{BINANCE_WS_BASE}/{SYMBOL_LOWER}@aggTrade"
    wal     = WALBuffer("trades", insert_trades)
    cvd     = CVDTracker()
    retries = 0
    wal.start_flush_thread()

    while retries < WS_MAX_RETRIES:
        try:
            log.info(f"[trades] connecting → {url}")
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                retries = 0
                log.info("[trades] ✓ connected")
                async for raw in ws:
                    msg  = json.loads(raw)
                    qty  = float(msg["q"])
                    is_bm = bool(msg["m"])
                    row  = {
                        "timestamp":   pd.Timestamp(msg["T"], unit="ms", tz="UTC"),
                        "price":       float(msg["p"]),
                        "quantity":    qty,
                        "is_buyer_mm": is_bm,
                        "trade_id":    int(msg["a"]),
                    }
                    wal.append(row)
                    live_cvd = cvd.update(qty, is_bm)
                    shared_state.setdefault("trades_live", deque(maxlen=500)).append(row)
                    shared_state["cvd_live"] = live_cvd

        except asyncio.CancelledError:
            wal.stop(); raise
        except Exception as e:
            retries += 1
            await asyncio.sleep(min(WS_RECONNECT_DELAY * retries, 60))
            log.warning(f"[trades] retry #{retries}: {e}")


async def stream_orderbook(shared_state: dict) -> None:
    wal    = WALBuffer("orderbook", insert_orderbook)
    syncer = OrderBookSyncer(shared_state, wal)
    wal.start_flush_thread()
    try:
        await syncer.run()
    except asyncio.CancelledError:
        wal.stop(); raise
