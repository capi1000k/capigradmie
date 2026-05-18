# src/orderbook_sync.py
"""
PROFESSIONAL ORDERBOOK SYNC
─────────────────────────────
Implements Binance official orderbook management algorithm:

  Step 1: Buffer incoming websocket events
  Step 2: GET /api/v3/depth snapshot (lastUpdateId)
  Step 3: Discard events where u < lastUpdateId
  Step 4: Apply events where U <= lastUpdateId+1 <= u
  Step 5: If gap detected → rebuild from snapshot

This prevents corrupted orderbook from:
  - packet drops
  - reconnects
  - latency spikes

Reference: https://binance-docs.github.io/apidocs/spot/en/#how-to-manage-a-local-order-book-correctly
"""

import asyncio
import json
import time
from collections import deque

import aiohttp
import websockets
import pandas as pd

from src.config import (
    BINANCE_WS_BASE, DEPTH_SNAPSHOT_URL,
    SYMBOL, SYMBOL_LOWER,
    ORDERBOOK_LEVELS,
    WS_RECONNECT_DELAY, WS_MAX_RETRIES,
)
from src.wal import WALBuffer
from src.db import insert_orderbook
from src.logger import get_logger

log = get_logger("orderbook")


class OrderBookSyncer:
    """
    Maintains a live, synchronized orderbook for one symbol.
    Implements the official Binance sync algorithm.
    """

    def __init__(self, shared_state: dict, wal: WALBuffer):
        self.shared_state    = shared_state
        self.wal             = wal

        # local book
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._last_update_id: int      = 0
        self._synced: bool             = False

        # event buffer (used during initial sync)
        self._event_buffer: deque      = deque(maxlen=1000)

        # OFI tracking (previous best bid/ask sizes)
        self._prev_bid_size: float     = 0.0
        self._prev_ask_size: float     = 0.0

    # ── Snapshot ──────────────────────────────────────────────────────────────

    async def _fetch_snapshot(self, session: aiohttp.ClientSession) -> None:
        """Fetch REST snapshot and initialize local book."""
        log.info("[orderbook] fetching depth snapshot…")
        params = {"symbol": SYMBOL, "limit": 1000}
        async with session.get(
            DEPTH_SNAPSHOT_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as r:
            r.raise_for_status()
            data = await r.json()

        self._last_update_id = data["lastUpdateId"]
        self._bids = {float(p): float(q) for p, q in data["bids"]}
        self._asks = {float(p): float(q) for p, q in data["asks"]}
        self._synced = False
        log.info(f"[orderbook] snapshot ok, lastUpdateId={self._last_update_id}")

    # ── Apply update ──────────────────────────────────────────────────────────

    def _apply(self, side: dict, updates: list) -> None:
        for p_str, q_str in updates:
            p, q = float(p_str), float(q_str)
            if q == 0.0:
                side.pop(p, None)
            else:
                side[p] = q

    def _process_event(self, event: dict) -> bool:
        """
        Apply one event to local book.
        Returns True if applied, False if gap detected (needs resync).
        """
        U = event["U"]   # first update id in event
        u = event["u"]   # final update id in event

        # discard stale events
        if u <= self._last_update_id:
            return True

        # check sequence continuity
        if not self._synced:
            if U <= self._last_update_id + 1 <= u:
                self._synced = True
            else:
                return True   # still buffering, not yet at sync point

        if U != self._last_update_id + 1:
            log.warning(
                f"[orderbook] GAP detected! "
                f"expected U={self._last_update_id+1}, got U={U} — resyncing"
            )
            return False  # signal caller to resync

        self._apply(self._bids, event["b"])
        self._apply(self._asks, event["a"])
        self._last_update_id = u
        return True

    # ── Compute snapshot ──────────────────────────────────────────────────────

    def _compute_microstructure(self) -> dict | None:
        if not self._bids or not self._asks:
            return None

        top_bids = sorted(self._bids.items(), reverse=True)[:ORDERBOOK_LEVELS]
        top_asks = sorted(self._asks.items())[:ORDERBOOK_LEVELS]

        best_bid, best_bid_size = top_bids[0]
        best_ask, best_ask_size = top_asks[0]

        bid_vol = sum(q for _, q in top_bids)
        ask_vol = sum(q for _, q in top_asks)
        total   = bid_vol + ask_vol

        imbalance  = (bid_vol - ask_vol) / total if total else 0.0
        spread     = best_ask - best_bid
        mid_price  = (best_bid + best_ask) / 2
        spread_bps = (spread / mid_price) * 10_000

        # ── Microprice (weighted mid) ─────────────────────────────────────────
        # Microprice = (ask_price * bid_size + bid_price * ask_size) / (bid_size + ask_size)
        denom = best_bid_size + best_ask_size
        microprice = (
            (best_ask * best_bid_size + best_bid * best_ask_size) / denom
            if denom > 0 else mid_price
        )

        # ── OFI (Order Flow Imbalance) ────────────────────────────────────────
        # OFI_t = ΔBidSize_t - ΔAskSize_t (change at best level)
        ofi = (best_bid_size - self._prev_bid_size) - (best_ask_size - self._prev_ask_size)
        self._prev_bid_size = best_bid_size
        self._prev_ask_size = best_ask_size

        return {
            "timestamp":  pd.Timestamp.utcnow(),
            "best_bid":   best_bid,
            "best_ask":   best_ask,
            "spread":     spread,
            "mid_price":  mid_price,
            "microprice": microprice,
            "bid_vol":    bid_vol,
            "ask_vol":    ask_vol,
            "imbalance":  imbalance,
            "ofi":        ofi,
            "spread_bps": spread_bps,
        }

    # ── Main stream loop ──────────────────────────────────────────────────────

    async def run(self) -> None:
        url     = f"{BINANCE_WS_BASE}/{SYMBOL_LOWER}@depth@100ms"
        retries = 0

        while retries < WS_MAX_RETRIES:
            try:
                async with aiohttp.ClientSession() as http_session:
                    # Step 1: open WebSocket first (buffer events)
                    log.info(f"[orderbook] connecting → {url}")
                    async with websockets.connect(
                        url, ping_interval=20, ping_timeout=10, close_timeout=5
                    ) as ws:
                        retries = 0
                        self._event_buffer.clear()
                        self._synced = False

                        # Step 2: fetch REST snapshot
                        await self._fetch_snapshot(http_session)

                        log.info("[orderbook] ✓ synced, processing events")

                        async for raw in ws:
                            event = json.loads(raw)

                            # buffer during initial sync
                            if not self._synced:
                                self._event_buffer.append(event)
                                # drain buffer once synced
                                while self._event_buffer:
                                    e = self._event_buffer.popleft()
                                    ok = self._process_event(e)
                                    if not ok:
                                        await self._fetch_snapshot(http_session)
                                        break
                                continue

                            ok = self._process_event(event)
                            if not ok:
                                # gap — rebuild
                                await self._fetch_snapshot(http_session)
                                continue

                            snap = self._compute_microstructure()
                            if snap:
                                self.shared_state["orderbook_live"] = snap
                                self.wal.append(snap)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                retries += 1
                delay = min(WS_RECONNECT_DELAY * retries, 60)
                log.warning(
                    f"[orderbook] disconnected ({e}) — retry #{retries} in {delay}s"
                )
                await asyncio.sleep(delay)

        log.critical("[orderbook] max retries exceeded")
