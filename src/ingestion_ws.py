# src/ingestion_ws.py
"""
DATA INGESTION — WEBSOCKET STREAMS v4
──────────────────────────────────────
v3 dan farqi:
  0. Trade stream @aggTrade → @trade ga o'zgartirildi — endi HAR BIR
     alohida execution (chin tick-by-tick) keladi, agregatlangan emas.
     trade_id maydoni ham "a" (aggTrade id) o'rniga "t" (trade id).

v2 dan farqi:
  1. CVD restart-persistence — DB'dagi oxirgi CVD'dan davom etadi
     (avval har restart CVD'ni 0'ga tushirib signalni buzardi).
  2. Liquidation stream qo'shildi (!forceOrder@arr, Futures WS) —
     narxni eng keskin harakatga keltiruvchi hodisalardan biri,
     endi to'g'ridan-to'g'ri kuzatiladi (taxmin emas).
  3. Mark Price + Funding Rate stream qo'shildi (markPrice@1s) —
     pozitsiya bosimi (funding) va mark/index spread.
  4. Open Interest — REST poll (WS'da yo'q, kam o'zgaradi).
  5. Xato handling aniqlashtirildi: parsing/struktura xatolari
     (KeyError, JSONDecodeError) endi network xatolaridan
     ajratilib, darhol log qilinadi — "sokin yutilib qolish" yo'q.
"""

import asyncio
import json
from collections import deque
import pandas as pd  # shared_state va WALBuffer uchun hali kerak

from src.config import (
    BINANCE_WS_BASE, FUTURES_WS_BASE, FUTURES_REST_BASE,
    SYMBOL, SYMBOL_LOWER,
    OPEN_INTEREST_ENDPOINT, OPEN_INTEREST_POLL,
    WS_RECONNECT_DELAY, WS_MAX_RETRIES,
)
from src.time_utils import utc_now, to_timestamp
from src.latency import LatencyMonitor
from src.wal import WALBuffer
from src.db import (
    insert_trades, insert_cvd, insert_orderbook,
    insert_liquidations, insert_open_interest, insert_mark_price,
    load_last_cvd,
)
from src import db
from src.orderbook_sync import OrderBookSyncer
from src.logger import get_logger

log = get_logger("websocket")

# Xatolarni ikki toifaga ajratish uchun — parsing xatosi ≠ network xatosi.
# Parsing xatosi kodda/sxemada muammo borligini bildiradi va DARHOL
# ko'rinishi kerak (warning emas, error + traceback), reconnect qilib
# "yashirib" bo'lmaydi.
_PARSE_ERRORS = (KeyError, ValueError, TypeError, json.JSONDecodeError)


class CVDTracker:
    """
    CVD = CVD_prev + (BuyVol - SellVol)
    Rising CVD + flat price  → absorption
    Falling CVD + flat price → distribution

    v3: restart-persistence — __init__ da DB'dan oxirgi CVD o'qiladi.
    Buning bo'lmasligi har restart'da CVD'ni 0'ga tashlab, ayni shu
    tickdagi slope/z-score signallarini bir necha daqiqaga buzardi.
    """
    def __init__(self):
        self._cvd = load_last_cvd()
        if self._cvd != 0.0:
            log.info(f"[cvd] restored from DB: {self._cvd:.4f}")
        self._wal = WALBuffer("cvd", insert_cvd)
        self._wal.start_flush_thread()

    def update(self, quantity: float, is_buyer_mm: bool) -> float:
        delta = -quantity if is_buyer_mm else quantity
        self._cvd += delta
        self._wal.append({
            "timestamp": utc_now(),
            "cvd":       self._cvd,
            "delta":     delta,
        })
        return self._cvd


async def stream_trades(shared_state: dict) -> None:
    import websockets
    url     = f"{BINANCE_WS_BASE}/{SYMBOL_LOWER}@trade"
    wal     = WALBuffer("trades", insert_trades)
    cvd     = CVDTracker()
    lat_mon = LatencyMonitor("trades")
    from src.data_quality import DataQualityMonitor
    dq = shared_state.setdefault("_dq_monitor", DataQualityMonitor())
    retries = 0
    wal.start_flush_thread()

    while retries < WS_MAX_RETRIES:
        try:
            log.info(f"[trades] connecting → {url}")
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                retries = 0
                log.info("[trades] ✓ connected")
                async for raw in ws:
                    receive_ts = utc_now()
                    try:
                        msg  = json.loads(raw)
                        qty  = float(msg["q"])
                        is_bm = bool(msg["m"])
                        exchange_ts = to_timestamp(int(msg["T"]))
                        trade_id = int(msg["t"])
                        price = float(msg["p"])
                    except _PARSE_ERRORS as e:
                        log.error(f"[trades] malformed message, skipping: {e} | raw={raw[:200]}")
                        continue

                    process_ts  = utc_now()
                    latency_ms  = (receive_ts - exchange_ts).total_seconds() * 1000
                    row  = {
                        "timestamp":   exchange_ts,
                        "price":       price,
                        "quantity":    qty,
                        "is_buyer_mm": is_bm,
                        "trade_id":    trade_id,
                        "exchange_ts": exchange_ts,
                        "receive_ts":  receive_ts,
                        "process_ts":  process_ts,
                        "latency_ms":  latency_ms,
                    }
                    wal.append(row)
                    live_cvd = cvd.update(qty, is_bm)
                    shared_state["trades_live"].append(row)
                    shared_state["cvd_live"] = live_cvd
                    lat_mon.update(latency_ms)
                    shared_state["latency_stats"] = lat_mon.stats()
                    shared_state["last_trade_latency_ms"] = latency_ms
                    dq.record_trade()
                    dq.snapshot(shared_state)

        except asyncio.CancelledError:
            wal.stop(); raise
        except Exception as e:
            retries += 1
            delay = min(WS_RECONNECT_DELAY * retries, 60)
            log.warning(f"[trades] disconnected ({e}) — retry #{retries} in {delay}s")
            await asyncio.sleep(delay)


async def stream_orderbook(shared_state: dict) -> None:
    wal    = WALBuffer("orderbook", insert_orderbook)
    syncer = OrderBookSyncer(shared_state, wal)
    wal.start_flush_thread()
    try:
        await syncer.run()
    except asyncio.CancelledError:
        wal.stop(); raise


# ─────────────────────────────────────────────────────────────────────────────
# LIQUIDATIONS — Futures forceOrder stream
# ─────────────────────────────────────────────────────────────────────────────
# Narxni eng keskin harakatga keltiruvchi hodisalardan biri: majburiy
# yopilgan pozitsiyalar. Bu stream barcha symbol'lar bo'yicha keladi
# (!forceOrder@arr — array stream), shuning uchun SYMBOL bo'yicha filtr
# qo'llaniladi.
#
# side="SELL" → LONG likvidatsiya qilindi (majburiy sotuv → narx pastga bosim)
# side="BUY"  → SHORT likvidatsiya qilindi (majburiy sotib olish → narx yuqoriga bosim)

async def stream_liquidations(shared_state: dict) -> None:
    import websockets
    url     = f"{FUTURES_WS_BASE}/!forceOrder@arr"
    wal     = WALBuffer("liquidations", insert_liquidations)
    lat_mon = LatencyMonitor("liquidations")
    wal.start_flush_thread()
    retries = 0

    shared_state.setdefault("liquidations_live", deque(maxlen=200))

    while retries < WS_MAX_RETRIES:
        try:
            log.info(f"[liquidations] connecting → {url}")
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                retries = 0
                log.info("[liquidations] ✓ connected")
                async for raw in ws:
                    receive_ts = utc_now()
                    try:
                        msg = json.loads(raw)
                        o   = msg["o"]              # order object
                        if o["s"] != SYMBOL:
                            continue                # boshqa symbol — skip
                        side        = o["S"]        # "BUY" | "SELL"
                        price       = float(o["ap"])  # average fill price
                        quantity    = float(o["q"])
                        exchange_ts = to_timestamp(int(o["T"]))
                    except _PARSE_ERRORS as e:
                        log.error(f"[liquidations] malformed message, skipping: {e} | raw={raw[:200]}")
                        continue

                    latency_ms = (receive_ts - exchange_ts).total_seconds() * 1000
                    row = {
                        "timestamp":   exchange_ts,
                        "side":        side,
                        "price":       price,
                        "quantity":    quantity,
                        "quote_qty":   price * quantity,
                        "exchange_ts": exchange_ts,
                        "receive_ts":  receive_ts,
                        "latency_ms":  latency_ms,
                    }
                    wal.append(row)
                    shared_state["liquidations_live"].append(row)
                    lat_mon.update(latency_ms)
                    log.info(
                        f"[liquidations] {side} {quantity:.4f} {SYMBOL} @ {price:.2f} "
                        f"(${row['quote_qty']:,.0f})"
                    )

        except asyncio.CancelledError:
            wal.stop(); raise
        except Exception as e:
            retries += 1
            delay = min(WS_RECONNECT_DELAY * retries, 60)
            log.warning(f"[liquidations] disconnected ({e}) — retry #{retries} in {delay}s")
            await asyncio.sleep(delay)


# ─────────────────────────────────────────────────────────────────────────────
# MARK PRICE + FUNDING RATE — Futures markPrice stream
# ─────────────────────────────────────────────────────────────────────────────
# Funding rate ekstremal qiymatlarda liquidation cascade ehtimolini
# oshiradi; mark-index spread esa futures/spot orasidagi bosim farqini
# ko'rsatadi.

async def stream_mark_price(shared_state: dict) -> None:
    import websockets
    url     = f"{FUTURES_WS_BASE}/{SYMBOL_LOWER}@markPrice@1s"
    wal     = WALBuffer("mark_price", insert_mark_price)
    lat_mon = LatencyMonitor("mark_price")
    wal.start_flush_thread()
    retries = 0

    while retries < WS_MAX_RETRIES:
        try:
            log.info(f"[mark_price] connecting → {url}")
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                retries = 0
                log.info("[mark_price] ✓ connected")
                async for raw in ws:
                    receive_ts = utc_now()
                    try:
                        msg = json.loads(raw)
                        mark_price      = float(msg["p"])
                        index_price     = float(msg["i"])
                        funding_rate    = float(msg["r"])
                        next_funding_ts = to_timestamp(int(msg["T"]))
                        exchange_ts     = to_timestamp(int(msg["E"]))
                    except _PARSE_ERRORS as e:
                        log.error(f"[mark_price] malformed message, skipping: {e} | raw={raw[:200]}")
                        continue

                    latency_ms = (receive_ts - exchange_ts).total_seconds() * 1000
                    row = {
                        "timestamp":       exchange_ts,
                        "mark_price":      mark_price,
                        "index_price":     index_price,
                        "funding_rate":    funding_rate,
                        "next_funding_ts": next_funding_ts,
                        "exchange_ts":     exchange_ts,
                        "receive_ts":      receive_ts,
                        "latency_ms":      latency_ms,
                    }
                    wal.append(row)
                    shared_state["mark_price_live"] = row
                    lat_mon.update(latency_ms)

        except asyncio.CancelledError:
            wal.stop(); raise
        except Exception as e:
            retries += 1
            delay = min(WS_RECONNECT_DELAY * retries, 60)
            log.warning(f"[mark_price] disconnected ({e}) — retry #{retries} in {delay}s")
            await asyncio.sleep(delay)


# ─────────────────────────────────────────────────────────────────────────────
# OPEN INTEREST — REST poll (WS'da mavjud emas, kam o'zgaradi)
# ─────────────────────────────────────────────────────────────────────────────

async def poll_open_interest(shared_state: dict) -> None:
    import aiohttp
    async with aiohttp.ClientSession() as session:
        while True:
            t0 = utc_now()
            try:
                async with session.get(
                    OPEN_INTEREST_ENDPOINT,
                    params={"symbol": SYMBOL},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    r.raise_for_status()
                    data = await r.json()

                oi          = float(data["openInterest"])
                mark        = (shared_state.get("mark_price_live") or {}).get("mark_price", 0.0)
                oi_value    = oi * mark if mark else 0.0
                row = {
                    "timestamp":     utc_now(),
                    "open_interest": oi,
                    "oi_value":      oi_value,
                }
                db.enqueue_write("open_interest", insert_open_interest, pd.DataFrame([row]))
                shared_state["open_interest_live"] = row

            except asyncio.CancelledError:
                raise
            except _PARSE_ERRORS as e:
                log.error(f"[open_interest] malformed response, skipping: {e}")
            except Exception as e:
                log.warning(f"[open_interest] fetch error: {e}")

            elapsed = (utc_now() - t0).total_seconds()
            await asyncio.sleep(max(0, OPEN_INTEREST_POLL - elapsed))