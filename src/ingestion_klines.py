# src/ingestion_klines.py
"""
DATA INGESTION — KLINES (REST polling)
───────────────────────────────────────
• M1 polled every 60 s
• M5 polled every 300 s
• Converts raw Binance response → clean DataFrame
• Hands off to storage layer
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta

import aiohttp
import pandas as pd

from src.config import (
    KLINE_ENDPOINT, SYMBOL, KLINE_LIMIT,
    KLINE_POLL_M1, KLINE_POLL_M5,
)
from src.storage import save_klines
from src.db import build_candle_features
from src.logger import get_logger

log = get_logger("klines")

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
]
KEEP = ["open_time", "open", "high", "low", "close", "volume",
        "num_trades", "taker_buy_base_vol"]
NUMERIC = ["open", "high", "low", "close", "volume",
           "taker_buy_base_vol", "num_trades"]

_INTERVAL_DELTA = {"1m": timedelta(minutes=1), "5m": timedelta(minutes=5)}


def _parse(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=COLUMNS)[KEEP].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # drop the last (incomplete) candle
    df = df.iloc[:-1]
    return df


async def _fetch_klines(session: aiohttp.ClientSession, interval: str) -> pd.DataFrame:
    params = {"symbol": SYMBOL, "interval": interval, "limit": KLINE_LIMIT}
    async with session.get(KLINE_ENDPOINT, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
        r.raise_for_status()
        data = await r.json()
    return _parse(data)


async def poll_klines(interval: str, period: int, shared_state: dict) -> None:
    """
    Continuously polls Binance REST for klines.
    interval: '1m' or '5m'
    period:   poll interval in seconds
    shared_state: dict updated with latest DataFrame

    Har yangi yopilgan sham uchun build_candle_features() chaqiriladi —
    o'sha sham davomidagi barcha xom oqim (trades/orderbook/liquidations/
    CVD/OI/funding) DuckDB'da agregatlanib candle_features'ga yoziladi.
    """
    key = f"klines_{interval.replace('m','m')}"
    state_key = "m1" if interval == "1m" else "m5"
    delta = _INTERVAL_DELTA[interval]
    last_seen_open_time = None   # oxirgi qayta ishlangan sham — takror agregatsiya qilmaslik uchun

    async with aiohttp.ClientSession() as session:
        while True:
            t0 = time.monotonic()
            try:
                df = await _fetch_klines(session, interval)
                save_klines(df, state_key)
                shared_state[f"klines_{state_key}"] = df
                log.info(f"[klines/{state_key}] fetched {len(df)} candles | "
                         f"last={df['close'].iloc[-1]:.2f}")

                # ── yangi yopilgan sham(lar) uchun mikrostruktura snapshot ──────
                if not df.empty:
                    new_rows = df if last_seen_open_time is None else \
                        df[df["open_time"] > last_seen_open_time]
                    for open_time in new_rows["open_time"]:
                        close_time = open_time + delta
                        snap = build_candle_features(interval, open_time, close_time)
                        if snap:
                            log.info(
                                f"[klines/{state_key}] candle_features saqlandi | "
                                f"open_time={open_time} trades={snap['trade_count']} "
                                f"cvd_delta={snap['cvd_delta']}"
                            )
                        else:
                            log.warning(
                                f"[klines/{state_key}] candle_features SAQLANMADI "
                                f"(xato — yuqoridagi [db] error qarang) open_time={open_time}"
                            )
                    last_seen_open_time = df["open_time"].iloc[-1]

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error(f"[klines/{state_key}] fetch error: {e}")

            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0, period - elapsed))