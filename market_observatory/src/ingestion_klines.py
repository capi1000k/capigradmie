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
from datetime import datetime, timezone

import aiohttp
import pandas as pd

from src.config import (
    KLINE_ENDPOINT, SYMBOL, KLINE_LIMIT,
    KLINE_POLL_M1, KLINE_POLL_M5,
)
from src.storage import save_klines
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
    """
    key = f"klines_{interval.replace('m','m')}"
    state_key = "m1" if interval == "1m" else "m5"

    async with aiohttp.ClientSession() as session:
        while True:
            t0 = time.monotonic()
            try:
                df = await _fetch_klines(session, interval)
                save_klines(df, state_key)
                shared_state[f"klines_{state_key}"] = df
                log.info(f"[klines/{state_key}] fetched {len(df)} candles | "
                         f"last={df['close'].iloc[-1]:.2f}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error(f"[klines/{state_key}] fetch error: {e}")

            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0, period - elapsed))
