# src/features.py
"""
FEATURE PREPARATION
────────────────────
Transforms raw data → market microstructure features.
All functions are stateless (df in → df out).
"""

import numpy as np
import pandas as pd


# ─── KLINES FEATURES ─────────────────────────────────────────────────────────

def kline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility and return features to klines DataFrame."""
    df = df.copy()
    df["volatility"]   = df["high"] - df["low"]
    df["returns"]      = df["close"].pct_change()
    df["log_returns"]  = np.log(df["close"] / df["close"].shift(1))
    df["body_size"]    = (df["close"] - df["open"]).abs()
    df["body_pct"]     = df["body_size"] / df["open"]
    df["is_bullish"]   = df["close"] > df["open"]
    df["vwap_proxy"]   = (df["high"] + df["low"] + df["close"]) / 3
    # rolling vol (10-bar)
    df["vol_10"]       = df["returns"].rolling(10).std()
    return df


# ─── TRADE FEATURES ──────────────────────────────────────────────────────────

def trade_features(df: pd.DataFrame, window: str = "1min") -> pd.DataFrame:
    """
    Aggregate trade-level data into time buckets.
    is_buyer_mm=True means the market maker is buyer → seller aggressor.
    """
    if df.empty:
        return df

    df = df.copy()
    df["side"] = df["is_buyer_mm"].map({True: "sell", False: "buy"})
    df["buy_vol"]  = df["quantity"].where(df["side"] == "buy",  0)
    df["sell_vol"] = df["quantity"].where(df["side"] == "sell", 0)

    # resample into buckets
    df = df.set_index("timestamp")
    agg = df.resample(window).agg(
        total_vol   = ("quantity",  "sum"),
        buy_vol     = ("buy_vol",   "sum"),
        sell_vol    = ("sell_vol",  "sum"),
        trade_count = ("quantity",  "count"),
        avg_price   = ("price",     "mean"),
        vwap        = ("price",     lambda x: np.average(x, weights=df.loc[x.index, "quantity"])),
    ).reset_index()

    agg["buy_sell_ratio"]   = agg["buy_vol"] / (agg["sell_vol"] + 1e-10)
    agg["trade_intensity"]  = agg["trade_count"]  # trades per window
    return agg


# ─── ORDERBOOK FEATURES ───────────────────────────────────────────────────────

def orderbook_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived microstructure metrics to orderbook DataFrame."""
    if df.empty:
        return df

    df = df.copy()
    # imbalance already stored; add rolling smoothing
    df["imbalance_5"]   = df["imbalance"].rolling(5,  min_periods=1).mean()
    df["imbalance_20"]  = df["imbalance"].rolling(20, min_periods=1).mean()

    # liquidity pressure: if bid_vol >> ask_vol → buyers dominate
    df["liq_pressure"]  = df["bid_vol"] / (df["ask_vol"] + 1e-10)

    # spread basis points
    df["spread_bps"]    = (df["spread"] / df["mid_price"]) * 10_000

    return df


# ─── SHARED STATE → FEATURE SNAPSHOT ─────────────────────────────────────────

def compute_live_snapshot(shared_state: dict) -> dict:
    """
    Derives a single-point feature snapshot from shared_state
    (used by dashboard for annotation).
    """
    snap = {}

    ob = shared_state.get("orderbook_live", {})
    if ob:
        snap["imbalance"]    = ob.get("imbalance", 0)
        snap["spread"]       = ob.get("spread", 0)
        snap["mid_price"]    = ob.get("mid_price", 0)
        snap["bid_vol"]      = ob.get("bid_vol", 0)
        snap["ask_vol"]      = ob.get("ask_vol", 0)
        snap["liq_pressure"] = snap["bid_vol"] / (snap["ask_vol"] + 1e-10)

    from collections import deque
    trades: deque = shared_state.get("trades_live", deque())
    if trades:
        trades_df = pd.DataFrame(list(trades))
        buys  = trades_df[~trades_df["is_buyer_mm"]]["quantity"].sum()
        sells = trades_df[ trades_df["is_buyer_mm"]]["quantity"].sum()
        snap["buy_sell_ratio"]  = buys / (sells + 1e-10)
        snap["trade_intensity"] = len(trades_df)

    return snap
