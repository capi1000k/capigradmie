"""
base_features.py
================
Fundamental price-action features derived from raw OHLCV candles.

These are the "atoms" of the pipeline — stationary representations of
what happened inside each bar. They answer: How far did price move?
Who won the battle between buyers and sellers within this candle?
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# 1. LOG RETURNS
# ──────────────────────────────────────────────

def log_return(close: pd.Series) -> pd.Series:
    """
    r_t = ln(P_t) - ln(P_{t-1})

    Why it matters:
        • Price-level agnostic: comparable across BTC at $10k vs $60k.
        • Additive over time (multi-bar returns = sum of single-bar returns).
        • Closer to normally distributed than simple % changes → better
          gradient behaviour in neural nets.
    """
    return np.log(close).diff().rename("log_return")


def rolling_log_return(close: pd.Series, window: int = 4) -> pd.Series:
    """
    Sum of log returns over `window` bars — captures the net directional
    drift over a short horizon without being path-dependent.
    """
    lr = np.log(close).diff()
    return lr.rolling(window).sum().rename(f"log_return_{window}b")


# ──────────────────────────────────────────────
# 2. CANDLE ANATOMY
# ──────────────────────────────────────────────

def candle_body(open_: pd.Series, close: pd.Series) -> pd.Series:
    """
    Signed body size as a fraction of the candle range.

    Positive → bullish bar; Negative → bearish bar.
    Captures the conviction behind a move:
        • |body| ≈ 1  → full-bodied momentum candle (no wicks)
        • |body| ≈ 0  → doji / indecision
    """
    return (close - open_).rename("candle_body")


def upper_wick(high: pd.Series, open_: pd.Series, close: pd.Series) -> pd.Series:
    """
    Upper wick = distance from the top of body to the high.

    upper_wick = High - max(Open, Close)

    Why it matters:
        Price was pushed up but then rejected — overhead supply / passive
        sellers absorbed the aggressive buying.  A large upper wick after
        a bullish run is a classic "shooting star" exhaustion signal.
    """
    return (high - np.maximum(open_, close)).rename("upper_wick")


def lower_wick(low: pd.Series, open_: pd.Series, close: pd.Series) -> pd.Series:
    """
    Lower wick = distance from the bottom of body to the low.

    lower_wick = min(Open, Close) - Low

    Why it matters:
        Price probed lower but buyers stepped in — demand at the lows.
        Large lower wick = liquidity grab / spring / hammer.
    """
    return (np.minimum(open_, close) - low).rename("lower_wick")


def upper_wick_ratio(high: pd.Series, low: pd.Series,
                     open_: pd.Series, close: pd.Series,
                     eps: float = 1e-9) -> pd.Series:
    """
    upper_wick / (high - low)  ∈ [0, 1]

    Normalised by the candle's total range so the ratio is comparable
    across volatility regimes.  Close to 1 → almost the whole candle
    is rejection wick with tiny body.
    """
    uw = high - np.maximum(open_, close)
    rng = (high - low).clip(lower=eps)
    return (uw / rng).rename("upper_wick_ratio")


def lower_wick_ratio(high: pd.Series, low: pd.Series,
                     open_: pd.Series, close: pd.Series,
                     eps: float = 1e-9) -> pd.Series:
    """
    lower_wick / (high - low)  ∈ [0, 1]

    Measures how much of the candle's range is below-body demand.
    """
    lw = np.minimum(open_, close) - low
    rng = (high - low).clip(lower=eps)
    return (lw / rng).rename("lower_wick_ratio")


def body_ratio(high: pd.Series, low: pd.Series,
               open_: pd.Series, close: pd.Series,
               eps: float = 1e-9) -> pd.Series:
    """
    |Close - Open| / (High - Low)  ∈ [0, 1]

    Pure body intensity — the fraction of the total range that is
    "decided" price movement.  High body_ratio → decisive, low → indecisive.
    """
    body = np.abs(close - open_)
    rng = (high - low).clip(lower=eps)
    return (body / rng).rename("body_ratio")


# ──────────────────────────────────────────────
# 3. RANGE NORMALIZATION
# ──────────────────────────────────────────────

def _wilder_atr(high: pd.Series, low: pd.Series,
                close: pd.Series, period: int = 14) -> pd.Series:
    """
    True Range and Wilder-smoothed ATR (internal helper).
    Uses EWM with alpha = 1/period, matching the original Wilder formula.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def relative_spread(high: pd.Series, low: pd.Series,
                    close: pd.Series, atr_period: int = 14,
                    eps: float = 1e-9) -> pd.Series:
    """
    (High - Low) / ATR(period)

    Identifies range expansion / contraction relative to recent
    average volatility.
        > 1.5  → wide, potentially breakout candle
        < 0.5  → compressed, inside-bar / coiling
    """
    atr = _wilder_atr(high, low, close, atr_period)
    rng = (high - low)
    return (rng / atr.clip(lower=eps)).rename("relative_spread")


def rolling_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling standard deviation of log returns.

    The "thermometer" of the market — how noisy is price right now?
    Regime-detection models condition on this to switch between
    trend-following and mean-reversion modes.
    """
    lr = np.log(close).diff()
    return lr.rolling(window).std().rename(f"rolling_vol_{window}")


# ──────────────────────────────────────────────
# 4. MASTER BUILDER
# ──────────────────────────────────────────────

def add_base_features(df: pd.DataFrame,
                      atr_period: int = 14,
                      vol_window: int = 20) -> pd.DataFrame:
    """
    Add all base price-action features to `df` in-place.

    Expected columns: open, high, low, close, volume
    Returns the same DataFrame with new columns appended.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    df["log_return"]         = log_return(c)
    df["log_return_4b"]      = rolling_log_return(c, 4)
    df["candle_body"]        = candle_body(o, c)
    df["upper_wick"]         = upper_wick(h, o, c)
    df["lower_wick"]         = lower_wick(l, o, c)
    df["upper_wick_ratio"]   = upper_wick_ratio(h, l, o, c)
    df["lower_wick_ratio"]   = lower_wick_ratio(h, l, o, c)
    df["body_ratio"]         = body_ratio(h, l, o, c)
    df["relative_spread"]    = relative_spread(h, l, c, atr_period)
    df[f"rolling_vol_{vol_window}"] = rolling_volatility(c, vol_window)

    return df