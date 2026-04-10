"""
time_features.py
================
Temporal / calendar features for trading models.

Markets are NOT time-homogeneous:
    • 08:00–11:00 UTC  — London open (high liquidity)
    • 13:30–16:00 UTC  — NY open (overlap with London = peak volatility)
    • 00:00–02:00 UTC  — Asian session (BTC market — OKX, Binance Asia)
    • Monday open / Friday close — weekend gap risk
    • End of month — institutional rebalancing

A naive model trained on 24/7 BTC data will average over these regimes
and learn nothing useful.  Time features allow the net to condition its
predictions on the market "session" without hard-coding session logic.

All features are:
    1. Cyclical (sin/cos encoded) — so distance between 23:00 and 01:00
       is correctly represented as 2 hours, not 22 hours.
    2. Normalised to [-1, +1].
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _sin_cos(value: np.ndarray, period: float):
    """Encode a cyclical variable using sin/cos."""
    angle = 2 * np.pi * value / period
    return np.sin(angle), np.cos(angle)


# ──────────────────────────────────────────────
# 1. INTRADAY: HOUR OF DAY
# ──────────────────────────────────────────────

def hour_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Sin/cos encoding of the hour of day (0–23).

    Why cyclical encoding?
        Hour 23 and hour 0 should be "close" in the feature space.
        Linear encoding (23 vs 0) creates a spurious gap.

    hour_sin, hour_cos together define a 2D point on the unit circle —
    the neural net can learn any function of the hour with full continuity.
    """
    hour = index.hour.astype(float)
    sin_h, cos_h = _sin_cos(hour, 24.0)

    # Also expose raw hour as an integer for interpretability
    return pd.DataFrame({
        "hour_sin": sin_h,
        "hour_cos": cos_h,
        "hour_raw": hour,         # 0–23, useful for attention heads
    }, index=index)


# ──────────────────────────────────────────────
# 2. DAY OF WEEK
# ──────────────────────────────────────────────

def dow_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Sin/cos encoding of day of week (Monday=0, Sunday=6).

    Monday open: weekend gaps unwind — high uncertainty.
    Friday close: position squaring before weekend — direction bias.
    Wednesday: historically lower volatility mid-week.
    """
    dow = index.dayofweek.astype(float)   # 0=Mon, 6=Sun
    sin_d, cos_d = _sin_cos(dow, 7.0)

    return pd.DataFrame({
        "dow_sin": sin_d,
        "dow_cos": cos_d,
        "dow_raw": dow,
    }, index=index)


# ──────────────────────────────────────────────
# 3. DAY OF MONTH / MONTH OF YEAR
# ──────────────────────────────────────────────

def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Sin/cos encoding of day-of-month and month-of-year.

    Day of month:
        • End-of-month (27–31): institutional rebalancing, option expiry.
        • Start-of-month (1–3): new capital deployment.

    Month of year:
        • Q1/Q4 tend to be higher volatility in crypto.
        • "Sell in May" seasonality (crypto has its own version).
    """
    dom = index.day.astype(float)
    moy = index.month.astype(float)

    sin_dom, cos_dom = _sin_cos(dom, 31.0)
    sin_moy, cos_moy = _sin_cos(moy, 12.0)

    return pd.DataFrame({
        "dom_sin": sin_dom,
        "dom_cos": cos_dom,
        "moy_sin": sin_moy,
        "moy_cos": cos_moy,
    }, index=index)


# ──────────────────────────────────────────────
# 4. TRADING SESSION FLAGS
# ──────────────────────────────────────────────

def session_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Binary flags for major trading sessions (UTC times).

    Sessions defined (approximate, BTC-optimised):
        Asian  session: 00:00–08:00 UTC
        London session: 07:00–16:00 UTC
        NY     session: 12:00–21:00 UTC
        Overlap (London+NY): 13:00–17:00 UTC — peak volatility window

    These can be used by the network to learn session-specific patterns
    (e.g., large moves on NY open, quiet Asian consolidation).
    """
    hour = index.hour

    asian_session   = ((hour >= 0)  & (hour < 8)).astype(float)
    london_session  = ((hour >= 7)  & (hour < 16)).astype(float)
    ny_session      = ((hour >= 12) & (hour < 21)).astype(float)
    overlap_session = ((hour >= 13) & (hour < 17)).astype(float)

    # Weekend flag (Saturday=5, Sunday=6)
    is_weekend = (index.dayofweek >= 5).astype(float)

    # Monday open (first 4 hours of Monday)
    is_monday_open = ((index.dayofweek == 0) & (hour < 4)).astype(float)

    return pd.DataFrame({
        "session_asian":   asian_session,
        "session_london":  london_session,
        "session_ny":      ny_session,
        "session_overlap": overlap_session,
        "is_weekend":      is_weekend,
        "is_monday_open":  is_monday_open,
    }, index=index)


# ──────────────────────────────────────────────
# 5. BARS SINCE MIDNIGHT (intraday position)
# ──────────────────────────────────────────────

def bars_since_midnight(index: pd.DatetimeIndex,
                         bar_minutes: int = 15) -> pd.Series:
    """
    How many bars have elapsed since 00:00 UTC today?

    Provides a fine-grained intraday position signal.
    Normalised to [0, 1] so scale is consistent with other features.

    For M15: max = 96 bars per day → divide by 96.
    """
    minutes_elapsed = index.hour * 60 + index.minute
    bars_elapsed    = minutes_elapsed / bar_minutes
    bars_per_day    = 1440 / bar_minutes   # 96 for M15
    return pd.Series(bars_elapsed / bars_per_day,
                     index=index, name="intraday_position")


# ──────────────────────────────────────────────
# 6. MASTER BUILDER
# ──────────────────────────────────────────────

def add_time_features(df: pd.DataFrame,
                      bar_minutes: int = 15) -> pd.DataFrame:
    """
    Add all time features to `df`.
    Requires df.index to be a DatetimeIndex (UTC).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex for time features.")

    idx = df.index

    df = pd.concat([df,
                    hour_features(idx),
                    dow_features(idx),
                    calendar_features(idx),
                    session_flags(idx)], axis=1)

    df["intraday_position"] = bars_since_midnight(idx, bar_minutes)

    return df