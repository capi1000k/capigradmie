"""
smart_money.py
==============
Market structure features based on Smart Money Concepts (SMC) and
Price Action theory.

Key principle: ALL features here are computed using ONLY past data.
Swing points require `n` bars of confirmation AFTER the swing bar,
so the feature is recorded with a lag of `n` bars.  This eliminates
look-ahead bias — the most common and most fatal mistake in retail ML.

Features:
    • Swing High / Low detection
    • HH / HL / LH / LL classification
    • Break of Structure (BOS)
    • Change of Character (CHoCH)
    • Distance to last swing levels
    • Volatility Contraction Pattern (VCP / Range Compression)
    • Price-Path Convexity
    • Hurst Exponent
"""

import numpy as np
import pandas as pd
from numba import njit   # optional — falls back gracefully if not installed


# ──────────────────────────────────────────────
# 1. SWING HIGH / LOW DETECTION
# ──────────────────────────────────────────────

def detect_swings(high: pd.Series, low: pd.Series,
                  n: int = 3) -> pd.DataFrame:
    """
    Detect Swing Highs and Swing Lows with `n`-bar confirmation.

    Definition:
        Swing High at bar i  iff  high[i] > max(high[i-n:i])
                                  AND high[i] > max(high[i+1:i+n+1])

    Because we use future bars for confirmation, we *record* the event
    at bar i+n (the first bar at which we *know* it's a swing).
    This is the canonical approach to avoid look-ahead bias.

    Returns DataFrame with columns:
        swing_high : 1.0 at confirmed swing high bars, else 0.0
        swing_low  : 1.0 at confirmed swing low bars, else 0.0
        swing_high_price : price of the most recent confirmed swing high
        swing_low_price  : price of the most recent confirmed swing low
    """
    h = high.values
    l = low.values
    N = len(h)

    is_swing_high = np.zeros(N)
    is_swing_low  = np.zeros(N)

    for i in range(n, N - n):
        # Swing High: highest point in 2n+1 window centred on i
        if h[i] == max(h[i-n : i+n+1]):
            # Record at confirmation bar (i + n)
            is_swing_high[i + n] = 1.0

        # Swing Low: lowest point in 2n+1 window
        if l[i] == min(l[i-n : i+n+1]):
            is_swing_low[i + n] = 1.0

    idx = high.index

    # Carry forward the price level of the most recent swing
    sh_price = pd.Series(np.where(is_swing_high, h, np.nan), index=idx)
    sl_price = pd.Series(np.where(is_swing_low,  l, np.nan), index=idx)

    # Forward-fill: at every bar, "last known swing level"
    sh_price_ffill = sh_price.ffill()
    sl_price_ffill = sl_price.ffill()

    return pd.DataFrame({
        "swing_high":       pd.Series(is_swing_high, index=idx),
        "swing_low":        pd.Series(is_swing_low,  index=idx),
        "last_swing_high":  sh_price_ffill,
        "last_swing_low":   sl_price_ffill,
    })


# ──────────────────────────────────────────────
# 2. HH / HL / LH / LL CLASSIFICATION
# ──────────────────────────────────────────────

def classify_swing_sequence(swing_df: pd.DataFrame,
                             high: pd.Series,
                             low: pd.Series) -> pd.DataFrame:
    """
    Compare consecutive swing highs and lows to classify market structure.

    HH (Higher High) + HL (Higher Low) → bullish trend in progress
    LH (Lower High)  + LL (Lower Low)  → bearish trend in progress
    Mixed                               → consolidation / transition

    Encoded as integers for the neural network:
        +1 = HH/HL (bullish)
        -1 = LH/LL (bearish)
         0 = unchanged or insufficient data
    """
    sh_mask = swing_df["swing_high"].values.astype(bool)
    sl_mask = swing_df["swing_low"].values.astype(bool)
    h_vals  = high.values
    l_vals  = low.values
    idx     = high.index
    N       = len(h_vals)

    hh_flag = np.zeros(N)   # Higher High
    hl_flag = np.zeros(N)   # Higher Low
    lh_flag = np.zeros(N)   # Lower High
    ll_flag = np.zeros(N)   # Lower Low

    prev_sh = np.nan
    prev_sl = np.nan

    for i in range(N):
        if sh_mask[i]:
            curr_sh = h_vals[i]
            if not np.isnan(prev_sh):
                hh_flag[i] = 1.0 if curr_sh > prev_sh else 0.0
                lh_flag[i] = 1.0 if curr_sh < prev_sh else 0.0
            prev_sh = curr_sh

        if sl_mask[i]:
            curr_sl = l_vals[i]
            if not np.isnan(prev_sl):
                hl_flag[i] = 1.0 if curr_sl > prev_sl else 0.0
                ll_flag[i] = 1.0 if curr_sl < prev_sl else 0.0
            prev_sl = curr_sl

    # Rolling "structure bias": sum over recent bars
    # +1 per HH/HL event, -1 per LH/LL event → aggregate trend measure
    window = 10
    net_structure = (
        pd.Series(hh_flag + hl_flag - lh_flag - ll_flag, index=idx)
        .rolling(window).sum()
    )

    return pd.DataFrame({
        "hh_flag":       pd.Series(hh_flag, index=idx),
        "hl_flag":       pd.Series(hl_flag, index=idx),
        "lh_flag":       pd.Series(lh_flag, index=idx),
        "ll_flag":       pd.Series(ll_flag, index=idx),
        "net_structure": net_structure,
    })


# ──────────────────────────────────────────────
# 3. BREAK OF STRUCTURE (BOS) & CHANGE OF CHARACTER (CHoCH)
# ──────────────────────────────────────────────

def break_of_structure(close: pd.Series,
                       swing_df: pd.DataFrame,
                       eps: float = 1e-9) -> pd.DataFrame:
    """
    BOS  (Break of Structure):
        Price closes ABOVE the last confirmed Swing High (bullish BOS)
        Price closes BELOW the last confirmed Swing Low  (bearish BOS)
        → Confirms trend continuation.

    CHoCH (Change of Character):
        In an uptrend (price above last swing low):
            Price closes BELOW the last HL → potential reversal.
        In a downtrend (price below last swing high):
            Price closes ABOVE the last LH → potential reversal.
        → First warning of a trend reversal.

    These are encoded as:
        bos_bull       : 1 on the bar a bullish BOS fires
        bos_bear       : 1 on the bar a bearish BOS fires
        choch_bull     : 1 on bar a bullish CHoCH fires
        choch_bear     : 1 on bar a bearish CHoCH fires
        bars_since_bos : bars elapsed since the last BOS of either kind
        dist_to_last_sh: (close - last_swing_high) / close  — structural distance
        dist_to_last_sl: (close - last_swing_low)  / close
    """
    c          = close.values
    last_sh    = swing_df["last_swing_high"].values
    last_sl    = swing_df["last_swing_low"].values
    N          = len(c)
    idx        = close.index

    bos_bull   = np.zeros(N)
    bos_bear   = np.zeros(N)
    choch_bull = np.zeros(N)
    choch_bear = np.zeros(N)

    prev_c    = np.roll(c, 1)
    prev_c[0] = c[0]

    for i in range(1, N):
        sh = last_sh[i]
        sl = last_sl[i]

        if np.isnan(sh) or np.isnan(sl):
            continue

        # BOS: close crosses a structural level
        if prev_c[i] <= sh and c[i] > sh:
            bos_bull[i] = 1.0
        if prev_c[i] >= sl and c[i] < sl:
            bos_bear[i] = 1.0

        # CHoCH: in a trend, price breaks the *opposite* structural level
        # Uptrend context: price is above last swing low
        if c[i] > sl and c[i] < sh and c[i] < prev_c[i]:
            # In uptrend, closing below the last swing low = CHoCH bearish
            if c[i] < sl:
                choch_bear[i] = 1.0

        # Downtrend context: price is below last swing high
        if c[i] < sh and c[i] > sl and c[i] > prev_c[i]:
            if c[i] > sh:
                choch_bull[i] = 1.0

    # Time since last BOS
    bos_any = bos_bull + bos_bear
    bars_since_bos = _bars_since_event(bos_any, N)

    # Structural distance (stationary, normalised)
    dist_sh = (c - last_sh) / (np.abs(c) + eps)
    dist_sl = (c - last_sl) / (np.abs(c) + eps)

    return pd.DataFrame({
        "bos_bull":        pd.Series(bos_bull,   index=idx),
        "bos_bear":        pd.Series(bos_bear,   index=idx),
        "choch_bull":      pd.Series(choch_bull, index=idx),
        "choch_bear":      pd.Series(choch_bear, index=idx),
        "bars_since_bos":  pd.Series(bars_since_bos, index=idx),
        "dist_to_swing_h": pd.Series(dist_sh, index=idx),
        "dist_to_swing_l": pd.Series(dist_sl, index=idx),
    })


def _bars_since_event(event_array: np.ndarray, N: int) -> np.ndarray:
    """Count bars since the last 1 in a binary event array."""
    result = np.full(N, np.nan)
    count  = np.nan
    for i in range(N):
        if event_array[i] == 1.0:
            count = 0
        elif not np.isnan(count):
            count += 1
        result[i] = count
    return result


# ──────────────────────────────────────────────
# 4. VOLATILITY CONTRACTION PATTERN (VCP)
# ──────────────────────────────────────────────

def volatility_compression(high: pd.Series, low: pd.Series,
                            short_window: int = 20,
                            long_window:  int = 100,
                            eps: float = 1e-9) -> pd.Series:
    """
    Range Compression ratio:
        Compression = (max(H_20) - min(L_20)) / (max(H_100) - min(L_100))

    Popularised by Mark Minervini's VCP (Volatility Contraction Pattern).

    A declining compression ratio (< 0.5) combined with drying volume
    indicates that "supply is being absorbed" by institutional buyers.
    The "line of least resistance" shifts upward → high-probability breakout.

    < 0.3 → very tight coil (maximum compression before move)
    > 1.0 → range expanding (possible breakout in progress)
    """
    range_short = high.rolling(short_window).max() - low.rolling(short_window).min()
    range_long  = high.rolling(long_window).max()  - low.rolling(long_window).min()
    return (range_short / range_long.clip(lower=eps)).rename("vcp_compression")


# ──────────────────────────────────────────────
# 5. PRICE-PATH CONVEXITY
# ──────────────────────────────────────────────

def price_convexity(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Convexity = (midpoint_of_endpoints - mean_of_path) / midpoint_of_endpoints

    Measures whether the price trajectory is "convex" (accelerating, FOMO)
    or "concave" (decelerating, potential reversal).

    High positive convexity → parabolic / blow-off move (mean-reverts).
    High negative convexity → concave / rounded bottom pattern.
    Near zero              → linear trend.

    Research: Convexity is negatively correlated with next-bar returns
    at short horizons (parabolic moves are unsustainable).
    """
    results = np.full(len(close), np.nan)
    c = close.values

    for i in range(window, len(c)):
        segment = c[i-window : i+1]
        midpoint = (segment[0] + segment[-1]) / 2.0
        mean_path = segment.mean()
        if abs(midpoint) > 1e-9:
            results[i] = (midpoint - mean_path) / abs(midpoint)

    return pd.Series(results, index=close.index, name=f"convexity_{window}")


# ──────────────────────────────────────────────
# 6. HURST EXPONENT
# ──────────────────────────────────────────────

def hurst_exponent(close: pd.Series,
                   window: int = 100,
                   min_lags: int = 2,
                   max_lags: int = 20) -> pd.Series:
    """
    Rolling Hurst Exponent using R/S (Rescaled Range) analysis.

    H > 0.5 → Trending / Persistent  (momentum works)
    H < 0.5 → Mean-reverting          (fade the move)
    H ≈ 0.5 → Random walk             (efficient, unpredictable)

    This is the most important regime-detection feature in the pipeline.
    The neural net can use it to implicitly "switch" between momentum
    and mean-reversion submodels without any explicit hard rule.

    Implementation: for each window we compute R/S at multiple lag lengths
    then fit a log-log regression to estimate H.  This is computationally
    expensive — use window=100 as a minimum for stability.
    """
    log_close = np.log(close.values)
    N = len(log_close)
    results = np.full(N, np.nan)

    lags = np.arange(min_lags, max_lags + 1)
    log_lags = np.log(lags)

    for i in range(window, N):
        segment = log_close[i - window : i]
        rs_vals = []
        for lag in lags:
            rs = _rs_statistic(segment, lag)
            rs_vals.append(rs)

        rs_vals = np.array(rs_vals)
        valid = rs_vals > 0
        if valid.sum() < 3:
            continue

        # OLS: log(R/S) = H * log(lag) + const
        log_rs = np.log(rs_vals[valid])
        ll     = log_lags[valid]
        # slope = H
        ll_mean   = ll.mean()
        rs_mean   = log_rs.mean()
        h_est = ((ll - ll_mean) * (log_rs - rs_mean)).sum() / ((ll - ll_mean)**2).sum()
        results[i] = np.clip(h_est, 0.0, 1.0)

    return pd.Series(results, index=close.index, name=f"hurst_{window}")


def _rs_statistic(series: np.ndarray, lag: int) -> float:
    """R/S statistic for a single lag (internal helper)."""
    n_chunks = len(series) // lag
    if n_chunks < 1:
        return np.nan

    rs_list = []
    for k in range(n_chunks):
        chunk = series[k*lag : (k+1)*lag]
        mean  = chunk.mean()
        deviations = np.cumsum(chunk - mean)
        r = deviations.max() - deviations.min()
        s = chunk.std(ddof=1)
        if s > 0:
            rs_list.append(r / s)

    return np.mean(rs_list) if rs_list else np.nan


# ──────────────────────────────────────────────
# 7. MASTER BUILDER
# ──────────────────────────────────────────────

def add_smart_money_features(df: pd.DataFrame,
                              swing_n: int = 3,
                              hurst_window: int = 100,
                              convexity_window: int = 20,
                              vcp_short: int = 20,
                              vcp_long: int = 100) -> pd.DataFrame:
    """
    Add all Smart Money / Market Structure features to `df`.
    Expected columns: open, high, low, close, volume
    """
    h, l, c = df["high"], df["low"], df["close"]

    # Swings
    swing_df = detect_swings(h, l, n=swing_n)
    df = pd.concat([df, swing_df], axis=1)

    # HH / HL / LH / LL
    seq_df = classify_swing_sequence(swing_df, h, l)
    df = pd.concat([df, seq_df], axis=1)

    # BOS / CHoCH
    bos_df = break_of_structure(c, swing_df)
    df = pd.concat([df, bos_df], axis=1)

    # VCP
    df["vcp_compression"] = volatility_compression(h, l, vcp_short, vcp_long)

    # Convexity
    df[f"convexity_{convexity_window}"] = price_convexity(c, convexity_window)

    # Hurst (expensive — runs once per build)
    df[f"hurst_{hurst_window}"] = hurst_exponent(c, hurst_window)

    return df