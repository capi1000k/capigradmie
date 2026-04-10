"""
target.py
=========
Triple Barrier labeling with H1 trend filter.

Architecture:
    M15 data  →  Triple Barrier  →  raw label {-1, 0, 1}
    H1  data  →  Trend filter    →  final label {-1, 0, 1}

Triple Barrier (Lopez de Prado, "Advances in Financial ML", Ch.3):
    For each M15 bar at time t:
        Upper barrier : entry_price × (1 + atr_mult_tp × ATR / entry_price)
        Lower barrier : entry_price × (1 - atr_mult_sl × ATR / entry_price)
        Time  barrier : t + max_bars

    First barrier hit determines the label:
        Upper → +1  (profit target hit)
        Lower → -1  (stop loss hit)
        Time  →  0  (neither — sideways / no edge)

    Transaction cost is subtracted from both barriers so the model
    learns to find moves that are PROFITABLE AFTER COSTS.

H1 Trend Filter:
    H1 trend = bullish  iff  EMA20 > EMA50 > EMA200
    H1 trend = bearish  iff  EMA20 < EMA50 < EMA200
    Otherwise: sideways (no trade)

    Alignment rule:
        label +1 kept   only if H1 trend = bullish
        label -1 kept   only if H1 trend = bearish
        label  0 always kept (confirmed non-edge)
        misaligned labels → 0  (override to no-trade)

Usage:
    from target import build_targets

    labels = build_targets(
        m15=df_m15,   # DatetimeIndex, columns: open high low close volume
        h1=df_h1,     # same structure, hourly
    )
    # labels: pd.Series aligned to m15.index, values in {-1, 0, 1}
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

# Transaction cost (round-trip): entry + exit
# Binance USDT-M futures maker: 0.02% each side → 0.04% total
# Add ~0.02% slippage conservatively → 0.06% round-trip
ROUND_TRIP_COST = 0.0006   # 0.06% — adjust to your exchange/tier


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _wilder_atr(high: pd.Series, low: pd.Series,
                close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder ATR — same as in base_features to avoid circular imports."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


# ──────────────────────────────────────────────
# 1. H1 TREND FILTER
# ──────────────────────────────────────────────

def h1_trend(h1: pd.DataFrame,
             fast: int = 20,
             mid:  int = 50,
             slow: int = 200) -> pd.Series:
    """
    Compute H1 trend direction aligned to M15 timestamps.

    Returns pd.Series with values:
         1 → bullish  (EMA20 > EMA50 > EMA200)
        -1 → bearish  (EMA20 < EMA50 < EMA200)
         0 → sideways (mixed, no clear trend)

    The H1 series is forward-filled onto M15 frequency so every M15
    bar knows its "current" H1 trend without look-ahead.
    """
    c = h1["close"]
    ema_f = _ema(c, fast)
    ema_m = _ema(c, mid)
    ema_s = _ema(c, slow)

    trend = pd.Series(0, index=h1.index, name="h1_trend")
    trend[( ema_f > ema_m) & (ema_m > ema_s)] =  1
    trend[(ema_f < ema_m) & (ema_m < ema_s)] = -1

    return trend


# ──────────────────────────────────────────────
# 2. TRIPLE BARRIER LABELING
# ──────────────────────────────────────────────

def triple_barrier(
    m15: pd.DataFrame,
    atr_period:    int   = 14,
    atr_mult_tp:   float = 1.5,   # take-profit multiplier  (upper barrier)
    atr_mult_sl:   float = 1.0,   # stop-loss multiplier    (lower barrier)
    max_bars:      int   = 4,     # time barrier in M15 bars (4 × 15 min = 1 h)
    cost:          float = ROUND_TRIP_COST,
) -> pd.Series:
    """
    Compute Triple Barrier labels for every M15 bar.

    Parameters
    ----------
    m15         : OHLCV DataFrame with DatetimeIndex
    atr_period  : lookback for ATR (default 14)
    atr_mult_tp : upper barrier = entry + atr_mult_tp × ATR
    atr_mult_sl : lower barrier = entry - atr_mult_sl × ATR
    max_bars    : time barrier (number of M15 bars to look forward)
    cost        : round-trip transaction cost (subtracted from both barriers)

    Returns
    -------
    pd.Series of int8, values in {-1, 0, 1}, indexed like m15
        +1 → upper barrier hit first (bullish edge after costs)
        -1 → lower barrier hit first (bearish edge after costs)
         0 → time barrier hit (no edge)
    """
    close  = m15["close"].values
    high   = m15["high"].values
    low    = m15["low"].values
    atr    = _wilder_atr(m15["high"], m15["low"], m15["close"], atr_period).values
    N      = len(close)
    labels = np.zeros(N, dtype=np.int8)

    for i in range(N - 1):
        if np.isnan(atr[i]):
            continue

        entry  = close[i]
        bar_atr = atr[i]

        # Barriers — net of transaction cost
        # Upper: price must exceed this to be profitable after paying costs
        upper = entry * (1.0 + atr_mult_tp * bar_atr / entry) - entry * cost
        lower = entry * (1.0 - atr_mult_sl * bar_atr / entry) + entry * cost

        # Safety check: degenerate candles or cost > barrier
        if upper <= entry or lower >= entry:
            continue

        # Look forward up to max_bars
        end = min(i + max_bars + 1, N)
        label = 0  # default: time barrier

        for j in range(i + 1, end):
            h = high[j]
            l = low[j]

            if h >= upper:
                label =  1
                break
            if l <= lower:
                label = -1
                break

        labels[i] = label

    # Last max_bars rows have incomplete forward windows → NaN-like → 0
    labels[N - max_bars:] = 0

    return pd.Series(labels, index=m15.index, name="label")


# ──────────────────────────────────────────────
# 3. APPLY H1 TREND FILTER
# ──────────────────────────────────────────────

def apply_trend_filter(
    raw_labels: pd.Series,
    h1_trend_series: pd.Series,
    m15_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Override misaligned labels with 0.

    Logic:
        label +1 and H1 = bullish  → keep  +1
        label +1 and H1 ≠ bullish  → force  0  (counter-trend long → skip)
        label -1 and H1 = bearish  → keep  -1
        label -1 and H1 ≠ bearish  → force  0  (counter-trend short → skip)
        label  0                   → keep   0  (confirmed no-edge)

    Why force counter-trend labels to 0 (not flip them)?
        The model should learn NOT to trade into the trend, not to
        blindly take the opposite side.  A counter-trend +1 is likely
        a noise spike, not a real reversal signal.
    """
    # Align H1 trend to M15 index using forward-fill
    # ffill: at any M15 bar, use the most recent completed H1 candle's trend
    # Drop duplicates first (keep last) to avoid reindex error
    h1_clean = h1_trend_series[~h1_trend_series.index.duplicated(keep="last")]

    # Combine indices and ffill, then select only M15 timestamps
    combined_idx = h1_clean.index.union(m15_index).sort_values()
    trend_m15 = (
        h1_clean
        .reindex(combined_idx)
        .ffill()
        .reindex(m15_index)
        .fillna(0)
        .astype(int)
    )

    filtered = raw_labels.copy()

    # Misaligned longs → 0
    filtered[(filtered ==  1) & (trend_m15 !=  1)] = 0
    # Misaligned shorts → 0
    filtered[(filtered == -1) & (trend_m15 != -1)] = 0

    return filtered.rename("label_filtered")


# ──────────────────────────────────────────────
# 4. CLASS BALANCE DIAGNOSTIC
# ──────────────────────────────────────────────

def label_stats(labels: pd.Series) -> pd.DataFrame:
    """
    Print class distribution.  Healthy range:
        0  (no-edge)   : 40–70%  — most bars should have no clear edge
        +1 (long)      : 15–30%
        -1 (short)     : 15–30%
    Heavy imbalance → re-tune atr_mult_tp / atr_mult_sl.
    """
    counts = labels.value_counts().sort_index()
    pct    = (counts / len(labels) * 100).round(1)
    stats  = pd.DataFrame({
        "label": counts.index.map({-1: "short (-1)", 0: "no-edge (0)", 1: "long (+1)"}),
        "count": counts.values,
        "pct":   pct.values,
    })
    return stats


# ──────────────────────────────────────────────
# 5. MASTER BUILDER
# ──────────────────────────────────────────────

def build_targets(
    m15: pd.DataFrame,
    h1:  pd.DataFrame,
    atr_period:    int   = 14,
    atr_mult_tp:   float = 1.5,
    atr_mult_sl:   float = 1.0,
    max_bars:      int   = 4,
    h1_ema_fast:   int   = 20,
    h1_ema_mid:    int   = 50,
    h1_ema_slow:   int   = 200,
    cost:          float = ROUND_TRIP_COST,
    verbose:       bool  = True,
) -> pd.Series:
    """
    Full pipeline: Triple Barrier → H1 trend filter → final labels.

    Parameters
    ----------
    m15, h1       : OHLCV DataFrames with DatetimeIndex (UTC)
    atr_mult_tp   : take-profit ATR multiplier  (1.5 = 1.5× ATR above entry)
    atr_mult_sl   : stop-loss ATR multiplier    (1.0 = 1.0× ATR below entry)
    max_bars      : time barrier in M15 bars    (4 = 1 hour)
    cost          : round-trip transaction cost (0.0006 = 0.06%)
    verbose       : print class distribution after labeling

    Returns
    -------
    pd.Series aligned to m15.index, values {-1, 0, 1}
    """
    # Step 1: compute raw Triple Barrier labels on M15
    raw = triple_barrier(
        m15,
        atr_period=atr_period,
        atr_mult_tp=atr_mult_tp,
        atr_mult_sl=atr_mult_sl,
        max_bars=max_bars,
        cost=cost,
    )

    # Step 2: compute H1 trend
    trend = h1_trend(h1, fast=h1_ema_fast, mid=h1_ema_mid, slow=h1_ema_slow)

    # Step 3: apply trend filter
    final = apply_trend_filter(raw, trend, m15.index)

    # Step 4: report
    if verbose:
        print("\n── Raw labels (before trend filter) ──")
        print(label_stats(raw).to_string(index=False))
        print(f"\n── Final labels (after H1 trend filter) ──")
        print(label_stats(final).to_string(index=False))
        aligned = (
            ((raw ==  1) & (final ==  1)).sum() +
            ((raw == -1) & (final == -1)).sum()
        )
        total_signals = (raw != 0).sum()
        if total_signals > 0:
            pct_kept = aligned / total_signals * 100
            print(f"\nSignals kept after filter : {aligned:,} / {total_signals:,} ({pct_kept:.1f}%)")

    return final


# ──────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Minimal smoke test with synthetic data
    np.random.seed(42)
    n_m15 = 5000
    n_h1  = n_m15 // 4

    dates_m15 = pd.date_range("2021-01-01", periods=n_m15, freq="15min", tz="UTC")
    dates_h1  = pd.date_range("2021-01-01", periods=n_h1,  freq="h",     tz="UTC")

    def _fake_ohlcv(dates, seed=0):
        rng   = np.random.default_rng(seed)
        close = 30000 + np.cumsum(rng.normal(0, 50, len(dates)))
        noise = rng.uniform(0.001, 0.005, len(dates))
        return pd.DataFrame({
            "open":   close * (1 - noise / 2),
            "high":   close * (1 + noise),
            "low":    close * (1 - noise),
            "close":  close,
            "volume": rng.uniform(100, 2000, len(dates)),
        }, index=dates)

    m15 = _fake_ohlcv(dates_m15, seed=1)
    h1  = _fake_ohlcv(dates_h1,  seed=2)

    labels = build_targets(m15, h1, verbose=True)
    print(f"\nSample:\n{labels[labels != 0].head(10)}")