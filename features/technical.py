"""
technical.py
============
Classical technical indicators re-implemented from scratch (no TA-Lib).

Philosophy: we don't use these as buy/sell signals.  We feed their
*derivatives* (slopes, distances, divergences) into the neural net so
it can discover non-linear combinations that precede profitable moves.
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    """Standard EMA using pandas EWM (span = period)."""
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's Smoothing (used in RSI, ATR).
    alpha = 1/period  →  slower decay than standard EMA.
    """
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)


# ──────────────────────────────────────────────
# 1. EMA SUITE
# ──────────────────────────────────────────────

def ema_features(close: pd.Series,
                 periods: tuple = (20, 50, 200),
                 slope_lag: int = 3,
                 eps: float = 1e-9) -> pd.DataFrame:
    """
    For each EMA period returns:
        ema_distance_{n}  →  (close - EMA) / EMA   (stationary)
        ema_slope_{n}     →  (EMA_t - EMA_{t-k}) / k  normalised by EMA

    Why it matters:
        • Raw EMA price is non-stationary → useless for ML.
        • Distance encodes "how stretched" price is from its mean.
        • Slope encodes the *velocity* of the trend.
        • Convergence of multiple EMAs = low-volatility coiling before
          an explosive move.
    """
    out = {}
    for p in periods:
        ema = _ema(close, p)
        out[f"ema_dist_{p}"]  = (close - ema) / ema.clip(lower=eps)
        out[f"ema_slope_{p}"] = (ema - ema.shift(slope_lag)) / (ema.shift(slope_lag).clip(lower=eps) * slope_lag)

    # EMA convergence: std of the three normalised EMAs
    # Low divergence → coiling; high → stretched trend
    dists = pd.DataFrame({p: out[f"ema_dist_{p}"] for p in periods})
    out["ema_convergence"] = dists.std(axis=1)

    return pd.DataFrame(out, index=close.index)


# ──────────────────────────────────────────────
# 2. RSI (Wilder)
# ──────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14, eps: float = 1e-9) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing.

    In this framework RSI is NOT an overbought/oversold signal.
    It is a measure of *internal strength*:
        • RSI persistently > 60 in an uptrend → institutional accumulation.
        • RSI divergence (price makes new high, RSI doesn't) → exhaustion.

    We also expose the RSI *velocity* (first difference) as a separate
    feature to catch momentum deceleration before price reverses.
    """
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)
    rs = avg_gain / avg_loss.clip(lower=eps)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{period}")


def rsi_velocity(close: pd.Series, period: int = 14) -> pd.Series:
    """First difference of RSI — rate of change of internal momentum."""
    r = rsi(close, period)
    return r.diff().rename(f"rsi_{period}_velocity")


# ──────────────────────────────────────────────
# 3. ATR (Wilder)
# ──────────────────────────────────────────────

def atr(high: pd.Series, low: pd.Series,
        close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range — the gold standard volatility measure for traders.
    Used internally by other features (relative spread, VCP, etc.).
    """
    tr = _true_range(high, low, close)
    return _wilder_smooth(tr, period).rename(f"atr_{period}")


def atr_ratio(high: pd.Series, low: pd.Series,
              close: pd.Series,
              short_period: int = 7,
              long_period: int = 50,
              eps: float = 1e-9) -> pd.Series:
    """
    ATR_short / ATR_long

    < 1  → volatility contraction (market is coiling)
    > 1  → range expansion / breakout context

    This is one of the best leading indicators of an impending breakout:
    the ratio declines for several bars before the explosive move.
    """
    short_atr = atr(high, low, close, short_period)
    long_atr  = atr(high, low, close, long_period)
    return (short_atr / long_atr.clip(lower=eps)).rename("atr_ratio")


# ──────────────────────────────────────────────
# 4. MACD
# ──────────────────────────────────────────────

def macd_features(close: pd.Series,
                  fast: int = 12,
                  slow: int = 26,
                  signal: int = 9,
                  eps: float = 1e-9) -> pd.DataFrame:
    """
    Returns: macd_line, macd_signal, macd_histogram (all price-normalised).

    Rather than using absolute MACD values (non-stationary), we divide
    by the slow EMA so values are price-agnostic and comparable across
    BTC market regimes.

    histogram > 0 and rising  → bullish momentum accelerating
    histogram crossing zero   → momentum regime change
    """
    ema_fast   = _ema(close, fast)
    ema_slow   = _ema(close, slow)
    norm       = ema_slow.clip(lower=eps)

    macd_line  = (ema_fast - ema_slow) / norm
    sig_line   = _ema(macd_line, signal)
    histogram  = macd_line - sig_line

    return pd.DataFrame({
        "macd_line":      macd_line,
        "macd_signal":    sig_line,
        "macd_histogram": histogram,
    }, index=close.index)


# ──────────────────────────────────────────────
# 5. STOCHASTIC OSCILLATOR
# ──────────────────────────────────────────────

def stochastic(high: pd.Series, low: pd.Series,
               close: pd.Series,
               k_period: int = 14,
               d_period: int = 3,
               eps: float = 1e-9) -> pd.DataFrame:
    """
    %K and %D stochastic.

    Where is the close relative to the recent high-low range?
    Used in tandem with RSI to confirm momentum exhaustion:
        Both overbought + bearish divergence → high-probability reversal.
    """
    lowest_low   = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    rng = (highest_high - lowest_low).clip(lower=eps)

    k = 100 * (close - lowest_low) / rng
    d = k.rolling(d_period).mean()

    return pd.DataFrame({"stoch_k": k, "stoch_d": d}, index=close.index)


# ──────────────────────────────────────────────
# 6. ROLLING SKEWNESS & KURTOSIS (Statistical)
# ──────────────────────────────────────────────

def rolling_skew(close: pd.Series, window: int = 50) -> pd.Series:
    """
    Rolling skewness of log returns.

    Negative skew → left tail dominates → crash risk present.
    Positive skew → rally potential / FOMO environment.
    Skew extremes often precede trend reversals.
    """
    lr = np.log(close).diff()
    return lr.rolling(window).skew().rename(f"rolling_skew_{window}")


def rolling_kurt(close: pd.Series, window: int = 50) -> pd.Series:
    """
    Rolling excess kurtosis of log returns.

    High kurtosis → fat tails → extreme moves are more likely than
    the Gaussian model predicts.  Useful for regime detection.
    """
    lr = np.log(close).diff()
    return lr.rolling(window).kurt().rename(f"rolling_kurt_{window}")


# ──────────────────────────────────────────────
# 7. MASTER BUILDER
# ──────────────────────────────────────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical features to `df`.
    Expected columns: open, high, low, close, volume
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # EMA suite
    ema_df = ema_features(c)
    df = pd.concat([df, ema_df], axis=1)

    # RSI
    df["rsi_14"]          = rsi(c, 14)
    df["rsi_14_velocity"] = rsi_velocity(c, 14)

    # ATR
    df["atr_14"]   = atr(h, l, c, 14)
    df["atr_ratio"] = atr_ratio(h, l, c)

    # MACD
    macd_df = macd_features(c)
    df = pd.concat([df, macd_df], axis=1)

    # Stochastic
    stoch_df = stochastic(h, l, c)
    df = pd.concat([df, stoch_df], axis=1)

    # Statistical moments
    df["rolling_skew_50"] = rolling_skew(c, 50)
    df["rolling_kurt_50"] = rolling_kurt(c, 50)

    return df