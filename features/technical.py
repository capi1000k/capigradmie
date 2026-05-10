"""
technical.py
============
Klassik texnik indikatorlar — scratch dan yozilgan (TA-Lib shart emas).

Falsafa: raw indikator qiymatlarini emas, ularning
HOSILALARINI (slope, distance, divergence) beramiz.
Model o'zi non-linear kombinatsiyalarni kashf qiladi.

Featurelar:
    EMA suite    : ema_dist_20/50/200, ema_slope_20/50/200, ema_convergence
    RSI          : rsi_14, rsi_14_velocity
    ATR          : atr_14, atr_ratio
    MACD         : macd_line, macd_signal, macd_histogram
    Stochastic   : stoch_k, stoch_d
    Statistical  : rolling_skew_50, rolling_kurt_50
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# YORDAMCHILAR
# ──────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _true_range(high: pd.Series,
                low: pd.Series,
                close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)


# ──────────────────────────────────────────────
# 1. EMA SUITE
# ──────────────────────────────────────────────

def ema_features(close: pd.Series,
                 periods: tuple = (20, 50, 200),
                 slope_lag: int = 3,
                 eps: float = 1e-9) -> pd.DataFrame:
    """
    Har bir EMA davri uchun:
        ema_dist_{n}  = (Close - EMA) / EMA      — stationary, qanchalik cho'zilgan
        ema_slope_{n} = (EMA_t - EMA_{t-k}) / EMA — trend tezligi

    ema_convergence = 3 ta EMA distance ning std
        Past → EMAdlar bir-biriga yaqin (coiling, breakout oldidan)
        Katta → trend cho'zilgan
    """
    out = {}
    for p in periods:
        ema = _ema(close, p)
        out[f"ema_dist_{p}"]  = (close - ema) / ema.clip(lower=eps)
        out[f"ema_slope_{p}"] = (
            (ema - ema.shift(slope_lag)) /
            (ema.shift(slope_lag).clip(lower=eps) * slope_lag)
        )

    dists = pd.DataFrame({p: out[f"ema_dist_{p}"] for p in periods})
    out["ema_convergence"] = dists.std(axis=1)

    return pd.DataFrame(out, index=close.index)


# ──────────────────────────────────────────────
# 2. RSI
# ──────────────────────────────────────────────

def rsi(close: pd.Series,
        period: int = 14,
        eps: float = 1e-9) -> pd.Series:
    """
    Wilder RSI.
    Bu yerda signal emas — ichki kuch o'lchovi sifatida ishlatiladi.
    RSI persistently > 60 → institutional accumulation.
    """
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)
    rs       = avg_gain / avg_loss.clip(lower=eps)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{period}")


def rsi_velocity(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI ning birinchi farqi — momentum tezlashishi/sekinlashishi."""
    return rsi(close, period).diff().rename(f"rsi_{period}_velocity")


# ──────────────────────────────────────────────
# 3. ATR
# ──────────────────────────────────────────────

def atr(high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return _wilder_smooth(tr, period).rename(f"atr_{period}")


def atr_ratio(high: pd.Series,
              low: pd.Series,
              close: pd.Series,
              short_period: int = 7,
              long_period: int = 50,
              eps: float = 1e-9) -> pd.Series:
    """
    ATR_short / ATR_long

    < 1 → volatilite qisqarmoqda (coiling)
    > 1 → range kengaymoqda (breakout)

    Breakout oldidan bu nisbat pasayadi — eng yaxshi leading indicator.
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
    Slow EMA bilan normalize qilingan MACD — price-agnostic.

    histogram > 0 va o'smoqda → bullish momentum tezlashmoqda
    histogram noldan o'tmoqda → momentum rejim o'zgarishi
    """
    ema_fast  = _ema(close, fast)
    ema_slow  = _ema(close, slow)
    norm      = ema_slow.clip(lower=eps)

    macd_line = (ema_fast - ema_slow) / norm
    sig_line  = _ema(macd_line, signal)
    histogram = macd_line - sig_line

    return pd.DataFrame({
        "macd_line":      macd_line,
        "macd_signal":    sig_line,
        "macd_histogram": histogram,
    }, index=close.index)


# ──────────────────────────────────────────────
# 5. STOCHASTIC
# ──────────────────────────────────────────────

def stochastic(high: pd.Series,
               low: pd.Series,
               close: pd.Series,
               k_period: int = 14,
               d_period: int = 3,
               eps: float = 1e-9) -> pd.DataFrame:
    """
    Close so'nggi range ichida qayerda turganini ko'rsatadi.
    RSI bilan birga momentum tugashi/boshlanishini tasdiqlaydi.
    """
    lowest_low   = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    rng          = (highest_high - lowest_low).clip(lower=eps)

    k = 100 * (close - lowest_low) / rng
    d = k.rolling(d_period).mean()

    return pd.DataFrame({"stoch_k": k, "stoch_d": d}, index=close.index)


# ──────────────────────────────────────────────
# 6. STATISTICAL MOMENTS
# ──────────────────────────────────────────────

def rolling_skew(close: pd.Series, window: int = 50) -> pd.Series:
    """
    Log return larnig rolling skewness.
    Manfiy skew → chap dum katta → crash xavfi bor.
    Musbat skew → o'ng dum katta → rally potentsial.
    """
    return np.log(close).diff().rolling(window).skew().rename(f"rolling_skew_{window}")


def rolling_kurt(close: pd.Series, window: int = 50) -> pd.Series:
    """
    Rolling excess kurtosis.
    Yuqori kurtosis → fat tail → ekstrem harakatlar ehtimoli yuqori.
    """
    return np.log(close).diff().rolling(window).kurt().rename(f"rolling_kurt_{window}")


# ──────────────────────────────────────────────
# MASTER BUILDER
# ──────────────────────────────────────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Barcha texnik featurelarni df ga qo'shadi.
    Kutilgan ustunlar: open, high, low, close
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    df = pd.concat([df, ema_features(c)], axis=1)

    df["rsi_14"]           = rsi(c, 14)
    df["rsi_14_velocity"]  = rsi_velocity(c, 14)

    df["atr_14"]           = atr(h, l, c, 14)
    df["atr_ratio"]        = atr_ratio(h, l, c)

    df = pd.concat([df, macd_features(c)], axis=1)
    df = pd.concat([df, stochastic(h, l, c)], axis=1)

    df["rolling_skew_50"]  = rolling_skew(c, 50)
    df["rolling_kurt_50"]  = rolling_kurt(c, 50)

    return df