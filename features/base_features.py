"""
base_features.py
================
Fundamental price-action features derived from raw OHLCV candles.

Avvalgi versiyadan farqi:
    1. candle_body, upper_wick, lower_wick — raw dollar emas,
       ATR bilan normalize qilingan (stationarity ta'minlandi).
    2. Barcha featurelar price-level agnostic:
       BTC $10k da va $100k da bir xil scale da chiqadi.

Featurelar:
    log_return          — bir barlik log daromad
    log_return_4b       — 4 barlik kumulativ log daromad
    candle_body_atr     — (Close - Open) / ATR
    upper_wick_atr      — upper wick / ATR
    lower_wick_atr      — lower wick / ATR
    upper_wick_ratio    — upper wick / (High - Low)
    lower_wick_ratio    — lower wick / (High - Low)
    body_ratio          — |Close - Open| / (High - Low)
    relative_spread     — (High - Low) / ATR
    rolling_vol_20      — 20 barlik rolling volatilite
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# YORDAMCHI
# ──────────────────────────────────────────────

def _wilder_atr(high: pd.Series,
                low: pd.Series,
                close: pd.Series,
                period: int = 14,
                eps: float = 1e-9) -> pd.Series:
    """
    Wilder ATR — normalize qilish uchun asosiy o'lchov birligi.

    Nima uchun ATR bilan normalize qilamiz?
        BTC 2019 da $10,000, 2024 da $70,000.
        Raw candle_body = Close - Open:
            2019: $50 harakat
            2024: $350 harakat
        ATR bilan normalize qilsak — ikkalasi ~0.5 chiqadi.
        Model 2019 va 2024 datani bir xil tushunadi.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr.clip(lower=eps)


# ──────────────────────────────────────────────
# 1. LOG RETURNS
# ──────────────────────────────────────────────

def log_return(close: pd.Series) -> pd.Series:
    """
    r_t = ln(P_t) - ln(P_{t-1})

    Nima uchun log return?
        • Price-level agnostic: $10k va $100k da comparable.
        • Additive: multi-bar return = barlar yig'indisi.
        • Normal taqsimotga yaqin → gradient yaxshi ishlaydi.
    """
    return np.log(close).diff().rename("log_return")


def rolling_log_return(close: pd.Series, window: int = 4) -> pd.Series:
    """
    window barlik kumulativ log return.
    Qisqa muddatli directional drift ni ushlaydi.
    """
    lr = np.log(close).diff()
    return lr.rolling(window).sum().rename(f"log_return_{window}b")


# ──────────────────────────────────────────────
# 2. CANDLE ANATOMY — ATR bilan normalize
# ──────────────────────────────────────────────

def candle_body_atr(open_: pd.Series,
                    close: pd.Series,
                    atr: pd.Series) -> pd.Series:
    """
    (Close - Open) / ATR

    Musbat → bullish bar; Manfiy → bearish bar.
    ATR bilan normalize → price-level agnostic.

    |value| > 1  → kuchli momentum (body ATR dan katta)
    |value| < 0.3 → doji / indecision
    """
    return ((close - open_) / atr).rename("candle_body_atr")


def upper_wick_atr(high: pd.Series,
                   open_: pd.Series,
                   close: pd.Series,
                   atr: pd.Series) -> pd.Series:
    """
    (High - max(Open, Close)) / ATR

    Katta qiymat → kuchli rejection yuqoridan (supply zone).
    Shooting star, bearish pin bar shu orqali aniqlanadi.
    """
    uw = high - np.maximum(open_, close)
    return (uw / atr).rename("upper_wick_atr")


def lower_wick_atr(low: pd.Series,
                   open_: pd.Series,
                   close: pd.Series,
                   atr: pd.Series) -> pd.Series:
    """
    (min(Open, Close) - Low) / ATR

    Katta qiymat → kuchli rejection pastdan (demand zone).
    Hammer, bullish pin bar shu orqali aniqlanadi.
    """
    lw = np.minimum(open_, close) - low
    return (lw / atr).rename("lower_wick_atr")


# ──────────────────────────────────────────────
# 3. RANGE RATIOS — ichki nisbatlar
# ──────────────────────────────────────────────

def upper_wick_ratio(high: pd.Series,
                     low: pd.Series,
                     open_: pd.Series,
                     close: pd.Series,
                     eps: float = 1e-9) -> pd.Series:
    """
    upper_wick / (High - Low)  ∈ [0, 1]

    Candle range ichida rejection qancha ulushni egallaydi?
    1 ga yaqin → deyarli butun candle rejection wick.
    """
    uw  = high - np.maximum(open_, close)
    rng = (high - low).clip(lower=eps)
    return (uw / rng).rename("upper_wick_ratio")


def lower_wick_ratio(high: pd.Series,
                     low: pd.Series,
                     open_: pd.Series,
                     close: pd.Series,
                     eps: float = 1e-9) -> pd.Series:
    """
    lower_wick / (High - Low)  ∈ [0, 1]
    """
    lw  = np.minimum(open_, close) - low
    rng = (high - low).clip(lower=eps)
    return (lw / rng).rename("lower_wick_ratio")


def body_ratio(high: pd.Series,
               low: pd.Series,
               open_: pd.Series,
               close: pd.Series,
               eps: float = 1e-9) -> pd.Series:
    """
    |Close - Open| / (High - Low)  ∈ [0, 1]

    Candle ning qanchalik "decisive" ekanligi.
    1 → full-body momentum candle (wick yo'q)
    0 → doji (butunlay indecision)
    """
    body = np.abs(close - open_)
    rng  = (high - low).clip(lower=eps)
    return (body / rng).rename("body_ratio")


# ──────────────────────────────────────────────
# 4. VOLATILITE
# ──────────────────────────────────────────────

def relative_spread(high: pd.Series,
                    low: pd.Series,
                    atr: pd.Series) -> pd.Series:
    """
    (High - Low) / ATR

    Joriy candle range o'rtacha range ga nisbatan qancha?
    > 2.0 → breakout / kengayish bari
    < 0.5 → compressed / inside bar
    """
    rng = high - low
    return (rng / atr).rename("relative_spread")


def rolling_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling std of log returns — bozor "harorat o'lchagichi".

    Yuqori → shovqinli / trend rejim
    Past   → coiling / breakout oldidan
    """
    lr = np.log(close).diff()
    return lr.rolling(window).std().rename(f"rolling_vol_{window}")


# ──────────────────────────────────────────────
# 5. MASTER BUILDER
# ──────────────────────────────────────────────

def add_base_features(df: pd.DataFrame,
                      atr_period: int = 14,
                      vol_window: int = 20) -> pd.DataFrame:
    """
    Barcha base featurelarni df ga qo'shadi.

    Kutilgan ustunlar: open, high, low, close, volume
    Qaytaradi: yangi ustunlar qo'shilgan df.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # ATR — normalize qilish uchun asos
    atr = _wilder_atr(h, l, c, atr_period)

    # Log returns
    df["log_return"]     = log_return(c)
    df["log_return_4b"]  = rolling_log_return(c, 4)

    # Candle anatomy — ATR normalize (stationarity ✅)
    df["candle_body_atr"]  = candle_body_atr(o, c, atr)
    df["upper_wick_atr"]   = upper_wick_atr(h, o, c, atr)
    df["lower_wick_atr"]   = lower_wick_atr(l, o, c, atr)

    # Range ratios — o'z ichida normalize (stationarity ✅)
    df["upper_wick_ratio"] = upper_wick_ratio(h, l, o, c)
    df["lower_wick_ratio"] = lower_wick_ratio(h, l, o, c)
    df["body_ratio"]       = body_ratio(h, l, o, c)

    # Volatilite
    df["relative_spread"]       = relative_spread(h, l, atr)
    df[f"rolling_vol_{vol_window}"] = rolling_volatility(c, vol_window)

    return df