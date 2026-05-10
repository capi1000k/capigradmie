"""
time_features.py
================
Temporal / calendar featurelar trading modeli uchun.

Bozorlar vaqt bo'yicha bir xil emas:
    00:00–08:00 UTC — Asian sessiya (BTC: OKX, Binance Asia)
    07:00–16:00 UTC — London sessiya
    13:00–17:00 UTC — London + NY overlap (eng yuqori volatilite)
    13:30–21:00 UTC — NY sessiya
    Dushanba ochilik / Juma yopilish — hafta sonu gap xavfi
    Oy oxiri — institutional rebalancing

Barcha featurelar:
    1. Siklik (sin/cos) — 23:00 va 01:00 o'rtasidagi masofa 2 soat,
       22 soat emas.
    2. Normalize qilingan [-1, +1].

Avvalgi versiyadan farq: o'zgarishsiz — yaxshi yozilgan edi.
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# YORDAMCHI
# ──────────────────────────────────────────────

def _sin_cos(value: np.ndarray, period: float):
    """Siklik o'zgaruvchini sin/cos orqali kodlash."""
    angle = 2 * np.pi * value / period
    return np.sin(angle), np.cos(angle)


# ──────────────────────────────────────────────
# 1. KUN SOATI
# ──────────────────────────────────────────────

def hour_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Kun soatini (0-23) sin/cos kodlash.

    Nima uchun sin/cos?
        23:00 va 00:00 "yaqin" bo'lishi kerak.
        Linear (23 vs 0) — noto'g'ri farq. Sin/cos — to'g'ri.
    """
    hour = index.hour.astype(float)
    sin_h, cos_h = _sin_cos(hour, 24.0)
    return pd.DataFrame({
        "hour_sin": sin_h,
        "hour_cos": cos_h,
        "hour_raw": hour,
    }, index=index)


# ──────────────────────────────────────────────
# 2. HAFTA KUNI
# ──────────────────────────────────────────────

def dow_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Hafta kunini (0=Dush, 6=Yak) sin/cos kodlash.

    Dushanba ochilik: hafta sonu gap unwind → yuqori noaniqlik.
    Juma yopilish: pozitsiya yopish → yo'nalish bias.
    """
    dow = index.dayofweek.astype(float)
    sin_d, cos_d = _sin_cos(dow, 7.0)
    return pd.DataFrame({
        "dow_sin": sin_d,
        "dow_cos": cos_d,
        "dow_raw": dow,
    }, index=index)


# ──────────────────────────────────────────────
# 3. OY KUNI va YIL OYI
# ──────────────────────────────────────────────

def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Oy kuni (1-31) va yil oyi (1-12) sin/cos kodlash.

    Oy oxiri (27-31): institutional rebalancing, option expiry.
    Oy boshi (1-3): yangi kapital kiritish.
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
# 4. SESSIYA FLAGLARI
# ──────────────────────────────────────────────

def session_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Asosiy trading sessiyalar uchun binary flaglar (UTC).

        Asian  sessiya: 00:00–08:00
        London sessiya: 07:00–16:00
        NY     sessiya: 13:30–21:00
        Overlap       : 13:00–17:00  (eng yuqori volatilite)
        Hafta sonu    : Shanba, Yakshanba
        Dushanba ochilik: Dush 00:00–04:00
    """
    hour = index.hour

    return pd.DataFrame({
        "session_asian":   ((hour >= 0)  & (hour < 8)).astype(float),
        "session_london":  ((hour >= 7)  & (hour < 16)).astype(float),
        "session_ny":      ((hour >= 13) & (hour < 21)).astype(float),
        "session_overlap": ((hour >= 13) & (hour < 17)).astype(float),
        "is_weekend":      (index.dayofweek >= 5).astype(float),
        "is_monday_open":  ((index.dayofweek == 0) & (hour < 4)).astype(float),
    }, index=index)


# ──────────────────────────────────────────────
# 5. INTRADAY POZITSIYA
# ──────────────────────────────────────────────

def bars_since_midnight(index: pd.DatetimeIndex,
                         bar_minutes: int = 15) -> pd.Series:
    """
    00:00 UTC dan beri o'tgan barlar soni / kun boshidagi barlar soni.
    Natija [0, 1] oralig'ida.

    M15 uchun: max 96 bar/kun → divide by 96.
    H1  uchun: max 24 bar/kun → divide by 24.
    H4  uchun: max 6  bar/kun → divide by 6.
    """
    minutes_elapsed = index.hour * 60 + index.minute
    bars_elapsed    = minutes_elapsed / bar_minutes
    bars_per_day    = 1440 / bar_minutes
    return pd.Series(
        bars_elapsed / bars_per_day,
        index=index,
        name="intraday_position"
    )


# ──────────────────────────────────────────────
# MASTER BUILDER
# ──────────────────────────────────────────────

def add_time_features(df: pd.DataFrame,
                      bar_minutes: int = 15) -> pd.DataFrame:
    """
    Barcha vaqt featurelarni df ga qo'shadi.
    df.index DatetimeIndex (UTC) bo'lishi shart.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index DatetimeIndex bo'lishi kerak (UTC).")

    idx = df.index

    df = pd.concat([
        df,
        hour_features(idx),
        dow_features(idx),
        calendar_features(idx),
        session_flags(idx),
    ], axis=1)

    df["intraday_position"] = bars_since_midnight(idx, bar_minutes)

    return df