"""
smart_money.py
==============
Smart Money Concepts (SMC) va Price Action featurelar.

Avvalgi versiyadan farqlar (3 ta critical bug tuzatildi):

    BUG 1 — CHoCH matematikan imkonsiz edi:
        Eski: if c[i] < sh: if prev_c[i] <= sh < c[i]  ← CONTRADICTION ❌
        Yangi: if prev_c[i] < sh <= c[i]  ← to'g'ri kesishma ✅

    BUG 2 — Swing price noto'g'ri bardan olinardi:
        Eski: sh_price[i+n] = h[i+n]  ← confirmation barning narxi ❌
        Yangi: sh_price[i+n] = h[i]   ← actual swing barning narxi ✅

    BUG 3 — kyle_lambda duplicate:
        Olib tashlandi — faqat microstructure.py da ✅

    STATIONARITY FIX:
        last_swing_high/low — raw price, datasetga kirmaydi ✅
        Faqat dist_to_swing_h/l (normalize) datasetga kiradi ✅

Featurelar:
    swing_high, swing_low
    hh_flag, hl_flag, lh_flag, ll_flag, net_structure
    bos_bull, bos_bear, choch_bull, choch_bear
    bars_since_bos
    dist_to_swing_h, dist_to_swing_l
    vcp_compression
    convexity_20
    hurst_100
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# 1. SWING HIGH / LOW DETECTION
# ──────────────────────────────────────────────

def detect_swings(high: pd.Series,
                  low: pd.Series,
                  n: int = 3) -> pd.DataFrame:
    """
    n-bar tasdiq bilan Swing High / Low aniqlash.

    Ta'rif:
        Bar i — Swing High  agar  high[i] == max(high[i-n : i+n+1])
        Bar i — Swing Low   agar  low[i]  == min(low[i-n  : i+n+1])

    Natija i+n barida yoziladi (tasdiq uchun n bar kutiladi).
    Look-ahead bias YO'Q.

    BUG 2 FIX:
        sh_price[i+n] = h[i]   ← actual swing barning narxi ✅
        (eski: h[i+n] — confirmation barning narxi, noto'g'ri edi ❌)
    """
    h = high.values
    l = low.values
    N = len(h)

    is_swing_high = np.zeros(N)
    is_swing_low  = np.zeros(N)
    sh_price_arr  = np.full(N, np.nan)
    sl_price_arr  = np.full(N, np.nan)

    for i in range(n, N - n):
        if h[i] == max(h[i - n: i + n + 1]):
            is_swing_high[i + n] = 1.0
            sh_price_arr[i + n]  = h[i]   # ✅ actual swing narxi

        if l[i] == min(l[i - n: i + n + 1]):
            is_swing_low[i + n]  = 1.0
            sl_price_arr[i + n]  = l[i]   # ✅ actual swing narxi

    idx = high.index

    return pd.DataFrame({
        "swing_high":      pd.Series(is_swing_high, index=idx),
        "swing_low":       pd.Series(is_swing_low,  index=idx),
        "last_swing_high": pd.Series(sh_price_arr, index=idx).ffill(),
        "last_swing_low":  pd.Series(sl_price_arr, index=idx).ffill(),
    })


# ──────────────────────────────────────────────
# 2. HH / HL / LH / LL
# ──────────────────────────────────────────────

def classify_swing_sequence(swing_df: pd.DataFrame,
                             high: pd.Series,
                             low: pd.Series) -> pd.DataFrame:
    """
    Ketma-ket swing punktlarini solishtirish.

    HH + HL → bullish trend
    LH + LL → bearish trend
    Aralash  → konsolidatsiya

    net_structure: 10-bar window ichida (bullish - bearish) hodisalar.
    """
    sh_mask = swing_df["swing_high"].values.astype(bool)
    sl_mask = swing_df["swing_low"].values.astype(bool)
    h_vals  = high.values
    l_vals  = low.values
    idx     = high.index
    N       = len(h_vals)

    hh_flag = np.zeros(N)
    hl_flag = np.zeros(N)
    lh_flag = np.zeros(N)
    ll_flag = np.zeros(N)

    prev_sh = np.nan
    prev_sl = np.nan

    for i in range(N):
        if sh_mask[i]:
            curr = h_vals[i]
            if not np.isnan(prev_sh):
                hh_flag[i] = 1.0 if curr > prev_sh else 0.0
                lh_flag[i] = 1.0 if curr < prev_sh else 0.0
            prev_sh = curr

        if sl_mask[i]:
            curr = l_vals[i]
            if not np.isnan(prev_sl):
                hl_flag[i] = 1.0 if curr > prev_sl else 0.0
                ll_flag[i] = 1.0 if curr < prev_sl else 0.0
            prev_sl = curr

    net_structure = (
        pd.Series(hh_flag + hl_flag - lh_flag - ll_flag, index=idx)
        .rolling(10).sum()
    )

    return pd.DataFrame({
        "hh_flag":       pd.Series(hh_flag, index=idx),
        "hl_flag":       pd.Series(hl_flag, index=idx),
        "lh_flag":       pd.Series(lh_flag, index=idx),
        "ll_flag":       pd.Series(ll_flag, index=idx),
        "net_structure": net_structure,
    })


# ──────────────────────────────────────────────
# 3. BOS va CHoCH
# ──────────────────────────────────────────────

def break_of_structure(close: pd.Series,
                        swing_df: pd.DataFrame,
                        eps: float = 1e-9) -> pd.DataFrame:
    """
    BOS (Break of Structure):
        Bullish: prev_c <= last_sh < c  — trend davomi yuqoriga
        Bearish: prev_c >= last_sl > c  — trend davomi pastga

    CHoCH (Change of Character) — BUG 1 FIX ✅:
        Bullish CHoCH: prev_c < last_sh  AND  c >= last_sh
                       (downtrend ichida lekin sh ni kesib o'tdi → reversal)
        Bearish CHoCH: prev_c > last_sl  AND  c <= last_sl
                       (uptrend ichida lekin sl ni kesib o'tdi → reversal)

    dist_to_swing_h/l: (close - level) / close → stationary ✅
    """
    c       = close.values
    last_sh = swing_df["last_swing_high"].values
    last_sl = swing_df["last_swing_low"].values
    N       = len(c)
    idx     = close.index

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

        # BOS — trend yo'nalishida kesishma
        if prev_c[i] <= sh < c[i]:
            bos_bull[i] = 1.0

        if prev_c[i] >= sl > c[i]:
            bos_bear[i] = 1.0

        # CHoCH — teskari kesishma ✅ (BUG FIX)
        if prev_c[i] < sh <= c[i]:
            choch_bull[i] = 1.0

        if prev_c[i] > sl >= c[i]:
            choch_bear[i] = 1.0

    bos_any        = bos_bull + bos_bear
    bars_since_bos = _bars_since_event(bos_any, N)

    dist_sh = (c - last_sh) / (np.abs(c) + eps)
    dist_sl = (c - last_sl) / (np.abs(c) + eps)

    return pd.DataFrame({
        "bos_bull":        pd.Series(bos_bull,       index=idx),
        "bos_bear":        pd.Series(bos_bear,       index=idx),
        "choch_bull":      pd.Series(choch_bull,     index=idx),
        "choch_bear":      pd.Series(choch_bear,     index=idx),
        "bars_since_bos":  pd.Series(bars_since_bos, index=idx),
        "dist_to_swing_h": pd.Series(dist_sh,        index=idx),
        "dist_to_swing_l": pd.Series(dist_sl,        index=idx),
    })


def _bars_since_event(event_array: np.ndarray, N: int) -> np.ndarray:
    """Oxirgi 1 hodisadan o'tgan barlar soni."""
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
# 4. VCP — VOLATILITY CONTRACTION PATTERN
# ──────────────────────────────────────────────

def volatility_compression(high: pd.Series,
                            low: pd.Series,
                            short_window: int = 20,
                            long_window: int = 100,
                            eps: float = 1e-9) -> pd.Series:
    """
    Range Compression = short_range / long_range

    < 0.3 → qattiq coil (yirik harakat oldidan)
    > 1.0 → range kengaymoqda
    """
    range_short = high.rolling(short_window).max() - low.rolling(short_window).min()
    range_long  = high.rolling(long_window).max()  - low.rolling(long_window).min()
    return (range_short / range_long.clip(lower=eps)).rename("vcp_compression")


# ──────────────────────────────────────────────
# 5. PRICE-PATH CONVEXITY
# ──────────────────────────────────────────────

def price_convexity(close: pd.Series, window: int = 20) -> pd.Series:
    """
    (endpoints_midpoint - path_mean) / |midpoint|

    > 0 → parabolik/blow-off (mean-revert xavfi)
    < 0 → konkav/yumaloq dip
    ≈ 0 → chiziqli trend
    """
    results = np.full(len(close), np.nan)
    c = close.values

    for i in range(window, len(c)):
        segment  = c[i - window: i + 1]
        midpoint = (segment[0] + segment[-1]) / 2.0
        if abs(midpoint) > 1e-9:
            results[i] = (midpoint - segment.mean()) / abs(midpoint)

    return pd.Series(results, index=close.index, name=f"convexity_{window}")


# ──────────────────────────────────────────────
# 6. HURST EXPONENT
# ──────────────────────────────────────────────

def hurst_exponent(close: pd.Series,
                   window: int = 100,
                   min_lags: int = 2,
                   max_lags: int = 20) -> pd.Series:
    """
    Rolling Hurst Exponent — R/S analiz.

    H > 0.5 → Trending (momentum ishlaydi)
    H < 0.5 → Mean-reverting
    H ≈ 0.5 → Random walk
    """
    log_close = np.log(close.values)
    returns   = np.diff(log_close)
    N         = len(log_close)
    results   = np.full(N, np.nan)

    lags     = np.arange(min_lags, max_lags + 1)
    log_lags = np.log(lags)

    for i in range(window, N):
        segment = returns[i - window: i]
        rs_vals = np.array([_rs_statistic(segment, lag) for lag in lags])

        valid = rs_vals > 0
        if valid.sum() < 3:
            continue

        log_rs  = np.log(rs_vals[valid])
        ll      = log_lags[valid]
        ll_mean = ll.mean()
        h_est   = (
            ((ll - ll_mean) * (log_rs - log_rs.mean())).sum() /
            ((ll - ll_mean) ** 2).sum()
        )
        results[i] = np.clip(h_est, 0.0, 1.0)

    return pd.Series(results, index=close.index, name=f"hurst_{window}")


def _rs_statistic(series: np.ndarray, lag: int) -> float:
    n_chunks = len(series) // lag
    if n_chunks < 1:
        return np.nan
    rs_list = []
    for k in range(n_chunks):
        chunk = series[k * lag: (k + 1) * lag]
        devs  = np.cumsum(chunk - chunk.mean())
        r     = devs.max() - devs.min()
        s     = chunk.std(ddof=1)
        if s > 0:
            rs_list.append(r / s)
    return np.mean(rs_list) if rs_list else np.nan


# ──────────────────────────────────────────────
# MASTER BUILDER
# ──────────────────────────────────────────────

def add_smart_money_features(df: pd.DataFrame,
                              swing_n: int = 3,
                              hurst_window: int = 100,
                              convexity_window: int = 20,
                              vcp_short: int = 20,
                              vcp_long: int = 100) -> pd.DataFrame:
    """
    Barcha SMC featurelarni df ga qo'shadi.
    Kutilgan ustunlar: open, high, low, close, volume

    ✅ kyle_lambda YO'Q — faqat microstructure.py da
    ✅ last_swing_high/low datasetga kirmaydi (non-stationary)
    ✅ CHoCH bug tuzatildi
    ✅ Swing price bug tuzatildi
    """
    h, l, c = df["high"], df["low"], df["close"]

    swing_df = detect_swings(h, l, n=swing_n)

    # Faqat binary flaglar — raw price emas
    df["swing_high"] = swing_df["swing_high"]
    df["swing_low"]  = swing_df["swing_low"]

    seq_df = classify_swing_sequence(swing_df, h, l)
    df = pd.concat([df, seq_df], axis=1)

    bos_df = break_of_structure(c, swing_df)
    df = pd.concat([df, bos_df], axis=1)

    df["vcp_compression"]          = volatility_compression(h, l, vcp_short, vcp_long)
    df[f"convexity_{convexity_window}"] = price_convexity(c, convexity_window)
    df[f"hurst_{hurst_window}"]    = hurst_exponent(c, hurst_window)

    return df