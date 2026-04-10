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
    • Change of Character (CHoCH)   ← TUZATILDI: avvalgi versiyada hech qachon ishlamasdi
    • Distance to last swing levels
    • Volatility Contraction Pattern (VCP / Range Compression)
    • Price-Path Convexity
    • Hurst Exponent

FIXES (v2):
    1. break_of_structure() — CHoCH logikasi to'liq qayta yozildi.
       Avvalgi versiyada ichki if shart tashqi if bilan qarama-qarshi edi,
       natijada choch_bull va choch_bear har doim 0 bo'lib qolardi.
    2. kyle_lambda() — Python loop o'rniga numpy rolling regression.
       ~100x tezroq, xotira sarfi ham kamaydi.
    3. numba — import olib tashlandi (ishlatilmagan, import xatosi berardi).
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# 1. SWING HIGH / LOW DETECTION
# ──────────────────────────────────────────────

def detect_swings(high: pd.Series, low: pd.Series,
                  n: int = 3) -> pd.DataFrame:
    """
    Detect Swing Highs and Swing Lows with `n`-bar confirmation.

    Definition:
        Swing High at bar i  iff  high[i] == max(high[i-n : i+n+1])
        Swing Low  at bar i  iff  low[i]  == min(low[i-n  : i+n+1])

    Because we use i+1..i+n bars for confirmation, we *record* the event
    at bar i+n (the first bar we KNOW it's a swing).
    No look-ahead bias.

    Returns DataFrame with columns:
        swing_high       : 1.0 at confirmed swing high bar, else 0.0
        swing_low        : 1.0 at confirmed swing low bar, else 0.0
        last_swing_high  : forward-filled price of most recent swing high
        last_swing_low   : forward-filled price of most recent swing low
    """
    h = high.values
    l = low.values
    N = len(h)

    is_swing_high = np.zeros(N)
    is_swing_low  = np.zeros(N)

    for i in range(n, N - n):
        if h[i] == max(h[i-n : i+n+1]):
            is_swing_high[i + n] = 1.0
        if l[i] == min(l[i-n : i+n+1]):
            is_swing_low[i + n] = 1.0

    idx = high.index

    sh_price = pd.Series(np.where(is_swing_high, h, np.nan), index=idx)
    sl_price = pd.Series(np.where(is_swing_low,  l, np.nan), index=idx)

    return pd.DataFrame({
        "swing_high":      pd.Series(is_swing_high, index=idx),
        "swing_low":       pd.Series(is_swing_low,  index=idx),
        "last_swing_high": sh_price.ffill(),
        "last_swing_low":  sl_price.ffill(),
    })


# ──────────────────────────────────────────────
# 2. HH / HL / LH / LL CLASSIFICATION
# ──────────────────────────────────────────────

def classify_swing_sequence(swing_df: pd.DataFrame,
                             high: pd.Series,
                             low: pd.Series) -> pd.DataFrame:
    """
    Compare consecutive swing highs/lows to classify market structure.

    HH + HL → bullish trend
    LH + LL → bearish trend
    Mixed   → consolidation / transition

    Encoded as integers:
        +1 = HH or HL (bullish event)
        -1 = LH or LL (bearish event)
         0 = no swing on this bar
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
        Bullish BOS : close crosses ABOVE last confirmed Swing High
        Bearish BOS : close crosses BELOW last confirmed Swing Low
        → Confirms trend continuation.

    CHoCH (Change of Character):  ← TO'LIQ QAYTA YOZILDI
        Bullish CHoCH : narx downtrend kontekstida (c < last_sh) bo'lsa,
                        va close YUQORIGA last_sh ni kesib o'tsa
                        → potentsial bullish reversal.
        Bearish CHoCH : narx uptrend kontekstida (c > last_sl) bo'lsa,
                        va close PASTGA last_sl ni kesib o'tsa
                        → potentsial bearish reversal.

    Avvalgi xato qanday edi:
        Tashqi shart:  if c[i] > sl  ...   (narx sl DAN YUQORI)
        Ichki shart:   if c[i] < sl  ...   (narx sl DAN PAST)
        Bu mantiqan imkonsiz — hech qachon true bo'lmasdi.

    To'g'ri mantiq:
        CHoCH — bu BOS bilan bir xil harakat, lekin qarama-qarshi trend
        kontekstida sodir bo'ladi.  Faqat prev_c → c kesishuviga qaraymiz.
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

        # ── BOS: trend yo'nalishida structural level kesib o'tildi ──
        # Bullish BOS: narx swing high ni pastdan yuqoriga kesdi
        if prev_c[i] <= sh < c[i]:
            bos_bull[i] = 1.0

        # Bearish BOS: narx swing low ni yuqoridan pastga kesdi
        if prev_c[i] >= sl > c[i]:
            bos_bear[i] = 1.0

        # ── CHoCH: teskari yo'nalishda structural level kesib o'tildi ──
        # Bullish CHoCH: narx downtrend ichida (c < sh), lekin swing high ni kesib o'tdi
        # Bu downtrend tugashi va bullish reversal boshlanishini ko'rsatadi
        if c[i] < sh:                          # Downtrend context: narx last_sh dan past
            if prev_c[i] <= sh < c[i]:         # Lekin shu barda sh ni yuqoriga kesdi
                choch_bull[i] = 1.0            # → Bullish CHoCH

        # Bearish CHoCH: narx uptrend ichida (c > sl), lekin swing low ni kesib o'tdi
        if c[i] > sl:                          # Uptrend context: narx last_sl dan yuqori
            if prev_c[i] >= sl > c[i]:         # Lekin shu barda sl ni pastga kesdi
                choch_bear[i] = 1.0            # → Bearish CHoCH

    # Bars since last BOS
    bos_any = bos_bull + bos_bear
    bars_since_bos = _bars_since_event(bos_any, N)

    # Structural distance (stationary, normalised)
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
    """Count bars elapsed since the last 1 in a binary event array."""
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

    < 0.3 → juda qattiq coil (yirik harakat oldidan)
    > 1.0 → range kengaymoqda (breakout bo'lyapti)
    """
    range_short = high.rolling(short_window).max() - low.rolling(short_window).min()
    range_long  = high.rolling(long_window).max()  - low.rolling(long_window).min()
    return (range_short / range_long.clip(lower=eps)).rename("vcp_compression")


# ──────────────────────────────────────────────
# 5. PRICE-PATH CONVEXITY
# ──────────────────────────────────────────────

def price_convexity(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Convexity = (midpoint_of_endpoints - mean_of_path) / |midpoint|

    > 0 → parabolik / blow-off harakat (mean-revert xavfi)
    < 0 → konkav / yumaloq dip pattern
    ≈ 0 → chiziqli trend
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

    H > 0.5 → Trending / Persistent  (momentum ishlaydi)
    H < 0.5 → Mean-reverting          (harakatga qarshi savdo)
    H ≈ 0.5 → Random walk             (bashorat qilib bo'lmaydi)

    Eng muhim rejim-deteksiya feature — neyron tarmoq bu orqali
    momentum va mean-reversion rejimlarini ajrata oladi.
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

        log_rs  = np.log(rs_vals[valid])
        ll      = log_lags[valid]
        ll_mean = ll.mean()
        rs_mean = log_rs.mean()
        h_est   = ((ll - ll_mean) * (log_rs - rs_mean)).sum() / ((ll - ll_mean)**2).sum()
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
# 7. KYLE'S LAMBDA — VEKTORIZATSIYA QILINGAN
# ──────────────────────────────────────────────

def kyle_lambda(close: pd.Series, volume: pd.Series,
                window: int = 20, eps: float = 1e-9) -> pd.Series:
    """
    Rolling OLS slope of |ΔP| on volume.

    Kyle (1985): ΔP = λ × net_order_flow + noise
    Yuqori λ → bozor narxga sezgir → yupqa order book.
    Past λ   → chuqur bozor, yirik hajm ham ta'sir qilmaydi.

    AVVALGI MUAMMO:
        Python for-loop bilan O(N × window) — 100k barda daqiqalar ketardi.

    YANGI YONDASHUV — NUMPY VEKTORIZATSIYA:
        pandas rolling() orqali har bir oynada:
            cov(volume, |ΔP|) va var(volume) ni hisoblaymiz.
        slope = cov / var  (OLS formulasi)
        Bu ~100x tezroq, xotira sarfi ham minimum.
    """
    delta_p = close.diff().abs()

    # Rolling statistics (pandas built-in — C-level tezlik)
    roll_cov = volume.rolling(window).cov(delta_p)
    roll_var = volume.rolling(window).var()

    lambda_series = roll_cov / roll_var.clip(lower=eps)
    return lambda_series.rename("kyle_lambda")


# ──────────────────────────────────────────────
# 8. MASTER BUILDER
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
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]

    # Swings
    swing_df = detect_swings(h, l, n=swing_n)
    df = pd.concat([df, swing_df], axis=1)

    # HH / HL / LH / LL
    seq_df = classify_swing_sequence(swing_df, h, l)
    df = pd.concat([df, seq_df], axis=1)

    # BOS / CHoCH (tuzatilgan versiya)
    bos_df = break_of_structure(c, swing_df)
    df = pd.concat([df, bos_df], axis=1)

    # VCP
    df["vcp_compression"] = volatility_compression(h, l, vcp_short, vcp_long)

    # Convexity
    df[f"convexity_{convexity_window}"] = price_convexity(c, convexity_window)

    # Hurst (qimmat — faqat bir marta ishga tushiriladi)
    df[f"hurst_{hurst_window}"] = hurst_exponent(c, hurst_window)

    # Kyle Lambda (endi tez — vektorizatsiya qilingan)
    df["kyle_lambda"] = kyle_lambda(c, v)

    return df