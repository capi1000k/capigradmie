"""
microstructure.py
=================
Institutional-grade market microstructure proxies — faqat OHLCV dan.

Level-2 order book yo'q, shuning uchun akademik adabiyotda
tasdiqlangan econometric proxylardan foydalanamiz.

Avvalgi versiyadan farqi:
    • volume_delta        — raw volume emas, rolling mean ga normalize
    • bvc_buy_vol/sell_vol — olib tashlandi (raw volume, non-stationary)
    • bvc_imbalance       — qoldirildi (o'zi normalized ∈ [-1, +1])
    • bvc_delta           — normalize qilingan versiyasi qo'shildi

Featurelar:
    roll_spread           — bid-ask spread proxy (Roll 1984)
    amihud_illiq          — price impact per dollar volume (Amihud 2002)
    volume_delta_norm     — signed volume / rolling mean volume
    volume_delta_ratio_14 — rolling net order flow ∈ [-1, +1]
    bvc_imbalance         — Bulk Volume Classification imbalance
    bvc_delta_norm        — normalize qilingan BVC delta
    effort_vs_result      — Wyckoff effort vs result (log)
    obv_zscore            — On-Balance Volume z-score
    kyle_lambda           — price impact slope (Kyle 1985)
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# 1. ROLL SPREAD
# ──────────────────────────────────────────────

def roll_spread(close: pd.Series,
                window: int = 20) -> pd.Series:
    """
    Roll (1984) effective spread estimator.

    s = 2 × sqrt(|Cov(ΔP_t, ΔP_{t-1})|) × sign(-Cov)

    Musbat → trending rejim
    Manfiy → mean-reverting / shovqinli rejim
    Kengaygan spread → liquidity krizi yoki volatilite spike oldidan
    """
    lr  = np.log(close).diff()
    cov = lr.rolling(window).cov(lr.shift(1))
    return (2 * np.sqrt(cov.abs()) * np.sign(-cov)).rename("roll_spread")


# ──────────────────────────────────────────────
# 2. AMIHUD ILLIQUIDITY
# ──────────────────────────────────────────────

def amihud_illiquidity(close: pd.Series,
                       volume: pd.Series,
                       window: int = 20,
                       eps: float = 1e-9) -> pd.Series:
    """
    Amihud (2002): ILLIQ = |R_t| / (Volume × Price)

    Dollar hajmiga nisbatan narx qanchalik harakat qiladi?
    Yuqori → yupqa bozor, slippage xavfi
    Past   → chuqur bozor, katta hajm ham ta'sir qilmaydi

    Log-transform: extreme qiymatlarni jilovlash uchun.
    """
    log_ret    = np.log(close).diff().abs()
    dollar_vol = (volume * close).clip(lower=eps)
    illiq_raw  = log_ret / dollar_vol
    illiq_sm   = illiq_raw.rolling(window).mean()
    return np.log1p(illiq_sm).rename("amihud_illiq")


# ──────────────────────────────────────────────
# 3. VOLUME DELTA — NORMALIZE QILINGAN
# ──────────────────────────────────────────────

def volume_delta_norm(close: pd.Series,
                      volume: pd.Series,
                      window: int = 20,
                      eps: float = 1e-9) -> pd.Series:
    """
    sign(ΔClose) × Volume / rolling_mean(Volume)

    Avvalgi versiya: sign × Volume → raw, non-stationary ❌
    Yangi versiya: rolling mean ga normalize → stationary ✅

    > 1  → o'rtacha dan yuqori buying pressure
    < -1 → o'rtacha dan yuqori selling pressure
    ≈ 0  → muvozanat
    """
    direction    = np.sign(close.diff())
    signed_vol   = direction * volume
    rolling_mean = volume.rolling(window).mean().clip(lower=eps)
    return (signed_vol / rolling_mean).rename("volume_delta_norm")


def volume_delta_ratio(close: pd.Series,
                       volume: pd.Series,
                       window: int = 14,
                       eps: float = 1e-9) -> pd.Series:
    """
    Rolling sum(signed_vol) / rolling sum(vol) ∈ [-1, +1]

    > +0.5 → kuchli buying dominance
    < -0.5 → kuchli selling dominance
    ≈  0   → muvozanat
    """
    signed_vol  = np.sign(close.diff()) * volume
    roll_signed = signed_vol.rolling(window).sum()
    roll_vol    = volume.rolling(window).sum().clip(lower=eps)
    return (roll_signed / roll_vol).rename(f"volume_delta_ratio_{window}")


# ──────────────────────────────────────────────
# 4. BULK VOLUME CLASSIFICATION (BVC)
# ──────────────────────────────────────────────

def bulk_volume_classification(close: pd.Series,
                                volume: pd.Series,
                                window: int = 14,
                                eps: float = 1e-9) -> pd.DataFrame:
    """
    Easley et al. Bulk Volume Classification.

        V_buy  = V × Φ(ΔP / σ_ΔP)
        V_sell = V - V_buy
        delta  = V_buy - V_sell

    Normal CDF orqali narx harakati kattaligiga qarab
    har bir barni buy/sell ga ajratadi.

    Qaytariladi:
        bvc_imbalance  ∈ [-1, +1]  — asosiy feature (stationary ✅)
        bvc_delta_norm             — delta / rolling_mean_vol (stationary ✅)
    """
    from scipy.special import ndtr

    delta_p = close.diff()
    sigma   = delta_p.rolling(window).std().clip(lower=eps)
    phi     = pd.Series(ndtr(delta_p / sigma), index=close.index)

    v_buy  = volume * phi
    v_sell = volume * (1 - phi)
    delta  = v_buy - v_sell

    imbalance = delta / volume.clip(lower=eps)

    # delta normalize: rolling mean volume ga bo'lamiz
    roll_mean_vol = volume.rolling(window).mean().clip(lower=eps)
    delta_norm    = delta / roll_mean_vol

    return pd.DataFrame({
        "bvc_imbalance":  imbalance,
        "bvc_delta_norm": delta_norm,
    }, index=close.index)


# ──────────────────────────────────────────────
# 5. EFFORT vs RESULT
# ──────────────────────────────────────────────

def effort_vs_result(close: pd.Series,
                     volume: pd.Series,
                     eps: float = 1e-9) -> pd.Series:
    """
    log(Volume / |ΔClose|)

    Wyckoff prinsipi: hajm (effort) vs narx harakati (result).
    Yuqori → katta hajm kam harakat → qarshilik / absorption
    Past   → kichik hajm katta harakat → thin market / FOMO

    Log-transform: ratio ko'p tartibga farq qilishi mumkin.
    """
    price_change = close.diff().abs().clip(lower=eps)
    evr = volume / price_change
    return np.log1p(evr).rename("effort_vs_result")


# ──────────────────────────────────────────────
# 6. OBV Z-SCORE
# ──────────────────────────────────────────────

def on_balance_volume(close: pd.Series,
                      volume: pd.Series) -> pd.Series:
    """
    OBV z-score — rolling normalized.

    Raw OBV kumulativ → non-stationary.
    Z-score: (OBV - rolling_mean) / rolling_std → stationary ✅

    OBV narxdan oldinda o'ssa → institutional accumulation
    OBV narxdan orqada qolsa → distribution (sotish)
    """
    direction = np.sign(close.diff()).fillna(0)
    obv       = (direction * volume).cumsum()
    obv_mean  = obv.rolling(50).mean()
    obv_std   = obv.rolling(50).std().clip(lower=1e-9)
    return ((obv - obv_mean) / obv_std).rename("obv_zscore")


# ──────────────────────────────────────────────
# 7. KYLE'S LAMBDA
# ──────────────────────────────────────────────

def kyle_lambda(close: pd.Series,
                volume: pd.Series,
                window: int = 20,
                eps: float = 1e-9) -> pd.Series:
    """
    Rolling OLS slope: |ΔP| ~ Volume

    Kyle (1985): ΔP = λ × net_order_flow + noise
    Yuqori λ → narx hajmga sezgir → yupqa order book
    Past λ   → chuqur bozor

    Vektorizatsiya: pandas rolling cov/var orqali (~100x tezroq loop dan).
    """
    delta_p  = close.diff().abs()
    roll_cov = volume.rolling(window).cov(delta_p)
    roll_var = volume.rolling(window).var()
    return (roll_cov / roll_var.clip(lower=eps)).rename("kyle_lambda")


# ──────────────────────────────────────────────
# MASTER BUILDER
# ──────────────────────────────────────────────

def add_microstructure_features(df: pd.DataFrame,
                                 use_bvc: bool = True) -> pd.DataFrame:
    """
    Barcha microstructure featurelarni df ga qo'shadi.
    Kutilgan ustunlar: open, high, low, close, volume
    """
    c, v = df["close"], df["volume"]

    df["roll_spread"]           = roll_spread(c)
    df["amihud_illiq"]          = amihud_illiquidity(c, v)
    df["volume_delta_norm"]     = volume_delta_norm(c, v)
    df["volume_delta_ratio_14"] = volume_delta_ratio(c, v, 14)
    df["effort_vs_result"]      = effort_vs_result(c, v)
    df["obv_zscore"]            = on_balance_volume(c, v)
    df["kyle_lambda"]           = kyle_lambda(c, v)

    if use_bvc:
        try:
            bvc_df = bulk_volume_classification(c, v)
            df = pd.concat([df, bvc_df], axis=1)
        except ImportError:
            # scipy yo'q — oddiy imbalance bilan almashtiramiz
            df["bvc_imbalance"]  = volume_delta_ratio(c, v)
            df["bvc_delta_norm"] = volume_delta_norm(c, v)

    return df