"""
microstructure.py
=================
Institutional-grade market microstructure proxies derived *solely* from OHLCV.

Without Level-2 order book data we cannot directly observe bid-ask spreads,
order-flow imbalance, or market depth.  The features in this module are
econometric proxies that approximate those quantities using price and volume —
validated in academic literature and used by systematic hedge funds.

References embedded in docstrings below.
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# 1. ROLL SPREAD ESTIMATOR
# ──────────────────────────────────────────────

def roll_spread(close: pd.Series,
                window: int = 20,
                eps: float = 1e-9) -> pd.Series:
    """
    Roll (1984) effective spread estimator.

    Theory:  In an efficient market, the bid-ask bounce causes *negative*
    serial autocorrelation in price changes.  Roll showed:

        s = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))

    Why it matters:
        • Proxy for transaction costs / liquidity conditions.
        • A widening Roll spread often precedes a liquidity crisis or
          high-volatility spike event.
        • Useful as a regime filter: tight spread = liquid & healthy;
          wide spread = fragile, slippage risk.

    Implementation note:
        When the covariance is positive (trending, not mean-reverting),
        the standard formula is undefined (sqrt of negative).  We use
        the "Absolute Roll" modification: take |cov| and preserve the sign
        so the feature stays real-valued.  Positive = trending regime;
        negative = mean-reverting / noisy regime.
    """
    lr = np.log(close).diff()
    cov = lr.rolling(window).cov(lr.shift(1))

    # Absolute Roll: 2*sqrt(|cov|) * sign(-cov)
    spread = 2 * np.sqrt(cov.abs()) * np.sign(-cov)
    return spread.rename("roll_spread")


# ──────────────────────────────────────────────
# 2. AMIHUD ILLIQUIDITY
# ──────────────────────────────────────────────

def amihud_illiquidity(close: pd.Series,
                       volume: pd.Series,
                       window: int = 20,
                       eps: float = 1e-9) -> pd.Series:
    """
    Amihud (2002) illiquidity ratio.

        ILLIQ_t = |R_t| / (Volume_t × Price_t)

    Why it matters:
        Measures price *impact* per unit of dollar volume — i.e., how much
        price moves per dollar traded.  High ILLIQ = thin market where
        even moderate volume causes large price shifts ("slippage-prone").

        For a deep-learning model this is a "Market Impact" proxy:
        high ILLIQ bars suggest price moves may be fragile and easily reversed
        when the volume dries up.

    We smooth with a rolling mean to reduce bar-level noise.
    """
    log_ret = np.log(close).diff().abs()
    dollar_vol = (volume * close).clip(lower=eps)
    illiq_raw = log_ret / dollar_vol

    # Rolling mean to get the "regime" level of illiquidity
    illiq_smooth = illiq_raw.rolling(window).mean()

    # Log-transform for heavy tails (ILLIQ has extreme values occasionally)
    return np.log1p(illiq_smooth).rename("amihud_illiq")


# ──────────────────────────────────────────────
# 3. VOLUME DELTA PROXY (Order Flow Direction)
# ──────────────────────────────────────────────

def volume_delta(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Signed volume proxy:
        delta_t = sign(close_t - close_{t-1}) × volume_t

    A simplified proxy for "net aggressive volume":
        + → buyers dominated (net buying pressure)
        - → sellers dominated (net selling pressure)

    Why it matters:
        Raw volume is unsigned — it doesn't tell you who initiated the trade.
        By signing with the price direction we get a crude but effective
        measure of order flow imbalance.  Sustained positive delta during
        a consolidation = institutional accumulation.
    """
    direction = np.sign(close.diff())
    return (direction * volume).rename("volume_delta")


def volume_delta_ratio(close: pd.Series, volume: pd.Series,
                       window: int = 14, eps: float = 1e-9) -> pd.Series:
    """
    Rolling sum of volume delta / rolling sum of total volume.

    Normalises the raw delta into [-1, +1].
        > +0.5  → strong buying dominance
        < -0.5  → strong selling dominance
        ≈  0    → balanced / uncertain market
    """
    delta = volume_delta(close, volume)
    roll_delta = delta.rolling(window).sum()
    roll_vol   = volume.rolling(window).sum().clip(lower=eps)
    return (roll_delta / roll_vol).rename(f"volume_delta_ratio_{window}")


# ──────────────────────────────────────────────
# 4. BULK VOLUME CLASSIFICATION (BVC)
# ──────────────────────────────────────────────

def bulk_volume_classification(close: pd.Series,
                                volume: pd.Series,
                                window: int = 14,
                                eps: float = 1e-9) -> pd.DataFrame:
    """
    Easley et al. Bulk Volume Classification.

        V_buy  = V_total × Φ(ΔP / σ_ΔP)
        V_sell = V_total - V_buy
        Delta  = V_buy - V_sell

    Where Φ is the standard normal CDF.

    Why it matters:
        More principled than simple price-direction signing.
        Uses the *magnitude* of the price change to weight buy vs sell
        probability — a large up-move classifies more volume as buy-initiated.

    Returns:
        bvc_buy_vol   : estimated buy volume
        bvc_sell_vol  : estimated sell volume
        bvc_delta     : V_buy - V_sell  (signed imbalance)
        bvc_imbalance : delta / total_vol ∈ [-1, +1]
    """
    from scipy.special import ndtr  # standard normal CDF (fast C implementation)

    delta_p = close.diff()
    sigma   = delta_p.rolling(window).std().clip(lower=eps)

    z_score  = delta_p / sigma
    phi      = pd.Series(ndtr(z_score), index=close.index)   # P(buy)

    v_buy  = volume * phi
    v_sell = volume * (1 - phi)
    delta  = v_buy - v_sell

    imbalance = delta / volume.clip(lower=eps)

    return pd.DataFrame({
        "bvc_buy_vol":   v_buy,
        "bvc_sell_vol":  v_sell,
        "bvc_delta":     delta,
        "bvc_imbalance": imbalance,
    }, index=close.index)


# ──────────────────────────────────────────────
# 5. EFFORT vs RESULT
# ──────────────────────────────────────────────

def effort_vs_result(close: pd.Series, volume: pd.Series,
                     eps: float = 1e-9) -> pd.Series:
    """
    Volume / |ΔClose|

    "Effort" (volume) relative to "Result" (price change).

    High ratio → much volume required to move price → resistance / absorption.
    Low  ratio → price moves easily on little volume → thin liquidity / FOMO.

    Inspired by Wyckoff's "Effort vs. Result" principle used by
    institutional tape readers since the 1930s.
    """
    price_change = close.diff().abs().clip(lower=eps)
    evr = volume / price_change

    # Log-transform: ratio spans many orders of magnitude
    return np.log1p(evr).rename("effort_vs_result")


# ──────────────────────────────────────────────
# 6. OBV (On-Balance Volume)
# ──────────────────────────────────────────────

def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Classic OBV — cumulative signed volume.

    We return the *slope* of OBV (change per bar) rather than the
    raw cumulative value, which is non-stationary.

    OBV slope diverging from price slope → order flow leading / lagging
    price — a powerful confirmation / divergence signal.
    """
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    # Normalise by rolling std so it's stationary and scale-free
    obv_std = obv.rolling(50).std().clip(lower=1e-9)
    return ((obv - obv.rolling(50).mean()) / obv_std).rename("obv_zscore")


# ──────────────────────────────────────────────
# 7. KYLE'S LAMBDA (PRICE IMPACT SLOPE)
# ──────────────────────────────────────────────

def kyle_lambda(close: pd.Series, volume: pd.Series,
                window: int = 20, eps: float = 1e-9) -> pd.Series:
    """
    Rolling regression slope of |ΔP| on volume.

    Kyle (1985) showed the price impact of order flow is:
        ΔP = λ × (V_buy - V_sell) + noise

    We approximate λ via OLS on a rolling window.
    High λ → market is sensitive to volume → thin order book.
    Low  λ → deep market, large trades absorbed with minimal impact.
    """
    delta_p = close.diff().abs()
    results = []

    for i in range(len(close)):
        if i < window:
            results.append(np.nan)
            continue
        y = delta_p.iloc[i-window:i].values
        x = volume.iloc[i-window:i].values
        # OLS: slope = cov(x,y) / var(x)
        xm, ym = x.mean(), y.mean()
        var_x = ((x - xm)**2).mean()
        cov_xy = ((x - xm) * (y - ym)).mean()
        slope = cov_xy / (var_x + eps)
        results.append(slope)

    return pd.Series(results, index=close.index, name="kyle_lambda")


# ──────────────────────────────────────────────
# 8. MASTER BUILDER
# ──────────────────────────────────────────────

def add_microstructure_features(df: pd.DataFrame,
                                 use_bvc: bool = True) -> pd.DataFrame:
    """
    Add all microstructure features to `df`.
    Expected columns: open, high, low, close, volume

    `use_bvc=False` skips scipy-dependent BVC (use if scipy unavailable).
    """
    c, v = df["close"], df["volume"]

    df["roll_spread"]            = roll_spread(c)
    df["amihud_illiq"]           = amihud_illiquidity(c, v)
    df["volume_delta"]           = volume_delta(c, v)
    df["volume_delta_ratio_14"]  = volume_delta_ratio(c, v, 14)
    df["effort_vs_result"]       = effort_vs_result(c, v)
    df["obv_zscore"]             = on_balance_volume(c, v)
    df["kyle_lambda"]            = kyle_lambda(c, v)

    if use_bvc:
        try:
            bvc_df = bulk_volume_classification(c, v)
            df = pd.concat([df, bvc_df], axis=1)
        except ImportError:
            # scipy not available — compute simplified delta only
            df["bvc_delta"]     = volume_delta(c, v)
            df["bvc_imbalance"] = volume_delta_ratio(c, v)

    return df