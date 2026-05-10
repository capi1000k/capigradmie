"""
target.py
=========
Yaxshilangan Triple Barrier Labeling.

Avvalgi versiyadan farqi:
    1. max_bars DINAMIK — ATR ga qarab o'zgaradi.
       Yuqori volatilite → kamroq bar kutish (narx tez harakat qiladi).
       Past volatilite   → ko'proq bar kutish (narx sekin harakat qiladi).

    2. Barrier kenglik ASIMMETRIK — TP va SL alohida sozlanadi.
       Trend following uchun: TP > SL (risk/reward 2:1)

    3. No-edge labellarini kamaytirish:
       Dinamik oyna tufayli signal soni oshadi.

    4. Hisobot: label taqsimoti + sifat baholash.

Ishga tushirish (test uchun):
    python3 target.py

Tavsiya etilgan parametrlar:
    atr_mult_tp  = 2.0   (TP = 2 × ATR)
    atr_mult_sl  = 1.0   (SL = 1 × ATR)  → risk/reward 2:1
    max_bars_min = 4     (minimum 1 soat)
    max_bars_max = 24    (maksimum 6 soat)
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# SOZLAMALAR
# ──────────────────────────────────────────────

ROUND_TRIP_COST = 0.0006   # Binance Futures maker+taker ~0.06%


# ──────────────────────────────────────────────
# 1. YORDAMCHI FUNKSIYALAR
# ──────────────────────────────────────────────

def _wilder_atr(high: pd.Series,
                low: pd.Series,
                close: pd.Series,
                period: int = 14) -> pd.Series:
    """Wilder ATR — Triple Barrier barrier kengligini hisoblash uchun."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _dynamic_max_bars(atr: np.ndarray,
                      close: np.ndarray,
                      max_bars_min: int = 4,
                      max_bars_max: int = 24) -> np.ndarray:
    """
    Har bir bar uchun dinamik max_bars hisoblash.

    Mantiq:
        vol_pct = ATR / Close  (normalised volatilite %)
        Yuqori vol → bozor tez harakat qiladi → kamroq bar kerak
        Past  vol  → bozor sekin harakat qiladi → ko'proq bar kerak

    Natija: [max_bars_min, max_bars_max] oralig'ida integer array.
    """
    eps = 1e-9
    vol_pct = atr / (np.abs(close) + eps)

    vol_series = pd.Series(vol_pct)

    # Oxirgi 500 bar ichida normalize
    vol_min   = vol_series.rolling(500, min_periods=50).min().values
    vol_max   = vol_series.rolling(500, min_periods=50).max().values
    vol_range = np.clip(vol_max - vol_min, eps, None)

    # vol_norm: 0 = eng past vol, 1 = eng yuqori vol
    vol_norm = np.clip((vol_pct - vol_min) / vol_range, 0.0, 1.0)

    # Yuqori vol → max_bars_min, Past vol → max_bars_max
    max_bars = max_bars_max - vol_norm * (max_bars_max - max_bars_min)
    max_bars = np.where(np.isnan(max_bars), max_bars_min, max_bars)
    max_bars = np.round(max_bars).astype(int)

    max_bars = np.clip(max_bars, max_bars_min, max_bars_max)

    return max_bars


# ──────────────────────────────────────────────
# 2. TRIPLE BARRIER — ASOSIY FUNKSIYA
# ──────────────────────────────────────────────

def triple_barrier(
    m15: pd.DataFrame,
    atr_mult_tp:  float = 2.0,
    atr_mult_sl:  float = 1.0,
    max_bars_min: int   = 4,
    max_bars_max: int   = 24,
    cost:         float = ROUND_TRIP_COST,
    atr_period:   int   = 14,
) -> pd.Series:
    """
    Yaxshilangan Triple Barrier Labeling.

    Har bir bar uchun:
        entry  = close[i]
        upper  = entry + atr_mult_tp × ATR[i]  (TP barrier, cost chiqarilgan)
        lower  = entry - atr_mult_sl × ATR[i]  (SL barrier, cost qo'shilgan)
        window = i+1 ... i+max_bars[i]          (dinamik oyna)

        high[j] >= upper  →  label = +1  (Long ✅)
        low[j]  <= lower  →  label = -1  (Short ✅)
        vaqt tugasa       →  label =  0  (No-edge)

    Parametrlar:
        atr_mult_tp  : TP barrier (ATR koeffitsienti)
        atr_mult_sl  : SL barrier (ATR koeffitsienti)
        max_bars_min : eng kam kutish (bar soni)
        max_bars_max : eng ko'p kutish (bar soni)
        cost         : round-trip transaction cost
        atr_period   : Wilder ATR davri

    Returns:
        pd.Series, dtype=int8, values in {-1, 0, +1}
    """
    close = m15["close"].values
    high  = m15["high"].values
    low   = m15["low"].values
    atr   = _wilder_atr(
                m15["high"], m15["low"], m15["close"], atr_period
            ).values
    N     = len(close)

    dyn_max_bars = _dynamic_max_bars(atr, close, max_bars_min, max_bars_max)

    labels = np.zeros(N, dtype=np.int8)

    for i in range(N - 1):
        if np.isnan(atr[i]):
            continue

        entry = close[i]
        a     = atr[i]

        # Cost hisobga olingan barrierlar
        upper = entry + atr_mult_tp * a - entry * cost
        lower = entry - atr_mult_sl * a + entry * cost

        if upper <= entry or lower >= entry:
            continue

        mb    = dyn_max_bars[i]
        label = 0

        for j in range(i + 1, min(i + mb + 1, N)):
            if high[j] >= upper:
                label =  1
                break
            if low[j] <= lower:
                label = -1
                break

        labels[i] = label

    # Oxirgi barlar — ishonchli label yo'q
    labels[N - max_bars_max:] = 0

    return pd.Series(labels, index=m15.index, name="label", dtype="int8")


# ──────────────────────────────────────────────
# 3. HISOBOT
# ──────────────────────────────────────────────

def label_report(labels: pd.Series) -> None:
    """
    Label taqsimoti va sifat baholash.

    Yaxshi dataset ko'rsatkichlari:
        signal (non-zero) ≥ 50%
        long/short farqi  < 20%
    """
    total = len(labels)
    vc    = labels.value_counts().sort_index()

    print("\n── Label taqsimoti ──")
    names = {-1: "Short  (-1)", 0: "No-edge ( 0)", 1: "Long   (+1)"}
    for lbl, cnt in vc.items():
        bar = "█" * int(cnt / total * 40)
        pct = cnt / total * 100
        print(f"  {names.get(lbl, str(lbl))}: {cnt:>8,}  ({pct:5.1f}%)  {bar}")

    signal_mask  = labels != 0
    signal_total = signal_mask.sum()
    signal_pct   = signal_total / total * 100

    print(f"\n  Jami bar         : {total:,}")
    print(f"  Signal (non-zero): {signal_total:,}  ({signal_pct:.1f}%)")

    if signal_total > 0:
        long_pct  = (labels == 1).sum()  / signal_total * 100
        short_pct = (labels == -1).sum() / signal_total * 100
        print(f"  Signal ichida long : {long_pct:.1f}%")
        print(f"  Signal ichida short: {short_pct:.1f}%")

    print("\n── Baholash ──")
    if signal_pct < 30:
        print("  ⚠️  Signal juda kam  → max_bars_max oshiring yoki atr_mult kamaytiring")
    elif signal_pct > 85:
        print("  ⚠️  No-edge juda kam → atr_mult oshiring")
    else:
        print("  ✅ Signal taqsimoti yaxshi")

    if signal_total > 0 and abs(long_pct - short_pct) > 25:
        print("  ⚠️  Long/Short nomutanosib — trend bozor davri ko'p (normal, lekin kuzating)")
    elif signal_total > 0:
        print("  ✅ Long/Short balansi yaxshi")


# ──────────────────────────────────────────────
# TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("── target.py test ──")

    m15 = pd.read_csv(
        "data/btcusdt_m15.csv",
        index_col="open_time",
        parse_dates=True,
    )
    m15 = m15[~m15.index.duplicated(keep="last")].sort_index()
    print(f"  M15: {len(m15):,} bar yuklanmdi")

    t0 = time.time()
    labels = triple_barrier(
    m15,
    atr_mult_tp  = 2.5,
    atr_mult_sl  = 2.5,
    max_bars_min = 4,
    max_bars_max = 16,
)
    print(f"  Hisoblash vaqti: {time.time() - t0:.1f}s")

    label_report(labels)