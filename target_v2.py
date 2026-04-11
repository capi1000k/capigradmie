"""
target_v2.py — H1 trend filter YO'Q, faqat Triple Barrier.

H1 trend endi label emas, feature sifatida dataset da qoladi.
Model o'zi H1 trend featurelardan foydalanishni hal qiladi.
"""

import numpy as np
import pandas as pd

ROUND_TRIP_COST = 0.0006

def _wilder_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()

def triple_barrier(m15, atr_mult_tp=1.5, atr_mult_sl=1.5, max_bars=8, cost=ROUND_TRIP_COST):
    close = m15["close"].values
    high  = m15["high"].values
    low   = m15["low"].values
    atr   = _wilder_atr(m15["high"], m15["low"], m15["close"]).values
    N     = len(close)
    labels = np.zeros(N, dtype=np.int8)

    for i in range(N - 1):
        if np.isnan(atr[i]):
            continue
        entry = close[i]
        upper = entry + atr_mult_tp * atr[i] - entry * cost
        lower = entry - atr_mult_sl * atr[i] + entry * cost
        if upper <= entry or lower >= entry:
            continue
        label = 0
        for j in range(i+1, min(i+max_bars+1, N)):
            if high[j] >= upper:
                label =  1; break
            if low[j]  <= lower:
                label = -1; break
        labels[i] = label

    labels[N-max_bars:] = 0
    return pd.Series(labels, index=m15.index, name="label")

if __name__ == "__main__":
    import pandas as pd
    m15 = pd.read_csv("data/btcusdt_m15.csv", index_col="open_time", parse_dates=True)
    m15 = m15[~m15.index.duplicated(keep="last")].sort_index()

    labels = triple_barrier(m15)
    vc = labels.value_counts().sort_index()
    total = len(labels)
    for lbl, cnt in vc.items():
        name = {-1:"short", 0:"no-edge", 1:"long"}[lbl]
        print(f"  {name:8s}: {cnt:,}  ({cnt/total*100:.1f}%)")
