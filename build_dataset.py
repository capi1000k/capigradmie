"""
build_dataset.py
================
To'liq dataset qurish pipeline:

    M15  →  M15 features (base, technical, microstructure, smart_money, time)
    H1   →  H1  features (base, technical, microstructure, smart_money)
             → "h1_" prefix bilan M15 ga forward-fill
    H4   →  H4  features (base, technical, microstructure, smart_money)
             → "h4_" prefix bilan M15 ga forward-fill
    hammasi  →  Triple Barrier labels
    hammasi  →  data/dataset.parquet

Ishga tushirish:
    cd ~/capigradmie
    python3 build_dataset.py
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

from features.base_features     import add_base_features
from features.technical         import add_technical_features
from features.microstructure    import add_microstructure_features
from features.smart_money       import add_smart_money_features
from features.time_features     import add_time_features
from target                     import triple_barrier, label_report


# ──────────────────────────────────────────────
# SOZLAMALAR
# ──────────────────────────────────────────────

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("data")

M15_FILE   = DATA_DIR / "btcusdt_m15.csv"
H1_FILE    = DATA_DIR / "btcusdt_h1.csv"
H4_FILE    = DATA_DIR / "btcusdt_h4.csv"
OUT_FILE   = OUTPUT_DIR / "dataset.parquet"

# Triple Barrier parametrlar (target.py da test qilingan)
TARGET_PARAMS = dict(
    atr_mult_tp  = 2.5,
    atr_mult_sl  = 2.5,
    max_bars_min = 4,
    max_bars_max = 16,
)

# OHLCV ustunlar — feature hisoblangandan keyin olib tashlanadi
OHLCV_COLS = ["open", "high", "low", "close", "volume",
              "quote_volume", "num_trades",
              "taker_buy_base_vol", "taker_buy_quote_vol"]


# ──────────────────────────────────────────────
# 1. DATA YUKLASH
# ──────────────────────────────────────────────

def load_data():
    print("── Data yuklanmoqda ──")

    def _load(path: Path, label: str) -> pd.DataFrame:
        df = pd.read_csv(path, index_col="open_time", parse_dates=True)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        # UTC timezone ta'minlash
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        print(f"  {label}: {len(df):>8,} bar  "
              f"({df.index[0].date()} → {df.index[-1].date()})")
        return df

    m15 = _load(M15_FILE, "M15")
    h1  = _load(H1_FILE,  "H1 ")
    h4  = _load(H4_FILE,  "H4 ")

    return m15, h1, h4


# ──────────────────────────────────────────────
# 2. FEATURE PIPELINE — bitta timeframe uchun
# ──────────────────────────────────────────────

def build_features(df: pd.DataFrame,
                   label: str,
                   bar_minutes: int = 15,
                   add_time: bool = True) -> pd.DataFrame:
    """
    Barcha feature modullarini ishga tushiradi.

    add_time=False — H1 va H4 uchun (time features M15 da yetarli,
    H1/H4 ga qo'shish redundant).
    """
    print(f"\n── {label} featurelar ──")
    df = df.copy()

    t0 = time.time()
    df = add_base_features(df)
    print(f"  base_features      : {time.time()-t0:.1f}s")

    t0 = time.time()
    df = add_technical_features(df)
    print(f"  technical          : {time.time()-t0:.1f}s")

    t0 = time.time()
    df = add_microstructure_features(df)
    print(f"  microstructure     : {time.time()-t0:.1f}s")

    t0 = time.time()
    df = add_smart_money_features(df)
    print(f"  smart_money (Hurst): {time.time()-t0:.1f}s  ← sekin, normal")

    if add_time:
        t0 = time.time()
        df = add_time_features(df, bar_minutes=bar_minutes)
        print(f"  time_features      : {time.time()-t0:.1f}s")

    # OHLCV olib tashlash — model uchun kerak emas
    drop_cols = [c for c in OHLCV_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    print(f"  Jami feature: {df.shape[1]}")
    return df


# ──────────────────────────────────────────────
# 3. YUQORI TIMEFRAME → M15 GA ALIGN
# ──────────────────────────────────────────────

def align_to_m15(htf_features: pd.DataFrame,
                 m15_index: pd.DatetimeIndex,
                 prefix: str,
                 tf_hours: int = 0) -> pd.DataFrame:
    """
    H1 yoki H4 featurelarni M15 ga forward-fill qilish.

    prefix = "h1_" yoki "h4_"

    Forward-fill: har M15 bar O'ZIDAN OLDINGI HTF barning
    feature qiymatini ko'radi. Look-ahead bias YO'Q.

    Masalan H1:
        H1 bar 08:00 → M15 barlarga 08:00, 08:15, 08:30, 08:45 ga ffill
        H1 bar 09:00 → 09:00 dan boshlab yangi qiymat
    """
    print(f"\n── {prefix.upper()} M15 ga align ──")

    if tf_hours > 0:
        htf_features = htf_features.copy()
        htf_features.index = htf_features.index + pd.Timedelta(hours=tf_hours)
    htf_renamed  = htf_features.add_prefix(prefix)
    combined_idx = htf_renamed.index.union(m15_index).sort_values()

    aligned = (
        htf_renamed
        .reindex(combined_idx)
        .ffill()
        .reindex(m15_index)
    )

    nan_rows = aligned.isna().any(axis=1).sum()
    print(f"  Feature ustunlar: {aligned.shape[1]}")
    print(f"  NaN qatorlar    : {nan_rows:,}  (boshlanishda normal)")

    return aligned


# ──────────────────────────────────────────────
# 4. DATASET SIFATINI TEKSHIRISH
# ──────────────────────────────────────────────

def quality_check(dataset: pd.DataFrame) -> None:
    """
    Datasetni tekshirish:
        - Inf qiymatlar
        - Barcha NaN olib tashlangandan keyin qolgan muammolar
        - Feature statistikasi
    """
    print("\n── Sifat tekshiruvi ──")

    # Inf tekshirish
    inf_count = np.isinf(dataset.select_dtypes(include=np.number)).sum().sum()
    if inf_count > 0:
        inf_cols = dataset.columns[np.isinf(dataset).any()].tolist()
        print(f"  ⚠️  Inf qiymatlar: {inf_count}  →  {inf_cols}")
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
    else:
        print(f"  ✅ Inf yo'q")

    # NaN tekshirish (dropna dan keyin)
    nan_count = dataset.isna().sum().sum()
    if nan_count > 0:
        print(f"  ⚠️  NaN qolgan: {nan_count}")
    else:
        print(f"  ✅ NaN yo'q")

    # Konstantа ustunlar (variance = 0)
    num_df    = dataset.select_dtypes(include=np.number)
    const_cols = num_df.columns[num_df.std() < 1e-10].tolist()
    if const_cols:
        print(f"  ⚠️  Konstant ustunlar: {const_cols}")
    else:
        print(f"  ✅ Konstant ustun yo'q")


# ──────────────────────────────────────────────
# 5. MASTER PIPELINE
# ──────────────────────────────────────────────

def build_dataset():
    total_start = time.time()

    print("=" * 55)
    print("  CAPIGRADMIE — Dataset Builder")
    print("=" * 55)

    # ── 1. Data yuklash ──
    m15_raw, h1_raw, h4_raw = load_data()

    # ── 2. Feature hisoblash ──
    m15_feat = build_features(m15_raw, "M15", bar_minutes=15, add_time=True)
    h1_feat  = build_features(h1_raw,  "H1",  bar_minutes=60, add_time=False)
    h4_feat  = build_features(h4_raw,  "H4",  bar_minutes=240, add_time=False)

    # ── 3. HTF → M15 align ──
    h1_aligned = align_to_m15(h1_feat, m15_raw.index, prefix="h1_", tf_hours=1)
    h4_aligned = align_to_m15(h4_feat, m15_raw.index, prefix="h4_", tf_hours=4)

    # ── 4. Labels ──
    print("\n── Triple Barrier labels ──")
    t0 = time.time()
    labels = triple_barrier(m15_raw, **TARGET_PARAMS)
    print(f"  Vaqt: {time.time()-t0:.1f}s")

    # ── 5. Birlashtirish ──
    print("\n── Dataset birlashtirilmoqda ──")
    dataset = pd.concat([m15_feat, h1_aligned, h4_aligned], axis=1)
    dataset["label"] = labels

    # ── 6. Inf → NaN ──
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ── 7. NaN olib tashlash ──
    before = len(dataset)
    dataset = dataset.dropna()
    after   = len(dataset)
    print(f"  NaN olib tashlangandan keyin: {before:,} → {after:,} qator")
    print(f"  Jami feature ustunlar       : {dataset.shape[1] - 1}")

    # ── 8. Sifat tekshiruvi ──
    quality_check(dataset)

    # ── 9. Label hisoboti ──
    label_report(dataset["label"])

    # ── 10. Saqlash ──
    OUTPUT_DIR.mkdir(exist_ok=True)
    dataset.to_parquet(OUT_FILE)
    size_mb = OUT_FILE.stat().st_size / 1024 / 1024
    print(f"\n  💾 Saqlandi: {OUT_FILE}  ({size_mb:.1f} MB)")

    elapsed = (time.time() - total_start) / 60
    print(f"\n{'='*55}")
    print(f"  ✅ DATASET TAYYOR  —  {elapsed:.1f} daqiqa")
    print(f"{'='*55}\n")

    return dataset


# ──────────────────────────────────────────────
# ISHGA TUSHIRISH
# ──────────────────────────────────────────────

if __name__ == "__main__":
    dataset = build_dataset()

    print("── Namunaviy ko'rinish ──")
    print(f"Shape : {dataset.shape}")
    print(f"Davr  : {dataset.index[0].date()} → {dataset.index[-1].date()}")
    print(f"\nBirinchi 3 ustun:")
    print(dataset.iloc[:3, :3])