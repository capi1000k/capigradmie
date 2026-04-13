"""
build_dataset.py
================
To'liq dataset qurish pipeline:

    M15 data  →  M15 features (base, technical, microstructure, smart_money, time)
    H1  data  →  H1  features (base, technical, microstructure, smart_money)
                 → "h1_" prefix bilan M15 ga forward-fill
    ikkalasi  →  Triple Barrier + H1 trend filter labels
    hammasi   →  birlashtirib  →  data/dataset.parquet

Oxirida Hurst Exponent vizualizatsiyasi chiqadi (matplotlib).

Ishga tushirish:
    cd ~/capigradmie
    python3 build_dataset.py
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from features import (
    add_base_features,
    add_technical_features,
    add_microstructure_features,
    add_smart_money_features,
    add_time_features,
)
from target import triple_barrier


# ──────────────────────────────────────────────
# SOZLAMALAR
# ──────────────────────────────────────────────

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("data")
M15_FILE   = DATA_DIR / "btcusdt_m15.csv"
H1_FILE    = DATA_DIR / "btcusdt_h1.csv"
OUT_FILE   = OUTPUT_DIR / "dataset.parquet"

TARGET_PARAMS = dict(
    atr_mult_tp = 1.5,
    atr_mult_sl = 1.5,
    max_bars    = 8,
)


# ──────────────────────────────────────────────
# 1. DATA YUKLASH
# ──────────────────────────────────────────────

def load_data():
    print("── Data yuklanmoqda ──")

    m15 = pd.read_csv(M15_FILE, index_col="open_time", parse_dates=True)
    h1  = pd.read_csv(H1_FILE,  index_col="open_time", parse_dates=True)

    # Duplicate indekslarni tozalash
    m15 = m15[~m15.index.duplicated(keep="last")].sort_index()
    h1  = h1[~h1.index.duplicated(keep="last")].sort_index()

    print(f"  M15: {m15.shape[0]:,} bar  ({m15.index[0].date()} → {m15.index[-1].date()})")
    print(f"  H1 : {h1.shape[0]:,} bar  ({h1.index[0].date()} → {h1.index[-1].date()})")

    return m15, h1


# ──────────────────────────────────────────────
# 2. FEATURE PIPELINE — bitta timeframe uchun
# ──────────────────────────────────────────────

def build_features(df: pd.DataFrame, label: str, bar_minutes: int = 15) -> pd.DataFrame:
    """
    Barcha feature modullarini ishga tushiradi.
    label = "M15" yoki "H1" (logging uchun)
    """
    print(f"\n── {label} featurelar hisoblanmoqda ──")
    df = df.copy()

    t0 = time.time()
    df = add_base_features(df)
    print(f"  base_features       : {time.time()-t0:.1f}s")

    t0 = time.time()
    df = add_technical_features(df)
    print(f"  technical_features  : {time.time()-t0:.1f}s")

    t0 = time.time()
    df = add_microstructure_features(df)
    print(f"  microstructure      : {time.time()-t0:.1f}s")

    t0 = time.time()
    df = add_smart_money_features(df)
    print(f"  smart_money (Hurst) : {time.time()-t0:.1f}s  ← sekin, normal")

    if bar_minutes > 0:
        t0 = time.time()
        df = add_time_features(df, bar_minutes=bar_minutes)
        print(f"  time_features       : {time.time()-t0:.1f}s")

    # OHLCV ustunlarini olib tashlaymiz — model uchun kerak emas
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    df = df.drop(columns=[c for c in ohlcv_cols if c in df.columns])

    print(f"  Jami featurelar: {df.shape[1]}")
    return df


# ──────────────────────────────────────────────
# 3. H1 FEATURELARNI M15 GA ALIGN QILISH
# ──────────────────────────────────────────────

def align_h1_to_m15(h1_features: pd.DataFrame,
                     m15_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    H1 feature ustunlarini "h1_" prefix bilan M15 ga forward-fill qiladi.

    Muhim: forward-fill = har M15 bar o'zidan OLDINGI H1 barning
    feature qiymatini ko'radi. Look-ahead bias yo'q.
    """
    print("\n── H1 featurelar M15 ga align qilinmoqda ──")

    # Ustun nomlariga "h1_" prefix qo'shamiz
    h1_renamed = h1_features.add_prefix("h1_")

    # H1 indeksini M15 bilan birlashtirb ffill qilamiz
    combined_idx = h1_renamed.index.union(m15_index).sort_values()
    aligned = (
        h1_renamed
        .reindex(combined_idx)
        .ffill()
        .reindex(m15_index)
    )

    print(f"  H1 feature ustunlar: {aligned.shape[1]}")
    print(f"  NaN qatorlar       : {aligned.isna().any(axis=1).sum():,} (kutiladi, boshlanishda)")
    return aligned


# ──────────────────────────────────────────────
# 4. HURST VIZUALIZATSIYASI
# ──────────────────────────────────────────────

def plot_hurst_analysis(dataset: pd.DataFrame, out_path: str = "data/hurst_analysis.png"):
    """
    Hurst Exponent ning amaliy foydasi ko'rsatiladi:

    1. Hurst vaqt bo'yicha — qachon trending, qachon mean-reverting
    2. BTC narxi bilan bir arada — korrelyatsiya ko'rinadi
    3. Hurst > 0.55 da long/short signal returns
    4. Hurst < 0.45 da long/short signal returns
       → Model Hurst orqali "rejim"ni ajrata oladimi?
    """
    print("\n── Hurst vizualizatsiyasi tayyorlanmoqda ──")

    # Kerakli ustunlar borligini tekshirish
    has_m15_hurst = "hurst_100" in dataset.columns
    has_h1_hurst  = "h1_hurst_100" in dataset.columns

    if not has_m15_hurst and not has_h1_hurst:
        print("  Hurst ustuni topilmadi, vizualizatsiya o'tkazib yuborildi.")
        return

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    COLORS = {
        "hurst_m15": "#7B68EE",
        "hurst_h1":  "#48D1CC",
        "price":     "#FFD700",
        "trending":  "#00FF88",
        "reverting": "#FF6B6B",
        "neutral":   "#888888",
        "long_ret":  "#00C853",
        "short_ret": "#FF3D00",
    }

    # Vaqt oralig'ini cheklash (oxirgi 2 yil — aniqroq ko'rinadi)
    cutoff = dataset.index[-1] - pd.Timedelta(days=365*2)
    ds = dataset[dataset.index >= cutoff].copy()

    # ── Panel 1: Hurst vaqt bo'yicha (M15) ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#1a1d27")

    if has_m15_hurst:
        h = ds["hurst_100"].dropna()
        ax1.plot(h.index, h.values, color=COLORS["hurst_m15"],
                 linewidth=0.6, alpha=0.8, label="M15 Hurst")

    if has_h1_hurst:
        h1h = ds["h1_hurst_100"].dropna()
        ax1.plot(h1h.index, h1h.values, color=COLORS["hurst_h1"],
                 linewidth=1.2, alpha=0.9, label="H1 Hurst")

    ax1.axhline(0.55, color=COLORS["trending"],  linestyle="--",
                linewidth=1.0, alpha=0.7, label="Trending (H>0.55)")
    ax1.axhline(0.50, color=COLORS["neutral"],   linestyle=":",
                linewidth=0.8, alpha=0.5, label="Random walk (H=0.5)")
    ax1.axhline(0.45, color=COLORS["reverting"], linestyle="--",
                linewidth=1.0, alpha=0.7, label="Mean-reverting (H<0.45)")

    ax1.fill_between(ds.index,
                     0.55, 1.0,
                     alpha=0.08, color=COLORS["trending"])
    ax1.fill_between(ds.index,
                     0.0, 0.45,
                     alpha=0.08, color=COLORS["reverting"])

    ax1.set_ylim(0.2, 0.85)
    ax1.set_ylabel("Hurst Exponent", color="white", fontsize=11)
    ax1.set_title("Hurst Exponent vaqt bo'yicha — bozor rejimi deteksiyasi",
                  color="white", fontsize=13, pad=10)
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#333344")
    ax1.legend(loc="upper right", fontsize=8,
               facecolor="#1a1d27", labelcolor="white")

    # ── Panel 2: Hurst taqsimoti ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#1a1d27")

    if has_m15_hurst:
        h_vals = ds["hurst_100"].dropna().values
        ax2.hist(h_vals[h_vals < 0.45], bins=30,
                 color=COLORS["reverting"], alpha=0.7, label="Mean-rev (H<0.45)")
        ax2.hist(h_vals[(h_vals >= 0.45) & (h_vals <= 0.55)], bins=30,
                 color=COLORS["neutral"], alpha=0.5, label="Random (0.45-0.55)")
        ax2.hist(h_vals[h_vals > 0.55], bins=30,
                 color=COLORS["trending"], alpha=0.7, label="Trending (H>0.55)")

    ax2.axvline(0.5, color="white", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Hurst qiymati", color="white", fontsize=10)
    ax2.set_ylabel("Chastota", color="white", fontsize=10)
    ax2.set_title("M15 Hurst taqsimoti", color="white", fontsize=11)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#333344")
    ax2.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white")

    # ── Panel 3: Hurst rejimi bo'yicha signal sifati ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#1a1d27")

    if has_m15_hurst and "label" in ds.columns:
        ds_signals = ds[ds["label"] != 0].copy()
        ds_signals["future_ret"] = (
            ds_signals["label"] *
            np.log(1 + ds_signals.get("log_return", pd.Series(0, index=ds_signals.index)))
        )

        # Hurst ni 5 ta bin ga ajratamiz
        try:
            ds_signals["hurst_bin"] = pd.qcut(
                ds_signals["hurst_100"].dropna(),
                q=5,
                labels=["<Q1", "Q1-Q2", "Q2-Q3", "Q3-Q4", ">Q4"]
            )
            bin_ret = ds_signals.groupby("hurst_bin", observed=True)["future_ret"].mean() * 100

            colors_bars = [COLORS["reverting"] if i < 2 else
                          (COLORS["neutral"] if i == 2 else COLORS["trending"])
                          for i in range(len(bin_ret))]
            bars = ax3.bar(range(len(bin_ret)), bin_ret.values,
                          color=colors_bars, alpha=0.8, width=0.6)

            for bar, val in zip(bars, bin_ret.values):
                ax3.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.0001,
                        f"{val:.4f}%", ha="center", va="bottom",
                        color="white", fontsize=8)

            ax3.set_xticks(range(len(bin_ret)))
            ax3.set_xticklabels(bin_ret.index, color="white", fontsize=9)
            ax3.axhline(0, color="white", linewidth=0.8, linestyle="--")
        except Exception:
            ax3.text(0.5, 0.5, "Signal data yetarli emas",
                    transform=ax3.transAxes, ha="center",
                    color="white", fontsize=11)

    ax3.set_ylabel("O'rtacha signal return (%)", color="white", fontsize=10)
    ax3.set_title("Hurst qiymatiga qarab signal sifati\n(yuqori Hurst → yaxshiroq trend signal)",
                  color="white", fontsize=11)
    ax3.tick_params(colors="white")
    ax3.spines[:].set_color("#333344")

    # ── Panel 4: M15 vs H1 Hurst korrelyatsiyasi ──
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#1a1d27")

    if has_m15_hurst and has_h1_hurst:
        scatter_df = ds[["hurst_100", "h1_hurst_100"]].dropna()
        # Sample for speed
        if len(scatter_df) > 5000:
            scatter_df = scatter_df.sample(5000, random_state=42)

        ax4.scatter(scatter_df["hurst_100"], scatter_df["h1_hurst_100"],
                   alpha=0.15, s=2, color=COLORS["hurst_m15"])

        # Trend chiziq
        z = np.polyfit(scatter_df["hurst_100"], scatter_df["h1_hurst_100"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0.3, 0.8, 100)
        ax4.plot(x_line, p(x_line), color=COLORS["hurst_h1"],
                linewidth=2, alpha=0.9)

        corr = scatter_df.corr().iloc[0, 1]
        ax4.set_title(f"M15 vs H1 Hurst korrelyatsiyasi\n(r = {corr:.3f})",
                     color="white", fontsize=11)
        ax4.set_xlabel("M15 Hurst", color="white", fontsize=10)
        ax4.set_ylabel("H1 Hurst", color="white", fontsize=10)
    else:
        ax4.text(0.5, 0.5, "Ikkala Hurst ham kerak",
                transform=ax4.transAxes, ha="center",
                color="white", fontsize=11)

    ax4.tick_params(colors="white")
    ax4.spines[:].set_color("#333344")

    # ── Panel 5: Hurst rejimi bo'yicha label taqsimoti ──
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#1a1d27")

    if has_m15_hurst and "label" in ds.columns:
        try:
            ds["hurst_regime"] = pd.cut(
                ds["hurst_100"],
                bins=[0, 0.45, 0.55, 1.0],
                labels=["Mean-rev\n(H<0.45)", "Random\n(0.45-0.55)", "Trending\n(H>0.55)"]
            )

            regime_labels = ds.groupby("hurst_regime", observed=True)["label"].value_counts(normalize=True).unstack()

            if -1 in regime_labels.columns and 1 in regime_labels.columns:
                x = np.arange(len(regime_labels.index))
                width = 0.25
                ax5.bar(x - width, regime_labels.get(-1, 0) * 100,
                       width, color=COLORS["short_ret"], alpha=0.8, label="Short (-1)")
                ax5.bar(x,        regime_labels.get(0, 0)  * 100,
                       width, color=COLORS["neutral"],   alpha=0.8, label="No-edge (0)")
                ax5.bar(x + width, regime_labels.get(1, 0) * 100,
                       width, color=COLORS["long_ret"],  alpha=0.8, label="Long (+1)")

                ax5.set_xticks(x)
                ax5.set_xticklabels(regime_labels.index, color="white", fontsize=9)
        except Exception:
            pass

    ax5.set_ylabel("Label % ulushi", color="white", fontsize=10)
    ax5.set_title("Hurst rejimi bo'yicha label taqsimoti\n(trending rejimda signal ko'proqmi?)",
                  color="white", fontsize=11)
    ax5.tick_params(colors="white")
    ax5.spines[:].set_color("#333344")
    ax5.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white")

    fig.suptitle("Hurst Exponent — Bozor Rejimi Deteksiyasi va Signal Sifatiga Ta'siri",
                color="white", fontsize=15, fontweight="bold", y=0.98)

    plt.savefig(out_path, dpi=140, bbox_inches="tight",
                facecolor="#0f1117", edgecolor="none")
    plt.close()
    print(f"  Grafik saqlandi: {out_path}")


# ──────────────────────────────────────────────
# 5. MASTER PIPELINE
# ──────────────────────────────────────────────

def build_dataset():
    total_start = time.time()

    # 1. Data yuklash
    m15_raw, h1_raw = load_data()

    # 2. M15 featurelar
    m15_features = build_features(m15_raw, label="M15", bar_minutes=15)

    # 3. H1 featurelar (time_features H1 da hisoblashning ma'nosi yo'q — M15 ga ffill)
    h1_features = build_features(h1_raw, label="H1", bar_minutes=0)

    # 4. H1 featurelarni M15 ga align qilish
    h1_aligned = align_h1_to_m15(h1_features, m15_raw.index)

    # 5. Target labellar
    print("\n── Target labellar hisoblanmoqda ──")
    t0 = time.time()
    labels = triple_barrier(
        m15_raw,
        **TARGET_PARAMS
    )
    print(f"  Vaqt: {time.time()-t0:.1f}s")

    # 6. Hammasini birlashtirish
    print("\n── Dataset birlashtirilmoqda ──")
    dataset = pd.concat([m15_features, h1_aligned], axis=1)
    dataset["label"] = labels

    # 7. NaN qatorlarni olib tashlash
    before = len(dataset)
    dataset = dataset.dropna()
    after  = len(dataset)
    print(f"  NaN olib tashlangandan keyin: {before:,} → {after:,} qator")
    print(f"  Jami feature ustunlar       : {dataset.shape[1] - 1}")

    # Label taqsimoti
    lc = dataset["label"].value_counts().sort_index()
    print(f"\n  Final label taqsimoti:")
    for lbl, cnt in lc.items():
        name = {-1: "short", 0: "no-edge", 1: "long"}.get(lbl, str(lbl))
        print(f"    {name:8s}: {cnt:,}  ({cnt/len(dataset)*100:.1f}%)")

    # 8. Saqlash
    OUTPUT_DIR.mkdir(exist_ok=True)
    dataset.to_parquet(OUT_FILE)
    size_mb = OUT_FILE.stat().st_size / 1024 / 1024
    print(f"\n  Dataset saqlandi: {OUT_FILE}  ({size_mb:.1f} MB)")

    print(f"\n── Jami vaqt: {(time.time()-total_start)/60:.1f} daqiqa ──")

    # 9. Hurst vizualizatsiyasi
    plot_hurst_analysis(dataset)

    return dataset


# ──────────────────────────────────────────────
# ISHGA TUSHIRISH
# ──────────────────────────────────────────────

if __name__ == "__main__":
    dataset = build_dataset()
    print("\nDataset tayyor!")
    print(dataset.head(3))