"""
demo_preview.py
───────────────
Generates a static PNG screenshot of the dashboard
using SIMULATED data (no Binance connection needed).

Run:
    python demo_preview.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from collections import deque
from datetime import datetime, timedelta, timezone

# ─── STYLE ───────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "figure.facecolor":  "#0d0d0f",
    "axes.facecolor":    "#111116",
    "axes.edgecolor":    "#2a2a35",
    "axes.labelcolor":   "#8888aa",
    "axes.grid":         True,
    "grid.color":        "#1e1e28",
    "grid.linewidth":    0.6,
    "xtick.color":       "#555577",
    "ytick.color":       "#555577",
    "text.color":        "#ccccdd",
    "font.family":       "monospace",
    "font.size":         8,
    "axes.titlesize":    9,
    "axes.titlecolor":   "#aaaacc",
    "lines.linewidth":   1.2,
})

BUY_COLOR  = "#00e676"
SELL_COLOR = "#ff1744"
UP_COLOR   = "#00e676"
DOWN_COLOR = "#ff1744"
ACCENT     = "#7c4dff"
ACCENT2    = "#00b0ff"
GOLD       = "#ffd740"

# ─── SIMULATE DATA ────────────────────────────────────────────────────────────

def sim_klines(n=80, start_price=67500):
    np.random.seed(42)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    times = [now - timedelta(minutes=n - i) for i in range(n)]

    prices = [start_price]
    for _ in range(n - 1):
        ret = np.random.normal(0, 0.0008)
        prices.append(prices[-1] * (1 + ret))

    rows = []
    for i, (t, p) in enumerate(zip(times, prices)):
        o = p * (1 + np.random.uniform(-0.001, 0.001))
        c = p * (1 + np.random.uniform(-0.001, 0.001))
        h = max(o, c) * (1 + abs(np.random.normal(0, 0.0005)))
        l = min(o, c) * (1 - abs(np.random.normal(0, 0.0005)))
        v = abs(np.random.normal(15, 5))
        buy_v = v * np.random.uniform(0.35, 0.65)
        rows.append({"open_time": t, "open": o, "high": h,
                     "low": l, "close": c, "volume": v,
                     "taker_buy_base_vol": buy_v, "num_trades": int(np.random.uniform(80, 300))})
    return pd.DataFrame(rows)


def sim_m5(df_m1):
    # Resample m1 → m5
    df = df_m1.set_index("open_time").resample("5min").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),   close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna().reset_index()
    df.rename(columns={"open_time": "open_time"}, inplace=True)
    return df


def sim_trades(n=400, price_center=67500):
    np.random.seed(7)
    now = datetime.now(timezone.utc)
    ts = [now - timedelta(seconds=i * 0.5) for i in range(n)][::-1]
    prices = price_center + np.cumsum(np.random.normal(0, 5, n))
    qtys = np.abs(np.random.exponential(0.08, n))
    is_bm = np.random.rand(n) > 0.52

    rows = [{"timestamp": t, "price": p, "quantity": q, "is_buyer_mm": b}
            for t, p, q, b in zip(ts, prices, qtys, is_bm)]
    return deque(rows, maxlen=500)


def sim_ob_history(n=300):
    np.random.seed(13)
    imb = np.cumsum(np.random.normal(0, 0.03, n))
    imb = np.tanh(imb)   # bound to [-1, 1]
    lp  = np.exp(imb * 0.5)
    return [{"imbalance": i, "liq_pressure": l, "bid_vol": 10*np.exp(i),
             "ask_vol": 10*np.exp(-i)} for i, l in zip(imb, lp)]


# ─── DRAW ────────────────────────────────────────────────────────────────────

def draw_candles(ax, df, df_m5):
    df = df.tail(60).copy()
    df["x"] = np.arange(len(df))
    w = 0.4
    for _, row in df.iterrows():
        color = UP_COLOR if row["close"] >= row["open"] else DOWN_COLOR
        ax.plot([row["x"], row["x"]], [row["low"], row["high"]],
                color=color, linewidth=0.8, alpha=0.7)
        bl = min(row["open"], row["close"])
        bh = max(row["open"], row["close"])
        ax.add_patch(mpatches.FancyBboxPatch(
            (row["x"] - w/2, bl), w, max(bh - bl, 1),
            boxstyle="square,pad=0",
            facecolor=color, edgecolor="none", alpha=0.85,
        ))

    last = df["close"].iloc[-1]
    ax.axhline(last, color=GOLD, linewidth=0.7, linestyle="--", alpha=0.6)
    ax.text(len(df) - 0.5, last, f" {last:,.2f}", color=GOLD, va="center", fontsize=8)

    # M5 overlay
    m1_times = df["open_time"].dt.tz_localize(None).values
    xs, ys = [], []
    for _, row in df_m5.iterrows():
        t = row["open_time"]
        if hasattr(t, 'tz_localize'):
            t = t.tz_localize(None) if t.tzinfo is None else t.tz_convert(None)
        idx = np.searchsorted(m1_times, np.datetime64(t))
        if 0 <= idx < len(df):
            xs.append(idx)
            ys.append(row["close"])
    if len(xs) > 1:
        ax.plot(xs, ys, color=ACCENT2, linewidth=1.8, alpha=0.65, label="M5 trend")
        ax.legend(loc="upper left", fontsize=7.5,
                  facecolor="#0d0d0f", edgecolor="#2a2a35", labelcolor=ACCENT2)

    ax.set_xlim(-1, 62)
    ax.set_xticks([])
    ax.set_title(f"  BTCUSDT  ·  M1 Candlestick    last: {last:,.2f}",
                 loc="left", color="#ccccdd", fontsize=9.5)
    ax.set_ylabel("Price (USDT)")


def draw_trades(ax, trades_deque):
    df = pd.DataFrame(list(trades_deque)).tail(300)
    df["x"] = np.arange(len(df))
    buys  = df[~df["is_buyer_mm"]]
    sells = df[ df["is_buyer_mm"]]
    sz_b = np.clip(buys["quantity"]  * 60, 3, 80)
    sz_s = np.clip(sells["quantity"] * 60, 3, 80)
    ax.scatter(buys["x"],  buys["price"],  c=BUY_COLOR,  s=sz_b, alpha=0.7, linewidths=0)
    ax.scatter(sells["x"], sells["price"], c=SELL_COLOR, s=sz_s, alpha=0.7, linewidths=0)
    ax.axhline(df["price"].mean(), color=GOLD, lw=0.5, ls=":", alpha=0.5)
    ax.set_xlim(0, 300)
    ax.set_xticks([])
    ax.set_title("  AggTrades  ·  ● buy  ● sell", loc="left", color="#ccccdd")
    ax.set_ylabel("Price")


def draw_volume(ax, df):
    df = df.tail(60).copy()
    df["x"] = np.arange(len(df))
    buy_v  = df["taker_buy_base_vol"]
    sell_v = df["volume"] - buy_v
    colors = [UP_COLOR if b >= s else DOWN_COLOR for b, s in zip(buy_v, sell_v)]
    ax.bar(df["x"], df["volume"], color=colors, width=0.8, alpha=0.75)
    ax.plot(df["x"], df["volume"].rolling(10).mean(), color=GOLD, lw=1.0, alpha=0.8)
    ax.set_xlim(-1, 62)
    ax.set_xticks([])
    ax.set_title("  Volume  ·  M1  (green=buy dom, red=sell dom)", loc="left", color="#ccccdd")
    ax.set_ylabel("Vol (BTC)")


def draw_imbalance(ax, history):
    df = pd.DataFrame(history[-300:])
    xs  = np.arange(len(df))
    imb = df["imbalance"].values
    pos = np.where(imb >= 0, imb, 0)
    neg = np.where(imb <  0, imb, 0)
    ax.fill_between(xs, pos, alpha=0.4, color=BUY_COLOR,  linewidth=0)
    ax.fill_between(xs, neg, alpha=0.4, color=SELL_COLOR, linewidth=0)
    ax.plot(xs, imb, color=ACCENT, linewidth=0.9, alpha=0.9, label="Imbalance")
    smooth = pd.Series(imb).rolling(20, min_periods=1).mean()
    ax.plot(xs, smooth, color=GOLD, linewidth=1.2, alpha=0.8, label="MA20")
    ax.axhline(0,    color="#444455", lw=0.8, ls="--")
    ax.axhline( 0.3, color=BUY_COLOR,  lw=0.4, ls=":", alpha=0.4)
    ax.axhline(-0.3, color=SELL_COLOR, lw=0.4, ls=":", alpha=0.4)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_title("  Order Book Imbalance  (purple) + MA20 (gold)", loc="left", color="#ccccdd")
    ax.set_ylabel("Imbalance")
    ax.legend(loc="upper right", fontsize=7, facecolor="#0d0d0f",
              edgecolor="#2a2a35", labelcolor="#aaaacc")
    lp = df["liq_pressure"].iloc[-1]
    ax.text(0.01, 0.08, f" LP: {lp:.2f}",
            transform=ax.transAxes, color=BUY_COLOR if lp > 1 else SELL_COLOR,
            fontsize=8.5)


# ─── COMPOSE ─────────────────────────────────────────────────────────────────

def make_preview():
    print("Simulating data…")
    df_m1   = sim_klines(80)
    df_m5   = sim_m5(df_m1)
    trades  = sim_trades(400, df_m1["close"].iloc[-1])
    ob_hist = sim_ob_history(300)

    fig = plt.figure(figsize=(18, 11), facecolor="#0d0d0f",
                     num="Market Behavior Observatory — Preview")
    fig.suptitle(
        "  ◈  MARKET BEHAVIOR OBSERVATORY  ·  BTCUSDT  ·  Binance   [SIMULATED DATA PREVIEW]",
        color="#aaaacc", fontsize=11, fontfamily="monospace", x=0.01, ha="left",
    )

    gs = gridspec.GridSpec(
        4, 1, figure=fig, hspace=0.1,
        height_ratios=[3, 1.5, 1, 1.2],
        left=0.06, right=0.98, top=0.95, bottom=0.04,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax in axes:
        ax.set_facecolor("#111116")
        ax.grid(True, color="#1e1e28", linewidth=0.5)

    draw_candles(axes[0], df_m1, df_m5)
    draw_trades(axes[1], trades)
    draw_volume(axes[2], df_m1)
    draw_imbalance(axes[3], ob_hist)

    # status bar
    ts = datetime.utcnow().strftime("%H:%M:%S UTC")
    fig.text(
        0.01, 0.005,
        f"  ⏱ {ts}   │   Mid: 67,493.21   │   Spread: 0.10   │   "
        f"OB Imbalance: +0.142   │   Liq Pressure: 1.18   │   "
        f"Buy/Sell Ratio: 1.24   │   Trade Intensity: 387",
        color="#888899", fontsize=7.5, fontfamily="monospace",
        ha="left", va="bottom", transform=fig.transFigure,
    )

    out = "dashboard_preview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0f")
    print(f"✓ Saved → {out}")
    plt.close(fig)
    return out


if __name__ == "__main__":
    make_preview()
