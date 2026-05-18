# src/dashboard.py
"""
VISUALIZATION LAYER — Market Behavior Observatory
──────────────────────────────────────────────────
4-panel real-time matplotlib dashboard:

  Panel 1 │ Candlestick (M1) + M5 trend overlay
  Panel 2 │ Trades scatter (buy=green, sell=red dots)
  Panel 3 │ Volume bars + buy/sell split
  Panel 4 │ Orderbook imbalance + liquidity pressure

Hover annotation: shows candle / trade / OB info on mouse move.
Auto-refreshes every DASHBOARD_REFRESH seconds.
"""

import warnings
warnings.filterwarnings("ignore")

import time
import threading
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import MultiCursor

from src.config import (
    DASHBOARD_REFRESH,
    CANDLES_SHOWN,
    TRADES_SHOWN,
    SYMBOL,
)
from src.features import kline_features, orderbook_features, compute_live_snapshot
from src.logger import get_logger

log = get_logger("dashboard")

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

BUY_COLOR  = "#00e676"   # green
SELL_COLOR = "#ff1744"   # red
UP_COLOR   = "#00e676"
DOWN_COLOR = "#ff1744"
ACCENT     = "#7c4dff"
ACCENT2    = "#00b0ff"
GOLD       = "#ffd740"


def _draw_candles(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Draw OHLC candlestick bars manually (no mplfinance dependency)."""
    ax.clear()
    ax.set_facecolor("#111116")
    ax.grid(True, color="#1e1e28", linewidth=0.5)

    if df.empty:
        ax.set_title(f"  {SYMBOL} M1  —  waiting for data…", loc="left")
        return

    df = df.tail(CANDLES_SHOWN).copy()
    df["x"] = np.arange(len(df))
    w = 0.4

    for _, row in df.iterrows():
        color = UP_COLOR if row["close"] >= row["open"] else DOWN_COLOR
        # wick
        ax.plot([row["x"], row["x"]], [row["low"], row["high"]],
                color=color, linewidth=0.8, alpha=0.7)
        # body
        body_lo = min(row["open"], row["close"])
        body_hi = max(row["open"], row["close"])
        ax.add_patch(mpatches.FancyBboxPatch(
            (row["x"] - w/2, body_lo), w, max(body_hi - body_lo, 0.5),
            boxstyle="square,pad=0",
            facecolor=color, edgecolor="none", alpha=0.85,
        ))

    # price label
    last_close = df["close"].iloc[-1]
    ax.axhline(last_close, color=GOLD, linewidth=0.7, linestyle="--", alpha=0.6)
    ax.text(len(df) - 0.5, last_close, f" {last_close:,.2f}",
            color=GOLD, va="center", fontsize=7.5)

    ax.set_xlim(-1, CANDLES_SHOWN + 1)
    ax.set_xticks([])
    ax.set_title(
        f"  {SYMBOL}  ·  M1 Candlestick    last: {last_close:,.2f}",
        loc="left", color="#ccccdd", fontsize=9,
    )
    ax.set_ylabel("Price (USDT)")


def _draw_m5_overlay(ax: plt.Axes, df_m1: pd.DataFrame, df_m5: pd.DataFrame) -> None:
    """Overlay M5 close as a trend line on the M1 chart."""
    if df_m5.empty or df_m1.empty:
        return

    df_m1 = df_m1.tail(CANDLES_SHOWN).copy()
    df_m1["x"] = np.arange(len(df_m1))

    # map M5 closes onto M1 x-axis by time alignment
    m1_times  = df_m1["open_time"].values
    m5_subset = df_m5[df_m5["open_time"] >= m1_times[0]].copy()

    xs, ys = [], []
    for _, row in m5_subset.iterrows():
        idx = np.searchsorted(m1_times, row["open_time"])
        if 0 <= idx < len(df_m1):
            xs.append(idx)
            ys.append(row["close"])

    if len(xs) > 1:
        ax.plot(xs, ys, color=ACCENT2, linewidth=1.5,
                linestyle="-", alpha=0.6, label="M5 trend")
        ax.legend(loc="upper left", fontsize=7,
                  facecolor="#0d0d0f", edgecolor="#2a2a35",
                  labelcolor=ACCENT2)


def _draw_trades(ax: plt.Axes, trades_deque: deque) -> None:
    ax.clear()
    ax.set_facecolor("#111116")
    ax.grid(True, color="#1e1e28", linewidth=0.5)
    ax.set_title("  AggTrades  ·  ● buy  ● sell", loc="left", color="#ccccdd")
    ax.set_ylabel("Price")
    ax.set_xticks([])

    if not trades_deque:
        return

    df = pd.DataFrame(list(trades_deque)).tail(TRADES_SHOWN)
    df["ts_num"] = np.arange(len(df))

    buys  = df[~df["is_buyer_mm"]]
    sells = df[ df["is_buyer_mm"]]

    sizes_buy  = np.clip(buys["quantity"]  * 8, 2, 60)
    sizes_sell = np.clip(sells["quantity"] * 8, 2, 60)

    if not buys.empty:
        ax.scatter(buys["ts_num"],  buys["price"],
                   c=BUY_COLOR,  s=sizes_buy,  alpha=0.7, linewidths=0)
    if not sells.empty:
        ax.scatter(sells["ts_num"], sells["price"],
                   c=SELL_COLOR, s=sizes_sell, alpha=0.7, linewidths=0)

    if not df.empty:
        ax.set_xlim(0, TRADES_SHOWN)
        mid = df["price"].mean()
        ax.axhline(mid, color=GOLD, linewidth=0.5, linestyle=":", alpha=0.5)


def _draw_volume(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.clear()
    ax.set_facecolor("#111116")
    ax.grid(True, color="#1e1e28", linewidth=0.5)
    ax.set_title("  Volume  ·  M1", loc="left", color="#ccccdd")
    ax.set_ylabel("Vol")
    ax.set_xticks([])

    if df.empty:
        return

    df = df.tail(CANDLES_SHOWN).copy()
    df["x"] = np.arange(len(df))

    # taker buy vs rest split
    buy_vol  = pd.to_numeric(df["taker_buy_base_vol"], errors="coerce").fillna(0)
    sell_vol = df["volume"] - buy_vol

    colors = [UP_COLOR if b >= s else DOWN_COLOR
              for b, s in zip(buy_vol, sell_vol)]
    ax.bar(df["x"], df["volume"], color=colors, width=0.8, alpha=0.75)

    # 10-bar rolling avg
    rolling = df["volume"].rolling(10).mean()
    ax.plot(df["x"], rolling, color=GOLD, linewidth=1.0, alpha=0.8)
    ax.set_xlim(-1, CANDLES_SHOWN + 1)


def _draw_imbalance(ax: plt.Axes, ob_history: list) -> None:
    ax.clear()
    ax.set_facecolor("#111116")
    ax.grid(True, color="#1e1e28", linewidth=0.5)
    ax.set_title("  Order Book Imbalance  ·  bid/(bid+ask)", loc="left", color="#ccccdd")
    ax.set_ylabel("Imbalance")
    ax.set_xticks([])
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="#444455", linewidth=0.8, linestyle="--")
    ax.axhline( 0.3, color=BUY_COLOR,  linewidth=0.4, linestyle=":", alpha=0.4)
    ax.axhline(-0.3, color=SELL_COLOR, linewidth=0.4, linestyle=":", alpha=0.4)

    if not ob_history:
        return

    df = pd.DataFrame(ob_history[-300:])
    xs = np.arange(len(df))

    # color fill
    imb = df["imbalance"].values
    pos = np.where(imb >= 0, imb, 0)
    neg = np.where(imb <  0, imb, 0)
    ax.fill_between(xs, pos, alpha=0.4, color=BUY_COLOR,  linewidth=0)
    ax.fill_between(xs, neg, alpha=0.4, color=SELL_COLOR, linewidth=0)
    ax.plot(xs, imb, color=ACCENT, linewidth=0.9, alpha=0.9)

    # smoothed
    smooth = pd.Series(imb).rolling(20, min_periods=1).mean()
    ax.plot(xs, smooth, color=GOLD, linewidth=1.1, alpha=0.7)

    # liquidity pressure side bar
    if "liq_pressure" in df.columns:
        lp = df["liq_pressure"].iloc[-1]
        label = f" LP: {lp:.2f}"
        color = BUY_COLOR if lp > 1 else SELL_COLOR
        ax.text(0.01, 0.05, label, transform=ax.transAxes,
                color=color, fontsize=8, alpha=0.9)


def _draw_status_bar(fig: plt.Figure, snap: dict) -> None:
    """Bottom text bar with live microstructure numbers."""
    imb    = snap.get("imbalance", 0)
    spread = snap.get("spread", 0)
    mid    = snap.get("mid_price", 0)
    bsr    = snap.get("buy_sell_ratio", 1)
    ti     = snap.get("trade_intensity", 0)
    lp     = snap.get("liq_pressure", 1)

    imb_col = BUY_COLOR if imb >= 0 else SELL_COLOR
    bsr_col = BUY_COLOR if bsr >= 1 else SELL_COLOR

    ts = datetime.utcnow().strftime("%H:%M:%S UTC")
    status = (
        f"  ⏱ {ts}   │   "
        f"Mid: {mid:,.2f}   │   "
        f"Spread: {spread:.2f}   │   "
        f"OB Imbalance: {imb:+.3f}   │   "
        f"Liq Pressure: {lp:.2f}   │   "
        f"Buy/Sell Ratio: {bsr:.2f}   │   "
        f"Trade Intensity: {ti}"
    )
    fig.texts.clear()
    fig.text(
        0.01, 0.005, status,
        color="#888899", fontsize=7.5,
        fontfamily="monospace",
        ha="left", va="bottom",
        transform=fig.transFigure,
    )


# ─── MAIN DASHBOARD CLASS ─────────────────────────────────────────────────────

class Dashboard:
    def __init__(self, shared_state: dict):
        self.state = shared_state
        self.ob_history: list[dict] = []

        # figure layout
        self.fig = plt.figure(
            figsize=(18, 11),
            facecolor="#0d0d0f",
            num="Market Behavior Observatory",
        )
        self.fig.suptitle(
            f"  ◈  MARKET BEHAVIOR OBSERVATORY  ·  {SYMBOL}  ·  Binance",
            color="#aaaacc", fontsize=11, fontfamily="monospace",
            x=0.01, ha="left",
        )

        gs = gridspec.GridSpec(
            4, 1,
            figure=self.fig,
            hspace=0.08,
            height_ratios=[3, 1.5, 1, 1.2],
            left=0.06, right=0.98,
            top=0.95, bottom=0.03,
        )

        self.ax_candle  = self.fig.add_subplot(gs[0])
        self.ax_trades  = self.fig.add_subplot(gs[1])
        self.ax_volume  = self.fig.add_subplot(gs[2])
        self.ax_imbal   = self.fig.add_subplot(gs[3])

        # hover annotation
        self._annot = self.ax_candle.annotate(
            "", xy=(0, 0), xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a22", ec="#4444aa", alpha=0.9),
            color="#ccccdd", fontsize=7.5,
            fontfamily="monospace",
            arrowprops=dict(arrowstyle="->", color="#4444aa"),
        )
        self._annot.set_visible(False)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)

        log.info("[dashboard] initialized")

    def _on_hover(self, event) -> None:
        """Show candle details on hover."""
        if event.inaxes != self.ax_candle:
            self._annot.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        df_m1 = self.state.get("klines_m1", pd.DataFrame())
        if df_m1.empty:
            return

        df_m1 = df_m1.tail(CANDLES_SHOWN).reset_index(drop=True)
        x_idx = int(round(event.xdata)) if event.xdata is not None else -1

        if 0 <= x_idx < len(df_m1):
            row = df_m1.iloc[x_idx]
            txt = (
                f"O: {row['open']:,.2f}\n"
                f"H: {row['high']:,.2f}\n"
                f"L: {row['low']:,.2f}\n"
                f"C: {row['close']:,.2f}\n"
                f"V: {row['volume']:,.3f}\n"
                f"T: {row['open_time'].strftime('%H:%M') if hasattr(row['open_time'], 'strftime') else row['open_time']}"
            )
            self._annot.xy = (x_idx, event.ydata)
            self._annot.set_text(txt)
            self._annot.set_visible(True)
        else:
            self._annot.set_visible(False)

        self.fig.canvas.draw_idle()

    def update(self, frame) -> None:
        df_m1 = self.state.get("klines_m1", pd.DataFrame())
        df_m5 = self.state.get("klines_m5", pd.DataFrame())

        if not df_m1.empty:
            df_m1 = kline_features(df_m1)

        trades_deque: deque = self.state.get("trades_live", deque())

        ob_live = self.state.get("orderbook_live")
        if ob_live:
            self.ob_history.append(ob_live)
            if len(self.ob_history) > 600:
                self.ob_history = self.ob_history[-600:]

        snap = compute_live_snapshot(self.state)

        # ── draw panels ──────────────────────────────────────────────────────
        _draw_candles(self.ax_candle, df_m1)
        if not df_m5.empty:
            _draw_m5_overlay(self.ax_candle, df_m1, df_m5)

        _draw_trades(self.ax_trades, trades_deque)
        _draw_volume(self.ax_volume, df_m1)
        _draw_imbalance(self.ax_imbal, self.ob_history)
        _draw_status_bar(self.fig, snap)

        # restore hover annot (draw_candles clears ax)
        self._annot = self.ax_candle.annotate(
            "", xy=(0, 0), xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a22", ec="#4444aa", alpha=0.9),
            color="#ccccdd", fontsize=7.5, fontfamily="monospace",
            arrowprops=dict(arrowstyle="->", color="#4444aa"),
        )
        self._annot.set_visible(False)

    def run(self) -> None:
        log.info("[dashboard] starting animation loop")
        self._anim = FuncAnimation(
            self.fig,
            self.update,
            interval=DASHBOARD_REFRESH * 1000,
            cache_frame_data=False,
        )
        plt.show()
