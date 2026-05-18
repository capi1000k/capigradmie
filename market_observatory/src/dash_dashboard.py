# src/dash_dashboard.py
"""
VISUALIZATION LAYER v2 — Plotly Dash Live Dashboard
Production-hardened: defensive callbacks, safe DB access, graceful fallbacks.
"""

import logging
import traceback
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

from src.config import (
    SYMBOL, DASH_HOST, DASH_PORT,
    DASHBOARD_REFRESH, CANDLES_SHOWN, TRADES_SHOWN,
)
from src.db import load_klines, load_trades, load_orderbook, load_cvd, db_stats
from src.logger import get_logger

log = get_logger("dash")

# ── COLORS ────────────────────────────────────────────────────────────────────
BG_DARK  = "#0d0d0f"
BG_PANEL = "#111116"
GRID     = "#1e1e28"
TEXT     = "#ccccdd"
GREEN    = "#00e676"
RED      = "#ff1744"
VIOLET   = "#7c4dff"
BLUE     = "#00b0ff"
GOLD     = "#ffd740"
ORANGE   = "#ff9100"

BASE = dict(
    paper_bgcolor=BG_DARK,
    plot_bgcolor=BG_PANEL,
    font=dict(color=TEXT, family="'Courier New', monospace", size=10),
    margin=dict(l=55, r=20, t=28, b=20),
    xaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=0.5, color=TEXT),
    yaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=0.5, color=TEXT),
)

# ── SHARED STATE ──────────────────────────────────────────────────────────────
_shared_state: dict = {}


def set_shared_state(state: dict) -> None:
    global _shared_state
    _shared_state = state


# ── HELPERS ───────────────────────────────────────────────────────────────────

def blank_figure(title: str = "") -> go.Figure:
    """Return a styled empty placeholder figure — never raises."""
    fig = go.Figure()
    fig.update_layout(
        **BASE,
        title=dict(text=title, font=dict(size=10, color="#555566"), x=0),
        annotations=[dict(
            text="waiting for data…",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#333344", size=13),
        )],
    )
    return fig


def safe_df(raw) -> pd.DataFrame:
    """
    Coerce *raw* to a clean DataFrame:
    - None / non-DataFrame → empty DataFrame
    - Replace np.inf / -np.inf with NaN
    - Numeric columns: NaN → 0
    """
    if raw is None or not isinstance(raw, pd.DataFrame):
        return pd.DataFrame()
    df = raw.copy()
    # Replace inf values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill numeric NaNs with 0
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    return df


def safe_float(value, default: float = 0.0) -> float:
    """Convert value to float safely; return default on any error."""
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def _strip_tz(series: pd.Series) -> pd.Series:
    """Convert any timezone-aware datetime series to naive UTC."""
    try:
        s = pd.to_datetime(series, errors="coerce")
        if hasattr(s.dt, "tz") and s.dt.tz is not None:
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        return s
    except Exception:
        return series


def _load_safe(loader_fn, *args, label: str = "data") -> pd.DataFrame:
    """Call a DB loader function and guarantee a safe DataFrame back."""
    try:
        result = loader_fn(*args)
        return safe_df(result)
    except Exception:
        log.error("[dash] failed to load %s:\n%s", label, traceback.format_exc())
        return pd.DataFrame()


# ── CHART BUILDERS ────────────────────────────────────────────────────────────

def chart_candle(df_m1: pd.DataFrame, df_m5: pd.DataFrame) -> go.Figure:
    """M1 Candlestick with optional M5 trend overlay."""
    try:
        df_m1 = safe_df(df_m1)
        df_m5 = safe_df(df_m5)

        required = {"open_time", "open", "high", "low", "close"}
        if df_m1.empty or not required.issubset(df_m1.columns):
            return blank_figure("M1 Candlestick")

        df = df_m1.tail(CANDLES_SHOWN).copy()
        df["open_time"] = _strip_tz(df["open_time"])

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df["open_time"],
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            increasing=dict(fillcolor=GREEN, line=dict(color=GREEN, width=1)),
            decreasing=dict(fillcolor=RED,   line=dict(color=RED,   width=1)),
            name="M1",
        ))

        if not df_m5.empty and required.issubset(df_m5.columns):
            df5 = df_m5.tail(30).copy()
            df5["open_time"] = _strip_tz(df5["open_time"])
            fig.add_trace(go.Scatter(
                x=df5["open_time"], y=df5["close"],
                mode="lines", line=dict(color=BLUE, width=2, dash="dot"),
                name="M5", opacity=0.75,
            ))

        # Safe last-close hline
        if len(df) > 0 and "close" in df.columns:
            last = safe_float(df["close"].iloc[-1])
            if last > 0:
                fig.add_hline(
                    y=last,
                    line=dict(color=GOLD, width=1, dash="dash"),
                    annotation_text=f" {last:,.2f}",
                    annotation_font=dict(color=GOLD, size=11, family="monospace"),
                    annotation_position="right",
                )

        fig.update_layout(
            **BASE,
            title=dict(
                text=f"  {SYMBOL}  ·  M1 Candlestick  +  M5 Trend",
                font=dict(size=11, color=TEXT), x=0,
            ),
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend=dict(orientation="h", x=0, y=1.05,
                        font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        )
        return fig

    except Exception:
        log.error("[dash] chart_candle error:\n%s", traceback.format_exc())
        return blank_figure("M1 Candlestick")


def chart_trades(trades_deque) -> go.Figure:
    """Scatter plot of recent aggTrades coloured by side."""
    try:
        if not trades_deque:
            return blank_figure("AggTrades")

        raw = list(trades_deque)
        if not raw:
            return blank_figure("AggTrades")

        df = pd.DataFrame(raw).tail(TRADES_SHOWN)
        required = {"timestamp", "is_buyer_mm", "price", "quantity"}
        if not required.issubset(df.columns):
            return blank_figure("AggTrades")

        df = safe_df(df)
        df["ts"] = _strip_tz(df["timestamp"])

        buys  = df[~df["is_buyer_mm"].astype(bool)]
        sells = df[ df["is_buyer_mm"].astype(bool)]

        fig = go.Figure()
        if not buys.empty:
            sizes = np.clip(buys["quantity"].values * 40, 3, 18)
            fig.add_trace(go.Scatter(
                x=buys["ts"], y=buys["price"], mode="markers",
                marker=dict(color=GREEN, size=sizes,
                            opacity=0.75, line=dict(width=0)),
                name="Buy",
                customdata=buys["quantity"],
                hovertemplate="<b>BUY</b> %{y:,.2f}  qty:%{customdata:.4f}<extra></extra>",
            ))
        if not sells.empty:
            sizes = np.clip(sells["quantity"].values * 40, 3, 18)
            fig.add_trace(go.Scatter(
                x=sells["ts"], y=sells["price"], mode="markers",
                marker=dict(color=RED, size=sizes,
                            opacity=0.75, line=dict(width=0)),
                name="Sell",
                customdata=sells["quantity"],
                hovertemplate="<b>SELL</b> %{y:,.2f}  qty:%{customdata:.4f}<extra></extra>",
            ))

        fig.update_layout(
            **BASE,
            title=dict(text="  AggTrades  ·  dot size = quantity",
                       font=dict(size=10, color=TEXT), x=0),
            hovermode="closest",
            legend=dict(orientation="h", x=0, y=1.05,
                        font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        )
        return fig

    except Exception:
        log.error("[dash] chart_trades error:\n%s", traceback.format_exc())
        return blank_figure("AggTrades")


def chart_volume(df_m1: pd.DataFrame) -> go.Figure:
    """Stacked buy/sell volume bars with MA10 overlay."""
    try:
        df_m1 = safe_df(df_m1)
        required = {"open_time", "volume", "taker_buy_base_vol"}
        if df_m1.empty or not required.issubset(df_m1.columns):
            return blank_figure("Volume")

        df = df_m1.tail(CANDLES_SHOWN).copy()
        df["open_time"] = _strip_tz(df["open_time"])
        buy_vol  = pd.to_numeric(df["taker_buy_base_vol"], errors="coerce").fillna(0)
        total    = pd.to_numeric(df["volume"],             errors="coerce").fillna(0)
        sell_vol = (total - buy_vol).clip(lower=0)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["open_time"], y=buy_vol,
                             name="Buy", marker_color=GREEN, opacity=0.8))
        fig.add_trace(go.Bar(x=df["open_time"], y=sell_vol,
                             name="Sell", marker_color=RED, opacity=0.8))
        ma10 = total.rolling(10, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df["open_time"], y=ma10,
            mode="lines", line=dict(color=GOLD, width=1.5),
            name="MA10", opacity=0.85,
        ))
        fig.update_layout(
            **BASE, barmode="stack",
            title=dict(text="  Volume  ·  Buy vs Sell",
                       font=dict(size=10, color=TEXT), x=0),
            legend=dict(orientation="h", x=0, y=1.05,
                        font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        )
        return fig

    except Exception:
        log.error("[dash] chart_volume error:\n%s", traceback.format_exc())
        return blank_figure("Volume")


def chart_cvd(df_cvd: pd.DataFrame, live_cvd) -> go.Figure:
    """Cumulative Volume Delta area chart."""
    try:
        df_cvd = safe_df(df_cvd)
        if df_cvd.empty or "cvd" not in df_cvd.columns or "timestamp" not in df_cvd.columns:
            return blank_figure("CVD")

        df = df_cvd.tail(300).copy()
        df["ts"] = _strip_tz(df["timestamp"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["ts"], y=df["cvd"],
            mode="lines", fill="tozeroy",
            fillcolor="rgba(0,230,118,0.12)",
            line=dict(color=BLUE, width=1.5),
            name="CVD",
        ))
        fig.add_hline(y=0, line=dict(color="#333344", width=1))

        live_val = safe_float(live_cvd, default=None) if live_cvd is not None else None
        suffix = f"  live: {live_val:+.4f}" if live_val is not None else ""
        fig.update_layout(
            **BASE,
            title=dict(
                text=f"  CVD  ·  Cumulative Volume Delta{suffix}",
                font=dict(size=10, color=TEXT), x=0,
            ),
        )
        return fig

    except Exception:
        log.error("[dash] chart_cvd error:\n%s", traceback.format_exc())
        return blank_figure("CVD")


def chart_imbalance(df_ob: pd.DataFrame) -> go.Figure:
    """Order-book imbalance + normalised OFI."""
    try:
        df_ob = safe_df(df_ob)
        if df_ob.empty or "imbalance" not in df_ob.columns or "timestamp" not in df_ob.columns:
            return blank_figure("Imbalance")

        df     = df_ob.tail(300).copy()
        df["ts"] = _strip_tz(df["timestamp"])
        imb    = df["imbalance"].values.astype(float)
        smooth = pd.Series(imb).rolling(20, min_periods=1).mean().values

        ofi_col = df["ofi"].values.astype(float) if "ofi" in df.columns else np.zeros(len(df))
        ofi_max = np.abs(ofi_col).max()
        ofi_norm = ofi_col / (ofi_max + 1e-10)  # safe division

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["ts"], y=imb, mode="lines", fill="tozeroy",
            fillcolor="rgba(124,77,255,0.15)",
            line=dict(color=VIOLET, width=1.5), name="Imbalance",
        ))
        fig.add_trace(go.Scatter(
            x=df["ts"], y=ofi_norm, mode="lines",
            line=dict(color=ORANGE, width=1, dash="dot"),
            name="OFI (norm)", opacity=0.7,
        ))
        fig.add_trace(go.Scatter(
            x=df["ts"], y=smooth, mode="lines",
            line=dict(color=GOLD, width=1.5), name="MA20",
        ))
        fig.add_hline(y=0,    line=dict(color="#333344", width=1))
        fig.add_hline(y= 0.3, line=dict(color=GREEN, width=0.5, dash="dot"))
        fig.add_hline(y=-0.3, line=dict(color=RED,   width=0.5, dash="dot"))

        # BASE already contains 'yaxis'; override it explicitly to avoid
        # "multiple values for keyword argument 'yaxis'" TypeError.
        layout = {**BASE}
        layout["yaxis"] = dict(range=[-1.1, 1.1], showgrid=True,
                                gridcolor=GRID, gridwidth=0.5, color=TEXT)
        fig.update_layout(
            **layout,
            title=dict(text="  OB Imbalance  +  OFI",
                       font=dict(size=10, color=TEXT), x=0),
            legend=dict(orientation="h", x=0, y=1.05,
                        font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        )
        return fig

    except Exception:
        log.error("[dash] chart_imbalance error:\n%s", traceback.format_exc())
        return blank_figure("Imbalance")


def chart_microprice(df_ob: pd.DataFrame) -> go.Figure:
    """Microprice vs midprice with optional spread BPS secondary axis."""
    try:
        df_ob = safe_df(df_ob)
        if (df_ob.empty
                or "microprice" not in df_ob.columns
                or "timestamp"  not in df_ob.columns):
            return blank_figure("Microprice")

        df = df_ob.tail(300).copy()
        df["ts"] = _strip_tz(df["timestamp"])

        fig = go.Figure()
        if "mid_price" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["ts"], y=df["mid_price"], mode="lines",
                line=dict(color="#555577", width=1), name="Midprice",
            ))
        fig.add_trace(go.Scatter(
            x=df["ts"], y=df["microprice"], mode="lines",
            line=dict(color=GOLD, width=1.5), name="Microprice",
        ))
        if "spread_bps" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["ts"], y=df["spread_bps"], mode="lines",
                line=dict(color=ORANGE, width=1, dash="dot"),
                name="Spread BPS", yaxis="y2", opacity=0.7,
            ))
            fig.update_layout(
                yaxis2=dict(
                    overlaying="y", side="right", showgrid=False,
                    tickfont=dict(size=8, color=ORANGE), color=ORANGE,
                ),
            )

        fig.update_layout(
            **BASE,
            title=dict(text="  Microprice  vs  Midprice  +  Spread BPS",
                       font=dict(size=10, color=TEXT), x=0),
            legend=dict(orientation="h", x=0, y=1.05,
                        font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        )
        return fig

    except Exception:
        log.error("[dash] chart_microprice error:\n%s", traceback.format_exc())
        return blank_figure("Microprice")


# ── APP LAYOUT ────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title=f"Market Observatory · {SYMBOL}",
    update_title=None,
)


def _card(label: str, id_: str, color: str = GOLD) -> dbc.Col:
    return dbc.Col(
        dbc.Card([dbc.CardBody([
            html.P(label, className="mb-0",
                   style={"fontSize": "10px", "color": "#888899",
                          "fontFamily": "monospace"}),
            html.H5(id=id_, children="—",
                    style={"color": color, "fontFamily": "monospace",
                           "marginBottom": 0}),
        ], style={"padding": "6px 10px"})],
            style={"background": "#16161e", "border": "1px solid #2a2a35"}),
        width="auto",
    )


app.layout = dbc.Container(
    fluid=True,
    style={"background": BG_DARK, "minHeight": "100vh", "padding": "10px"},
    children=[
        dbc.Row([dbc.Col(html.H5(
            f"◈ MARKET BEHAVIOR OBSERVATORY  ·  {SYMBOL}  ·  Binance",
            style={"color": "#aaaacc", "fontFamily": "monospace",
                   "fontWeight": "bold", "marginBottom": "8px"},
        ))]),

        dbc.Row([
            _card("Price",        "stat-price",  GOLD),
            _card("Spread",       "stat-spread", TEXT),
            _card("Spread BPS",   "stat-sbps",   TEXT),
            _card("OB Imbalance", "stat-imb",    GREEN),
            _card("OFI",          "stat-ofi",    VIOLET),
            _card("CVD",          "stat-cvd",    BLUE),
            _card("B/S Ratio",    "stat-bsr",    ORANGE),
            _card("Microprice",   "stat-micro",  GOLD),
            _card("DB rows",      "stat-db",     "#555577"),
        ], className="mb-2 g-2"),

        dbc.Row([dbc.Col(dcc.Graph(
            id="g-candle",
            figure=blank_figure("M1 Candlestick"),
            config={"displayModeBar": False},
            style={"height": "320px"},
        ))], className="mb-1"),

        dbc.Row([
            dbc.Col(dcc.Graph(
                id="g-trades",
                figure=blank_figure("AggTrades"),
                config={"displayModeBar": False},
                style={"height": "200px"},
            ), width=8),
            dbc.Col(dcc.Graph(
                id="g-volume",
                figure=blank_figure("Volume"),
                config={"displayModeBar": False},
                style={"height": "200px"},
            ), width=4),
        ], className="mb-1"),

        dbc.Row([
            dbc.Col(dcc.Graph(
                id="g-cvd",
                figure=blank_figure("CVD"),
                config={"displayModeBar": False},
                style={"height": "180px"},
            ), width=4),
            dbc.Col(dcc.Graph(
                id="g-imbalance",
                figure=blank_figure("Imbalance"),
                config={"displayModeBar": False},
                style={"height": "180px"},
            ), width=4),
            dbc.Col(dcc.Graph(
                id="g-microprice",
                figure=blank_figure("Microprice"),
                config={"displayModeBar": False},
                style={"height": "180px"},
            ), width=4),
        ]),

        dcc.Interval(id="tick", interval=DASHBOARD_REFRESH, n_intervals=0),
    ],
)


# ── CALLBACK ──────────────────────────────────────────────────────────────────

# Total outputs: 6 figures + 9 stat strings = 15
_N_OUTPUTS = 15


def _safe_stats(ob: dict) -> tuple:
    """
    Extract stat-card strings from the live orderbook snapshot.
    Always returns a 7-tuple of strings — never raises.
    """
    if not ob:
        return ("—",) * 7

    def _fmt_f(key, fmt):
        try:
            return fmt.format(safe_float(ob.get(key, 0)))
        except Exception:
            return "—"

    price_s = _fmt_f("mid_price",  "{:,.2f}")
    spr_s   = _fmt_f("spread",     "{:.3f}")
    sbps_s  = _fmt_f("spread_bps", "{:.2f}")
    imb_s   = _fmt_f("imbalance",  "{:+.3f}")
    ofi_s   = _fmt_f("ofi",        "{:+.4f}")
    micro_s = _fmt_f("microprice", "{:,.2f}")

    return price_s, spr_s, sbps_s, imb_s, ofi_s, micro_s


@app.callback(
    # ── 6 graph figures ────────────────────────────────────────────────────────
    Output("g-candle",     "figure"),
    Output("g-trades",     "figure"),
    Output("g-volume",     "figure"),
    Output("g-cvd",        "figure"),
    Output("g-imbalance",  "figure"),
    Output("g-microprice", "figure"),
    # ── 9 stat-card texts ──────────────────────────────────────────────────────
    Output("stat-price",  "children"),
    Output("stat-spread", "children"),
    Output("stat-sbps",   "children"),
    Output("stat-imb",    "children"),
    Output("stat-ofi",    "children"),
    Output("stat-cvd",    "children"),
    Output("stat-bsr",    "children"),
    Output("stat-micro",  "children"),
    Output("stat-db",     "children"),
    # ── trigger ────────────────────────────────────────────────────────────────
    Input("tick", "n_intervals"),
    prevent_initial_call=False,
)
def update(_n):
    """
    Master refresh callback.
    Contract: must return exactly 15 values (6 figures + 9 strings).
    Any exception is caught; blank figures + "ERR" strings are returned.
    """
    # ── define a safe fallback tuple with correct arity ───────────────────────
    def _fallback(reason: str = "error"):
        log.error("[dash] update() fallback triggered: %s", reason)
        figs = (
            blank_figure("M1 Candlestick"),
            blank_figure("AggTrades"),
            blank_figure("Volume"),
            blank_figure("CVD"),
            blank_figure("Imbalance"),
            blank_figure("Microprice"),
        )
        stats = ("ERR",) * 9
        return figs + stats  # 15 total

    try:
        # ── load data (each call is independently safe) ───────────────────────
        df_m1    = _load_safe(load_klines,    "m1", 100, label="klines_m1")
        df_m5    = _load_safe(load_klines,    "m5",  30, label="klines_m5")
        df_ob    = _load_safe(load_orderbook, 300,       label="orderbook")
        df_cvd   = _load_safe(load_cvd,       300,       label="cvd")
        trades   = _shared_state.get("trades_live", deque())
        live_cvd = _shared_state.get("cvd_live")
        ob       = _shared_state.get("orderbook_live") or {}

        # ── stat cards ────────────────────────────────────────────────────────
        price_s, spr_s, sbps_s, imb_s, ofi_s, micro_s = _safe_stats(ob)

        # CVD live
        try:
            cvd_val = safe_float(live_cvd)
            cvd_s   = f"{cvd_val:+.4f}" if live_cvd is not None else "—"
        except Exception:
            cvd_s = "—"

        # Buy/Sell ratio
        bsr_s = "—"
        try:
            if trades:
                df_t  = pd.DataFrame(list(trades))
                if "is_buyer_mm" in df_t.columns and "quantity" in df_t.columns:
                    df_t  = safe_df(df_t)
                    buys  = df_t[~df_t["is_buyer_mm"].astype(bool)]["quantity"].sum()
                    sells = df_t[ df_t["is_buyer_mm"].astype(bool)]["quantity"].sum()
                    ratio = safe_float(buys) / (safe_float(sells) + 1e-10)
                    bsr_s = f"{ratio:.3f}"
        except Exception:
            log.warning("[dash] bsr calculation failed:\n%s", traceback.format_exc())

        # DB stats
        db_s = "—"
        try:
            stats_raw = db_stats()
            if stats_raw:
                db_s = "  ".join(
                    f"{k.split('_')[-1]}:{v}" for k, v in stats_raw.items()
                )
        except Exception:
            log.warning("[dash] db_stats failed:\n%s", traceback.format_exc())

        # ── charts ────────────────────────────────────────────────────────────
        return (
            chart_candle(df_m1, df_m5),
            chart_trades(trades),
            chart_volume(df_m1),
            chart_cvd(df_cvd, live_cvd),
            chart_imbalance(df_ob),
            chart_microprice(df_ob),
            price_s, spr_s, sbps_s, imb_s, ofi_s, cvd_s, bsr_s, micro_s, db_s,
        )

    except Exception:
        log.error("[dash] update() unhandled exception:\n%s", traceback.format_exc())
        return _fallback("unhandled")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def run_dashboard(shared_state: dict, debug: bool = False) -> None:
    set_shared_state(shared_state)
    log.info("[dash] starting at http://%s:%s", DASH_HOST, DASH_PORT)
    app.run(
        host=DASH_HOST,
        port=DASH_PORT,
        debug=debug,
        use_reloader=False,
    )