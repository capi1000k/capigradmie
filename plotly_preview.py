"""
plotly_preview.py — Static Plotly dashboard screenshot (no network needed)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BG_DARK="#0d0d0f"; BG_PANEL="#111116"; GRID="#1e1e28"; TEXT="#ccccdd"
GREEN="#00e676"; RED="#ff1744"; VIOLET="#7c4dff"; BLUE="#00b0ff"; GOLD="#ffd740"; ORANGE="#ff9100"

np.random.seed(42)
N = 80
now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
times = [now - timedelta(minutes=N-i) for i in range(N)]
prices = [67500.0]
for _ in range(N-1): prices.append(prices[-1]*(1+np.random.normal(0,0.0008)))

rows=[]
for t,p in zip(times,prices):
    o=p*(1+np.random.uniform(-0.001,0.001)); c=p*(1+np.random.uniform(-0.001,0.001))
    h=max(o,c)*(1+abs(np.random.normal(0,0.0005))); l=min(o,c)*(1-abs(np.random.normal(0,0.0005)))
    v=abs(np.random.normal(15,5)); bv=v*np.random.uniform(0.35,0.65)
    rows.append({"open_time":t,"open":o,"high":h,"low":l,"close":c,"volume":v,"taker_buy_base_vol":bv})
df_m1 = pd.DataFrame(rows)
df_m1["open_time"] = pd.to_datetime(df_m1["open_time"]).dt.tz_localize(None)

# M5
df_m5 = df_m1.set_index("open_time").resample("5min").agg(
    open=("open","first"),high=("high","max"),low=("low","min"),
    close=("close","last"),volume=("volume","sum")).dropna().reset_index()

# Trades
T=300; ts=[now-timedelta(seconds=i*0.5) for i in range(T)][::-1]
tprices=prices[-1]+np.cumsum(np.random.normal(0,5,T))
qtys=np.abs(np.random.exponential(0.08,T)); isbm=np.random.rand(T)>0.52
df_t=pd.DataFrame({"ts":[pd.to_datetime(t).tz_localize(None) for t in ts],"price":tprices,"qty":qtys,"isbm":isbm})

# OB history
imb=np.tanh(np.cumsum(np.random.normal(0,0.03,300)))
ofi_norm=np.random.normal(0,0.3,300)
micro=prices[-1]+imb*2; mid=np.full(300,prices[-1])
sbps=0.5+np.abs(np.random.normal(0,0.1,300))
ts_ob=[now-timedelta(seconds=300-i) for i in range(300)]
df_ob=pd.DataFrame({"ts":[pd.to_datetime(t).tz_localize(None) for t in ts_ob],
                    "imbalance":imb,"ofi_norm":ofi_norm,"microprice":micro,"mid_price":mid,"spread_bps":sbps})

# CVD
cvd=np.cumsum(np.random.normal(0,0.5,300))
df_cvd=pd.DataFrame({"ts":df_ob["ts"],"cvd":cvd})

LAYOUT=dict(paper_bgcolor=BG_DARK,plot_bgcolor=BG_PANEL,
    font=dict(color=TEXT,family="'Courier New',monospace",size=10),
    xaxis=dict(showgrid=True,gridcolor=GRID,gridwidth=0.5,color=TEXT),
    yaxis=dict(showgrid=True,gridcolor=GRID,gridwidth=0.5,color=TEXT))

fig = make_subplots(
    rows=4,cols=3,
    row_heights=[0.38,0.22,0.22,0.18],
    specs=[[{"colspan":3},None,None],
           [{"colspan":2},None,{"colspan":1}],
           [{"colspan":1},{"colspan":1},{"colspan":1}],
           [{"colspan":3},None,None]],
    vertical_spacing=0.04,horizontal_spacing=0.04,
)

# ── Candles ──────────────────────────────────────────────────────────────────
df=df_m1.tail(60)
fig.add_trace(go.Candlestick(
    x=df["open_time"],open=df["open"],high=df["high"],low=df["low"],close=df["close"],
    increasing=dict(fillcolor=GREEN,line=dict(color=GREEN,width=1)),
    decreasing=dict(fillcolor=RED,line=dict(color=RED,width=1)),
    name="M1",showlegend=True),row=1,col=1)
fig.add_trace(go.Scatter(x=df_m5["open_time"],y=df_m5["close"],
    mode="lines",line=dict(color=BLUE,width=2,dash="dot"),name="M5",opacity=0.7),row=1,col=1)
last=float(df["close"].iloc[-1])
fig.add_hline(y=last,line=dict(color=GOLD,width=1,dash="dash"),
    annotation_text=f" {last:,.2f}",annotation_font=dict(color=GOLD,size=11),
    annotation_position="right",row=1,col=1)

# ── Trades ────────────────────────────────────────────────────────────────────
buys=df_t[~df_t["isbm"]]; sells=df_t[df_t["isbm"]]
fig.add_trace(go.Scatter(x=buys["ts"],y=buys["price"],mode="markers",
    marker=dict(color=GREEN,size=np.clip(buys["qty"]*40,3,16),opacity=0.75,line=dict(width=0)),
    name="Buy"),row=2,col=1)
fig.add_trace(go.Scatter(x=sells["ts"],y=sells["price"],mode="markers",
    marker=dict(color=RED,size=np.clip(sells["qty"]*40,3,16),opacity=0.75,line=dict(width=0)),
    name="Sell"),row=2,col=1)

# ── Volume ────────────────────────────────────────────────────────────────────
bv=df["taker_buy_base_vol"]; sv=df["volume"]-bv
fig.add_trace(go.Bar(x=df["open_time"],y=bv,name="Buy Vol",marker_color=GREEN,opacity=0.8),row=2,col=3)
fig.add_trace(go.Bar(x=df["open_time"],y=sv,name="Sell Vol",marker_color=RED,opacity=0.8),row=2,col=3)

# ── CVD ───────────────────────────────────────────────────────────────────────
fig.add_trace(go.Scatter(x=df_cvd["ts"],y=df_cvd["cvd"],mode="lines",
    fill="tozeroy",fillcolor="rgba(0,230,118,0.12)",
    line=dict(color=BLUE,width=1.5),name="CVD"),row=3,col=1)
fig.add_hline(y=0,line=dict(color="#333344",width=1),row=3,col=1)

# ── Imbalance + OFI ───────────────────────────────────────────────────────────
ma20=pd.Series(imb).rolling(20,min_periods=1).mean()
fig.add_trace(go.Scatter(x=df_ob["ts"],y=df_ob["imbalance"],mode="lines",
    fill="tozeroy",fillcolor="rgba(124,77,255,0.15)",
    line=dict(color=VIOLET,width=1.5),name="Imbalance"),row=3,col=2)
fig.add_trace(go.Scatter(x=df_ob["ts"],y=ofi_norm,mode="lines",
    line=dict(color=ORANGE,width=1,dash="dot"),name="OFI",opacity=0.7),row=3,col=2)
fig.add_trace(go.Scatter(x=df_ob["ts"],y=ma20,mode="lines",
    line=dict(color=GOLD,width=1.5),name="MA20"),row=3,col=2)

# ── Microprice ────────────────────────────────────────────────────────────────
fig.add_trace(go.Scatter(x=df_ob["ts"],y=df_ob["mid_price"],mode="lines",
    line=dict(color="#555577",width=1),name="Mid"),row=3,col=3)
fig.add_trace(go.Scatter(x=df_ob["ts"],y=df_ob["microprice"],mode="lines",
    line=dict(color=GOLD,width=1.5),name="Microprice"),row=3,col=3)

# ── Status bar ────────────────────────────────────────────────────────────────
ts_now = datetime.utcnow().strftime("%H:%M:%S UTC")
fig.add_annotation(
    text=f"⏱ {ts_now}  │  Price: {last:,.2f}  │  Spread: 0.10  │  Spread BPS: 0.15  │  "
         f"OB Imbalance: +0.142  │  OFI: +0.038  │  CVD: +12.47  │  B/S Ratio: 1.24  │  Microprice: {last+0.8:,.2f}",
    xref="paper",yref="paper",x=0,y=-0.01,
    showarrow=False,font=dict(color="#888899",size=9,family="monospace"),
    align="left",
)

fig.update_layout(
    **LAYOUT,
    height=900,
    title=dict(
        text="◈  MARKET BEHAVIOR OBSERVATORY  ·  BTCUSDT  ·  Binance  [SIMULATED PREVIEW]",
        font=dict(size=13,color="#aaaacc",family="monospace"),x=0.01,
    ),
    xaxis_rangeslider_visible=False,
    barmode="stack",
    margin=dict(l=60,r=30,t=50,b=40),
    legend=dict(orientation="h",x=0,y=1.01,font=dict(size=9),bgcolor="rgba(0,0,0,0)"),
    showlegend=True,
)

# kill rangeslider on candlestick axis
for axis in fig.layout:
    if axis.startswith("xaxis"):
        fig.layout[axis].update(rangeslider_visible=False)

out = "dashboard_plotly_preview.png"
fig.write_image(out, width=1600, height=900, scale=1.5)
print(f"✓ Saved → {out}")
