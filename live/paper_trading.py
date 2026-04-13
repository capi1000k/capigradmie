"""
Paper Trading — Binance Testnet
================================
Real pulsiz sinov. Har 15 daqiqada:
  1. Binance dan oxirgi M15 va H1 barlarni oladi
  2. Featurelarni hisoblaydi
  3. Model prediction qiladi
  4. Confidence > 0.70 bo'lsa — virtual pozitsiya ochadi
  5. Natijani log ga yozadi
"""

import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import requests
from datetime import datetime, timezone
from pathlib import Path

# ──────────────────────────────────────────────
# SOZLAMALAR
# ──────────────────────────────────────────────

SYMBOL          = "BTCUSDT"
CAPITAL         = 1000.0      # Virtual kapital (USD)
HALF_KELLY      = 0.081
CONF_THRESHOLD  = 0.70
MODEL_PATH      = "model/mlp_fold4.pt"
LOG_PATH        = "live/paper_trades.csv"

# Binance Testnet (real pul yo'q)
BASE_URL = "https://testnet.binance.vision/api"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("live/paper_trading.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# MODEL YUKLASH
# ──────────────────────────────────────────────

class TradingMLP(nn.Module):
    def __init__(self, input_dim, hidden=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_model(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TradingMLP(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["features"], ckpt["scaler_mean"].astype(np.float32), ckpt["scaler_std"].astype(np.float32)


# ──────────────────────────────────────────────
# BINANCE DAN DATA OLISH
# ──────────────────────────────────────────────

def fetch_klines(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    """Binance testnet dan OHLCV data olish"""
    url = f"{BASE_URL}/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Oxirgi bar hali yopilmagan — olib tashlaymiz
    return df[["open", "high", "low", "close", "volume"]].iloc[:-1]


# ──────────────────────────────────────────────
# FEATURE HISOBLASH
# ──────────────────────────────────────────────

def compute_features(m15: pd.DataFrame, h1: pd.DataFrame,
                     feature_names: list,
                     scaler_mean: np.ndarray,
                     scaler_std: np.ndarray) -> np.ndarray:
    """Oxirgi M15 bar uchun featurelarni hisoblaydi"""
    from features import (
        add_base_features, add_technical_features,
        add_microstructure_features, add_smart_money_features,
        add_time_features
    )

    # M15 featurelar
    m15_f = m15.copy()
    m15_f = add_base_features(m15_f)
    m15_f = add_technical_features(m15_f)
    m15_f = add_microstructure_features(m15_f)
    m15_f = add_smart_money_features(m15_f)
    m15_f = add_time_features(m15_f, bar_minutes=15)
    m15_f = m15_f.drop(columns=["open","high","low","close","volume"], errors="ignore")

    # H1 featurelar
    h1_f = h1.copy()
    h1_f = add_base_features(h1_f)
    h1_f = add_technical_features(h1_f)
    h1_f = add_microstructure_features(h1_f)
    h1_f = add_smart_money_features(h1_f)
    h1_f = h1_f.drop(columns=["open","high","low","close","volume"], errors="ignore")
    h1_f = h1_f.add_prefix("h1_")

    # H1 ni M15 ga align
    combined_idx = h1_f.index.union(m15_f.index).sort_values()
    h1_aligned = h1_f.reindex(combined_idx).ffill().reindex(m15_f.index)

    # Birlashtirish
    dataset = pd.concat([m15_f, h1_aligned], axis=1)

    # Oxirgi bar
    last_row = dataset.iloc[-1]

    # Faqat kerakli featurelar
    x = np.array([last_row.get(f, 0.0) for f in feature_names], dtype=np.float32)

    # Normalize
    x = (x - scaler_mean) / (scaler_std + 1e-8)
    return x


# ──────────────────────────────────────────────
# PAPER TRADING STATE
# ──────────────────────────────────────────────

class PaperTrader:
    def __init__(self, capital: float):
        self.capital       = capital
        self.initial       = capital
        self.position      = None   # {'direction': 1/-1, 'entry': float, 'entry_time': ts, 'exit_time': ts}
        self.trades        = []
        self.trade_log     = Path(LOG_PATH)
        self.trade_log.parent.mkdir(exist_ok=True)

        # CSV header
        if not self.trade_log.exists():
            with open(self.trade_log, "w") as f:
                f.write("entry_time,exit_time,direction,entry_price,exit_price,ret,capital\n")

    def open_position(self, direction: int, price: float, ts):
        self.position = {
            "direction" : direction,
            "entry"     : price,
            "entry_time": ts,
            "exit_time" : ts + pd.Timedelta(minutes=15 * 8),  # 8 bar = 2 soat
        }
        side = "LONG" if direction == 1 else "SHORT"
        log.info(f"OPEN  {side:5s} @ {price:,.2f}  |  Capital: ${self.capital:,.2f}")

    def check_exit(self, current_price: float, current_time):
        if self.position is None:
            return
        if current_time < self.position["exit_time"]:
            return

        # Pozitsiyani yopish
        direction   = self.position["direction"]
        entry_price = self.position["entry"]
        ret         = direction * (current_price / entry_price - 1) - 0.0006

        pnl = self.capital * HALF_KELLY * ret
        self.capital += pnl

        side = "LONG" if direction == 1 else "SHORT"
        log.info(f"CLOSE {side:5s} @ {current_price:,.2f}  |  "
                 f"PnL: {ret*100:+.3f}%  |  Capital: ${self.capital:,.2f}")

        # Log
        with open(self.trade_log, "a") as f:
            f.write(f"{self.position['entry_time']},"
                    f"{current_time},"
                    f"{direction},"
                    f"{entry_price:.2f},"
                    f"{current_price:.2f},"
                    f"{ret:.6f},"
                    f"{self.capital:.2f}\n")

        self.trades.append(ret)
        self.position = None

    def stats(self):
        if not self.trades:
            return
        t = np.array(self.trades)
        total_ret = (self.capital / self.initial - 1) * 100
        log.info(f"── Stats ──  Trades:{len(t)}  "
                 f"WR:{(t>0).mean()*100:.1f}%  "
                 f"AvgPnL:{t.mean()*100:.3f}%  "
                 f"Return:{total_ret:.1f}%")


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────

def main():
    log.info("Paper trading boshlandi")
    log.info(f"Model: {MODEL_PATH}")
    log.info(f"Conf threshold: {CONF_THRESHOLD}")
    log.info(f"Capital: ${CAPITAL}")

    model, features, scaler_mean, scaler_std = load_model(MODEL_PATH)
    trader = PaperTrader(CAPITAL)

    log.info(f"Model yuklandi | Features: {len(features)}")

    while True:
        try:
            now = datetime.now(timezone.utc)
            log.info(f"── Yangi bar tekshirilmoqda: {now.strftime('%H:%M')} UTC ──")

            # Data olish
            m15 = fetch_klines(SYMBOL, "15m", limit=300)
            h1  = fetch_klines(SYMBOL, "1h",  limit=250)

            current_price = m15["close"].iloc[-1]
            current_time  = m15.index[-1]

            log.info(f"BTC narxi: ${current_price:,.2f}")

            # Ochiq pozitsiyani tekshirish
            trader.check_exit(current_price, current_time)

            # Yangi signal faqat pozitsiya yo'q bo'lganda
            if trader.position is None:
                x = compute_features(m15, h1, features, scaler_mean, scaler_std)

                with torch.no_grad():
                    logit = model(torch.tensor(x).unsqueeze(0))
                    prob  = torch.sigmoid(logit).item()

                confidence = max(prob, 1 - prob)
                direction  = 1 if prob > 0.5 else -1
                side       = "LONG" if direction == 1 else "SHORT"

                log.info(f"Signal: {side:5s}  Prob: {prob:.3f}  Conf: {confidence:.3f}")

                if confidence >= CONF_THRESHOLD:
                    trader.open_position(direction, current_price, current_time)
                else:
                    log.info(f"Signal zaif — savdo yo'q (conf={confidence:.3f} < {CONF_THRESHOLD})")
            else:
                entry = trader.position["entry"]
                exit_t = trader.position["exit_time"]
                log.info(f"Pozitsiya ochiq | Entry: ${entry:,.2f} | Exit: {exit_t.strftime('%H:%M')}")

            # Har 10 savdoda stats
            if len(trader.trades) % 10 == 0 and trader.trades:
                trader.stats()

            # 15 daqiqa kutish — keyingi M15 bar
            next_bar = (15 - now.minute % 15) * 60 - now.second + 5
            log.info(f"Keyingi bar: {next_bar//60}m {next_bar%60}s kutilmoqda...\n")
            time.sleep(next_bar)

        except KeyboardInterrupt:
            log.info("\nTo'xtatildi.")
            trader.stats()
            break
        except Exception as e:
            log.error(f"Xato: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
