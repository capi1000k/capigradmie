"""
Futures Trading v2
==================
Tuzatishlar:
  A. Kapital — Binance dan real balans olinadi, $200 emas
  C. Hurst cache — dataset dan olinadi, qayta hisoblanmaydi
"""

import os, sys, time, logging, csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import *
from datetime import datetime, timezone
from pathlib import Path

load_dotenv()
sys.path.insert(0, '/home/zanerhon/capigradmie')

# ── SOZLAMALAR ──
SYMBOL         = "BTCUSDT"
LEVERAGE       = 1
HALF_KELLY     = 0.081
CONF_THRESHOLD = 0.70
MODEL_PATH     = "model/mlp_fold4.pt"
LOG_PATH       = "live/futures_trades.csv"
MAX_CAPITAL    = 300.0   # USDT — max ishlatish

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("live/futures_v2.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── MODEL ──
class TradingMLP(nn.Module):
    def __init__(self, input_dim, hidden=[256,128,64], dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev,h), nn.BatchNorm1d(h),
                      nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TradingMLP(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return (model,
            ckpt["features"],
            ckpt["scaler_mean"].astype(np.float32),
            ckpt["scaler_std"].astype(np.float32))

# ── VARIANT C: HURST CACHE ──
def load_hurst_cache():
    """Hurst ni dataset dan oladi — qayta hisoblanmaydi (10 daqiqa tejaladi)"""
    try:
        cache = pd.read_parquet('data/hurst_cache.parquet')
        log.info(f"Hurst cache yuklandi: {len(cache):,} bar")
        return cache
    except:
        log.warning("Hurst cache topilmadi — dataset dan olinadi")
        ds = pd.read_parquet('data/dataset_v2.parquet')
        return ds[['hurst_100', 'h1_hurst_100']]

# ── FEATURELAR ──
def get_features(client, features, scaler_mean, scaler_std, hurst_cache):
    from features import (add_base_features, add_technical_features,
                          add_microstructure_features, add_smart_money_features,
                          add_time_features)

    def klines_to_df(klines):
        df = pd.DataFrame(klines, columns=[
            'open_time','open','high','low','close','volume',
            'close_time','qv','trades','tbb','tbq','ignore'])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df = df.set_index('open_time')
        for c in ['open','high','low','close','volume']:
            df[c] = df[c].astype(float)
        return df[['open','high','low','close','volume']].iloc[:-1]

    m15 = klines_to_df(client.futures_klines(
        symbol=SYMBOL, interval='15m', limit=300))
    h1  = klines_to_df(client.futures_klines(
        symbol=SYMBOL, interval='1h',  limit=250))

    current_price = float(m15['close'].iloc[-1])
    current_time  = m15.index[-1]

    # M15 features (Hurst olib tashlanadi — cache dan olinadi)
    m15_f = add_base_features(m15.copy())
    m15_f = add_technical_features(m15_f)
    m15_f = add_microstructure_features(m15_f)

    # VARIANT C: smart_money dan faqat Hurst olib tashlanadi
    # Qolgan smart money featurelar hisoblanadi
    from features.smart_money import (detect_swings, classify_swing_sequence,
                                       break_of_structure, volatility_compression,
                                       price_convexity, kyle_lambda)
    h_sm, l_sm = m15['high'], m15['low']
    c_sm, v_sm = m15['close'], m15['volume']

    swing_df = detect_swings(h_sm, l_sm, n=3)
    m15_f = pd.concat([m15_f, swing_df], axis=1)
    seq_df = classify_swing_sequence(swing_df, h_sm, l_sm)
    m15_f = pd.concat([m15_f, seq_df], axis=1)
    bos_df = break_of_structure(c_sm, swing_df)
    m15_f = pd.concat([m15_f, bos_df], axis=1)
    m15_f['vcp_compression'] = volatility_compression(h_sm, l_sm)
    m15_f['convexity_20']    = price_convexity(c_sm)
    m15_f['kyle_lambda']     = kyle_lambda(c_sm, v_sm)

    # Hurst — cache dan oxirgi mavjud qiymat
    if current_time in hurst_cache.index:
        m15_f['hurst_100'] = hurst_cache.loc[current_time, 'hurst_100']
    else:
        # Cache da yo'q — oxirgi mavjud qiymatni ishlatamiz
        last_hurst = hurst_cache['hurst_100'].dropna().iloc[-1]
        m15_f['hurst_100'] = last_hurst
        log.warning(f"Hurst cache da yo'q — oxirgi qiymat: {last_hurst:.3f}")

    m15_f = add_time_features(m15_f, bar_minutes=15)
    m15_f = m15_f.drop(
        columns=['open','high','low','close','volume'], errors='ignore')

    # H1 features
    h1_f = add_base_features(h1.copy())
    h1_f = add_technical_features(h1_f)
    h1_f = add_microstructure_features(h1_f)

    h_h1, l_h1 = h1['high'], h1['low']
    c_h1, v_h1 = h1['close'], h1['volume']
    swing_h1 = detect_swings(h_h1, l_h1, n=3)
    h1_f = pd.concat([h1_f, swing_h1], axis=1)
    seq_h1 = classify_swing_sequence(swing_h1, h_h1, l_h1)
    h1_f = pd.concat([h1_f, seq_h1], axis=1)
    bos_h1 = break_of_structure(c_h1, swing_h1)
    h1_f = pd.concat([h1_f, bos_h1], axis=1)
    h1_f['vcp_compression'] = volatility_compression(h_h1, l_h1)
    h1_f['convexity_20']    = price_convexity(c_h1)
    h1_f['kyle_lambda']     = kyle_lambda(c_h1, v_h1)

    # H1 Hurst — cache dan
    h1_last_time = h1.index[-1]
    if h1_last_time in hurst_cache.index:
        h1_f['hurst_100'] = hurst_cache.loc[h1_last_time, 'h1_hurst_100']
    else:
        last_h1_hurst = hurst_cache['h1_hurst_100'].dropna().iloc[-1]
        h1_f['hurst_100'] = last_h1_hurst

    h1_f = h1_f.drop(
        columns=['open','high','low','close','volume'], errors='ignore')
    h1_f = h1_f.add_prefix('h1_')

    # Align H1 → M15
    combined = h1_f.index.union(m15_f.index).sort_values()
    h1_aligned = h1_f.reindex(combined).ffill().reindex(m15_f.index)
    dataset = pd.concat([m15_f, h1_aligned], axis=1)

    last = dataset.iloc[-1]
    x = np.array([last.get(f, 0.0) for f in features], dtype=np.float32)
    x = (x - scaler_mean) / (scaler_std + 1e-8)
    return x, current_price, current_time

# ── VARIANT A: HAQIQIY KAPITAL ──
def get_real_capital(client):
    """Binance dan haqiqiy USDT balansni oladi"""
    balance = client.futures_account_balance()
    for b in balance:
        if b['asset'] == 'USDT':
            real = float(b['balance'])
            # MAX_CAPITAL dan oshmasin
            return min(real, MAX_CAPITAL)
    return MAX_CAPITAL

def get_position(client):
    positions = client.futures_position_information(symbol=SYMBOL)
    for p in positions:
        amt = float(p['positionAmt'])
        if abs(amt) > 0:
            return {
                'side'  : 'LONG' if amt > 0 else 'SHORT',
                'amount': abs(amt),
                'entry' : float(p['entryPrice']),
            }
    return None

def open_order(client, direction, price, capital):
    quantity = 0.002
    while quantity * price < 100.0:
        quantity = round(quantity + 0.001, 3)

    side = SIDE_BUY if direction == 1 else SIDE_SELL
    order = client.futures_create_order(
        symbol   = SYMBOL,
        side     = side,
        type     = ORDER_TYPE_MARKET,
        quantity = quantity,
    )
    log.info(f"ORDER: {order['side']} {quantity} BTC @ ~{price:,.2f} "
             f"(notional: ${quantity*price:.0f})")
    return order, quantity

def close_order(client, position):
    side = SIDE_SELL if position['side'] == 'LONG' else SIDE_BUY
    order = client.futures_create_order(
        symbol     = SYMBOL,
        side       = side,
        type       = ORDER_TYPE_MARKET,
        quantity   = position['amount'],
        reduceOnly = True,
    )
    log.info(f"CLOSE: {position['side']} {position['amount']} BTC")
    return order

# ── LOG FAYL ──
def init_log():
    Path(LOG_PATH).parent.mkdir(exist_ok=True)
    if not Path(LOG_PATH).exists():
        with open(LOG_PATH, 'w', newline='') as f:
            csv.writer(f).writerow([
                'time','side','entry','exit','qty','ret_pct','pnl_usdt','capital'
            ])

def write_trade(ts, side, entry, exit_price, qty, capital):
    direction = 1 if side == 'LONG' else -1
    ret = direction * (exit_price / entry - 1) - 0.0006
    pnl = capital * HALF_KELLY * ret
    with open(LOG_PATH, 'a', newline='') as f:
        csv.writer(f).writerow([
            ts.strftime('%Y-%m-%d %H:%M'),
            side,
            f"{entry:.2f}",
            f"{exit_price:.2f}",
            qty,
            f"{ret*100:.4f}",
            f"{pnl:.4f}",
            f"{capital:.4f}",
        ])
    return ret, pnl

# ── MAIN ──
def main():
    log.info("="*55)
    log.info("Futures Trading v2 boshlandi")
    log.info(f"Conf: {CONF_THRESHOLD} | Kelly: {HALF_KELLY} | Max: ${MAX_CAPITAL}")

    client = Client(
        os.getenv('BINANCE_API_KEY'),
        os.getenv('BINANCE_API_SECRET'),
        testnet=True
    )
    client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'

    client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)

    model, features, scaler_mean, scaler_std = load_model(MODEL_PATH)
    hurst_cache = load_hurst_cache()

    # VARIANT A: haqiqiy kapital
    capital = get_real_capital(client)
    log.info(f"Haqiqiy kapital: ${capital:,.2f} USDT")
    log.info("="*55)

    init_log()
    entry_info = {}

    while True:
        try:
            now = datetime.now(timezone.utc)
            log.info(f"── {now.strftime('%H:%M')} UTC ──")

            # VARIANT A: har doim real kapital
            capital = get_real_capital(client)

            # Feature va narx
            x, price, current_time = get_features(
                client, features, scaler_mean, scaler_std, hurst_cache)
            log.info(f"BTC: ${price:,.2f} | Kapital: ${capital:,.2f}")

            position = get_position(client)

            if position:
                elapsed = None
                if 'open_time' in entry_info:
                    elapsed = (now - entry_info['open_time']).total_seconds() / 60
                    remaining = 120 - elapsed
                    log.info(f"Pozitsiya: {position['side']} | "
                            f"Entry: ${position['entry']:,.2f} | "
                            f"Unrealized: ${(price/position['entry']-1)*100:.3f}% | "
                            f"Qoldi: {remaining:.0f} min")

                    if elapsed >= 120:
                        # Yopish
                        close_order(client, position)
                        ret, pnl = write_trade(
                            now, position['side'],
                            position['entry'], price,
                            position['amount'], capital
                        )
                        log.info(f"PnL: {ret*100:+.3f}% | ${pnl:+.4f}")
                        entry_info = {}

                        # Dashboard yangilash
                        try:
                            from live.visualize import plot_dashboard
                            plot_dashboard()
                            log.info("Dashboard yangilandi")
                        except Exception as ve:
                            log.warning(f"Vizualizatsiya: {ve}")
                else:
                    # entry_info yo'q — restart bo'lgan
                    log.info(f"Pozitsiya bor lekin entry_info yo'q — "
                            f"{position['side']} @ ${position['entry']:,.2f}")
                    entry_info = {
                        'open_time': now,
                        'direction': 1 if position['side'] == 'LONG' else -1
                    }
            else:
                # Signal
                with torch.no_grad():
                    logit = model(torch.tensor(x).unsqueeze(0))
                    prob  = torch.sigmoid(logit).item()

                conf      = max(prob, 1 - prob)
                direction = 1 if prob > 0.5 else -1
                side      = "LONG" if direction == 1 else "SHORT"
                log.info(f"Signal: {side} | Prob: {prob:.3f} | Conf: {conf:.3f}")

                if conf >= CONF_THRESHOLD:
                    order, qty = open_order(client, direction, price, capital)
                    entry_info = {
                        'open_time': now,
                        'direction': direction,
                        'qty'      : qty,
                    }
                    log.info(f"Pozitsiya ochildi: {side} @ ${price:,.2f}")
                else:
                    log.info(f"Signal zaif — savdo yo'q")

            # 15 daqiqa kutish
            next_bar = (15 - now.minute % 15) * 60 - now.second + 10
            log.info(f"Keyingi bar: {next_bar//60}m {next_bar%60}s\n")
            time.sleep(next_bar)

        except KeyboardInterrupt:
            log.info("To'xtatildi.")
            pos = get_position(client)
            if pos:
                log.info("Ochiq pozitsiya yopilmoqda...")
                close_order(client, pos)
            break
        except Exception as e:
            log.error(f"Xato: {e}")
            import traceback
            log.error(traceback.format_exc())
            time.sleep(60)

if __name__ == '__main__':
    main()
