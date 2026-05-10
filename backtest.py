"""
backtest.py
===========
Capigradmie — To'g'rilangan Backtest

Tuzatilgan xatolar:
    1. total_ret NaN edi — close.reindex NaN qoldirar edi,
       endi dropna() bilan tozalanadi
    2. Sharpe annualizatsiya — M15 uchun yiliga 35,040 bar,
       lekin Sharpe juda katta chiqardi. Sababini tekshirib,
       trade_ret bo'sh barlar (signal=0) ni o'z ichiga olar edi.
       Endi faqat trade bo'lgan barlar bo'yicha Sharpe hisoblanadi —
       bu "per-trade Sharpe" bo'ladi, real va tushunarli.
    3. Oylik PnL % sifatida ko'rsatiladi (log → exp)

Ishga tushirish:
    python3 backtest.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path


# ──────────────────────────────────────────────
# SOZLAMALAR
# ──────────────────────────────────────────────

MODEL_FILE   = Path("models/lgbm_model.pkl")
DATASET_FILE = Path("data/dataset.parquet")
M15_FILE     = Path("data/btcusdt_m15.csv")
OUTPUT_CSV   = Path("models/backtest_results.csv")

COST_BPS = 6   # 0.06% — Binance Futures taker fee (round-trip yarim)

THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# Annualizatsiya: faqat trade bo'lgan barlar bo'yicha
# M15 uchun yiliga ~35,040 bar → lekin signallar ~30-50% vaqt
# Per-trade Sharpe annualized: sqrt(n_trades_per_year)
# Biz avg trades/kun hisoblabmiz → yiliga
BARS_PER_DAY  = 96   # 24*60/15 = 96 bar/kun
DAYS_PER_YEAR = 365


# ──────────────────────────────────────────────
# ASOSIY FUNKSIYALAR
# ──────────────────────────────────────────────

def load_model_and_data():
    """Model va test datani yuklash."""
    with open(MODEL_FILE, "rb") as f:
        saved = pickle.load(f)

    model     = saved["model"]
    threshold = saved.get("threshold", 0.50)

    # Dataset
    df = pd.read_parquet(DATASET_FILE)

    # Konstant ustunlarni olib tashlash
    num_df     = df.select_dtypes(include=np.number)
    const_cols = num_df.columns[num_df.std() < 1e-10].tolist()
    if const_cols:
        df.drop(columns=const_cols, inplace=True)

    # Test set: oxirgi 15% (train.py bilan mos)
    n          = len(df)
    test_start = n - int(n * 0.15)
    df_test    = df.iloc[test_start:]

    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    # M15 close narxlari
    m15   = pd.read_csv(M15_FILE, index_col="open_time", parse_dates=True)
    close = m15["close"].reindex(df_test.index)

    # Log return: keyingi barda chiqish narxi
    log_ret = np.log(close).diff().shift(-1)

    print(f"── Yuklanmoqda ──")
    print(f"  Model threshold (o'qitilganda): {threshold:.2f}")
    print(f"  Test set: {len(df_test):,} bar  "
          f"({df_test.index[0].date()} → {df_test.index[-1].date()})")
    print(f"  Close NaN: {close.isna().sum()}")
    print(f"  LogRet NaN: {log_ret.isna().sum()} (oxirgi bar normal)")

    return model, X_test, y_test, log_ret, df_test.index


def run_backtest_threshold(model, X_test, log_ret, threshold):
    """
    Bitta threshold uchun backtest.

    Returns dict: n_trades, signal_pct, win_rate,
                  total_ret_pct, sharpe, max_dd, profit_factor, avg_trade_pct
    """
    proba     = model.predict(X_test)
    pred_cls  = proba.argmax(axis=1)   # 0=Short, 1=No-edge, 2=Long
    max_prob  = proba.max(axis=1)

    # Signal: threshold dan yuqori confidence da
    direction = pred_cls - 1           # -1, 0, +1
    signal    = np.where(max_prob >= threshold, direction, 0)

    # Trade return (NaN bo'lgan oxirgi barni o'chirish)
    lr    = log_ret.values
    valid = ~np.isnan(lr)

    sig_v = signal[valid]
    lr_v  = lr[valid]

    cost      = np.abs(sig_v) * (COST_BPS / 10_000)
    trade_ret = sig_v * lr_v - cost

    # Faqat trade bo'lgan barlar
    trade_mask = sig_v != 0
    n_trades   = trade_mask.sum()

    if n_trades == 0:
        return None

    trade_rets = trade_ret[trade_mask]   # faqat active trade barlar

    # Win rate
    win_rate = (trade_rets > 0).mean() * 100

    # Total return (kumulativ log → %)
    total_log = trade_ret.sum()          # barcha barlar (0 * lr = 0 ta'sir qilmaydi)
    total_ret_pct = (np.exp(total_log) - 1) * 100

    # Equity curve (barcha barlar, kumulativ)
    equity = np.exp(np.cumsum(trade_ret)) - 1  # % sifatida

    # Max drawdown
    roll_max  = np.maximum.accumulate(equity + 1)
    drawdown  = (equity + 1) / roll_max - 1
    max_dd    = drawdown.min() * 100

    # Sharpe — per-trade annualized
    # n_trades_per_year = n_trades / test_days * 365
    n_bars    = valid.sum()
    test_days = n_bars / BARS_PER_DAY
    trades_per_year = n_trades / test_days * DAYS_PER_YEAR

    if trade_rets.std() > 0:
        sharpe = (trade_rets.mean() / trade_rets.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Profit factor
    gross_win  = trade_rets[trade_rets > 0].sum()
    gross_loss = abs(trade_rets[trade_rets < 0].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Signal %
    signal_pct = trade_mask.mean() * 100

    # Avg trade %
    avg_trade_pct = trade_rets.mean() * 100

    return dict(
        n_trades      = n_trades,
        signal_pct    = signal_pct,
        win_rate      = win_rate,
        total_ret_pct = total_ret_pct,
        sharpe        = sharpe,
        max_dd        = max_dd,
        profit_factor = pf,
        avg_trade_pct = avg_trade_pct,
        trade_ret_arr = trade_ret,   # oylik tahlil uchun
        signal_arr    = signal,
    )


def monthly_pnl(trade_ret_arr, index, signal_arr):
    """Oylik PnL hisoblash."""
    valid = ~np.isnan(np.array([0.0] * len(index)))  # placeholder
    ret_s = pd.Series(trade_ret_arr[:len(index)], index=index)

    monthly = ret_s.resample("ME").sum()
    monthly_pct = (np.exp(monthly) - 1) * 100
    return monthly_pct


def print_results_table(results):
    """Threshold jadvali."""
    print(f"\n── Threshold bo'yicha natijalar ──")
    print(f"   {'Thresh':>6}  {'Trades':>7}  {'Signal%':>7}  "
          f"{'WinRate':>7}  {'TotalRet%':>9}  {'Sharpe':>7}  "
          f"{'MaxDD':>7}  {'PF':>5}")
    print("  " + "─" * 75)

    for thr, r in results.items():
        if r is None:
            continue
        print(f"   {thr:>6.2f}  {r['n_trades']:>7,}  {r['signal_pct']:>6.1f}%  "
              f"{r['win_rate']:>6.1f}%  {r['total_ret_pct']:>+9.2f}%  "
              f"{r['sharpe']:>7.2f}  {r['max_dd']:>+7.2f}%  {r['profit_factor']:>5.2f}")


# ──────────────────────────────────────────────
# MASTER PIPELINE
# ──────────────────────────────────────────────

def run():
    print("=" * 60)
    print("  CAPIGRADMIE — Backtest (To'g'rilangan)")
    print("=" * 60)

    model, X_test, y_test, log_ret, idx = load_model_and_data()

    print(f"\n── {len(THRESHOLDS)} threshold sinab ko'rilmoqda ──")

    results = {}
    for thr in THRESHOLDS:
        r = run_backtest_threshold(model, X_test, log_ret, thr)
        results[thr] = r

    print_results_table(results)

    # Eng yaxshi threshold (Sharpe bo'yicha)
    valid_results = {t: r for t, r in results.items() if r is not None}
    best_thr = max(valid_results, key=lambda t: valid_results[t]["sharpe"])
    best     = valid_results[best_thr]

    print(f"\n{'='*60}")
    print(f"  ✅ ENG YAXSHI: threshold = {best_thr}")
    print(f"{'='*60}")
    print(f"  Jami trade        : {best['n_trades']:,}")
    print(f"  Signal %          : {best['signal_pct']:.1f}%")
    print(f"  Win Rate          : {best['win_rate']:.1f}%")
    print(f"  Total Return      : {best['total_ret_pct']:+.2f}%")
    print(f"  Sharpe Ratio      : {best['sharpe']:.2f}")
    print(f"  Max Drawdown      : {best['max_dd']:.2f}%")
    print(f"  Profit Factor     : {best['profit_factor']:.2f}")
    print(f"  Avg Trade         : {best['avg_trade_pct']:.4f}%")

    # Buy & Hold solishtirish
    m15   = pd.read_csv(M15_FILE, index_col="open_time", parse_dates=True)
    close = m15["close"].reindex(idx)
    bnh   = (np.exp(np.log(close.dropna()).diff().sum()) - 1) * 100
    print(f"\n  📊 Buy & Hold     : {bnh:+.2f}%  (solishtiruv uchun)")
    print(f"  📊 Alpha          : {best['total_ret_pct'] - bnh:+.2f}%")

    # Oylik PnL
    trade_ret = best["trade_ret_arr"]
    ret_s     = pd.Series(trade_ret[:len(idx)], index=idx)
    monthly   = (np.exp(ret_s.resample("ME").sum()) - 1) * 100

    print(f"\n── Oylik PnL (threshold={best_thr}) ──")
    pos_months = 0
    neg_months = 0
    for period, val in monthly.items():
        if abs(val) < 0.01:
            continue
        icon = "✅" if val >= 0 else "❌"
        bar  = "█" * min(30, int(abs(val) / 3))
        print(f"  {icon} {period.strftime('%Y-%m')}: {val:+.2f}%  {bar}")
        if val >= 0:
            pos_months += 1
        else:
            neg_months += 1

    print(f"\n  Musbat oylar: {pos_months}  |  Manfiy oylar: {neg_months}")

    # CSV saqlash
    rows = []
    for thr, r in valid_results.items():
        rows.append(dict(
            threshold    = thr,
            n_trades     = r["n_trades"],
            signal_pct   = r["signal_pct"],
            win_rate     = r["win_rate"],
            total_ret    = r["total_ret_pct"],
            sharpe       = r["sharpe"],
            max_dd       = r["max_dd"],
            profit_factor= r["profit_factor"],
            avg_trade    = r["avg_trade_pct"],
        ))

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"\n  💾 Natijalar saqlandi: {OUTPUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    run()