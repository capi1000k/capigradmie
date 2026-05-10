"""
backtest.py
===========
Model signallari asosida vektorlashtirilgan backtest.

Metodologiya:
    - Test set da model signallarini ishlatamiz
    - Har signal uchun: entry close[i], exit close[i+1] (next bar)
    - Transaction cost: round-trip 0.06%
    - Position sizing: fixed (har trade 1 unit kapital)
    - Leverage yo'q (natijani toza ko'rish uchun)

Ko'rsatkichlar:
    Total Return     — jami daromad %
    Sharpe Ratio     — risk-adjusted return (annualized)
    Max Drawdown     — eng katta cho'kish %
    Win Rate         — g'olibiyat foizi
    Profit Factor    — jami foyda / jami zarar
    Signal Count     — necha trade
    Avg Trade        — o'rtacha trade daromadi

Ishga tushirish:
    python3 backtest.py
"""

import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path


# ──────────────────────────────────────────────
# SOZLAMALAR
# ──────────────────────────────────────────────

DATASET_FILE  = Path("data/dataset.parquet")
M15_FILE      = Path("data/btcusdt_m15.csv")
MODEL_FILE    = Path("models/lgbm_model.pkl")
RESULTS_FILE  = Path("models/backtest_results.csv")

COST          = 0.0006    # round-trip 0.06%
BARS_PER_YEAR = 365 * 24 * 4   # M15: 35,040 bar/yil

# Sinab ko'rish uchun threshold ro'yxati
TEST_THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


# ──────────────────────────────────────────────
# 1. YUKLASH
# ──────────────────────────────────────────────

def load_all():
    print("── Yuklanmoqda ──")

    # Model
    with open(MODEL_FILE, "rb") as f:
        saved = pickle.load(f)

    model         = saved["model"]
    label_map_inv = saved["label_map_inv"]
    feature_names = saved["feature_names"]
    print(f"  Model: {MODEL_FILE}")

    # Dataset
    df = pd.read_parquet(DATASET_FILE)

    # Konstant ustunlar
    num_df     = df.select_dtypes(include=np.number)
    const_cols = num_df.columns[num_df.std() < 1e-10].tolist()
    if const_cols:
        df.drop(columns=const_cols, inplace=True)

    X = df.drop(columns=["label"])
    y = df["label"]

    # Raw M15 narxlar (exit uchun kerak)
    m15 = pd.read_csv(M15_FILE, index_col="open_time", parse_dates=True)
    m15 = m15[~m15.index.duplicated(keep="last")].sort_index()
    if m15.index.tzinfo is None:
        m15.index = m15.index.tz_localize("UTC")

    # Test set — oxirgi 10%
    n          = len(X)
    test_start = int(n * 0.90)
    X_test     = X.iloc[test_start:]
    y_test     = y.iloc[test_start:]
    idx_test   = df.index[test_start:]

    print(f"  Test set: {len(X_test):,} bar  "
          f"({idx_test[0].date()} → {idx_test[-1].date()})")

    # Prediction
    proba = model.predict(X_test)

    return proba, y_test, idx_test, m15, label_map_inv, feature_names


# ──────────────────────────────────────────────
# 2. BACKTEST CORE
# ──────────────────────────────────────────────

def run_backtest(proba: np.ndarray,
                 idx: pd.DatetimeIndex,
                 m15: pd.DataFrame,
                 threshold: float,
                 cost: float = COST) -> dict:
    """
    Vektorlashtirilgan backtest.

    Signal logikasi:
        max(proba) >= threshold  AND  argmax != 1 (no-edge)
        → Long (+1) yoki Short (-1) signal

    Trade logikasi:
        Entry: signal bari yopilganda (close[i])
        Exit:  keyingi bar yopilganda (close[i+1])
        PnL:   log(close[i+1] / close[i]) × direction - cost
    """
    pred_class = proba.argmax(axis=1)   # 0=short, 1=no-edge, 2=long
    max_proba  = proba.max(axis=1)

    # Label: 0→-1, 1→0, 2→+1
    direction = pred_class - 1          # {-1, 0, +1}

    # Threshold filter
    signal_mask = (max_proba >= threshold) & (direction != 0)
    signals     = direction * signal_mask.astype(int)

    # M15 close narxlarini test indeksiga align qilish
    close = m15["close"].reindex(idx)

    # PnL hisoblash
    log_ret = np.log(close).diff().shift(-1)   # keyingi bar return

    trade_returns = signals * log_ret - np.abs(signals) * cost

    # Kunlik equity curve (cumulative)
    equity = trade_returns.cumsum()

    # Ko'rsatkichlar
    trades      = trade_returns[signal_mask]
    n_trades    = signal_mask.sum()
    win_rate    = (trades > 0).mean() if n_trades > 0 else 0
    total_ret   = equity.iloc[-1] if len(equity) > 0 else 0

    # Sharpe (annualized)
    if trade_returns.std() > 0:
        sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(BARS_PER_YEAR)
    else:
        sharpe = 0.0

    # Max Drawdown
    cummax    = equity.cummax()
    drawdown  = equity - cummax
    max_dd    = drawdown.min()

    # Profit Factor
    gross_profit = trades[trades > 0].sum()
    gross_loss   = trades[trades < 0].abs().sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Avg trade
    avg_trade = trades.mean() if n_trades > 0 else 0

    return {
        "threshold":   threshold,
        "n_trades":    int(n_trades),
        "signal_pct":  signal_mask.mean() * 100,
        "win_rate":    win_rate * 100,
        "total_ret":   total_ret * 100,
        "sharpe":      sharpe,
        "max_dd":      max_dd * 100,
        "profit_factor": pf,
        "avg_trade":   avg_trade * 100,
        "equity":      equity,
        "trade_returns": trade_returns,
    }


# ──────────────────────────────────────────────
# 3. HISOBOT
# ──────────────────────────────────────────────

def print_results_table(results: list):
    """Barcha threshold uchun natijalar jadvali."""
    print("\n── Threshold bo'yicha natijalar ──")
    print(f"  {'Thresh':>7} {'Trades':>7} {'Signal%':>8} "
          f"{'WinRate':>8} {'TotalRet':>9} {'Sharpe':>7} "
          f"{'MaxDD':>7} {'PF':>6}")
    print("  " + "─" * 70)

    for r in results:
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != np.inf else "  ∞"
        print(
            f"  {r['threshold']:>7.2f} "
            f"{r['n_trades']:>7,} "
            f"{r['signal_pct']:>7.1f}% "
            f"{r['win_rate']:>7.1f}% "
            f"{r['total_ret']:>8.2f}% "
            f"{r['sharpe']:>7.2f} "
            f"{r['max_dd']:>6.2f}% "
            f"{pf_str:>6}"
        )


def print_best_result(best: dict):
    """Eng yaxshi threshold uchun to'liq hisobot."""
    print(f"\n{'='*55}")
    print(f"  ✅ ENG YAXSHI: threshold = {best['threshold']:.2f}")
    print(f"{'='*55}")
    print(f"  Jami trade        : {best['n_trades']:,}")
    print(f"  Signal %          : {best['signal_pct']:.1f}%")
    print(f"  Win Rate          : {best['win_rate']:.1f}%")
    print(f"  Total Return      : {best['total_ret']:.2f}%  (log return)")
    print(f"  Sharpe Ratio      : {best['sharpe']:.2f}")
    print(f"  Max Drawdown      : {best['max_dd']:.2f}%")
    pf = best['profit_factor']
    pf_str = f"{pf:.2f}" if pf != np.inf else "∞"
    print(f"  Profit Factor     : {pf_str}")
    print(f"  Avg Trade         : {best['avg_trade']:.4f}%")

    # Baholash
    print(f"\n── Baholash ──")
    if best['sharpe'] > 1.5:
        print(f"  ✅ Sharpe {best['sharpe']:.2f} — yaxshi (1.5+ maqsad)")
    elif best['sharpe'] > 1.0:
        print(f"  ⚠️  Sharpe {best['sharpe']:.2f} — o'rtacha (1.5+ maqsad)")
    else:
        print(f"  ❌ Sharpe {best['sharpe']:.2f} — past, model yaxshilanishi kerak")

    if best['max_dd'] > -20:
        print(f"  ✅ MaxDD {best['max_dd']:.1f}% — boshqarsa bo'ladi")
    else:
        print(f"  ⚠️  MaxDD {best['max_dd']:.1f}% — katta, position sizing muhim")

    if best['win_rate'] > 50:
        print(f"  ✅ Win Rate {best['win_rate']:.1f}%")
    else:
        print(f"  ⚠️  Win Rate {best['win_rate']:.1f}% — profit factor muhimroq")


def print_monthly_breakdown(best: dict, idx: pd.DatetimeIndex):
    """Oylik PnL taqsimoti."""
    print(f"\n── Oylik PnL (eng yaxshi threshold) ──")

    equity = best["equity"]
    monthly = equity.groupby([idx.year, idx.month]).last().diff()
    monthly.index = [f"{y}-{m:02d}" for y, m in monthly.index]

    positive = 0
    negative = 0

    for period, ret in monthly.items():
        if pd.isna(ret):
            continue
        sign = "+" if ret >= 0 else ""
        bar  = "█" * min(int(abs(ret) * 200), 30)
        color_hint = "✅" if ret >= 0 else "❌"
        print(f"  {color_hint} {period}: {sign}{ret*100:.2f}%  {bar}")
        if ret >= 0:
            positive += 1
        else:
            negative += 1

    print(f"\n  Musbat oylar: {positive}  |  Manfiy oylar: {negative}")


# ──────────────────────────────────────────────
# 4. MASTER PIPELINE
# ──────────────────────────────────────────────

def backtest():
    print("=" * 55)
    print("  CAPIGRADMIE — Backtest")
    print("=" * 55)

    proba, y_test, idx_test, m15, label_map_inv, feature_names = load_all()

    # Barcha threshold uchun backtest
    print(f"\n── {len(TEST_THRESHOLDS)} threshold sinab ko'rilmoqda ──")

    all_results = []
    for thr in TEST_THRESHOLDS:
        r = run_backtest(proba, idx_test, m15, threshold=thr)
        all_results.append(r)

    # Jadval
    print_results_table(all_results)

    # Eng yaxshi: Sharpe bo'yicha
    best = max(all_results, key=lambda r: r["sharpe"])

    # To'liq hisobot
    print_best_result(best)

    # Oylik breakdown
    print_monthly_breakdown(best, idx_test)

    # Saqlash
    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("equity", "trade_returns")}
        for r in all_results
    ])
    summary.to_csv(RESULTS_FILE, index=False)
    print(f"\n  💾 Natijalar saqlandi: {RESULTS_FILE}")

    return all_results, best


# ──────────────────────────────────────────────
# ISHGA TUSHIRISH
# ──────────────────────────────────────────────

if __name__ == "__main__":
    all_results, best = backtest()