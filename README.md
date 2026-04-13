# capigradmicat > README.md << 'EOF'
# Capigradmie — BTC/USDT ML Trading System

> 18 yoshli o'zbek yigiti tomonidan noldan qurilgan professional ML trading pipeline.
> Tashkent, O'zbekiston. 2025-2026.

---

## Loyiha haqida

Bu loyiha BTC/USDT M15 timeframe uchun Machine Learning asosida ishlaydi.  
Model narx yo'nalishini bashorat qilib, Binance Futures da avtomatik savdo qiladi.

**Asosiy natijalar:**
- Walk-forward backtest: har davrda musbat return
- Confidence > 0.70 filtr bilan Win Rate: **67-70%**
- Avg PnL per trade: **0.242%** (cost dan keyin)
- Max Drawdown: **< 2%** (hech qachon)
- Shuffle test: **49.6%** → look-ahead bias yo'q

---

## Arxitektura

Raw OHLCV Data (M15 + H1)
↓
Feature Engineering (129 feature)
↓
Triple Barrier Labeling
↓
PyTorch MLP Model
↓
Confidence Filter (> 0.70)
↓
Binance Futures Order

---

## Feature Engineering

### `features/base_features.py`
- Log return, rolling log return
- Candle anatomy: body, upper/lower wick, ratios
- ATR, relative spread, rolling volatility

### `features/technical.py`
- EMA (20, 50, 200): distance, slope, convergence
- RSI (14) + velocity
- ATR ratio (7/50)
- MACD: line, signal, histogram
- Stochastic %K/%D
- Rolling skewness, kurtosis (50)

### `features/microstructure.py`
Institutsional darajadagi proxy'lar:
- **Roll Spread** — bid-ask spread estimator (Roll 1984)
- **Amihud Illiquidity** — narx ta'siri proxy (Amihud 2002)
- **Volume Delta** — order flow yo'nalishi
- **BVC** — Bulk Volume Classification (Easley et al.)
- **Effort vs Result** — Wyckoff prinsipi
- **OBV Z-score** — On-Balance Volume
- **Kyle's Lambda** — price impact slope (Kyle 1985)

### `features/smart_money.py`
- Swing High/Low detection (n-bar confirmation)
- HH/HL/LH/LL klassifikatsiya
- BOS (Break of Structure) — tuzatilgan CHoCH bilan
- Distance to swing levels
- VCP (Volatility Contraction Pattern)
- Price-path convexity
- **Hurst Exponent** — bozor rejimi deteksiyasi

### `features/time_features.py`
- Sin/cos hour, day of week, day of month, month
- Trading session flags: Asian, London, NY, Overlap
- Intraday position

---

## Target: Triple Barrier

Lopez de Prado ("Advances in Financial ML") metodologiyasi:

Har M15 bar uchun:
Upper barrier: entry + 1.5 × ATR  (TP)
Lower barrier: entry - 1.5 × ATR  (SL)
Time barrier : 8 bar (2 soat)
First barrier hit:
Upper → label +1 (long)
Lower → label -1 (short)
Time  → label  0 (no-edge)
Transaction cost: 0.06% round-trip

**Label taqsimoti:**
short   : 39.1%
no-edge : 22.7%
long    : 38.2%


---

## Model

### PyTorch MLP
```python
Input(125)
  → Linear(256) → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(128) → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(64)  → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(1)   → Sigmoid
```

**Overfitting oldini olish:**
- Early Stopping (patience=12)
- BatchNorm + Dropout(0.3)
- AdamW (lr=1e-3, weight_decay=1e-4)
- ReduceLROnPlateau scheduler
- Walk-forward validation (time-aware, 4 fold)

---

## Backtest Natijalari

### Walk-forward (conf > 0.70, Half-Kelly 8.1%)

| Davr | Win Rate | Return | Max DD | Trades |
|------|----------|--------|--------|--------|
| 2024 Q1 | 63.8% | +40.4% | 0.5% | 1,786 |
| 2024 Q3 | 68.0% | +45.4% | 0.6% | 1,715 |
| 2025 Q1 | 66.9% | +41.2% | 0.5% | 1,753 |
| 2025 Q3 | 66.8% | +65.4% | 0.6% | 2,705 |

**Jami avg PnL/trade: 0.242%**

### Bias tekshiruvi
| Test | Natija | Xulosa |
|------|--------|--------|
| Shuffle test | 49.6% ≈ 50% | Look-ahead bias YO'Q |
| Walk-forward | Har davrda musbat | Overfitting YO'Q |
| Tautology test | H1 filter labeldan olib tashlandi | Strukturaviy bias YO'Q |
| Swing price test | match=True | Narx to'g'ri |

---

## Risk Management

**Half-Kelly Criterion:**

Win Rate  : 56.7%
Avg Win   : 0.605%
Avg Loss  : 0.540%
Risk/Reward: 1.12
Kelly %   : 16.2%
Half-Kelly: 8.1%  ← ishlatiladi

**Per trade risk:** kapitalning 8.1%

---

## Live Trading

Binance Futures Testnet da real-time savdo:

```bash
PYTHONPATH=/path/to/capigradmie python3 live/futures_trading_v2.py
```

**Sozlamalar:**
- Symbol: BTCUSDT
- Leverage: 1x
- Confidence threshold: 0.70
- Half-Kelly: 8.1%
- Max capital: $300 USDT
- Exit: 8 bar (2 soat) dan keyin

**Birinchi savdo:**
Date  : 2026-04-13
Signal: SHORT (Conf: 0.965)
Entry : $70,994.70
Result: ✅

---

## O'rnatish

```bash
git clone https://github.com/capi1000k/capigradmie.git
cd capigradmie

# Virtual environment
uv venv
source .venv/bin/activate

# Kutubxonalar
uv pip install pandas numpy torch lightgbm scikit-learn \
               matplotlib pyarrow python-binance python-dotenv

# .env fayl
cp .env.example .env
# .env ga Binance API kalitlarini kiriting

# Dataset qurish (12-15 daqiqa)
python3 build_dataset.py

# Model o'qitish
python3 model/train_v2.py

# Live trading
python3 live/futures_trading_v2.py
```

---

## Fayl tuzilmasi
capigradmie/
├── features/
│   ├── init.py
│   ├── base_features.py
│   ├── technical.py
│   ├── microstructure.py
│   ├── smart_money.py
│   └── time_features.py
├── model/
│   ├── train_v2.py          # PyTorch MLP training
│   └── mlp_fold{1-4}.pt     # Saqlangan modellar
├── live/
│   ├── futures_trading_v2.py # Live trading
│   ├── visualize.py          # Dashboard
│   └── paper_trading.py      # Paper trade
├── data/
│   ├── btcusdt_m15.csv       # M15 data (175k bar)
│   ├── btcusdt_h1.csv        # H1 data (44k bar)
│   └── dataset_v2.parquet    # Tayyor dataset
├── build_dataset.py           # Dataset pipeline
├── target.py                  # Triple Barrier
└── README.md

---

## Texnologiyalar

| Kutubxona | Versiya | Maqsad |
|-----------|---------|--------|
| PyTorch | 2.x | Neural network |
| LightGBM | 4.x | Gradient boosting |
| pandas | 2.x | Data manipulation |
| numpy | 1.x | Numerical computing |
| scikit-learn | 1.x | Preprocessing |
| python-binance | 1.x | Exchange API |
| matplotlib | 3.x | Visualization |

---

## Muallif

**Capi (Alisher Muzafarov)**  
18 yosh | Tashkent, O'zbekiston  
GitHub: [@capi1000k](https://github.com/capi1000k)

> "Narxni bashorat qilib bo'lmaydi, lekin trend kontekstida  
>  aniq entry/exit topish mumkin." — Capigradmie falsafasi

---

*Built with curiosity, discipline, and Claude AI* 🚀
EOF

echo "README tayyor!"