# src/config.py
"""
Market Observatory — Central Configuration
"""

from pathlib import Path

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
LOG_DIR    = BASE_DIR / "logs"

KLINES_M1_DIR  = DATA_DIR / "klines_m1"
KLINES_M5_DIR  = DATA_DIR / "klines_m5"
TRADES_DIR     = DATA_DIR / "trades"
ORDERBOOK_DIR  = DATA_DIR / "orderbook"
DB_PATH        = DATA_DIR / "observatory.duckdb"   # ← NEW: DuckDB file

for d in [KLINES_M1_DIR, KLINES_M5_DIR, TRADES_DIR, ORDERBOOK_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── BINANCE ──────────────────────────────────────────────────────────────────
BINANCE_REST_BASE  = "https://api.binance.com"
BINANCE_WS_BASE    = "wss://stream.binance.com:9443/ws"

SYMBOL             = "BTCUSDT"
SYMBOL_LOWER       = SYMBOL.lower()

KLINE_ENDPOINT     = f"{BINANCE_REST_BASE}/api/v3/klines"
DEPTH_SNAPSHOT_URL = f"{BINANCE_REST_BASE}/api/v3/depth"

# ─── INTERVALS ────────────────────────────────────────────────────────────────
KLINE_POLL_M1      = 60          # seconds
KLINE_POLL_M5      = 300         # seconds
KLINE_LIMIT        = 100         # candles per fetch

# ─── STORAGE ──────────────────────────────────────────────────────────────────
ORDERBOOK_LEVELS   = 20          # top N bid/ask levels
WAL_FLUSH_SECONDS  = 1.0         # WAL: flush every N seconds
WAL_FLUSH_ROWS     = 100         # WAL: flush every N rows (whichever comes first)

# ─── DASHBOARD (Plotly Dash) ──────────────────────────────────────────────────
DASH_HOST          = "127.0.0.1"
DASH_PORT          = 8050
DASHBOARD_REFRESH  = 2000        # milliseconds (Dash interval)
CANDLES_SHOWN      = 60
TRADES_SHOWN       = 300

# ─── RECONNECT ────────────────────────────────────────────────────────────────
WS_RECONNECT_DELAY = 3
WS_MAX_RETRIES     = 999_999
