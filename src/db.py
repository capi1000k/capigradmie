# src/db.py
"""
STORAGE LAYER v2 — DuckDB Backend
───────────────────────────────────
DuckDB — embedded OLAP database.
- Columnar storage (like Parquet, but queryable)
- No read-concat-rewrite pattern
- True append via INSERT
- Dedup via UNIQUE constraint + INSERT OR IGNORE
- WAL mode enabled (crash-safe)

Tables:
  klines_m1   — M1 OHLCV candles
  klines_m5   — M5 OHLCV candles
  trades      — aggTrade stream
  orderbook   — OB snapshots with microstructure features
"""

import threading
import duckdb
import pandas as pd
import numpy as np

from src.config import DB_PATH
from src.logger import get_logger

log = get_logger("db")

# ─── CONNECTION POOL (thread-local) ──────────────────────────────────────────
# DuckDB connections are NOT thread-safe, so each thread gets its own.
_local = threading.local()
_init_lock = threading.Lock()


def _get_conn() -> duckdb.DuckDBPyConnection:
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = duckdb.connect(str(DB_PATH))
        conn.execute("PRAGMA threads=4")
        conn.execute("PRAGMA memory_limit='512MB'")
        _local.conn = conn
        _ensure_schema(conn)
    return _local.conn


def _ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS klines_m1 (
            open_time           TIMESTAMPTZ PRIMARY KEY,
            open                DOUBLE,
            high                DOUBLE,
            low                 DOUBLE,
            close               DOUBLE,
            volume              DOUBLE,
            num_trades          INTEGER,
            taker_buy_base_vol  DOUBLE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS klines_m5 (
            open_time           TIMESTAMPTZ PRIMARY KEY,
            open                DOUBLE,
            high                DOUBLE,
            low                 DOUBLE,
            close               DOUBLE,
            volume              DOUBLE,
            num_trades          INTEGER,
            taker_buy_base_vol  DOUBLE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            timestamp     TIMESTAMPTZ PRIMARY KEY,
            price         DOUBLE,
            quantity      DOUBLE,
            is_buyer_mm   BOOLEAN,
            trade_id      BIGINT,
            exchange_ts   TIMESTAMPTZ,
            receive_ts    TIMESTAMPTZ,
            process_ts    TIMESTAMPTZ,
            latency_ms    DOUBLE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orderbook (
            timestamp     TIMESTAMPTZ PRIMARY KEY,
            best_bid      DOUBLE,
            best_ask      DOUBLE,
            spread        DOUBLE,
            mid_price     DOUBLE,
            microprice    DOUBLE,
            bid_vol       DOUBLE,
            ask_vol       DOUBLE,
            imbalance     DOUBLE,
            ofi           DOUBLE,
            spread_bps    DOUBLE,
            exchange_ts   TIMESTAMPTZ,
            receive_ts    TIMESTAMPTZ,
            process_ts    TIMESTAMPTZ,
            latency_ms    DOUBLE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cvd (
            timestamp   TIMESTAMPTZ PRIMARY KEY,
            cvd         DOUBLE,
            delta       DOUBLE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS liquidations (
            timestamp     TIMESTAMPTZ PRIMARY KEY,
            side          VARCHAR,      -- BUY (short liquidated) / SELL (long liquidated)
            price         DOUBLE,
            quantity      DOUBLE,
            quote_qty     DOUBLE,       -- price * quantity, USDT notional
            exchange_ts   TIMESTAMPTZ,
            receive_ts    TIMESTAMPTZ,
            latency_ms    DOUBLE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS open_interest (
            timestamp     TIMESTAMPTZ PRIMARY KEY,
            open_interest DOUBLE,       -- contracts (BTC)
            oi_value      DOUBLE        -- open_interest * mark_price (USDT notional), filled by caller
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mark_price (
            timestamp       TIMESTAMPTZ PRIMARY KEY,
            mark_price      DOUBLE,
            index_price     DOUBLE,
            funding_rate    DOUBLE,
            next_funding_ts TIMESTAMPTZ,
            exchange_ts     TIMESTAMPTZ,
            receive_ts      TIMESTAMPTZ,
            latency_ms      DOUBLE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candle_features (
            interval          VARCHAR,
            open_time         TIMESTAMPTZ,
            close_time         TIMESTAMPTZ,
            -- trades
            trade_count        INTEGER,
            buy_count          INTEGER,
            sell_count         INTEGER,
            buy_vol            DOUBLE,
            sell_vol           DOUBLE,
            -- CVD (shu sham davomida)
            cvd_open           DOUBLE,
            cvd_close          DOUBLE,
            cvd_delta          DOUBLE,
            -- orderbook (shu sham davomidagi o'rtacha/agregat)
            ob_update_count    INTEGER,
            avg_imbalance      DOUBLE,
            avg_spread_bps     DOUBLE,
            min_spread_bps     DOUBLE,
            max_spread_bps     DOUBLE,
            ofi_sum            DOUBLE,
            avg_bid_vol        DOUBLE,
            avg_ask_vol        DOUBLE,
            -- likvidatsiyalar
            liq_buy_count      INTEGER,   -- SHORT likvidatsiya (majburiy sotib olish)
            liq_buy_qty        DOUBLE,
            liq_sell_count     INTEGER,   -- LONG likvidatsiya (majburiy sotuv)
            liq_sell_qty       DOUBLE,
            -- open interest / funding
            oi_open            DOUBLE,
            oi_close           DOUBLE,
            oi_delta           DOUBLE,
            funding_rate_avg   DOUBLE,
            mark_price_close   DOUBLE,
            PRIMARY KEY (interval, open_time)
        )
    """)
    log.debug("Schema ensured")


# ─── WRITE FUNCTIONS ─────────────────────────────────────────────────────────

def insert_klines(df: pd.DataFrame, interval: str) -> int:
    """INSERT OR REPLACE klines. Returns rows inserted."""
    if df.empty:
        return 0
    table = f"klines_{interval}"
    conn  = _get_conn()
    try:
        # DuckDB can query pandas directly
        conn.execute(f"""
            INSERT OR REPLACE INTO {table}
            SELECT * FROM df
        """)
        log.debug(f"[db] klines/{interval} upserted {len(df)} rows")
        return len(df)
    except Exception as e:
        log.error(f"[db] klines insert error: {e}")
        return 0


def insert_trades(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn = _get_conn()
    try:
        conn.execute("""
            INSERT OR IGNORE INTO trades
            SELECT * FROM df
        """)
        return len(df)
    except Exception as e:
        log.error(f"[db] trades insert error: {e}")
        return 0


def insert_orderbook(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn = _get_conn()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO orderbook
            SELECT * FROM df
        """)
        return len(df)
    except Exception as e:
        log.error(f"[db] orderbook insert error: {e}")
        return 0


def insert_cvd(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn = _get_conn()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO cvd
            SELECT * FROM df
        """)
        return len(df)
    except Exception as e:
        log.error(f"[db] cvd insert error: {e}")
        return 0


def insert_liquidations(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn = _get_conn()
    try:
        conn.execute("""
            INSERT OR IGNORE INTO liquidations
            SELECT * FROM df
        """)
        return len(df)
    except Exception as e:
        log.error(f"[db] liquidations insert error: {e}")
        return 0


def insert_open_interest(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn = _get_conn()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO open_interest
            SELECT * FROM df
        """)
        return len(df)
    except Exception as e:
        log.error(f"[db] open_interest insert error: {e}")
        return 0


def insert_mark_price(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn = _get_conn()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO mark_price
            SELECT * FROM df
        """)
        return len(df)
    except Exception as e:
        log.error(f"[db] mark_price insert error: {e}")
        return 0


def load_last_cvd() -> float:
    """
    Restart-persistence: oxirgi saqlangan CVD qiymatini qaytaradi.
    Dastur qayta ishga tushganda CVDTracker shu yerdan davom etadi,
    aks holda restart CVD'ni 0'ga tushirib signalni buzadi.
    Ma'lumot yo'q bo'lsa 0.0 qaytaradi.
    """
    conn = _get_conn()
    try:
        row = conn.execute("""
            SELECT cvd FROM cvd ORDER BY timestamp DESC LIMIT 1
        """).fetchone()
        return float(row[0]) if row else 0.0
    except Exception as e:
        log.error(f"[db] load_last_cvd error: {e}")
        return 0.0


# ─── CANDLE-LEVEL FEATURE SNAPSHOT ────────────────────────────────────────────
# "Devorga g'isht qo'yish" — har sham yopilganda, o'sha sham davomidagi
# BARCHA xom oqimlarni (trades, orderbook, liquidations, CVD, OI, funding)
# bitta qatorga agregatlab yozadi. Naqsh yo'q — faqat arxiv.

def build_candle_features(interval: str, open_time, close_time) -> dict | None:
    """
    [open_time, close_time) oralig'idagi barcha xom datani agregatlab,
    candle_features jadvaliga yozadi. Har chaqiriqda faqat bitta (yangi
    yopilgan) sham uchun ishlaydi — ingestion_klines.py chaqiradi.

    Returns yozilgan dict, yoki xato bo'lsa None.
    """
    conn = _get_conn()
    try:
        trades = conn.execute("""
            SELECT
                COUNT(*)                                          AS trade_count,
                SUM(CASE WHEN is_buyer_mm THEN 0 ELSE 1 END)       AS buy_count,
                SUM(CASE WHEN is_buyer_mm THEN 1 ELSE 0 END)       AS sell_count,
                SUM(CASE WHEN is_buyer_mm THEN 0 ELSE quantity END) AS buy_vol,
                SUM(CASE WHEN is_buyer_mm THEN quantity ELSE 0 END) AS sell_vol
            FROM trades
            WHERE timestamp >= ? AND timestamp < ?
        """, [open_time, close_time]).fetchone()

        cvd_row = conn.execute("""
            SELECT
                FIRST(cvd ORDER BY timestamp) AS cvd_open,
                LAST(cvd ORDER BY timestamp)  AS cvd_close
            FROM cvd
            WHERE timestamp >= ? AND timestamp < ?
        """, [open_time, close_time]).fetchone()

        ob = conn.execute("""
            SELECT
                COUNT(*)              AS ob_update_count,
                AVG(imbalance)        AS avg_imbalance,
                AVG(spread_bps)       AS avg_spread_bps,
                MIN(spread_bps)       AS min_spread_bps,
                MAX(spread_bps)       AS max_spread_bps,
                SUM(ofi)              AS ofi_sum,
                AVG(bid_vol)          AS avg_bid_vol,
                AVG(ask_vol)          AS avg_ask_vol
            FROM orderbook
            WHERE timestamp >= ? AND timestamp < ?
        """, [open_time, close_time]).fetchone()

        liq = conn.execute("""
            SELECT
                SUM(CASE WHEN side = 'BUY'  THEN 1 ELSE 0 END)        AS liq_buy_count,
                SUM(CASE WHEN side = 'BUY'  THEN quantity ELSE 0 END) AS liq_buy_qty,
                SUM(CASE WHEN side = 'SELL' THEN 1 ELSE 0 END)        AS liq_sell_count,
                SUM(CASE WHEN side = 'SELL' THEN quantity ELSE 0 END) AS liq_sell_qty
            FROM liquidations
            WHERE timestamp >= ? AND timestamp < ?
        """, [open_time, close_time]).fetchone()

        oi_row = conn.execute("""
            SELECT
                FIRST(open_interest ORDER BY timestamp) AS oi_open,
                LAST(open_interest ORDER BY timestamp)  AS oi_close
            FROM open_interest
            WHERE timestamp >= ? AND timestamp < ?
        """, [open_time, close_time]).fetchone()

        mp = conn.execute("""
            SELECT
                AVG(funding_rate)                    AS funding_rate_avg,
                LAST(mark_price ORDER BY timestamp)   AS mark_price_close
            FROM mark_price
            WHERE timestamp >= ? AND timestamp < ?
        """, [open_time, close_time]).fetchone()

        cvd_open, cvd_close = (cvd_row or (None, None))
        oi_open, oi_close   = (oi_row or (None, None))

        row = {
            "interval":         interval,
            "open_time":        open_time,
            "close_time":       close_time,
            "trade_count":      trades[0] or 0,
            "buy_count":        trades[1] or 0,
            "sell_count":       trades[2] or 0,
            "buy_vol":          trades[3] or 0.0,
            "sell_vol":         trades[4] or 0.0,
            "cvd_open":         cvd_open,
            "cvd_close":        cvd_close,
            "cvd_delta":        (cvd_close - cvd_open) if (cvd_open is not None and cvd_close is not None) else None,
            "ob_update_count":  ob[0] or 0,
            "avg_imbalance":    ob[1],
            "avg_spread_bps":   ob[2],
            "min_spread_bps":   ob[3],
            "max_spread_bps":   ob[4],
            "ofi_sum":          ob[5],
            "avg_bid_vol":      ob[6],
            "avg_ask_vol":      ob[7],
            "liq_buy_count":    liq[0] or 0,
            "liq_buy_qty":      liq[1] or 0.0,
            "liq_sell_count":   liq[2] or 0,
            "liq_sell_qty":     liq[3] or 0.0,
            "oi_open":          oi_open,
            "oi_close":         oi_close,
            "oi_delta":         (oi_close - oi_open) if (oi_open is not None and oi_close is not None) else None,
            "funding_rate_avg": mp[0],
            "mark_price_close": mp[1],
        }

        df = pd.DataFrame([row])
        conn.execute("INSERT OR REPLACE INTO candle_features SELECT * FROM df")
        return row

    except Exception as e:
        log.error(f"[db] build_candle_features error ({interval}, {open_time}): {e}")
        return None


def load_candle_features(interval: str, limit: int = 200) -> pd.DataFrame:
    conn = _get_conn()
    try:
        return conn.execute("""
            SELECT * FROM candle_features
            WHERE interval = ?
            ORDER BY open_time DESC
            LIMIT ?
        """, [interval, limit]).df().iloc[::-1].reset_index(drop=True)
    except Exception as e:
        log.error(f"[db] load_candle_features error: {e}")
        return pd.DataFrame()


# ─── READ FUNCTIONS ───────────────────────────────────────────────────────────

def load_klines(interval: str, limit: int = 100) -> pd.DataFrame:
    conn = _get_conn()
    try:
        return conn.execute(f"""
            SELECT * FROM klines_{interval}
            ORDER BY open_time DESC
            LIMIT {limit}
        """).df().iloc[::-1].reset_index(drop=True)
    except Exception as e:
        log.error(f"[db] load_klines error: {e}")
        return pd.DataFrame()


def load_trades(limit: int = 500) -> pd.DataFrame:
    conn = _get_conn()
    try:
        return conn.execute(f"""
            SELECT * FROM trades
            ORDER BY timestamp DESC
            LIMIT {limit}
        """).df().iloc[::-1].reset_index(drop=True)
    except Exception as e:
        log.error(f"[db] load_trades error: {e}")
        return pd.DataFrame()


def load_orderbook(limit: int = 300) -> pd.DataFrame:
    conn = _get_conn()
    try:
        return conn.execute(f"""
            SELECT * FROM orderbook
            ORDER BY timestamp DESC
            LIMIT {limit}
        """).df().iloc[::-1].reset_index(drop=True)
    except Exception as e:
        log.error(f"[db] load_orderbook error: {e}")
        return pd.DataFrame()


def load_cvd(limit: int = 300) -> pd.DataFrame:
    conn = _get_conn()
    try:
        return conn.execute(f"""
            SELECT * FROM cvd
            ORDER BY timestamp DESC
            LIMIT {limit}
        """).df().iloc[::-1].reset_index(drop=True)
    except Exception as e:
        log.error(f"[db] load_cvd error: {e}")
        return pd.DataFrame()


def db_stats() -> dict:
    """Return row counts for all tables."""
    conn = _get_conn()
    tables = ["klines_m1", "klines_m5", "trades", "orderbook", "cvd"]
    stats = {}
    for t in tables:
        try:
            stats[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception:
            stats[t] = 0
    return stats