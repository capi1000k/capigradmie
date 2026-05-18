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
            trade_id      BIGINT
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
            spread_bps    DOUBLE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cvd (
            timestamp   TIMESTAMPTZ PRIMARY KEY,
            cvd         DOUBLE,
            delta       DOUBLE
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
