# main.py
"""
MARKET BEHAVIOR OBSERVATORY v2
────────────────────────────────
Usage:
    python main.py                  # full: data + Dash browser dashboard
    python main.py --data-only      # headless data collection
    python main.py --dashboard-only # dashboard from saved DB data
    python main.py --preview        # static PNG preview (no network)
"""

import asyncio
import argparse
import sys
import signal
import threading
from collections import deque

from src.config import KLINE_POLL_M1, KLINE_POLL_M5
from src.ingestion_klines import poll_klines
from src.ingestion_ws import stream_trades, stream_orderbook
from src.logger import get_logger

log = get_logger("main")

shared_state: dict = {
    "klines_m1":      None,
    "klines_m5":      None,
    "trades_live":    deque(maxlen=500),
    "orderbook_live": None,
    "cvd_live":       None,
}


async def run_data_layer() -> None:
    log.info("▶  Starting data ingestion layer")
    tasks = [
        asyncio.create_task(poll_klines("1m", KLINE_POLL_M1, shared_state), name="klines_m1"),
        asyncio.create_task(poll_klines("5m", KLINE_POLL_M5, shared_state), name="klines_m5"),
        asyncio.create_task(stream_trades(shared_state),                      name="trades_ws"),
        asyncio.create_task(stream_orderbook(shared_state),                   name="orderbook_ws"),
    ]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def start_data_in_thread() -> threading.Thread:
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_data_layer())
        except Exception as e:
            log.error(f"Data thread error: {e}")
        finally:
            loop.close()

    t = threading.Thread(target=_run, name="data-layer", daemon=True)
    t.start()
    return t


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-only",      action="store_true")
    p.add_argument("--dashboard-only", action="store_true")
    p.add_argument("--preview",        action="store_true",
                   help="Generate static dashboard PNG (no network)")
    p.add_argument("--debug",          action="store_true",
                   help="Dash debug mode")
    return p.parse_args()


def main():
    args = parse_args()

    log.info("═" * 60)
    log.info("  MARKET BEHAVIOR OBSERVATORY  v2.0")
    log.info(f"  Symbol: BTCUSDT  │  Backend: DuckDB  │  UI: Plotly Dash")
    log.info("═" * 60)

    if args.preview:
        log.info("Mode: PREVIEW (static PNG)")
        from demo_preview import make_preview
        make_preview()
        return

    if args.data_only:
        log.info("Mode: DATA ONLY")
        asyncio.run(run_data_layer())
        return

    if args.dashboard_only:
        log.info("Mode: DASHBOARD ONLY")
        from src.dash_dashboard import run_dashboard
        run_dashboard(shared_state, debug=args.debug)
        return

    # ── FULL ──────────────────────────────────────────────────────────────────
    log.info("Mode: FULL  (data + Dash dashboard)")
    start_data_in_thread()

    import time
    log.info("Waiting 3s for initial data…")
    time.sleep(3)

    from src.dash_dashboard import run_dashboard

    def _sigint(sig, frame):
        log.info("Shutting down")
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)

    log.info("Open your browser → http://127.0.0.1:8050")
    run_dashboard(shared_state, debug=args.debug)


if __name__ == "__main__":
    main()
