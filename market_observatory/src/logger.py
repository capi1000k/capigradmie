# src/logger.py
"""
Color-coded structured logger.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

try:
    import colorlog
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

from src.config import LOG_DIR


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Console handler ──────────────────────────────────────────────────────
    if HAS_COLOR:
        fmt = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(name)s] %(levelname)s%(reset)s  %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG":    "cyan",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        fmt = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s  %(message)s",
            datefmt="%H:%M:%S",
        )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── File handler ─────────────────────────────────────────────────────────
    today = datetime.utcnow().strftime("%Y%m%d")
    fh = logging.FileHandler(LOG_DIR / f"{today}_{name}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s  %(message)s"
    ))
    logger.addHandler(fh)

    return logger
