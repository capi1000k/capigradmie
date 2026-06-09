# utils/time_utils.py
"""
TIMESTAMP UTILITY LAYER  —  Market Behavior Observatory
=========================================================
Single source of truth for every datetime / timestamp
operation across the codebase.

Design rules
────────────
• All internal datetimes are UTC-aware (timezone-aware).
• Public functions are pure / stateless — no side effects.
• No silent coercions: if a value can't be converted safely,
  raise a descriptive TypeError / ValueError.
• pandas, numpy, and plain Python datetime objects are all
  accepted as input where sensible (ISO 8601 strings too).

Sections
────────
  1. NOW helpers
  2. Type conversion  (→ datetime, → Timestamp, → int ms)
  3. Alignment / rounding
  4. Formatting / parsing
  5. Validation / guards
  6. Range / sequence generation
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Type alias for anything this module accepts as a "timestamp-like" value
# ---------------------------------------------------------------------------
TimestampLike = Union[
    datetime,          # stdlib — tz-aware or naive
    pd.Timestamp,      # pandas
    np.datetime64,     # numpy
    int,               # Unix ms  (13-digit epoch)
    float,             # Unix seconds (10-digit epoch)
    str,               # ISO 8601 string
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_UTC = timezone.utc
_PANDAS_UTC = "UTC"

# Binance epoch reference (ms since Unix epoch)
_MS_PER_S  = 1_000
_MS_PER_M  = 60_000
_MS_PER_H  = 3_600_000
_MS_PER_D  = 86_400_000


# ===========================================================================
# 1.  NOW HELPERS
# ===========================================================================

def utc_now() -> pd.Timestamp:
    """
    Return the current UTC time as a **tz-aware** ``pd.Timestamp``.

    Preferred over ``datetime.utcnow()`` (which returns a *naive* datetime)
    and ``datetime.now(timezone.utc)`` (stdlib only) for consistency across
    the codebase — pandas operations accept ``pd.Timestamp`` natively.

    Returns
    -------
    pd.Timestamp
        Current UTC moment with ``tzinfo=UTC``.

    Examples
    --------
    >>> t = utc_now()
    >>> t.tzinfo
    <UTC>
    >>> isinstance(t, pd.Timestamp)
    True
    """
    return pd.Timestamp.now(tz=_PANDAS_UTC)


def utc_now_ms() -> int:
    """
    Return current UTC time as **Unix milliseconds** (int).

    Suitable for Binance REST / WebSocket timestamps.

    Returns
    -------
    int
        Milliseconds since Unix epoch (UTC).

    Examples
    --------
    >>> ms = utc_now_ms()
    >>> len(str(ms))
    13
    """
    return int(time.time() * _MS_PER_S)


def utc_now_dt() -> datetime:
    """
    Return current UTC time as a **tz-aware** stdlib ``datetime``.

    Use when a plain ``datetime`` is required (e.g. logging, strftime).

    Returns
    -------
    datetime
        ``datetime.now(timezone.utc)``
    """
    return datetime.now(_UTC)


def monotonic_ms() -> int:
    """
    Return a monotonic clock reading in **milliseconds**.

    Not wall-clock time; safe for measuring durations / latency.

    Returns
    -------
    int
        ``time.monotonic_ns() // 1_000_000``
    """
    return time.monotonic_ns() // 1_000_000


# ===========================================================================
# 2.  TYPE CONVERSION
# ===========================================================================

def to_timestamp(value: TimestampLike) -> pd.Timestamp:
    """
    Coerce any supported timestamp-like value to a **UTC-aware**
    ``pd.Timestamp``.

    Accepted types
    ──────────────
    * ``pd.Timestamp``   — returned as-is if already UTC, else converted
    * ``datetime``       — naive → assumed UTC; aware → converted to UTC
    * ``np.datetime64``  — unit inferred automatically
    * ``int``            — treated as **Unix milliseconds** (13-digit epoch)
    * ``float``          — treated as **Unix seconds** (10-digit epoch)
    * ``str``            — ISO 8601 string, parsed via pandas

    Parameters
    ----------
    value : TimestampLike
        Input to convert.

    Returns
    -------
    pd.Timestamp
        UTC-aware timestamp.

    Raises
    ------
    TypeError
        If ``value`` is not a recognised type.
    ValueError
        If a string cannot be parsed.

    Examples
    --------
    >>> to_timestamp(1_700_000_000_000)          # Unix ms
    Timestamp('2023-11-14 22:13:20+0000', tz='UTC')
    >>> to_timestamp("2024-01-01T12:00:00Z")
    Timestamp('2024-01-01 12:00:00+0000', tz='UTC')
    """
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            return value.tz_localize(_PANDAS_UTC)
        return value.tz_convert(_PANDAS_UTC)

    if isinstance(value, datetime):
        if value.tzinfo is None:
            # naive → assume UTC (matches Binance convention)
            return pd.Timestamp(value, tz=_PANDAS_UTC)
        return pd.Timestamp(value).tz_convert(_PANDAS_UTC)

    if isinstance(value, np.datetime64):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize(_PANDAS_UTC)
        return ts.tz_convert(_PANDAS_UTC)

    if isinstance(value, int):
        # Binance uses ms; accept both 10-digit (s) and 13-digit (ms)
        if value > 1e12:
            return pd.Timestamp(value, unit="ms", tz=_PANDAS_UTC)
        return pd.Timestamp(value, unit="s", tz=_PANDAS_UTC)

    if isinstance(value, float):
        return pd.Timestamp(value, unit="s", tz=_PANDAS_UTC)

    if isinstance(value, str):
        try:
            ts = pd.Timestamp(value)
        except Exception as exc:
            raise ValueError(f"Cannot parse timestamp string: {value!r}") from exc
        if ts.tzinfo is None:
            return ts.tz_localize(_PANDAS_UTC)
        return ts.tz_convert(_PANDAS_UTC)

    raise TypeError(
        f"Unsupported timestamp type: {type(value).__name__!r}. "
        f"Expected one of: pd.Timestamp, datetime, np.datetime64, int, float, str."
    )


def to_datetime(value: TimestampLike) -> datetime:
    """
    Coerce any supported timestamp-like value to a **UTC-aware** stdlib
    ``datetime``.

    Delegates to :func:`to_timestamp` and calls ``.to_pydatetime()``.

    Parameters
    ----------
    value : TimestampLike

    Returns
    -------
    datetime
        UTC-aware ``datetime`` object.
    """
    return to_timestamp(value).to_pydatetime()


def to_unix_ms(value: TimestampLike) -> int:
    """
    Convert any timestamp-like value to **Unix milliseconds** (int).

    Suitable for Binance REST query parameters.

    Parameters
    ----------
    value : TimestampLike

    Returns
    -------
    int
        Unix epoch in milliseconds.

    Examples
    --------
    >>> to_unix_ms(pd.Timestamp("2024-01-01", tz="UTC"))
    1704067200000
    """
    ts = to_timestamp(value)
    return int(ts.value // 1_000_000)   # nanoseconds → milliseconds


def to_unix_s(value: TimestampLike) -> float:
    """
    Convert any timestamp-like value to **Unix seconds** (float).

    Parameters
    ----------
    value : TimestampLike

    Returns
    -------
    float
        Unix epoch in seconds.
    """
    return to_unix_ms(value) / _MS_PER_S


def strip_tz(series: pd.Series) -> pd.Series:
    """
    Convert a tz-aware datetime Series to **timezone-naive UTC**.

    Useful before passing to matplotlib / plotly (some backends reject
    tz-aware timestamps).

    Parameters
    ----------
    series : pd.Series
        Datetime Series (tz-aware or naive).

    Returns
    -------
    pd.Series
        Timezone-naive datetime Series in UTC.

    Examples
    --------
    >>> s = pd.Series([pd.Timestamp("2024-01-01", tz="UTC")])
    >>> strip_tz(s).dt.tz
    # None
    """
    s = pd.to_datetime(series, errors="coerce", utc=True)
    return s.dt.tz_localize(None)


# ===========================================================================
# 3.  ALIGNMENT / ROUNDING
# ===========================================================================

def floor_to_minute(value: TimestampLike) -> pd.Timestamp:
    """
    Floor a timestamp to the nearest **closed minute** (seconds = 0).

    Matches the ``open_time`` semantics of Binance M1 klines.

    Parameters
    ----------
    value : TimestampLike

    Returns
    -------
    pd.Timestamp
        UTC-aware, seconds and sub-seconds zeroed.

    Examples
    --------
    >>> floor_to_minute("2024-03-15T12:34:56.789Z")
    Timestamp('2024-03-15 12:34:00+0000', tz='UTC')
    """
    ts = to_timestamp(value)
    return ts.floor("min")


def floor_to_interval(value: TimestampLike, interval_minutes: int) -> pd.Timestamp:
    """
    Floor a timestamp to the nearest N-minute interval boundary.

    Parameters
    ----------
    value : TimestampLike
    interval_minutes : int
        Interval width in minutes (e.g. 1, 5, 15, 60).

    Returns
    -------
    pd.Timestamp
        UTC-aware, floored to the interval boundary.

    Raises
    ------
    ValueError
        If ``interval_minutes < 1``.

    Examples
    --------
    >>> floor_to_interval("2024-01-01T12:37:00Z", 5)
    Timestamp('2024-01-01 12:35:00+0000', tz='UTC')
    """
    if interval_minutes < 1:
        raise ValueError(f"interval_minutes must be >= 1, got {interval_minutes}")
    ts = to_timestamp(value)
    freq = f"{interval_minutes}min"
    return ts.floor(freq)


def current_candle_open(interval_minutes: int = 1) -> pd.Timestamp:
    """
    Return the **open time** of the currently running Binance kline.

    Parameters
    ----------
    interval_minutes : int
        Kline interval in minutes (default: 1).

    Returns
    -------
    pd.Timestamp
        UTC-aware open time of the current candle.
    """
    return floor_to_interval(utc_now(), interval_minutes)


def candle_close_time(open_time: TimestampLike, interval_minutes: int = 1) -> pd.Timestamp:
    """
    Compute the **close time** of a kline given its open time.

    Binance close_time = open_time + interval - 1 ms.

    Parameters
    ----------
    open_time : TimestampLike
    interval_minutes : int

    Returns
    -------
    pd.Timestamp
        UTC-aware close time.
    """
    ts = to_timestamp(open_time)
    return ts + pd.Timedelta(minutes=interval_minutes) - pd.Timedelta(milliseconds=1)


# ===========================================================================
# 4.  FORMATTING / PARSING
# ===========================================================================

def fmt_utc(value: TimestampLike, fmt: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    """
    Format a timestamp as a human-readable UTC string.

    Parameters
    ----------
    value : TimestampLike
    fmt   : str
        strftime format string (default: ``"%Y-%m-%d %H:%M:%S UTC"``).

    Returns
    -------
    str

    Examples
    --------
    >>> fmt_utc(pd.Timestamp("2024-06-01 09:30:00", tz="UTC"))
    '2024-06-01 09:30:00 UTC'
    """
    return to_timestamp(value).strftime(fmt)


def fmt_hms(value: TimestampLike) -> str:
    """
    Format a timestamp as ``HH:MM:SS`` (UTC, no date).

    Useful for dashboard status bars.

    Parameters
    ----------
    value : TimestampLike

    Returns
    -------
    str
        e.g. ``"14:07:33"``
    """
    return to_timestamp(value).strftime("%H:%M:%S")


def fmt_iso(value: TimestampLike) -> str:
    """
    Format a timestamp as an ISO 8601 string with UTC offset.

    Parameters
    ----------
    value : TimestampLike

    Returns
    -------
    str
        e.g. ``"2024-06-01T09:30:00+00:00"``
    """
    return to_timestamp(value).isoformat()


def parse_binance_open_time(ms: int) -> pd.Timestamp:
    """
    Parse a Binance ``open_time`` field (Unix milliseconds) to a
    UTC-aware ``pd.Timestamp``.

    Convenience alias for ``to_timestamp(ms)`` with explicit intent.

    Parameters
    ----------
    ms : int
        Unix milliseconds from Binance API response.

    Returns
    -------
    pd.Timestamp
    """
    return pd.Timestamp(ms, unit="ms", tz=_PANDAS_UTC)


# ===========================================================================
# 5.  VALIDATION / GUARDS
# ===========================================================================

def is_tz_aware(value: Union[datetime, pd.Timestamp]) -> bool:
    """
    Return ``True`` if *value* is timezone-aware.

    Parameters
    ----------
    value : datetime | pd.Timestamp

    Returns
    -------
    bool
    """
    if isinstance(value, pd.Timestamp):
        return value.tzinfo is not None
    if isinstance(value, datetime):
        return value.tzinfo is not None and value.tzinfo.utcoffset(value) is not None
    return False


def assert_utc(value: Union[datetime, pd.Timestamp], label: str = "timestamp") -> None:
    """
    Raise ``ValueError`` if *value* is not UTC-aware.

    Use as a guard at function boundaries.

    Parameters
    ----------
    value : datetime | pd.Timestamp
    label : str
        Human-readable name for error messages.

    Raises
    ------
    ValueError
        If *value* is timezone-naive or not UTC.
    """
    if not is_tz_aware(value):
        raise ValueError(
            f"{label} must be UTC-aware, got naive datetime: {value!r}"
        )
    ts = to_timestamp(value)
    if ts.utcoffset().total_seconds() != 0:
        raise ValueError(
            f"{label} must be UTC (offset=0), got offset={ts.utcoffset()}: {value!r}"
        )


def is_stale(value: TimestampLike, max_age_seconds: float = 10.0) -> bool:
    """
    Return ``True`` if *value* is older than *max_age_seconds* ago (UTC).

    Useful for detecting stale orderbook / kline data.

    Parameters
    ----------
    value : TimestampLike
    max_age_seconds : float
        Freshness threshold in seconds.

    Returns
    -------
    bool

    Examples
    --------
    >>> old = pd.Timestamp("2000-01-01", tz="UTC")
    >>> is_stale(old, max_age_seconds=60)
    True
    """
    ts = to_timestamp(value)
    age = (utc_now() - ts).total_seconds()
    return age > max_age_seconds


def is_future(value: TimestampLike) -> bool:
    """
    Return ``True`` if *value* is strictly in the future (UTC).

    Parameters
    ----------
    value : TimestampLike

    Returns
    -------
    bool
    """
    return to_timestamp(value) > utc_now()


def validate_kline_open_time(
    open_time: TimestampLike,
    interval_minutes: int = 1,
    tolerance_ms: int = 500,
) -> bool:
    """
    Validate that *open_time* is aligned to an N-minute boundary
    within a millisecond tolerance.

    Parameters
    ----------
    open_time : TimestampLike
    interval_minutes : int
    tolerance_ms : int
        Maximum deviation in milliseconds from the expected boundary.

    Returns
    -------
    bool
        ``True`` if aligned within tolerance.
    """
    ts = to_timestamp(open_time)
    expected = floor_to_interval(ts, interval_minutes)
    diff_ms  = abs((ts - expected).total_seconds() * _MS_PER_S)
    return diff_ms <= tolerance_ms


# ===========================================================================
# 6.  RANGE / SEQUENCE GENERATION
# ===========================================================================

def kline_time_range(
    start: TimestampLike,
    end: TimestampLike,
    interval_minutes: int = 1,
) -> pd.DatetimeIndex:
    """
    Generate a sequence of kline **open times** between *start* and *end*.

    Both bounds are floored to the interval boundary.

    Parameters
    ----------
    start : TimestampLike
    end   : TimestampLike
    interval_minutes : int

    Returns
    -------
    pd.DatetimeIndex
        UTC-aware DatetimeIndex at every N-minute boundary.

    Examples
    --------
    >>> idx = kline_time_range("2024-01-01T00:00Z", "2024-01-01T00:10Z", 5)
    >>> list(idx)
    [Timestamp('2024-01-01 00:00:00+0000', tz='UTC'),
     Timestamp('2024-01-01 00:05:00+0000', tz='UTC'),
     Timestamp('2024-01-01 00:10:00+0000', tz='UTC')]
    """
    t_start = floor_to_interval(start, interval_minutes)
    t_end   = floor_to_interval(end,   interval_minutes)
    freq    = f"{interval_minutes}min"
    return pd.date_range(start=t_start, end=t_end, freq=freq, tz=_PANDAS_UTC)


def ms_range(
    start: TimestampLike,
    end: TimestampLike,
    chunk_ms: int = 60_000,
) -> list[tuple[int, int]]:
    """
    Divide a time range into (start_ms, end_ms) chunks for paginated
    Binance REST requests.

    Parameters
    ----------
    start    : TimestampLike
    end      : TimestampLike
    chunk_ms : int
        Width of each chunk in milliseconds (default: 60 000 = 1 minute).

    Returns
    -------
    list of (int, int)
        Each tuple is ``(open_time_ms, close_time_ms)`` for one request.

    Examples
    --------
    >>> chunks = ms_range("2024-01-01T00:00Z", "2024-01-01T00:03Z", 60_000)
    >>> len(chunks)
    3
    """
    s_ms = to_unix_ms(start)
    e_ms = to_unix_ms(end)
    chunks = []
    cur = s_ms
    while cur < e_ms:
        chunk_end = min(cur + chunk_ms - 1, e_ms)
        chunks.append((cur, chunk_end))
        cur = chunk_end + 1
    return chunks


def recent_window(seconds: int = 60) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return ``(start, now)`` covering the last *seconds* seconds.

    Useful for fetching a sliding window of recent data.

    Parameters
    ----------
    seconds : int
        Window size.

    Returns
    -------
    tuple[pd.Timestamp, pd.Timestamp]
        ``(utc_now() - timedelta(seconds=seconds), utc_now())``
    """
    now = utc_now()
    return now - pd.Timedelta(seconds=seconds), now


# ===========================================================================
# Quick sanity test (run with: python -m utils.time_utils)
# ===========================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  time_utils — sanity check")
    print("=" * 60)

    failures = []

    def check(label, result, expected=None, condition=None):
        ok = (result == expected) if expected is not None else bool(condition(result))
        status = "✓" if ok else "✗"
        print(f"  {status}  {label}: {result}")
        if not ok:
            failures.append(label)

    # 1. utc_now
    now_ts = utc_now()
    check("utc_now() is pd.Timestamp",    isinstance(now_ts, pd.Timestamp), True)
    check("utc_now() is UTC-aware",       is_tz_aware(now_ts), True)

    # 2. utc_now_ms
    ms = utc_now_ms()
    check("utc_now_ms() length",          len(str(ms)), 13)

    # 3. to_timestamp — int ms
    ref_ms = 1_700_000_000_000
    ts_from_ms = to_timestamp(ref_ms)
    check("to_timestamp(int ms) UTC",     str(ts_from_ms.tz), "UTC")

    # 4. to_timestamp — ISO string
    ts_iso = to_timestamp("2024-01-01T00:00:00Z")
    check("to_timestamp(str) year",       ts_iso.year, 2024)

    # 5. to_unix_ms round-trip
    ms_rt = to_unix_ms(ts_iso)
    ts_rt = to_timestamp(ms_rt)
    check("to_unix_ms round-trip",        ts_rt.year, 2024)

    # 6. floor_to_minute
    floored = floor_to_minute("2024-03-15T12:34:56.789Z")
    check("floor_to_minute second=0",     floored.second, 0)

    # 7. floor_to_interval
    fi = floor_to_interval("2024-01-01T12:37:00Z", 5)
    check("floor_to_interval 5m minute",  fi.minute, 35)

    # 8. kline_time_range
    idx = kline_time_range("2024-01-01T00:00Z", "2024-01-01T00:10Z", 5)
    check("kline_time_range len",         len(idx), 3)

    # 9. ms_range
    chunks = ms_range("2024-01-01T00:00Z", "2024-01-01T00:02:59.999Z", 60_000)
    check("ms_range 3 chunks",            len(chunks), 3)

    # 10. is_stale
    old = pd.Timestamp("2000-01-01", tz="UTC")
    check("is_stale(2000-01-01)",         is_stale(old, 60), True)

    # 11. strip_tz
    s = pd.Series([pd.Timestamp("2024-01-01", tz="UTC")])
    stripped = strip_tz(s)
    check("strip_tz has no tz",           stripped.dt.tz is None, True)

    # 12. fmt_hms
    hms = fmt_hms("2024-06-24T07:00:00Z")
    check("fmt_hms",                      hms, "07:00:00")

    # 13. candle_close_time
    ct = candle_close_time("2024-01-01T00:00:00Z", 1)
    check("candle_close_time ms",         ct.microsecond, 999000)

    # 14. recent_window
    w_start, w_end = recent_window(60)
    check("recent_window diff ~60s",
          condition=lambda _: 59 < (w_end - w_start).total_seconds() < 61,
          result=(w_end - w_start).total_seconds())

    print()
    if failures:
        print(f"  FAILED: {failures}")
        sys.exit(1)
    else:
        print("  All checks passed ✓")