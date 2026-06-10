# src/events.py
"""
MARKET MICROSTRUCTURE SNAPSHOT COLLECTOR — Stage 2.2
═══════════════════════════════════════════════════════

Instrument turlari (3 ta):
  ┌─────────────────────┬──────────────────────────────────────────────────┐
  │ ABSORPTION          │ Katta flow bor, lekin narx qotgan.               │
  │                     │ Passiv devol (buyer yoki seller) ushlab turibdi. │
  ├─────────────────────┼──────────────────────────────────────────────────┤
  │ LIQUIDITY_GRAB      │ Thin book + narx keskin siljidi.                 │
  │                     │ Passiv devol yo'q — narx bo'sh joyga "uchdi".   │
  ├─────────────────────┼──────────────────────────────────────────────────┤
  │ MOMENTUM            │ Flow + narx bir tomonga, hech kim to'xtatmaydi. │
  │                     │ Trend kuchaymoqda yoki davom etmoqda.            │
  └─────────────────────┴──────────────────────────────────────────────────┘

Har instrument uchun 2 ta yo'nalish:
  ABSORPTION_BUY / ABSORPTION_SELL
  LIQUIDITY_GRAB_UP / LIQUIDITY_GRAB_DOWN
  MOMENTUM_UP / MOMENTUM_DOWN

Falsafa:
  - Threshold YO'Q. Faqat sign agreement tekshiriladi.
  - Label = "_CANDIDATE" — bu hipoteza, fact emas.
  - Barcha xom signallar saqlanadi.
  - Calibration keyinroq — 1000+ eventdan so'ng calibrate.py da.
  - MIXED label — signallar zid bo'lganda, bu ham qimmatli data.

Outcome:
  5min va 20min keyin narx nima bo'ldi → return % yoziladi.
  CONFIRMED / FAILED — calibration uchun, real-time qaror uchun emas.

reads:
    shared_state["regime"]["signals"]  → ofi_z, slope_z, imbalance, spread_bps_z, micro_vs_mid
    shared_state["orderbook_live"]     → mid_price
    shared_state["cvd_live"]           → float

writes:
    logs/events.log
"""

from __future__ import annotations

import logging
import pathlib
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ─── Logger ───────────────────────────────────────────────────────────────────

_LOG_DIR = pathlib.Path("logs")
_LOG_DIR.mkdir(exist_ok=True)

_elog = logging.getLogger("events")
if not _elog.handlers:
    _fh = logging.FileHandler(_LOG_DIR / "events.log", encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    _elog.addHandler(_fh)
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(logging.Formatter("%(asctime)s [events] %(message)s", datefmt="%H:%M:%S"))
    _elog.addHandler(_ch)
    _elog.propagate = False
_elog.setLevel(logging.DEBUG)


# ─── Instrument turlari ───────────────────────────────────────────────────────

class InstrumentType(str, Enum):
    """
    3 ta asosiy microstructure instrument.
    Har biri bozorning BOSHQACHA mexanikasini ifodalaydi.
    """
    ABSORPTION   = "ABSORPTION"     # flow absorbed → narx qotgan
    LIQUIDITY    = "LIQUIDITY"      # thin book → narx uchdi
    MOMENTUM     = "MOMENTUM"       # flow + narx birga → trend


class Direction(str, Enum):
    BUY  = "BUY"    # bullish (narx yuqoriga ketishi kutiladi)
    SELL = "SELL"   # bearish (narx pastga ketishi kutiladi)


class CandidateLabel(str, Enum):
    """
    Sign agreement asosida qo'yilgan hipoteza labeli.
    Threshold yo'q — faqat signallar qaysi tomonga ko'rsatayotgani.
    """
    ABSORPTION_BUY_CANDIDATE    = "ABSORPTION_BUY_CANDIDATE"
    ABSORPTION_SELL_CANDIDATE   = "ABSORPTION_SELL_CANDIDATE"
    LIQUIDITY_GRAB_UP_CANDIDATE = "LIQUIDITY_GRAB_UP_CANDIDATE"
    LIQUIDITY_GRAB_DOWN_CANDIDATE = "LIQUIDITY_GRAB_DOWN_CANDIDATE"
    MOMENTUM_UP_CANDIDATE       = "MOMENTUM_UP_CANDIDATE"
    MOMENTUM_DOWN_CANDIDATE     = "MOMENTUM_DOWN_CANDIDATE"
    MIXED                       = "MIXED"   # signallar zid — shovqin bo'lishi mumkin


# ─── Signal strengths (calibration uchun) ────────────────────────────────────

@dataclass
class SignalStrength:
    """
    Har bir signal qanchalik kuchli ekanini ko'rsatadi.
    Threshold yo'q — faqat magnitude.
    Calibration da: "ofi_z > 0.8 bo'lganda win rate necha?" — shu savol uchun.
    """
    ofi_z_abs:      float   # |ofi_z|    — flow kuchi
    slope_z_abs:    float   # |slope_z|  — CVD trend kuchi
    imbalance_abs:  float   # |imbalance| — orderbook taraf kuchi
    micro_abs:      float   # |micro_vs_mid| — narx siljishi
    spread_z_val:   float   # spread_bps_z — spread holati (signed)

    @property
    def composite(self) -> float:
        """
        Umumiy signal kuchi — calibration da tier ajratish uchun ishlatiladi.
        Weighted sum: ofi va slope eng muhim.
        """
        return (
            0.35 * self.ofi_z_abs +
            0.30 * self.slope_z_abs +
            0.20 * self.imbalance_abs +
            0.10 * self.micro_abs +
            0.05 * abs(self.spread_z_val)
        )


# ─── Snapshot (asosiy data unit) ──────────────────────────────────────────────

@dataclass
class MarketSnapshot:
    """
    Bir momentdagi bozor holati — to'liq xom signallar bilan.
    Bu yozilgan data unit. Threshold yo'q, faqat observation.
    """
    snapshot_id:        str
    candidate_label:    CandidateLabel
    instrument_type:    InstrumentType
    direction:          Direction

    timestamp:          datetime
    price:              float

    # regime context
    regime_label:       str
    regime_confidence:  float

    # xom signallar — o'zgartirilmagan
    ofi_z:              float
    slope_z:            float
    imbalance:          float
    micro_vs_mid:       float
    spread_bps_z:       float
    cvd:                float

    # signal kuchi (offline calibration uchun)
    strength:           SignalStrength

    # outcomes (OutcomeTracker tomonidan to'ldiriladi)
    outcome_5m:         Optional[float] = None
    outcome_20m:        Optional[float] = None


# ─── Pending outcome ──────────────────────────────────────────────────────────

@dataclass
class _PendingOutcome:
    snapshot_id:     str
    candidate_label: CandidateLabel
    instrument_type: InstrumentType
    direction:       Direction
    entry_price:     float
    trigger_ts:      float
    logged_5m:       bool = False
    logged_20m:      bool = False


# ─── Outcome evaluation ───────────────────────────────────────────────────────

_SUCCESS_THRESHOLD: float = 0.0005   # 0.05% — calibration da o'zgartiriladi

def _evaluate(direction: Direction, ret: float) -> bool:
    """
    Narx kutilgan tomonga ketdimi?
    BUY  → narx ko'tarilishi kerak edi
    SELL → narx tushishi kerak edi
    """
    if direction == Direction.BUY:
        return ret > _SUCCESS_THRESHOLD
    else:
        return ret < -_SUCCESS_THRESHOLD


# ─── Outcome Tracker ──────────────────────────────────────────────────────────

_OUTCOME_5MIN_S:  float = 300.0
_OUTCOME_20MIN_S: float = 1200.0


class _OutcomeTracker:

    def __init__(self):
        self._pending: list[_PendingOutcome] = []

    def register(self, snap: MarketSnapshot) -> None:
        self._pending.append(_PendingOutcome(
            snapshot_id=snap.snapshot_id,
            candidate_label=snap.candidate_label,
            instrument_type=snap.instrument_type,
            direction=snap.direction,
            entry_price=snap.price,
            trigger_ts=time.monotonic(),
        ))

    def tick(self, current_price: float) -> None:
        if current_price <= 0:
            return
        now = time.monotonic()
        still_pending = []

        for p in self._pending:
            elapsed = now - p.trigger_ts
            ret = (current_price - p.entry_price) / p.entry_price

            if not p.logged_5m and elapsed >= _OUTCOME_5MIN_S:
                ok = _evaluate(p.direction, ret)
                _elog.info(
                    "[OUTCOME_5M]  id=%s label=%-35s instr=%-12s dir=%s "
                    "entry=%.2f now=%.2f return=%+.4f%% → %s",
                    p.snapshot_id[:8],
                    p.candidate_label.value,
                    p.instrument_type.value,
                    p.direction.value,
                    p.entry_price, current_price,
                    ret * 100,
                    "CONFIRMED ✓" if ok else "FAILED ✗",
                )
                p.logged_5m = True

            if not p.logged_20m and elapsed >= _OUTCOME_20MIN_S:
                ok = _evaluate(p.direction, ret)
                _elog.info(
                    "[OUTCOME_20M] id=%s label=%-35s instr=%-12s dir=%s "
                    "entry=%.2f now=%.2f return=%+.4f%% → %s",
                    p.snapshot_id[:8],
                    p.candidate_label.value,
                    p.instrument_type.value,
                    p.direction.value,
                    p.entry_price, current_price,
                    ret * 100,
                    "CONFIRMED ✓" if ok else "FAILED ✗",
                )
                p.logged_20m = True

            if not (p.logged_5m and p.logged_20m):
                still_pending.append(p)

        self._pending = still_pending


# ─── Cooldown Manager ─────────────────────────────────────────────────────────

_COOLDOWN_S: float = 15.0        # GLOBAL — istalgan snapshot yozilsa, 15s kutiladi
_MIN_STRENGTH: float = 0.15      # bu dan past strength → shovqin, yozilmaydi


class _CooldownManager:

    def __init__(self):
        self._last_ts: float = 0.0   # global — istalgan snapshot bo'lsa timer reset

    def can_fire(self, instrument: InstrumentType, direction: Direction) -> bool:
        return (time.monotonic() - self._last_ts) >= _COOLDOWN_S

    def record(self, instrument: InstrumentType, direction: Direction) -> None:
        self._last_ts = time.monotonic()


# ─── Event Store ──────────────────────────────────────────────────────────────

class EventStore:

    def __init__(self):
        self._snapshots: list[MarketSnapshot] = []

    def append(self, snap: MarketSnapshot) -> None:
        self._snapshots.append(snap)

    @property
    def snapshots(self) -> list[MarketSnapshot]:
        return list(self._snapshots)

    @property
    def count(self) -> int:
        return len(self._snapshots)

    def summary(self) -> dict:
        from collections import Counter
        c = Counter(s.candidate_label.value for s in self._snapshots)
        return dict(sorted(c.items()))

    def instrument_summary(self) -> dict:
        from collections import Counter
        c = Counter(s.instrument_type.value for s in self._snapshots)
        return dict(sorted(c.items()))


event_store = EventStore()


# ─── Classifier — sign agreement, threshold yo'q ─────────────────────────────

def _classify(
    ofi_z:     float,
    slope_z:   float,
    imbalance: float,
    micro:     float,
    spread_z:  float,
) -> tuple[CandidateLabel, InstrumentType, Direction]:
    """
    Faqat SIGN AGREEMENT tekshiriladi — threshold yo'q.

    ┌─────────────────────────────────────────────────────────────┐
    │ ABSORPTION_BUY:                                             │
    │   sell flow:     ofi_z < 0  AND  slope_z < 0              │
    │   passive buyer: imbalance > 0  (bid side kuchli)          │
    │   price held:    |micro| kichik  (spread_z past)           │
    │                                                             │
    │ ABSORPTION_SELL:                                            │
    │   buy flow:      ofi_z > 0  AND  slope_z > 0              │
    │   passive seller: imbalance < 0  (ask side kuchli)         │
    │   price held:    |micro| kichik  (spread_z past)           │
    │                                                             │
    │ LIQUIDITY_GRAB_UP:                                          │
    │   thin book:     spread_z > 0  (elevated)                  │
    │   buy flow:      ofi_z > 0                                 │
    │   price spiked:  micro > 0                                 │
    │                                                             │
    │ LIQUIDITY_GRAB_DOWN:                                        │
    │   thin book:     spread_z > 0                              │
    │   sell flow:     ofi_z < 0                                 │
    │   price dropped: micro < 0                                 │
    │                                                             │
    │ MOMENTUM_UP:                                                │
    │   buy flow:      ofi_z > 0  AND  slope_z > 0              │
    │   price moving:  micro > 0  AND  imbalance > 0             │
    │   (no resistance — imbalance buyer side)                    │
    │                                                             │
    │ MOMENTUM_DOWN:                                              │
    │   sell flow:     ofi_z < 0  AND  slope_z < 0              │
    │   price moving:  micro < 0  AND  imbalance < 0             │
    │                                                             │
    │ MIXED:                                                      │
    │   signallar zid — hech bir pattern aniq emas               │
    └─────────────────────────────────────────────────────────────┘

    Ustunlik tartibi (overlap bo'lganda):
      1. ABSORPTION  — narx qotgan (eng muhim signal)
      2. LIQUIDITY   — spread elevated + narx siljigan
      3. MOMENTUM    — flow + narx birga
      4. MIXED       — hech biri emas
    """

    # sign shorthand
    ofi_dn    = ofi_z < 0
    ofi_up    = ofi_z > 0
    slope_dn  = slope_z < 0
    slope_up  = slope_z > 0
    imbal_bid = imbalance > 0    # bid side kuchli
    imbal_ask = imbalance < 0    # ask side kuchli
    micro_dn  = micro < 0
    micro_up  = micro > 0
    micro_sm  = abs(micro) < abs(imbalance) * 0.5  # micro kichik — narx qotgan (nisbiy)
    spread_el = spread_z > 0    # spread elevated

    # ── 1. ABSORPTION ─────────────────────────────────────────────────────────
    # sell flow + passive buyer + narx qotgan
    if ofi_dn and slope_dn and imbal_bid and not micro_up:
        return (
            CandidateLabel.ABSORPTION_BUY_CANDIDATE,
            InstrumentType.ABSORPTION,
            Direction.BUY,
        )

    # buy flow + passive seller + narx qotgan
    if ofi_up and slope_up and imbal_ask and not micro_dn:
        return (
            CandidateLabel.ABSORPTION_SELL_CANDIDATE,
            InstrumentType.ABSORPTION,
            Direction.SELL,
        )

    # ── 2. LIQUIDITY GRAB ─────────────────────────────────────────────────────
    # thin book + buy flow + narx yuqoriga siljidi
    if spread_el and ofi_up and micro_up:
        return (
            CandidateLabel.LIQUIDITY_GRAB_UP_CANDIDATE,
            InstrumentType.LIQUIDITY,
            Direction.BUY,
        )

    # thin book + sell flow + narx pastga siljidi
    if spread_el and ofi_dn and micro_dn:
        return (
            CandidateLabel.LIQUIDITY_GRAB_DOWN_CANDIDATE,
            InstrumentType.LIQUIDITY,
            Direction.SELL,
        )

    # ── 3. MOMENTUM ───────────────────────────────────────────────────────────
    # buy flow + narx yuqoriga + imbalance ham buyer tomonda
    if ofi_up and slope_up and micro_up and imbal_bid:
        return (
            CandidateLabel.MOMENTUM_UP_CANDIDATE,
            InstrumentType.MOMENTUM,
            Direction.BUY,
        )

    # sell flow + narx pastga + imbalance ham seller tomonda
    if ofi_dn and slope_dn and micro_dn and imbal_ask:
        return (
            CandidateLabel.MOMENTUM_DOWN_CANDIDATE,
            InstrumentType.MOMENTUM,
            Direction.SELL,
        )

    # ── 4. MIXED ──────────────────────────────────────────────────────────────
    # signallar zid — bu ham qimmatli data (shovqin baseline)
    # direction: ofi_z yo'nalishiga qarab
    direction = Direction.BUY if ofi_up else Direction.SELL
    return (
        CandidateLabel.MIXED,
        InstrumentType.MOMENTUM,   # placeholder — calibration da ignored
        direction,
    )


# ─── Main Detector ────────────────────────────────────────────────────────────

class EventDetector:
    """
    detector.update(shared_state) — har tick chaqiriladi.

    Threshold yo'q. Sign agreement asosida label qo'yiladi.
    Xom signallar + strength + outcome — hammasi logga yoziladi.
    Calibration keyinroq calibrate.py da.
    """

    def __init__(self):
        self._cooldown   = _CooldownManager()
        self._outcome    = _OutcomeTracker()
        self._tick_count = 0

    def update(self, shared_state: dict) -> Optional[MarketSnapshot]:

        self._tick_count += 1

        # ── narx ──────────────────────────────────────────────────────────────
        ob = shared_state.get("orderbook_live") or {}
        price = float(ob.get("mid_price") or 0.0)
        if price <= 0:
            return None

        # ── outcome tick ──────────────────────────────────────────────────────
        self._outcome.tick(price)

        # ── regime warmup tekshiruvi ───────────────────────────────────────────
        regime = shared_state.get("regime") or {}
        if regime.get("label") == "UNDEFINED":
            return None

        signals = regime.get("signals") or {}
        if not signals:
            return None

        # ── signallarni o'qish ────────────────────────────────────────────────
        ofi_z     = float(signals.get("ofi_z",        0.0))
        slope_z   = float(signals.get("slope_z",      0.0))
        imbalance = float(signals.get("imbalance",    0.0))
        micro     = float(signals.get("micro_vs_mid", 0.0))
        spread_z  = float(signals.get("spread_bps_z", 0.0))
        cvd       = float(shared_state.get("cvd_live") or 0.0)
        label     = regime.get("label", "UNKNOWN")
        conf      = float(regime.get("confidence", 0.0))

        # ── classify ──────────────────────────────────────────────────────────
        candidate, instrument, direction = _classify(
            ofi_z, slope_z, imbalance, micro, spread_z
        )

        # ── cooldown ──────────────────────────────────────────────────────────
        if not self._cooldown.can_fire(instrument, direction):
            return None

        # ── signal strength ───────────────────────────────────────────────────
        strength = SignalStrength(
            ofi_z_abs=round(abs(ofi_z), 4),
            slope_z_abs=round(abs(slope_z), 4),
            imbalance_abs=round(abs(imbalance), 4),
            micro_abs=round(abs(micro), 4),
            spread_z_val=round(spread_z, 4),
        )

        # shovqin filtri — juda zaif signallar yozilmaydi
        if strength.composite < _MIN_STRENGTH:
            return None

        # ── snapshot yaratish ─────────────────────────────────────────────────
        snap = MarketSnapshot(
            snapshot_id=str(uuid.uuid4()),
            candidate_label=candidate,
            instrument_type=instrument,
            direction=direction,
            timestamp=datetime.now(timezone.utc),
            price=price,
            regime_label=label,
            regime_confidence=conf,
            ofi_z=round(ofi_z, 4),
            slope_z=round(slope_z, 4),
            imbalance=round(imbalance, 4),
            micro_vs_mid=round(micro, 4),
            spread_bps_z=round(spread_z, 4),
            cvd=round(cvd, 1),
            strength=strength,
        )

        # ── store + log + outcome ─────────────────────────────────────────────
        event_store.append(snap)
        self._cooldown.record(instrument, direction)
        self._outcome.register(snap)
        self._log_snapshot(snap)

        return snap

    @staticmethod
    def _log_snapshot(snap: MarketSnapshot) -> None:
        _elog.info(
            "[SNAPSHOT] id=%s  label=%-35s  instr=%-12s  dir=%s  "
            "price=%-10.2f  regime=%-14s  conf=%.2f  "
            "strength=%.3f | "
            "ofi_z=%+.3f  slope_z=%+.3f  imbal=%+.3f  "
            "micro=%+.4f  spread_z=%+.3f  cvd=%+.1f",
            snap.snapshot_id[:8],
            snap.candidate_label.value,
            snap.instrument_type.value,
            snap.direction.value,
            snap.price,
            snap.regime_label,
            snap.regime_confidence,
            snap.strength.composite,
            snap.ofi_z,
            snap.slope_z,
            snap.imbalance,
            snap.micro_vs_mid,
            snap.spread_bps_z,
            snap.cvd,
        )

        # har 50 ta snapshotda summary
        if event_store.count % 50 == 0 and event_store.count > 0:
            _elog.info(
                "[SUMMARY] total=%d  by_label=%s  by_instrument=%s",
                event_store.count,
                event_store.summary(),
                event_store.instrument_summary(),
            )


# ─── Singleton ────────────────────────────────────────────────────────────────

detector = EventDetector()