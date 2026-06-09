
# src/regime.py
"""
MARKET REGIME DETECTION ENGINE — Stage 2.1E
─────────────────────────────────────────────
Final dizayn qarorlari:

  1. CVD slope  — raw CVD z-score emas. Slope history ustida z-score.
  2. UNDEFINED  — warmup tugamaguncha hech qachon RANGING qaytarilmaydi.
  3. RANGING aktiv gate — "signal yo'q" ≠ RANGING.
  4. Rezerv signallar — spread_bps_z, spread_contraction (COMPRESSION uchun).
  5. Discriminator arxitekturasi — Base + Discriminator qatlamlari.
  6. Hysteresis — rejim flickering oldini oladi.
  7. To'liq izolyatsiya — faqat shared_state orqali muloqot.

Stage 2.1E additions (observation only — not wired to weights):
  8. Efficiency Ratio (er, er_near_zero) — trend quality signal.
  9. CVD/OFI Consistency (cvd_ofi_consistent) — direction agreement signal.
 10. Spread-Adjusted Microprice (micro_vs_mid_spread_adj) — spread-normalised pressure.
 11. ValidationTracker: confidence buckets + Brier Score calibration.

Bug fixes:
  - Oscillation: removed h[i]!=0 guard that silently dropped 0-state transitions.
  - RANGING gate: Variant A — spread_ok AND (oscillation_ok OR micro_ok).

Reads:
    shared_state["orderbook_live"]   → imbalance, ofi, mid_price, microprice, spread_bps
    shared_state["cvd_live"]         → float

Writes:
    shared_state["regime"] → {
        label, confidence, scores, signals_snapshot, ts
        warmup_progress (only during UNDEFINED)
    }

Future phases:
    COMPRESSION → spread_bps_z + spread_contraction + volume_decline signallari tayyor
    PANIC       → spread_bps_z extreme + slope extreme
    ABSORPTION  → high trade intensity + micro stable
"""

import logging
import time
import pathlib as _pl
from collections import deque
from datetime import datetime, timezone

import numpy as np


# ── ValidationTracker logging setup (module-level, thread-safe) ───────────────
_VLOG_HANDLER_ADDED: bool = False

log = logging.getLogger("regime")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# CVD slope
_SLOPE_LOOKBACK: int = 10      # slope = (cvd_now - cvd_10_ago) / 10
_SLOPE_WINDOW:   int = 100     # slope history uzunligi (z-score uchun)

# OFI rolling aggregation (shovqin kamaytirish)
_OFI_ROLL:       int = 5       # oxirgi N tick OFI summasi

# Warmup
_WARMUP_TICKS:   int = 30      # shu tickgacha UNDEFINED qaytariladi

# Hysteresis
_HYSTERESIS_DELTA: float = 0.08  # rejim almashish uchun minimal confidence farqi

# Discriminator weight split
# score = BASE_W × base_score + DISC_W × disc_score
_BASE_W: float = 0.55
_DISC_W: float = 0.45

# RANGING gate — kamida bitta shart bajarilishi kerak
_RANGING_GATE_SPREAD_Z_MAX:    float = 1.2   # spread z-score chegarasi (normal zona)
_RANGING_GATE_OSCILLATION_MIN: float = 0.20  # slope sign change nisbati
# Recalibrated 0.25 → 0.20 after oscillation bug fix (Phase 1):
# Previously h[i] != 0 guard skipped transitions INTO zero-state (+1→0, -1→0),
# under-counting by ~15-25%. Fix counts ALL neighbor changes. Threshold
# lowered to keep gate sensitivity equivalent to pre-fix behaviour.
_RANGING_GATE_MICRO_MAX:       float = 0.25  # |micro_vs_mid| chegarasi

# Z-score minimal tarix
_ZSCORE_MIN_SAMPLES: int = 5

# Spread contraction threshold (rezerv — COMPRESSION uchun)
_SPREAD_CONTRACTION_WINDOW: int = 20  # oxirgi N tick ichida spread trendi

# Efficiency Ratio (Phase 4 — observation only, not wired to weights yet)
_ER_LOOKBACK: int = 30   # price history window for ER calculation


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
# Har rejim uchun: {"base": {...}, "discriminator": {...}}
# Qiymatlar normalize qilinmagan — _score_layer() ichida normalize qilinadi.
#
# COMPRESSION, PANIC, ABSORPTION — hozir yo'q, lekin rezerv signallar
# SignalExtractor da hozirdan tayyor. Keyingi fazada faqat shu dict kengayadi.

_WEIGHTS: dict = {
    "TRENDING_UP": {
        "base": {
            "slope_z":      0.35,   # CVD tez o'smoqda
            "ofi_z":        0.30,   # aggressive buyers at best level
            "imbalance":    0.35,   # bid depth > ask depth
        },
        "discriminator": {
            "micro_vs_mid": 1.00,   # microprice > midprice → buyer pressure confirmed
        },
    },
    "TRENDING_DOWN": {
        "base": {
            "slope_z_neg":  0.35,   # CVD tez tushmoqda
            "ofi_z_neg":    0.30,   # aggressive sellers
            "imbalance_neg":0.35,   # ask depth > bid depth
        },
        "discriminator": {
            "micro_vs_mid_neg": 1.00,  # microprice < midprice → seller pressure confirmed
        },
    },
    "RANGING": {
        "base": {
            "imbalance_near_zero":    0.30,
            "ofi_near_zero":          0.30,
            "micro_vs_mid_near_zero": 0.40,
        },
        "discriminator": {
            "slope_oscillation":      0.60,  # CVD yo'nalishi tez-tez o'zgaryapti
            "spread_normal":          0.40,  # spread odatiy chegarada
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(value, default: float = 0.0) -> float:
    """NaN / Inf / None ga qarshi himoya."""
    try:
        v = float(value)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return default


def _zscore_clipped(value: float, history: deque, min_samples: int = _ZSCORE_MIN_SAMPLES) -> float:
    """
    Z-score → clip[-3,+3] → scale[-1,+1].
    Tarix yetarli bo'lmasa 0.0 qaytaradi.
    """
    if len(history) < min_samples:
        return 0.0
    arr  = np.array(history, dtype=float)
    mean = arr.mean()
    std  = arr.std()
    if std < 1e-10:
        return 0.0
    z = (value - mean) / std
    return float(np.clip(z, -3.0, 3.0)) / 3.0


def _score_layer(signals: dict, weights: dict) -> float:
    """
    Bir qatlam uchun weighted relu score.

    score = Σ( relu(signal_i) × weight_i ) / Σ(weight_i)

    relu() — manfiy signal hissa qo'shmaydi, lekin jarima ham yo'q.
    Natija ∈ [0, 1].
    """
    total_w = sum(weights.values())
    if total_w < 1e-10:
        return 0.0
    weighted = sum(
        max(0.0, signals.get(k, 0.0)) * w
        for k, w in weights.items()
    )
    return weighted / total_w


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class _SignalExtractor:
    """
    Vazifasi:
        shared_state dan xom qiymatlarni tortadi,
        normallashtiradi va classifier uchun tayyor signals dict qaytaradi.

    CVD:
        raw cvd_live → slope (stasionar) → slope history → zscore

    OFI:
        raw ofi (tick-level, shovqinli) → rolling sum → zscore

    Rezerv signallar (COMPRESSION / PANIC uchun, hozir unused):
        spread_bps_z        — spread tarixiy o'rtasiga nisbatan
        spread_contraction  — spread so'nggi N tickda qisqaryaptimi
    """

    def __init__(self):
        # CVD slope
        self._cvd_raw_history:   deque = deque(maxlen=_SLOPE_LOOKBACK + 1)
        self._slope_history:     deque = deque(maxlen=_SLOPE_WINDOW)

        # OFI rolling
        self._ofi_raw_history:   deque = deque(maxlen=_OFI_ROLL)
        self._ofi_roll_history:  deque = deque(maxlen=_SLOPE_WINDOW)

        # Spread (rezerv)
        self._spread_bps_history: deque = deque(maxlen=_SLOPE_WINDOW)

        # Slope sign tracking (RANGING oscillation uchun)
        self._slope_sign_history: deque = deque(maxlen=20)

        # Mid-price history — Efficiency Ratio uchun (Phase 4, observation only)
        self._mid_price_history:  deque = deque(maxlen=_ER_LOOKBACK)

    def extract(self, shared_state: dict) -> dict | None:
        """
        Returns normalized signals dict, yoki None agar ma'lumot yo'q.

        Barcha qaytariladigan qiymatlar float, asosan [-1, +1] oralig'ida.
        Rezerv signallar ham chiqariladi (hozir unused bo'lsa ham).
        """
        ob = shared_state.get("orderbook_live")
        if not ob:
            return None

        cvd_raw = shared_state.get("cvd_live")
        if cvd_raw is None:
            return None

        # ── raw qiymatlar ─────────────────────────────────────────────────────
        cvd_raw     = _safe_float(cvd_raw)
        imbalance   = _safe_float(ob.get("imbalance",   0.0))  # [-1,+1]
        ofi_raw     = _safe_float(ob.get("ofi",         0.0))  # unbounded
        mid_price   = _safe_float(ob.get("mid_price",   0.0))
        microprice  = _safe_float(ob.get("microprice",  0.0))
        spread_bps  = _safe_float(ob.get("spread_bps",  0.0))

        # ── CVD slope ─────────────────────────────────────────────────────────
        self._cvd_raw_history.append(cvd_raw)
        slope = self._compute_slope()
        self._slope_history.append(slope)

        slope_z = _zscore_clipped(slope, self._slope_history)

        # slope oscillation — RANGING gate va discriminator uchun
        # sign: +1 musbat slope, -1 manfiy slope, 0 sifirga yaqin
        slope_sign = 1 if slope > 1e-6 else (-1 if slope < -1e-6 else 0)
        self._slope_sign_history.append(slope_sign)
        slope_oscillation = self._compute_oscillation()

        # ── OFI rolling sum + zscore ──────────────────────────────────────────
        self._ofi_raw_history.append(ofi_raw)
        ofi_rolled = float(sum(self._ofi_raw_history))
        self._ofi_roll_history.append(ofi_rolled)
        ofi_z = _zscore_clipped(ofi_rolled, self._ofi_roll_history)

        # ── Microprice vs Midprice ────────────────────────────────────────────
        if mid_price > 1e-10:
            raw_diff     = (microprice - mid_price) / mid_price
            micro_vs_mid = float(np.clip(raw_diff * 10_000, -1.0, 1.0))
        else:
            micro_vs_mid = 0.0

        # ── Spread BPS — rezerv signallar ─────────────────────────────────────
        self._spread_bps_history.append(spread_bps)
        spread_bps_z       = _zscore_clipped(spread_bps, self._spread_bps_history)
        spread_contraction = self._compute_spread_contraction()

        # ── Spread "normal" signal (RANGING discriminator) ────────────────────
        # spread_bps tarixiy o'rtasiga yaqin bo'lsa → spread_normal yuqori
        spread_normal = float(np.clip(1.0 - abs(spread_bps_z), 0.0, 1.0))

        # ── Efficiency Ratio (Phase 4 — observation only) ─────────────────────
        # ER = |price[-1] - price[0]| / sum(|diff(prices)|)
        # ER → 1.0: smooth directional move (trend)
        # ER → 0.0: noisy back-and-forth (ranging / chop)
        self._mid_price_history.append(mid_price)
        er, er_near_zero = self._compute_er()

        # ── CVD/OFI Consistency (Phase 5 — observation only) ──────────────────
        # 1.0 if CVD slope and OFI agree in direction; 0.0 if diverging
        # Divergence can precede trend exhaustion or false breakouts
        cvd_ofi_consistent = 1.0 if (
            (slope_z > 0 and ofi_z > 0) or (slope_z < 0 and ofi_z < 0)
        ) else 0.0

        # ── Spread-Adjusted Microprice (Phase 6 — observation only) ───────────
        # Normalises microprice deviation by current spread regime.
        # Wide spread → microprice deviation less informative → shrink the signal.
        # Existing micro_vs_mid kept unchanged for weight continuity.
        spread_z_clamped        = float(np.clip(abs(spread_bps_z), 0.0, 2.0))
        micro_vs_mid_spread_adj = float(
            np.clip(micro_vs_mid / (1.0 + spread_z_clamped), -1.0, 1.0)
        )

        # ── Signals dict ──────────────────────────────────────────────────────
        return {
            # ── directional ───────────────────────────────────────────────────
            "slope_z":                slope_z,
            "ofi_z":                  ofi_z,
            "imbalance":              imbalance,
            "micro_vs_mid":           micro_vs_mid,

            # ── negated (TRENDING_DOWN uchun) ─────────────────────────────────
            "slope_z_neg":            -slope_z,
            "ofi_z_neg":              -ofi_z,
            "imbalance_neg":          -imbalance,
            "micro_vs_mid_neg":       -micro_vs_mid,

            # ── near-zero (RANGING base uchun) ────────────────────────────────
            "imbalance_near_zero":    1.0 - abs(imbalance),
            "ofi_near_zero":          float(np.clip(1.0 - abs(ofi_z), 0.0, 1.0)),
            "micro_vs_mid_near_zero": 1.0 - abs(micro_vs_mid),

            # ── RANGING discriminator ─────────────────────────────────────────
            "slope_oscillation":      slope_oscillation,
            "spread_normal":          spread_normal,

            # ── rezerv — COMPRESSION / PANIC uchun (hozir unused) ────────────
            "spread_bps_z":           spread_bps_z,
            "spread_contraction":     spread_contraction,

            # ── Phase 4: Efficiency Ratio (observation only) ──────────────────
            "er":                     er,
            "er_near_zero":           er_near_zero,

            # ── Phase 5: CVD/OFI Consistency (observation only) ───────────────
            "cvd_ofi_consistent":     cvd_ofi_consistent,

            # ── Phase 6: Spread-Adjusted Microprice (observation only) ─────────
            "micro_vs_mid_spread_adj":        micro_vs_mid_spread_adj,

            # ── debug snapshot ────────────────────────────────────────────────
            "_raw_slope":             slope,
            "_raw_spread_bps":        spread_bps,
        }

    # ── ichki hisob-kitoblar ──────────────────────────────────────────────────

    def _compute_slope(self) -> float:
        """
        CVD slope = (cvd_hozir - cvd_N_tick_oldin) / N

        History to'lmagan bo'lsa 0.0 qaytaradi.
        Stasionar metrika — z-score uchun xavfsiz.
        """
        h = self._cvd_raw_history
        if len(h) < 2:
            return 0.0
        lookback = min(_SLOPE_LOOKBACK, len(h) - 1)
        return (h[-1] - h[-1 - lookback]) / lookback

    def _compute_oscillation(self) -> float:
        """
        Slope sign o'zgarish nisbati → [0, 1].

        RANGING da yo'nalish tez-tez o'zgaradi → yuqori oscillation.
        TRENDING da yo'nalish bir tomonda qoladi → past oscillation.

        Formula:
            changes = count( sign_i ≠ sign_{i-1} ) for i in history
            oscillation = changes / (len(history) - 1)

        BUG FIX (Phase 1):
            Removed the `h[i] != 0` guard that was silently discarding
            transitions INTO the zero-state (e.g. +1→0, −1→0).
            Such transitions are genuine direction changes and were causing
            systematic under-counting in ranging markets where slope
            repeatedly collapses through zero.
            _RANGING_GATE_OSCILLATION_MIN recalibrated 0.25→0.20 accordingly.
        """
        h = list(self._slope_sign_history)
        if len(h) < 3:
            return 0.0
        changes = sum(1 for i in range(1, len(h)) if h[i] != h[i - 1])
        return float(changes / (len(h) - 1))

    def _compute_spread_contraction(self) -> float:
        """
        Rezerv signal — COMPRESSION uchun.
        Spread so'nggi N tickda qisqaryaptimi?

        Formula:
            half = N // 2
            recent_mean  = mean(spread[-half:])
            earlier_mean = mean(spread[:half])
            contraction  = clip((earlier_mean - recent_mean) / (earlier_mean + ε), 0, 1)

        Spread kichraysa → contraction → 1.0
        Spread o'zgarmasa yoki kattaraysa → 0.0

        Hozir WEIGHTS da ishlatilmaydi. Phase 2B da aktivlashadi.
        """
        h = list(self._spread_bps_history)
        n = len(h)
        if n < _SPREAD_CONTRACTION_WINDOW:
            return 0.0
        half         = n // 2
        earlier_mean = float(np.mean(h[:half]))
        recent_mean  = float(np.mean(h[half:]))
        if earlier_mean < 1e-10:
            return 0.0
        contraction  = (earlier_mean - recent_mean) / earlier_mean
        return float(np.clip(contraction, 0.0, 1.0))

    def _compute_er(self) -> tuple[float, float]:
        """
        Efficiency Ratio (Phase 4 — observation only, not wired to weights).

        ER = |price[-1] - price[0]| / sum(|diff(prices)|)

        Measures how "efficiently" price moved over the lookback window:
            ER → 1.0 : price moved in a straight line (clean trend)
            ER → 0.0 : price oscillated back-and-forth (chop / ranging)

        er_near_zero = 1.0 - er  (analogous to near_zero pattern for RANGING signals)

        Returns (er, er_near_zero). Both ∈ [0, 1].
        Requires at least 2 samples; returns (0.0, 1.0) during warmup.
        """
        h = list(self._mid_price_history)
        if len(h) < 2:
            return 0.0, 1.0
        prices        = np.array(h, dtype=float)
        net_move      = abs(prices[-1] - prices[0])
        total_path    = float(np.sum(np.abs(np.diff(prices))))
        if total_path < 1e-10:
            # Price completely flat → perfect efficiency (no noise), but also no direction.
            # Return 0.0 to avoid misleading trend signal; 1.0 near_zero for ranging hint.
            return 0.0, 1.0
        er        = float(np.clip(net_move / total_path, 0.0, 1.0))
        er_near_zero = 1.0 - er
        return er, er_near_zero


# ─────────────────────────────────────────────────────────────────────────────
# REGIME CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class _RegimeClassifier:
    """
    Vazifasi:
        Har rejim uchun [0,1] oralig'ida raw score hisoblaydi.

    Discriminator arxitekturasi:
        score(R) = _BASE_W × base_score + _DISC_W × disc_score

        Base signals       — yo'nalish mavjudligini aniqlaydi
        Discriminator      — rejimni aniq tasdiqlaydi

    RANGING gate:
        Aktiv gate sharti bajarilmasa RANGING score = 0.
        "Signal yo'q" → RANGING emas.
    """

    def score(self, signals: dict) -> dict[str, float]:
        """
        Returns {regime_label: score ∈ [0,1]} barcha Phase 1 rejimlar uchun.
        """
        results = {}
        for regime, layer_weights in _WEIGHTS.items():
            base_score = _score_layer(signals, layer_weights["base"])
            disc_score = _score_layer(signals, layer_weights["discriminator"])
            raw        = _BASE_W * base_score + _DISC_W * disc_score

            if regime == "RANGING":
                raw = raw if self._gate_ranging(signals) else 0.0

            results[regime] = float(np.clip(raw, 0.0, 1.0))

        return results

    @staticmethod
    def _gate_ranging(signals: dict) -> bool:
        """
        RANGING aktiv gate — Variant A (Phase 2).

        Spread normality is a MANDATORY precondition:
            spread_ok = |spread_bps_z| < _RANGING_GATE_SPREAD_Z_MAX

        PLUS at least ONE behavioral confirmation:
            oscillation_ok = slope_oscillation > _RANGING_GATE_OSCILLATION_MIN
            micro_ok       = |micro_vs_mid|     < _RANGING_GATE_MICRO_MAX

        Gate = spread_ok AND (oscillation_ok OR micro_ok)

        Rationale for Variant A over original OR:
            Old `spread_ok OR oscillation_ok OR micro_ok` allowed any single
            weak signal to pass. In particular, spread_ok alone fired during
            pre-breakout compression phases (spread is "normal" but market is
            coiling, not ranging). Requiring spread_ok as a mandatory condition
            eliminates this class of false positives. The behavioral clause
            (oscillation OR micro) then confirms actual two-sided indecision.

        Uchala shart birgalikda bajarilmasa → RANGING score = 0.
        """
        spread_ok      = abs(signals.get("spread_bps_z",   0.0)) < _RANGING_GATE_SPREAD_Z_MAX
        oscillation_ok = signals.get("slope_oscillation",  0.0) > _RANGING_GATE_OSCILLATION_MIN
        micro_ok       = abs(signals.get("micro_vs_mid",   0.0)) < _RANGING_GATE_MICRO_MAX

        return spread_ok and (oscillation_ok or micro_ok)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class _ConfidenceModel:
    """
    Vazifasi:
        Xom score lug'atidan g'olib rejim va confidence chiqaradi.

    Ikki komponent alohida o'lchanadi:
        signal_strength = winner_score                    ∈ [0,1]
        separation      = margin / (winner_score + ε)    ∈ [0,1]

        margin = winner_score - runner_up_score

        confidence = α × signal_strength + (1−α) × separation
                   = 0.6 × winner + 0.4 × (margin / winner)

    Nima uchun bunday:
        Oldingi formula  winner × (1+margin)/2  chekka holatlarda
        (past winner + katta margin) noto'g'ri intuitsiya berardi.
        Bu formula ikkala komponentni mustaqil baholaydi.

    Natija ∈ [0.0, 1.0].
    """

    _ALPHA: float = 0.60   # signal_strength og'irligi

    def compute(self, scores: dict[str, float]) -> tuple[str, float]:
        """
        Returns (winner_label, confidence).
        Barcha score sifir bo'lsa → ("UNDEFINED", 0.0).
        """
        if not scores or all(v < 1e-10 for v in scores.values()):
            return "UNDEFINED", 0.0

        sorted_scores  = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        winner_label, winner_score = sorted_scores[0]
        runner_up_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

        margin           = winner_score - runner_up_score
        signal_strength  = winner_score
        separation       = margin / (winner_score + 1e-10)

        confidence = (
            self._ALPHA * signal_strength
            + (1.0 - self._ALPHA) * separation
        )
        return winner_label, float(np.clip(confidence, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# HYSTERESIS FILTER
# ─────────────────────────────────────────────────────────────────────────────

class _HysteresisFilter:
    """
    Vazifasi:
        Tez-tez rejim almashuvini — "flickering" — oldini oladi.

    Qoida:
        Rejim almashishi uchun:
            new_confidence − current_confidence ≥ HYSTERESIS_DELTA

        Istisnolar:
            1. Xuddi shu rejim → har doim qabul qilinadi (confidence yangilanadi)
            2. current_label == "UNDEFINED" → har doim qabul qilinadi (warmup chiqishi)

    Warmup davri bu class da boshqarilmaydi —
    RegimeEngine o'zi warmup ni boshqaradi va bu classni chaqirmaydi.
    """

    def __init__(self):
        self._delta: float = _HYSTERESIS_DELTA

    def should_switch(
        self,
        current_label: str,
        new_label:     str,
        current_conf:  float,
        new_conf:      float,
    ) -> bool:
        # UNDEFINED dan chiqish — minimal confidence kerak (LOGIC-02 fix)
        # Warmup tugagandan darhol confidence=0.15 da rejim o'rnatilishi noto'g'ri.
        if current_label == "UNDEFINED":
            return new_conf >= 0.30

        # Xuddi shu rejim — confidence yangilanadi
        if current_label == new_label:
            return True

        # Yangi rejim yetarlicha ustun bo'lishi kerak
        return (new_conf - current_conf) >= self._delta



# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION TRACKER
# ─────────────────────────────────────────────────────────────────────────────

_RANGING_SUCCESS_THRESHOLD: float = 0.0005  # LOGIC-01 fix: was 0.0010 (2x TREND threshold).
                                             # 0.0010 gave artificially high RANGING win rate.
                                             # Now equal to TREND threshold for fair calibration.
_TREND_SUCCESS_THRESHOLD:   float = 0.0005  # 0.05% — minimum economically meaningful move.
                                             # Prevents directional coin-flip validation:
                                             # pure noise (< 0.05%) does not count as success.
_VALIDATION_DELAY_S:        float = 20.0
_STATS_INTERVAL_S:          float = 60.0
_PERIODIC_SNAPSHOT_INTERVAL_S: float = 60.0  # periodic snapshot: har N sekunda joriy rejim pending ga

# Confidence bucket boundaries for calibration tracking (Phase 3)
_CONF_BUCKETS: list[tuple[float, float]] = [
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.0),
]

class _ValidationTracker:
    """
    Tracks regime prediction outcomes and confidence calibration.

    Phase 3 additions:
        - Per-confidence-bucket win rate tracking
        - Brier Score accumulation
          BS = (1/N) * Σ (predicted_prob − outcome)²
          Lower is better; 0.0 = perfect, 0.25 = random (for binary outcomes)
    """

    def __init__(self):
        self._pending        = []
        self._stats          = {
            "TRENDING_UP":   {"count": 0, "wins": 0},
            "TRENDING_DOWN": {"count": 0, "wins": 0},
            "RANGING":       {"count": 0, "wins": 0},
        }
        # ── Phase 3: confidence buckets ───────────────────────────────────────
        # keyed by bucket lower bound for stable ordering
        self._buckets: dict[float, dict] = {
            lo: {"count": 0, "wins": 0, "predicted_sum": 0.0}
            for lo, _ in _CONF_BUCKETS
        }
        # Brier Score accumulation: sum of (p - o)^2, count N
        self._brier_sum:   float = 0.0
        self._brier_count: int   = 0

        self._last_stats_ts:    float = 0.0
        self._last_snapshot_ts: float = 0.0   # periodic snapshot uchun

        # ── logging setup (module-level flag, thread-safe) ────────────────────
        global _VLOG_HANDLER_ADDED
        _vlog = logging.getLogger("regime.validation")
        if not _VLOG_HANDLER_ADDED and not _vlog.handlers:
            _fh = logging.FileHandler(_pl.Path("1.log"), encoding="utf-8")
            _fh.setLevel(logging.DEBUG)
            _fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
            _vlog.addHandler(_fh)
            _vlog.propagate = False
            _VLOG_HANDLER_ADDED = True
        _vlog.setLevel(logging.DEBUG)
        self._vlog = _vlog

    def on_regime_change(self, label, confidence, price):
        if label not in self._stats or price < 1e-3:
            return
        self._pending.append({
            "ts": time.monotonic(), "regime": label,
            "confidence": confidence, "entry_price": price,
        })

    def tick(self, current_price: float, current_label: str = "", current_conf: float = 0.0):
        now = time.monotonic()
        remaining = []
        for snap in self._pending:
            if now - snap["ts"] < _VALIDATION_DELAY_S:
                remaining.append(snap)
                continue
            self._evaluate(snap, current_price)
        self._pending = remaining

        # ── Periodic snapshot — rejim o'zgarmasa ham validation yig'iladi ──────
        # Har _PERIODIC_SNAPSHOT_INTERVAL_S sekunda joriy rejim pending ga qo'shiladi.
        # Bu validation sekinligi muammosini hal qiladi: bozor bir rejimda uzoq
        # qolsa ham calibration data to'planadi.
        if (
            current_label in self._stats
            and current_price > 1e-3
            and now - self._last_snapshot_ts >= _PERIODIC_SNAPSHOT_INTERVAL_S
        ):
            self._pending.append({
                "ts":          now,
                "regime":      current_label,
                "confidence":  current_conf,
                "entry_price": current_price,
            })
            self._last_snapshot_ts = now

        if now - self._last_stats_ts >= _STATS_INTERVAL_S:
            self._log_stats()
            self._last_stats_ts = now

    def _evaluate(self, snap, future_price):
        entry   = snap["entry_price"]
        ret_pct = (future_price - entry) / entry
        regime  = snap["regime"]
        conf    = snap["confidence"]

        if regime == "TRENDING_UP":
            success = ret_pct > _TREND_SUCCESS_THRESHOLD
        elif regime == "TRENDING_DOWN":
            success = ret_pct < -_TREND_SUCCESS_THRESHOLD
        else:
            success = abs(ret_pct) < _RANGING_SUCCESS_THRESHOLD

        outcome = 1.0 if success else 0.0

        if regime in self._stats:
            self._stats[regime]["count"] += 1
            if success:
                self._stats[regime]["wins"] += 1

        # ── Phase 3: bucket update ────────────────────────────────────────────
        bucket_lo = self._find_bucket(conf)
        if bucket_lo is not None:
            b = self._buckets[bucket_lo]
            b["count"]         += 1
            b["wins"]          += int(success)
            b["predicted_sum"] += conf

        # ── Phase 3: Brier Score update ───────────────────────────────────────
        # threshold va margin — faqat TRENDING rejimlari uchun mazmunli;
        # RANGING uchun ham chiqariladi, lekin interpretation farqli.
        threshold = (
            _TREND_SUCCESS_THRESHOLD
            if regime in ("TRENDING_UP", "TRENDING_DOWN")
            else _RANGING_SUCCESS_THRESHOLD
        )
        margin = abs(ret_pct) - threshold   # musbat → threshold oshildi, manfiy → oshilmadi

        self._brier_sum   += (conf - outcome) ** 2
        self._brier_count += 1

        self._vlog.info(
            "[REGIME_VALIDATION] regime=%s confidence=%.2f "
            "entry_price=%.2f future_price=%.2f "
            "return=%+.5f threshold=%.5f margin=%+.5f success=%s",
            regime, conf, entry, future_price,
            ret_pct, threshold, margin, success,
        ) 

    @staticmethod
    def _find_bucket(conf: float) -> float | None:
        """Returns the lower bound of the matching confidence bucket, or None."""
        for lo, hi in _CONF_BUCKETS:
            if lo <= conf < hi:
                return lo
        # conf == 1.0 edge case → assign to top bucket
        if conf >= _CONF_BUCKETS[-1][1]:
            return _CONF_BUCKETS[-1][0]
        return None

    def _log_stats(self):
        # ── existing per-regime stats ──────────────────────────────────────────
        parts = []
        for label, s in self._stats.items():
            if s["count"] == 0:
                parts.append(f"{label}: count=0")
            else:
                wr = 100.0 * s["wins"] / s["count"]
                parts.append(f"{label}: count={s['count']} win_rate={wr:.1f}%")
        self._vlog.info("[REGIME_STATS] %s", "  ".join(parts))

        # ── Phase 3: bucket calibration stats ────────────────────────────────
        bucket_parts = []
        for lo, hi in _CONF_BUCKETS:
            b = self._buckets[lo]
            n = b["count"]
            if n == 0:
                bucket_parts.append(f"[{lo:.1f}-{hi:.1f}]: n=0")
            else:
                actual_wr   = b["wins"] / n
                predicted   = b["predicted_sum"] / n
                bucket_parts.append(
                    f"[{lo:.1f}-{hi:.1f}]: n={n} "
                    f"pred={predicted:.3f} actual={actual_wr:.3f}"
                )
        self._vlog.info("[REGIME_BUCKETS] %s", "  ".join(bucket_parts))

        # ── Phase 3: Brier Score ──────────────────────────────────────────────
        if self._brier_count > 0:
            brier = self._brier_sum / self._brier_count
            self._vlog.info(
                "[REGIME_BRIER] brier_score=%.4f n=%d  "
                "(0.0=perfect 0.25=random)",
                brier, self._brier_count,
            )

# ─────────────────────────────────────────────────────────────────────────────
# REGIME ENGINE  (public API)
# ─────────────────────────────────────────────────────────────────────────────

class RegimeEngine:
    """
    Market Regime Detection Engine — Stage 2.1A.

    Yagona public class. Tashqaridan faqat compute() ko'rinadi.

    Ishlatish:
        engine = RegimeEngine()

        # har 500ms da (async loop yoki thread):
        result = engine.compute(shared_state)

    Reads:
        shared_state["orderbook_live"]
        shared_state["cvd_live"]

    Writes:
        shared_state["regime"] = {
            "label":      str,      # UNDEFINED | TRENDING_UP | TRENDING_DOWN | RANGING
            "confidence": float,    # 0.0 – 1.0
            "scores":     dict,     # {regime: score} debug uchun
            "signals":    dict,     # signal snapshot debug uchun
            "ts":         datetime,

            # faqat UNDEFINED davrida:
            "warmup_progress": float  # 0.0 → 1.0
        }
    """

    def __init__(self):
        self._extractor  = _SignalExtractor()
        self._classifier = _RegimeClassifier()
        self._confidence = _ConfidenceModel()
        self._hysteresis = _HysteresisFilter()

        self._current_label: str   = "UNDEFINED"
        self._current_conf:  float = 0.0
        self._tick_count:    int   = 0
        self._validator      = _ValidationTracker()

    # ── public ────────────────────────────────────────────────────────────────

    def compute(self, shared_state: dict) -> dict | None:
        """
        Bir detection cycle.

        Warmup davrida:
            signal extraction bajarilmaydi.
            UNDEFINED + warmup_progress qaytariladi.

        Aktiv davrida:
            to'liq pipeline: extract → score → confidence → hysteresis.

        Returns dict yoki None (ma'lumot yo'q yoki xato).
        """
        try:
            self._tick_count += 1

            # ── WARMUP ────────────────────────────────────────────────────────
            if self._tick_count <= _WARMUP_TICKS:
                # WARN-01 fix: warmup davrida ham extract() chaqiramiz.
                # Natija ishlatilmaydi, lekin deque history to'ldiriladi.
                # Warmup tugagandan keyin z-score lar darhol ma'noli bo'ladi.
                _ = self._extractor.extract(shared_state)
                result = {
                    "label":           "UNDEFINED",
                    "confidence":      0.0,
                    "scores":          {},
                    "signals":         {},
                    "warmup_progress": round(self._tick_count / _WARMUP_TICKS, 3),
                    "ts":              datetime.now(timezone.utc),
                }
                shared_state["regime"] = result
                return result

            # ── SIGNAL EXTRACTION ─────────────────────────────────────────────
            signals = self._extractor.extract(shared_state)
            if signals is None:
                log.debug("[regime] shared_state ma'lumot yo'q — skip")
                return None

            # ── SCORING ───────────────────────────────────────────────────────
            scores = self._classifier.score(signals)

            # ── CONFIDENCE ────────────────────────────────────────────────────
            new_label, new_conf = self._confidence.compute(scores)

            # ── HYSTERESIS ────────────────────────────────────────────────────
            if self._hysteresis.should_switch(
                self._current_label, new_label,
                self._current_conf,  new_conf,
            ):
                prev = self._current_label
                self._current_label = new_label
                self._current_conf  = new_conf
                if prev != new_label:
                    price = _safe_float((shared_state.get("orderbook_live") or {}).get("mid_price", 0))
                    self._validator.on_regime_change(new_label, new_conf, price)

            price = _safe_float((shared_state.get("orderbook_live") or {}).get("mid_price", 0))
            self._validator.tick(price, self._current_label, self._current_conf)

            # ── debug signallarni tozalash (_raw_ prefiksli) ──────────────────
            public_signals = {
                k: round(v, 5)
                for k, v in signals.items()
                if not k.startswith("_raw")
            }

            # ── Phase 4/5/6: log observation-only signals periodically ─────────
            if self._tick_count % 20 == 0:
                log.debug(
                    "[regime_obs] er=%.3f er_near_zero=%.3f "
                    "cvd_ofi_consistent=%.0f "
                    "micro_vs_mid_spread_adj=%.4f",
                    signals.get("er", 0.0),
                    signals.get("er_near_zero", 0.0),
                    signals.get("cvd_ofi_consistent", 0.0),
                    signals.get("micro_vs_mid_spread_adj", 0.0),
                )

            result = {
                "label":      self._current_label,
                "confidence": round(self._current_conf, 4),
                "scores":     {k: round(v, 4) for k, v in scores.items()},
                "signals":    public_signals,
                "ts":         datetime.now(timezone.utc),
            }

            shared_state["regime"] = result
            return result

        except Exception:
            log.exception("[regime] compute() unhandled exception")
            return None

    # ── convenience properties ────────────────────────────────────────────────

    @property
    def current_label(self) -> str:
        return self._current_label

    @property
    def current_confidence(self) -> float:
        return self._current_conf

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def is_warmed_up(self) -> bool:
        return self._tick_count > _WARMUP_TICKS