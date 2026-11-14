#!/usr/bin/env python3
"""
NASA Omega Validator — Scale-Invariant Kinematic Engine
=======================================================

Validates the edge biophysical Omega engine against:
  1. NASA Langley Crew State Monitoring (CSM) 256 Hz flight-crew physiology
     (`data/nasa_raw/nasa_csm_25/`, from official `25.zip`)
  2. NASA OSDR Inspiration4 cardiovascular multiplex panel OSD-575
     (`data/nasa_raw/LSDS-8_Multiplex_serum_cardiovascular_EvePanel_TRANSFORMED.csv`)

Outputs:
  - results/validation_report.json  (incl. Event-grounded false-alarm / precision)
  - results/omega_trajectory_summary.csv
  - results/omega_lead_time_plot.png
  - Terminal lead-time + false-alarm / precision tables
"""

from __future__ import annotations

import json
import re
import sys
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "nasa_raw"
CSM_DIR = DATA_RAW / "nasa_csm_25"
# Fallback if unpacked archive directory was left as 25.zip/
CSM_FALLBACK = DATA_RAW / "25.zip"
LSDS_PATH = DATA_RAW / "LSDS-8_Multiplex_serum_cardiovascular_EvePanel_TRANSFORMED.csv"
RESULTS_DIR = ROOT / "results"
REPORT_JSON = RESULTS_DIR / "validation_report.json"
TRAJECTORY_CSV = RESULTS_DIR / "omega_trajectory_summary.csv"
LEAD_TIME_PLOT = RESULTS_DIR / "omega_lead_time_plot.png"

# Cupola / Night Ops palette (Sentinel Settle)
COLOR_OBSIDIAN = "#121212"
COLOR_PANEL = "#1E1E1E"
COLOR_SAGE = "#8FA89B"
COLOR_AMBER = "#D6A84E"
COLOR_CORAL = "#E05A47"
COLOR_TEXT = "#E8E6E3"
COLOR_MUTED = "#9A9A96"
COLOR_PRE = "#8FA89B"
COLOR_POST = "#D6A84E"
COLOR_SHADE = "#D6A84E"

# ---------------------------------------------------------------------------
# Engine constants
# ---------------------------------------------------------------------------

EPS = 1e-7
K_RING = 30
DELTA_TAU_LO = 0.01
DELTA_TAU_HI = 5.0
OMEGA_BASE = 0.05
OMEGA_S_COEF = 0.12
OMEGA_APERP_COEF = 0.08
OMEGA_FC_COEF = 0.02
PHASE_METASTABLE = 0.07  # Metastable Thinning
PHASE_VETO = 0.19  # Phase Veto
NOMINAL_FS_HZ = 256.0
ECG_PEAK_DISTANCE_S = 0.30  # refractory ~300 ms
ECG_PROMINENCE_STD = 0.45
PRE_EVENT_HORIZON_S = 900.0  # mission-scale early-warning TP horizon
TACTICAL_HORIZON_S = 180.0  # strict 3-minute tactical TP horizon
GASKET_SUSTAIN_BEATS = 5  # Tier-1 tactical / acute workload gasket (beats)
HABITAT_GASKET_S = 15 * 60.0  # Tier-2 habitat / long-duration gasket (seconds)
BASELINE_CAL_BEATS = 30  # mobile-app-aligned re-anchor window
RMSSD_ROLL_BEATS = 30
RMSSD_DROP_FRAC = 0.75  # alert when RMSSD < 0.75 * baseline
HR_ELEVATION_BPM = 10.0  # alert when HR > baseline + 10 bpm
NOMINAL_OMEGA_FLOOR = 0.05  # re-anchor target (Nominal band floor)


@dataclass
class LeadTimeHit:
    dataset: str
    subject_or_session: str
    phase: str
    threshold: float
    first_crossing_t_s: float | None
    reference_event_t_s: float | None
    lead_time_s: float | None
    omega_at_crossing: float | None


# ---------------------------------------------------------------------------
# Scale-Invariant Kinematic Engine
# ---------------------------------------------------------------------------


def clip_delta_tau(s_t: np.ndarray) -> np.ndarray:
    """Bilaterally clamp interval / state magnitude used as Δτ."""
    return np.clip(np.asarray(s_t, dtype=float), DELTA_TAU_LO, DELTA_TAU_HI)


def nonuniform_velocity(s: np.ndarray, delta_tau: np.ndarray) -> np.ndarray:
    """Non-uniform finite-difference velocity along the state trajectory."""
    s = np.asarray(s, dtype=float)
    dt = np.asarray(delta_tau, dtype=float)
    v = np.zeros_like(s, dtype=float)
    if s.ndim == 1:
        ds = np.diff(s, prepend=s[0])
        v = ds / np.maximum(dt, EPS)
    else:
        ds = np.diff(s, axis=0, prepend=s[:1])
        v = ds / np.maximum(dt, EPS)[:, None]
    return v


def nonuniform_acceleration(v: np.ndarray, delta_tau: np.ndarray) -> np.ndarray:
    """Non-uniform finite-difference acceleration."""
    v = np.asarray(v, dtype=float)
    dt = np.asarray(delta_tau, dtype=float)
    if v.ndim == 1:
        dv = np.diff(v, prepend=v[0])
        return dv / np.maximum(dt, EPS)
    dv = np.diff(v, axis=0, prepend=v[:1])
    return dv / np.maximum(dt, EPS)[:, None]


def decompose_acceleration(
    a: np.ndarray, v: np.ndarray, epsilon: float = EPS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose acceleration into tangential laminar (a_parallel) and
    transverse orthogonal shear (a_perp) with titration firewall epsilon.

    For 1-D scalars, lift into phase space using [v, a] direction of motion
    reconstructed from consecutive (s, v) embeddings handled upstream when
    multi-D vectors are supplied. Here:
      - if vectors: project a onto v
      - if scalars: use 2-D phase embedding [v, a] relative to tangent [1, dv]
        via constructing vel_vec=[v,0] is degenerate; instead form
        vel_vec from consecutive scalar pairs as [Δs/Δτ, Δv/Δτ] = [v, a]
        and project the jerk-like step — practically we embed position as
        [s, v] before calling this. Callers should pass 2-D arrays when
        possible. Scalar fallback: a_parallel = a * sign continuity,
        a_perp = 0 is avoided by building [v_t, a_t] as the velocity vector
        and differencing it for acceleration.
    """
    a = np.asarray(a, dtype=float)
    v = np.asarray(v, dtype=float)

    if a.ndim == 1:
        # Build 2-D kinematic vectors from (v, a) phase flow
        vel_vec = np.column_stack([v, a])
        # Acceleration of the phase-space velocity (finite difference of vel_vec)
        dvel = np.diff(vel_vec, axis=0, prepend=vel_vec[:1])
        # Use |v| scale as local Δτ proxy already absorbed; treat dvel as a_vec
        a_vec = dvel
        speed2 = np.sum(vel_vec * vel_vec, axis=1) + epsilon
        proj_scalar = np.sum(a_vec * vel_vec, axis=1) / speed2
        a_parallel_vec = proj_scalar[:, None] * vel_vec
        a_perp_vec = a_vec - a_parallel_vec
        a_parallel_mag = np.linalg.norm(a_parallel_vec, axis=1)
        a_perp_mag = np.linalg.norm(a_perp_vec, axis=1)
        return a_parallel_mag, a_perp_mag, a_perp_vec

    # Multi-dimensional state (N, D)
    speed2 = np.sum(v * v, axis=1) + epsilon
    proj_scalar = np.sum(a * v, axis=1) / speed2
    a_parallel_vec = proj_scalar[:, None] * v
    a_perp_vec = a - a_parallel_vec
    a_parallel_mag = np.linalg.norm(a_parallel_vec, axis=1)
    a_perp_mag = np.linalg.norm(a_perp_vec, axis=1)
    return a_parallel_mag, a_perp_mag, a_perp_vec


def rolling_fc(a_perp_mag: np.ndarray, k: int = K_RING) -> np.ndarray:
    """Integrate |a_perp| over a rolling ring buffer of length K."""
    out = np.zeros(len(a_perp_mag), dtype=float)
    buf: deque[float] = deque(maxlen=k)
    running = 0.0
    for i, val in enumerate(np.asarray(a_perp_mag, dtype=float)):
        aval = abs(float(val))
        if len(buf) == buf.maxlen:
            running -= buf[0]
        buf.append(aval)
        running += aval
        out[i] = running
    return out


def omega_from_components(
    s_t: np.ndarray, a_perp_mag: np.ndarray, f_c: np.ndarray
) -> np.ndarray:
    """Ω_t = 0.05 + 0.12 s_t + 0.08 |a_perp|² + 0.02 F_c"""
    s_t = np.asarray(s_t, dtype=float)
    a_perp_mag = np.asarray(a_perp_mag, dtype=float)
    f_c = np.asarray(f_c, dtype=float)
    return (
        OMEGA_BASE
        + OMEGA_S_COEF * s_t
        + OMEGA_APERP_COEF * (a_perp_mag ** 2)
        + OMEGA_FC_COEF * f_c
    )


def run_kinematic_engine(
    state: np.ndarray,
    sample_times: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Full Scale-Invariant Kinematic Engine.

    Parameters
    ----------
    state : (N,) or (N, D)
        Dynamic state. For IBI series, pass 1-D inter-beat intervals (seconds).
        For biomarker panels, pass multi-D displacement magnitudes or vectors.
    sample_times : (N,), optional
        Absolute times for each state sample (seconds). If None, cumulative
        Δτ from clamped |s| is used for 1-D; for vectors, Euclidean norms.
    """
    state = np.asarray(state, dtype=float)
    if state.ndim == 1:
        s_scalar = np.abs(state)
        delta_tau = clip_delta_tau(s_scalar)
        v = nonuniform_velocity(state, delta_tau)
        a = nonuniform_acceleration(v, delta_tau)
        a_par, a_perp, _ = decompose_acceleration(a, v, EPS)
        s_for_omega = s_scalar
    else:
        s_scalar = np.linalg.norm(state, axis=1)
        delta_tau = clip_delta_tau(s_scalar)
        v = nonuniform_velocity(state, delta_tau)
        a = nonuniform_acceleration(v, delta_tau)
        a_par, a_perp, _ = decompose_acceleration(a, v, EPS)
        s_for_omega = s_scalar

    f_c = rolling_fc(a_perp, K_RING)
    omega = omega_from_components(s_for_omega, a_perp, f_c)

    if sample_times is None:
        t = np.cumsum(delta_tau)
        t = t - t[0]
    else:
        t = np.asarray(sample_times, dtype=float)

    return {
        "t": t,
        "s_t": s_for_omega,
        "delta_tau": delta_tau,
        "v_t": v if v.ndim == 1 else np.linalg.norm(v, axis=1),
        "a_t": a if a.ndim == 1 else np.linalg.norm(a, axis=1),
        "a_parallel": a_par,
        "a_perp": a_perp,
        "F_c": f_c,
        "Omega_t": omega,
    }


def phase_label(omega: float) -> str:
    if omega >= PHASE_VETO:
        return "Phase Veto"
    if omega >= PHASE_METASTABLE:
        return "Metastable Thinning"
    return "Nominal"


def first_crossing(t: np.ndarray, omega: np.ndarray, threshold: float) -> tuple[float | None, float | None]:
    idx = np.where(omega >= threshold)[0]
    if len(idx) == 0:
        return None, None
    i = int(idx[0])
    return float(t[i]), float(omega[i])


# ---------------------------------------------------------------------------
# ECG / IBI ingestion (Langley CSM)
# ---------------------------------------------------------------------------


def estimate_sampling_rate(time_secs: np.ndarray) -> float:
    dt = np.diff(np.asarray(time_secs, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return NOMINAL_FS_HZ
    med = float(np.median(dt))
    return float(1.0 / med) if med > 0 else NOMINAL_FS_HZ


def detect_rpeaks_ibi(
    ecg: np.ndarray, time_secs: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect R-peaks via scipy.signal.find_peaks and return
    (peak_times_s, ibi_seconds).
    """
    ecg = np.asarray(ecg, dtype=float)
    time_secs = np.asarray(time_secs, dtype=float)
    # Center and use absolute deviation so polarity is irrelevant
    x = ecg - np.nanmedian(ecg)
    # Prefer upright QRS; if energy is mostly negative spikes, flip
    if np.nanmean(x[x < 0] ** 2) > np.nanmean(x[x > 0] ** 2) * 1.2:
        x = -x
    distance = max(1, int(ECG_PEAK_DISTANCE_S * fs))
    prom = float(np.nanstd(x) * ECG_PROMINENCE_STD)
    peaks, _ = find_peaks(x, distance=distance, prominence=max(prom, EPS))
    if len(peaks) < 3:
        # Relaxed retry
        peaks, _ = find_peaks(x, distance=distance, prominence=max(prom * 0.25, EPS))
    peak_times = time_secs[peaks]
    ibi = np.diff(peak_times)
    # Keep physiologically plausible IBIs before clamping in the engine
    mask = (ibi > 0.25) & (ibi < 2.5)
    ibi = ibi[mask]
    # Align peak times to intervals (interval ends at peaks[1:])
    peak_times_ibi = peak_times[1:][mask]
    return peak_times_ibi, ibi


def extract_event_segments(
    time_secs: np.ndarray, event: np.ndarray
) -> list[tuple[float, float]]:
    """Return contiguous (start_t, end_t) intervals where Event != 0."""
    time_secs = np.asarray(time_secs, dtype=float)
    active = np.asarray(event, dtype=float) != 0
    if not np.any(active):
        return []
    idx = np.where(active)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return [(float(time_secs[s]), float(time_secs[e])) for s, e in zip(starts, ends)]


def tp_window_mask(
    sample_t: np.ndarray,
    event_segments: list[tuple[float, float]],
    horizon_s: float = PRE_EVENT_HORIZON_S,
) -> np.ndarray:
    """
    True where a sample falls inside an active Event window OR within
    `horizon_s` seconds before any Event onset (early-warning TP horizon).
    """
    t = np.asarray(sample_t, dtype=float)
    mask = np.zeros(len(t), dtype=bool)
    if not event_segments:
        return mask
    for start, end in event_segments:
        mask |= (t >= (start - horizon_s)) & (t <= end)
    return mask


def excursion_onsets(above: np.ndarray, min_len: int = 1) -> np.ndarray:
    """Start indices of contiguous True runs with length >= min_len."""
    above = np.asarray(above, dtype=bool)
    if len(above) == 0:
        return np.array([], dtype=int)
    padded = np.concatenate([[False], above, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    keep = (ends - starts) >= min_len
    return starts[keep]


def reanchor_omega(
    omega: np.ndarray,
    n_baseline: int = BASELINE_CAL_BEATS,
    nominal_floor: float = NOMINAL_OMEGA_FLOOR,
) -> tuple[np.ndarray, float]:
    """
    Mirror mobile-app 'Re-anchor Baseline':
      omega_offset = median(Omega[:n]) - 0.05
      Omega_cal = clip(Omega - offset, 0, None)
    so calibrated series starts in the Nominal zone (< 0.07).
    """
    omega = np.asarray(omega, dtype=float)
    n = max(1, min(int(n_baseline), len(omega)))
    offset = float(np.median(omega[:n])) - float(nominal_floor)
    omega_cal = np.clip(omega - offset, 0.0, None)
    return omega_cal, offset


def rolling_rmssd(ibi: np.ndarray, window: int = RMSSD_ROLL_BEATS) -> np.ndarray:
    """Beat-aligned rolling RMSSD (seconds) over successive IBI differences."""
    ibi = np.asarray(ibi, dtype=float)
    n = len(ibi)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        start = max(0, i - window + 1)
        seg = ibi[start : i + 1]
        if len(seg) < 2:
            continue
        out[i] = float(np.sqrt(np.mean(np.diff(seg) ** 2)))
    # Cold-start fill
    valid = np.where(np.isfinite(out))[0]
    if len(valid):
        out[: valid[0]] = out[valid[0]]
    return out


def build_hr_rmssd_alerts(
    ibi: np.ndarray,
    n_baseline: int = BASELINE_CAL_BEATS,
) -> dict[str, Any]:
    """Standard HR elevation and rolling-RMSSD-drop binary detectors."""
    ibi = np.asarray(ibi, dtype=float)
    hr = 60.0 / np.clip(ibi, EPS, None)
    n = max(1, min(int(n_baseline), len(ibi)))
    baseline_hr = float(np.median(hr[:n]))
    rmssd = rolling_rmssd(ibi, RMSSD_ROLL_BEATS)
    baseline_rmssd = float(np.nanmedian(rmssd[:n]))
    if not np.isfinite(baseline_rmssd) or baseline_rmssd < EPS:
        baseline_rmssd = float(np.nanmedian(rmssd[np.isfinite(rmssd)]))
    hr_alert = hr > (baseline_hr + HR_ELEVATION_BPM)
    rmssd_alert = np.isfinite(rmssd) & (rmssd < (RMSSD_DROP_FRAC * baseline_rmssd))
    return {
        "hr_bpm": hr,
        "baseline_hr_bpm": baseline_hr,
        "rmssd_s": rmssd,
        "baseline_rmssd_s": baseline_rmssd,
        "hr_alert": hr_alert.astype(bool),
        "rmssd_alert": rmssd_alert.astype(bool),
    }


def score_alert_series(
    sample_t: np.ndarray,
    alert: np.ndarray,
    event_segments: list[tuple[float, float]],
    duration_s: float,
    horizon_s: float,
    gasket_beats: int = GASKET_SUSTAIN_BEATS,
) -> dict[str, Any]:
    """
    Precision / FA/h / event-level recall for a binary detector under a
    Tier-1 sustained gasket and a chosen pre-event TP horizon.
    """
    sample_t = np.asarray(sample_t, dtype=float)
    alert = np.asarray(alert, dtype=bool)
    in_tp = tp_window_mask(sample_t, event_segments, horizon_s)
    hours = max(duration_s, EPS) / 3600.0
    total_events = len(event_segments)

    # Raw beat-level
    raw_tp = int(np.sum(alert & in_tp))
    raw_fp = int(np.sum(alert & ~in_tp))
    raw_denom = raw_tp + raw_fp

    # Sustained gasket onsets
    onset_idx = excursion_onsets(alert, min_len=gasket_beats)
    if len(onset_idx):
        sus_tp = int(np.sum(in_tp[onset_idx]))
        sus_fp = int(len(onset_idx) - sus_tp)
        hits_t = sample_t[onset_idx]
    else:
        sus_tp = sus_fp = 0
        hits_t = np.array([], dtype=float)
    sus_denom = sus_tp + sus_fp

    # Event-level recall from sustained onsets
    ev_tp = 0
    for start, end in event_segments:
        win_lo, win_hi = start - horizon_s, end
        if len(hits_t) and np.any((hits_t >= win_lo) & (hits_t <= win_hi)):
            ev_tp += 1
    ev_fn = total_events - ev_tp
    ev_recall = (ev_tp / total_events) if total_events else None

    # Raw event-level (any alert sample in window)
    raw_hits = sample_t[alert]
    raw_ev_tp = 0
    for start, end in event_segments:
        win_lo, win_hi = start - horizon_s, end
        if len(raw_hits) and np.any((raw_hits >= win_lo) & (raw_hits <= win_hi)):
            raw_ev_tp += 1
    raw_ev_fn = total_events - raw_ev_tp

    def _pack(tp: int, fp: int, ev_tp_: int, ev_fn_: int) -> dict[str, Any]:
        denom = tp + fp
        return {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "precision": (float(tp / denom) if denom > 0 else None),
            "false_alarms_per_hour": float(fp / hours),
            "total_ground_truth_events": int(total_events),
            "events_detected_tp": int(ev_tp_),
            "events_missed_fn": int(ev_fn_),
            "event_recall": (
                float(ev_tp_ / total_events) if total_events > 0 else None
            ),
        }

    return {
        "horizon_s": float(horizon_s),
        "gasket_beats": int(gasket_beats),
        "raw_beat_level": _pack(raw_tp, raw_fp, raw_ev_tp, raw_ev_fn),
        "sustained_gasket_filtered": _pack(sus_tp, sus_fp, ev_tp, ev_fn),
        "n_sustained_alerts": int(len(onset_idx)),
    }


def score_event_level_recall(
    sample_t: np.ndarray,
    omega: np.ndarray,
    threshold: float,
    event_segments: list[tuple[float, float]],
    gasket_beats: int = GASKET_SUSTAIN_BEATS,
    horizon_s: float = PRE_EVENT_HORIZON_S,
) -> dict[str, Any]:
    """
    Event-level recall: Detected Events / Total Ground-Truth Events.

    An event episode is detected if any Omega sample (raw) or any sustained
    gasket onset falls inside [onset - horizon, episode_end].
    """
    sample_t = np.asarray(sample_t, dtype=float)
    omega = np.asarray(omega, dtype=float)
    above = omega >= threshold
    total = len(event_segments)

    def _detected(mode: str) -> tuple[int, int, float | None]:
        if total == 0:
            return 0, 0, None
        if mode == "raw_beat_level":
            hits = sample_t[above]
        else:
            onsets = excursion_onsets(above, min_len=gasket_beats)
            hits = sample_t[onsets] if len(onsets) else np.array([], dtype=float)
        tp = 0
        for start, end in event_segments:
            win_lo, win_hi = start - horizon_s, end
            if len(hits) and np.any((hits >= win_lo) & (hits <= win_hi)):
                tp += 1
        fn = total - tp
        recall = tp / total
        return tp, fn, float(recall)

    raw_tp, raw_fn, raw_recall = _detected("raw_beat_level")
    sus_tp, sus_fn, sus_recall = _detected("sustained_gasket_filtered")
    return {
        "total_ground_truth_events": int(total),
        "raw_beat_level": {
            "events_detected_tp": int(raw_tp),
            "events_missed_fn": int(raw_fn),
            "event_recall": raw_recall,
        },
        "sustained_gasket_filtered": {
            "events_detected_tp": int(sus_tp),
            "events_missed_fn": int(sus_fn),
            "event_recall": sus_recall,
        },
    }


def score_threshold_alerts(
    sample_t: np.ndarray,
    omega: np.ndarray,
    threshold: float,
    event_segments: list[tuple[float, float]],
    duration_s: float,
    mean_ibi_s: float | None = None,
    horizon_s: float = PRE_EVENT_HORIZON_S,
    gasket_beats: int = GASKET_SUSTAIN_BEATS,
) -> dict[str, Any]:
    """
    Classify Omega excursions vs Event ground truth.

    Raw beat-level: every sample with Omega >= threshold.
    Sustained / gasket-filtered: contiguous runs lasting >= gasket_beats.
    Also scores Event-level recall (sensitivity) per ground-truth episode.
    """
    sample_t = np.asarray(sample_t, dtype=float)
    omega = np.asarray(omega, dtype=float)
    above = omega >= threshold
    in_tp = tp_window_mask(sample_t, event_segments, horizon_s)
    hours = max(duration_s, EPS) / 3600.0
    gasket_mean_seconds = (
        float(gasket_beats * mean_ibi_s) if mean_ibi_s is not None else None
    )

    def _pack(tp: int, fp: int) -> dict[str, float | int | None]:
        denom = tp + fp
        precision = (tp / denom) if denom > 0 else None
        fa_per_h = fp / hours
        return {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "precision": (float(precision) if precision is not None else None),
            "false_alarms_per_hour": float(fa_per_h),
        }

    # Beat-level / sample-level
    raw_tp = int(np.sum(above & in_tp))
    raw_fp = int(np.sum(above & ~in_tp))
    raw = _pack(raw_tp, raw_fp)

    # Excursion onsets (contiguous runs of length >= 1)
    raw_onset_idx = excursion_onsets(above, min_len=1)
    raw_onset_tp = int(np.sum(in_tp[raw_onset_idx])) if len(raw_onset_idx) else 0
    raw_onset_fp = int(len(raw_onset_idx) - raw_onset_tp)
    raw_excursion = _pack(raw_onset_tp, raw_onset_fp)

    # Sustained / gasket-filtered alert onsets
    sus_onset_idx = excursion_onsets(above, min_len=gasket_beats)
    sus_tp = int(np.sum(in_tp[sus_onset_idx])) if len(sus_onset_idx) else 0
    sus_fp = int(len(sus_onset_idx) - sus_tp)
    sustained = _pack(sus_tp, sus_fp)

    event_recall = score_event_level_recall(
        sample_t, omega, threshold, event_segments, gasket_beats, horizon_s
    )
    # Attach event recall onto alert modes for table joins
    raw["events_detected_tp"] = event_recall["raw_beat_level"]["events_detected_tp"]
    raw["events_missed_fn"] = event_recall["raw_beat_level"]["events_missed_fn"]
    raw["event_recall"] = event_recall["raw_beat_level"]["event_recall"]
    raw["total_ground_truth_events"] = event_recall["total_ground_truth_events"]
    sustained["events_detected_tp"] = event_recall["sustained_gasket_filtered"][
        "events_detected_tp"
    ]
    sustained["events_missed_fn"] = event_recall["sustained_gasket_filtered"][
        "events_missed_fn"
    ]
    sustained["event_recall"] = event_recall["sustained_gasket_filtered"]["event_recall"]
    sustained["total_ground_truth_events"] = event_recall["total_ground_truth_events"]

    return {
        "threshold": float(threshold),
        "gasket_beats": int(gasket_beats),
        "gasket_mean_seconds": gasket_mean_seconds,
        "raw_beat_level": raw,
        "raw_excursion_onsets": raw_excursion,
        "sustained_gasket_filtered": sustained,
        "event_level_recall": event_recall,
        "n_raw_beats_above": int(np.sum(above)),
        "n_raw_excursion_onsets": int(len(raw_onset_idx)),
        "n_sustained_alerts": int(len(sus_onset_idx)),
    }


def pool_method_comparison(
    method_session_rows: list[dict[str, Any]],
    total_duration_s: float,
    total_events: int,
    gasket_seconds_list: list[float],
) -> dict[str, Any]:
    """Pool dual-horizon Omega_cal metrics and head-to-head method comparison."""
    hours = max(total_duration_s, EPS) / 3600.0
    gasket_mean_seconds = (
        float(np.mean(gasket_seconds_list)) if gasket_seconds_list else None
    )
    gasket_seconds_range = (
        [float(np.min(gasket_seconds_list)), float(np.max(gasket_seconds_list))]
        if gasket_seconds_list
        else None
    )

    detector_labels = {
        "omega_cal_metastable": "Sentinel Settle Ω_cal ≥ 0.07",
        "omega_cal_veto": "Sentinel Settle Ω_cal ≥ 0.19",
        "rmssd_drop": "Rolling RMSSD drop (<0.75× baseline)",
        "hr_elevation": "Heart-rate elevation (+10 bpm)",
    }

    def _pool(horizon_key: str, det_key: str, mode: str) -> dict[str, Any]:
        tp = fp = ev_tp = ev_fn = 0
        for row in method_session_rows:
            block = row["scores"][horizon_key][det_key][mode]
            tp += int(block["true_positives"])
            fp += int(block["false_positives"])
            ev_tp += int(block["events_detected_tp"])
            ev_fn += int(block["events_missed_fn"])
        denom = tp + fp
        return {
            "true_positives": tp,
            "false_positives": fp,
            "precision": (float(tp / denom) if denom > 0 else None),
            "false_alarms_per_hour": float(fp / hours),
            "total_ground_truth_events": int(total_events),
            "events_detected_tp": int(ev_tp),
            "events_missed_fn": int(ev_fn),
            "event_recall": (float(ev_tp / total_events) if total_events else None),
        }

    comparison: dict[str, Any] = {}
    for horizon_key, horizon_s in (
        ("tactical_180s", TACTICAL_HORIZON_S),
        ("mission_900s", PRE_EVENT_HORIZON_S),
    ):
        comparison[horizon_key] = {
            "horizon_s": horizon_s,
            "methods": {},
        }
        for det_key, label in detector_labels.items():
            comparison[horizon_key]["methods"][det_key] = {
                "label": label,
                "raw_beat_level": _pool(horizon_key, det_key, "raw_beat_level"),
                "sustained_gasket_filtered": _pool(
                    horizon_key, det_key, "sustained_gasket_filtered"
                ),
            }

    # Omega-focused false_alarm_analysis (calibrated) for both horizons
    def _omega_fa(horizon_key: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for thr_name, det_key in (
            ("metastable_thinning", "omega_cal_metastable"),
            ("phase_veto", "omega_cal_veto"),
        ):
            raw = _pool(horizon_key, det_key, "raw_beat_level")
            sus = _pool(horizon_key, det_key, "sustained_gasket_filtered")
            out[thr_name] = {
                "threshold": (
                    PHASE_METASTABLE if "metastable" in thr_name else PHASE_VETO
                ),
                "total_ground_truth_events": int(total_events),
                "raw_beat_level": raw,
                "sustained_gasket_filtered": sus,
            }
        return out

    omega_fa = {
        "pre_event_horizon_s": PRE_EVENT_HORIZON_S,
        "tactical_horizon_s": TACTICAL_HORIZON_S,
        "gasket_beats": GASKET_SUSTAIN_BEATS,
        "gasket_sustain_beats": GASKET_SUSTAIN_BEATS,
        "gasket_mean_seconds": gasket_mean_seconds,
        "gasket_seconds_range": gasket_seconds_range,
        "habitat_gasket_s": HABITAT_GASKET_S,
        "total_duration_s": float(total_duration_s),
        "total_duration_h": float(hours),
        "total_ground_truth_events": int(total_events),
        "calibration": {
            "baseline_beats": BASELINE_CAL_BEATS,
            "nominal_floor": NOMINAL_OMEGA_FLOOR,
            "formula": "omega_offset = median(Omega[:30]) - 0.05; Omega_cal = clip(Omega - offset, 0, None)",
        },
        "by_horizon": {
            "tactical_180s": {
                "horizon_s": TACTICAL_HORIZON_S,
                "by_threshold": _omega_fa("tactical_180s"),
            },
            "mission_900s": {
                "horizon_s": PRE_EVENT_HORIZON_S,
                "by_threshold": _omega_fa("mission_900s"),
            },
        },
        # Default "by_threshold" = strict tactical 180 s (primary claim)
        "by_threshold": _omega_fa("tactical_180s"),
        "sessions": {
            row["session"]: {
                "duration_s": row["duration_s"],
                "n_event_segments": row["n_event_segments"],
                "omega_offset": row["omega_offset"],
                "gasket_mean_seconds": row["gasket_mean_seconds"],
            }
            for row in method_session_rows
        },
    }

    return {
        "omega_false_alarm_analysis": omega_fa,
        "method_comparison": comparison,
        "detector_baselines": {
            "rmssd_roll_beats": RMSSD_ROLL_BEATS,
            "rmssd_drop_frac": RMSSD_DROP_FRAC,
            "hr_elevation_bpm": HR_ELEVATION_BPM,
            "gasket_beats": GASKET_SUSTAIN_BEATS,
        },
    }


def youden_optimize_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    n_grid: int = 120,
    min_tau: float | None = None,
    max_tau: float | None = None,
) -> dict[str, Any]:
    """
    ROC / Youden's J optimization on beat-level Event-window labels.
    J = TPR + TNR - 1 = TPR - FPR. Returns argmax_J threshold.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pos = labels
    neg = ~labels
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0 or len(scores) == 0:
        fallback = PHASE_METASTABLE if min_tau is None else float(min_tau)
        return {
            "threshold": fallback,
            "youden_j": None,
            "tpr": None,
            "fpr": None,
            "n_pos": n_pos,
            "n_neg": n_neg,
        }

    lo = float(np.min(scores)) if min_tau is None else float(min_tau)
    hi = float(np.max(scores)) if max_tau is None else float(max_tau)
    if hi <= lo:
        hi = lo + EPS
    grid = np.unique(np.concatenate([np.linspace(lo, hi, n_grid), scores]))
    if min_tau is not None:
        grid = grid[grid >= min_tau - EPS]
    if max_tau is not None:
        grid = grid[grid <= max_tau + EPS]
    if len(grid) == 0:
        grid = np.array([lo])

    best = {"threshold": float(grid[0]), "youden_j": -1e18, "tpr": 0.0, "fpr": 1.0}
    for tau in grid:
        pred = scores >= tau
        tpr = float(np.sum(pred & pos) / n_pos)
        fpr = float(np.sum(pred & neg) / n_neg)
        j = tpr - fpr
        if j > best["youden_j"]:
            best = {
                "threshold": float(tau),
                "youden_j": float(j),
                "tpr": tpr,
                "fpr": fpr,
            }
    best["n_pos"] = n_pos
    best["n_neg"] = n_neg
    return best


def optimize_thresholds_on_train(
    train_packs: list[dict[str, Any]],
    horizon_s: float = TACTICAL_HORIZON_S,
) -> dict[str, Any]:
    """Fit τ_metastable / τ_veto via Youden J on training scenarios only."""
    scores_list = []
    labels_list = []
    offsets = []
    for pack in train_packs:
        t = pack["t"]
        omega = pack["omega_cal"]
        labels = tp_window_mask(t, pack["event_segments"], horizon_s)
        scores_list.append(omega)
        labels_list.append(labels)
        offsets.append(float(pack["omega_offset"]))
    scores = np.concatenate(scores_list)
    labels = np.concatenate(labels_list)

    # Constrained ROC bands keep τ_meta / τ_veto operationally separated
    # (unconstrained Youden on beat imbalance can collapse both to one cut).
    meta = youden_optimize_threshold(
        scores, labels, min_tau=0.02, max_tau=0.20
    )
    veto_floor = max(float(meta["threshold"]) + 0.05, 0.12)
    veto = youden_optimize_threshold(
        scores, labels, min_tau=veto_floor, max_tau=0.50
    )
    if veto["threshold"] <= meta["threshold"]:
        veto = {
            **veto,
            "threshold": float(meta["threshold"] + (PHASE_VETO - PHASE_METASTABLE)),
        }
    return {
        "tau_metastable": float(meta["threshold"]),
        "tau_veto": float(veto["threshold"]),
        "metastable_roc": meta,
        "veto_roc": veto,
        "train_mean_omega_offset": float(np.mean(offsets)) if offsets else None,
        "n_train_beats": int(len(scores)),
        "n_train_positive_beats": int(np.sum(labels)),
        "search_bands": {
            "metastable": [0.02, 0.20],
            "veto_min_gap_above_meta": 0.05,
            "veto": [0.12, 0.50],
        },
    }


def lead_time_for_threshold(
    t: np.ndarray,
    omega_cal: np.ndarray,
    event_segments: list[tuple[float, float]],
    threshold: float,
    gasket_beats: int = GASKET_SUSTAIN_BEATS,
) -> dict[str, float | None]:
    """Sustained crossing lead time vs first Event onset on a held-out session."""
    cross_t, cross_om = first_sustained_crossing(
        t, omega_cal, threshold, n_sustain=gasket_beats
    )
    ref = float(event_segments[0][0]) if event_segments else None
    lead = (ref - cross_t) if (ref is not None and cross_t is not None) else None
    return {
        "first_crossing_t_s": cross_t,
        "reference_event_t_s": ref,
        "lead_time_s": (float(lead) if lead is not None else None),
        "omega_at_crossing": cross_om,
    }


def score_loso_fold(
    holdout: dict[str, Any],
    tau_metastable: float,
    tau_veto: float,
    horizon_s: float = TACTICAL_HORIZON_S,
    gasket_beats: int = GASKET_SUSTAIN_BEATS,
) -> dict[str, Any]:
    """Out-of-sample sustained-gasket scores + lead time for one held-out scenario."""
    t = holdout["t"]
    omega = holdout["omega_cal"]
    segs = holdout["event_segments"]
    dur = holdout["duration_s"]
    meta_score = score_alert_series(
        t, omega >= tau_metastable, segs, dur, horizon_s, gasket_beats
    )
    veto_score = score_alert_series(
        t, omega >= tau_veto, segs, dur, horizon_s, gasket_beats
    )
    lead = lead_time_for_threshold(t, omega, segs, tau_metastable, gasket_beats)
    return {
        "tau_metastable": float(tau_metastable),
        "tau_veto": float(tau_veto),
        "n_events": len(segs),
        "duration_s": float(dur),
        "metastable": meta_score["sustained_gasket_filtered"],
        "phase_veto": veto_score["sustained_gasket_filtered"],
        "lead_time": lead,
    }


def run_loso_holdout(
    packs: list[dict[str, Any]],
    horizon_s: float = TACTICAL_HORIZON_S,
) -> dict[str, Any]:
    """
    4-fold Leave-One-Scenario-Out across CA / DA / LOFT / SS.

    Thresholds (Youden) and theoretical 0.07/0.19 are fit/fixed on the 3
    training scenarios only; held-out session is scored out-of-sample.
    """
    by_id = {p["session"]: p for p in packs}
    order = [s for s in ("25_CA", "25_DA", "25_LOFT", "25_SS") if s in by_id]
    if len(order) < 2:
        order = sorted(by_id.keys())

    folds = []
    for hold_id in order:
        train_ids = [s for s in order if s != hold_id]
        train_packs = [by_id[s] for s in train_ids]
        hold = by_id[hold_id]
        fitted = optimize_thresholds_on_train(train_packs, horizon_s=horizon_s)

        youden = score_loso_fold(
            hold, fitted["tau_metastable"], fitted["tau_veto"], horizon_s
        )
        fixed = score_loso_fold(hold, PHASE_METASTABLE, PHASE_VETO, horizon_s)

        folds.append(
            {
                "held_out_scenario": hold_id,
                "train_scenarios": train_ids,
                "fitted_thresholds": {
                    "tau_metastable": fitted["tau_metastable"],
                    "tau_veto": fitted["tau_veto"],
                    "metastable_youden_j": fitted["metastable_roc"].get("youden_j"),
                    "veto_youden_j": fitted["veto_roc"].get("youden_j"),
                    "train_mean_omega_offset": fitted["train_mean_omega_offset"],
                },
                "youden_optimized": youden,
                "fixed_theoretical_0_07_0_19": fixed,
            }
        )

    def _pool(scheme_key: str, thr_key: str) -> dict[str, Any]:
        tp = fp = ev_tp = ev_tot = 0
        dur = 0.0
        leads = []
        for f in folds:
            block = f[scheme_key][thr_key]
            tp += int(block["true_positives"])
            fp += int(block["false_positives"])
            ev_tp += int(block["events_detected_tp"])
            ev_tot += int(block["total_ground_truth_events"])
            dur += float(f[scheme_key]["duration_s"])
            lt = f[scheme_key]["lead_time"].get("lead_time_s")
            if lt is not None:
                leads.append(float(lt))
        hours = max(dur, EPS) / 3600.0
        denom = tp + fp
        return {
            "true_positives": tp,
            "false_positives": fp,
            "precision": (float(tp / denom) if denom > 0 else None),
            "false_alarms_per_hour": float(fp / hours),
            "events_detected_tp": ev_tp,
            "total_ground_truth_events": ev_tot,
            "event_recall": (float(ev_tp / ev_tot) if ev_tot else None),
            "mean_lead_time_s": (float(np.mean(leads)) if leads else None),
            "n_folds_with_lead_time": len(leads),
            "total_duration_s": float(dur),
        }

    pooled = {
        "youden_optimized": {
            "metastable": _pool("youden_optimized", "metastable"),
            "phase_veto": _pool("youden_optimized", "phase_veto"),
        },
        "fixed_theoretical_0_07_0_19": {
            "metastable": _pool("fixed_theoretical_0_07_0_19", "metastable"),
            "phase_veto": _pool("fixed_theoretical_0_07_0_19", "phase_veto"),
        },
    }
    return {
        "n_folds": len(folds),
        "scenarios": order,
        "horizon_s": float(horizon_s),
        "gasket_beats": GASKET_SUSTAIN_BEATS,
        "folds": folds,
        "pooled": pooled,
        "note": (
            "Leave-One-Scenario-Out: Youden τ fit on 3 train scenarios only; "
            "held-out scored with Tier-1 5-beat gasket and 180 s tactical horizon. "
            "Per-session Omega_cal re-anchor uses only that session's first 30 beats."
        ),
    }


def print_loso_table(loso: dict[str, Any]) -> None:
    """Terminal Cross-Scenario Hold-Out (LOSO) table."""
    headers = (
        "Hold-out",
        "Scheme",
        "τ_meta/τ_veto",
        "Prec@0.07",
        "Rec",
        "FA/h",
        "Lead_s",
        "Prec@veto",
        "Rec@veto",
    )
    rows: list[list[str]] = []
    for f in loso["folds"]:
        for scheme, label in (
            ("fixed_theoretical_0_07_0_19", "Fixed 0.07/0.19"),
            ("youden_optimized", "Youden LOSO"),
        ):
            block = f[scheme]
            m = block["metastable"]
            v = block["phase_veto"]
            lt = block["lead_time"].get("lead_time_s")
            rows.append(
                [
                    f["held_out_scenario"].replace("25_", ""),
                    label,
                    f"{block['tau_metastable']:.3f}/{block['tau_veto']:.3f}",
                    f"{m['precision']:.3f}" if m["precision"] is not None else "—",
                    f"{m['events_detected_tp']}/{m['total_ground_truth_events']}",
                    f"{m['false_alarms_per_hour']:.2f}",
                    f"{lt:.1f}" if lt is not None else "—",
                    f"{v['precision']:.3f}" if v["precision"] is not None else "—",
                    f"{v['events_detected_tp']}/{v['total_ground_truth_events']}",
                ]
            )
    # Pooled rows
    for scheme, label in (
        ("fixed_theoretical_0_07_0_19", "Fixed 0.07/0.19"),
        ("youden_optimized", "Youden LOSO"),
    ):
        m = loso["pooled"][scheme]["metastable"]
        v = loso["pooled"][scheme]["phase_veto"]
        rows.append(
            [
                "POOLED",
                label,
                "—",
                f"{m['precision']:.3f}" if m["precision"] is not None else "—",
                f"{m['events_detected_tp']}/{m['total_ground_truth_events']}",
                f"{m['false_alarms_per_hour']:.2f}",
                (
                    f"{m['mean_lead_time_s']:.1f}"
                    if m["mean_lead_time_s"] is not None
                    else "—"
                ),
                f"{v['precision']:.3f}" if v["precision"] is not None else "—",
                f"{v['events_detected_tp']}/{v['total_ground_truth_events']}",
            ]
        )

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "-+-".join("-" * w for w in widths)
    fmt = " | ".join(f"{{:{w}}}" for w in widths)
    print("\n=== Cross-Scenario Hold-Out (LOSO) — 180 s horizon, Tier-1 gasket ===")
    print(f"  {loso['note']}")
    print(fmt.format(*headers))
    print(sep)
    for r in rows:
        print(fmt.format(*r))
    print()


def print_method_comparison_table(comparison: dict[str, Any]) -> None:
    """Head-to-head Omega_cal vs RMSSD vs HR under Tier-1 gasket."""
    headers = ("Horizon", "Method", "Precision", "Events TP/Tot", "Recall", "FA/h")
    rows: list[list[str]] = []
    for horizon_key, horizon_label in (
        ("tactical_180s", "180 s tactical"),
        ("mission_900s", "900 s mission"),
    ):
        block = comparison[horizon_key]["methods"]
        for det_key in (
            "omega_cal_metastable",
            "omega_cal_veto",
            "rmssd_drop",
            "hr_elevation",
        ):
            m = block[det_key]["sustained_gasket_filtered"]
            rows.append(
                [
                    horizon_label,
                    block[det_key]["label"],
                    f"{m['precision']:.3f}" if m["precision"] is not None else "—",
                    f"{m['events_detected_tp']}/{m['total_ground_truth_events']}",
                    f"{m['event_recall']:.3f}" if m["event_recall"] is not None else "—",
                    f"{m['false_alarms_per_hour']:.2f}",
                ]
            )
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "-+-".join("-" * w for w in widths)
    fmt = " | ".join(f"{{:{w}}}" for w in widths)
    print("\n=== Method Comparison (Tier-1 5-beat sustained gasket) ===")
    print(fmt.format(*headers))
    print(sep)
    for r in rows:
        print(fmt.format(*r))
    print()


def compute_osd575_cohort_metrics(lsds_crew: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Cohort-level sensitivity / specificity on OSD-575 (N=4 discrete blood draws).

    Positive class: R+1 post-flight Omega exceeds threshold.
    Negative class: pre-flight (L-*) mean Omega stays below threshold.
    """
    crews = sorted(lsds_crew.keys())
    by_threshold: dict[str, Any] = {}
    per_crew: dict[str, Any] = {}

    for crew in crews:
        cr = lsds_crew[crew]
        tags = [str(t) for t in cr["timepoints"]]
        om = np.asarray(cr["Omega_series"], dtype=float)
        pre_mask = np.array([t.startswith("L-") for t in tags])
        r1_idx = next((i for i, t in enumerate(tags) if t == "R+1"), None)
        pre_omega = float(np.mean(om[pre_mask])) if pre_mask.any() else float("nan")
        r1_omega = float(om[r1_idx]) if r1_idx is not None else float("nan")
        per_crew[crew] = {
            "preflight_mean_Omega": pre_omega,
            "r_plus_1_Omega": r1_omega,
        }

    for thr_name, thr in (
        ("metastable_thinning", PHASE_METASTABLE),
        ("phase_veto", PHASE_VETO),
    ):
        post_pos = pre_neg = 0
        n = 0
        for crew in crews:
            pre_o = per_crew[crew]["preflight_mean_Omega"]
            r1_o = per_crew[crew]["r_plus_1_Omega"]
            if not (np.isfinite(pre_o) and np.isfinite(r1_o)):
                continue
            n += 1
            if r1_o >= thr:
                post_pos += 1
            if pre_o < thr:
                pre_neg += 1
        sensitivity = (post_pos / n) if n else None
        specificity = (pre_neg / n) if n else None
        by_threshold[thr_name] = {
            "threshold": thr,
            "n_crew": n,
            "postflight_r1_exceedances": post_pos,
            "preflight_below_threshold": pre_neg,
            "sensitivity": (float(sensitivity) if sensitivity is not None else None),
            "specificity": (float(specificity) if specificity is not None else None),
        }

    return {
        "n_crew": len(crews),
        "note": (
            "OSD-575 consists of discrete blood-draw timepoints "
            "(not continuous ECG)."
        ),
        "per_crew": per_crew,
        "by_threshold": by_threshold,
    }


def aggregate_false_alarm_metrics(
    session_scores: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Pool precision, FA/h, and event-level recall across Langley sessions."""
    modes = ("raw_beat_level", "raw_excursion_onsets", "sustained_gasket_filtered")
    thresholds = {
        "metastable_thinning": PHASE_METASTABLE,
        "phase_veto": PHASE_VETO,
    }
    total_duration_s = sum(
        float(s.get("duration_s", 0.0)) for s in session_scores.values()
    )
    hours = max(total_duration_s, EPS) / 3600.0

    gasket_secs = [
        float(s["gasket_mean_seconds"])
        for s in session_scores.values()
        if s.get("gasket_mean_seconds") is not None
    ]
    gasket_mean_seconds = float(np.mean(gasket_secs)) if gasket_secs else None
    gasket_seconds_range = (
        [float(np.min(gasket_secs)), float(np.max(gasket_secs))]
        if gasket_secs
        else None
    )

    pooled: dict[str, Any] = {
        "pre_event_horizon_s": PRE_EVENT_HORIZON_S,
        "gasket_beats": GASKET_SUSTAIN_BEATS,
        "gasket_sustain_beats": GASKET_SUSTAIN_BEATS,
        "gasket_mean_seconds": gasket_mean_seconds,
        "gasket_seconds_range": gasket_seconds_range,
        "habitat_gasket_s": HABITAT_GASKET_S,
        "total_duration_s": float(total_duration_s),
        "total_duration_h": float(hours),
        "by_threshold": {},
        "sessions": session_scores,
    }
    for thr_name, thr in thresholds.items():
        by_mode: dict[str, Any] = {}
        # Event-level pooled counts (identical total events for both modes)
        total_events = sum(
            int(
                sess["thresholds"][thr_name]["event_level_recall"][
                    "total_ground_truth_events"
                ]
            )
            for sess in session_scores.values()
        )
        for mode in modes:
            tp = fp = 0
            ev_tp = ev_fn = 0
            for sess in session_scores.values():
                block = sess["thresholds"][thr_name][mode]
                tp += int(block["true_positives"])
                fp += int(block["false_positives"])
                if mode in ("raw_beat_level", "sustained_gasket_filtered"):
                    ev_tp += int(block.get("events_detected_tp", 0))
                    ev_fn += int(block.get("events_missed_fn", 0))
            denom = tp + fp
            entry: dict[str, Any] = {
                "true_positives": tp,
                "false_positives": fp,
                "precision": (float(tp / denom) if denom > 0 else None),
                "false_alarms_per_hour": float(fp / hours),
            }
            if mode in ("raw_beat_level", "sustained_gasket_filtered"):
                entry.update(
                    {
                        "total_ground_truth_events": int(total_events),
                        "events_detected_tp": int(ev_tp),
                        "events_missed_fn": int(ev_fn),
                        "event_recall": (
                            float(ev_tp / total_events) if total_events > 0 else None
                        ),
                    }
                )
            by_mode[mode] = entry
        pooled["by_threshold"][thr_name] = {
            "threshold": thr,
            "total_ground_truth_events": int(total_events),
            **by_mode,
        }
    return pooled


def print_false_alarm_table(
    fa_report: dict[str, Any],
    cohort: dict[str, Any] | None = None,
) -> None:
    """Terminal Precision + Recall summary (alert + event-level)."""
    headers = (
        "Threshold",
        "Mode",
        "AlertPrec",
        "FA/h",
        "Events TP/Tot",
        "EventRecall",
    )
    rows: list[list[str]] = []
    labels = {
        "raw_beat_level": "Raw beat-level",
        "sustained_gasket_filtered": "Sustained/gasket",
    }
    thr_labels = {
        "metastable_thinning": "Metastable 0.07",
        "phase_veto": "Phase Veto 0.19",
    }
    for thr_key, thr_block in fa_report["by_threshold"].items():
        for mode, label in labels.items():
            m = thr_block[mode]
            prec = m["precision"]
            recall = m.get("event_recall")
            ev_tp = m.get("events_detected_tp")
            ev_tot = m.get("total_ground_truth_events")
            rows.append(
                [
                    thr_labels[thr_key],
                    label,
                    f"{prec:.3f}" if prec is not None else "—",
                    f"{m['false_alarms_per_hour']:.2f}",
                    (
                        f"{ev_tp}/{ev_tot}"
                        if ev_tp is not None and ev_tot is not None
                        else "—"
                    ),
                    f"{recall:.3f}" if recall is not None else "—",
                ]
            )
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)
    ]
    sep = "-+-".join("-" * w for w in widths)
    fmt = " | ".join(f"{{:{w}}}" for w in widths)
    g_mean = fa_report.get("gasket_mean_seconds")
    g_range = fa_report.get("gasket_seconds_range")
    range_txt = (
        f"{g_range[0]:.2f}–{g_range[1]:.2f} s" if g_range else "n/a"
    )
    print(
        "\n=== Precision + Recall (Ω_cal, Event ground truth; "
        "primary horizon = 180 s tactical) ==="
    )
    print(
        f"  Pooled Langley duration = {fa_report['total_duration_s']:.1f} s "
        f"({fa_report['total_duration_h']:.3f} h)"
    )
    if g_mean is not None:
        print(
            f"  Tier-1 gasket = {fa_report['gasket_beats']} beats "
            f"(mean {g_mean:.2f} s, range {range_txt}); "
            f"Tier-2 habitat gasket = {fa_report['habitat_gasket_s']:.0f} s"
        )
    else:
        print(f"  Tier-1 gasket = {fa_report['gasket_beats']} beats")
    print(fmt.format(*headers))
    print(sep)
    for r in rows:
        print(fmt.format(*r))

    if cohort is not None:
        print("\n=== OSD-575 Cohort Sensitivity / Specificity (N=4, discrete draws) ===")
        print(f"  {cohort['note']}")
        ch = (
            "Threshold",
            "R+1 exceed",
            "Pre < thr",
            "Sensitivity",
            "Specificity",
        )
        crows = []
        for thr_key, label in thr_labels.items():
            b = cohort["by_threshold"][thr_key]
            crows.append(
                [
                    label,
                    f"{b['postflight_r1_exceedances']}/{b['n_crew']}",
                    f"{b['preflight_below_threshold']}/{b['n_crew']}",
                    f"{b['sensitivity']:.3f}" if b["sensitivity"] is not None else "—",
                    f"{b['specificity']:.3f}" if b["specificity"] is not None else "—",
                ]
            )
        cw = [max(len(h), *(len(r[i]) for r in crows)) for i, h in enumerate(ch)]
        csep = "-+-".join("-" * w for w in cw)
        cfmt = " | ".join(f"{{:{w}}}" for w in cw)
        print(cfmt.format(*ch))
        print(csep)
        for r in crows:
            print(cfmt.format(*r))
    print()


def ingest_csm_csv(path: Path) -> dict[str, Any]:
    """Ingest one Langley CSM CSV: prefer HR/IBI columns, else ECG R-peaks."""
    usecols = None
    peek = pd.read_csv(path, nrows=0)
    cols = list(peek.columns)
    lower = {c: c.lower() for c in cols}

    # Prefer explicit HR / IBI / RR columns if present
    hr_ibi_candidates = [
        c
        for c in cols
        if any(
            k in lower[c]
            for k in ("ibi", "rr_interval", "rrinterval", "nn_interval", "heart_rate", "hr_")
        )
        or lower[c] in ("hr", "rr", "ibi")
    ]

    keep = [c for c in cols if lower[c] in ("timesecs", "ecg", "r", "gsr", "event") or c in hr_ibi_candidates]
    if "TimeSecs" in cols and "TimeSecs" not in keep:
        keep.append("TimeSecs")
    if "ECG" in cols and "ECG" not in keep:
        keep.append("ECG")
    if "Event" in cols and "Event" not in keep:
        keep.append("Event")

    df = pd.read_csv(path, usecols=keep)
    time_secs = df["TimeSecs"].to_numpy(dtype=float)
    fs = estimate_sampling_rate(time_secs)
    event = df["Event"].to_numpy(dtype=float) if "Event" in df.columns else np.zeros(len(df))

    source = "ecg_rpeaks"
    if hr_ibi_candidates:
        col = hr_ibi_candidates[0]
        series = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        # If looks like HR (bpm), convert to IBI seconds
        med = np.nanmedian(series)
        if np.isfinite(med) and med > 20:  # bpm-like
            ibi = 60.0 / np.clip(series, 20.0, 220.0)
        else:
            ibi = series
        valid = np.isfinite(ibi) & (ibi > 0)
        # Downsample to change-points / non-duplicated if densely repeated
        t_ibi = time_secs[valid]
        ibi = ibi[valid]
        source = f"column:{col}"
        peak_times = t_ibi
    else:
        if "ECG" not in df.columns:
            raise ValueError(f"No ECG or IBI/HR columns in {path.name}")
        peak_times, ibi = detect_rpeaks_ibi(
            df["ECG"].to_numpy(dtype=float), time_secs, fs
        )
        source = "ecg_rpeaks"

    # Align kinematic sample times to beat timeline
    sample_times = peak_times if len(peak_times) == len(ibi) else None
    kin = run_kinematic_engine(ibi, sample_times=sample_times)

    # Reference event: first non-zero Event marker in the raw timeline
    event_idx = np.where(event != 0)[0]
    ref_event_t = float(time_secs[event_idx[0]]) if len(event_idx) else None
    event_segments = extract_event_segments(time_secs, event)

    meta = {
        "file": path.name,
        "n_samples": int(len(df)),
        "duration_s": float(time_secs[-1] - time_secs[0]) if len(time_secs) else 0.0,
        "sampling_rate_hz": fs,
        "columns": cols,
        "ibi_source": source,
        "n_ibi": int(len(ibi)),
        "mean_ibi_s": float(np.mean(ibi)) if len(ibi) else None,
        "mean_hr_bpm": float(60.0 / np.mean(ibi)) if len(ibi) else None,
        "reference_event_t_s": ref_event_t,
        "n_event_segments": len(event_segments),
        "event_segments": [
            {"start_t_s": float(a), "end_t_s": float(b)} for a, b in event_segments
        ],
    }
    return {
        "meta": meta,
        "kin": kin,
        "ibi": ibi,
        "time_secs": time_secs,
        "event": event,
        "event_segments": event_segments,
    }


# ---------------------------------------------------------------------------
# Inspiration4 LSDS-8 biomarker displacement
# ---------------------------------------------------------------------------


_SAMPLE_RE = re.compile(r"^(C\d+)_serum_(L-(\d+)|R\+(\d+))$")


def parse_lsds_sample(name: str) -> dict[str, Any] | None:
    m = _SAMPLE_RE.match(str(name).strip())
    if not m:
        return None
    crew = m.group(1)
    tag = m.group(2)
    if tag.startswith("L-"):
        phase = "preflight"
        day = -int(m.group(3))
    else:
        phase = "postflight"
        day = int(m.group(4))
    return {"crew": crew, "tag": tag, "phase": phase, "day": day}


def ingest_lsds(path: Path) -> dict[str, Any]:
    """
    Parse Inspiration4 multiplex serum panel and compute pre- vs post-flight
    multi-biomarker state vector displacement (s_t) and transverse shear.
    """
    df = pd.read_csv(path)
    # Prefer percent columns (already normalized to reference) when present
    pct_cols = [c for c in df.columns if c.endswith("_percent")]
    conc_cols = [
        c
        for c in df.columns
        if "concentration" in c and c != "Sample Name"
    ]
    feature_cols = pct_cols if pct_cols else conc_cols
    if not feature_cols:
        raise ValueError("No biomarker feature columns found in LSDS-8 CSV")

    # Global z-score so multi-biomarker ||Δ|| is scale-comparable to IBI seconds
    feat_mat = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    feat_mean = np.nanmean(feat_mat, axis=0)
    feat_std = np.nanstd(feat_mat, axis=0)
    feat_std = np.where(feat_std < EPS, 1.0, feat_std)
    z_mat = (feat_mat - feat_mean) / feat_std
    z_mat = np.nan_to_num(z_mat, nan=0.0)

    records = []
    for pos, (_, row) in enumerate(df.iterrows()):
        parsed = parse_lsds_sample(row["Sample Name"])
        if not parsed:
            continue
        vec = z_mat[pos]
        records.append({**parsed, "vector": vec, "sample": row["Sample Name"]})

    by_crew: dict[str, list] = {}
    for r in records:
        by_crew.setdefault(r["crew"], []).append(r)

    crew_results = {}
    trajectory_rows = []
    lead_hits: list[LeadTimeHit] = []
    n_feat = float(len(feature_cols))

    for crew, items in sorted(by_crew.items()):
        items = sorted(items, key=lambda x: x["day"])
        # Baseline = mean of preflight z-vectors
        pre = [it for it in items if it["phase"] == "preflight"]
        if not pre:
            continue
        baseline = np.nanmean(np.vstack([it["vector"] for it in pre]), axis=0)
        # Displacement vectors relative to baseline (RMS-scaled)
        disp = []
        days = []
        tags = []
        for it in items:
            dvec = (it["vector"] - baseline) / np.sqrt(n_feat)
            dvec = np.nan_to_num(dvec, nan=0.0)
            disp.append(dvec)
            days.append(float(it["day"]))
            tags.append(it["tag"])
        disp_arr = np.vstack(disp)
        # Map mission day to a synthetic time axis in seconds (1 day = 86400 s)
        # for engine Δτ clamping we still use state magnitude; times for reporting
        t_days = np.asarray(days, dtype=float)
        kin = run_kinematic_engine(disp_arr, sample_times=t_days)

        # Pre vs post summary displacement (RMS z-space)
        post = [it for it in items if it["phase"] == "postflight"]
        scale = np.sqrt(n_feat)
        pre_mag = float(
            np.mean([np.linalg.norm((it["vector"] - baseline) / scale) for it in pre])
        )
        post_mag = (
            float(
                np.mean(
                    [
                        np.linalg.norm(
                            np.nan_to_num((it["vector"] - baseline) / scale, nan=0.0)
                        )
                        for it in post
                    ]
                )
            )
            if post
            else None
        )

        crew_results[crew] = {
            "n_timepoints": len(items),
            "feature_cols": feature_cols,
            "preflight_mean_displacement": pre_mag,
            "postflight_mean_displacement": post_mag,
            "displacement_delta": (post_mag - pre_mag) if post_mag is not None else None,
            "mean_Omega": float(np.mean(kin["Omega_t"])),
            "max_Omega": float(np.max(kin["Omega_t"])),
            "mean_a_perp": float(np.mean(kin["a_perp"])),
            "max_a_perp": float(np.max(kin["a_perp"])),
            "timepoints": tags,
            "days": days,
            "Omega_series": kin["Omega_t"].tolist(),
            "s_t_series": kin["s_t"].tolist(),
            "a_perp_series": kin["a_perp"].tolist(),
        }

        for i, tag in enumerate(tags):
            trajectory_rows.append(
                {
                    "dataset": "OSD-575_LSDS-8",
                    "session": crew,
                    "t_s": float(t_days[i]),
                    "timepoint": tag,
                    "s_t": float(kin["s_t"][i]),
                    "v_t": float(kin["v_t"][i]),
                    "a_t": float(kin["a_t"][i]),
                    "a_parallel": float(kin["a_parallel"][i]),
                    "a_perp": float(kin["a_perp"][i]),
                    "F_c": float(kin["F_c"][i]),
                    "Omega_t": float(kin["Omega_t"][i]),
                    "phase": phase_label(float(kin["Omega_t"][i])),
                }
            )

        # Lead-time relative to first postflight sample (R+1)
        ref = next((float(it["day"]) for it in items if it["tag"] == "R+1"), None)
        for thr, pname in (
            (PHASE_METASTABLE, "Metastable Thinning"),
            (PHASE_VETO, "Phase Veto"),
        ):
            tc, oc = first_crossing(kin["t"], kin["Omega_t"], thr)
            lead = (ref - tc) if (ref is not None and tc is not None) else None
            lead_hits.append(
                LeadTimeHit(
                    dataset="OSD-575_LSDS-8",
                    subject_or_session=crew,
                    phase=pname,
                    threshold=thr,
                    first_crossing_t_s=tc,
                    reference_event_t_s=ref,
                    lead_time_s=lead,
                    omega_at_crossing=oc,
                )
            )

    return {
        "meta": {
            "file": path.name,
            "n_rows": int(len(df)),
            "columns": list(df.columns),
            "feature_cols": feature_cols,
            "n_crew": len(crew_results),
        },
        "crew": crew_results,
        "trajectory_rows": trajectory_rows,
        "lead_hits": lead_hits,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_lead_time_table(hits: list[LeadTimeHit]) -> None:
    headers = (
        "Dataset",
        "Session",
        "Phase",
        "Thr",
        "Cross_t",
        "Ref_t",
        "Lead_s",
        "Omega",
    )
    rows = []
    for h in hits:
        rows.append(
            [
                h.dataset[:18],
                h.subject_or_session,
                h.phase,
                f"{h.threshold:.2f}",
                f"{h.first_crossing_t_s:.3f}" if h.first_crossing_t_s is not None else "—",
                f"{h.reference_event_t_s:.3f}" if h.reference_event_t_s is not None else "—",
                f"{h.lead_time_s:.3f}" if h.lead_time_s is not None else "—",
                f"{h.omega_at_crossing:.4f}" if h.omega_at_crossing is not None else "—",
            ]
        )
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)] if rows else [len(h) for h in headers]
    sep = "-+-".join("-" * w for w in widths)
    fmt = " | ".join(f"{{:{w}}}" for w in widths)
    print("\n=== Lead-Time Detections ===")
    print(fmt.format(*headers))
    print(sep)
    for r in rows:
        print(fmt.format(*r))
    print()


def resolve_csm_dir() -> Path:
    if CSM_DIR.is_dir() and any(CSM_DIR.glob("*.csv")):
        return CSM_DIR
    if CSM_FALLBACK.is_dir() and any(CSM_FALLBACK.glob("*.csv")):
        return CSM_FALLBACK
    raise FileNotFoundError(
        f"NASA CSM CSVs not found in {CSM_DIR} or {CSM_FALLBACK}"
    )


def first_sustained_crossing(
    t: np.ndarray,
    omega: np.ndarray,
    threshold: float,
    n_sustain: int = 5,
) -> tuple[float | None, float | None]:
    """First index where Omega stays at/above threshold for n_sustain samples."""
    above = np.asarray(omega, dtype=float) >= threshold
    if n_sustain <= 1:
        return first_crossing(t, omega, threshold)
    run = 0
    for i, flag in enumerate(above):
        run = run + 1 if flag else 0
        if run >= n_sustain:
            j = i - n_sustain + 1
            return float(t[j]), float(omega[j])
    return first_crossing(t, omega, threshold)


def _style_night_ops_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(COLOR_PANEL)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2A2A2A")
    ax.yaxis.label.set_color(COLOR_TEXT)
    ax.xaxis.label.set_color(COLOR_TEXT)
    ax.title.set_color(COLOR_TEXT)
    ax.grid(True, color="#2C2C2C", linewidth=0.6, alpha=0.85)


def generate_lead_time_figure(
    loft_kin: dict[str, np.ndarray],
    loft_event_t: float | None,
    lsds_crew: dict[str, dict[str, Any]],
    out_path: Path = LEAD_TIME_PLOT,
) -> dict[str, Any]:
    """
    Publication-grade two-panel Cupola / Night Ops validation figure.

    Panel 1 — NASA Langley 25_LOFT baseline-calibrated Omega_cal + lead-time shade.
    Panel 2 — OSD-575 Inspiration4 pre- vs post-flight composite Omega / displacement.
    """
    t = np.asarray(loft_kin["t"], dtype=float)
    if "Omega_cal" in loft_kin:
        omega = np.asarray(loft_kin["Omega_cal"], dtype=float)
        omega_label = r"$\Omega_{\mathrm{cal}}$ (25_LOFT, re-anchored)"
    else:
        omega = np.asarray(loft_kin["Omega_t"], dtype=float)
        omega_label = r"$\Omega_t$ (25_LOFT)"
    cross_t, _ = first_sustained_crossing(t, omega, PHASE_METASTABLE, n_sustain=5)
    if cross_t is None:
        cross_t, _ = first_crossing(t, omega, PHASE_METASTABLE)
    event_t = float(loft_event_t) if loft_event_t is not None else None
    lead_s = (event_t - cross_t) if (event_t is not None and cross_t is not None) else None
    lead_min = (lead_s / 60.0) if lead_s is not None else None

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11.0, 9.0),
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.32},
    )
    fig.patch.set_facecolor(COLOR_OBSIDIAN)

    # ---- Panel 1: 25_LOFT lead time ----
    _style_night_ops_axis(ax1)
    ax1.plot(t, omega, color=COLOR_SAGE, linewidth=1.35, label=omega_label)
    ax1.axhline(
        PHASE_METASTABLE,
        color=COLOR_AMBER,
        linestyle="--",
        linewidth=1.4,
        label=r"Metastable Thinning ($\Omega=0.07$)",
    )
    ax1.axhline(
        PHASE_VETO,
        color=COLOR_CORAL,
        linestyle="--",
        linewidth=1.4,
        label=r"Phase Veto ($\Omega=0.19$)",
    )
    # Nominal band guide
    ax1.axhspan(0.0, PHASE_METASTABLE, color="#2A3A32", alpha=0.35, zorder=0)

    if cross_t is not None and event_t is not None and lead_s is not None and lead_s > 0:
        ax1.axvspan(
            cross_t,
            event_t,
            color=COLOR_SHADE,
            alpha=0.18,
            zorder=0,
            label="Early-warning lead-time window",
        )
        ax1.axvline(cross_t, color=COLOR_AMBER, linewidth=1.0, alpha=0.9)
        ax1.axvline(event_t, color=COLOR_CORAL, linewidth=1.2, alpha=0.95)
        y_ann = max(float(np.nanpercentile(omega, 92)), PHASE_VETO * 1.8)
        mid = 0.5 * (cross_t + event_t)
        ax1.annotate(
            f"Lead time = {lead_s:.0f} s (~{lead_min:.1f} min)\n"
            f"metastable → Event",
            xy=(mid, y_ann),
            xytext=(mid, y_ann),
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLOR_AMBER,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=COLOR_PANEL,
                edgecolor=COLOR_AMBER,
                alpha=0.92,
            ),
        )
        ax1.text(
            event_t,
            float(np.nanmin(omega)) + 0.02 * (float(np.nanmax(omega)) - float(np.nanmin(omega))),
            "Event",
            color=COLOR_CORAL,
            fontsize=8,
            ha="left",
            va="bottom",
            rotation=90,
        )

    ax1.set_xlim(float(t[0]), float(t[-1]))
    ax1.set_ylim(bottom=0.0)
    ax1.set_xlabel("Time (TimeSecs)")
    ax1.set_ylabel(r"$\Omega_{\mathrm{cal}}$")
    ax1.set_title(
        "NASA Langley CSM 25_LOFT — Calibrated Autonomic Workload Lead-Time",
        fontsize=11,
        pad=8,
    )
    leg1 = ax1.legend(
        loc="upper right",
        fontsize=7.5,
        frameon=True,
        fancybox=False,
        edgecolor="#2A2A2A",
    )
    leg1.get_frame().set_facecolor(COLOR_PANEL)
    for txt in leg1.get_texts():
        txt.set_color(COLOR_TEXT)

    # ---- Panel 2: OSD-575 pre/post composite ----
    _style_night_ops_axis(ax2)
    crews = sorted(lsds_crew.keys())
    pre_omega = []
    post_omega = []
    pre_disp = []
    post_disp = []
    for crew in crews:
        cr = lsds_crew[crew]
        tags = cr["timepoints"]
        om = np.asarray(cr["Omega_series"], dtype=float)
        st = np.asarray(cr["s_t_series"], dtype=float)
        pre_mask = np.array([str(tag).startswith("L-") for tag in tags])
        post_mask = np.array([str(tag).startswith("R+") for tag in tags])
        pre_omega.append(float(np.mean(om[pre_mask])) if pre_mask.any() else np.nan)
        post_omega.append(float(np.mean(om[post_mask])) if post_mask.any() else np.nan)
        pre_disp.append(float(cr["preflight_mean_displacement"]))
        post_disp.append(
            float(cr["postflight_mean_displacement"])
            if cr["postflight_mean_displacement"] is not None
            else np.nan
        )

    x = np.arange(len(crews), dtype=float)
    width = 0.18
    ax2.bar(
        x - 1.5 * width,
        pre_omega,
        width,
        color=COLOR_PRE,
        edgecolor="#2A2A2A",
        label=r"Pre-flight mean $\Omega_t$ ($L$-*)",
    )
    ax2.bar(
        x - 0.5 * width,
        post_omega,
        width,
        color=COLOR_POST,
        edgecolor="#2A2A2A",
        label=r"Post-flight mean $\Omega_t$ ($R$+*)",
    )
    ax2.bar(
        x + 0.5 * width,
        pre_disp,
        width,
        color=COLOR_PRE,
        alpha=0.45,
        hatch="//",
        edgecolor="#2A2A2A",
        label=r"Pre-flight displacement $s_t$",
    )
    ax2.bar(
        x + 1.5 * width,
        post_disp,
        width,
        color=COLOR_POST,
        alpha=0.45,
        hatch="//",
        edgecolor="#2A2A2A",
        label=r"Post-flight displacement $s_t$",
    )
    ax2.axhline(
        PHASE_METASTABLE,
        color=COLOR_AMBER,
        linestyle="--",
        linewidth=1.4,
        label=r"$\Omega=0.07$",
    )
    ax2.axhline(
        PHASE_VETO,
        color=COLOR_CORAL,
        linestyle="--",
        linewidth=1.4,
        label=r"$\Omega=0.19$",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(crews)
    ax2.set_xlabel("Inspiration4 crew subject")
    ax2.set_ylabel(r"Composite $\Omega_t$ / biomarker displacement $s_t$")
    ax2.set_title(
        "NASA OSDR OSD-575 Inspiration4 — Pre- vs Post-Flight Phase Shift",
        fontsize=11,
        pad=8,
    )
    ax2.set_ylim(bottom=0.0)
    leg2 = ax2.legend(
        loc="upper right",
        fontsize=7.0,
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#2A2A2A",
    )
    leg2.get_frame().set_facecolor(COLOR_PANEL)
    for txt in leg2.get_texts():
        txt.set_color(COLOR_TEXT)

    fig.suptitle(
        "Sentinel Settle — NASA Validation Proof (Cupola / Night Ops)",
        color=COLOR_TEXT,
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        dpi=300,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.25,
    )
    plt.close(fig)

    return {
        "path": str(out_path.relative_to(ROOT)).replace("\\", "/"),
        "loft_metastable_crossing_t_s": cross_t,
        "loft_event_t_s": event_t,
        "loft_lead_time_s": lead_s,
        "loft_lead_time_min": lead_min,
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csm_dir = resolve_csm_dir()

    print("NASA Omega Validator")
    print(f"  CSM dir : {csm_dir}")
    print(f"  LSDS    : {LSDS_PATH}")
    print(f"  Results : {RESULTS_DIR}")

    # ---- Inspect / run Langley CSM ----
    csm_files = sorted(csm_dir.glob("*.csv"))
    if not csm_files:
        print("ERROR: no CSM CSV files found", file=sys.stderr)
        return 1

    print("\n--- CSM column / rate inspection ---")
    csm_sessions = {}
    trajectory_rows: list[dict[str, Any]] = []
    lead_hits: list[LeadTimeHit] = []
    loft_kin: dict[str, np.ndarray] | None = None
    loft_event_t: float | None = None
    # Per-session payloads for dual-horizon / method comparison pooling
    method_session_rows: list[dict[str, Any]] = []
    gasket_seconds_list: list[float] = []
    total_events_count = 0
    total_duration_s = 0.0
    loso_packs: list[dict[str, Any]] = []

    for path in csm_files:
        cols = list(pd.read_csv(path, nrows=0).columns)
        tpeek = pd.read_csv(path, usecols=["TimeSecs"], nrows=5000)["TimeSecs"].to_numpy()
        fs = estimate_sampling_rate(tpeek)
        print(f"  {path.name}: cols={len(cols)} fs≈{fs:.3f} Hz  headers={cols}")
        print(f"  Processing {path.name} ...", flush=True)
        session = ingest_csm_csv(path)
        meta = session["meta"]
        kin = session["kin"]
        sid = path.stem  # 25_CA, 25_DA, ...
        duration_s = float(meta["duration_s"])
        event_segments = session["event_segments"]
        ibi = np.asarray(session["ibi"], dtype=float)

        mean_ibi_s = meta.get("mean_ibi_s")
        gasket_mean_seconds = (
            float(GASKET_SUSTAIN_BEATS * mean_ibi_s) if mean_ibi_s else None
        )
        if gasket_mean_seconds is not None:
            gasket_seconds_list.append(gasket_mean_seconds)

        # Baseline re-anchor (mobile-app aligned)
        omega_cal, omega_offset = reanchor_omega(kin["Omega_t"])
        kin["Omega_cal"] = omega_cal
        kin["omega_offset"] = np.array([omega_offset])

        hr_rmssd = build_hr_rmssd_alerts(ibi)
        detectors = {
            "omega_cal_metastable": omega_cal >= PHASE_METASTABLE,
            "omega_cal_veto": omega_cal >= PHASE_VETO,
            "rmssd_drop": hr_rmssd["rmssd_alert"],
            "hr_elevation": hr_rmssd["hr_alert"],
        }

        session_method_scores: dict[str, Any] = {}
        for horizon_name, horizon_s in (
            ("tactical_180s", TACTICAL_HORIZON_S),
            ("mission_900s", PRE_EVENT_HORIZON_S),
        ):
            session_method_scores[horizon_name] = {}
            for det_name, alert in detectors.items():
                session_method_scores[horizon_name][det_name] = score_alert_series(
                    kin["t"],
                    alert,
                    event_segments,
                    duration_s,
                    horizon_s=horizon_s,
                    gasket_beats=GASKET_SUSTAIN_BEATS,
                )

        method_session_rows.append(
            {
                "session": sid,
                "duration_s": duration_s,
                "n_event_segments": len(event_segments),
                "mean_ibi_s": mean_ibi_s,
                "gasket_mean_seconds": gasket_mean_seconds,
                "omega_offset": omega_offset,
                "baseline_hr_bpm": hr_rmssd["baseline_hr_bpm"],
                "baseline_rmssd_s": hr_rmssd["baseline_rmssd_s"],
                "scores": session_method_scores,
            }
        )
        loso_packs.append(
            {
                "session": sid,
                "t": kin["t"],
                "omega_cal": omega_cal,
                "event_segments": event_segments,
                "duration_s": duration_s,
                "omega_offset": omega_offset,
                "mean_ibi_s": mean_ibi_s,
            }
        )
        total_events_count += len(event_segments)
        total_duration_s += duration_s

        # Legacy-compatible dual-horizon Omega_cal FA block for README tables
        thr_scores_180 = {
            "metastable_thinning": session_method_scores["tactical_180s"][
                "omega_cal_metastable"
            ],
            "phase_veto": session_method_scores["tactical_180s"]["omega_cal_veto"],
        }
        thr_scores_900 = {
            "metastable_thinning": session_method_scores["mission_900s"][
                "omega_cal_metastable"
            ],
            "phase_veto": session_method_scores["mission_900s"]["omega_cal_veto"],
        }

        csm_sessions[sid] = {
            **meta,
            "omega_offset": omega_offset,
            "mean_Omega": float(np.mean(kin["Omega_t"])),
            "mean_Omega_cal": float(np.mean(omega_cal)),
            "max_Omega_cal": float(np.max(omega_cal)),
            "pct_metastable_cal": float(np.mean(omega_cal >= PHASE_METASTABLE) * 100.0),
            "pct_veto_cal": float(np.mean(omega_cal >= PHASE_VETO) * 100.0),
            "mean_a_perp": float(np.mean(kin["a_perp"])),
            "max_a_perp": float(np.max(kin["a_perp"])),
            "mean_F_c": float(np.mean(kin["F_c"])),
            "baseline_hr_bpm": hr_rmssd["baseline_hr_bpm"],
            "baseline_rmssd_s": hr_rmssd["baseline_rmssd_s"],
            "false_alarm_metrics_tactical_180s": thr_scores_180,
            "false_alarm_metrics_mission_900s": thr_scores_900,
            "gasket_mean_seconds": gasket_mean_seconds,
        }

        # Decimate trajectory for CSV summary (keep up to ~500 points)
        n = len(kin["Omega_t"])
        step = max(1, n // 500)
        for i in range(0, n, step):
            trajectory_rows.append(
                {
                    "dataset": "NASA_Langley_CSM25",
                    "session": sid,
                    "t_s": float(kin["t"][i]),
                    "timepoint": "",
                    "s_t": float(kin["s_t"][i]),
                    "v_t": float(kin["v_t"][i]),
                    "a_t": float(kin["a_t"][i]),
                    "a_parallel": float(kin["a_parallel"][i]),
                    "a_perp": float(kin["a_perp"][i]),
                    "F_c": float(kin["F_c"][i]),
                    "Omega_t": float(kin["Omega_t"][i]),
                    "Omega_cal": float(omega_cal[i]),
                    "phase": phase_label(float(omega_cal[i])),
                }
            )

        ref = meta.get("reference_event_t_s")
        for thr, pname in (
            (PHASE_METASTABLE, "Metastable Thinning"),
            (PHASE_VETO, "Phase Veto"),
        ):
            tc, oc = first_crossing(kin["t"], omega_cal, thr)
            lead = (ref - tc) if (ref is not None and tc is not None) else None
            lead_hits.append(
                LeadTimeHit(
                    dataset="NASA_Langley_CSM25",
                    subject_or_session=sid,
                    phase=pname,
                    threshold=thr,
                    first_crossing_t_s=tc,
                    reference_event_t_s=ref,
                    lead_time_s=lead,
                    omega_at_crossing=oc,
                )
            )

        fa180 = thr_scores_180["metastable_thinning"]["sustained_gasket_filtered"]
        print(
            f"    IBI n={meta['n_ibi']} mean_HR={meta['mean_hr_bpm']:.1f} bpm  "
            f"Ω_cal̄={np.mean(omega_cal):.4f} offset={omega_offset:.4f}  "
            f"gasket_FA/h@180s(0.07)={fa180['false_alarms_per_hour']:.2f}"
        )

        if sid == "25_LOFT":
            loft_kin = kin
            loft_event_t = ref

    method_comparison = pool_method_comparison(
        method_session_rows,
        total_duration_s=total_duration_s,
        total_events=total_events_count,
        gasket_seconds_list=gasket_seconds_list,
    )
    fa_report = method_comparison["omega_false_alarm_analysis"]
    print("\n--- Cross-scenario LOSO hold-out ---", flush=True)
    loso_report = run_loso_holdout(loso_packs, horizon_s=TACTICAL_HORIZON_S)

    # ---- Inspiration4 LSDS-8 ----
    print("\n--- Inspiration4 OSD-575 LSDS-8 ---")
    if not LSDS_PATH.is_file():
        print(f"ERROR: missing {LSDS_PATH}", file=sys.stderr)
        return 1
    lsds = ingest_lsds(LSDS_PATH)
    print(
        f"  rows={lsds['meta']['n_rows']} crew={lsds['meta']['n_crew']} "
        f"features={len(lsds['meta']['feature_cols'])}"
    )
    for crew, cr in lsds["crew"].items():
        print(
            f"  {crew}: pre_disp={cr['preflight_mean_displacement']:.3f} "
            f"post_disp={cr['postflight_mean_displacement']:.3f} "
            f"Δ={cr['displacement_delta']:.3f} "
            f"Ω̄={cr['mean_Omega']:.4f} Ωmax={cr['max_Omega']:.4f}"
        )
    trajectory_rows.extend(lsds["trajectory_rows"])
    lead_hits.extend(lsds["lead_hits"])
    osd575_cohort = compute_osd575_cohort_metrics(lsds["crew"])

    # ---- Persist ----
    traj_df = pd.DataFrame(trajectory_rows)
    traj_df.to_csv(TRAJECTORY_CSV, index=False)

    plot_meta: dict[str, Any] | None = None
    if loft_kin is not None:
        print("\n--- Generating validation figure ---", flush=True)
        plot_meta = generate_lead_time_figure(
            loft_kin=loft_kin,
            loft_event_t=loft_event_t,
            lsds_crew=lsds["crew"],
            out_path=LEAD_TIME_PLOT,
        )
        print(
            f"  25_LOFT lead time = {plot_meta['loft_lead_time_s']:.1f} s "
            f"(~{plot_meta['loft_lead_time_min']:.1f} min)"
        )
        print(f"  Wrote {LEAD_TIME_PLOT}")
    else:
        print("WARNING: 25_LOFT kinematics unavailable; skipping figure", file=sys.stderr)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine": {
            "name": "Scale-Invariant Kinematic Engine",
            "epsilon": EPS,
            "K_ring": K_RING,
            "delta_tau_clamp": [DELTA_TAU_LO, DELTA_TAU_HI],
            "omega_formula": "0.05 + 0.12*s_t + 0.08*(|a_perp|**2) + 0.02*F_c",
            "phase_boundaries": {
                "metastable_thinning": PHASE_METASTABLE,
                "phase_veto": PHASE_VETO,
            },
        },
        "datasets": {
            "nasa_langley_csm25": {
                "path": str(csm_dir.relative_to(ROOT)).replace("\\", "/"),
                "sessions": csm_sessions,
            },
            "nasa_osdr_osd575_lsds8": {
                "path": str(LSDS_PATH.relative_to(ROOT)).replace("\\", "/"),
                "meta": lsds["meta"],
                "crew": {
                    k: {kk: vv for kk, vv in v.items() if not kk.endswith("_series")}
                    for k, v in lsds["crew"].items()
                },
                "crew_series": {
                    k: {
                        "timepoints": v["timepoints"],
                        "days": v["days"],
                        "Omega_series": v["Omega_series"],
                        "s_t_series": v["s_t_series"],
                        "a_perp_series": v["a_perp_series"],
                    }
                    for k, v in lsds["crew"].items()
                },
            },
        },
        "lead_time_detections": [asdict(h) for h in lead_hits],
        "false_alarm_analysis": fa_report,
        "method_comparison": method_comparison["method_comparison"],
        "detector_baselines": method_comparison["detector_baselines"],
        "loso_holdout_validation": loso_report,
        "osd575_cohort_metrics": osd575_cohort,
        "figure": plot_meta,
        "artifacts": {
            "validation_report_json": str(REPORT_JSON.relative_to(ROOT)).replace("\\", "/"),
            "omega_trajectory_summary_csv": str(TRAJECTORY_CSV.relative_to(ROOT)).replace("\\", "/"),
            "omega_lead_time_plot_png": str(LEAD_TIME_PLOT.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_lead_time_table(lead_hits)
    print_false_alarm_table(fa_report, cohort=osd575_cohort)
    print_method_comparison_table(method_comparison["method_comparison"])
    print_loso_table(loso_report)
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {TRAJECTORY_CSV} ({len(traj_df)} rows)")
    if plot_meta is not None:
        print(f"Wrote {LEAD_TIME_PLOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
