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
PRE_EVENT_HORIZON_S = 900.0  # early-warning TP window before Event onset
GASKET_SUSTAIN_BEATS = 5  # sustained / gasket-filtered alert length


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


def score_threshold_alerts(
    sample_t: np.ndarray,
    omega: np.ndarray,
    threshold: float,
    event_segments: list[tuple[float, float]],
    duration_s: float,
    horizon_s: float = PRE_EVENT_HORIZON_S,
    gasket_beats: int = GASKET_SUSTAIN_BEATS,
) -> dict[str, Any]:
    """
    Classify Omega excursions vs Event ground truth.

    Raw beat-level: every sample with Omega >= threshold.
    Sustained / gasket-filtered: contiguous runs lasting >= gasket_beats.
    """
    sample_t = np.asarray(sample_t, dtype=float)
    omega = np.asarray(omega, dtype=float)
    above = omega >= threshold
    in_tp = tp_window_mask(sample_t, event_segments, horizon_s)
    hours = max(duration_s, EPS) / 3600.0

    def _pack(tp: int, fp: int) -> dict[str, float | int]:
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

    return {
        "threshold": float(threshold),
        "raw_beat_level": raw,
        "raw_excursion_onsets": raw_excursion,
        "sustained_gasket_filtered": sustained,
        "n_raw_beats_above": int(np.sum(above)),
        "n_raw_excursion_onsets": int(len(raw_onset_idx)),
        "n_sustained_alerts": int(len(sus_onset_idx)),
    }


def aggregate_false_alarm_metrics(
    session_scores: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Pool TP/FP across Langley sessions for each threshold and filter mode."""
    modes = ("raw_beat_level", "raw_excursion_onsets", "sustained_gasket_filtered")
    thresholds = {
        "metastable_thinning": PHASE_METASTABLE,
        "phase_veto": PHASE_VETO,
    }
    total_duration_s = sum(
        float(s.get("duration_s", 0.0)) for s in session_scores.values()
    )
    hours = max(total_duration_s, EPS) / 3600.0
    pooled: dict[str, Any] = {
        "pre_event_horizon_s": PRE_EVENT_HORIZON_S,
        "gasket_sustain_beats": GASKET_SUSTAIN_BEATS,
        "total_duration_s": float(total_duration_s),
        "total_duration_h": float(hours),
        "by_threshold": {},
        "sessions": session_scores,
    }
    for thr_name, thr in thresholds.items():
        by_mode: dict[str, Any] = {}
        for mode in modes:
            tp = fp = 0
            for sess in session_scores.values():
                block = sess["thresholds"][thr_name][mode]
                tp += int(block["true_positives"])
                fp += int(block["false_positives"])
            denom = tp + fp
            by_mode[mode] = {
                "true_positives": tp,
                "false_positives": fp,
                "precision": (float(tp / denom) if denom > 0 else None),
                "false_alarms_per_hour": float(fp / hours),
            }
        pooled["by_threshold"][thr_name] = {
            "threshold": thr,
            **by_mode,
        }
    return pooled


def print_false_alarm_table(fa_report: dict[str, Any]) -> None:
    """Terminal summary of TP / FP / precision / FA-per-hour."""
    headers = (
        "Threshold",
        "Mode",
        "TP",
        "FP",
        "Precision",
        "FA/hour",
    )
    rows: list[list[str]] = []
    labels = {
        "raw_beat_level": "Raw beat-level",
        "raw_excursion_onsets": "Raw excursion",
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
            rows.append(
                [
                    thr_labels[thr_key],
                    label,
                    str(m["true_positives"]),
                    str(m["false_positives"]),
                    f"{prec:.3f}" if prec is not None else "—",
                    f"{m['false_alarms_per_hour']:.2f}",
                ]
            )
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)
    ]
    sep = "-+-".join("-" * w for w in widths)
    fmt = " | ".join(f"{{:{w}}}" for w in widths)
    print("\n=== False-Alarm & Precision (Event ground truth, 900 s pre-event horizon) ===")
    print(
        f"  Pooled Langley duration = {fa_report['total_duration_s']:.1f} s "
        f"({fa_report['total_duration_h']:.3f} h); "
        f"gasket sustain = {fa_report['gasket_sustain_beats']} beats"
    )
    print(fmt.format(*headers))
    print(sep)
    for r in rows:
        print(fmt.format(*r))
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

    Panel 1 — NASA Langley 25_LOFT Omega trajectory + early-warning shade.
    Panel 2 — OSD-575 Inspiration4 pre- vs post-flight composite Omega / displacement.
    """
    t = np.asarray(loft_kin["t"], dtype=float)
    omega = np.asarray(loft_kin["Omega_t"], dtype=float)
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
    ax1.plot(t, omega, color=COLOR_SAGE, linewidth=1.35, label=r"$\Omega_t$ (25_LOFT)")
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
    ax1.set_ylabel(r"$\Omega_t$")
    ax1.set_title(
        "NASA Langley CSM 25_LOFT — Flight-Crew ECG Lead-Time Detection",
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
    fa_session_scores: dict[str, dict[str, Any]] = {}

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

        # Event-grounded false-alarm / precision scoring
        thr_scores = {
            "metastable_thinning": score_threshold_alerts(
                kin["t"],
                kin["Omega_t"],
                PHASE_METASTABLE,
                event_segments,
                duration_s,
            ),
            "phase_veto": score_threshold_alerts(
                kin["t"],
                kin["Omega_t"],
                PHASE_VETO,
                event_segments,
                duration_s,
            ),
        }
        fa_session_scores[sid] = {
            "duration_s": duration_s,
            "n_event_segments": len(event_segments),
            "thresholds": thr_scores,
        }

        csm_sessions[sid] = {
            **meta,
            "mean_Omega": float(np.mean(kin["Omega_t"])),
            "max_Omega": float(np.max(kin["Omega_t"])),
            "mean_a_perp": float(np.mean(kin["a_perp"])),
            "max_a_perp": float(np.max(kin["a_perp"])),
            "mean_F_c": float(np.mean(kin["F_c"])),
            "pct_metastable": float(np.mean(kin["Omega_t"] >= PHASE_METASTABLE) * 100.0),
            "pct_veto": float(np.mean(kin["Omega_t"] >= PHASE_VETO) * 100.0),
            "false_alarm_metrics": thr_scores,
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
                    "phase": phase_label(float(kin["Omega_t"][i])),
                }
            )

        ref = meta.get("reference_event_t_s")
        for thr, pname in (
            (PHASE_METASTABLE, "Metastable Thinning"),
            (PHASE_VETO, "Phase Veto"),
        ):
            tc, oc = first_crossing(kin["t"], kin["Omega_t"], thr)
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

        meta_fa = thr_scores["metastable_thinning"]["sustained_gasket_filtered"]
        print(
            f"    IBI n={meta['n_ibi']} mean_HR={meta['mean_hr_bpm']:.1f} bpm  "
            f"Ω̄={csm_sessions[sid]['mean_Omega']:.4f} Ωmax={csm_sessions[sid]['max_Omega']:.4f}  "
            f"sustained_FA/h(0.07)={meta_fa['false_alarms_per_hour']:.2f}"
        )

        if sid == "25_LOFT":
            loft_kin = kin
            loft_event_t = ref

    fa_report = aggregate_false_alarm_metrics(fa_session_scores)

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
        "figure": plot_meta,
        "artifacts": {
            "validation_report_json": str(REPORT_JSON.relative_to(ROOT)).replace("\\", "/"),
            "omega_trajectory_summary_csv": str(TRAJECTORY_CSV.relative_to(ROOT)).replace("\\", "/"),
            "omega_lead_time_plot_png": str(LEAD_TIME_PLOT.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_lead_time_table(lead_hits)
    print_false_alarm_table(fa_report)
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {TRAJECTORY_CSV} ({len(traj_df)} rows)")
    if plot_meta is not None:
        print(f"Wrote {LEAD_TIME_PLOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
