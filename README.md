# Sentinel Settle — NASA Space Apps 2026 Validation

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](requirements.txt)
[![Challenge](https://img.shields.io/badge/NASA_Space_Apps-2026-0B3D91.svg)](https://www.spaceappschallenge.org/)
[![Demo 30s](https://img.shields.io/badge/Demo-30s_Galaxy_S22-FF0000.svg)](https://youtu.be/YOUR_VIDEO_ID)

**Operational claim:** Baseline-calibrated $\Omega_{\mathrm{cal}}$ is an autonomic workload / cognitive-overload early-warning signal that, on NASA Langley `25_LOFT`, gives **678 s (~11.3 min)** lead time to a labeled workload `Event` under a Tier-1 ~4 s gasket — supporting NASA’s *Risk of Inadequate Systems Operations Support* and *Risk of Behavioral Changes and Psychiatric Disorders*.

**30-second Galaxy S22 demo:** [https://youtu.be/YOUR_VIDEO_ID](https://youtu.be/YOUR_VIDEO_ID) *(replace with final YouTube ID)*

![Sentinel Settle NASA Validation Proof](results/omega_lead_time_plot.png)

*Figure — Cupola / Night Ops validation proof: (top) NASA Langley `25_LOFT` **baseline-calibrated** $\Omega_{\mathrm{cal}}$ (Nominal band shaded) with **678 s (~11.3 min)** early-warning to the workload Event; (bottom) OSD-575 Inspiration4 multi-biomarker displacement (**supporting** serum evidence — not continuous ECG).*

## Validation Scope, False-Alarm Analysis & Limitations

### Proof-of-Concept Sample Size

Validated on 1.31 hours of Langley flight-crew telemetry (Subject 25 across CA, DA, SS, and LOFT scenarios) plus N = 4 Inspiration4 orbital crew profiles (OSD-575). Requires longer, prospective validation across more crew and mission conditions.

Session $\Omega$ is **re-anchored** to mirror the mobile app’s “Re-anchor Baseline” (`omega_offset = median(Omega[:30]) - 0.05`; $\Omega_{\mathrm{cal}} = \mathrm{clip}(\Omega - \mathrm{offset}, 0, \infty)$) so trajectories begin in Nominal (&lt; 0.07) rather than sitting permanently elevated.

### Cross-Scenario Hold-Out (LOSO)

4-fold **Leave-One-Scenario-Out** across `CA`, `DA`, `LOFT`, `SS`: Youden $\tau_{\mathrm{metastable}}$ / $\tau_{\mathrm{veto}}$ are fit on the **3 training scenarios only** (constrained ROC bands); the held-out scenario is scored out-of-sample with the Tier-1 **5-beat** gasket and strict **180 s** horizon. Fixed theoretical **0.07 / 0.19** boundaries are reported alongside. Per-session $\Omega_{\mathrm{cal}}$ re-anchor uses only that session’s first 30 beats (no Event-label leak).

| Hold-out | Scheme | τ_meta / τ_veto | Prec (meta) | Event Recall | FA/h | Lead time (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| **LOFT** | Fixed 0.07 / 0.19 | 0.070 / 0.190 | **0.405** | **5 / 6 (0.833)** | **24.84** | **677.7** |
| **LOFT** | Youden LOSO | 0.166 / 0.217 | 0.417 | 6 / 6 (1.000) | 34.78 | 590.1 |
| CA | Fixed 0.07 / 0.19 | 0.070 / 0.190 | 1.000 | 1 / 1 | 0.00 | −18.0 |
| CA | Youden LOSO | 0.165 / 0.267 | 1.000 | 1 / 1 | 0.00 | −54.7 |
| DA | Fixed 0.07 / 0.19 | 0.070 / 0.190 | 1.000 | 10 / 10 | 0.00 | 18.0 |
| DA | Youden LOSO | 0.181 / 0.493 | 1.000 | 10 / 10 | 0.00 | 7.2 |
| SS | Fixed 0.07 / 0.19 | 0.070 / 0.190 | 0.688 | 2 / 2 | 50.02 | 62.6 |
| SS | Youden LOSO | 0.180 / 0.259 | 1.000 | 2 / 2 | 0.00 | 20.3 |
| **POOLED** | Fixed 0.07 / 0.19 | — | **0.639** | **18 / 19 (0.947)** | **22.97** | mean 185.1 |
| **POOLED** | Youden LOSO | — | 0.588 | 19 / 19 (1.000) | 26.79 | mean 140.7 |

**Read:** Holding out `LOFT` (the hardest long scenario), fixed $\Omega_{\mathrm{cal}}$ ≥ 0.07 still recovers **5/6 Events** with **677.7 s** lead time — evidence that the signal generalizes to an unseen workload regime rather than memorizing in-sample labels. Youden-fit thresholds raise LOFT recall to **6/6** at the cost of higher FA/h (**34.78**).

### Method Comparison (Strict 180 s Tactical Horizon)

Head-to-head on the **same 19** Langley `Event` episodes, **Tier-1 5-beat sustained gasket** (~**3.99 s**; range **3.56–4.65 s**). Primary claim uses a strict **180-second (3-minute)** pre-event TP horizon (900 s mission horizon retained in `validation_report.json` for comparison only).

Exact values from `results/validation_report.json` (`python src/nasa_omega_validator.py`):

| Method | Alert Precision | Event Recall (TP / 19) | False Alarms / Hour |
| --- | ---: | ---: | ---: |
| **Sentinel Settle $\Omega_{\mathrm{cal}}$ ≥ 0.07** | **0.639** | **18 / 19 (0.947)** | **22.97** |
| **Sentinel Settle $\Omega_{\mathrm{cal}}$ ≥ 0.19** | **0.587** | **19 / 19 (1.000)** | **19.90** |
| Rolling RMSSD drop (&lt; 0.75 × baseline, 30-beat) | 0.625 | 8 / 19 (0.421) | 4.59 |
| Heart-rate elevation (+10 bpm vs baseline) | 0.761 | 13 / 19 (0.684) | 8.42 |

**Read:** Under the strict 180 s horizon, $\Omega_{\mathrm{cal}}$ **out-recalls** standard RMSSD-drop and HR-elevation detectors on labeled workload Events (0.947–1.000 vs 0.421–0.684) at the cost of higher FA/h than the simpler baselines. That is the operational trade space: earlier autonomic/workload sensitivity vs. quieter HR/RMSSD triggers. Raw beat-level $\Omega_{\mathrm{cal}}$ remains noisy (Metastable **1695.68 FA/h**); the Tier-1 gasket is what makes the method usable.

### Two-Tier Operational Gasket Architecture

| Tier | Role | Window | Effect on this archive |
| --- | --- | --- | --- |
| **Tier 1 — Tactical / Acute Workload Gasket** | Filters single-beat R-peak jitter for cockpit/EVA-scale events (`DA`, `SS`, `CA`, `LOFT`) | **5 consecutive beats** / **~4 s** | Collapses raw Metastable noise toward **22.97 FA/h** while keeping Event recall **18/19** |
| **Tier 2 — Habitat / Long-Duration Gasket** | Suppresses alarm fatigue during multi-week orbital drift | **15 minutes** | Implemented in the mobile prototype for habitat monitoring |

### Crew Human-Factors & Alarm Philosophy

| Phase | Crew / ops action |
| --- | --- |
| **Nominal ($\Omega &lt; 0.07$)** | Silent monitoring; **zero crew interruption**. |
| **Metastable Thinning ($0.07 \le \Omega &lt; 0.19$, sustained ≥ Tier-1 ~4 s tactical / Tier-2 15 min habitat)** | Low-arousal cue; crew may initiate eyes-closed haptic pacing (**Breath Sanctuary**), **designed for closed-loop haptic pacing** to support autonomic recovery. Prospective closed-loop $\Omega$ reduction trials are the next milestone — **not** claimed here. |
| **Phase Veto ($\Omega \ge 0.19$)** | High-priority workload redline; pause non-critical checklist items and queue a **zero-PII Flight Surgeon JSON bundle** for downlink. |

### OSD-575 Supporting Biomarker Evidence (N = 4)

`OSD-575` is framed as **supporting multi-biomarker displacement evidence** after flight — **not** continuous cardiac monitoring. It consists of discrete blood-draw timepoints:

| Threshold | R+1 exceedances | Pre-flight below threshold | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: |
| Metastable Thinning (Ω ≥ 0.07) | 4 / 4 | 0 / 4 | 1.000 | 0.000 |
| Phase Veto (Ω ≥ 0.19) | 4 / 4 | 4 / 4 | 1.000 | 1.000 |

### Closed-Loop Haptic Intervention Status

Breath Sanctuary is **designed for closed-loop haptic pacing** to support autonomic recovery. The app logs session coherence and heart-rate settling; controlled prospective trials measuring closed-loop $\Omega_t$ reduction remain the next validation milestone and are **not** claimed by this repository.

## Prospective Validation Roadmap ("What's Next")

- **Target flight analogs:** Prospective multi-crew testing in **NASA HERA (Human Exploration Research Analog)** isolation campaigns and **6° Head-Down Tilt Bed Rest (HDBR)** microgravity deconditioning studies.
- **Primary closed-loop endpoint:** Statistically significant reduction in $\Omega_{\mathrm{cal}}$ magnitude and faster recovery slope ($\Delta\Omega / \Delta t$) during **Breath Sanctuary** closed-loop haptic pacing versus unguided passive rest.
- **Edge hardware qualification:** On-device latency and battery profiling across radiation-tolerant wearable BLE hardware for Artemis EVA/IVA operations.

## What this repository validates

`src/nasa_omega_validator.py` ingests two NASA datasets and evaluates calibrated Omega against workload `Event` labels, with HR/RMSSD benchmarks:

| Dataset | Path | Role |
| --- | --- | --- |
| NASA Langley Crew State Monitoring (`25.zip`) | `data/nasa_raw/nasa_csm_25/` | 256 Hz ECG / EEG / GSR flight-crew sessions (`25_CA`, `25_DA`, `25_SS`, `25_LOFT`) — primary continuous validation |
| NASA OSDR Inspiration4 OSD-575 | `data/nasa_raw/LSDS-8_Multiplex_serum_cardiovascular_EvePanel_TRANSFORMED.csv` | Supporting pre-/post-flight multiplex serum panel (EvePanel / LSDS-8) |

### Engine (summary)

1. Build dynamic state `s_t` from IBI (seconds) or multi-biomarker displacement vectors.
2. Bilaterally clamp: `delta_tau = clip(s_t, 0.01, 5.0)`.
3. Non-uniform finite-difference velocity `v_t` and acceleration `a_t`.
4. Decompose into `a_parallel` / `a_perp` with `epsilon = 1e-7`.
5. Integrate `|a_perp|` over rolling `K = 30` → `F_c`.
6. Evaluate `Omega_t = 0.05 + 0.12*s_t + 0.08*(|a_perp|**2) + 0.02*F_c`, then **re-anchor** to $\Omega_{\mathrm{cal}}$.
7. Score alerts under **180 s** (primary) and **900 s** (mission) horizons; benchmark vs RMSSD drop and HR+10 bpm.

Empirical lead-time proof: Langley **`25_LOFT` 678-second (~11.3-minute)** calibrated metastable-to-Event detection.

## Challenge resources cited

- **Risk of Inadequate Systems Operations Support** — ops-support risk during mission systems work; addressed here as autonomic / cognitive-overload early warning.
- **Risk of Behavioral Changes and Psychiatric Disorders** — behavioral-health risk pathways relevant to continuous state monitoring and low-arousal recovery cues on long-duration missions.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt

# Place unpacked Langley CSVs in data/nasa_raw/nasa_csm_25/
# and the LSDS-8 CSV under data/nasa_raw/
python src/nasa_omega_validator.py
```

### Outputs

| Artifact | Description |
| --- | --- |
| `results/validation_report.json` | Calibrated Ω metrics, 180/900 s horizons, method comparison vs RMSSD/HR, **LOSO hold-out**, OSD-575 cohort stats |
| `results/omega_trajectory_summary.csv` | Decimated $\Omega$ / $\Omega_{\mathrm{cal}}$ time series |
| `results/omega_lead_time_plot.png` | 300 DPI Cupola / Night Ops figure (calibrated LOFT lead time + OSD-575 shift) |
| Terminal tables | Lead-time, precision/recall, method comparison |

## Data notes

- Official Langley archive is distributed as `25.zip`. Unpacked session CSVs (`25_CA.csv`, `25_DA.csv`, `25_SS.csv`, `25_LOFT.csv`) are expected under `data/nasa_raw/nasa_csm_25/`. Zip archives under `data/nasa_raw/` are gitignored.
- Column headers observed: `TimeSecs`, 20 EEG channels, `ECG`, `R`, `GSR`, `Event`. Measured sampling rate ≈ **256 Hz**.
- OSD-575 LSDS-8 columns: `Sample Name` plus concentration / percent fields for a2-macroglobulin, AGP, CRP, fetuin-A, fibrinogen, haptoglobin, L-selectin, PF4, and SAP.

## Prior Work & Attribution

- **Sentinel Settle** Flutter/BLE mobile edge app (pre-existing), including Breath Sanctuary **designed for closed-loop haptic pacing**: [https://github.com/Regsiren/sentinel-jolt-app](https://github.com/Regsiren/sentinel-jolt-app)
- Zenodo preprints: [10.5281/zenodo.20549045](https://doi.org/10.5281/zenodo.20549045), [10.5281/zenodo.20548468](https://doi.org/10.5281/zenodo.20548468)
- The underlying kinematic governance method is the subject of a pre-existing UKIPO patent application filed by Thohat Ventures Ltd. This repository publishes an independent NASA-dataset validation implementation under Apache-2.0; it does not expand, waive, or transfer those patent rights.

## License

Copyright 2026 Thohat Ventures Ltd  
Licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE).
