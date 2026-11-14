# Sentinel Settle — NASA Space Apps 2026 Validation

Health monitoring software for astronauts on space missions: open validation of an edge biophysical **Scale-Invariant Kinematic Engine** against NASA flight-crew physiology and Inspiration4 cardiovascular biomarkers.

**Challenge:** [2026 NASA Space Apps Challenge — Health Monitoring Software for Astronauts on Space Missions](https://www.spaceappschallenge.org/)

## What this repository validates

`src/nasa_omega_validator.py` ingests two NASA datasets and evaluates the Omega phase metric used for early instability detection:

| Dataset | Path | Role |
| --- | --- | --- |
| NASA Langley Crew State Monitoring (`25.zip`) | `data/nasa_raw/nasa_csm_25/` | 256 Hz ECG / EEG / GSR flight-crew sessions (`25_CA`, `25_DA`, `25_SS`, `25_LOFT`) |
| NASA OSDR Inspiration4 OSD-575 | `data/nasa_raw/LSDS-8_Multiplex_serum_cardiovascular_EvePanel_TRANSFORMED.csv` | Pre- vs post-flight multiplex serum cardiovascular panel (EvePanel / LSDS-8) |

### Engine (summary)

1. Build a dynamic state series `s_t` from inter-beat intervals (IBI, seconds) or multi-biomarker displacement vectors.
2. Bilaterally clamp intervals: `delta_tau = clip(s_t, 0.01, 5.0)`.
3. Non-uniform finite-difference velocity `v_t` and acceleration `a_t`.
4. Decompose `a_t` into tangential laminar (`a_parallel`) and transverse orthogonal shear (`a_perp`) with titration firewall `epsilon = 1e-7`.
5. Integrate `|a_perp|` over a rolling ring buffer `K = 30` → `F_c`.
6. Evaluate  
   `Omega_t = 0.05 + 0.12*s_t + 0.08*(|a_perp|**2) + 0.02*F_c`  
   against phase boundaries **0.07** (Metastable Thinning) and **0.19** (Phase Veto).

Langley CSM files contain raw `ECG` at ~256 Hz (`TimeSecs` median Δt = 1/256 s). The validator detects R-peaks with `scipy.signal.find_peaks` when HR/IBI columns are absent. Inspiration4 samples (`L-*` preflight, `R+*` postflight) yield crew-wise state-vector displacement and transverse shear across mission timepoints.

## Challenge resources cited

This validation aligns with the official challenge **Resources** tab articles:

- **Risk of Inadequate Systems Operations Support** — crew health and performance risk from insufficient operations support during mission systems work.
- **Risk of Behavioral Changes and Psychiatric Disorders** — behavioral health risk pathways relevant to continuous physiological / cognitive state monitoring on long-duration missions.

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
| `results/validation_report.json` | Reproducible summary metrics, session/crew stats, lead-time detections |
| `results/omega_trajectory_summary.csv` | Decimated Omega / kinematic time series |
| Terminal table | Lead-time detections vs. Event markers (CSM) or first `R+1` (OSD-575) |

## Data notes

- Official Langley archive is distributed as `25.zip`. Unpacked session CSVs (`25_CA.csv`, `25_DA.csv`, `25_SS.csv`, `25_LOFT.csv`) are expected under `data/nasa_raw/nasa_csm_25/`. Zip archives under `data/nasa_raw/` are gitignored.
- Column headers observed: `TimeSecs`, 20 EEG channels, `ECG`, `R`, `GSR`, `Event`. Measured sampling rate ≈ **256 Hz**.
- OSD-575 LSDS-8 columns: `Sample Name` plus concentration / percent fields for a2-macroglobulin, AGP, CRP, fetuin-A, fibrinogen, haptoglobin, L-selectin, PF4, and SAP.

## Prior Work & Attribution

- **Sentinel Settle** Flutter/BLE mobile edge app (pre-existing): [https://github.com/Regsiren/sentinel-jolt-app](https://github.com/Regsiren/sentinel-jolt-app)
- Zenodo preprints: [10.5281/zenodo.20549045](https://doi.org/10.5281/zenodo.20549045), [10.5281/zenodo.20548468](https://doi.org/10.5281/zenodo.20548468)
- The underlying kinematic governance method is the subject of a pre-existing UKIPO patent application filed by Thohat Ventures Ltd. This repository publishes an independent NASA-dataset validation implementation under Apache-2.0; it does not expand, waive, or transfer those patent rights.

## License

Copyright 2026 Thohat Ventures Ltd  
Licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE).
