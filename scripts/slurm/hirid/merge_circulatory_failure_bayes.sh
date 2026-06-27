#!/usr/bin/env bash
# Merge Bayesian trajectory cohort files for HiRID circulatory failure.
# Run after all run_hirid_circulatory_failure_bayes_trajectories.slurm tasks complete.
#
# Unlike MIMIC/eICU, the HiRID trajectory script does not produce a merged
# prediction dataset, so this script builds it by joining the per-biomarker
# trajectory files into the base prediction dataset.

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/gaga/data/physionet}"
DATA_DIR="${DATA_ROOT}/hirid/circulatory_failure"
N_COHORTS=16

echo "Merging HiRID Circulatory Failure Bayesian trajectory files..."

python - <<'PY'
from pathlib import Path
import pandas as pd
import os

data_dir = Path(os.environ.get("DATA_ROOT", "/home/gaga/data/physionet")) / "hirid" / "circulatory_failure"
n_cohorts = 16

def merge_cohorts(base_name: str) -> pd.DataFrame | None:
    parts = []
    for i in range(n_cohorts):
        p = data_dir / f"{base_name}_cohort{i:02d}.csv"
        if p.exists():
            parts.append(pd.read_csv(p))
        else:
            print(f"  WARNING: Missing cohort file {p}")

    if not parts:
        print(f"  WARNING: No cohort files found for {base_name}")
        return None

    merged = pd.concat(parts, ignore_index=True)
    out = data_dir / f"{base_name}.csv"
    merged.to_csv(out, index=False)
    print(f"  ✓ {out} ({len(merged):,} rows)")
    return merged


# Step 1: merge per-biomarker cohort files
biomarker_dfs = {}
for biomarker in ["lactate", "heartrate", "systolic"]:
    print(f"Merging {biomarker}...")
    df = merge_cohorts(f"{biomarker}_trajectory_probs_bayes")
    if df is not None:
        biomarker_dfs[biomarker] = df

# Step 2: build merged prediction dataset
pred_path = data_dir / "circulatory_failure_prediction_dataset.csv"
if not pred_path.exists():
    print(f"ERROR: Base prediction dataset not found: {pred_path}")
    raise SystemExit(1)

dataset = pd.read_csv(pred_path)
print(f"\nLoaded base prediction dataset: {len(dataset):,} rows")

# Infer ID and windowing columns
id_col = next((c for c in ["patientid", "stay_id", "hadm_id"] if c in dataset.columns), None)
time_col = next((c for c in ["time_hour", "time_day"] if c in dataset.columns), None)
if id_col is None or time_col is None:
    print(f"ERROR: Could not determine ID ({id_col}) or time ({time_col}) columns.")
    raise SystemExit(1)

print(f"Merge keys: {id_col}, {time_col}")

# Column renames: prob_stable -> {bm}_stable, etc.
col_map = {
    "lactate":   {"prob_stable": "lactate_stable",   "prob_gradual_increase": "lactate_gradual",   "prob_rapid_increase": "lactate_rapid"},
    "heartrate": {"prob_stable": "heartrate_stable",  "prob_gradual_increase": "heartrate_gradual",  "prob_rapid_increase": "heartrate_rapid"},
    "systolic":  {"prob_stable": "systolic_stable",   "prob_gradual_decline":  "systolic_gradual",   "prob_rapid_decline":  "systolic_rapid"},
}

for biomarker, traj_df in biomarker_dfs.items():
    rename = col_map[biomarker]
    traj_cols = [c for c in rename if c in traj_df.columns]
    if not traj_cols:
        print(f"  WARNING: No expected prob columns found in {biomarker} trajectory file")
        continue

    traj_df = traj_df[[id_col, time_col] + traj_cols].rename(columns=rename)
    dataset = dataset.merge(traj_df, on=[id_col, time_col], how="left")
    print(f"  Merged {biomarker}: {len(dataset):,} rows")

out_path = data_dir / "circulatory_failure_prediction_dataset_with_probs.csv"
dataset.to_csv(out_path, index=False)
print(f"\n✓ {out_path} ({len(dataset):,} rows)")
print("\n✓ Merge complete!")
PY
