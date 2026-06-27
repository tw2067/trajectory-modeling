#!/usr/bin/env bash
# Merge Bayesian trajectory cohort files for MIMIC circulatory failure.
# Run after all run_mimic_circulatory_failure_bayes_trajectories.slurm tasks complete.

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/gaga/data/physionet}"
DATA_DIR="${DATA_ROOT}/mimic/circulatory_failure"
N_COHORTS=4

echo "Merging MIMIC Circulatory Failure Bayesian trajectory files..."

python - <<'PY'
from pathlib import Path
import pandas as pd
import os

data_dir = Path(os.environ.get("DATA_ROOT", "/home/gaga/data/physionet")) / "mimic" / "circulatory_failure"
n_cohorts = 4

def merge_base(base_name: str) -> pd.DataFrame | None:
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


for biomarker in ["lactate", "heartrate", "systolic"]:
    print(f"Merging {biomarker}...")
    merge_base(f"{biomarker}_trajectory_probs_bayes")

print("Merging prediction dataset with probs...")
merge_base("circulatory_failure_prediction_dataset_with_probs")

print("\n✓ Merge complete!")
PY
