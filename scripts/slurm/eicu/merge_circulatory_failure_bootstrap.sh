#!/bin/bash
# Merge bootstrap trajectory cohort files for eICU circulatory failure
# Run after all SLURM array jobs complete

set -euo pipefail

DATA_DIR="/home/gaga/data/physionet/eicu/circulatory_failure"
N_COHORTS=8

echo "Merging eICU Circulatory Failure Bootstrap trajectory files..."

python - <<'PY'
from pathlib import Path
import pandas as pd

data_dir = Path("/home/gaga/data/physionet/eicu/circulatory_failure")
n_cohorts = 8

def merge_base(base_name: str):
    parts = []
    for i in range(n_cohorts):
        p_parquet = data_dir / f"{base_name}_cohort{i:02d}.parquet"
        p_csv = data_dir / f"{base_name}_cohort{i:02d}.csv"
        if p_parquet.exists():
            parts.append(pd.read_parquet(p_parquet))
        elif p_csv.exists():
            parts.append(pd.read_csv(p_csv))
        else:
            print(f"  WARNING: Missing cohort file for {base_name}, cohort={i:02d}")

    if not parts:
        print(f"  WARNING: No cohort files found for {base_name}")
        return

    merged = pd.concat(parts, ignore_index=True)
    out_parquet = data_dir / f"{base_name}.parquet"
    out_csv = data_dir / f"{base_name}.csv"
    merged.to_parquet(out_parquet, index=False, compression="zstd")
    merged.to_csv(out_csv, index=False)
    print(f"  ✓ Created {out_parquet} ({len(merged):,} rows)")
    print(f"  ✓ Created {out_csv} ({len(merged):,} rows)")


for biomarker in ["lactate", "heartrate", "systolic"]:
    print(f"Merging {biomarker}...")
    merge_base(f"{biomarker}_trajectory_probs_bootstrap")

print("Merging prediction dataset...")
merge_base("circulatory_failure_prediction_dataset_with_bootstrap_probs")

print("\n✓ Merge complete!")
PY
