#!/usr/bin/env bash
# Merge all eICU sepsis trajectory sub-cohort files into single datasets

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "Could not locate repo root"; exit 1; }

DATA_ROOT="${DATA_ROOT:-/home/gaga/data/physionet}"
cd "$REPO_ROOT" || exit 1

echo "Merging eICU sepsis trajectory probability sub-cohorts..."

echo ""
python3 << 'EOF'
import pandas as pd
from pathlib import Path
import os

result_dir = Path(os.environ.get("DATA_ROOT", "/home/gaga/data/physionet")) / "eicu" / "sepsis"

def merge_cohorts(pattern: str, output_name: str, id_col: str):
    cohort_files = sorted(result_dir.glob(pattern))
    if not cohort_files:
        print(f"ERROR: No cohort files found for {pattern}!")
        return

    print(f"Found {len(cohort_files)} cohort files for {output_name}:")
    for f in cohort_files:
        print(f"  - {f.name}")

    print("\nMerging...")
    cohorts = []
    for f in cohort_files:
        df = pd.read_csv(f)
        cohorts.append(df)
        print(f"  {f.name}: {len(df):,} rows, {df[id_col].nunique():,} patients")

    merged = pd.concat(cohorts, ignore_index=True)
    output_path = result_dir / output_name
    merged.to_csv(output_path, index=False)

    print(f"\n✓ Merged dataset saved: {output_path}")
    print(f"  Total rows: {len(merged):,}")
    print(f"  Total patients: {merged[id_col].nunique():,}")

merge_cohorts("lactate_trajectory_probs_bayes_cohort*.csv", "lactate_trajectory_probs_bayes.csv", "stay_id")
merge_cohorts("wbc_trajectory_probs_bayes_cohort*.csv", "wbc_trajectory_probs_bayes.csv", "stay_id")
merge_cohorts("platelet_trajectory_probs_bayes_cohort*.csv", "platelet_trajectory_probs_bayes.csv", "stay_id")
merge_cohorts("sepsis_trajectory_probs_cohort*.csv", "sepsis_trajectory_probs.csv", "stay_id")
EOF

echo ""
echo "Merge complete!"