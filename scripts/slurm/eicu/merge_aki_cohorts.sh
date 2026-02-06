#!/bin/bash
# Merge all AKI trajectory sub-cohort files into a single dataset

cd /home/gaga/tamarw1/trajectory-modeling

echo "Merging AKI trajectory probability sub-cohorts..."
echo ""

python3 << 'EOF'
import pandas as pd
from pathlib import Path

result_dir = Path("results/eicu/aki")
cohort_files = sorted(result_dir.glob("aki_trajectory_probs_cohort*.csv"))

if not cohort_files:
    print("ERROR: No cohort files found!")
    exit(1)

print(f"Found {len(cohort_files)} cohort files:")
for f in cohort_files:
    print(f"  - {f.name}")

print("\nMerging...")
cohorts = []
for f in cohort_files:
    df = pd.read_csv(f)
    cohorts.append(df)
    print(f"  {f.name}: {len(df):,} rows, {df['stay_id'].nunique():,} patients")

merged = pd.concat(cohorts, ignore_index=True)

output_path = result_dir / "aki_trajectory_probs.csv"
merged.to_csv(output_path, index=False)

print(f"\n✓ Merged dataset saved: {output_path}")
print(f"  Total rows: {len(merged):,}")
print(f"  Total patients: {merged['stay_id'].nunique():,}")
EOF

echo ""
echo "Merge complete!"
