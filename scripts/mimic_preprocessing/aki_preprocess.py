"""
MIMIC AKI Preprocessing Script

Extracts:
- ICU cohort
- Creatinine time series

Outputs:
- results/mimic/aki/creatinine_timeseries.csv
- results/mimic/aki/hadm_cohort.csv

Usage:
    python scripts/mimic_preprocessing/aki_preprocess.py
"""

import os
import pandas as pd
from pathlib import Path

from common import MIMICPaths, connect_db, load_metadata, fetch_labevents_timeseries, ensure_time_day

OUTPUT_DIR = Path("results/mimic/aki")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MIMIC AKI Preprocessing")
print("=" * 80)

paths = MIMICPaths()
conn = connect_db(paths.db_path)

hadm_query = """
SELECT 
    i.subject_id::INTEGER AS subject_id,
    i.hadm_id::INTEGER AS hadm_id,
    a.admittime::TIMESTAMP as admittime,
    a.dischtime::TIMESTAMP as dischtime,
    i.intime::TIMESTAMP as intime,
    i.outtime::TIMESTAMP as outtime,
    CAST(a.dischtime::DATE - a.admittime::DATE AS REAL) as los_days,
    p.gender,
    CAST((a.admittime::DATE - p.dob::DATE) / 365.25 AS INT) as age,
    a.hospital_expire_flag::INT as hospital_expire_flag,
    a.discharge_location
FROM icustays i
INNER JOIN patients p ON i.subject_id = p.subject_id
INNER JOIN admissions a ON i.hadm_id = a.hadm_id
WHERE 
    CAST((a.admittime::DATE - p.dob::DATE) / 365.25 AS INT) BETWEEN 18 AND 90
    AND CAST(a.dischtime::DATE - a.admittime::DATE AS REAL) >= 2.0
"""

hadm_df = conn.execute(hadm_query).fetchdf()
hadm_df.columns = hadm_df.columns.str.lower()
print(f"✓ Loaded {len(hadm_df):,} ICU admissions")

max_adm = int(os.environ.get("MIMIC_MAX_HADM", "5000"))
if len(hadm_df) > max_adm:
    hadm_subset = (
        hadm_df.sort_values("admittime")
        .drop_duplicates(subset=["subject_id"])["hadm_id"]
        .sample(min(max_adm, hadm_df["subject_id"].nunique()), random_state=920)
        .tolist()
    )
    hadm_df = hadm_df[hadm_df["hadm_id"].isin(hadm_subset)]
    print(f"✓ Using {len(hadm_df):,} admissions after sampling")

hadm_ids = hadm_df["hadm_id"].astype(int).tolist()

_, labs_meta = load_metadata(paths)
labs_meta["feature_upper"] = labs_meta["feature name"].str.upper()

creat_items = labs_meta[labs_meta["feature_upper"].str.contains("CREATININE")]["itemid"].tolist()

creat_df = fetch_labevents_timeseries(conn, hadm_ids, creat_items, "creatinine", 0.1, 20.0)
creat_df = ensure_time_day(creat_df)

creat_df.to_csv(OUTPUT_DIR / "creatinine_timeseries.csv", index=False)
hadm_df.to_csv(OUTPUT_DIR / "hadm_cohort.csv", index=False)

print("✓ Saved creatinine_timeseries.csv and hadm_cohort.csv")
conn.close()
