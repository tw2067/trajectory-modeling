"""
MIMIC Sepsis Preprocessing Script

Extracts:
- Sepsis cohort (ICD-9 codes)
- Lactate, WBC, Platelet time series

Outputs:
- results/mimic/sepsis/lactate_timeseries.csv
- results/mimic/sepsis/wbc_timeseries.csv
- results/mimic/sepsis/platelet_timeseries.csv
- results/mimic/sepsis/hadm_cohort.csv

Usage:
    python scripts/mimic_preprocessing/sepsis_preprocess.py
"""

import os
import sys
import pandas as pd
from pathlib import Path

from common import MIMICPaths, connect_db, load_metadata, fetch_labevents_timeseries, ensure_time_day

OUTPUT_DIR = Path("results/mimic/sepsis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MIMIC Sepsis Preprocessing")
print("=" * 80)

paths = MIMICPaths()
conn = connect_db(paths.db_path)

hadm_query = """
WITH sepsis_patients AS (
    SELECT DISTINCT a.hadm_id
    FROM admissions a
    INNER JOIN diagnoses_icd d ON a.hadm_id = d.hadm_id
    WHERE 
        (d.icd9_code LIKE '99591' OR
         d.icd9_code LIKE '99592' OR
         d.icd9_code LIKE '78552')
)
SELECT 
    a.subject_id::INTEGER AS subject_id,
    a.hadm_id::INTEGER AS hadm_id,
    a.admittime::TIMESTAMP as admittime,
    a.dischtime::TIMESTAMP as dischtime,
    CAST(a.dischtime::DATE - a.admittime::DATE AS REAL) as los_days,
    p.gender,
    CAST((a.admittime::DATE - p.dob::DATE) / 365.25 AS INT) as age,
    a.hospital_expire_flag::INT as hospital_expire_flag,
    a.discharge_location
FROM admissions a
INNER JOIN patients p ON a.subject_id = p.subject_id
INNER JOIN sepsis_patients sp ON a.hadm_id = sp.hadm_id
WHERE 
    CAST((a.admittime::DATE - p.dob::DATE) / 365.25 AS INT) BETWEEN 18 AND 90
    AND CAST(a.dischtime::DATE - a.admittime::DATE AS REAL) >= 2.0
"""

hadm_df = conn.execute(hadm_query).fetchdf()
hadm_df.columns = hadm_df.columns.str.lower()
print(f"✓ Loaded {len(hadm_df):,} sepsis admissions")

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

vital_meta, labs_meta = load_metadata(paths)

labs_meta["feature_upper"] = labs_meta["feature name"].str.upper()

lactate_items = labs_meta[labs_meta["feature_upper"].str.contains("LACTATE")]["itemid"].tolist()
wbc_items = labs_meta[labs_meta["feature_upper"].str.contains("WBC")]["itemid"].tolist()
platelet_items = labs_meta[labs_meta["feature_upper"].str.contains("PLATELET")]["itemid"].tolist()

lactate_df = fetch_labevents_timeseries(conn, hadm_ids, lactate_items, "lactate", 0.1, 50.0)
wbc_df = fetch_labevents_timeseries(conn, hadm_ids, wbc_items, "wbc", 0.1, 300.0)
platelet_df = fetch_labevents_timeseries(conn, hadm_ids, platelet_items, "platelet", 1.0, 2000.0)

lactate_df = ensure_time_day(lactate_df)
wbc_df = ensure_time_day(wbc_df)
platelet_df = ensure_time_day(platelet_df)

lactate_df.to_csv(OUTPUT_DIR / "lactate_timeseries.csv", index=False)
wbc_df.to_csv(OUTPUT_DIR / "wbc_timeseries.csv", index=False)
platelet_df.to_csv(OUTPUT_DIR / "platelet_timeseries.csv", index=False)
hadm_df.to_csv(OUTPUT_DIR / "hadm_cohort.csv", index=False)

print("✓ Saved sepsis time series and cohort")
conn.close()
