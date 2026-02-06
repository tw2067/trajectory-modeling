"""
MIMIC Ventilator Preprocessing Script

Extracts:
- Mechanically ventilated cohort
- P/F ratio time series (PaO2 / FiO2 daily means)

Outputs:
- results/mimic/ventilator/pf_ratio_timeseries.csv
- results/mimic/ventilator/hadm_cohort.csv

Usage:
    python scripts/mimic_preprocessing/ventilator_preprocess.py
"""

import os
import pandas as pd
from pathlib import Path

from common import MIMICPaths, connect_db, load_metadata, fetch_labevents_timeseries, fetch_vitals_labs, ensure_time_day

OUTPUT_DIR = Path("results/mimic/ventilator")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MIMIC Ventilator Preprocessing")
print("=" * 80)

paths = MIMICPaths()
conn = connect_db(paths.db_path)

hadm_query = """
WITH ventilated_patients AS (
    SELECT DISTINCT
        i.hadm_id::INTEGER AS hadm_id,
        MIN(ce.charttime::TIMESTAMP) as vent_start
    FROM chartevents ce
    INNER JOIN icustays i ON ce.icustay_id = i.icustay_id
    WHERE 
        ce.itemid IN (720, 223849)
        AND ce.value IS NOT NULL
    GROUP BY i.hadm_id
)
SELECT 
    i.subject_id::INTEGER AS subject_id,
    i.hadm_id::INTEGER AS hadm_id,
    a.admittime::TIMESTAMP as admittime,
    a.dischtime::TIMESTAMP as dischtime,
    vp.vent_start::TIMESTAMP as vent_start,
    CAST(a.dischtime::DATE - a.admittime::DATE AS REAL) as los_days,
    p.gender,
    CAST((a.admittime::DATE - p.dob::DATE) / 365.25 AS INT) as age,
    a.hospital_expire_flag::INT as hospital_expire_flag,
    a.discharge_location
FROM icustays i
INNER JOIN patients p ON i.subject_id = p.subject_id
INNER JOIN admissions a ON i.hadm_id = a.hadm_id
INNER JOIN ventilated_patients vp ON i.hadm_id = vp.hadm_id
WHERE 
    CAST((a.admittime::DATE - p.dob::DATE) / 365.25 AS INT) BETWEEN 18 AND 90
    AND CAST(a.dischtime::DATE - a.admittime::DATE AS REAL) >= 2.0
"""

hadm_df = conn.execute(hadm_query).fetchdf()
hadm_df.columns = hadm_df.columns.str.lower()
print(f"✓ Loaded {len(hadm_df):,} ventilated admissions")

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
vital_meta["feature_upper"] = vital_meta["feature name"].str.upper()

pao2_items = labs_meta[labs_meta["feature_upper"].str.contains("PAO2")]["itemid"].tolist()
fio2_items = vital_meta[vital_meta["feature_upper"].str.contains("FIO2")]["itemid"].tolist()

# Load PaO2 from labevents
pao2_df = fetch_labevents_timeseries(conn, hadm_ids, pao2_items, "pao2", 20.0, 600.0)
pao2_df = ensure_time_day(pao2_df)

# Load FiO2 from chartevents via vitals query
vitals_df, _ = fetch_vitals_labs(conn, hadm_ids, vital_items=fio2_items, lab_items=[])
if len(vitals_df) > 0:
    fio2_df = vitals_df.copy()
    fio2_df = fio2_df.rename(columns={"valuenum": "fio2"})
    fio2_df["fio2"] = fio2_df["fio2"].astype(float)
    fio2_df["time_days"] = (pd.to_datetime(fio2_df["charttime"]) - pd.to_datetime(fio2_df["admittime"])).dt.total_seconds() / 86400.0
    fio2_df = ensure_time_day(fio2_df)
else:
    fio2_df = pd.DataFrame()

if len(pao2_df) == 0 or len(fio2_df) == 0:
    print("⚠️ Missing PaO2 or FiO2 data. Skipping P/F ratio computation.")
else:
    pao2_daily = pao2_df.groupby(["hadm_id", "time_day"])["pao2"].mean().reset_index()
    fio2_daily = fio2_df.groupby(["hadm_id", "time_day"])["fio2"].mean().reset_index()

    pf = pao2_daily.merge(fio2_daily, on=["hadm_id", "time_day"], how="inner")
    pf["pf_ratio"] = pf["pao2"] / pf["fio2"]

    # Add time_days as mid-day
    pf["time_days"] = pf["time_day"].astype(float)

    pf = pf[["hadm_id", "time_days", "time_day", "pf_ratio"]]
    pf.to_csv(OUTPUT_DIR / "pf_ratio_timeseries.csv", index=False)
    print("✓ Saved pf_ratio_timeseries.csv")

hadm_df.to_csv(OUTPUT_DIR / "hadm_cohort.csv", index=False)
print("✓ Saved hadm_cohort.csv")
conn.close()
