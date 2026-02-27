#!/usr/bin/env python
import duckdb
import pandas as pd
import numpy as np

DB_PATH = '/home/gaga/data/physionet/HiRiD/hirid.duckdb'
conn = duckdb.connect(DB_PATH, read_only=True)

cohort_query = """
SELECT patientid, admission_time, discharge_time, los_days
FROM patient_info
WHERE los_days >= 1.0
"""

cohort = conn.execute(cohort_query).fetchdf()
print(f"  Total ICU stays: {len(cohort):,}")

patient_ids = cohort['patientid'].tolist()[:100]  # Use just first 100 for speed

VARS = {
    200: 'HeartRate',
    100: 'SysBP',
    600: 'SysBP',
    120: 'DiasBP',
    620: 'DiasBP',
    110: 'MAP',
    610: 'MAP',
    24000524: 'Lactate',
    24000732: 'Lactate',
    24000485: 'Lactate',
}

# Check variable counts first
vars_query = f"""
SELECT variableid, COUNT(*) as cnt
FROM observations
WHERE variableid IN {tuple(VARS.keys())}
AND patientid IN ({','.join(str(x) for x in patient_ids)})
GROUP BY variableid
ORDER BY cnt DESC
"""

result = conn.execute(vars_query).fetchdf()
print("\nVariable counts (first 100 patients):")
print(result)

obs_query = f"""
SELECT 
    o.patientid,
    o.datetime as charttime,
    o.variableid,
    CAST(o.value AS DOUBLE) as valuenum,
    p.admission_time as admittime
FROM observations o
INNER JOIN patient_info p ON o.patientid = p.patientid
WHERE 
    o.variableid IN {tuple(VARS.keys())}
    AND o.value IS NOT NULL
    AND o.patientid IN ({','.join(str(x) for x in patient_ids)})
LIMIT 100
"""

obs_df = conn.execute(obs_query).fetchdf()
print(f"\nFirst 100 rows from obs_df:")
print(obs_df)

obs_df['feature'] = obs_df['variableid'].map(VARS)
print(f"\nAfter mapping features:")
print(obs_df[['variableid', 'valuenum', 'feature']].head(10))

RANGES = {
    'HeartRate': (20, 300),
    'SysBP': (40, 300),
    'DiasBP': (20, 200),
    'MAP': (20, 200),
    'Lactate': (0.1, 20),
}

print(f"\nBefore filtering: {len(obs_df)} rows")
for feat, (min_v, max_v) in RANGES.items():
    mask = (obs_df['feature'] == feat) & (~obs_df['valuenum'].between(min_v, max_v))
    obs_df = obs_df[~mask]
    print(f"After filtering {feat}: {len(obs_df)} rows")

print(f"\nAfter filtering: {len(obs_df)} rows")
print(obs_df[['feature', 'valuenum']].head(20))

conn.close()
