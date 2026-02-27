#!/usr/bin/env python
import duckdb
import pandas as pd
from pathlib import Path

DB_PATH = '/home/gaga/data/physionet/HiRiD/hirid.duckdb'
conn = duckdb.connect(DB_PATH, read_only=True)

print("Getting cohort...")
cohort_query = """
SELECT patientid, admission_time, discharge_time, los_days
FROM patient_info
WHERE los_days >= 1.0
"""

cohort = conn.execute(cohort_query).fetchdf()
print(f"✓ Cohort: {len(cohort):,} patients")

patient_ids = cohort['patientid'].tolist()
print(f"Patient IDs range: {min(patient_ids)} to {max(patient_ids)}")

# Check how many rows per variable ID
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

print("\nVariable counts...")
for vid, fname in VARS.items():
    try:
        count = conn.execute(f"""
        SELECT COUNT(*) FROM observations
        WHERE variableid = {vid}
        """).fetchone()[0]
        print(f"  {fname:12} ({vid:8}): {count:12,}")
    except Exception as e:
        print(f"  {fname:12} ({vid:8}): ERROR - {e}")

conn.close()
print("\n✓ Done")
