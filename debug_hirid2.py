#!/usr/bin/env python
import duckdb
import pandas as pd
import numpy as np

DB_PATH = '/home/gaga/data/physionet/HiRiD/hirid.duckdb'
conn = duckdb.connect(DB_PATH, read_only=True)

# Get cohort
cohort = conn.execute('SELECT patientid FROM patient_info WHERE los_days >= 1.0').fetchdf()
patient_ids = cohort['patientid'].tolist()
print(f'Cohort: {len(patient_ids)} patients')

# Check data for first 100 patients
sample_ids = patient_ids[:100]
sample = conn.execute(f"""
SELECT variableid, COUNT(*) as cnt
FROM observations
WHERE variableid IN (110, 610, 100, 600, 120, 620, 200, 24000524, 24000732, 24000485)
AND patientid IN ({','.join(str(x) for x in sample_ids)})
GROUP BY variableid
ORDER BY cnt DESC
""").fetchdf()

print(f'\nSample of 100 patients:')
print(sample)
print(f'Total rows in sample: {sample["cnt"].sum():,}')

# Now get full cohort but with variable count
full_vars = conn.execute(f"""
SELECT variableid, COUNT(*) as cnt
FROM observations
WHERE variableid IN (110, 610, 100, 600, 120, 620, 200, 24000524, 24000732, 24000485)
AND patientid IN ({','.join(str(x) for x in patient_ids)})
GROUP BY variableid
ORDER BY cnt DESC
""").fetchdf()

print(f'\nFull cohort variables:')
print(full_vars)
print(f'Total rows: {full_vars["cnt"].sum():,}')

conn.close()
