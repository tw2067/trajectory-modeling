#!/usr/bin/env python
import duckdb
import pandas as pd

conn = duckdb.connect('/home/gaga/data/physionet/HiRiD/hirid.duckdb', read_only=True)

# Check variable IDs available
vars_query = '''
SELECT variableid, COUNT(*) as cnt
FROM observations
WHERE variableid IN (110, 610, 100, 600, 120, 620, 200, 24000524, 24000732, 24000485)
GROUP BY variableid
ORDER BY cnt DESC
'''
result = conn.execute(vars_query).fetchdf()
print('Variable counts:')
print(result)

# Check if there are MAP measurements specifically
map_count = conn.execute('SELECT COUNT(*) FROM observations WHERE variableid IN (110, 610)').fetchone()[0]
print(f'\nMAP measurements (110, 610): {map_count:,}')

sysbp_count = conn.execute('SELECT COUNT(*) FROM observations WHERE variableid IN (100, 600)').fetchone()[0]
print(f'SysBP measurements (100, 600): {sysbp_count:,}')

diasbp_count = conn.execute('SELECT COUNT(*) FROM observations WHERE variableid IN (120, 620)').fetchone()[0]
print(f'DiasBP measurements (120, 620): {diasbp_count:,}')

# Check obs_daily generation
print('\n--- Checking hourly aggregation ---')
from pathlib import Path

OUTPUT_DIR = Path("/home/gaga/data/physionet/hirid/circulatory_failure")
cohort_query = """
SELECT patientid, admission_time, discharge_time, los_days
FROM patient_info
WHERE los_days >= 1.0
"""

cohort = conn.execute(cohort_query).fetchdf()
patient_ids = cohort['patientid'].tolist()
print(f'Cohort size: {len(patient_ids):,}')

# Variable IDs
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

import numpy as np

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
    AND o.patientid IN {tuple(patient_ids)}
ORDER BY o.patientid, o.datetime
"""

obs_df = conn.execute(obs_query).fetchdf()
print(f'obs_df rows: {len(obs_df):,}')
obs_df['feature'] = obs_df['variableid'].map(VARS)
print(f'Feature value counts:\n{obs_df["feature"].value_counts()}')

obs_df['charttime'] = pd.to_datetime(obs_df['charttime'])
obs_df['admittime'] = pd.to_datetime(obs_df['admittime'])
obs_df['time_hours'] = (obs_df['charttime'] - obs_df['admittime']).dt.total_seconds() / 3600.0
obs_df['time_hour'] = np.floor(obs_df['time_hours']).astype(int)

# Filter by ranges
RANGES = {
    'HeartRate': (20, 300),
    'SysBP': (40, 300),
    'DiasBP': (20, 200),
    'MAP': (20, 200),
    'Lactate': (0.1, 20),
}

for feat, (min_v, max_v) in RANGES.items():
    mask = obs_df['feature'] == feat
    if mask.any():
        obs_df = obs_df[~mask | obs_df['valuenum'].between(min_v, max_v)]

print(f'\nAfter filtering, obs_df rows: {len(obs_df):,}')
print(f'Feature value counts after filtering:\n{obs_df["feature"].value_counts()}')

# Check hourly aggregation
obs_daily = obs_df.groupby(['patientid', 'time_hour', 'feature'])['valuenum'].agg(['min', 'max', 'mean']).reset_index()
obs_daily = obs_daily.pivot_table(
    index=['patientid', 'time_hour'],
    columns='feature',
    values=['min', 'max', 'mean'],
    aggfunc='first'
).reset_index()

obs_daily.columns = ['_'.join(col).strip('_') for col in obs_daily.columns]
print(f'\nobs_daily columns: {obs_daily.columns.tolist()}')
print(f'obs_daily rows: {len(obs_daily):,}')

# Check if mean_MAP exists
if 'mean_MAP' in obs_daily.columns:
    print(f'\nmean_MAP non-NA count: {obs_daily["mean_MAP"].notna().sum():,}')
else:
    print('\nmean_MAP not in columns!')
    if 'mean_SysBP' in obs_daily.columns and 'mean_DiasBP' in obs_daily.columns:
        obs_daily['mean_MAP'] = (2 * obs_daily['mean_DiasBP'] + obs_daily['mean_SysBP']) / 3.0
        print(f'Computed mean_MAP non-NA count: {obs_daily["mean_MAP"].notna().sum():,}')
    else:
        print(f'Cannot compute MAP. SysBP present: {"mean_SysBP" in obs_daily.columns}, DiasBP present: {"mean_DiasBP" in obs_daily.columns}')

conn.close()
