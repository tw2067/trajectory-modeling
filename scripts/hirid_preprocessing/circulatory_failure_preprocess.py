"""
HiRiD Circulatory Failure Preprocessing Script

Creates:
- Lactate, Heart Rate, and Systolic BP time series
- Prediction dataset with circulatory failure outcome

Outcome definition (default):
- Future window contains vasopressor start OR hypotension (SBP below threshold)
- Excludes windows where patient is already in circulatory failure at prediction time

Usage:
    python scripts/hirid_preprocessing/circulatory_failure_preprocess.py
"""

from __future__ import annotations

import os
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np

OUTPUT_DIR = Path("results/hirid/circulatory_failure")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = "/home/gaga/data/physionet/HiRiD/hirid.duckdb"
MIN_LOS_DAYS = 1.0
PREDICTION_GAP_HOURS = 1.0
PREDICTION_WINDOW_HOURS = 8.0
MAP_THRESHOLD = 65.0
LACTATE_THRESHOLD = 2.0

print("=" * 80)
print("HiRiD Circulatory Failure Preprocessing")
print("=" * 80)

conn = duckdb.connect(DB_PATH, read_only=True)

cohort_query = f"""
SELECT patientid, admission_time, discharge_time, los_days
FROM patient_info
WHERE los_days >= {MIN_LOS_DAYS}
"""

cohort = conn.execute(cohort_query).fetchdf()
print(f"  Total ICU stays: {len(cohort):,}")

patient_ids = cohort['patientid'].tolist()

# Variable IDs
VARS = {
    200: 'HeartRate',
    210: 'SysBP',
    220: 'DiasBP',
    20001600: 'Lactate',
}

# Ranges for sanity filtering
RANGES = {
    'HeartRate': (20, 300),
    'SysBP': (40, 300),
    'DiasBP': (20, 200),
    'Lactate': (0.1, 20),
}

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
obs_df['feature'] = obs_df['variableid'].map(VARS)

# Filter by ranges
for feat, (min_v, max_v) in RANGES.items():
    mask = obs_df['feature'] == feat
    if mask.any():
        obs_df = obs_df[~mask | obs_df['valuenum'].between(min_v, max_v)]

obs_df['charttime'] = pd.to_datetime(obs_df['charttime'])
obs_df['admittime'] = pd.to_datetime(obs_df['admittime'])
obs_df['time_hours'] = (obs_df['charttime'] - obs_df['admittime']).dt.total_seconds() / 3600.0
obs_df['time_hour'] = np.floor(obs_df['time_hours']).astype(int)

# Time series outputs
lactate_ts = obs_df[obs_df['feature'] == 'Lactate'].copy()
lactate_ts = lactate_ts.rename(columns={'patientid': 'patientid', 'valuenum': 'lactate'})
heartrate_ts = obs_df[obs_df['feature'] == 'HeartRate'].copy()
heartrate_ts = heartrate_ts.rename(columns={'patientid': 'patientid', 'valuenum': 'heartrate'})
sbp_ts = obs_df[obs_df['feature'] == 'SysBP'].copy()
sbp_ts = sbp_ts.rename(columns={'patientid': 'patientid', 'valuenum': 'systolic'})

# Baselines (first 24h)

def add_baseline(df: pd.DataFrame, value_col: str, baseline_col: str) -> pd.DataFrame:
    if len(df) == 0:
        return df
    baseline_window = df[df['time_hours'] <= 24.0]
    baseline = (
        baseline_window.sort_values('time_hours')
        .groupby('patientid')[value_col]
        .first()
        .reset_index()
        .rename(columns={value_col: baseline_col})
    )
    return df.merge(baseline, on='patientid', how='left')


lactate_ts = add_baseline(lactate_ts, 'lactate', 'baseline_lactate')
heartrate_ts = add_baseline(heartrate_ts, 'heartrate', 'baseline_heartrate')
sbp_ts = add_baseline(sbp_ts, 'systolic', 'baseline_systolic')

lactate_ts.to_csv(OUTPUT_DIR / "lactate_timeseries.csv", index=False)
heartrate_ts.to_csv(OUTPUT_DIR / "heartrate_timeseries.csv", index=False)
sbp_ts.to_csv(OUTPUT_DIR / "systolic_timeseries.csv", index=False)
cohort.to_csv(OUTPUT_DIR / "patient_cohort.csv", index=False)

print("✓ Saved circulatory time series and cohort")

# Hourly aggregation
obs_daily = obs_df.groupby(['patientid', 'time_hour', 'feature'])['valuenum'].agg(['min', 'max', 'mean']).reset_index()
obs_daily = obs_daily.pivot_table(
    index=['patientid', 'time_hour'],
    columns='feature',
    values=['min', 'max', 'mean'],
    aggfunc='first'
).reset_index()

obs_daily.columns = ['_'.join(col).strip('_') for col in obs_daily.columns]

# Compute MAP if SysBP & DiasBP available
if 'mean_SysBP' in obs_daily.columns and 'mean_DiasBP' in obs_daily.columns:
    obs_daily['mean_MAP'] = (2 * obs_daily['mean_DiasBP'] + obs_daily['mean_SysBP']) / 3.0

# Vasopressor detection (HiRiD pharma IDs)
VASOPRESSOR_IDS = [
    1000462, 1000656, 1000657, 1000658,  # Noradrenalin
    71, 1000750, 1000649, 1000650, 1000655,  # Adrenalin
    112, 113,  # Vasopressin
    426,  # Dobutrex (inotrope)
]

vaso_query = f"""
SELECT r.patientid, r.datetime as charttime, p.admission_time as admittime
FROM pharma_records r
INNER JOIN patient_info p ON r.patientid = p.patientid
WHERE r.pharmaid IN {tuple(VASOPRESSOR_IDS)}
AND r.patientid IN {tuple(patient_ids)}
"""

vaso_df = conn.execute(vaso_query).fetchdf()
if len(vaso_df) > 0:
    vaso_df['charttime'] = pd.to_datetime(vaso_df['charttime'])
    vaso_df['admittime'] = pd.to_datetime(vaso_df['admittime'])
    vaso_df['time_hour'] = ((vaso_df['charttime'] - vaso_df['admittime']).dt.total_seconds() / 3600.0).astype(int)
    vaso_start = vaso_df.groupby('patientid')['time_hour'].min().reset_index().rename(columns={'time_hour': 'vaso_start_hour'})
else:
    vaso_start = pd.DataFrame(columns=['patientid', 'vaso_start_hour'])

failure_events = []
excluded = {'already_failure': 0, 'no_future_data': 0, 'ambiguous': 0}

for patientid, grp in obs_daily.groupby('patientid'):
    grp = grp.sort_values('time_hour')
    for i in range(len(grp)):
        current_time = grp.iloc[i]['time_hour']
        current_map = grp.iloc[i].get('mean_MAP', np.nan)
        current_lactate = grp.iloc[i].get('mean_Lactate', np.nan)

        current_vaso = False
        if patientid in vaso_start['patientid'].values:
            start_hour = vaso_start[vaso_start['patientid'] == patientid]['vaso_start_hour'].iloc[0]
            current_vaso = start_hour <= current_time

        if pd.isna(current_map) or pd.isna(current_lactate):
            excluded['ambiguous'] += 1
            continue

        if current_lactate >= LACTATE_THRESHOLD and (current_map <= MAP_THRESHOLD or current_vaso):
            excluded['already_failure'] += 1
            continue

        prediction_start = current_time + PREDICTION_GAP_HOURS
        prediction_end = current_time + PREDICTION_GAP_HOURS + PREDICTION_WINDOW_HOURS

        future_grp = grp[(grp['time_hour'] >= prediction_start) & (grp['time_hour'] <= prediction_end)]
        if len(future_grp) == 0:
            excluded['no_future_data'] += 1
            continue

        if 'mean_MAP' not in future_grp.columns or 'mean_Lactate' not in future_grp.columns:
            excluded['ambiguous'] += 1
            continue

        future_hypotension = (future_grp['mean_MAP'] <= MAP_THRESHOLD).any()
        future_lactate = (future_grp['mean_Lactate'] >= LACTATE_THRESHOLD).any()

        vasopressor_in_window = False
        if patientid in vaso_start['patientid'].values:
            start_hour = vaso_start[vaso_start['patientid'] == patientid]['vaso_start_hour'].iloc[0]
            vasopressor_in_window = prediction_start <= start_hour <= prediction_end

        target = int(future_lactate and (future_hypotension or vasopressor_in_window))
        row = grp.iloc[i].to_dict()
        row['target_circulatory_failure'] = target
        failure_events.append(row)

prediction_dataset = pd.DataFrame(failure_events)
positive_first = prediction_dataset[prediction_dataset['target_circulatory_failure'] == 1].groupby('patientid').first().reset_index()
all_negatives = prediction_dataset[prediction_dataset['target_circulatory_failure'] == 0]
prediction_dataset = pd.concat([all_negatives, positive_first], ignore_index=True).sort_values(['patientid', 'time_hour']).reset_index(drop=True)

prediction_dataset.to_csv(OUTPUT_DIR / "circulatory_failure_prediction_dataset.csv", index=False)

print(f"✓ Saved {len(prediction_dataset):,} rows")
print(f"   Positive events: {prediction_dataset['target_circulatory_failure'].sum():,} ({100*prediction_dataset['target_circulatory_failure'].mean():.1f}%)")
print(f"   Skipped (already failure): {excluded['already_failure']:,}")
print(f"   Skipped (no future data): {excluded['no_future_data']:,}")

conn.close()
print("\n✓ Circulatory failure preprocessing complete!")
