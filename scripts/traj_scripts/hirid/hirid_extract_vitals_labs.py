"""
Helper script to extract and prepare HiRiD vitals and labs data.

This creates processed dataframes similar to MIMIC preprocessing.
"""

import pandas as pd
import numpy as np
import duckdb
import os

print("="*80)
print("HiRiD Vitals & Labs Extraction")
print("="*80)

# Connect to database
db_path = '/home/gaga/data/physionet/HiRiD/hirid.duckdb'
conn = duckdb.connect(db_path, read_only=True)

# Load patient cohort
patient_final = pd.read_csv('../../results/hirid/aki/patient_cohort.csv')
patient_ids = patient_final['patientid'].tolist()

print(f"\n📊 Cohort: {len(patient_ids):,} patients")

# ============================================================================
# VITAL SIGNS
# ============================================================================

vital_variables = {
    200: 'HeartRate',
    210: 'SysBP',
    220: 'DiasBP',
    300: 'SpO2',
    310: 'RespRate',
    330: 'TempC',
}

print(f"\n💓 Loading vital signs...")
vitals_query = f"""
SELECT 
    o.patientid,
    o.datetime as charttime,
    o.variableid,
    CAST(o.value AS DOUBLE) as valuenum,
    p.admission_time as admittime
FROM observations o
INNER JOIN patient_info p ON o.patientid = p.patientid
WHERE 
    o.variableid IN {tuple(vital_variables.keys())}
    AND o.value IS NOT NULL
    AND o.patientid IN {tuple(patient_ids)}
ORDER BY o.patientid, o.datetime
"""

vitals_df = conn.execute(vitals_query).fetchdf()
vitals_df['feature name'] = vitals_df['variableid'].map(vital_variables)

# Add reasonable ranges (similar to MIMIC vital_metadata)
ranges = {
    'HeartRate': (20, 300),
    'SysBP': (40, 300),
    'DiasBP': (20, 200),
    'SpO2': (50, 100),
    'RespRate': (4, 60),
    'TempC': (30, 45)
}

vitals_df['min'] = vitals_df['feature name'].map(lambda x: ranges[x][0])
vitals_df['max'] = vitals_df['feature name'].map(lambda x: ranges[x][1])

# Filter outliers
vitals_df = vitals_df[
    vitals_df['valuenum'].between(vitals_df['min'], vitals_df['max'])
]

print(f"   ✓ Loaded {len(vitals_df):,} vital sign measurements")
for var_name, count in vitals_df['feature name'].value_counts().items():
    print(f"      {var_name:15s}: {count:,}")

# ============================================================================
# LABORATORY VALUES
# ============================================================================

lab_variables = {
    20000350: 'CREATININE',
    20000400: 'BUN',  # Urea
    20000450: 'POTASSIUM',
    20001300: 'SODIUM',
    20001600: 'LACTATE',
    20000500: 'GLUCOSE',
    20000550: 'CHLORIDE',
    20000600: 'BICARBONATE',
    20002000: 'HEMOGLOBIN',
    20002050: 'HEMATOCRIT',
    20002100: 'WBC',
    20002150: 'PLATELET',
}

print(f"\n🔬 Loading laboratory values...")
labs_query = f"""
SELECT 
    o.patientid,
    o.datetime as charttime,
    o.variableid,
    CAST(o.value AS DOUBLE) as valuenum,
    p.admission_time as admittime
FROM observations o
INNER JOIN patient_info p ON o.patientid = p.patientid
WHERE 
    o.variableid IN {tuple(lab_variables.keys())}
    AND o.value IS NOT NULL
    AND o.patientid IN {tuple(patient_ids)}
ORDER BY o.patientid, o.datetime
"""

labs_df = conn.execute(labs_query).fetchdf()
labs_df['feature name'] = labs_df['variableid'].map(lab_variables)

# Unit conversions (HiRiD uses µmol/L, mmol/L, etc.)
# Convert to MIMIC-like units for consistency
conversions = {
    'CREATININE': lambda x: x / 88.4,  # µmol/L → mg/dL
    'BUN': lambda x: x * 2.8,          # mmol/L → mg/dL
    # Others are already in compatible units
}

for lab, convert_fn in conversions.items():
    mask = labs_df['feature name'] == lab
    if mask.any():
        labs_df.loc[mask, 'valuenum'] = labs_df.loc[mask, 'valuenum'].apply(convert_fn)

# Add reasonable ranges
lab_ranges = {
    'CREATININE': (0.1, 20),
    'BUN': (1, 200),
    'POTASSIUM': (2.0, 8.0),
    'SODIUM': (100, 180),
    'LACTATE': (0.1, 20),
    'GLUCOSE': (20, 1000),
    'CHLORIDE': (70, 140),
    'BICARBONATE': (5, 50),
    'HEMOGLOBIN': (3, 20),
    'HEMATOCRIT': (10, 70),
    'WBC': (0.1, 100),
    'PLATELET': (10, 2000),
}

labs_df['min'] = labs_df['feature name'].map(lambda x: lab_ranges.get(x, (-np.inf, np.inf))[0])
labs_df['max'] = labs_df['feature name'].map(lambda x: lab_ranges.get(x, (-np.inf, np.inf))[1])

# Filter outliers
labs_df = labs_df[
    labs_df['valuenum'].between(labs_df['min'], labs_df['max'])
]

print(f"   ✓ Loaded {len(labs_df):,} laboratory measurements")
for var_name, count in labs_df['feature name'].value_counts().head(10).items():
    print(f"      {var_name:15s}: {count:,}")

# ============================================================================
# SAVE
# ============================================================================

os.makedirs('../../results/hirid/aki', exist_ok=True)

vitals_df.to_csv('../../results/hirid/aki/vitals_raw.csv', index=False)
labs_df.to_csv('../../results/hirid/aki/labs_raw.csv', index=False)

print(f"\n💾 Saved:")
print(f"   ../../results/hirid/aki/vitals_raw.csv")
print(f"   ../../results/hirid/aki/labs_raw.csv")

# Create metadata files for compatibility with shared utilities
vital_meta = pd.DataFrame([
    {
        'itemid': var_id,
        'feature name': var_name,
        'min': ranges[var_name][0],
        'max': ranges[var_name][1],
        'units': 'varies'
    }
    for var_id, var_name in vital_variables.items()
])

lab_meta = pd.DataFrame([
    {
        'itemid': var_id,
        'feature name': var_name,
        'min': lab_ranges[var_name][0],
        'max': lab_ranges[var_name][1],
        'units': 'varies'
    }
    for var_id, var_name in lab_variables.items()
])

vital_meta.to_csv('/home/gaga/data/physionet/HiRiD/vital_metadata.csv', index=False)
lab_meta.to_csv('/home/gaga/data/physionet/HiRiD/labs_metadata.csv', index=False)

print(f"   /home/gaga/data/physionet/HiRiD/vital_metadata.csv")
print(f"   /home/gaga/data/physionet/HiRiD/labs_metadata.csv")

print("\n✅ Done! You can now use these with aggregate_vitals_labs() utility")
print("="*80)
