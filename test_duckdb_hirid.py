#!/usr/bin/env python
import duckdb
import time

DB_PATH = '/home/gaga/data/physionet/HiRiD/hirid.duckdb'
print(f"Connecting to {DB_PATH}...")
start = time.time()
conn = duckdb.connect(DB_PATH, read_only=True)
print(f"Connected in {time.time() - start:.1f}s")

# Try a simple count query
print("\nTesting simple query...")
start = time.time()
result = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
print(f"Count: {result:,} (took {time.time() - start:.1f}s)")

# Try with WHERE clause
print("\nTesting query with WHERE variableid IN...")
start = time.time()
result = conn.execute("""
SELECT COUNT(*) FROM observations
WHERE variableid IN (200, 100, 600, 120, 620, 110, 610, 24000524, 24000732, 24000485)
""").fetchone()[0]
print(f"Count: {result:,} (took {time.time() - start:.1f}s)")

# Try a small fetch
print("\nTesting small fetch...")
start = time.time()
result = conn.execute("""
SELECT 
    o.patientid,
    o.datetime as charttime,
    o.variableid,
    CAST(o.value AS DOUBLE) as valuenum,
    p.admission_time as admittime
FROM observations o
INNER JOIN patient_info p ON o.patientid = p.patientid
WHERE 
    o.variableid IN (200, 100, 600, 120, 620, 110, 610, 24000524, 24000732, 24000485)
    AND o.value IS NOT NULL
    AND p.los_days >= 1.0
LIMIT 100
""").fetchdf()
print(f"Fetched: {len(result):,} rows (took {time.time() - start:.1f}s)")
print(result.head())

conn.close()
