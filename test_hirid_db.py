#!/usr/bin/env python
import duckdb
import pandas as pd

DB_PATH = '/home/gaga/data/physionet/HiRiD/hirid.duckdb'
conn = duckdb.connect(DB_PATH, read_only=True)

# Test: get observations table info
try:
    result = conn.execute("SELECT COUNT(*) FROM observations").fetchone()
    print(f"Total observations: {result[0]:,}")
except Exception as e:
    print(f"Error querying observations: {e}")

# Test: check what tables exist
try:
    tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='memory'").fetchall()
    print(f"Tables: {tables}")
except Exception as e:
    print(f"Error listing tables: {e}")

# Test: get sample row
try:
    sample = conn.execute("SELECT * FROM observations LIMIT 1").fetchall()
    print(f"Sample row: {sample}")
except Exception as e:
    print(f"Error getting sample: {e}")

conn.close()
