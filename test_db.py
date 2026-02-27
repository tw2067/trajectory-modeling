#!/usr/bin/env python
import duckdb
import pandas as pd

DB_PATH = '/home/gaga/data/physionet/HiRiD/hirid.duckdb'
conn = duckdb.connect(DB_PATH, read_only=True)

# Just check if the observations table has data
total_count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
print(f"Total observations: {total_count:,}")

# Check for our specific variables
our_vars = (110, 610, 100, 600, 120, 620, 200, 24000524, 24000732, 24000485)
var_count = conn.execute(f"SELECT COUNT(*) FROM observations WHERE variableid IN {our_vars}").fetchone()[0]
print(f"Our variables: {var_count:,}")

# Check if any have non-null values
null_count = conn.execute(f"SELECT COUNT(*) FROM observations WHERE variableid IN {our_vars} AND value IS NULL").fetchone()[0]
print(f"Null values: {null_count:,}")
print(f"Non-null values: {var_count - null_count:,}")

conn.close()
