#!/usr/bin/env python3
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[4]
runpy.run_path(str(ROOT / "scripts" / "traj_scripts" / "hirid" / "hirid_extract_vitals_labs.py"), run_name="__main__")
