#!/usr/bin/env python3
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[4]
runpy.run_path(str(ROOT / "scripts" / "traj_scripts" / "eicu" / "sepsis_all_biomarkers.py"), run_name="__main__")
