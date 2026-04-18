#!/usr/bin/env python3
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[4]
runpy.run_path(str(ROOT / "scripts" / "mimic_preprocessing" / "sepsis_preprocess.py"), run_name="__main__")
