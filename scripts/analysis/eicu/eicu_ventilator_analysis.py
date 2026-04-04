#!/usr/bin/env python
"""
eICU Ventilator Duration Prediction Analysis with Trajectory Features

Run from workspace root:
    python scripts/eicu_ventilator_analysis.py
    python scripts/eicu_ventilator_analysis.py --cv-repeats 10 --train-n-jobs 2
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from analysis.analysis_template import AnalysisConfig, main_cli


class eICUVentilatorConfig(AnalysisConfig):
    """eICU Ventilator prediction analysis configuration."""
    
    DATASET = "eicu"
    TASK = "ventilator"
    
    BIOMARKERS = {
        "pf_ratio": {
            "file": "pf_ratio_timeseries.csv",
            "value_col": "pf_ratio",
        },
    }
    
    TARGET_COL = "target_weaning_success"
    OUTCOME_FILE = None
    OUTCOME_MERGE_COLS = []
    
    CV_N_REPEATS = 10
    CV_N_SPLITS = 5
    LOOKBACK_WINDOW = 3
    LOOKBACK_UNIT = "days"


if __name__ == "__main__":
    main_cli(
        eICUVentilatorConfig,
        description="eICU Ventilator Duration Prediction with Trajectory Features",
    )
