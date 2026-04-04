#!/usr/bin/env python
"""
HiRiD Sepsis Prediction Analysis with Trajectory Features and Joblib Parallelization

Run from workspace root:
    python scripts/hirid_sepsis_analysis.py
    python scripts/hirid_sepsis_analysis.py --cv-repeats 10 --train-n-jobs 2 --train-backend threads
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from analysis.analysis_template import AnalysisConfig, main_cli


class HiRiDSepsisConfig(AnalysisConfig):
    """HiRiD Sepsis analysis configuration."""
    
    DATASET = "hirid"
    TASK = "sepsis"
    
    BIOMARKERS = {
        "lactate": {
            "file": "lactate_timeseries.csv",
            "value_col": "lactate",
        },
        "wbc": {
            "file": "wbc_timeseries.csv",
            "value_col": "wbc",
        },
        "platelets": {
            "file": "platelets_timeseries.csv",
            "value_col": "platelets",
        },
    }
    
    TARGET_COL = "target_septic_shock"
    OUTCOME_FILE = "septic_shock_outcomes.csv"
    OUTCOME_MERGE_COLS = ["patientid", "time_hour", "target_septic_shock"]
    
    CV_N_REPEATS = 10
    CV_N_SPLITS = 5
    LOOKAHEAD_HOURS = 24


if __name__ == "__main__":
    main_cli(
        HiRiDSepsisConfig,
        description="HiRiD Sepsis Prediction with Trajectory Features (parallelized)",
    )
