#!/usr/bin/env python
"""
eICU Liver ACLF Prediction Analysis with Trajectory Features

Run from workspace root:
    python scripts/eicu_liver_analysis.py
    python scripts/eicu_liver_analysis.py --cv-repeats 10 --train-n-jobs 2
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from analysis.analysis_template import AnalysisConfig, main_cli


class eICULiverConfig(AnalysisConfig):
    """eICU Liver ACLF analysis configuration."""
    
    DATASET = "eicu"
    TASK = "liver"
    
    BIOMARKERS = {
        "bilirubin": {
            "file": "bilirubin_timeseries.csv",
            "value_col": "bilirubin",
            "bootstrap": {
                "class_func": "pos",
                "flat_thr": 0.5,
                "decline_thr": 1.0,
                "nonlinear_gap": 3.0,
                "traj_types": ("prolonged_nonprogression", "linear_decline", "nonlinear"),
                "label_map": {
                    "nonprogression": "prolonged_nonprogression",
                    "linear": "linear_decline",
                    "nonlinear": "nonlinear",
                },
                "window": 5.0,
            },
        },
    }
    
    TARGET_COL = "target_aclf"
    OUTCOME_FILE = "aclf_outcomes.csv"
    OUTCOME_MERGE_COLS = ["stay_id", "time_day", "target_aclf"]
    
    CV_N_REPEATS = 10
    CV_N_SPLITS = 5
    LOOKBACK_WINDOW = 5
    LOOKBACK_UNIT = "days"


if __name__ == "__main__":
    main_cli(
        eICULiverConfig,
        description="eICU Liver ACLF Prediction with Trajectory Features",
    )
