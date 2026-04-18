#!/usr/bin/env python
"""
eICU Sepsis Prediction Analysis with Trajectory Features

Run from workspace root:
    python scripts/eicu_sepsis_analysis.py
    python scripts/eicu_sepsis_analysis.py --cv-repeats 10 --train-n-jobs 2
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from analysis.analysis_template import AnalysisConfig, main_cli


class eICUSepsisConfig(AnalysisConfig):
    """eICU Sepsis analysis configuration."""
    
    DATASET = "eicu"
    TASK = "sepsis"
    
    BIOMARKERS = {
        "lactate": {
            "file": "lactate_timeseries.csv",
            "value_col": "lactate",
            "bootstrap": {
                "class_func": "pos",
                "flat_thr": 0.1,
                "decline_thr": 0.3,
                "nonlinear_gap": 0.5,
                "traj_types": ("stable", "gradual_increase", "rapid_increase"),
                "label_map": {
                    "nonprogression": "stable",
                    "linear": "gradual_increase",
                    "nonlinear": "rapid_increase",
                },
                "window": 3.0,
            },
        },
        "wbc": {
            "file": "wbc_timeseries.csv",
            "value_col": "wbc",
            "bootstrap": {
                "class_func": "pos",
                "flat_thr": 1.0,
                "decline_thr": 3.0,
                "nonlinear_gap": 2.0,
                "traj_types": ("stable", "gradual_increase", "rapid_increase"),
                "label_map": {
                    "nonprogression": "stable",
                    "linear": "gradual_increase",
                    "nonlinear": "rapid_increase",
                },
                "window": 3.0,
            },
        },
        "platelets": {
            "file": "platelets_timeseries.csv",
            "value_col": "platelet",
            "bootstrap": {
                "class_func": "neg",
                "flat_thr": -20.0,
                "decline_thr": -50.0,
                "nonlinear_gap": 30.0,
                "traj_types": ("stable", "gradual_decline", "rapid_decline"),
                "label_map": {
                    "nonprogression": "stable",
                    "linear": "gradual_decline",
                    "nonlinear": "rapid_decline",
                },
                "window": 3.0,
            },
        },
    }
    
    TARGET_COL = "target_septic_shock"
    OUTCOME_FILE = None
    OUTCOME_MERGE_COLS = []
    
    CV_N_REPEATS = 10
    CV_N_SPLITS = 5
    LOOKBACK_WINDOW = 3
    LOOKBACK_UNIT = "days"


if __name__ == "__main__":
    main_cli(
        eICUSepsisConfig,
        description="eICU Sepsis Prediction with Trajectory Features",
    )
