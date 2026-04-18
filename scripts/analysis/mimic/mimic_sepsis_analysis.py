#!/usr/bin/env python
"""
MIMIC Sepsis Prediction Analysis with Trajectory Features

Run from workspace root:
    python scripts/mimic_sepsis_analysis.py
    python scripts/mimic_sepsis_analysis.py --cv-repeats 2 --cv-splits 3 --train-n-jobs 2
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from analysis.analysis_template import AnalysisConfig, main_cli


class MIMICSepsisConfig(AnalysisConfig):
    """MIMIC Sepsis analysis configuration."""
    
    DATASET = "mimic"
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
        "platelet": {
            "file": "platelet_timeseries.csv",
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
    OUTCOME_FILE = "septic_shock_outcomes.csv"
    OUTCOME_MERGE_COLS = ["hadm_id", "time_day", "target_septic_shock"]
    
    CV_N_REPEATS = 10
    CV_N_SPLITS = 5
    LOOKBACK_WINDOW = 3
    LOOKBACK_UNIT = "days"


if __name__ == "__main__":
    main_cli(
        MIMICSepsisConfig,
        description="MIMIC Sepsis Shock Prediction with Trajectory Features",
    )
