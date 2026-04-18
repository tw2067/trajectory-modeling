#!/usr/bin/env python
"""
MIMIC AKI Prediction Analysis with Trajectory Features

Run from workspace root:
    python scripts/mimic_aki_analysis.py
    python scripts/mimic_aki_analysis.py --cv-repeats 2 --cv-splits 3 --train-n-jobs 2
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from analysis.analysis_template import AnalysisConfig, main_cli


class MIMICAKIConfig(AnalysisConfig):
    """MIMIC AKI analysis configuration."""
    
    DATASET = "mimic"
    TASK = "aki"
    
    BIOMARKERS = {
        "creatinine": {
            "file": "creatinine_timeseries.csv",
            "value_col": "creatinine",
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
                "window": 7.0,
            },
        },
    }
    
    TARGET_COL = "target_aki_stage3"
    OUTCOME_FILE = "aki_outcomes.csv"
    OUTCOME_MERGE_COLS = ["hadm_id", "time_day", "target_aki_stage3"]
    
    CV_N_REPEATS = 10
    CV_N_SPLITS = 5
    LOOKBACK_WINDOW = 7
    LOOKBACK_UNIT = "days"


if __name__ == "__main__":
    main_cli(
        MIMICAKIConfig,
        description="MIMIC AKI Prediction with Trajectory Features",
    )
