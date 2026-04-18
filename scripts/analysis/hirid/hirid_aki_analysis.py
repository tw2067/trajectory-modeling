#!/usr/bin/env python
"""
HiRiD AKI Prediction Analysis with Trajectory Features

Run from workspace root:
    python scripts/hirid_aki_analysis.py
    python scripts/hirid_aki_analysis.py --cv-repeats 10 --train-n-jobs 2 --train-backend threads
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from analysis.analysis_template import AnalysisConfig, main_cli


class HiRiDAKIConfig(AnalysisConfig):
    """HiRiD AKI analysis configuration."""
    
    DATASET = "hirid"
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
                "window": 24.0,
            },
        },
    }
    
    TARGET_COL = "target_aki"
    OUTCOME_FILE = "aki_outcomes.csv"
    OUTCOME_MERGE_COLS = ["patientid", "time_hour", "target_aki"]
    
    CV_N_REPEATS = 10
    CV_N_SPLITS = 5
    LOOKAHEAD_HOURS = 24


if __name__ == "__main__":
    main_cli(
        HiRiDAKIConfig,
        description="HiRiD AKI Prediction with Trajectory Features",
    )
