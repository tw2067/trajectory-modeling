#!/usr/bin/env python
"""
HiRiD Liver ACLF Prediction Analysis with Trajectory Features

Run from workspace root:
    python scripts/hirid_liver_analysis.py
    python scripts/hirid_liver_analysis.py --cv-repeats 10 --train-n-jobs 2 --train-backend threads
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from analysis.analysis_template import AnalysisConfig, main_cli


class HiRiDLiverConfig(AnalysisConfig):
    """HiRiD Liver ACLF analysis configuration."""
    
    DATASET = "hirid"
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
                "window": 24.0,
            },
        },
    }
    
    TARGET_COL = "target_aclf"
    OUTCOME_FILE = "aclf_outcomes.csv"
    OUTCOME_MERGE_COLS = ["patientid", "time_hour", "target_aclf"]
    
    CV_N_REPEATS = 10
    CV_N_SPLITS = 5
    LOOKAHEAD_HOURS = 24


if __name__ == "__main__":
    main_cli(
        HiRiDLiverConfig,
        description="HiRiD Liver ACLF Prediction with Trajectory Features",
    )
