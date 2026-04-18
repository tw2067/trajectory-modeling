#!/usr/bin/env python
"""
MIMIC Ventilator Duration Prediction Analysis with Trajectory Features

Run from workspace root:
    python scripts/mimic_ventilator_analysis.py
    python scripts/mimic_ventilator_analysis.py --cv-repeats 2 --cv-splits 3 --train-n-jobs 2
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from analysis.analysis_template import AnalysisConfig, main_cli


class MIMICVentilatorConfig(AnalysisConfig):
    """MIMIC Ventilator prediction analysis configuration."""
    
    DATASET = "mimic"
    TASK = "ventilator"
    
    BIOMARKERS = {
        "pf_ratio": {
            "file": "pf_ratio_timeseries.csv",
            "value_col": "pf_ratio",
            "bootstrap": {
                "class_func": "pos",
                "flat_thr": 10.0,
                "decline_thr": 30.0,
                "nonlinear_gap": 20.0,
                "traj_types": ("prolonged_nonprogression", "linear_decline", "nonlinear"),
                "label_map": {
                    "nonprogression": "prolonged_nonprogression",
                    "linear": "linear_decline",
                    "nonlinear": "nonlinear",
                },
                "window": 3.0,
            },
        },
    }
    
    TARGET_COL = "target_weaning_success"
    OUTCOME_FILE = "ventilator_outcomes.csv"
    OUTCOME_MERGE_COLS = ["hadm_id", "time_day", "target_weaning_success"]
    
    CV_N_REPEATS = 10
    CV_N_SPLITS = 5
    LOOKBACK_WINDOW = 3
    LOOKBACK_UNIT = "days"


if __name__ == "__main__":
    main_cli(
        MIMICVentilatorConfig,
        description="MIMIC Ventilator Duration Prediction with Trajectory Features",
    )
