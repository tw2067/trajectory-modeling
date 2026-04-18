#!/usr/bin/env python
"""
HiRiD Ventilator Duration Prediction Analysis with Trajectory Features

Run from workspace root:
    python scripts/hirid_ventilator_analysis.py
    python scripts/hirid_ventilator_analysis.py --cv-repeats 10 --train-n-jobs 2 --train-backend threads
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from analysis.analysis_template import AnalysisConfig, main_cli


class HiRiDVentilatorConfig(AnalysisConfig):
    """HiRiD Ventilator prediction analysis configuration."""
    
    DATASET = "hirid"
    TASK = "ventilator"
    
    BIOMARKERS = {
        "pao2": {
            "file": "pao2_timeseries.csv",
            "value_col": "pao2",
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
                "window": 24.0,
            },
        },
        "peep": {
            "file": "peep_timeseries.csv",
            "value_col": "peep",
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
                "window": 24.0,
            },
        },
    }
    
    TARGET_COL = "target_vent_duration"
    OUTCOME_FILE = "ventilator_outcomes.csv"
    OUTCOME_MERGE_COLS = ["patientid", "time_hour", "target_vent_duration"]
    
    CV_N_REPEATS = 10
    CV_N_SPLITS = 5
    LOOKAHEAD_HOURS = 24


if __name__ == "__main__":
    main_cli(
        HiRiDVentilatorConfig,
        description="HiRiD Ventilator Duration Prediction with Trajectory Features",
    )
