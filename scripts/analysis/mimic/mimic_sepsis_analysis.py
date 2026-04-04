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
from traj_features.backends.bayes.classify import pos_flags_from_traj, flags_from_traj


class MIMICSepsisConfig(AnalysisConfig):
    """MIMIC Sepsis analysis configuration."""
    
    DATASET = "mimic"
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
        "platelet": {
            "file": "platelet_timeseries.csv",
            "value_col": "platelet",
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
