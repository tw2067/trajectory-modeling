#!/usr/bin/env python
"""
eICU AKI Prediction Analysis with Trajectory Features

Run from workspace root:
    python scripts/eicu_aki_analysis.py
    python scripts/eicu_aki_analysis.py --cv-repeats 10 --train-n-jobs 2
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

from analysis.analysis_template import AnalysisConfig, main_cli


class eICUAKIConfig(AnalysisConfig):
    """eICU AKI analysis configuration."""
    
    DATASET = "eicu"
    TASK = "aki"
    
    BIOMARKERS = {
        "creatinine": {
            "file": "creatinine_timeseries.csv",
            "value_col": "creatinine",
        },
    }
    
    TARGET_COL = "target_aki_stage3"
    OUTCOME_FILE = None
    OUTCOME_MERGE_COLS = []
    
    CV_N_REPEATS = 10
    CV_N_SPLITS = 5
    LOOKBACK_WINDOW = 7
    LOOKBACK_UNIT = "days"


if __name__ == "__main__":
    main_cli(
        eICUAKIConfig,
        description="eICU AKI Prediction with Trajectory Features",
    )
