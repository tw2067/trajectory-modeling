"""
Shared analysis utilities for normalizing preprocessing, scaling, imputation across all trajectory analysis tasks.

Provides:
- Auto-detection of ID, time, and windowing columns
- Safe data loading (parquet-first, CSV fallback)
- Robust biomarker summary statistics (mean, max, min, trend, std)
- Safe merging with duplicate handling and validation
- Feature set preparation (baseline, trajectory, summary stats)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed as jdelayed

logger = logging.getLogger(__name__)


# ============================================================================
# Column Detection
# ============================================================================


def pick_id_col(df: pd.DataFrame) -> str:
    """Auto-detect ID column (hadm_id, stay_id, patientid)."""
    for col in ["hadm_id", "stay_id", "patientid"]:
        if col in df.columns:
            return col
    raise ValueError(f"No ID column found in {list(df.columns)}")


def pick_time_cols(df: pd.DataFrame) -> tuple[str, str]:
    """
    Auto-detect time columns.
    Returns: (continuous_time_col, windowing_col)
    
    Examples:
    - (time_hours, time_hour)
    - (time_days, time_day)
    """
    if "time_hours" in df.columns and "time_hour" in df.columns:
        return "time_hours", "time_hour"
    if "time_days" in df.columns and "time_day" in df.columns:
        return "time_days", "time_day"
    if "time_hour" in df.columns:
        return "time_hour", "time_hour"
    if "time_day" in df.columns:
        return "time_day", "time_day"
    raise ValueError(f"No time columns found in {list(df.columns)}")


# ============================================================================
# Data Loading
# ============================================================================


def read_table(path: Path) -> pd.DataFrame:
    """Read table (parquet-first, fallback to CSV)."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_named_table(base_dir: Path, stem_name: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load named table (parquet-first fallback to CSV).
    
    Args:
        base_dir: Directory containing the file
        stem_name: Base name without extension (e.g., 'lactate_timeseries')
        verbose: Log which file was loaded
    
    Returns:
        DataFrame
    
    Raises:
        FileNotFoundError: If neither parquet nor CSV exists
    """
    base_dir = Path(base_dir)
    candidates = [base_dir / f"{stem_name}.parquet", base_dir / f"{stem_name}.csv"]
    
    for p in candidates:
        if p.exists():
            if verbose:
                logger.info(f"  Loading: {p}")
            return read_table(p)
    
    raise FileNotFoundError(f"Missing table for {stem_name}. Tried: {candidates}")


# ============================================================================
# Safe Merging
# ============================================================================


def safe_left_merge(
    base_df: pd.DataFrame,
    add_df: pd.DataFrame,
    key_cols: list[str],
    add_name: str = "add_df",
) -> pd.DataFrame:
    """
    Safely merge two datasets, handling duplicate keys by aggregating on probability columns.
    
    Args:
        base_df: Left (base) dataframe
        add_df: Right (addition) dataframe to merge
        key_cols: Key columns for merge
        add_name: Name for logging
    
    Returns:
        Merged DataFrame with validate='one_to_one'
    """
    base_dup = int(base_df.duplicated(subset=key_cols).sum())
    add_dup = int(add_df.duplicated(subset=key_cols).sum())

    if base_dup > 0:
        logger.warning(f"base dataset has {base_dup:,} duplicate keys; keeping first")
        base_df = base_df.drop_duplicates(subset=key_cols, keep="first")

    if add_dup > 0:
        prob_cols = [c for c in add_df.columns if c not in key_cols]
        logger.warning(f"{add_name} has {add_dup:,} duplicate keys; aggregating means")
        add_df = add_df.groupby(key_cols, as_index=False)[prob_cols].mean()

    return base_df.merge(add_df, on=key_cols, how="left", validate="one_to_one")


# ============================================================================
# Biomarker Summary Statistics
# ============================================================================


def _patient_window_stats(
    id_val,
    pid_df: pd.DataFrame,
    time_col: str,
    windowing_col: str,
    value_col: str,
    lookback_hours: float,
) -> list[tuple]:
    """
    Compute windowed summary statistics for a single patient.
    
    Returns list of:
        (id, windowing_val, mean, max, min, change, trend, std)
    """
    pid_df = pid_df.sort_values(time_col)
    times = pd.to_numeric(pid_df[time_col], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(pid_df[value_col], errors="coerce").to_numpy(dtype=float)
    windows = pd.to_numeric(pid_df[windowing_col], errors="coerce").to_numpy(dtype=float)

    keep = np.isfinite(times) & np.isfinite(values) & np.isfinite(windows)
    times = times[keep]
    values = values[keep]
    windows = windows[keep]

    unique_windows = np.unique(windows)
    rows = []

    for w in unique_windows:
        mask = (times >= w - lookback_hours) & (times <= w)
        wv = values[mask]
        wt = times[mask]
        n = len(wv)

        if n == 0:
            rows.append((id_val, w, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue

        trend = 0.0
        if n > 1:
            x = wt[np.isfinite(wt) & np.isfinite(wv)]
            y = wv[np.isfinite(wt) & np.isfinite(wv)]
            if len(x) > 1 and np.unique(x).size > 1:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", np.exceptions.RankWarning)
                        trend = float(np.polyfit(x, y, 1)[0])
                except Exception:
                    trend = 0.0

        rows.append(
            (
                id_val,
                w,
                float(np.mean(wv)),
                float(np.max(wv)),
                float(np.min(wv)),
                float(wv[-1] - wv[0]) if n > 0 else np.nan,
                trend,
                float(np.std(wv)) if n > 1 else 0.0,
            )
        )

    return rows


def biomarker_summary_stats(
    ts_df: pd.DataFrame,
    value_col: str,
    lookback_hours: float = 12,
    n_jobs: int = 4,
) -> pd.DataFrame:
    """
    Compute windowed summary statistics for a biomarker time series.
    
    Args:
        ts_df: Time series dataframe with columns [id_col, time_col, windowing_col, value_col]
        value_col: Name of the value column to summarize
        lookback_hours: Hours to look back from each window
        n_jobs: Number of parallel jobs
    
    Returns:
        DataFrame with columns:
            [id_col, windowing_col, {value_col}_mean, _max, _min, _change, _trend, _std]
    """
    # Auto-detect columns
    id_col = pick_id_col(ts_df)
    time_col, windowing_col = pick_time_cols(ts_df)
    
    # Ensure numeric
    ts_df = ts_df.copy()
    ts_df[time_col] = pd.to_numeric(ts_df[time_col], errors="coerce")
    ts_df[windowing_col] = pd.to_numeric(ts_df[windowing_col], errors="coerce")
    ts_df[value_col] = pd.to_numeric(ts_df[value_col], errors="coerce")
    
    # Drop NaN
    ts_df = ts_df.dropna(subset=[time_col, windowing_col, value_col])
    
    if len(ts_df) == 0:
        logger.warning(f"Empty time series after cleaning")
        return pd.DataFrame(columns=[id_col, windowing_col])
    
    # Compute per-patient
    all_rows = Parallel(n_jobs=n_jobs)(
        jdelayed(_patient_window_stats)(
            pid,
            pid_df,
            time_col,
            windowing_col,
            value_col,
            lookback_hours,
        )
        for pid, pid_df in ts_df.groupby(id_col)
    )
    
    # Flatten
    flat_rows = [row for rows in all_rows for row in rows]
    
    # Convert to DataFrame
    result = pd.DataFrame(
        flat_rows,
        columns=[
            id_col,
            windowing_col,
            f"{value_col}_mean",
            f"{value_col}_max",
            f"{value_col}_min",
            f"{value_col}_change",
            f"{value_col}_trend",
            f"{value_col}_std",
        ],
    )
    
    # Convert to numeric and handle infinities
    for col in result.columns:
        if col not in [id_col, windowing_col]:
            result[col] = pd.to_numeric(result[col], errors="coerce")
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)
    
    return result


# ============================================================================
# Feature Set Organization
# ============================================================================


def organize_feature_sets(df: pd.DataFrame, id_col: str, windowing_col: str) -> dict[str, list[str]]:
    """
    Auto-organize features into baseline, trajectory, summary stats categories.
    
    Args:
        df: Feature dataframe
        id_col: ID column name
        windowing_col: Windowing column name
    
    Returns:
        Dictionary with keys:
        - 'trajectory': trajectory probability columns
        - 'summary': summary statistic columns
        - 'baseline': baseline/static features
        - 'all_numeric': all numeric columns (excluding ID, windowing, outcome)
    """
    exclude_cols = {
        id_col, windowing_col,
        "charttime", "admittime", "dischtime", "subject_id",
        "los_days", "hospital_expire_flag", "in_hospital_mortality",
        "icu_expire_flag", "icu_mortality",
        "target_septic_shock", "target_aclf", "target_aki", "target_ventilator",
        "outcome", "label", "y",
    }
    
    numeric_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and not c.startswith("target_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    
    trajectory_cols = [
        c for c in numeric_cols
        if c.endswith("_stable")
        or c.endswith("_gradual")
        or c.endswith("_rapid")
        or c.endswith("_increase")
        or c.endswith("_decline")
        or c.startswith("prob_")
        or c.startswith("trajtype_")
        or c.endswith("_prob")
        or "_boot" in c
    ]
    
    summary_cols = [
        c for c in numeric_cols
        if c.endswith("_mean")
        or c.endswith("_max")
        or c.endswith("_min")
        or c.endswith("_change")
        or c.endswith("_trend")
        or c.endswith("_std")
        or "_5d" in c
        or "_10d" in c
    ]
    
    baseline_cols = [
        c for c in numeric_cols
        if c not in trajectory_cols
        and c not in summary_cols
    ]
    
    # Further separate static vs dynamic in baseline
    static_tokens = {
        "age", "gender", "sex", "baseline", "admit", "admission",
        "ethnicity", "race", "height", "weight", "bmi", "baseline_",
    }
    static_cols = [
        c for c in baseline_cols
        if any(tok in c.lower() for tok in static_tokens)
    ]
    dynamic_cols = [c for c in baseline_cols if c not in static_cols]
    
    return {
        "trajectory": trajectory_cols,
        "summary": summary_cols,
        "baseline": baseline_cols,
        "static": static_cols,
        "dynamic": dynamic_cols,
        "all_numeric": numeric_cols,
    }


# ============================================================================
# Model & Data Configuration
# ============================================================================


def get_dataset_config(dataset: str) -> dict:
    """
    Get dataset-specific configuration (ID col, paths, etc.).
    
    Args:
        dataset: 'mimic', 'hirid', or 'eicu'
    
    Returns:
        Dictionary with keys: id_col, time_col_base (hours/days), base_path
    """
    configs = {
        "mimic": {
            "id_col": "hadm_id",
            "time_unit": "hours",  # MIMIC uses hours
            "base_path": Path("/home/gaga/data/physionet/mimic"),
        },
        "hirid": {
            "id_col": "patientid",
            "time_unit": "hours",  # HiRiD uses hours
            "base_path": Path("/home/gaga/data/physionet/hirid"),
        },
        "eicu": {
            "id_col": "stay_id",
            "time_unit": "days",  # eICU uses days
            "base_path": Path("/home/gaga/data/physionet/eicu"),
        },
    }
    
    if dataset not in configs:
        raise ValueError(f"Unknown dataset: {dataset}. Choices: {list(configs.keys())}")
    
    return configs[dataset]


# ============================================================================
# Model Training Utilities
# ============================================================================


def get_default_models(include_boosting: bool = True) -> dict:
    """
    Get default scikit-learn models for comparison.
    
    Args:
        include_boosting: Whether to include XGBoost and HistGradientBoosting
    
    Returns:
        Dictionary of {name: model_class}
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=920 , class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=920, n_jobs=-1, class_weight="balanced"),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=920, class_weight="balanced"),
    }
    
    if include_boosting:
        try:
            from xgboost import XGBClassifier
            models["XGBoost"] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=920,
            )
        except ImportError:
            pass
    
    return models
