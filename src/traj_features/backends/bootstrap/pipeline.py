"""
Bootstrap trajectory pipeline for parallel processing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Callable
from .classify import bootstrap_trajectory_probs


def process_patient_window(
    pid: int | str,
    window_id: int,
    df_window: pd.DataFrame,
    time_col: str,
    values_col: str,
    n_bootstrap: int,
    grid_freq: int,
    flat_thr: float,
    decline_thr: float,
    nonlinear_gap: float,
    smoothing: float,
    class_func: Callable,
    traj_types: tuple[str, ...],
    label_map: dict | None = None
) -> dict:
    """
    Process a single patient window with bootstrap resampling.
    
    Returns
    -------
    dict
        Results including patient ID, window ID, and trajectory probabilities
    """
    # Extract time and values
    time = df_window[time_col].values
    values = df_window[values_col].values
    
    # Create evaluation grid
    t_min, t_max = time.min(), time.max()
    time_grid = np.linspace(t_min, t_max, grid_freq)
    
    # Compute probabilities
    probs = bootstrap_trajectory_probs(
        time=time,
        values=values,
        n_bootstrap=n_bootstrap,
        time_grid=time_grid,
        flat_thr=flat_thr,
        decline_thr=decline_thr,
        nonlinear_gap=nonlinear_gap,
        smoothing=smoothing,
        class_func=class_func,
        label_map=label_map
    )
    
    # Map to trajectory type labels
    result = {
        'pid': pid,
        'window_id': window_id,
    }
    
    # Add probabilities with proper labels
    for generic, specific in (label_map or {}).items():
        if generic in probs:
            result[f'trajtype_{specific}_prob'] = probs[generic]
    
    # Fallback if no label_map
    if not label_map:
        result['trajtype_prolonged_nonprogression_prob'] = probs.get('nonprogression', 0.0)
        result['trajtype_linear_decline_prob'] = probs.get('linear', 0.0)
        result['trajtype_nonlinear_prob'] = probs.get('nonlinear', 0.0)
    
    return result


def compute_bootstrap_trajectory_covariates_parallel(
    lab_df: pd.DataFrame,
    window_years: float = 2.0,
    flat_thr: float = -1.0,
    decline_thr: float = -2.0,
    nonlinear_gap: float = 3.0,
    n_bootstrap: int = 500,
    smoothing: float = None,
    min_points_per_window: int = 5,
    grid_freq: int = 12,
    pids: str = "patient_id",
    values: str = "lab_value",
    time_col: str = "time",
    windowing_col: str = "time",
    n_jobs: int = -1,
    traj_types: tuple[str, ...] = ('prolonged_nonprogression', 'linear_decline', 'nonlinear'),
    class_func: Callable = None,
    label_map: dict | None = None,
    progressbar: bool = False
) -> pd.DataFrame:
    """
    Compute trajectory probabilities using bootstrap resampling in parallel.
    
    Parameters
    ----------
    lab_df : DataFrame
        Long-format lab data with columns [pids, time_col, values]
    window_years : float
        Lookback window size in same units as time_col
    flat_thr, decline_thr, nonlinear_gap : float
        Classification thresholds
    n_bootstrap : int
        Number of bootstrap resamples per window
    smoothing : float, optional
        Spline smoothing parameter (default: auto-select based on data)
    min_points_per_window : int
        Minimum observations required per window
    grid_freq : int
        Number of points for trajectory evaluation grid
    pids, values, time_col, windowing_col : str
        Column names
    n_jobs : int
        Number of parallel jobs
    traj_types : tuple
        Trajectory type labels
    class_func : callable
        Classification function (default: flags_from_traj)
    label_map : dict, optional
        Mapping of generic to specific labels
    progressbar : bool
        Show progress bar
        
    Returns
    -------
    DataFrame
        Trajectory probabilities per (patient, windowing_col)
    """
    if class_func is None:
        from ..bayes.classify import flags_from_traj
        class_func = flags_from_traj
    
    # Auto-select smoothing if not specified
    if smoothing is None:
        # Use cross-validation-like heuristic: smoothing ~ variance of residuals
        # A reasonable default is to allow some roughness but not overfit
        smoothing = None  # Let UnivariateSpline auto-select with s=None
    
    # Create windows
    lab_df = lab_df.sort_values(by=[pids, time_col])
    lab_df['_window_time'] = lab_df[windowing_col]

    # Round/floor to integer window ids using requested windowing column.
    # IMPORTANT: Keep the output key name aligned with `windowing_col`.
    lab_df['_window_id'] = np.floor(pd.to_numeric(lab_df['_window_time'], errors='coerce')).astype('Int64')
    lab_df = lab_df.dropna(subset=['_window_id'])
    lab_df['_window_id'] = lab_df['_window_id'].astype(int)
    
    # Define windows: for each patient-day, look back window_years
    windows = []
    for (pid, window_id), group in lab_df.groupby([pids, '_window_id']):
        window_start = window_id - window_years
        # Select observations within window
        window_data = lab_df[
            (lab_df[pids] == pid) &
            (lab_df['_window_time'] >= window_start) &
            (lab_df['_window_time'] <= window_id)
        ]
        
        if len(window_data) >= min_points_per_window:
            windows.append((pid, window_id, window_data))
    
    if not windows:
        # No valid windows - return empty DataFrame
        cols = [pids, windowing_col] + [f'trajtype_{t}_prob' for t in traj_types]
        return pd.DataFrame(columns=cols)
    
    # Process windows in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10 if progressbar else 0)(
        delayed(process_patient_window)(
            pid=pid,
            window_id=window_id,
            df_window=window_data,
            time_col=time_col,
            values_col=values,
            n_bootstrap=n_bootstrap,
            grid_freq=grid_freq,
            flat_thr=flat_thr,
            decline_thr=decline_thr,
            nonlinear_gap=nonlinear_gap,
            smoothing=smoothing,
            class_func=class_func,
            traj_types=traj_types,
            label_map=label_map
        )
        for pid, window_id, window_data in windows
    )
    
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    result_df = result_df.rename(columns={'pid': pids, 'window_id': windowing_col})
    
    return result_df
