"""
Bootstrap-based trajectory classification.

Fast alternative to Bayesian sampling that uses:
- Bootstrap resampling for uncertainty quantification
- Cubic splines for smooth trajectory fits
- Same classification logic as Bayesian backend

Typically 10-100x faster than MCMC with comparable trajectory probabilities.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from typing import Callable
from ..bayes.classify import flags_from_traj


def classify_trajectory_from_spline(
    spline: UnivariateSpline,
    time_grid: np.ndarray,
    flat_thr: float,
    decline_thr: float,
    nonlinear_gap: float,
    class_func: Callable,
    label_map: dict | None = None
) -> str:
    """
    Classify a spline trajectory into one of the trajectory types.
    
    Parameters
    ----------
    spline : UnivariateSpline
        Fitted spline function
    time_grid : array
        Time points for evaluation
    flat_thr, decline_thr, nonlinear_gap : float
        Classification thresholds
    class_func : callable
        Classification function (e.g., flags_from_traj, pos_flags_from_traj)
    label_map : dict, optional
        Label mapping for trajectory types
        
    Returns
    -------
    str
        Trajectory type: 'nonprogression', 'linear', or 'nonlinear'
    """
    # Evaluate spline on grid
    traj_values = spline(time_grid)
    
    # Get classification flags
    flags = class_func(
        traj_values,
        time_grid,
        flat_thr=flat_thr,
        decline_thr=decline_thr,
        nonlinear_gap=nonlinear_gap,
        label_map=label_map
    )
    
    # Determine dominant type (priority: nonlinear > linear > nonprogression)
    # Get the mapped labels
    if label_map:
        nonlin_label = label_map.get('nonlinear', 'nonlinear')
        linear_label = label_map.get('linear', 'linear_decline')
        nonprog_label = label_map.get('nonprogression', 'prolonged_nonprogression')
    else:
        nonlin_label = 'nonlinear'
        linear_label = 'linear_decline'
        nonprog_label = 'prolonged_nonprogression'
    
    if flags.get(nonlin_label, False):
        return 'nonlinear'
    elif flags.get(linear_label, False):
        return 'linear'
    else:
        return 'nonprogression'


def bootstrap_trajectory_probs(
    time: np.ndarray,
    values: np.ndarray,
    n_bootstrap: int,
    time_grid: np.ndarray,
    flat_thr: float,
    decline_thr: float,
    nonlinear_gap: float,
    smoothing: float,
    class_func: Callable,
    label_map: dict | None = None,
    random_state: int | None = None
) -> dict[str, float]:
    """
    Compute trajectory probabilities via bootstrap resampling.
    
    Parameters
    ----------
    time : array
        Time points (sorted)
    values : array
        Biomarker values
    n_bootstrap : int
        Number of bootstrap samples
    time_grid : array
        Grid for trajectory evaluation
    flat_thr, decline_thr, nonlinear_gap : float
        Classification thresholds
    smoothing : float
        Spline smoothing parameter (s parameter for UnivariateSpline)
    class_func : callable
        Classification function
    label_map : dict, optional
        Label mapping for trajectory types
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Trajectory probabilities: {'prob_stable': float, 'prob_linear': float, 'prob_nonlinear': float}
    """
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState()
    
    n_points = len(time)
    if n_points < 4:
        # Not enough points for bootstrap - return uniform probabilities
        return {
            'nonprogression': 1.0 / 3,
            'linear': 1.0 / 3,
            'nonlinear': 1.0 / 3
        }
    
    classifications = []
    
    for _ in range(n_bootstrap):
        # Bootstrap resample
        indices = rng.choice(n_points, size=n_points, replace=True)
        boot_time = time[indices]
        boot_values = values[indices]
        
        # Sort by time (spline requires monotonic x)
        sort_idx = np.argsort(boot_time)
        boot_time = boot_time[sort_idx]
        boot_values = boot_values[sort_idx]
        
        # Handle duplicate time points (take mean)
        unique_times, inverse_indices = np.unique(boot_time, return_inverse=True)
        if len(unique_times) < len(boot_time):
            # Average values at duplicate times
            unique_values = np.array([
                boot_values[inverse_indices == i].mean()
                for i in range(len(unique_times))
            ])
            boot_time = unique_times
            boot_values = unique_values
        
        if len(boot_time) < 4:
            # Not enough unique points - skip this bootstrap
            continue
        
        try:
            # Fit spline (k=3 for cubic, or less if not enough points)
            k = min(3, len(boot_time) - 1)
            spline = UnivariateSpline(boot_time, boot_values, k=k, s=smoothing)
            
            # Classify trajectory
            traj_type = classify_trajectory_from_spline(
                spline, time_grid, flat_thr, decline_thr, nonlinear_gap,
                class_func, label_map
            )
            classifications.append(traj_type)
            
        except Exception:
            # Spline fitting failed - skip this bootstrap
            continue
    
    if not classifications:
        # All bootstraps failed - return uniform
        return {
            'nonprogression': 1.0 / 3,
            'linear': 1.0 / 3,
            'nonlinear': 1.0 / 3
        }
    
    # Count classifications
    counts = pd.Series(classifications).value_counts()
    total = len(classifications)
    
    return {
        'nonprogression': counts.get('nonprogression', 0) / total,
        'linear': counts.get('linear', 0) / total,
        'nonlinear': counts.get('nonlinear', 0) / total
    }
