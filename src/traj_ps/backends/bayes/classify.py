from __future__ import annotations
import numpy as np
from collections.abc import Callable, Iterable

def flags_from_traj(traj, time_grid, flat_thr=-1, decline_thr=-2, nonlinear_gap=3, label_map=None):
    """
    Classify trajectory into types using thresholds.
    
    Parameters
    ----------
    label_map : dict, optional
        Maps generic labels to disease-specific labels:
        {'nonprogression': 'prolonged_nonprogression', 'linear': 'linear_decline', 'nonlinear': 'nonlinear'}
    
    Returns
    -------
    dict
        Dictionary with disease-specific labels as keys, boolean flags as values
    """
    dt = np.diff(time_grid)
    dy = np.diff(traj)
    slopes = dy / dt

    # prolonged nonprogression: mostly flat and average slope of non-progression
    frac_flat = (slopes >= decline_thr).mean()
    total_flat = np.mean(slopes) >= flat_thr

    # linear decline: mostly steeply negative
    frac_decl = (slopes < decline_thr).mean()

    # nonlinear: mean slope of faster half vs slower half differs by > nonlinear_gap
    s = np.sort(slopes)                       # ascending (more negative first)
    half = len(s) // 2
    fast = s[:half].mean()                    # faster decline half (more negative)
    slow = s[half:].mean()                    # slower half
    nonlinear = abs(fast - slow) > nonlinear_gap

    # Generic flags (internal logic)
    generic_flags = {
        'nonprogression': (frac_flat >= 0.8) & total_flat,
        'linear':         frac_decl >= 0.8,
        'nonlinear':      nonlinear
    }

    # Apply label mapping if provided
    if label_map is None:
        # Default CKD mapping
        label_map = {
            'nonprogression': 'prolonged_nonprogression',
            'linear': 'linear_decline',
            'nonlinear': 'nonlinear'
        }
    
    # Return disease-specific labels
    return {label_map[k]: v for k, v in generic_flags.items()}


def pos_flags_from_traj(traj, time_grid, flat_thr=0.5, decline_thr=1, nonlinear_gap=2, label_map=None):
    """
    Classify positive-slope trajectory (increasing values over time).
    
    This function works for ANY increasing trajectory, regardless of whether
    increasing is "good" (e.g., CD4 count recovery) or "bad" (e.g., MDS-UPDRS-III worsening).
    
    The interpretation is determined by the label_map parameter:
    - For HIV (CD4 increasing = better):
      label_map = {'nonprogression': 'stable', 'linear': 'gradual_decline', 'nonlinear': 'rapid_decline'}
    - For Parkinson's (MDS-UPDRS-III increasing = worse):
      label_map = {'nonprogression': 'stable', 'linear': 'slow_progression', 'nonlinear': 'rapid_progression'}
    
    Parameters
    ----------
    traj : np.ndarray
        Trajectory values over time
    time_grid : np.ndarray
        Time points
    flat_thr : float
        Threshold for stable/flat slope (near zero or small positive)
    decline_thr : float
        Threshold for steep positive slope (rapid increase)
    nonlinear_gap : float
        Gap for detecting nonlinear acceleration
    label_map : dict, optional
        Maps generic labels to disease-specific labels.
        Default: {'nonprogression': 'stable', 'linear': 'slow_decline', 'nonlinear': 'rapid_decline'}
    
    Returns
    -------
    dict
        Disease-specific trajectory flags
        
    Examples
    --------
    For HIV (CD4 count):
    >>> label_map = {'nonprogression': 'stable', 'linear': 'gradual_decline', 'nonlinear': 'rapid_decline'}
    >>> flags = pos_flags_from_traj(cd4_trajectory, times, label_map=label_map)
    
    For Parkinson's (MDS-UPDRS-III):
    >>> label_map = {'nonprogression': 'stable', 'linear': 'slow_progression', 'nonlinear': 'rapid_progression'}
    >>> flags = pos_flags_from_traj(updrs_trajectory, times, label_map=label_map)
    """
    dt = np.diff(time_grid)
    dy = np.diff(traj)
    slopes = dy / dt

    frac_flat = (slopes <= decline_thr).mean()
    total_flat = np.mean(slopes) <= flat_thr

    frac_decl = (slopes > decline_thr).mean()

    # nonlinear: mean slope of faster half vs slower half differs by > nonlinear_gap
    s = np.sort(slopes)
    half = len(s) // 2
    fast = s[half:].mean()                    # faster increase half (more positive)
    slow = s[:half].mean()                    # slower half
    nonlinear = abs(fast - slow) > nonlinear_gap

    generic_flags = {
        'nonprogression': (frac_flat >= 0.8) & total_flat,
        'linear': frac_decl >= 0.8,
        'nonlinear': nonlinear
    }

    # Apply label mapping
    if label_map is None:
        # Default HIV mapping (positive direction)
        label_map = {
            'nonprogression': 'stable',
            'linear': 'slow_decline',
            'nonlinear': 'rapid_decline'
        }
    
    return {label_map[k]: v for k, v in generic_flags.items()}


def mmse_flag_from_traj(traj, time_grid, flat_thr=-0.3, decline_thr=-1.5, nonlinear_gap=1,
                        aging_thr=-0.5, label_map=None):
    """
    Classify MMSE trajectory (Alzheimer's - negative direction).
    """
    dt = np.diff(time_grid)
    dy = np.diff(traj)
    slopes = dy / dt

    frac_flat = (slopes >= aging_thr).mean()
    total_flat = np.mean(slopes) >= flat_thr

    frac_slow = ((decline_thr <= slopes) & (slopes < aging_thr)).mean()
    frac_fast = (decline_thr > slopes).mean()

    s = np.sort(slopes)  # ascending (more negative first)
    half = len(s) // 2
    fast = s[:half].mean()  # faster decline half (more negative)
    slow = s[half:].mean()  # slower half
    nonlinear = abs(fast - slow) > nonlinear_gap

    # For MMSE we have 4 categories potentially, but map to 3 main ones
    generic_flags = {
        'nonprogression': (frac_flat >= 0.8) & total_flat,
        'linear': (frac_slow >= 0.8) or (frac_fast >= 0.8),  # Combine slow and fast linear
        'nonlinear': nonlinear
    }
    
    # Apply label mapping
    if label_map is None:
        # Default Alzheimer's mapping
        label_map = {
            'nonprogression': 'stable',
            'linear': 'slow_decline',
            'nonlinear': 'rapid_decline'
        }
    
    return {label_map[k]: v for k, v in generic_flags.items()}

def _posterior_feature_probs_from_samples(
    y_samples, 
    time_grid, 
    flat_thr=-1.0, 
    decline_thr=-2.0, 
    nonlinear_gap=3.0,
    class_func: Callable[..., dict] = flags_from_traj,
    traj_types: Iterable[str] = ('prolonged_nonprogression', 'linear_decline', 'nonlinear'),
    label_map: dict = None
) -> dict[str, float]:
    """
    Compute posterior probabilities for trajectory types from MCMC samples.
    
    Parameters
    ----------
    y_samples : np.ndarray
        MCMC samples of trajectories, shape (n_samples, n_timepoints)
    time_grid : np.ndarray
        Time points
    flat_thr, decline_thr, nonlinear_gap : float
        Thresholds for classification
    class_func : callable
        Classification function (flags_from_traj, pos_flags_from_traj, mmse_flag_from_traj)
    traj_types : tuple of str
        Disease-specific trajectory type labels to count
    label_map : dict
        Maps generic labels to disease-specific labels
    
    Returns
    -------
    dict
        {f"trajtype_{label}_prob": probability} for each label in traj_types
    """
    counts = {k: 0 for k in traj_types}
    S = y_samples.shape[0]
    
    for s in range(S):
        # Get flags for this sample (returns disease-specific labels)
        flags = class_func(
            y_samples[s], 
            time_grid, 
            flat_thr, 
            decline_thr, 
            nonlinear_gap,
            label_map=label_map
        )
        
        # Count which types are flagged
        for k in traj_types:
            if flags.get(k, False):
                counts[k] += 1
    
    # Convert counts to probabilities
    return {f"trajtype_{k}_prob": counts[k] / S for k in traj_types}