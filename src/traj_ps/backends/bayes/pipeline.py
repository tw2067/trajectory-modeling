from __future__ import annotations
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from .sampling import _sample_post_trajs_scaled
from .classify import _posterior_feature_probs_from_samples, flags_from_traj

def _window_worker(
    window_df: pd.DataFrame,
    flat_thr=-1.0, decline_thr=-2.0, nonlinear_gap=3.0,
    df_basis=5, n_samples=1000, tune=1000, min_points=5, grid_freq=12,
    class_func=flags_from_traj, values='lab_value', time_col='time',
    traj_types=('prolonged_nonprogression','linear_decline','nonlinear'),
):
    tg, ys = _sample_post_trajs_scaled(window_df, df_basis, n_samples, tune, min_points, grid_freq,
                                       0.95, values, time_col)
    return _posterior_feature_probs_from_samples(
        ys, tg, flat_thr, decline_thr, nonlinear_gap, class_func, traj_types
    )

def compute_time_varying_trajectory_covariates_parallel(
    lab_df: pd.DataFrame,
    window_years: float = 2.0,
    flat_thr=-1.0, decline_thr=-2.0, nonlinear_gap=3.0,
    df_basis: int = 5, n_samples: int = 1000, tune: int = 1000,
    n_jobs: int = -1, min_points_per_window: int = 5, grid_freq: int = 12,
    class_func=flags_from_traj, pids='patient_id', values='lab_value',
    time_col='time', traj_types=('prolonged_nonprogression','linear_decline','nonlinear'),
    verbose: int = 0,
) -> pd.DataFrame:
    """
    For each (pid, anchor time), use the previous window_years to compute
    posterior probabilities for each trajectory type. Returns a long DF:
      [pids, time_col, traj_prob_* ...]
    """
    windows = []
    for pid, g in lab_df.groupby(pids):
        times = np.asarray(sorted(g[time_col].unique()))
        for t in times:
            win = g[(g[time_col] >= t - window_years) & (g[time_col] <= t)]
            if len(win) >= min_points_per_window:
                windows.append((pid, t, win[[pids, time_col, values]].copy()))

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(
        delayed(_window_worker)(
            win_df, flat_thr, decline_thr, nonlinear_gap,
            df_basis, n_samples, tune, min_points_per_window, grid_freq,
            class_func, values, time_col, traj_types
        )
        for (_, _, win_df) in windows
    )

    rows = []
    for (pid, t, _), probs in zip(windows, results):
        rec = {pids: pid, time_col: t}
        rec.update(probs)
        rows.append(rec)
    return pd.DataFrame(rows)