import numpy as np
import pandas as pd
import pymc as pm
from joblib import Parallel, delayed
import aesara.tensor as at
from scipy.interpolate import interp1d
from lifelines import CoxTimeVaryingFitter
# noinspection PyUnresolvedReferences
from patsy import dmatrix


# TODO: code review and debugging using appropriate data (maybe simulate if no DB is good enough)

SEED = 920

def _sample_posterior_trajectories(df_window, df_basis=5, n_samples=1000, window_years=2):
    """Single-window Bayesian spline + posterior samples projected to monthly grid."""
    df_window = df_window.sort_values('time').copy()
    # monthly resolution across the window
    tmin, tmax = df_window['time'].min(), df_window['time'].max()
    n_grid = max(6, int(round((tmax - tmin) * 12)))  # at least a few points
    time_grid = np.linspace(tmin, tmax, n_grid)

    X_basis = dmatrix(f"bs(time, df={df_basis}, include_intercept=True)",
                      df_window, return_type='dataframe').values
    X_pred  = dmatrix(f"bs(x, df={df_basis}, include_intercept=True)",
                      {"x": time_grid}, return_type='dataframe').values

    y = df_window['lab_value'].values

    with pm.Model() as model:
        beta  = pm.Normal("beta", mu=0.0, sigma=1.0, shape=X_basis.shape[1])
        sigma = pm.HalfNormal("sigma", 1.0)
        mu    = pm.math.dot(X_basis, beta)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(draws=n_samples, tune=500, chains=4, cores=1,
                          target_accept=0.95, random_seed=SEED, progressbar=False, chain_method="sequential")

    post = trace.posterior["beta"]
    post = post.stack(sample=("chain", "draw"))
    if "sample" not in post.dims or post.dims[0] != "sample":
        post = post.transpose("sample", ...)

    beta_samples = post.values
    beta_samples = np.atleast_2d(beta_samples)
    # project to time grid: [S, T]
    y_samples = beta_samples @ X_pred.T

    return time_grid, y_samples

def _flags_from_traj(traj, time_grid, flat_thr=-1, decline_thr=-2, nonlinear_gap=3.0):
    dt = np.diff(time_grid)
    dy = np.diff(traj)
    slopes = dy / dt

    # prolonged nonprogression: mostly flat and average slope of non-progression
    frac_flat = (slopes >= decline_thr).mean()
    total_flat = np.mean(slopes) >= flat_thr

    # linear decline: mostly steeply negative
    frac_decl = (slopes < decline_thr).mean()

    # nonlinear: mean slope of faster half vs slower half differs by > 3 (as requested)
    s = np.sort(slopes)                       # ascending (more negative first)
    half = len(s) // 2
    fast = s[:half].mean()                    # faster decline half (more negative)
    slow = s[half:].mean()                    # slower half
    nonlinear = abs(fast - slow) > nonlinear_gap

    return {
        'prolonged_nonprogression': (frac_flat >= 0.8) & total_flat,
        'linear_decline':            frac_decl >= 0.8,
        'nonlinear':                 nonlinear
    }

def _posterior_feature_probs_from_samples(y_samples, time_grid):
    counts = {'prolonged_nonprogression': 0, 'linear_decline': 0, 'nonlinear': 0}
    S = y_samples.shape[0]
    for s in range(S):
        flags = _flags_from_traj(y_samples[s], time_grid)
        for k in counts:
            counts[k] += int(flags[k])
    return {f"traj_prob_{k}": counts[k] / S for k in counts}

def _window_worker(window_df, df_basis=5, n_samples=1000):
    """Compute posterior feature probabilities for a single window."""
    time_grid, y_samples = _sample_posterior_trajectories(window_df, df_basis=df_basis, n_samples=n_samples)
    return _posterior_feature_probs_from_samples(y_samples, time_grid)

def compute_time_varying_trajectory_covariates_parallel(
    lab_df: pd.DataFrame,
    window_years: float = 2.0,
    df_basis: int = 5,
    n_samples: int = 1000,
    n_jobs: int = -1,
    min_points_per_window: int = 5,
    verbose=0
):
    """
    Parallel step-5: For each patient and anchor time, use the previous `window_years`
    to compute posterior trajectory-type probabilities. Returns a long DF with
    (patient_id, time, traj_prob_linear_decline, traj_prob_nonlinear, traj_prob_prolonged_nonprogression)
    """
    # Build window index
    windows = []
    for pid, g in lab_df.groupby('patient_id'):
        times = np.asarray(sorted(g['time'].unique()))
        for t in times:
            win = g[(g['time'] >= t - window_years) & (g['time'] <= t)]
            if len(win) >= min_points_per_window:
                windows.append((pid, t, win[['patient_id', 'time', 'lab_test', 'lab_value']].copy()))

    # Process in parallel
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(
        delayed(_window_worker)(win_df, df_basis=df_basis, n_samples=n_samples)
        for (_, _, win_df) in windows
    )

    # Stitch back
    records = []
    for (pid, t, _), probs in zip(windows, results):
        rec = {'patient_id': pid, 'time': t}
        rec.update({
            'traj_prob_prolonged_nonprogression': probs['traj_prob_prolonged_nonprogression'],
            'traj_prob_linear_decline':            probs['traj_prob_linear_decline'],
            'traj_prob_nonlinear':                 probs['traj_prob_nonlinear'],
        })
        records.append(rec)
    return pd.DataFrame(records)



