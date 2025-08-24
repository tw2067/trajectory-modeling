from collections.abc import Callable, Iterable
import numpy as np
import pandas as pd
import pymc as pm
from joblib import Parallel, delayed
from traj_class_functions import *
import aesara.tensor as at
from scipy.interpolate import interp1d
# noinspection PyUnresolvedReferences
from patsy import dmatrix


SEED = 920


def _sample_post_trajs_scaled(df_window: pd.DataFrame, df_basis: int = 5,
    n_samples: int = 1000, tune: int = 1000, min_points: int = 5, grid_freq = 12,
    target_accept: float = 0.95,  values: str = 'lab_value', time_col: str = 'time'
):
    """
    Fits a Bayesian spline on a sliding window using internally scaled time and standardized y,
    then returns posterior samples back-transformed to original units.

    Assumptions:
      - df_window has columns: ['time', 'lab_value'] ; time is in YEARS
      - Output time_grid is in YEARS, y_samples are in ORIGINAL lab units
    """
    dfw = df_window.sort_values(time_col).copy()
    if len(dfw) < min_points:
        raise ValueError("Too few points in this window.")

    tmin, tmax = dfw[time_col].min(), dfw[time_col].max()
    span = max(1e-8, (tmax - tmin))

    X, y_mu, y_sd, y_std, df_eff = _scale_vals_and_grid(df_basis, dfw,values, time_col, tmin, span)

    # prediction grid:
    n_grid = max(grid_freq, int(round(span * grid_freq)))
    tg_scaled = np.linspace(0.0, 1.0, n_grid)               # scaled grid
    X_pred = dmatrix(f"bs(x, df={df_eff}, include_intercept=True)",
                 {"x": tg_scaled}, return_type='dataframe').to_numpy()

    with pm.Model() as m:
        beta  = pm.Normal("beta", mu=0.0, sigma=1.0, shape=X.shape[1])   # mildly informative
        sigma = pm.HalfNormal("sigma", 1.0)
        mu    = pm.math.dot(X, beta)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_std)

        trace = pm.sample(
            draws=n_samples, tune=tune, chains=4, cores=1, target_accept=target_accept,
            init="jitter+adapt_diag", random_seed=SEED, progressbar=False
        )

    post = trace.posterior["beta"]
    post = post.stack(sample=("chain", "draw"))
    if "sample" not in post.dims or post.dims[0] != "sample":
        post = post.transpose("sample", ...)

    beta_samples = post.values
    beta_samples = np.atleast_2d(beta_samples)
    # project to time grid: [S, T]
    y_samples = beta_samples @ X_pred.T

    # 4) BACK-TRANSFORM to original units
    y_pred = y_samples * y_sd + y_mu                                           # original lab units
    time_grid = tmin + tg_scaled * span                                         # YEARS

    return time_grid, y_pred  # time in years, y in original units


def _scale_vals_and_grid(df_basis, dfw, values, time_col, tmin, span):
    # 1) scale time -> [0,1]  (only for modeling)
    t_scaled = (dfw[time_col] - tmin) / span
    dfw['time_scaled'] = t_scaled

    # 2) standardize lab values (only for modeling)
    y_raw = dfw[values].to_numpy()
    y_mu = float(np.mean(y_raw))
    y_sd = float(np.std(y_raw)) if np.std(y_raw) > 0 else 1.0
    y_std = (y_raw - y_mu) / y_sd

    # 3) spline basis on scaled time
    df_eff = int(min(df_basis, max(2, len(dfw) - 1), 8))
    X = dmatrix(f"bs(time_scaled, df={df_eff}, include_intercept=True)",
                dfw, return_type='dataframe').to_numpy()

    return X, y_mu, y_sd, y_std, df_eff


def _posterior_feature_probs_from_samples(y_samples,
                                          time_grid,
                                          flat_thr: float = -1.0,
                                          decline_thr: float= -2.0,
                                          nonlinear_gap: float = 3.0,
                                          class_func: Callable[..., dict] = flags_from_traj,
                                          traj_types: Iterable[str] = ('prolonged_nonprogression',
                                                                       'linear_decline',
                                                                       'nonlinear')
                                          ) -> dict[str, float]:
    counts = {traj_t: 0 for traj_t in traj_types}
    n_samples = y_samples.shape[0]
    for i in range(n_samples):
        flags = class_func(y_samples[i], time_grid, flat_thr, decline_thr, nonlinear_gap)
        for k in counts:
            counts[k] += int(flags[k])
    return {f"traj_prob_{k}": counts[k] / n_samples for k in counts}


def _window_worker(window_df: pd.DataFrame,
                   flat_thr: float = -1.0,
                   decline_thr: float = -2.0,
                   nonlinear_gap: float = 3.0,
                   df_basis: int = 5,
                   n_samples: int = 1000,
                   tune: int = 1000,
                   min_points: int = 5,
                   grid_freq: int = 12,
                   class_func: Callable[..., dict] = flags_from_traj,
                   values: str = 'lab_value',
                   time_col: str = 'time',
                   traj_types: Iterable[str] = ('prolonged_nonprogression',
                                                'linear_decline',
                                                'nonlinear')
                   ):
    """Compute posterior feature probabilities for a single window."""
    time_grid, y_samples = _sample_post_trajs_scaled(window_df, df_basis=df_basis, n_samples=n_samples,
                                                     tune=tune, min_points=min_points ,grid_freq=grid_freq,
                                                     values=values, time_col=time_col)
    return _posterior_feature_probs_from_samples(y_samples, time_grid, flat_thr=flat_thr, decline_thr=decline_thr,
                                                 nonlinear_gap=nonlinear_gap, class_func=class_func,
                                                 traj_types=traj_types)


def compute_time_varying_trajectory_covariates_parallel(
        lab_df: pd.DataFrame,
        window_years: float = 2.0,
        flat_thr = -1,
        decline_thr = -2,
        nonlinear_gap = 3.0,
        df_basis: int = 5,
        n_samples: int = 1000,
        tune: int = 1000,
        n_jobs: int = -1,
        min_points_per_window: int = 5,
        grid_freq: int = 12,
        class_func: Callable[..., dict] = flags_from_traj,
        pids: str = 'patient_id',
        values: str = 'lab_value',
        time_col: str = 'time',
        traj_types: Iterable[str] = ('prolonged_nonprogression',
                                     'linear_decline',
                                     'nonlinear'),
        verbose: int = 0
):
    """
    Parallel step-5: For each patient and anchor time, use the previous `window_years`
    to compute posterior trajectory-type probabilities. Returns a long DF with
    (patient_id, time, traj_prob_linear_decline, traj_prob_nonlinear, traj_prob_prolonged_nonprogression)
    """

    # Build window index
    windows = []
    for pid, g in lab_df.groupby(pids):
        times = np.asarray(sorted(g[time_col].unique()))
        for t in times:
            win = g[(g[time_col] >= t - window_years) & (g[time_col] <= t)]
            if len(win) >= min_points_per_window:
                windows.append((pid, t, win[[pids, time_col, values]].copy()))

    # Process in parallel
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(
        delayed(_window_worker)(win_df, flat_thr=flat_thr, decline_thr=decline_thr, nonlinear_gap=nonlinear_gap,
                                df_basis=df_basis, n_samples=n_samples, tune=tune, grid_freq=grid_freq,
                                class_func=class_func, values=values, time_col=time_col, traj_types=traj_types)
        for (_, _, win_df) in windows
    )

    # Stitch back
    records = []
    for (pid, t, _), probs in zip(windows, results):
        rec = {pids: pid, time_col: t}
        rec.update({
            k: probs[k] for k in probs})
        records.append(rec)
    return pd.DataFrame(records)


################################################ older versions #######################################################

def _sample_post_trajs(df_window, df_basis=5, n_samples=1000):
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


