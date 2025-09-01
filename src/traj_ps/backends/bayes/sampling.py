from __future__ import annotations
from collections.abc import Iterable, Callable
import numpy as np, pandas as pd, pymc as pm
# If your environment prefers pytensor: from pytensor import tensor as at
import aesara.tensor as at  # keep as-is if this is what you use today
# noinspection PyUnresolvedReferences
from patsy import dmatrix
SEED = 920

def _scale_vals_and_grid(df_basis, dfw, values, time_col, tmin, span):
    t_scaled = (dfw[time_col] - tmin) / span
    dfw = dfw.copy(); dfw['time_scaled'] = t_scaled
    y_raw = dfw[values].to_numpy()
    y_mu  = float(np.mean(y_raw))
    y_sd  = float(np.std(y_raw)) if np.std(y_raw) > 0 else 1.0
    y_std = (y_raw - y_mu) / y_sd
    df_eff = int(min(df_basis, max(2, len(dfw) - 1), 8))
    X = dmatrix(f"bs(time_scaled, df={df_eff}, include_intercept=True)",
                dfw, return_type='dataframe').to_numpy()
    return X, y_mu, y_sd, y_std, df_eff

def _sample_post_trajs_scaled(
    df_window: pd.DataFrame,
    df_basis: int = 5,
    n_samples: int = 1000,
    tune: int = 1000,
    min_points: int = 5,
    grid_freq: int = 12,
    target_accept: float = 0.95,
    values: str = 'lab_value',
    time_col: str = 'time',
):
    """
    Fit Bayesian spline on a window (time scaled to [0,1]; y standardized), then
    back-transform posterior samples to ORIGINAL units on a uniform grid.
    Returns (time_grid_years, y_samples) with y_samples shape [S, Tgrid].
    """
    dfw = df_window.sort_values(time_col).copy()
    if len(dfw) < min_points:
        raise ValueError("Too few points in this window.")
    tmin, tmax = dfw[time_col].min(), dfw[time_col].max()
    span = max(1e-8, (tmax - tmin))

    X, y_mu, y_sd, y_std, df_eff = _scale_vals_and_grid(df_basis, dfw, values, time_col, tmin, span)

    # prediction grid on scaled time:
    n_grid = max(grid_freq, int(round(span * grid_freq)))
    tg_scaled = np.linspace(0.0, 1.0, n_grid)
    X_pred = dmatrix(f"bs(x, df={df_eff}, include_intercept=True)",
                     {"x": tg_scaled}, return_type='dataframe').to_numpy()

    with pm.Model() as m:
        beta  = pm.Normal("beta", mu=0.0, sigma=1.0, shape=X.shape[1])
        sigma = pm.HalfNormal("sigma", 1.0)
        mu    = at.dot(X, beta)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_std)
        trace = pm.sample(
            draws=n_samples, tune=tune, chains=4, cores=1,
            target_accept=target_accept, init="jitter+adapt_diag",
            random_seed=SEED, progressbar=False
        )

    post = trace.posterior["beta"].stack(sample=("chain","draw"))
    if "sample" not in post.dims or post.dims[0] != "sample":
        post = post.transpose("sample", ...)
    beta_samples = np.atleast_2d(post.values)
    y_samples = beta_samples @ X_pred.T  # standardized units
    y_pred = y_samples * y_sd + y_mu     # back to original lab units
    time_grid = tmin + tg_scaled * span  # back to YEARS
    return time_grid, y_pred
