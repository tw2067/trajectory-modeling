from __future__ import annotations
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pygam import LinearGAM, s
from sklearn.model_selection import KFold  # only used if cv_splits > 0
from scipy.sparse import issparse
import pygam.utils as _pgutils

SEED = 920

# -----------------------------
# monkey‑patch: robust Cholesky for some pyGAM/SciPy combos
# -----------------------------
def _safe_cholesky(A, **kwargs):
    if issparse(A):
        A = A.toarray()
    else:
        A = np.asarray(A)
    return np.linalg.cholesky(A)
_pgutils.cholesky = _safe_cholesky  # patch once

# -----------------------------
# Utilities
# -----------------------------
def _scale_time_to_unit(t: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Scale time array to [0,1]. Returns (scaled, tmin, span).
    Robust to zero-span (returns zeros).
    """
    t = np.asarray(t, dtype=float)
    tmin, tmax = np.min(t), np.max(t)
    span = max(1e-12, tmax - tmin)
    return (t - tmin) / span, float(tmin), float(span)

def _beta_column_names(prefix: str, n_splines: int) -> list[str]:
    """
    Column names for 1 intercept + n_splines coefficients for a single lab.
    """
    return [f"{prefix}_beta_intercept"] + [f"{prefix}_beta_s{i+1}" for i in range(n_splines)]


# -----------------------------
# Core: fit GAM adaptively & return betas
# -----------------------------
def _fit_gam_get_betas_adaptive(
    time_scaled: np.ndarray,
    y: np.ndarray,
    spline_order: int = 3,
    n_splines_range: tuple[int, int] = (4, 6),
    lam_grid: tuple[float, ...] = (0.3, 1.0, 3.0),
    max_iter: int = 5000,
    seed: int = SEED,
    cv_splits: int = 0,
) -> tuple[np.ndarray, int, float]:
    """
    Fit a univariate GAM: y ~ s(time_scaled) with an adaptive number of splines and (optional) λ selection.
    Returns:
        betas: np.ndarray, shape (1 + n_best,)
        n_best: int, number of splines chosen
        lam_best: float, smoothing parameter used
    Notes:
        - When cv_splits == 0 (default), selection uses pyGAM's GCV across lam_grid and candidate n_splines.
        - When cv_splits > 0 and enough points, selection uses KFold MSE.
    """
    np.random.RandomState(seed)
    X = time_scaled.reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    n = len(y)

    # Upper bound on number of splines given data size: require parameters < samples
    max_n_by_data = max(1, n - (spline_order + 1))
    lo, hi = n_splines_range
    hi = min(hi, max_n_by_data)
    if hi < lo:
        hi = lo
    candidate_ns = list(range(lo, hi + 1))

    # Selection
    best_score = np.inf
    best_model = None
    best_n = candidate_ns[0]
    best_lam = lam_grid[0]

    if cv_splits and n >= max(cv_splits, 5):
        # K-fold CV on MSE
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        for n_spl in candidate_ns:
            for lam in lam_grid:
                mses = []
                for tr, te in kf.split(X):
                    gam = LinearGAM(s(0, n_splines=n_spl, spline_order=spline_order, lam=lam), max_iter=max_iter)
                    gam.fit(X[tr], y[tr])
                    y_hat = gam.predict(X[te])
                    mses.append(np.mean((y[te] - y_hat) ** 2))
                score = float(np.mean(mses))
                if score < best_score:
                    best_score, best_n, best_lam = score, n_spl, lam
        best_model = _fit_one(X, y, best_n, best_lam, spline_order, max_iter)
    else:
        # Use pyGAM's GCV
        for n_spl in candidate_ns:
            for lam in lam_grid:
                gam = _fit_one(X, y, n_spl, lam, spline_order, max_iter)
                gcv = float(gam.statistics_["GCV"])  # lower is better
                if gcv < best_score:
                    best_score, best_model, best_n, best_lam = gcv, gam, n_spl, lam

    betas = best_model.coef_.astype(float)  # length: 1 + best_n
    return betas, int(best_n), float(best_lam)


def _fit_one(X, y, n_spl: int, lam: float, spline_order: int, max_iter: int) -> LinearGAM:
    gam = LinearGAM(s(0, n_splines=n_spl, spline_order=spline_order, lam=lam), max_iter=max_iter)
    gam.fit(X, y)
    return gam

# -----------------------------
# work unit: one patient
# -----------------------------
def _process_one_patient(
    g_pid: pd.DataFrame,
    labs: list[str],
    window_years: float,
    n_splines_range: tuple[int, int],
    lam_grid: tuple[float, ...],
    min_points_per_window: int,
    standardize_y: bool,
    cv_splits: int,
    max_iter: int,
) -> list[dict]:
    pid = g_pid["patient_id"].iat[0]
    times_anchor = np.sort(g_pid["time"].unique())
    max_k_global = n_splines_range[1]
    recs = []

    for t_anchor in times_anchor:
        t_start = t_anchor - window_years
        rec = {"patient_id": pid, "time": float(t_anchor)}
        any_lab = False

        for lab in labs:
            g = g_pid[(g_pid["lab_test"] == lab) &
                      (g_pid["time"] >= t_start) &
                      (g_pid["time"] <= t_anchor)].sort_values("time")

            beta_cols = _beta_column_names(lab, max_k_global)
            # zero‑init block + flags
            for c in beta_cols:
                rec[c] = 0.0
            rec[f"{lab}_n_splines"] = 0
            rec[f"{lab}_lam"] = np.nan

            if len(g) < min_points_per_window:
                continue

            ts, _, _ = _scale_time_to_unit(g["time"].to_numpy(float))
            y = g["lab_value"].to_numpy(float)
            if standardize_y:
                mu, sd = float(np.mean(y)), float(np.std(y)) or 1.0
                y = (y - mu) / sd

            try:
                betas, n_best, lam_best = _fit_gam_get_betas_adaptive(
                    ts, y,
                    n_splines_range=n_splines_range,
                    lam_grid=lam_grid,
                    max_iter=max_iter,
                    cv_splits=cv_splits,
                )
                pad = np.zeros(1 + max_k_global, dtype=float)
                pad[:1 + n_best] = betas[:1 + n_best]
                for c, v in zip(beta_cols, pad):
                    rec[c] = float(v)
                rec[f"{lab}_n_splines"] = int(n_best)
                rec[f"{lab}_lam"] = float(lam_best)
                any_lab = True
            except Exception as e:
                print(e)

        if any_lab:
            recs.append(rec)

    return recs

# -----------------------------
# Public API: compute covariates on sliding windows
# -----------------------------

def compute_gam_beta_covariates_adaptive_parallel(
    lab_df: pd.DataFrame,
    window_years: float = 2.0,
    labs: list[str] | None = None,
    n_splines_range: tuple[int, int] = (4, 6),
    lam_grid: tuple[float, ...] = (0.3, 1.0, 3.0),
    min_points_per_window: int = 6,
    standardize_y: bool = True,
    cv_splits: int = 0,
    max_iter: int = 5000,
    n_jobs: int = -1,
    verbose: int = 5,
) -> pd.DataFrame:
    """
    Parallel GAM-beta covariates:
      - parallelized by patient (safe, process-based via joblib 'loky')
      - zero-padded beta blocks with flags

    Returns tidy DataFrame: ['patient_id','time', <beta cols>, <flags>]
    """
    if labs is None:
        labs = sorted(lab_df["lab_test"].unique().tolist())

    # joblib prefers lightweight payloads; pre-split by patient
    patient_groups = [g for _, g in lab_df.groupby("patient_id", sort=False)]

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(
        delayed(_process_one_patient)(
            g_pid=g,
            labs=labs,
            window_years=window_years,
            n_splines_range=n_splines_range,
            lam_grid=lam_grid,
            min_points_per_window=min_points_per_window,
            standardize_y=standardize_y,
            cv_splits=cv_splits,
            max_iter=max_iter,
        )
        for g in patient_groups
    )

    # flatten
    flat = [rec for sub in results for rec in sub]
    cov_df = (pd.DataFrame.from_records(flat)
              .sort_values(["patient_id", "time"])
              .reset_index(drop=True))
    return cov_df

def compute_gam_beta_covariates_adaptive(
    lab_df: pd.DataFrame,
    window_years: float = 2.0,
    labs: list[str] | None = None,
    n_splines_range: tuple[int, int] = (3, 6),
    lam_grid: tuple[float, ...] = (0.3, 1.0, 3.0),
    min_points_per_window: int = 6,
    standardize_y: bool = True,
    cv_splits: int = 0,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Build time-varying GAM beta covariates over sliding windows.

    Input
    -----
    lab_df: long DataFrame with columns:
        'patient_id' (hashable), 'time' (float YEARS), 'lab_test' (str), 'lab_value' (float)
        Irregular sampling supported.
    window_years: retrospective window length (e.g., 2.0)
    labs: subset of lab_test to include (None -> all)
    n_splines_range: tuple(min, max) candidate spline counts per window (adaptive)
    lam_grid: candidate λ values (pyGAM penalty)
    min_points_per_window: minimum points to attempt a GAM fit
    standardize_y: if True, z-score y within window (betas used as representations, not slopes)
    cv_splits: 0 -> use GCV; >0 -> K-fold CV on MSE (slower)
    seed: random seed

    Output
    ------
    DataFrame with one row per (patient_id, time) anchor, containing:
        - For each lab:
            {lab}_beta_intercept, {lab}_beta_s1, ... ,{lab}_beta_sK (zero-padded to K=max in n_splines_range)
            {lab}_n_splines (int), {lab}_lam (float)
        - plus 'patient_id', 'time'
    """
    if labs is None:
        labs = sorted(lab_df["lab_test"].unique().tolist())

    max_k_global = n_splines_range[1]  # for fixed-width padding
    out_records = []

    # Iterate patients
    for pid, g_pid in lab_df.groupby("patient_id"):
        # Anchor only at observed measurement times for efficiency
        for t_anchor in np.sort(g_pid["time"].unique()):
            t_start = t_anchor - window_years
            rec = {"patient_id": pid, "time": float(t_anchor)}
            any_lab_fit = False

            for lab in labs:
                g = g_pid[
                    (g_pid["lab_test"] == lab)
                    & (g_pid["time"] >= t_start)
                    & (g_pid["time"] <= t_anchor)
                ].sort_values("time")

                # Predeclare zero-padded betas + flags
                beta_cols = _beta_column_names(lab, max_k_global)
                for c in beta_cols:
                    rec[c] = 0.0
                rec[f"{lab}_n_splines"] = 0
                rec[f"{lab}_lam"] = np.nan

                if len(g) < min_points_per_window:
                    continue

                t = g["time"].to_numpy(dtype=float)
                ts, _, _ = _scale_time_to_unit(t)

                y = g["lab_value"].to_numpy(dtype=float)
                if standardize_y:
                    ymu = float(np.mean(y))
                    ysd = float(np.std(y)) or 1.0
                    y_fit = (y - ymu) / ysd
                else:
                    y_fit = y

                try:
                    betas, n_best, lam_best = _fit_gam_get_betas_adaptive(
                        ts,
                        y_fit,
                        n_splines_range=n_splines_range,
                        lam_grid=lam_grid,
                        seed=seed,
                        cv_splits=cv_splits,
                    )
                    # Zero-pad to (1 + max_k_global)
                    pad = np.zeros(1 + max_k_global, dtype=float)
                    pad[: 1 + n_best] = betas[: 1 + n_best]
                    for c, v in zip(beta_cols, pad):
                        rec[c] = float(v)

                    rec[f"{lab}_n_splines"] = int(n_best)
                    rec[f"{lab}_lam"] = float(lam_best)
                    any_lab_fit = True
                except Exception as e:
                    # Keep zeros + n_splines=0 to mark failure/insufficient data
                    print(e)

            if any_lab_fit:
                out_records.append(rec)


    cov_df = (
        pd.DataFrame.from_records(out_records)
        .sort_values(["patient_id", "time"])
        .reset_index(drop=True)
    )
    return cov_df

# -----------------------------
# Optional: PCA compression of beta blocks (fixed width already)
# -----------------------------
def pca_reduce_gam_betas(
    cov_df: pd.DataFrame,
    labs: list[str],
    n_components: int = 3,
    impute_value: float = 0.0,
    random_state: int = SEED,
) -> pd.DataFrame:
    """
    Reduce each lab's zero-padded beta block to n principal components.
    Returns a DataFrame with ['patient_id','time', <lab>_beta_pc1..pcK] per lab.
    """
    from sklearn.decomposition import PCA

    out = cov_df[["patient_id", "time"]].copy()

    for lab in labs:
        cols = [c for c in cov_df.columns if c.startswith(f"{lab}_beta_")]
        X = cov_df[cols].to_numpy(dtype=float)
        # Impute any residual NAs (should be none due to zero padding, but safe):
        X = np.where(np.isfinite(X), X, impute_value)

        pca = PCA(n_components=n_components, random_state=random_state)
        Z = pca.fit_transform(X)

        for k in range(n_components):
            out[f"{lab}_beta_pc{k+1}"] = Z[:, k]

        # Optionally keep n_splines & lam flags:
        out[f"{lab}_n_splines"] = cov_df[f"{lab}_n_splines"] if f"{lab}_n_splines" in cov_df.columns else 0
        out[f"{lab}_lam"] = cov_df[f"{lab}_lam"] if f"{lab}_lam" in cov_df.columns else np.nan

    return out