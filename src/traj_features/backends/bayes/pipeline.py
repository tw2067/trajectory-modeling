from __future__ import annotations
import numpy as np, pandas as pd
import os
from joblib import Parallel, delayed
from .sampling import _sample_post_trajs_scaled
from .classify import _posterior_feature_probs_from_samples, flags_from_traj

def _window_worker(
    window_df: pd.DataFrame,
    flat_thr=-1.0, decline_thr=-2.0, nonlinear_gap=3.0,
    df_basis=5, n_samples=1000, tune=1000, min_points=5, grid_freq=12,
    class_func=flags_from_traj, values='lab_value', time_col='time',
    traj_types=('prolonged_nonprogression','linear_decline','nonlinear'),
    label_map=None,
    *,
    sampler: str = 'pymc',
    chains: int = 4,
    cores: int = 1,
    progressbar: bool = False,
    chain_method: str = "vectorized",
    target_accept: float = 0.95,

):
    # Give each process its own PyTensor compiledir to prevent file lock contention.
    # We must update pytensor.config directly — loky workers inherit the parent's
    # PYTENSOR_FLAGS at spawn time (pytensor is already imported before _window_worker
    # runs), so changing os.environ alone does not affect the live pytensor config.
    import pytensor
    base = os.environ.get("SLURM_TMPDIR", "/tmp")
    pid = os.getpid()
    compiledir = os.path.join(base, f"pytensor_{pid}")
    os.makedirs(compiledir, exist_ok=True)
    pytensor.config.base_compiledir = compiledir  # update live config
    os.environ["PYTENSOR_FLAGS"] = f"base_compiledir={compiledir},floatX=float64"
    # Also make BLAS single-threaded inside each worker to avoid oversubscription
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1") 

    tg, ys = _sample_post_trajs_scaled(
        window_df, df_basis, n_samples, tune, min_points, grid_freq,
        target_accept, values, time_col,
        sampler=sampler, chains=chains, cores=cores, progressbar=progressbar,
        chain_method=chain_method
    )
    return _posterior_feature_probs_from_samples(
        ys, tg, flat_thr, decline_thr, nonlinear_gap, class_func, traj_types, label_map=label_map
    )


def _window_worker_gpu_pinned(
    window_df: pd.DataFrame,
    gpu_id: int,
    *args, sampler: str, chains: int, cores: int, progressbar: bool, chain_method: str, target_accept: float, **kwargs
):
    # Pin this child process to a specific GPU, then import JAX lazily
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if sampler in ("numpyro", "blackjax"):
        import jax  # ensures CUDA_VISIBLE_DEVICES is honored in this process
        jax.config.update("jax_enable_x64", True)
    return _window_worker(
        window_df, *args, sampler=sampler, chains=chains, cores=cores,
        progressbar=progressbar, chain_method=chain_method, target_accept=target_accept, **kwargs
    )


def compute_time_varying_trajectory_covariates_parallel(
    lab_df: pd.DataFrame,
    window_years: float = 2.0,
    flat_thr=-1.0, decline_thr=-2.0, nonlinear_gap=3.0,
    df_basis: int = 5, n_samples: int = 1000, tune: int = 1000,
    n_jobs: int = -1, min_points_per_window: int = 5, grid_freq: int = 12,
    batch_size: int = 500,
    class_func=flags_from_traj, pids='patient_id', values='lab_value',
    time_col='time', traj_types=('prolonged_nonprogression','linear_decline','nonlinear'),
    label_map=None,
    verbose: int = 0,
    *,
    sampler: str = 'pymc',
    chains: int = 4,
    cores: int = 1,
    progressbar: bool = False,
    chain_method: str = "vectorized", 
    target_accept: float = 0.95,
    available_gpus: list[int]|None = None,
    windowing_col: str|None = None  # allows windows in different granullarity than the measuring frequency
) -> pd.DataFrame:
    """
    For each (pid, anchor time), use the previous window_years to compute
    posterior probabilities for each trajectory type. Returns a long DF:
      [pids, time_col, traj_prob_* ...]
    """

    if label_map is None:
        label_map = {
            'nonprogression': 'prolonged_nonprogression',
            'linear': 'linear_decline',
            'nonlinear': 'nonlinear'
        }

    windowing_col_effective = time_col if windowing_col is None else windowing_col

    def _process_batch(batch):
        if not batch:
            return []

        if sampler in ("numpyro", "blackjax") and available_gpus and len(available_gpus) > 1:
            nj_eff = min(len(available_gpus), len(batch))
            print(f"[INFO] Using {nj_eff} GPUs in parallel: {available_gpus[:nj_eff]}")
            gpu_rr = [available_gpus[i % len(available_gpus)] for i in range(len(batch))]
            results = Parallel(n_jobs=nj_eff, backend="loky", verbose=verbose)(
                delayed(_window_worker_gpu_pinned)(
                    win_df, gpu_id,
                    flat_thr, decline_thr, nonlinear_gap, df_basis, n_samples, tune,
                    min_points_per_window, grid_freq, class_func, values, time_col, traj_types, label_map=label_map,
                    sampler=sampler, chains=chains, cores=cores, progressbar=progressbar, chain_method=chain_method,
                    target_accept=target_accept
                )
                for (_, _, win_df), gpu_id in zip(batch, gpu_rr)
            )
        else:
            nj_eff = n_jobs if sampler == "nutpie" else 1 if sampler in ("numpyro", "blackjax") else n_jobs
            results = Parallel(n_jobs=nj_eff, backend="loky", verbose=verbose)(
                delayed(_window_worker)(
                    win_df, flat_thr, decline_thr, nonlinear_gap, df_basis, n_samples, tune,
                    min_points_per_window, grid_freq, class_func, values, time_col, traj_types, label_map=label_map,
                    sampler=sampler, chains=chains, cores=cores, progressbar=progressbar, chain_method=chain_method,
                    target_accept=target_accept
                )
                for (_, _, win_df) in batch
            )

        out_rows = []
        for (pid, t, _), probs in zip(batch, results):
            rec = {pids: pid, windowing_col_effective: t}
            rec.update(probs)
            out_rows.append(rec)
        return out_rows

    if available_gpus is None and sampler in ("numpyro", "blackjax"):
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            # Parse CUDA_VISIBLE_DEVICES (e.g., "0,1,2" -> [0, 1, 2])
            available_gpus = [int(x.strip()) for x in cuda_visible.split(',') if x.strip()]
            print(f"[INFO] Auto-detected GPUs: {available_gpus}")

    rows = []
    batch = []
    for pid, g in lab_df.groupby(pids):
        times = np.asarray(sorted(g[windowing_col_effective].unique()))
        for t in times:
            win = g[(g[windowing_col_effective] >= t - window_years) & (g[windowing_col_effective] <= t)]
            if len(win) >= min_points_per_window:
                batch.append((pid, t, win[[pids, time_col, values]].copy()))

            if len(batch) >= batch_size:
                rows.extend(_process_batch(batch))
                batch = []

    if batch:
        rows.extend(_process_batch(batch))

    return pd.DataFrame(rows)

