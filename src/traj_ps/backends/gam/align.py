from __future__ import annotations
import numpy as np, pandas as pd

def align_gam_betas_to_bins(
    gam_cov_df: pd.DataFrame,      # output of compute_gam_beta_covariates_adaptive_parallel
    bin_map: dict[str, np.ndarray],# pid -> array of bin starts
    pid_col: str = "patient_id",
    time_col: str = "time",
) -> pd.DataFrame:
    """
    LOCF-align zero-padded GAM beta blocks to PS bin starts.
    Returns [pid, time, <lab>_beta_* , <lab>_n_splines, <lab>_lam, ...]
    """
    # identify all exported beta/flag columns
    keep_cols = [c for c in gam_cov_df.columns if c not in (pid_col, time_col)]
    rows = []
    for pid, starts in bin_map.items():
        dfp = gam_cov_df[gam_cov_df[pid_col] == pid].sort_values(time_col)
        t = dfp[time_col].to_numpy(float)
        for t0 in starts:
            rec = {"pid": pid, "time": float(t0)}
            if t.size == 0:
                for c in keep_cols: rec[c] = 0.0
            else:
                idx = np.searchsorted(t, t0, side="right") - 1
                if idx < 0:
                    for c in keep_cols: rec[c] = 0.0
                else:
                    for c in keep_cols: rec[c] = float(dfp.iloc[idx][c])
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(["pid","time"]).reset_index(drop=True)
