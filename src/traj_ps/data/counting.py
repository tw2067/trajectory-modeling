# src/traj_ps/data/counting.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Dict, List, Tuple
from .aggregate import make_bins, DEFAULT_BIN_W

def _treatment_start_time(dynamic_df: pd.DataFrame, pid: str) -> float | None:
    """Return first time with treated==1 for pid, else None."""
    g = dynamic_df[dynamic_df["pid"] == pid]
    t = g.loc[g["treated"] == 1, "time"]
    return float(t.min()) if not t.empty else None

def aggregate_covariates_to_bins(
    dynamic_df: pd.DataFrame,
    static_df: pd.DataFrame,
    agg_features: Iterable[str],
    bin_width: float = DEFAULT_BIN_W,
    binary_prefixes: Tuple[str,...] = ("Med",),
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Build aggregated covariates on a fixed bin grid.
    Returns:
      agg_df: one row per (pid, bin), columns: pid, start, stop, treated_start, <agg_features...>
      bin_map: pid -> bin_starts
    """
    agg_features = list(agg_features)
    out_rows = []
    bin_map: Dict[str, np.ndarray] = {}

    for pid, g in dynamic_df.groupby("pid"):
        Tmax = float(static_df.loc[static_df.pid == pid, "Tmax"].iloc[0])
        starts = make_bins(Tmax, bin_width)                  # bin starts
        stops  = np.r_[starts[1:], starts[-1] + bin_width]   # bin stops
        bin_map[pid] = starts

        # treatment start (first treated==1 anywhere in dynamic)
        t_start = _treatment_start_time(dynamic_df, pid)

        # pre-extract for speed
        g = g.sort_values("time")
        for s, e in zip(starts, stops):
            row = {"pid": pid, "start": float(s), "stop": float(e)}
            # treated at START of interval
            row["treatment"] = int(0 if (t_start is None or s < t_start) else 1)
            # aggregated features: mean in [s,e) for continuous; LOCF at start for binary-like
            for feat in agg_features:
                gf = g[g["feature_name"] == feat]
                t = gf["time"].to_numpy(float)
                v = gf["value"].to_numpy(float)
                is_binary = feat.startswith(binary_prefixes) or (
                    v.size > 0 and np.isin(np.unique(v), [0, 1]).all()
                )
                if is_binary:
                    if t.size == 0:
                        row[feat] = 0.0
                    else:
                        idx = np.searchsorted(t, s, side="right") - 1
                        idx = 0 if idx < 0 else idx
                        row[feat] = float(v[np.clip(idx, 0, v.size - 1)])
                else:
                    inb = (t >= s) & (t < e)
                    row[feat] = float(np.nanmean(v[inb])) if inb.any() else np.nan
            out_rows.append(row)

    agg_df = pd.DataFrame(out_rows).sort_values(["pid", "start"]).reset_index(drop=True)
    # fill continuous NAs by forward fill then median
    for feat in agg_features:
        cont = ~agg_df[feat].isin([0.0, 1.0]) if agg_df[feat].notna().any() else None
        if cont is not None:
            agg_df[feat] = agg_df.groupby("pid")[feat].ffill()
            agg_df[feat] = agg_df[feat].fillna(agg_df[feat].median())
    return agg_df, bin_map

def align_traj_probs_to_bins(
    traj_probs_by_lab: Dict[str, pd.DataFrame],
    bin_map: Dict[str, np.ndarray],
    pid_col: str = "patient_id",
    time_col: str = "time",
) -> pd.DataFrame:
    """
    For each lab: we have a DF with columns [patient_id, time, traj_prob_*].
    Align (LOCF) those probabilities to each pid's bin starts, suffixing columns with "__<lab>".
    Returns long table with [pid, time] and all aligned prob columns.
    """
    wide_rows = []
    # collect all prob column names across labs
    prob_cols_by_lab = {
        lab: [c for c in df.columns if c.startswith("traj_prob_")]
        for lab, df in traj_probs_by_lab.items()
    }

    for pid, starts in bin_map.items():
        rec = {"pid": pid}
        # set time now (bin start)
        for t0 in starts:
            rec = {"pid": pid, "time": float(t0)}
            # for each lab, LOCF at bin start
            for lab, probs_df in traj_probs_by_lab.items():
                dfp = probs_df[probs_df[pid_col] == pid].sort_values(time_col)
                t = dfp[time_col].to_numpy(float)
                if t.size == 0:
                    # no info yet → zeros
                    for col in prob_cols_by_lab[lab]:
                        rec[f"{col}__{lab}"] = 0.0
                else:
                    idx = np.searchsorted(t, t0, side="right") - 1
                    if idx < 0:
                        for col in prob_cols_by_lab[lab]:
                            rec[f"{col}__{lab}"] = 0.0
                    else:
                        for col in prob_cols_by_lab[lab]:
                            rec[f"{col}__{lab}"] = float(dfp.iloc[idx][col])
            wide_rows.append(rec)
    return pd.DataFrame(wide_rows).sort_values(["pid", "time"]).reset_index(drop=True)

def build_counting_process(
    static_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    traj_aligned_df: pd.DataFrame,
    id_col: str = "pid",
) -> pd.DataFrame:
    """
    Merge static covariates (repeat per row), aggregated covariates, and aligned trajectory covariates
    into a single counting-process table for lifelines.CoxTimeVaryingFitter.
    """
    # merge on pid+time: agg_df has start; traj_aligned_df has time
    base = agg_df.merge(traj_aligned_df, left_on=[id_col, "start"], right_on=[id_col, "time"], how="left")
    base = base.drop(columns=["time"])
    # attach static covariates
    static_cols = [c for c in static_df.columns if c not in (id_col,)]
    design = base.merge(static_df[[id_col] + static_cols], on=id_col, how="left")
    # safety: fill any remaining NaNs with 0
    design = design.fillna(0.0)
    return design
