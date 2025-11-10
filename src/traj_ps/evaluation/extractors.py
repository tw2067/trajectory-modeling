from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Literal, Tuple
from scipy.stats import linregress
from sklearn.linear_model import TheilSenRegressor, HuberRegressor

from traj_ps.backends.gam.model import GAMTrajPS, GAMConfig
from traj_ps.backends.bayes.model import BayesianTrajPS, BayesConfig
from traj_ps.backends.gam.features import extract_gam_trajectories
from ..config import DiseaseConfig

def _subset_by_lookback(g: pd.DataFrame, time_col: str, lookback_window: Optional[float]) -> pd.DataFrame:
    if lookback_window is None or not np.isfinite(lookback_window):
        return g
    tmax = float(g[time_col].max())
    g_win = g[g[time_col] >= (tmax - lookback_window)]
    return g_win if len(g_win) >= 2 else g


def _robust_linear_features(
    dynamic_df: pd.DataFrame,
    feature: str = "eGFR",
    pid_col: str = "pid",
    time_col: str = "time",
    feat_col: str = "feature_name",
    val_col: str = "value",
    lookback_window: Optional[float] = None,
) -> pd.DataFrame:
    """
    Robust per-patient slope/intercept:
    - Uses Theil-Sen (robust to outliers); falls back to Huber/OLS if needed
    - Intercept reported as patient's first observed value (baseline)
    - Uses all history by default (lookback_window=None)
    """
    rows = []
    df = dynamic_df[dynamic_df[feat_col] == feature].copy()
    for pid, g in df.groupby(pid_col):
        g = g.sort_values(time_col)
        if len(g) == 0:
            rows.append({"pid": pid, f"{feature}_slope": np.nan, f"{feature}_intercept": np.nan})
            continue

        g_use = _subset_by_lookback(g, time_col, lookback_window)
        x = g_use[time_col].to_numpy(dtype=float).reshape(-1, 1)
        y = g_use[val_col].to_numpy(dtype=float)

        slope = 0.0
        if len(y) >= 2:
            try:
                # Theil-Sen: robust slope for linear trends
                ts = TheilSenRegressor(random_state=0)
                ts.fit(x, y)
                slope = float(ts.coef_[0])
            except Exception:
                try:
                    hb = HuberRegressor()
                    hb.fit(x, y)
                    slope = float(hb.coef_[0])
                except Exception:
                    # Fallback to OLS
                    s, _, _, _, _ = linregress(x.ravel(), y)
                    slope = float(s)

        baseline = float(g[val_col].iloc[0])  # first observed value

        rows.append({
            "pid": pid,
            f"{feature}_slope": slope,
            f"{feature}_intercept": baseline,
        })
    return pd.DataFrame(rows)


def _dynamic_to_wide(
    dynamic_df: pd.DataFrame,
    labs: List[str],
    pid_col: str = "pid",
    feat_col: str = "feature_name",
    val_col: str = "value",
    time_col: str = "time",
    target_pid_col: str = "patient_id",
) -> pd.DataFrame:
    """
    Convert long dynamic_df (pid,time,feature_name,value) to wide lab table:
      columns: [patient_id, time, <lab1>, <lab2>, ...]
    """
    df = dynamic_df[[pid_col, time_col, feat_col, val_col]].copy()
    df = df[df[feat_col].isin(labs)]
    # Resolve duplicates at same (pid,time,feature) by last-observation
    df = df.sort_values([pid_col, feat_col, time_col]).drop_duplicates(
        subset=[pid_col, time_col, feat_col], keep="last"
    )
    wide = df.pivot_table(
        index=[pid_col, time_col], columns=feat_col, values=val_col, aggfunc="last"
    ).reset_index()
    wide = wide.rename(columns={pid_col: target_pid_col})
    # Ensure requested labs exist (fill with NaN if missing)
    for lab in labs:
        if lab not in wide.columns:
            wide[lab] = np.nan
    # Sort and return
    wide = wide[[target_pid_col, time_col] + labs].sort_values([target_pid_col, time_col]).reset_index(drop=True)
    return wide


def _per_patient_linear_features(
    dynamic_df: pd.DataFrame,
    feature: str = "eGFR",
    pid_col: str = "pid",
    time_col: str = "time",
    feat_col: str = "feature_name",
    val_col: str = "value",
    lookback_window: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute per-patient OLS slope and intercept for a biomarker.
    If lookback_window is set (in same time units as time_col), restrict to last window.
    Intercept returned is the first observed value (baseline) within the window if available,
    else overall first value.
    """
    rows = []
    df = dynamic_df[dynamic_df[feat_col] == feature].copy()
    for pid, g in df.groupby(pid_col):
        g = g.sort_values(time_col)
        if len(g) == 0:
            rows.append({"pid": pid, f"{feature}_slope": np.nan, f"{feature}_intercept": np.nan})
            continue

        # Determine window
        if lookback_window is not None and np.isfinite(lookback_window):
            tmax = float(g[time_col].max())
            g_win = g[g[time_col] >= (tmax - lookback_window)]
            if len(g_win) >= 2:
                g_use = g_win
            else:
                g_use = g
        else:
            g_use = g

        x = g_use[time_col].to_numpy(dtype=float)
        y = g_use[val_col].to_numpy(dtype=float)

        if len(x) >= 2:
            slope, intercept, _, _, _ = linregress(x, y)
        else:
            slope = 0.0
            intercept = float(y[0])

        # Baseline (intercept proxy): first observed value overall
        baseline = float(g[val_col].iloc[0])

        rows.append({
            "pid": pid,
            f"{feature}_slope": float(slope),
            f"{feature}_intercept": baseline if np.isfinite(baseline) else float(intercept),
        })
    return pd.DataFrame(rows)


def _standardize_bayes_probs(embed_out: Any) -> pd.DataFrame:
    """
    Normalize various possible embed() outputs to a per-patient probability table:
      columns: pid, trajtype_<label>_prob ...
    Accepts:
      - DataFrame (wide with prob columns or long with ['pid','type','prob'])
      - dict with key 'probs'/'posterior_probs' mapping to such DataFrame
      - tuple containing a suitable DataFrame
    Aggregates across windows/time by averaging per patient if needed.
    """
    df = None

    # Pull a DataFrame from common containers
    if isinstance(embed_out, pd.DataFrame):
        df = embed_out.copy()
    elif isinstance(embed_out, dict):
        for k in ("probs", "posterior_probs", "probabilities"):
            v = embed_out.get(k, None)
            if isinstance(v, pd.DataFrame):
                df = v.copy()
                break
    elif isinstance(embed_out, (list, tuple)):
        for e in embed_out:
            if isinstance(e, pd.DataFrame):
                df = e.copy()
                break

    if df is None or df.empty:
        return pd.DataFrame(columns=["pid"])

    # Identify patient id column
    pid_col = "pid" if "pid" in df.columns else ("patient_id" if "patient_id" in df.columns else None)
    if pid_col is None:
        # Try index
        if "pid" in getattr(df.index, "names", []):
            df = df.reset_index()
            pid_col = "pid"
        else:
            return pd.DataFrame(columns=["pid"])

    # If long format with 'type' and 'prob', pivot to wide
    if "type" in df.columns and ("prob" in df.columns or "p" in df.columns):
        prob_col = "prob" if "prob" in df.columns else "p"
        wide = df.pivot_table(index=[pid_col], columns="type", values=prob_col, aggfunc="mean")
        wide = wide.reset_index()
        wide = wide.rename(columns={pid_col: "pid"})
        # Rename probability columns to trajtype_<label>_prob
        prob_cols = {c: f"trajtype_{c}_prob" for c in wide.columns if c != "pid"}
        wide = wide.rename(columns=prob_cols)
        return wide

    # If wide format: take columns that look like probabilities and average per patient if windows exist
    # Candidates: columns starting with 'prob_', 'p_', or already 'trajtype_..._prob'
    prob_like = [c for c in df.columns if c.startswith(("prob_", "p_", "trajtype_")) and c != pid_col]
    if not prob_like:
        # Also accept numeric columns except pid/time/window
        cand = [c for c in df.columns if c not in {pid_col, "time", "window", "window_start", "window_end"}]
        if cand and np.all([np.issubdtype(df[c].dtype, np.number) for c in cand]):
            prob_like = cand

    if not prob_like:
        return pd.DataFrame({"pid": df[pid_col].unique()})

    # Aggregate to per-patient mean probs if multiple rows per patient (e.g., windows)
    agg = df.groupby(pid_col)[prob_like].mean().reset_index()
    agg = agg.rename(columns={pid_col: "pid"})
    # Normalize names to trajtype_*_prob
    rename_map = {}
    for c in prob_like:
        if not c.startswith("trajtype_"):
            # strip common prefixes
            base = c
            for pref in ("prob_", "p_"):
                if base.startswith(pref):
                    base = base[len(pref):]
            rename_map[c] = f"trajtype_{base}_prob"
    if rename_map:
        agg = agg.rename(columns=rename_map)

    return agg


def extract_features_gam(
    dynamic_df: pd.DataFrame,
    feature: str = "eGFR",
    cfg: Optional[Any] = None,
    lookback_window: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Use real GAM backend to get per-patient features and predicted trajectories.
    - features: pid, {feature}_slope (derivative at last time), {feature}_intercept (baseline)
    - trajectories: pid, time, predicted_value (GAM predictions at observed times)
    """
    # Optional tunables from cfg (if provided)
    n_splines = getattr(cfg, "n_splines", 10) if cfg is not None else 10
    lam = getattr(cfg, "lam", 0.6) if cfg is not None else 0.6

    features, trajectories = extract_gam_trajectories(
        dynamic_df=dynamic_df,
        feature=feature,
        n_jobs=-1,
        n_splines=n_splines,
        lam=lam,
        min_points=3,
    )

    # If for any reason GAM returns empty, fall back to robust linear
    if features.empty or trajectories.empty:
        feats = _robust_linear_features(dynamic_df, feature=feature, lookback_window=lookback_window)
        trajs = _reconstruct_trajectories_from_linear(dynamic_df, feats, feature=feature)
        return feats, trajs

    return features, trajectories


def extract_features_bayes(
    dynamic_df: pd.DataFrame,
    feature: str = "eGFR",
    cfg: Optional[BayesConfig] = None,
    disease_cfg: Optional[DiseaseConfig] = None,
    lookback_window: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Use Bayesian backend to compute per-patient trajectory-type probabilities.
    Returns:
      - features: pid + trajtype_*_prob columns (no slope/intercept)
      - trajectories: empty DataFrame (we don't reconstruct for Bayes)
    """
    from traj_ps.backends.bayes.model import BayesConfig
    from traj_ps.backends.bayes.pipeline import compute_time_varying_trajectory_covariates_parallel
    cfg = cfg or BayesConfig()
    
    # Prepare data
    primary_feature = disease_cfg.primary_feature.name
    lab_long = dynamic_df[dynamic_df["feature_name"] == primary_feature][["pid", "time", "value"]].rename(
        columns={
            "pid": cfg.pids,
            "time": cfg.time_col,
            "value": cfg.values
        }
    ).copy()
    
    # Call pipeline WITH the sampler config
    prob_feats = compute_time_varying_trajectory_covariates_parallel(
        lab_df=lab_long,
        window_years=cfg.window_years,
        flat_thr=disease_cfg.primary_feature.flat_threshold,
        decline_thr=disease_cfg.primary_feature.decline_threshold,
        nonlinear_gap=disease_cfg.primary_feature.nonlinear_gap,
        df_basis=cfg.df_basis,
        n_samples=cfg.n_samples,
        tune=cfg.tune,
        n_jobs=cfg.n_jobs,
        min_points_per_window=cfg.min_points_per_window,
        grid_freq=cfg.grid_freq,
        pids=cfg.pids,
        values=cfg.values,
        time_col=cfg.time_col,
        sampler=cfg.sampler,
        chains=cfg.chains,
        cores=cfg.cores,
        progressbar=cfg.progressbar,
        chain_method=cfg.chain_method,
        target_accept=cfg.target_accept,
        available_gpus=cfg.available_gpus,
    )
    
    # Ensure 'pid' column
    if cfg.pids in prob_feats.columns and "pid" not in prob_feats.columns:
        prob_feats = prob_feats.rename(columns={cfg.pids: "pid"})
    
    # No trajectories for Bayes
    empty_trajs = pd.DataFrame(columns=["pid", "time", "predicted_value"])
    return prob_feats, empty_trajs


def extract_features_deep(
    dynamic_df: pd.DataFrame,
    feature: str = "eGFR",
    lookback_window: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deep backend:
      - if an export function is available, prefer its predicted trajectories
      - else fall back to robust linear reconstruction
    """
    # Attempt to use export if available
    try:
        from traj_ps.inference.export_embed import export_embeddings, predict_trajectories  # type: ignore
        # export_embeddings could train/load a model and produce per-patient embeddings
        # predict_trajectories should return DataFrame: pid, time, predicted_value
        trajs = predict_trajectories(dynamic_df, feature=feature)
        # Derive simple features from predicted trajectories:
        # slope ~ OLS over predicted series; intercept ~ first predicted value
        rows = []
        df = trajs.merge(
            dynamic_df[dynamic_df["feature_name"] == feature][["pid", "time"]],
            on=["pid", "time"], how="inner"
        )
        for pid, g in df.groupby("pid"):
            g = g.sort_values("time")
            x = g["time"].to_numpy(dtype=float)
            y = g["predicted_value"].to_numpy(dtype=float)
            if len(x) >= 2:
                s, _, _, _, _ = linregress(x, y)
                slope = float(s)
            else:
                slope = 0.0
            intercept = float(y[0]) if len(y) else np.nan
            rows.append({"pid": pid, f"{feature}_slope": slope, f"{feature}_intercept": intercept})
        feats = pd.DataFrame(rows)
        return feats, trajs
    except Exception:
        # Fallback: robust linear features + reconstructed linear trajectories
        feats = _robust_linear_features(dynamic_df, feature=feature, lookback_window=lookback_window)
        trajs = _reconstruct_trajectories_from_linear(dynamic_df, feats, feature=feature)
        return feats, trajs


def extract_trajectory_features(
    backend: Literal["deep", "bayes", "gam"],
    dynamic_df: pd.DataFrame,
    feature: str = "eGFR",
    config: Optional[Dict[str, Any]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dispatch to backend-specific extractor and return standardized features:
      pid, {feature}_slope, {feature}_intercept

    Returns
    -------
    features : pd.DataFrame
    trajectories : pd.DataFrame
    """
    config = config or {}
    lw = config.get("lookback_window", None)
    b = backend.lower()

    if b == "gam":
        out = extract_features_gam(dynamic_df, feature=feature, cfg=config.get("gam_cfg"), lookback_window=lw)
    elif b == "bayes":
        out = extract_features_bayes(dynamic_df, feature=feature, cfg=config.get("bayes_cfg"), lookback_window=lw)
    elif b == "deep":
        out = extract_features_deep(dynamic_df, feature=feature, lookback_window=lw)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Normalize to (features, trajectories)
    if isinstance(out, tuple) and len(out) == 2:
        features, trajectories = out
    else:
        features = out
        trajectories = _reconstruct_trajectories_from_linear(dynamic_df, features, feature=feature)

    return features, trajectories


def _reconstruct_trajectories_from_linear(
    dynamic_df: pd.DataFrame,
    features_df: pd.DataFrame,
    feature: str = "eGFR"
) -> pd.DataFrame:
    """
    Reconstruct predicted trajectories using extracted slope/intercept.
    
    Returns DataFrame with: pid, time, predicted_value
    """
    predictions = []
    
    for pid in features_df['pid'].unique():
        slope = features_df.loc[features_df['pid'] == pid, f'{feature}_slope'].iloc[0]
        intercept = features_df.loc[features_df['pid'] == pid, f'{feature}_intercept'].iloc[0]
        
        # Get time points for this patient
        patient_times = dynamic_df[
            (dynamic_df['pid'] == pid) & 
            (dynamic_df['feature_name'] == feature)
        ]['time'].values
        
        for t in patient_times:
            predicted_value = intercept + slope * t
            predictions.append({
                'pid': pid,
                'time': t,
                'predicted_value': predicted_value
            })
    
    return pd.DataFrame(predictions)