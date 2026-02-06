from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Literal, Tuple
from scipy.stats import linregress
from sklearn.linear_model import TheilSenRegressor, HuberRegressor
import sys

from traj_ps.backends.gam.model import GAMTrajPS, GAMConfig
from traj_ps.backends.bayes.model import BayesianTrajPS, BayesConfig
from traj_ps.backends.gam.features import extract_gam_trajectories
from traj_ps.config import DiseaseConfig

def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching column name from candidates list."""
    for col in candidates:
        if col in df.columns:
            return col
    return None

def _subset_by_lookback(g: pd.DataFrame, time_col: str, lookback_window: Optional[float]) -> pd.DataFrame:
    if lookback_window is None or not np.isfinite(lookback_window):
        return g
    tmax = float(g[time_col].max())
    g_win = g[g[time_col] >= (tmax - lookback_window)]
    return g_win if len(g_win) >= 2 else g


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


def extract_features_bayes(
    dynamic_df: pd.DataFrame,
    feature: str = "eGFR",
    cfg: Optional[Any] = None,
    disease_cfg: Optional[DiseaseConfig] = None,
    lookback_window: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Use Bayesian backend to compute per-patient trajectory-type probabilities.
    Returns:
      - features: pid + trajtype_*_prob columns (disease-specific labels)
      - trajectories: empty DataFrame (we don't reconstruct for Bayes)
    """
    from traj_ps.backends.bayes.model import BayesConfig
    from traj_ps.backends.bayes.pipeline import compute_time_varying_trajectory_covariates_parallel
    from traj_ps.backends.bayes.classify import (
        flags_from_traj, 
        pos_flags_from_traj,
        mmse_flag_from_traj,
    )
    
    # Get config
    if isinstance(cfg, dict):
        bayes_cfg = cfg.get('bayes_cfg', BayesConfig())
    else:
        bayes_cfg = cfg or BayesConfig()
    
    # Get disease-specific settings
    if disease_cfg is not None:
        flat_thr = disease_cfg.primary_feature.flat_threshold
        decline_thr = disease_cfg.primary_feature.decline_threshold
        nonlinear_gap = disease_cfg.primary_feature.nonlinear_gap
        traj_types = tuple(disease_cfg.trajectory_types)
        label_map = disease_cfg.trajectory_type_map
        
        # Choose classification function based on feature direction
        direction = disease_cfg.primary_feature.direction
        higher_better = disease_cfg.primary_feature.higher_better
        disease_name = disease_cfg.name.lower()
        
        if direction == "increasing":
            # Use pos_flags_from_traj for ANY increasing trajectory
            # (works for both CD4 count and MDS-UPDRS-III)
            # The label_map distinguishes between good and bad outcomes
            class_func = pos_flags_from_traj
            outcome_type = "better" if higher_better else "worse"
            print(f"  [Bayes] Using pos_flags_from_traj for {disease_name} (increasing = {outcome_type})")
        elif disease_name == "alzheimers":
            # Special thresholds for MMSE
            class_func = mmse_flag_from_traj
            print(f"  [Bayes] Using mmse_flag_from_traj for Alzheimer's")
        else:
            # Default: decreasing features (eGFR, etc.)
            class_func = flags_from_traj
            print(f"  [Bayes] Using flags_from_traj for {disease_name} (decreasing)")
        
        print(f"  [Bayes] Using disease config for {disease_cfg.name}")
        print(f"  [Bayes] Trajectory types: {traj_types}")
        print(f"  [Bayes] Label mapping: {label_map}")
        print(f"  [Bayes] Thresholds: flat={flat_thr}, decline={decline_thr}, nonlinear_gap={nonlinear_gap}")
    else:
        # Default CKD
        flat_thr = -1.0
        decline_thr = -2.0
        nonlinear_gap = 3.0
        traj_types = ('prolonged_nonprogression', 'linear_decline', 'nonlinear')
        label_map = {
            'nonprogression': 'prolonged_nonprogression',
            'linear': 'linear_decline',
            'nonlinear': 'nonlinear'
        }
        class_func = flags_from_traj
        print(f"  [Bayes] Using default CKD thresholds")
    
    # Prepare data
    lab_long = dynamic_df[dynamic_df["feature_name"] == feature][["pid", "time", "value"]].rename(
        columns={
            "pid": bayes_cfg.pids,
            "time": bayes_cfg.time_col,
            "value": bayes_cfg.values
        }
    ).copy()
    
    if lab_long.empty:
        print(f"  [WARNING] No data for feature '{feature}'")
        empty_cols = ["pid"] + [f"trajtype_{t}_prob" for t in traj_types]
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=["pid", "time", "predicted_value"])
    
    # Call pipeline with disease-specific settings
    prob_feats = compute_time_varying_trajectory_covariates_parallel(
        lab_df=lab_long,
        window_years=bayes_cfg.window_years,
        flat_thr=flat_thr,
        decline_thr=decline_thr,
        nonlinear_gap=nonlinear_gap,
        df_basis=bayes_cfg.df_basis,
        n_samples=bayes_cfg.n_samples,
        tune=bayes_cfg.tune,
        n_jobs=bayes_cfg.n_jobs,
        min_points_per_window=bayes_cfg.min_points_per_window,
        grid_freq=bayes_cfg.grid_freq,
        pids=bayes_cfg.pids,
        values=bayes_cfg.values,
        time_col=bayes_cfg.time_col,
        sampler=bayes_cfg.sampler,
        chains=bayes_cfg.chains,
        cores=bayes_cfg.cores,
        progressbar=bayes_cfg.progressbar,
        chain_method=bayes_cfg.chain_method,
        target_accept=bayes_cfg.target_accept,
        class_func=class_func,           # Pass selected function
        traj_types=traj_types,
        label_map=label_map,
        available_gpus=getattr(bayes_cfg, 'available_gpus', None),
    )
    
    # Ensure 'pid' column
    if bayes_cfg.pids in prob_feats.columns and "pid" not in prob_feats.columns:
        prob_feats = prob_feats.rename(columns={bayes_cfg.pids: "pid"})
    
    # Verify expected columns
    expected_cols = [f"trajtype_{t}_prob" for t in traj_types]
    missing_cols = [c for c in expected_cols if c not in prob_feats.columns]
    if missing_cols:
        print(f"  [WARNING] Missing probability columns: {missing_cols}")
        for col in missing_cols:
            prob_feats[col] = 0.0
    
    print(f"  [Bayes] Output columns: {[c for c in prob_feats.columns if '_prob' in c]}")
    
    empty_trajs = pd.DataFrame(columns=["pid", "time", "predicted_value"])
    return prob_feats, empty_trajs


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
    
    # ADD: Check if data exists for this feature
    if df.empty:
        print(f"  [WARNING] No data found for feature '{feature}'")
        return pd.DataFrame(columns=["pid", f"{feature}_slope", f"{feature}_intercept"])
    
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
    """
    rows = []
    df = dynamic_df[dynamic_df[feat_col] == feature].copy()
    
    # ADD: Check if data exists for this feature
    if df.empty:
        print(f"  [WARNING] No data found for feature '{feature}'")
        return pd.DataFrame(columns=["pid", f"{feature}_slope", f"{feature}_intercept"])
    
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


def extract_features_gam(
    dynamic_df: pd.DataFrame,
    feature: str = "eGFR",
    cfg: Optional[Any] = None,
    disease_cfg: Optional[DiseaseConfig] = None,
    lookback_window: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """GAM backend: Extract trajectory features using global spline fitting."""
    
    print(f"  [GAM] Extracting trajectory features for {feature}...")
    
    # Get GAMConfig
    from traj_ps.backends.gam.model import GAMConfig
    
    if isinstance(cfg, dict):
        gam_cfg = cfg.get('gam_cfg', GAMConfig())
    elif isinstance(cfg, GAMConfig):
        gam_cfg = cfg
    else:
        gam_cfg = GAMConfig()
    
    # GAM uses GLOBAL spline fitting, not windows
    # The extract_gam_trajectories function in features.py does this correctly
    n_splines = gam_cfg.n_splines_range[0] if hasattr(gam_cfg, 'n_splines_range') else 10
    lam = gam_cfg.lam_grid[1] if hasattr(gam_cfg, 'lam_grid') else 0.6
    
    print(f"  [GAM] Using global spline fitting: n_splines={n_splines}, lam={lam}")
    
    try:
        # Use the EXISTING extract_gam_trajectories function (correct implementation)
        features, trajectories = extract_gam_trajectories(
            dynamic_df=dynamic_df,
            feature=feature,
            n_jobs=-1,
            n_splines=n_splines,
            lam=lam,
            min_points=3,
        )
        
        if not features.empty:
            print(f"  [GAM] ✓ Successfully extracted features for {len(features)} patients")
            return features, trajectories
            
    except Exception as e:
        print(f"  [GAM] GAM extraction failed: {e}")
        import traceback
        traceback.print_exc()

    # Fallback: robust linear features
    print(f"  [GAM] Using robust linear fallback for {feature}")
    feats = _robust_linear_features(dynamic_df, feature=feature, lookback_window=lookback_window)
    trajs = _reconstruct_trajectories_from_linear(dynamic_df, feats, feature=feature)
    return feats, trajs


def extract_features_deep(
    dynamic_df: pd.DataFrame,
    feature: str = "eGFR",
    cfg: Optional[Any] = None,
    disease_cfg: Optional[DiseaseConfig] = None,
    lookback_window: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deep backend: Extract trajectory features using window-based modeling."""
    
    print(f"  [Deep] Extracting trajectory features for {feature}...")
    print(f"  [Deep] Python path: {sys.path[:3]}")  # Show first 3 paths
    
    # Try to use window-based extraction first
    try:
        print(f"  [Deep] Attempting import: traj_ps.backends.deep.features")
        from traj_ps.backends.deep.features import extract_deep_trajectory_features
        print(f"  [Deep] ✓ Import successful!")
        
        # Get window config
        if isinstance(cfg, dict):
            window_cfg = cfg.get('window', {})
        else:
            window_cfg = {}
        
        window_size = window_cfg.get('size', 1.0)
        min_obs = window_cfg.get('min_obs', 3)
        
        print(f"  [Deep] Using window-based extraction: window_size={window_size}, min_obs={min_obs}")
        
        feats, trajs = extract_deep_trajectory_features(
            dynamic_df=dynamic_df,
            feature=feature,
            window_size=window_size,
            min_obs=min_obs
        )
        
        if not feats.empty:
            print(f"  [Deep] ✓ Successfully extracted features for {len(feats)} patients")
            return feats, trajs
        else:
            print(f"  [Deep] ⚠️  Window-based extraction returned empty, falling back...")
            
    except ImportError as e:
        print(f"  [Deep] ✗ ImportError: {e}")
        print(f"  [Deep] File should be at: src/traj_ps/backends/deep/features.py")
        import traceback
        print("  [Deep] Full traceback:")
        traceback.print_exc()
    except Exception as e:
        print(f"  [Deep] ✗ Exception during extraction: {e}")
        import traceback
        print("  [Deep] Full traceback:")
        traceback.print_exc()
    
    # Fallback to robust linear features
    print(f"  [Deep] Using robust linear fallback for {feature}")
    feats = _robust_linear_features(dynamic_df, feature=feature, lookback_window=lookback_window)
    trajs = _reconstruct_trajectories_from_linear(dynamic_df, feats, feature=feature)
    return feats, trajs


def _reconstruct_trajectories_from_linear(
    dynamic_df: pd.DataFrame,
    features_df: pd.DataFrame,
    feature: str = "eGFR"
) -> pd.DataFrame:
    """
    Reconstruct predicted trajectories using extracted slope/intercept.
    
    Parameters
    ----------
    dynamic_df : pd.DataFrame
        Original dynamic data with columns: pid, time, feature_name, value
    features_df : pd.DataFrame
        Extracted features with columns: pid, {feature}_slope, {feature}_intercept
    feature : str
        Feature name to reconstruct (e.g., 'eGFR', 'CD4_count', 'MMSE')
    
    Returns
    -------
    pd.DataFrame
        Predicted trajectories with columns: pid, time, predicted_value
    """
    if features_df.empty:
        return pd.DataFrame(columns=['pid', 'time', 'predicted_value'])
    
    slope_col = f'{feature}_slope'
    intercept_col = f'{feature}_intercept'
    
    # Check that required columns exist
    if slope_col not in features_df.columns or intercept_col not in features_df.columns:
        print(f"  [WARNING] Missing slope/intercept columns for {feature}")
        print(f"  [WARNING] Available columns: {features_df.columns.tolist()}")
        return pd.DataFrame(columns=['pid', 'time', 'predicted_value'])
    
    predictions = []
    
    for pid in features_df['pid'].unique():
        pid_features = features_df[features_df['pid'] == pid]
        if pid_features.empty:
            continue
            
        slope = pid_features[slope_col].iloc[0]
        intercept = pid_features[intercept_col].iloc[0]
        
        # Skip if NaN or inf
        if not np.isfinite(slope) or not np.isfinite(intercept):
            continue
        
        # Get time points for this patient and feature
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


def extract_trajectory_features(
    backend: Literal["deep", "bayes", "gam"],
    dynamic_df: pd.DataFrame,
    feature: str = "eGFR",
    config: Optional[Dict[str, Any]] = None,
    disease_cfg: Optional[DiseaseConfig] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dispatch to backend-specific extractor and return standardized features.

    Parameters
    ----------
    backend : {"deep", "bayes", "gam"}
        Backend to use for extraction
    dynamic_df : pd.DataFrame
        Longitudinal data with columns: pid, time, feature_name, value
    feature : str
        Primary feature to extract (e.g., 'eGFR', 'CD4_count', 'MMSE')
    config : dict, optional
        Backend-specific configuration
    disease_cfg : DiseaseConfig, optional
        Disease-specific configuration (thresholds, labels, etc.)

    Returns
    -------
    features : pd.DataFrame
        Columns: pid, {feature}_slope, {feature}_intercept (or trajtype_*_prob for Bayes)
    trajectories : pd.DataFrame
        Columns: pid, time, predicted_value (empty for Bayes)
    """
    config = config or {}
    lw = config.get("lookback_window", None)
    b = backend.lower()

    if b == "gam":
        out = extract_features_gam(
            dynamic_df, 
            feature=feature, 
            cfg=config.get("gam_cfg"), 
            disease_cfg=disease_cfg,
            lookback_window=lw
        )
    elif b == "bayes":
        out = extract_features_bayes(
            dynamic_df, 
            feature=feature, 
            cfg=config.get("bayes_cfg"), 
            disease_cfg=disease_cfg,
            lookback_window=lw
        )
    elif b == "deep":
        out = extract_features_deep(
            dynamic_df, 
            feature=feature, 
            cfg=config,
            disease_cfg=disease_cfg,
            lookback_window=lw
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Normalize to (features, trajectories)
    if isinstance(out, tuple) and len(out) == 2:
        features, trajectories = out
    else:
        features = out
        trajectories = _reconstruct_trajectories_from_linear(dynamic_df, features, feature=feature)

    return features, trajectories