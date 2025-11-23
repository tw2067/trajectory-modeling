import numpy as np, pandas as pd

def get_default_features(disease: str = "ckd") -> tuple:
    """Get default features for a disease."""
    if disease == "ckd":
        return ("eGFR", "Creatinine", "SBP", "DBP", "HbA1c", "MedA")
    elif disease == "hiv":
        return ("CD4_count", "viral_load", "weight", "hemoglobin", "ART")
    elif disease == "alzheimers":
        return ("MMSE", "ADAS_cog", "hippocampal_volume", "CSF_tau", "medications")
    elif disease == "parkinsons":
        return ("MDS_UPDRS_III", "LEDD", "tremor_score", "rigidity_score", "medications")
    else:
        return ("eGFR", "Creatinine", "SBP", "DBP", "HbA1c", "MedA")


def get_feature_params(disease: str, feature: str) -> dict:
    """Get simulation parameters for a specific feature."""
    params = {
        # CKD
        ("ckd", "eGFR"): {
            "baseline_mean": 90,
            "baseline_std": 20,
            "slope_mean": -3.0,
            "slope_std": 1.0,
            "noise_std": 3,
            "min_val": 5,
            "max_val": 200,
            "rate": 2,  # measurements per year
        },
        ("ckd", "HbA1c"): {
            "baseline_mean": 7.5,
            "baseline_std": 1.0,
            "slope_mean": 0.1,
            "slope_std": 0.2,
            "noise_std": 0.3,
            "min_val": 5,
            "max_val": 14,
            "rate": 3,
        },
        ("ckd", "SBP"): {
            "baseline_mean": 120,
            "baseline_std": 15,
            "slope_mean": 0,
            "slope_std": 5,
            "noise_std": 8,
            "min_val": 90,
            "max_val": 200,
            "rate": 4,
        },
        
        # HIV
        ("hiv", "CD4_count"): {
            "baseline_mean": 500,
            "baseline_std": 200,
            "slope_mean": -20,
            "slope_std": 10,
            "noise_std": 50,
            "min_val": 0,
            "max_val": 1500,
            "rate": 4,
        },
        ("hiv", "viral_load"): {
            "baseline_mean": 50000,
            "baseline_std": 30000,
            "slope_mean": 5000,
            "slope_std": 2000,
            "noise_std": 10000,
            "min_val": 0,
            "max_val": 1000000,
            "rate": 3,
        },
        ("hiv", "weight"): {
            "baseline_mean": 70,
            "baseline_std": 15,
            "slope_mean": -0.5,
            "slope_std": 0.3,
            "noise_std": 2,
            "min_val": 40,
            "max_val": 150,
            "rate": 2,
        },
        
        # Alzheimer's
        ("alzheimers", "MMSE"): {
            "baseline_mean": 24,
            "baseline_std": 4,
            "slope_mean": -2.0,
            "slope_std": 1.0,
            "noise_std": 1.5,
            "min_val": 0,
            "max_val": 30,
            "rate": 2,
        },
        ("alzheimers", "ADAS_cog"): {
            "baseline_mean": 15,
            "baseline_std": 5,
            "slope_mean": 3.0,
            "slope_std": 1.5,
            "noise_std": 2,
            "min_val": 0,
            "max_val": 70,
            "rate": 2,
        },
        # Parkinson's Disease
        ("parkinsons", "MDS_UPDRS_III"): {
            "baseline_mean": 20,  # Moderate baseline motor symptoms
            "baseline_std": 8,
            "slope_mean": 3.0,  # Worsens ~3 points/year (INCREASING)
            "slope_std": 1.5,
            "noise_std": 2.0,
            "min_val": 0,
            "max_val": 132,  # Max possible MDS-UPDRS-III score
            "rate": 2,  # Assessments twice per year
        },
        ("parkinsons", "LEDD"): {
            "baseline_mean": 600,  # Levodopa equivalent daily dose (mg)
            "baseline_std": 200,
            "slope_mean": 50,  # Dose escalation
            "slope_std": 30,
            "noise_std": 50,
            "min_val": 0,
            "max_val": 2000,
            "rate": 4,
        },
        ("parkinsons", "tremor_score"): {
            "baseline_mean": 2.0,
            "baseline_std": 0.8,
            "slope_mean": 0.3,  # Worsens gradually
            "slope_std": 0.2,
            "noise_std": 0.3,
            "min_val": 0,
            "max_val": 4,
            "rate": 2,
        },
    }
    
    # Return params or defaults
    key = (disease, feature)
    return params.get(key, {
        "baseline_mean": 50,
        "baseline_std": 10,
        "slope_mean": 0,
        "slope_std": 1,
        "noise_std": 5,
        "min_val": 0,
        "max_val": 100,
        "rate": 2,
    })


def simulate_dynamic_static(
    n_pat=120, 
    features=None,  
    disease="ckd", 
    disease_cfg=None,
    seed=920
):
    """
    Simulate longitudinal data for any disease.
    
    Parameters
    ----------
    n_pat : int
        Number of patients
    features : tuple, optional
        Feature names to simulate. If None, uses disease defaults.
    disease : str
        Disease type: 'ckd', 'hiv', 'alzheimers'
    seed : int
        Random seed
    """
    if features is None:
        features = get_default_features(disease)

    # Get disease config if not provided
    if disease_cfg is None:
        from traj_ps.config import DiseaseConfig
        if disease == "ckd":
            disease_cfg = DiseaseConfig.for_ckd()
        elif disease == "hiv":
            disease_cfg = DiseaseConfig.for_hiv()
        elif disease == "alzheimers":
            disease_cfg = DiseaseConfig.for_alzheimers()
        elif disease == "parkinsons":
            disease_cfg = DiseaseConfig.for_parkinsons()
        else:
            disease_cfg = None

    # Build feature metadata lookup
    feature_metadata = {}
    if disease_cfg is not None:
        feature_metadata[disease_cfg.primary_feature.name] = disease_cfg.primary_feature
        for feat_cfg in disease_cfg.secondary_features:
            feature_metadata[feat_cfg.name] = feat_cfg
    
    rows_dyn, rows_sta = [], []
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    
    # Get primary feature (first in list)
    primary_feature = features[0]
    
    for i in range(n_pat):
        pid = f"P{i:05d}"
        T = rng.uniform(5.0, 10.0)
        trt_flag = rng.random() < 0.5
        age = int(rng.integers(45, 85))
        sex = int(rng.integers(0, 2))
        cci = int(rng.poisson(2))
        rows_sta.append((pid, age, sex, cci, float(T)))

        def pp(rate):
            """Poisson process for measurement times."""
            t, out = 0.0, []
            while True:
                t += rng.exponential(1 / max(rate, 1e-6))
                if t >= T:
                    break
                out.append(t)
            return np.array(out) if out else np.array([rng.uniform(0.1, T)])

        # Generate observations for each feature
        for feat in features:
            params = get_feature_params(disease, feat)

            # Get feature metadata
            feat_meta = feature_metadata.get(feat)
            is_binary = feat_meta.feature_type == 'binary' if feat_meta else False
            has_periodicity = feat_meta.has_periodicity if feat_meta else False
            
            if is_binary:
                # Binary medication features
                t_med = pp(params["rate"])
                med = 0
                for tm in t_med:
                    if rng.random() < 0.4:
                        med = 1 - med
                    rows_dyn.append((pid, float(tm), feat, float(med), 0))
            else:
                # Continuous features
                times = pp(params["rate"])
                
                # Generate trajectory
                baseline = params["baseline_mean"]
                slope = params["slope_mean"]
                values = baseline + slope * times + rng.normal(
                    0, params["noise_std"], size=times.size
                )
                
                # Add periodic variation for some features
                if has_periodicity:
                    values += 5 * np.sin(2 * np.pi * times)
                
                values = np.clip(values, params["min_val"], params["max_val"])
                
                for t, v in zip(times, values):
                    rows_dyn.append((pid, float(t), feat, float(v), 0))

        # Treatment assignment based on primary feature
        grid = np.arange(0.0, T, 0.1)
        
        # Get primary feature observations for this patient
        primary_obs = [(t, v) for p, t, f, v, _ in rows_dyn 
                       if p == pid and f == primary_feature]
        
        if primary_obs:
            t_prim, v_prim = zip(*primary_obs)
            t_prim, v_prim = np.array(t_prim), np.array(v_prim)
            
            # LOCF for propensity score
            def locf(tt, vv, g):
                if tt.size == 0:
                    return np.zeros_like(g)
                idx = np.searchsorted(tt, g, side="right") - 1
                idx[idx < 0] = 0
                return vv[np.clip(idx, 0, vv.size - 1)]
            
            v_loc = locf(t_prim, v_prim, grid)

            # Get primary feature metadata for propensity model
            prim_meta = feature_metadata.get(primary_feature)
            
            # Disease-specific propensity model based on feature direction
            if prim_meta and prim_meta.direction == "increasing" and not prim_meta.higher_better:
                # Increasing = worsening (e.g., MDS-UPDRS-III)
                # Higher values → higher treatment probability
                score = -3.0 + 0.05 * np.maximum(0, v_loc - prim_meta.normal_max) + 0.1 * age / 70
            elif prim_meta and prim_meta.direction == "increasing" and prim_meta.higher_better:
                # Increasing = improving (e.g., CD4 count)
                # Lower values → higher treatment probability
                score = -3.0 + 0.01 * np.maximum(0, prim_meta.normal_min - v_loc) + 0.1 * sex
            elif prim_meta and prim_meta.direction == "decreasing":
                # Decreasing features (e.g., eGFR, MMSE)
                # Lower values → higher treatment probability
                score = -3.2 + 0.04 * np.maximum(0, prim_meta.normal_min - v_loc) + 0.2 * sex + 0.05 * cci
            else:
                # Fallback
                score = -3.0 + 0.01 * sex
            
            p = 1 - np.exp(-np.exp(score))
            
            treated = 0
            t_start = None
            for gg, pp in zip(grid, p):
                if treated == 0 and rng.random() < pp and trt_flag:
                    treated = 1
                    t_start = float(gg)
                    break
            
            if t_start is not None:
                # Mark all observations after treatment start
                for k, (ppid, tt, fn, val, tr) in enumerate(rows_dyn):
                    if ppid == pid and tt >= t_start:
                        rows_dyn[k] = (ppid, tt, fn, val, 1)

    dynamic = pd.DataFrame(
        rows_dyn, columns=["pid", "time", "feature_name", "value", "treated"]
    ).sort_values(["pid", "time", "feature_name"]).reset_index(drop=True)
    
    static = pd.DataFrame(
        rows_sta, columns=["pid", "age", "sex", "cci", "Tmax"]
    ).sort_values("pid")

    treatment = dynamic.groupby("pid")["treated"].max().reset_index().rename(
        columns={"treated": "treatment"}
    )
    static = static.merge(treatment, on="pid", how="left")

    time_to_event = dynamic.groupby("pid")["time"].max().reset_index().rename(
        columns={"time": "time_to_event"}
    )
    static = static.merge(time_to_event, on="pid", how="left")
    
    return dynamic, static


def simulate_with_known_trajectories(
    n_pat: int,
    scenario: str = "linear_decline",
    treatment_effect_on_slope: float = -0.3,
    disease: str = "ckd",
    disease_cfg = None,
    seed: int = 920,
    **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Simulate longitudinal data with known trajectory parameters.
    
    This extends simulate_dynamic_static() to return ground truth trajectory
    parameters for validation purposes.
    
    Parameters
    ----------
    n_pat : int
        Number of patients to simulate.
    scenario : str, default="linear_decline"
        Trajectory pattern to simulate:
        - "linear_decline": eGFR declines linearly, treatment slows decline
        - "nonlinear": HbA1c follows quadratic pattern
        - "heterogeneous": Treatment effect varies by baseline
    treatment_effect_on_slope : float, default=-0.3
        Effect of treatment on trajectory slope.
    seed : int
        Random seed for reproducibility.
    **kwargs
        Additional arguments passed to simulate_dynamic_static().
    
    Returns
    -------
    dynamic_df : pd.DataFrame
        Long-format dynamic data with columns: pid, time, feature_name, value, treated.
    static_df : pd.DataFrame
        Static covariates with columns: pid, age, sex, cci, treatment, Tmax, time_to_event.
    ground_truth : dict
        Dictionary containing:
        - scenario: str - scenario name
        - true_slopes: dict[pid, slope] - true slope for each patient
        - true_intercepts: dict[pid, intercept] - true baseline for each patient
        - treatment_effect_on_slope: float - known treatment effect
        - feature_params: dict - per-feature trajectory parameters
    """
    np.random.seed(seed)

    # Get or create disease config
    if disease_cfg is None:
        from traj_ps.config import DiseaseConfig
        if disease == "ckd":
            disease_cfg = DiseaseConfig.for_ckd()
        elif disease == "hiv":
            disease_cfg = DiseaseConfig.for_hiv()
        elif disease == "alzheimers":
            disease_cfg = DiseaseConfig.for_alzheimers()
        elif disease == "parkinsons":
            disease_cfg = DiseaseConfig.for_parkinsons()
        else:
            disease_cfg = DiseaseConfig.for_ckd()

     # Get features for this disease
    features = disease_cfg.covariate_names.copy() + [disease_cfg.primary_feature.name]
    primary_feature = disease_cfg.primary_feature.name
    label_map = disease_cfg.trajectory_type_map
    
    # Generate base data
    dynamic_df, static_df = simulate_dynamic_static(n_pat=n_pat, seed=seed, disease=disease,
                                                     disease_cfg=disease_cfg, features=features, **kwargs)
    
    # Initialize ground truth storage
    ground_truth = {
        'scenario': scenario,
        'disease': disease,
        'disease_cfg': disease_cfg,
        'primary_feature': primary_feature,
        'treatment_effect_on_slope': treatment_effect_on_slope,
        'true_slopes': {},
        'true_intercepts': {},
        'trajectory_types': {},
        'feature_params': {},
        'label_map': label_map
    }

    # Generate trajectory parameters based on scenario
    if scenario == "nonprogression":
        ground_truth = _simulate_nonprogression(
            dynamic_df, static_df, primary_feature, treatment_effect_on_slope, ground_truth
        )
    elif scenario == "linear_decline":
        ground_truth = _simulate_linear_decline(
            dynamic_df, static_df, primary_feature, treatment_effect_on_slope, ground_truth
        )
    elif scenario == "nonlinear":
        ground_truth = _simulate_nonlinear(
            dynamic_df, static_df, primary_feature, treatment_effect_on_slope, ground_truth
        )
    elif scenario == "heterogeneous":
        ground_truth = _simulate_heterogeneous(
            dynamic_df, static_df, primary_feature, treatment_effect_on_slope, ground_truth
        )
    elif scenario == "mixed":
        ground_truth = _simulate_mixed(
            dynamic_df, static_df, primary_feature, treatment_effect_on_slope, ground_truth
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    return dynamic_df, static_df, ground_truth


def _simulate_linear_decline(
    dynamic_df: pd.DataFrame,
    static_df: pd.DataFrame,
    primary_feature: str,
    treatment_effect: float,
    ground_truth: dict
) -> dict:
    """
    Simulate linear decline (disease-agnostic).
    
    Model: feature(t) = baseline + slope * t
    - slope reflects natural disease progression
    - treatment reduces decline magnitude
    """
    trajectory_types = {}
    
    # Get disease-specific params
    disease = ground_truth.get('disease', 'ckd')
    disease_cfg = ground_truth.get('disease_cfg')
    params = get_feature_params(disease, primary_feature)
    label_map = ground_truth.get('label_map', {})

    # Get thresholds from disease config
    if disease_cfg:
        flat_thr = disease_cfg.primary_feature.flat_threshold
        decline_thr = disease_cfg.primary_feature.decline_threshold
        nonlinear_gap = disease_cfg.primary_feature.nonlinear_gap
        direction = disease_cfg.primary_feature.direction
        higher_better = disease_cfg.primary_feature.higher_better
    else:
        # Fallback defaults (CKD)
        flat_thr = -1.0
        decline_thr = -2.0
        nonlinear_gap = 3.0
        direction = "decreasing"
        higher_better = True

    for pid in static_df["pid"].unique():
        is_treated = static_df.loc[static_df["pid"] == pid, "treatment"].iloc[0]
        
        baseline = np.random.normal(params["baseline_mean"], params["baseline_std"])
        
        # Generate slopes that clearly fall into "linear decline" category
        # For decreasing: slopes < decline_thr (e.g., < -2.0)
        # For increasing (worse): slopes > decline_thr (e.g., > 2.0)
        if direction == "decreasing":
            # Slopes more negative than decline_thr: range [decline_thr * 2, decline_thr * 1.25]
            base_slope = np.random.uniform(decline_thr * 2, decline_thr * 1.25)
        else:  # increasing
            # Slopes more positive than decline_thr: range [decline_thr * 1.25, decline_thr * 2]
            base_slope = np.random.uniform(decline_thr * 1.25, decline_thr * 2)
        
        # Treatment modifies slope
        if is_treated:
            # For decreasing: treatment makes slope less negative (adds positive value)
            # For increasing (worse): treatment makes slope less positive (subtracts)
            if direction == "decreasing" or (direction == "increasing" and higher_better):
                slope = base_slope - treatment_effect  # Makes decline slower
            else:  # increasing and worse
                slope = base_slope - treatment_effect  # Makes worsening slower
        else:
            slope = base_slope
        
        ground_truth['true_intercepts'][pid] = baseline
        ground_truth['true_slopes'][pid] = slope
        
        # Classify based on thresholds
        if direction == "decreasing":
            if np.mean([slope]) >= flat_thr and slope >= decline_thr:
                generic_type = 'nonprogression'
            elif slope < decline_thr:
                generic_type = 'linear'
            else:
                generic_type = 'linear'  # In-between
        else:  # increasing
            if np.mean([slope]) <= flat_thr and slope <= decline_thr:
                generic_type = 'nonprogression'
            elif slope > decline_thr:
                generic_type = 'linear'
            else:
                generic_type = 'linear'
        
        trajectory_types[pid] = label_map.get(generic_type, generic_type)
        
        # Update dynamic_df
        mask = (dynamic_df["pid"] == pid) & (dynamic_df["feature_name"] == primary_feature)
        times = dynamic_df.loc[mask, "time"].values
        
        noise = np.random.normal(0, params["noise_std"], size=len(times))
        true_values = baseline + slope * times + noise
        true_values = np.clip(true_values, params["min_val"], params["max_val"])
        
        dynamic_df.loc[mask, "value"] = true_values

    ground_truth['trajectory_types'] = trajectory_types
    
    ground_truth['feature_params'][primary_feature] = {
        'model': 'linear',
        'mean_baseline': params["baseline_mean"],
        'mean_slope_control': base_slope,
        'mean_slope_treated': base_slope - treatment_effect,
        'noise_std': params["noise_std"],
        'decline_threshold': decline_thr,
        'flat_threshold': flat_thr
    }
    
    return ground_truth


def _simulate_nonlinear(
    dynamic_df: pd.DataFrame,
    static_df: pd.DataFrame,
    primary_feature: str,
    treatment_effect: float,
    ground_truth: dict
) -> dict:
    """
    Simulate nonlinear trajectories with acceleration meeting nonlinear_gap threshold.
    """
    trajectory_types = {}
    
    disease = ground_truth.get('disease', 'ckd')
    disease_cfg = ground_truth.get('disease_cfg')
    params = get_feature_params(disease, primary_feature)
    label_map = ground_truth.get('label_map', {})
    
    # Get thresholds from disease config
    if disease_cfg:
        flat_thr = disease_cfg.primary_feature.flat_threshold
        decline_thr = disease_cfg.primary_feature.decline_threshold
        nonlinear_gap = disease_cfg.primary_feature.nonlinear_gap
        direction = disease_cfg.primary_feature.direction
        higher_better = disease_cfg.primary_feature.higher_better
    else:
        flat_thr = -1.0
        decline_thr = -2.0
        nonlinear_gap = 3.0
        direction = "decreasing"
        higher_better = True
    
    for pid in static_df["pid"].unique():
        is_treated = static_df.loc[static_df["pid"] == pid, "treatment"].iloc[0]
        
        baseline = np.random.normal(params["baseline_mean"], params["baseline_std"])
        
        # Generate trajectory with sufficient slope gap
        # For nonlinear: abs(fast_half_slope - slow_half_slope) > nonlinear_gap
        
        # Start with moderate linear term
        if direction == "decreasing":
            b1 = np.random.uniform(decline_thr * 1.5, decline_thr * 0.8)
        else:  # increasing
            b1 = np.random.uniform(decline_thr * 0.8, decline_thr * 1.5)
        
        # Quadratic term strong enough to create gap > nonlinear_gap
        # For trajectory over T years, slope(t) = b1 + 2*b2*t
        # At t=T: slope_max = b1 + 2*b2*T
        # At t=0: slope_min = b1
        # Gap = |slope_max - slope_min| = |2*b2*T|
        # Want: |2*b2*T| > nonlinear_gap
        T = 5.0  # typical follow-up
        min_b2 = nonlinear_gap / (2 * T) * 1.5  # 1.5x for safety margin
        
        if direction == "decreasing":
            b2 = -np.random.uniform(min_b2, min_b2 * 2)  # Accelerating decline
        else:  # increasing
            b2 = np.random.uniform(min_b2, min_b2 * 2)  # Accelerating worsening
        
        if is_treated:
            b1 = b1 - treatment_effect * 0.5
            b2 = b2 - treatment_effect * 0.3
        
        ground_truth['true_intercepts'][pid] = baseline
        ground_truth['true_slopes'][pid] = {'b1': b1, 'b2': b2}
        trajectory_types[pid] = label_map.get('nonlinear', 'nonlinear')
        
        mask = (dynamic_df["pid"] == pid) & (dynamic_df["feature_name"] == primary_feature)
        times = dynamic_df.loc[mask, "time"].values
        
        noise = np.random.normal(0, params["noise_std"], size=len(times))
        true_values = baseline + b1 * times + b2 * (times ** 2) + noise
        true_values = np.clip(true_values, params["min_val"], params["max_val"])
        
        dynamic_df.loc[mask, "value"] = true_values

    ground_truth['trajectory_types'] = trajectory_types  
    
    ground_truth['feature_params'][primary_feature] = {
        'model': 'quadratic',
        'mean_baseline': params["baseline_mean"],
        'mean_b1': b1,
        'mean_b2': b2,
        'nonlinear_gap_threshold': nonlinear_gap,
        'noise_std': params["noise_std"]
    }
    
    return ground_truth


def _simulate_heterogeneous(
    dynamic_df: pd.DataFrame,
    static_df: pd.DataFrame,
    primary_feature: str,
    treatment_effect: float,
    ground_truth: dict
) -> dict:
    """
    Simulate TRULY heterogeneous treatment effects.
    
    Strategy:
    - Control group: All have declining trajectories (homogeneous)
    - Treated group: Heterogeneous responses based on baseline severity
      * 30% strong responders → move to nonprogression
      * 50% moderate responders → remain in linear decline (slower)
      * 20% non-responders → same as control
    
    This creates a mix of trajectory types in the treated group.
    """
    trajectory_types = {}
    
    disease = ground_truth.get('disease', 'ckd')
    disease_cfg = ground_truth.get('disease_cfg')
    params = get_feature_params(disease, primary_feature)
    label_map = ground_truth.get('label_map', {})
    
    # Get thresholds from disease config
    if disease_cfg:
        flat_thr = disease_cfg.primary_feature.flat_threshold
        decline_thr = disease_cfg.primary_feature.decline_threshold
        direction = disease_cfg.primary_feature.direction
        higher_better = disease_cfg.primary_feature.higher_better
    else:
        flat_thr = -1.0
        decline_thr = -2.0
        direction = "decreasing"
        higher_better = True
    
    print(f"\n[Heterogeneous Simulation] Disease: {disease}, Direction: {direction}")
    print(f"[Heterogeneous Simulation] Thresholds: flat={flat_thr}, decline={decline_thr}")
    print(f"[Heterogeneous Simulation] Treatment effect: {treatment_effect}")
    
    response_stats = {
        'strong_responder': [],
        'moderate_responder': [],
        'non_responder': [],
        'control': []
    }
    response_types = {}  # Track response type per patient
    
    for pid in static_df["pid"].unique():
        is_treated = static_df.loc[static_df["pid"] == pid, "treatment"].iloc[0]
        
        baseline = np.random.normal(params["baseline_mean"], params["baseline_std"])
        
        # Start with decline-range slopes for ALL patients
        if direction == "decreasing":
            # For decreasing (eGFR, MMSE): negative slopes indicate decline
            base_slope = np.random.uniform(decline_thr * 2, decline_thr * 1.25)  # e.g., [-4.0, -2.5]
        else:  # increasing (worse) - CD4, MDS-UPDRS-III
            # For increasing worse: positive slopes indicate worsening
            base_slope = np.random.uniform(decline_thr * 1.25, decline_thr * 2)  # e.g., [2.5, 4.0]
        
        if is_treated:
            # Baseline-dependent heterogeneity: worse baseline → better response
            baseline_deviation = (params["baseline_mean"] - baseline) / params["baseline_std"]
            
            # Assign response type based on baseline severity
            # More severe (worse baseline) → higher chance of strong response
            if direction == "decreasing" or higher_better:
                # For eGFR/MMSE/CD4: lower baseline → worse disease → better response
                severity_score = -baseline_deviation  # Negative deviation = worse
            else:
                # For MDS-UPDRS-III: higher baseline → worse disease → better response
                severity_score = baseline_deviation  # Positive deviation = worse
            
            # Convert severity to response probability
            # High severity → more likely to be strong responder
            strong_prob = 0.15 + 0.30 * (1 / (1 + np.exp(-severity_score)))  # Sigmoid: 15-45%
            non_prob = 0.10 + 0.20 * (1 / (1 + np.exp(severity_score)))      # Inverse: 10-30%
            moderate_prob = 1 - strong_prob - non_prob                        # Remainder: 35-65%
            
            response_type = np.random.choice(
                ['strong_responder', 'moderate_responder', 'non_responder'],
                p=[strong_prob, moderate_prob, non_prob]
            )
            
            if response_type == 'strong_responder':
                # Strong response: move to nonprogression zone
                if direction == "decreasing":
                    # Make slope positive or slightly negative (stable/improving)
                    slope = np.random.uniform(flat_thr * 0.5, min(flat_thr + abs(flat_thr) * 0.8, -0.1))
                else:
                    # Make slope small positive or near zero (stable/slow worsening)
                    slope = np.random.uniform(max(flat_thr - abs(flat_thr) * 0.8, 0.1), flat_thr * 0.5)
                
            elif response_type == 'moderate_responder':
                # Moderate response: still declining but slower
                # Apply treatment effect to reduce decline magnitude
                if direction == "decreasing":
                    # Make slope less negative (slower decline)
                    slope = base_slope + treatment_effect * 0.6  # e.g., -3.5 + 1.2 = -2.3
                else:
                    # Make slope less positive (slower worsening)
                    slope = base_slope - treatment_effect * 0.6  # e.g., 3.5 - 1.2 = 2.3
            else:  # non_responder
                # No response: same as control (small random variation)
                slope = base_slope + np.random.normal(0, 0.2)
            
            response_stats[response_type].append(slope)
            response_types[pid] = response_type
        else:
            # Control: uniform decline
            slope = base_slope
            response_stats['control'].append(slope)
            response_types[pid] = 'control'
        
        ground_truth['true_intercepts'][pid] = baseline
        ground_truth['true_slopes'][pid] = slope
        ground_truth['response_types'] = response_types  # Store for analysis
        
        # Classify based on final slope and thresholds
        if direction == "decreasing":
            if slope >= flat_thr:
                generic_type = 'nonprogression'
            elif slope < decline_thr:
                generic_type = 'linear'
            else:
                # In-between zone: closer to which threshold?
                generic_type = 'nonprogression' if abs(slope - flat_thr) < abs(slope - decline_thr) else 'linear'
        else:  # increasing
            if slope <= flat_thr:
                generic_type = 'nonprogression'
            elif slope > decline_thr:
                generic_type = 'linear'
            else:
                generic_type = 'nonprogression' if abs(slope - flat_thr) < abs(slope - decline_thr) else 'linear'
        
        trajectory_types[pid] = label_map.get(generic_type, generic_type)
        
        # Update dynamic_df
        mask = (dynamic_df["pid"] == pid) & (dynamic_df["feature_name"] == primary_feature)
        times = dynamic_df.loc[mask, "time"].values
        
        noise = np.random.normal(0, params["noise_std"], size=len(times))
        true_values = baseline + slope * times + noise
        true_values = np.clip(true_values, params["min_val"], params["max_val"])
        
        dynamic_df.loc[mask, "value"] = true_values
    
    ground_truth['trajectory_types'] = trajectory_types
    
    # Print response distribution
    print(f"\n[Heterogeneous Simulation] Response Statistics:")
    for rtype, slopes in response_stats.items():
        if slopes:
            print(f"  {rtype:20s}: n={len(slopes):3d}, mean={np.mean(slopes):7.3f}, "
                  f"std={np.std(slopes):6.3f}, range=[{np.min(slopes):7.3f}, {np.max(slopes):7.3f}]")
    
    # Check trajectory type distribution
    type_counts = pd.Series(list(trajectory_types.values())).value_counts()
    print(f"\n[Heterogeneous Simulation] Trajectory Type Distribution:")
    for ttype, count in type_counts.items():
        print(f"  {ttype:30s}: {count:3d} ({100*count/len(trajectory_types):.1f}%)")
    
    # Verify we have heterogeneity
    all_slopes = [s for slopes in response_stats.values() for s in slopes]
    print(f"\n[Heterogeneous Simulation] Overall slope variance: {np.var(all_slopes):.6f}")
    if len(type_counts) < 2:
        print("  ⚠️  WARNING: Only one trajectory type detected! Heterogeneity failed.")
    else:
        print(f"  ✓ Success: {len(type_counts)} trajectory types detected")
    
    ground_truth['feature_params'][primary_feature] = {
        'model': 'heterogeneous_linear',
        'mean_baseline': params["baseline_mean"],
        'base_slope_range': (decline_thr * 2, decline_thr * 1.25) if direction == "decreasing" 
                            else (decline_thr * 1.25, decline_thr * 2),
        'base_treatment_effect': treatment_effect,
        'response_distribution': {k: len(v) for k, v in response_stats.items()},
        'actual_type_counts': dict(type_counts),
        'decline_threshold': decline_thr,
        'flat_threshold': flat_thr,
        'noise_std': params["noise_std"]
    }
    
    return ground_truth


def _simulate_nonprogression(
    dynamic_df: pd.DataFrame,
    static_df: pd.DataFrame,
    primary_feature: str, 
    treatment_effect: float,
    ground_truth: dict
) -> dict:
    """
    Simulate stable trajectories with both local and global checks.
    
    Nonprogression criteria (from flags_from_traj):
    1. Local: >= 80% of slopes satisfy the "not declining" threshold
       - For decreasing: slope >= decline_thr (not rapidly declining)
       - For increasing: slope <= decline_thr (not rapidly worsening)
    2. Global: mean slope satisfies the "flat" threshold
       - For decreasing: mean(slopes) >= flat_thr
       - For increasing: mean(slopes) <= flat_thr
    """
    trajectory_types = {}
    
    disease = ground_truth.get('disease', 'ckd')
    disease_cfg = ground_truth.get('disease_cfg')
    params = get_feature_params(disease, primary_feature)
    label_map = ground_truth.get('label_map', {})
    
    # Get thresholds from disease config
    if disease_cfg:
        flat_thr = disease_cfg.primary_feature.flat_threshold
        decline_thr = disease_cfg.primary_feature.decline_threshold
        direction = disease_cfg.primary_feature.direction
        higher_better = disease_cfg.primary_feature.higher_better
    else:
        flat_thr = -1.0
        decline_thr = -2.0
        direction = "decreasing"
        higher_better = True
    
    for pid in static_df["pid"].unique():
        baseline = np.random.normal(params["baseline_mean"], params["baseline_std"])
        
        # Generate slope that satisfies BOTH conditions:
        # For decreasing: slope >= flat_thr AND most local slopes >= decline_thr
        # For increasing: slope <= flat_thr AND most local slopes <= decline_thr
        
        if direction == "decreasing":
            # Range: [flat_thr, flat_thr + small_positive]
            # E.g., for flat_thr=-1.0: range [-1.0, -0.2]
            base_slope = np.random.uniform(flat_thr, flat_thr + abs(flat_thr) * 0.8)
        else:  # increasing
            # Range: [flat_thr - small_positive, flat_thr]
            # E.g., for flat_thr=0.5: range [-0.3, 0.5]
            base_slope = np.random.uniform(flat_thr - abs(flat_thr) * 0.8, flat_thr)
        
        # Add small variability to create month-to-month slopes
        # Most will be >= decline_thr (for decreasing) or <= decline_thr (for increasing)
        n_timepoints = 20  # Approximate number of measurements
        local_slopes = []
        
        for _ in range(n_timepoints):
            # Small random perturbation around base_slope
            noise_magnitude = abs(decline_thr - flat_thr) * 0.3
            local_slope = base_slope + np.random.uniform(-noise_magnitude, noise_magnitude)
            
            # Ensure >= 80% satisfy the local threshold
            if direction == "decreasing":
                # Keep local_slope >= decline_thr with high probability
                if local_slope < decline_thr and np.random.random() > 0.15:  # 85% above threshold
                    local_slope = decline_thr + abs(np.random.normal(0, 0.5))
            else:  # increasing
                # Keep local_slope <= decline_thr with high probability
                if local_slope > decline_thr and np.random.random() > 0.15:
                    local_slope = decline_thr - abs(np.random.normal(0, 0.5))
            
            local_slopes.append(local_slope)
        
        # Use mean of local slopes as the effective slope
        slope = np.mean(local_slopes)
        
        ground_truth['true_intercepts'][pid] = baseline
        ground_truth['true_slopes'][pid] = {'mean': slope, 'local': local_slopes}
        trajectory_types[pid] = label_map.get('nonprogression', 'nonprogression')
        
        # Update values
        mask = (dynamic_df["pid"] == pid) & (dynamic_df["feature_name"] == primary_feature)
        times = dynamic_df.loc[mask, "time"].values
        noise = np.random.normal(0, params["noise_std"], size=len(times))
        true_values = baseline + slope * times + noise
        true_values = np.clip(true_values, params["min_val"], params["max_val"])
        
        dynamic_df.loc[mask, "value"] = true_values
    
    ground_truth['trajectory_types'] = trajectory_types
    
    ground_truth['feature_params'][primary_feature] = {
        'model': 'stable',
        'mean_baseline': params["baseline_mean"],
        'mean_slope': 0.0,
        'slope_range': (flat_thr, flat_thr + abs(flat_thr) * 0.8) if direction == "decreasing" 
                       else (flat_thr - abs(flat_thr) * 0.8, flat_thr),
        'flat_threshold': flat_thr,
        'decline_threshold': decline_thr,
        'noise_std': params["noise_std"]
    }
    
    return ground_truth


def _simulate_mixed(
    dynamic_df: pd.DataFrame,
    static_df: pd.DataFrame,
    primary_feature: str,
    treatment_effect: float,
    ground_truth: dict
) -> dict:
    """
    Simulate mixed trajectories using disease-specific thresholds.
    """
    trajectory_types = {}
    n_patients = len(static_df)
    
    disease = ground_truth.get('disease', 'ckd')
    disease_cfg = ground_truth.get('disease_cfg')
    params = get_feature_params(disease, primary_feature)
    label_map = ground_truth.get('label_map', {})
    
    # Get thresholds from disease config
    if disease_cfg:
        flat_thr = disease_cfg.primary_feature.flat_threshold
        decline_thr = disease_cfg.primary_feature.decline_threshold
        nonlinear_gap = disease_cfg.primary_feature.nonlinear_gap
        direction = disease_cfg.primary_feature.direction
        higher_better = disease_cfg.primary_feature.higher_better
    else:
        flat_thr = -1.0
        decline_thr = -2.0
        nonlinear_gap = 3.0
        direction = "decreasing"
        higher_better = True
    
    print(f"\n[Mixed Simulation] Disease: {disease}, Direction: {direction}")
    print(f"[Mixed Simulation] Thresholds: flat={flat_thr}, decline={decline_thr}, gap={nonlinear_gap}")
    
    type_assignments = np.random.choice(
        ['nonprogression', 'linear', 'nonlinear'],
        size=n_patients,
        p=[0.3, 0.5, 0.2]
    )
    
    # Track statistics for diagnostics
    slope_stats = {'nonprogression': [], 'linear': [], 'nonlinear': []}
    
    for idx, pid in enumerate(static_df["pid"].unique()):
        is_treated = static_df.loc[static_df["pid"] == pid, "treatment"].iloc[0]
        generic_type = type_assignments[idx]
        
        baseline = np.random.normal(params["baseline_mean"], params["baseline_std"])
        
        if generic_type == 'nonprogression':
            # Use nonprogression slope range
            if direction == "decreasing":
                slope = np.random.uniform(flat_thr, min(flat_thr + abs(flat_thr) * 0.8, -0.1))
            else:
                slope = np.random.uniform(max(flat_thr - abs(flat_thr) * 0.8, 0.1), flat_thr)
            b2 = 0
            slope_stats['nonprogression'].append(slope)
            
        elif generic_type == 'linear':
            # Use linear decline slope range - WIDER RANGE for variation
            if direction == "decreasing":
                slope = np.random.uniform(decline_thr * 2, decline_thr * 1.1)  # More variation
            else:
                slope = np.random.uniform(decline_thr * 1.1, decline_thr * 2)
            
            if is_treated:
                slope = slope - treatment_effect
            b2 = 0
            slope_stats['linear'].append(slope)
            
        else:  # nonlinear
            # Use nonlinear parameters
            if direction == "decreasing":
                slope = np.random.uniform(decline_thr * 1.5, decline_thr * 0.8)
            else:
                slope = np.random.uniform(decline_thr * 0.8, decline_thr * 1.5)
            
            T = 5.0
            min_b2 = nonlinear_gap / (2 * T) * 1.5
            
            if direction == "decreasing":
                b2 = -np.random.uniform(min_b2, min_b2 * 2)
            else:
                b2 = np.random.uniform(min_b2, min_b2 * 2)
            
            if is_treated:
                slope = slope - treatment_effect * 0.5
                b2 = b2 - treatment_effect * 0.3
            
            slope_stats['nonlinear'].append(slope)
        
        ground_truth['true_intercepts'][pid] = baseline
        ground_truth['true_slopes'][pid] = {'linear': slope, 'quadratic': b2, 'type': generic_type}
        trajectory_types[pid] = label_map.get(generic_type, generic_type)
        
        mask = (dynamic_df["pid"] == pid) & (dynamic_df["feature_name"] == primary_feature)
        times = dynamic_df.loc[mask, "time"].values
        
        if generic_type == 'nonlinear':
            true_values = baseline + slope * times + b2 * (times ** 2)
        else:
            true_values = baseline + slope * times
        
        noise = np.random.normal(0, params["noise_std"], size=len(times))
        true_values += noise
        true_values = np.clip(true_values, params["min_val"], params["max_val"])
        
        dynamic_df.loc[mask, "value"] = true_values
    
    ground_truth['trajectory_types'] = trajectory_types
    
    # Print slope distribution diagnostics
    print(f"\n[Mixed Simulation] Slope Statistics:")
    for ttype, slopes in slope_stats.items():
        if slopes:
            print(f"  {ttype:20s}: n={len(slopes):3d}, mean={np.mean(slopes):7.3f}, "
                  f"std={np.std(slopes):6.3f}, range=[{np.min(slopes):7.3f}, {np.max(slopes):7.3f}]")
    
    # Check if slopes have variance
    all_slopes = [s for slopes in slope_stats.values() for s in slopes]
    print(f"\n[Mixed Simulation] Overall slope variance: {np.var(all_slopes):.6f}")
    if np.var(all_slopes) < 0.01:
        print("  WARNING: Very low slope variance! This will cause correlation=nan")
    
    ground_truth['feature_params'][primary_feature] = {
        'model': 'mixed',
        'mean_baseline': params["baseline_mean"],
        'noise_std': params["noise_std"],
        'type_proportions': {'nonprogression': 0.3, 'linear': 0.5, 'nonlinear': 0.2},
        'actual_type_counts': {k: len(v) for k, v in slope_stats.items()},
        'slope_stats': {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in slope_stats.items() if v},
        'flat_threshold': flat_thr,
        'decline_threshold': decline_thr,
        'nonlinear_gap': nonlinear_gap
    }
    
    return ground_truth
    
