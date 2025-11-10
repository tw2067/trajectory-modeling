import numpy as np, pandas as pd
DEFAULT_FEATURES = ("eGFR","Creatinine","SBP","DBP","HbA1c","MedA")

def simulate_dynamic_static(n_pat=120, features=DEFAULT_FEATURES, seed=920):
    rows_dyn, rows_sta = [], []
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    for i in range(n_pat):
        pid = f"P{i:05d}"
        T = rng.uniform(5.0, 10.0)
        trt_flag = rng.random()<0.5
        age, sex, cci = int(rng.integers(45,85)), int(rng.integers(0,2)), int(rng.poisson(2))
        rows_sta.append((pid, age, sex, cci, float(T)))

        def pp(rate):
            t, out = .0, []
            while True:
                t += rng.exponential(1/max(rate,1e-6))
                if t >= T: break
                out.append(t)
            return np.array(out) if out else np.array([rng.uniform(.1,T)])
        t_e, t_h, t_s, t_m = pp(2), pp(3), pp(4), pp(3)

        eg = np.clip(90 - 3.0*t_e + rng.normal(0,3,size=t_e.size), 5, 200)
        hb = np.clip(7.5 + 0.1*t_h + 0.2*np.sin(2*np.pi*t_h) + rng.normal(0,.3,size=t_h.size), 5, 14)
        sb = np.clip(120 + 5*np.sin(2*np.pi*t_s) + rng.normal(0,8,size=t_s.size), 90, 200)
        med=0
        for tm in t_m:
            if rng.random()<0.4: med=1-med
            rows_dyn.append((pid, float(tm), "MedA", float(med), 0))
        for t,v in zip(t_e,eg): rows_dyn.append((pid,float(t),"eGFR",float(v),0))
        for t,v in zip(t_h,hb): rows_dyn.append((pid,float(t),"HbA1c",float(v),0))
        for t,v in zip(t_s,sb): rows_dyn.append((pid,float(t),"SBP",float(v),0))

        grid = np.arange(0.0, T, 0.1)
        # LOCF approximations
        def locf(tt, vv, g):
            if tt.size==0: return np.zeros_like(g)
            idx = np.searchsorted(tt, g, side="right")-1; idx[idx<0]=0
            return vv[np.clip(idx,0,vv.size-1)]
        eg_loc = locf(t_e, eg, grid)
        score = -3.2 + 0.04*np.maximum(0, 90-eg_loc) + 0.2*sex + 0.05*cci
        p = 1 - np.exp(-np.exp(score))
        treated=0; t_start=None
        for gg,pp in zip(grid,p):
            if treated==0 and rng.random()<pp and trt_flag:
                treated=1; t_start=float(gg); break
        if t_start is not None:
            for k,(ppid,tt,fn,val,tr) in enumerate(rows_dyn):
                if ppid==pid and tt>=t_start:
                    rows_dyn[k]=(ppid,tt,fn,val,1)

    dynamic = pd.DataFrame(rows_dyn, columns=["pid","time","feature_name","value","treated"]) \
                .sort_values(["pid","time","feature_name"]).reset_index(drop=True)
    static  = pd.DataFrame(rows_sta, columns=["pid","age","sex","cci","Tmax"]).sort_values("pid")

    treatment = dynamic.groupby("pid")["treated"].max().reset_index().rename(columns={"treated":"treatment"})
    static = static.merge(treatment, on="pid", how="left")

    time_to_event = dynamic.groupby("pid")["time"].max().reset_index().rename(columns={"time":"time_to_event"})
    static = static.merge(time_to_event, on="pid", how="left")
    return dynamic, static


def simulate_with_known_trajectories(
    n_pat: int,
    scenario: str = "linear_decline",
    treatment_effect_on_slope: float = -0.3,
    seed: int = 42,
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
    
    # Generate base data
    dynamic_df, static_df = simulate_dynamic_static(n_pat=n_pat, seed=seed, **kwargs)
    
    # Initialize ground truth storage
    ground_truth = {
        'scenario': scenario,
        'treatment_effect_on_slope': treatment_effect_on_slope,
        'true_slopes': {},
        'true_intercepts': {},
        'feature_params': {}
    }
    
    # Generate trajectory parameters based on scenario
    if scenario == "linear_decline":
        ground_truth = _simulate_linear_decline(
            dynamic_df, static_df, treatment_effect_on_slope, ground_truth
        )
    elif scenario == "nonlinear":
        ground_truth = _simulate_nonlinear(
            dynamic_df, static_df, treatment_effect_on_slope, ground_truth
        )
    elif scenario == "heterogeneous":
        ground_truth = _simulate_heterogeneous(
            dynamic_df, static_df, treatment_effect_on_slope, ground_truth
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    return dynamic_df, static_df, ground_truth


def _simulate_linear_decline(
    dynamic_df: pd.DataFrame,
    static_df: pd.DataFrame,
    treatment_effect: float,
    ground_truth: dict
) -> dict:
    """
    Simulate linear decline in eGFR, with treatment slowing the decline.
    
    Model: eGFR(t) = baseline + slope * t
    - slope is negative (declining eGFR)
    - treatment reduces the magnitude of decline (makes slope less negative)
    """
    feature_name = "eGFR"
    trajectory_types = {}

    # Store per-patient parameters
    for pid in static_df["pid"].unique():
        is_treated = static_df.loc[static_df["pid"] == pid, "treatment"].iloc[0]
        
        # Baseline eGFR: Normal(60, 15)
        baseline = np.random.normal(60, 15)
        
        # Baseline slope: Normal(-2, 0.5) mL/min/year decline
        base_slope = np.random.normal(-2, 0.5)
        
        # Treatment modifies slope
        if is_treated:
            slope = base_slope + treatment_effect  # less negative (slower decline)
        else:
            slope = base_slope
        
        ground_truth['true_intercepts'][pid] = baseline
        ground_truth['true_slopes'][pid] = slope

        # Assign trajectory type based on slope characteristics
        # Linear decline: consistent negative slope
        if slope < -1.5:
            trajectory_types[pid] = 'linear_decline'
        elif slope < -0.5:
            trajectory_types[pid] = 'linear_decline'  # Still declining but slower
        else:
            trajectory_types[pid] = 'prolonged_nonprogression'  # Minimal decline
        
        # Update dynamic_df with true trajectory
        mask = (dynamic_df["pid"] == pid) & (dynamic_df["feature_name"] == feature_name)
        times = dynamic_df.loc[mask, "time"].values
        
        # Add noise to observations
        noise = np.random.normal(0, 3, size=len(times))
        true_values = baseline + slope * times + noise
        
        dynamic_df.loc[mask, "value"] = true_values

    ground_truth['trajectory_types'] = trajectory_types
    
    # Store feature-level parameters
    ground_truth['feature_params'][feature_name] = {
        'model': 'linear',
        'mean_baseline': 60,
        'mean_slope_control': -2,
        'mean_slope_treated': -2 + treatment_effect,
        'noise_std': 3
    }
    
    return ground_truth


def _simulate_nonlinear(
    dynamic_df: pd.DataFrame,
    static_df: pd.DataFrame,
    treatment_effect: float,
    ground_truth: dict
) -> dict:
    """
    Simulate nonlinear (quadratic) trajectory in HbA1c.
    
    Model: HbA1c(t) = baseline + b1*t + b2*t^2
    - Treatment affects both linear and quadratic terms
    """
    feature_name = "HbA1c"
    trajectory_types = {}
    
    for pid in static_df["pid"].unique():
        is_treated = static_df.loc[static_df["pid"] == pid, "treatment"].iloc[0]
        
        # Baseline HbA1c: Normal(7.5, 1)
        baseline = np.random.normal(7.5, 1)
        
        # Linear term: slight increase
        b1 = np.random.normal(0.5, 0.2)
        
        # Quadratic term: acceleration
        b2 = np.random.normal(0.1, 0.05)
        
        # Treatment reduces both linear and quadratic growth
        if is_treated:
            b1 = b1 - treatment_effect
            b2 = b2 - treatment_effect * 0.5
        
        ground_truth['true_intercepts'][pid] = baseline
        ground_truth['true_slopes'][pid] = {'b1': b1, 'b2': b2}

        # All patients have nonlinear trajectories in this scenario
        trajectory_types[pid] = 'nonlinear'
        
        # Update dynamic_df
        mask = (dynamic_df["pid"] == pid) & (dynamic_df["feature_name"] == feature_name)
        times = dynamic_df.loc[mask, "time"].values
        
        noise = np.random.normal(0, 0.3, size=len(times))
        true_values = baseline + b1 * times + b2 * (times ** 2) + noise
        
        dynamic_df.loc[mask, "value"] = true_values

    ground_truth['trajectory_types'] = trajectory_types  
    
    ground_truth['feature_params'][feature_name] = {
        'model': 'quadratic',
        'mean_baseline': 7.5,
        'mean_b1_control': 0.5,
        'mean_b2_control': 0.1,
        'noise_std': 0.3
    }
    
    return ground_truth


def _simulate_heterogeneous(
    dynamic_df: pd.DataFrame,
    static_df: pd.DataFrame,
    treatment_effect: float,
    ground_truth: dict
) -> dict:
    """
    Simulate heterogeneous treatment effects.
    
    Treatment effect on eGFR slope varies by baseline eGFR:
    - Low baseline → larger treatment benefit
    - High baseline → smaller treatment benefit
    """
    feature_name = "eGFR"
    trajectory_types = {}
    
    for pid in static_df["pid"].unique():
        is_treated = static_df.loc[static_df["pid"] == pid, "treatment"].iloc[0]
        
        # Baseline eGFR
        baseline = np.random.normal(60, 15)
        
        # Base slope
        base_slope = np.random.normal(-2, 0.5)
        
        # Heterogeneous treatment effect: stronger for lower baseline
        if is_treated:
            # Effect decreases as baseline increases
            het_effect = treatment_effect * (1 + (60 - baseline) / 30)
            slope = base_slope + het_effect
        else:
            slope = base_slope
        
        ground_truth['true_intercepts'][pid] = baseline
        ground_truth['true_slopes'][pid] = slope


        # Mix of trajectory types based on final slope
        if abs(slope) < 0.5:
            trajectory_types[pid] = 'prolonged_nonprogression'
        elif abs(slope) < 1.5:
            trajectory_types[pid] = 'linear_decline'
        else:
            # Assign some as nonlinear for heterogeneity
            if np.random.random() < 0.3:
                trajectory_types[pid] = 'nonlinear'
            else:
                trajectory_types[pid] = 'linear_decline'

        # Update dynamic_df
        mask = (dynamic_df["pid"] == pid) & (dynamic_df["feature_name"] == feature_name)
        times = dynamic_df.loc[mask, "time"].values
        
        noise = np.random.normal(0, 3, size=len(times))
        true_values = baseline + slope * times + noise
        
        dynamic_df.loc[mask, "value"] = true_values
    
    ground_truth['trajectory_types'] = trajectory_types
    
    ground_truth['feature_params'][feature_name] = {
        'model': 'heterogeneous_linear',
        'mean_baseline': 60,
        'base_treatment_effect': treatment_effect,
        'heterogeneity_factor': 'baseline_dependent',
        'noise_std': 3
    }
    
    return ground_truth
