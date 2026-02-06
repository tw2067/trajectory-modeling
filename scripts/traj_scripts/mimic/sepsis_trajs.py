"""
Compute trajectory probabilities for multiple biomarkers (sepsis progression)
- Lactate (primary marker)
- WBC (infection/inflammation)
- Platelets (coagulopathy/organ dysfunction)
"""

import numpy as np
import pandas as pd
from traj_ps.backends.bayes import BayesianTrajPS, BayesConfig
from traj_ps.backends.bayes.classify import pos_flags_from_traj, flags_from_traj
import importlib
import sys
import gc
import pymc as pm
from pytensor import tensor as at
from patsy import dmatrix
import os


# Use job-specific compile directory to avoid lock contention
job_id = os.environ.get('SLURM_JOB_ID', 'local')
os.environ['PYTENSOR_FLAGS'] = f"base_compiledir={os.path.expanduser('~')}/.pytensor_{job_id},optimizer=fast_compile,exception_verbosity=high"

# # Limit OpenBLAS/MKL threading (prevents oversubscription with joblib)
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Force PyTensor to use C linker (more stable with parallel workers)
os.environ['PYTENSOR_FLAGS'] += ',cxx='

print(f"[Setup] PyTensor compile dir: ~/.pytensor_{job_id}")
print(f"[Setup] Thread limits: OMP/MKL/OpenBLAS = 1")


def clear_cache():
    """Clear Python module cache and garbage collect."""
    modules_to_reload = [
        'traj_ps.backends.bayes.model',
        'traj_ps.backends.bayes.pipeline',
        'traj_ps.backends.bayes.classify',
        'traj_ps.backends.deep.model',
        'traj_ps.backends.gam.model',
    ]
    
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
    
    gc.collect()
    
    try:
        import jax
        jax.clear_caches()
        print("  [Cache] Cleared JAX cache")
    except ImportError:
        pass
    
    print("  [Cache] Cleared module cache and collected garbage")


def precompile_pytensor_functions(config):
    """
    Pre-compile PyTensor functions with a dummy run to avoid lock contention.
    """
    print("\n[Precompilation] Compiling PyTensor functions...")
    
    dummy_data = pd.DataFrame({
        'hadm_id': [1] * 10,
        'time_days': np.linspace(0, 5, 10),
        'lab_value': np.random.randn(10) + 2.0
    })
    
    try:
        df_window = dummy_data.copy()
        tmin, tmax = df_window['time_days'].min(), df_window['time_days'].max()
        span = max(1e-8, tmax - tmin)
        
        t_scaled = (df_window['time_days'] - tmin) / span
        df_window['time_scaled'] = t_scaled
        
        y_raw = df_window['lab_value'].to_numpy()
        y_mu = float(np.mean(y_raw))
        y_sd = float(np.std(y_raw)) if np.std(y_raw) > 0 else 1.0
        y_std = (y_raw - y_mu) / y_sd
        
        X = dmatrix(f"bs(x, df={config.df_basis}, include_intercept=True)",
                   {"x": t_scaled}, return_type='dataframe').to_numpy()
        
        with pm.Model() as m:
            beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=X.shape[1])
            sigma = pm.HalfNormal("sigma", 1.0)
            mu = pm.math.dot(X, beta)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_std)
            
            print("  [Precompilation] Compiling logp function...")
            logp_fn = m.compile_logp()
            test_point = m.initial_point()
            _ = logp_fn(test_point)
            
        print("  [Precompilation] ✓ PyTensor functions compiled successfully")
        return True
        
    except Exception as e:
        print(f"  [Precompilation] ⚠️ Warning: Could not precompile: {e}")
        return False


clear_cache()


def process_biomarker_trajectories(biomarker_name, ts_data, value_col, config, output_prefix, column_map=None):
    """
    Process trajectories for a single biomarker
    
    Parameters
    ----------
    biomarker_name : str
        Name of the biomarker (for display)
    ts_data : pd.DataFrame
        Time series data
    value_col : str
        Column name for the biomarker values
    config : BayesConfig
        Configuration for trajectory modeling
    output_prefix : str
        Prefix for output files
    column_map : dict, optional
        Mapping from trajectory type labels to output column names.
        If None, uses default mapping based on traj_types in config.
    """
    print(f"\n{'='*60}")
    print(f"Processing {biomarker_name.upper()} trajectories")
    print(f"{'='*60}")
    
    ts_clean = ts_data.dropna(subset=[value_col]).sort_values(by=['hadm_id', 'time_days'])
    
    print(f"   Rows: {len(ts_clean):,}")
    print(f"   Patients: {ts_clean['hadm_id'].nunique():,}")
    print(f"   Timepoints per patient: {ts_clean.groupby('hadm_id')['time_days'].count().mean():.1f}")
    
    # Prepare data for BayesianTrajPS
    traj_input = ts_clean[[
        'hadm_id',
        'time_days',
        'time_day',
        value_col
    ]].rename(columns={value_col: 'lab_value'})
    
    # PRE-COMPILE PyTensor functions (only once)
    if biomarker_name == 'lactate':
        precompile_pytensor_functions(config)
    
    # Initialize and compute trajectories
    traj_model = BayesianTrajPS(cfg=config)
    
    print(f"\n   Computing trajectory probabilities...")
    
    hadms = traj_input['hadm_id'].unique()
    npts = hadms.size
    n_batches = 10
    batch_size = npts // n_batches
    
    trajectory_probs_list = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < n_batches - 1 else npts
        subset_hadms = hadms[start_idx:end_idx]
        
        print(f"   Batch {i+1}/{n_batches} ({len(subset_hadms)} patients)...")
        
        batch_probs = traj_model.embed(traj_input[traj_input['hadm_id'].isin(subset_hadms)])
        batch_probs.to_csv(f'{output_prefix}_batch_{i+1}.csv', index=False)
        trajectory_probs_list.append(batch_probs)
        
        gc.collect()
    
    trajectory_probs = pd.concat(trajectory_probs_list, ignore_index=True)
    
    print(f"   ✓ Computed {len(trajectory_probs):,} trajectory probabilities")
    
    # Rename columns based on trajectory types in config
    if column_map is None:
        # Auto-generate column mapping from config.traj_types
        traj_types = config.traj_types
        column_map = {
            f'trajtype_{traj_types[0]}_prob': 'prob_stable',
            f'trajtype_{traj_types[1]}_prob': 'prob_gradual_increase',
            f'trajtype_{traj_types[2]}_prob': 'prob_rapid_increase'
        }
    
    trajectory_probs = trajectory_probs.rename(columns=column_map)
    
    # Merge with original data
    biomarker_with_probs = ts_clean.merge(
        trajectory_probs[['hadm_id', 'time_day', 'prob_stable', 'prob_gradual_increase', 'prob_rapid_increase']],
        on=['hadm_id', 'time_day'],
        how='right'
    ).sort_values(by='time_days').drop_duplicates(subset=['hadm_id', 'time_day'], keep='last')
    
    # Dominant trajectory per time window
    biomarker_with_probs['dominant_traj'] = biomarker_with_probs[
        ['prob_stable', 'prob_gradual_increase', 'prob_rapid_increase']
    ].idxmax(axis=1).str.replace('prob_', '')
    
    print(f"\n   Trajectory Distribution:")
    for traj in ['stable', 'gradual_increase', 'rapid_increase']:
        subset = biomarker_with_probs[biomarker_with_probs['dominant_traj'] == traj]
        if len(subset) > 0:
            print(f"     {traj.replace('_', ' ').title():20s}: {len(subset):5,} ({100*len(subset)/len(biomarker_with_probs):5.1f}%)")
            if value_col in subset.columns:
                print(f"        Mean {value_col}: {subset[value_col].mean():.2f}")
    
    return biomarker_with_probs


# ============================================================================
# 1. LACTATE Trajectories (Primary marker - perfusion)
# ============================================================================
print("\n1. Loading lactate time series...")
lactate_ts = pd.read_csv('/home/gaga/tamarw1/trajectory-modeling/results/mimic/sepsis/lactate_timeseries.csv')

# Lactate INCREASES = worsening (like creatinine)
lactate_config = BayesConfig(
    window_years=3.0,           # 3 days lookback
    df_basis=5,
    n_samples=200,
    tune=300,
    min_points_per_window=4,
    grid_freq=2,
    flat_thr=0.1,               # Stable: ±0.1 mmol/L/day
    decline_thr=0.3,            # Gradual increase: 0.3+ mmol/L/day (POSITIVE = worsening)
    nonlinear_gap=0.5,
    pids='hadm_id',
    values='lab_value',
    time_col='time_days',
    windowing_col='time_day',
    use_gpu=False,
    sampler='pymc',
    target_accept=0.99,
    chains=4,
    n_jobs=-1,
    class_func=pos_flags_from_traj,
    traj_types=('stable', 'slow_decline', 'rapid_decline'),
    label_map={'nonprogression': 'stable', 'linear': 'slow_decline', 'nonlinear': 'rapid_decline'}
)

lactate_with_probs = process_biomarker_trajectories(
    'lactate', 
    lactate_ts, 
    'lactate', 
    lactate_config,
    '/home/gaga/tamarw1/trajectory-modeling/results/mimic/sepsis/lactate_trajectory_probs_bayes'
)

# Save final lactate output
lactate_output = lactate_with_probs[[
    'subject_id', 'hadm_id', 'time_days', 'time_day',
    'lactate', 'baseline_lactate',
    'prob_stable', 'prob_gradual_increase', 'prob_rapid_increase'
]]
lactate_output.to_csv('/home/gaga/tamarw1/trajectory-modeling/results/mimic/sepsis/lactate_trajectory_probs_bayes.csv', index=False)
print(f"\n✓ Saved: results/mimic/sepsis/lactate_trajectory_probs_bayes.csv")


# ============================================================================
# 2. WBC Trajectories (Infection/inflammation)
# ============================================================================
print("\n\n2. Loading WBC time series...")
wbc_ts = pd.read_csv('/home/gaga/tamarw1/trajectory-modeling/results/mimic/sepsis/wbc_timeseries.csv')

# WBC can increase OR decrease (both can be abnormal)
# Use pos_flags to detect increases (leukocytosis, more common in sepsis)
wbc_config = BayesConfig(
    window_years=3.0,
    df_basis=5,
    n_samples=200,
    tune=300,
    min_points_per_window=4,
    grid_freq=2,
    flat_thr=1.0,               # Stable: ±1.0 K/μL/day
    decline_thr=3.0,            # Gradual increase: 3.0+ K/μL/day (leukocytosis)
    nonlinear_gap=2.0,
    pids='hadm_id',
    values='lab_value',
    time_col='time_days',
    windowing_col='time_day',
    use_gpu=False,
    sampler='pymc',
    target_accept=0.99,
    chains=4,
    n_jobs=-1,
    class_func=pos_flags_from_traj,
    traj_types=('stable', 'slow_decline', 'rapid_decline'),
    label_map={'nonprogression': 'stable', 'linear': 'slow_decline', 'nonlinear': 'rapid_decline'}
)

wbc_with_probs = process_biomarker_trajectories(
    'wbc',
    wbc_ts,
    'wbc',
    wbc_config,
    '/home/gaga/tamarw1/trajectory-modeling/results/mimic/sepsis/wbc_trajectory_probs_bayes'
)

# Save WBC output
wbc_output = wbc_with_probs[[
    'subject_id', 'hadm_id', 'time_days', 'time_day',
    'wbc', 'baseline_wbc',
    'prob_stable', 'prob_gradual_increase', 'prob_rapid_increase'
]]
wbc_output.to_csv('/home/gaga/tamarw1/trajectory-modeling/results/mimic/sepsis/wbc_trajectory_probs_bayes.csv', index=False)
print(f"\n✓ Saved: results/mimic/sepsis/wbc_trajectory_probs_bayes.csv")


# ============================================================================
# 3. PLATELET Trajectories (Coagulopathy/organ dysfunction)
# ============================================================================
print("\n\n3. Loading platelet time series...")
platelet_ts = pd.read_csv('/home/gaga/tamarw1/trajectory-modeling/results/mimic/sepsis/platelet_timeseries.csv')

# Platelets DECREASE = worsening (thrombocytopenia)
# Use flags_from_traj (negative slope function) for declining values
platelet_config = BayesConfig(
    window_years=3.0,
    df_basis=5,
    n_samples=200,
    tune=300,
    min_points_per_window=4,
    grid_freq=2,
    flat_thr=-20.0,             # Stable: >-20 K/μL/day (near zero or small negative)
    decline_thr=-50.0,          # Gradual decline: <-50 K/μL/day (more negative = worse)
    nonlinear_gap=30.0,
    pids='hadm_id',
    values='lab_value',
    time_col='time_days',
    windowing_col='time_day',
    use_gpu=False,
    sampler='pymc',
    target_accept=0.99,
    chains=4,
    n_jobs=-1,
    class_func=flags_from_traj,
    traj_types=('prolonged_nonprogression', 'linear_decline', 'nonlinear'),
    label_map={'nonprogression': 'prolonged_nonprogression', 'linear': 'linear_decline', 'nonlinear': 'nonlinear'}
)

platelet_with_probs = process_biomarker_trajectories(
    'platelet',
    platelet_ts,
    'platelet',
    platelet_config,
    '/home/gaga/tamarw1/trajectory-modeling/results/mimic/sepsis/platelet_trajectory_probs_bayes',
    column_map={
        'trajtype_prolonged_nonprogression_prob': 'prob_stable',
        'trajtype_linear_decline_prob': 'prob_gradual_increase',
        'trajtype_nonlinear_prob': 'prob_rapid_increase'
    }
)

# Save platelet output
platelet_output = platelet_with_probs[[
    'subject_id', 'hadm_id', 'time_days', 'time_day',
    'platelet', 'baseline_platelet',
    'prob_stable', 'prob_gradual_increase', 'prob_rapid_increase'
]]
platelet_output.to_csv('/home/gaga/tamarw1/trajectory-modeling/results/mimic/sepsis/platelet_trajectory_probs_bayes.csv', index=False)
print(f"\n✓ Saved: results/mimic/sepsis/platelet_trajectory_probs_bayes.csv")


print("\n" + "="*60)
print("ALL TRAJECTORY COMPUTATIONS COMPLETE!")
print("="*60)
print(f"\nGenerated 3 trajectory probability files:")
print(f"  1. Lactate:  {len(lactate_output):,} timepoints")
print(f"  2. WBC:      {len(wbc_output):,} timepoints")
print(f"  3. Platelet: {len(platelet_output):,} timepoints")
