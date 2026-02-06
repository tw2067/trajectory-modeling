import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
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

# Limit OpenBLAS/MKL threading (prevents oversubscription with joblib)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Force PyTensor to use C linker (more stable with parallel workers)
os.environ['PYTENSOR_FLAGS'] += ',cxx='

print(f"[Setup] PyTensor compile dir: ~/.pytensor_{job_id}")
print(f"[Setup] Thread limits: OMP/MKL/OpenBLAS = 1")



def clear_cache():
    """Clear Python module cache and garbage collect."""
    # Clear imported modules related to backends
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
    
    # Force garbage collection
    gc.collect()
    
    # Clear JAX cache if using GPU
    try:
        import jax
        jax.clear_caches()
        print("  [Cache] Cleared JAX cache")
    except ImportError:
        pass
    
    # Clear PyMC cache
    try:
        import pymc as pm
        # PyMC doesn't have explicit cache clearing, but gc should help
        print("  [Cache] Garbage collected PyMC objects")
    except ImportError:
        pass
    
    print("  [Cache] Cleared module cache and collected garbage")

clear_cache()


def precompile_pytensor_functions(config):
    """
    Pre-compile PyTensor functions with a dummy run to avoid lock contention.
    This ensures all workers can reuse compiled code.
    """
    print("\n[Precompilation] Compiling PyTensor functions...")
    
    # Create a small dummy dataset for compilation
    dummy_data = pd.DataFrame({
        'hadm_id': [1] * 10,
        'time_days': np.linspace(0, 5, 10),
        'lab_value': np.random.randn(10) + 2.0
    })
    
    # Create dummy model to trigger compilation
    try:
        df_window = dummy_data.copy()
        tmin, tmax = df_window['time_days'].min(), df_window['time_days'].max()
        span = max(1e-8, tmax - tmin)
        
        # Scale time
        t_scaled = (df_window['time_days'] - tmin) / span
        df_window['time_scaled'] = t_scaled
        
        # Get values
        y_raw = df_window['lab_value'].to_numpy()
        y_mu = float(np.mean(y_raw))
        y_sd = float(np.std(y_raw)) if np.std(y_raw) > 0 else 1.0
        y_std = (y_raw - y_mu) / y_sd
        
        # Build design matrix
        X = dmatrix(f"bs(x, df={config.df_basis}, include_intercept=True)",
                   {"x": t_scaled}, return_type='dataframe').to_numpy()
        
        # Build PyMC model and compile
        with pm.Model() as m:
            beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=X.shape[1])
            sigma = pm.HalfNormal("sigma", 1.0)
            mu = pm.math.dot(X, beta)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_std)
            
            # Compile (don't sample)
            print("  [Precompilation] Compiling logp function...")
            logp_fn = m.compile_logp()
            test_point = m.initial_point()
            _ = logp_fn(test_point)
            
        print("  [Precompilation] ✓ PyTensor functions compiled successfully")
        return True
        
    except Exception as e:
        print(f"  [Precompilation] ⚠️ Warning: Could not precompile: {e}")
        return False

creatinine_ts = pd.read_csv('/home/gaga/tamarw1/trajectory-modeling/results/mimic/aki/creatinine_timeseries.csv')

creatinine_ts = creatinine_ts.dropna(subset=['creatinine']).sort_values(by=['hadm_id', 'time_days'])


# Prepare data for BayesianTrajPS
traj_input = creatinine_ts[[
    'hadm_id',
    'time_days',
    'time_day',
    'creatinine'
]].rename(columns={
    'creatinine': 'lab_value'
})

print(f"\n1. Input data:")
print(f"   Rows: {len(traj_input):,}")
print(f"   Patients: {traj_input['hadm_id'].nunique():,}")
print(f"   Timepoints per patient: {traj_input.groupby('hadm_id')['time_days'].count().mean():.1f}")

# Configure BayesianTrajPS
config = BayesConfig(
    window_years=7.0,
    df_basis=5,
    n_samples=200,              # Posterior samples (reduce from 1000 for speed)
    tune=300,                   # MCMC tuning iterations
    min_points_per_window=6,
    grid_freq=2,
    flat_thr=0.1,
    decline_thr=0.3,
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
    traj_types=('prolonged_nonprogression', 'linear_decline', 'nonlinear'),
)

print(f"\n2. BayesianTrajPS Configuration:")
print(f"   Window: {config.window_years} days")
print(f"   Chains: {config.chains}")
print(f"   Samples: {config.n_samples} (warmup: {config.tune})")

# PRE-COMPILE PyTensor functions
precompile_pytensor_functions(config)

# Initialize and compute trajectories
traj_model = BayesianTrajPS(cfg=config)


print(f"\n3. Computing trajectory probabilities...")

hadms = traj_input['hadm_id'].unique()
npts = hadms.size
n_batches = 10  # Increased from 5 for better memory management
batch_size = npts // n_batches

trajectory_probs_list = []

for i in range(n_batches):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size if i < n_batches - 1 else npts
    subset_hadms = hadms[start_idx:end_idx]
    
    print(f"\n   Processing batch {i+1}/{n_batches} ({len(subset_hadms)} patients)...")
    
    batch_probs = traj_model.embed(traj_input[traj_input['hadm_id'].isin(subset_hadms)])
    batch_probs.to_csv(f'/home/gaga/tamarw1/trajectory-modeling/results/mimic/aki/aki_trajectory_probs_bayes_batch_{i+1}.csv', index=False)
    trajectory_probs_list.append(batch_probs)
    
    # Clear memory after each batch
    gc.collect()

trajectory_probs = pd.concat(trajectory_probs_list, ignore_index=True)

print(f"\n✓ Trajectory probabilities computed!")
print(f"   Output shape: {trajectory_probs.shape}")
print(f"   Columns: {list(trajectory_probs.columns)}")

# Rename columns

trajectory_probs = trajectory_probs.rename(columns={
    'trajtype_prolonged_nonprogression_prob': 'prob_stable',
    'trajtype_linear_decline_prob': 'prob_gradual_increase',
    'trajtype_nonlinear_prob': 'prob_rapid_increase'
})


creatinine_with_probs = creatinine_ts.merge(
    trajectory_probs[['hadm_id', 'time_day', 'prob_stable', 'prob_gradual_increase', 'prob_rapid_increase']],
    on=['hadm_id', 'time_day'],
    how='right'
).sort_values(by='time_days').drop_duplicates(subset=['hadm_id', 'time_day'], keep='last')

print(f"\n4. Merged trajectory probabilities:")
print(f"   Rows: {len(creatinine_with_probs):,}")
print(f"   Multiple time windows per patient: ✓")
print(f"   Missing probabilities: {creatinine_with_probs['prob_stable'].isna().sum()}")

# Dominant trajectory per time window
creatinine_with_probs['dominant_traj'] = creatinine_with_probs[
    ['prob_stable', 'prob_gradual_increase', 'prob_rapid_increase']
].idxmax(axis=1).str.replace('prob_', '')

print(f"\n5. Trajectory Distribution:")
for traj in ['stable', 'gradual_increase', 'rapid_increase']:
    subset = creatinine_with_probs[creatinine_with_probs['dominant_traj'] == traj]
    if len(subset) > 0:
        print(f"   {traj.replace('_', ' ').title():20s}: {len(subset):5,} observations ({100*len(subset)/len(creatinine_with_probs):5.1f}%)")
        print(f"      Mean creatinine: {subset['creatinine'].mean():.2f} mg/dL")

# Save
trajectory_probs_full = creatinine_with_probs[['subject_id', 'hadm_id', 'time_days', 'time_day',
    'creatinine', 'baseline_creatinine', 'creat_fold_change',
    'prob_stable', 'prob_gradual_increase', 'prob_rapid_increase',
    'age', 'gender', 'hospital_expire_flag'
]]

trajectory_probs_full.to_csv('/home/gaga/tamarw1/trajectory-modeling/results/mimic/aki/aki_trajectory_probs_bayes.csv', index=False)
print(f"\n✓ Saved: results/mimic/aki/aki_trajectory_probs_bayes.csv")