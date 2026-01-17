"""
Compute trajectory probabilities for P/F ratio in ventilated patients.

P/F ratio (PaO2/FiO2 × 100):
- Higher values = better oxygenation
- Lower values = worse oxygenation 
- <200 = ARDS
- >300 = weaning readiness

Trajectory interpretation:
- Declining P/F ratio = worsening respiratory function
- Stable P/F ratio = steady state
- Improving P/F ratio = better oxygenation, potential weaning readiness
"""

import numpy as np
import pandas as pd
from traj_ps.backends.bayes import BayesianTrajPS, BayesConfig
from traj_ps.backends.bayes.classify import pos_flags_from_traj
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
        'lab_value': np.random.randn(10) + 300.0
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


def process_pf_trajectories():
    """Process P/F ratio trajectories for ventilated patients."""
    
    print(f"\n{'='*60}")
    print("Processing P/F RATIO trajectories")
    print(f"{'='*60}")
    
    pf_ts = pd.read_csv('/home/gaga/tamarw1/trajectory-modeling/results/mimic/ventilator/pf_ratio_timeseries.csv')
    
    pf_ts_clean = pf_ts.dropna(subset=['pf_ratio']).sort_values(by=['hadm_id', 'time_days'])
    
    print(f"   Rows: {len(pf_ts_clean):,}")
    print(f"   Patients: {pf_ts_clean['hadm_id'].nunique():,}")
    print(f"   Timepoints per patient: {pf_ts_clean.groupby('hadm_id')['time_days'].count().mean():.1f}")
    
    # Prepare data for BayesianTrajPS
    traj_input = pf_ts_clean[[
        'hadm_id',
        'time_days',
        'time_day',
        'pf_ratio'
    ]].rename(columns={'pf_ratio': 'lab_value'})
    
    # P/F ratio INCREASES = improvement (better oxygenation)
    # Use pos_flags_from_traj with positive thresholds
    pf_config = BayesConfig(
        window_years=3.0,           # 3 days lookback
        df_basis=5,
        n_samples=200,
        tune=300,
        min_points_per_window=4,
        grid_freq=2,
        flat_thr=10.0,              # Stable: ±10 points/day (near zero or small change)
        decline_thr=30.0,           # Gradual improvement: >30 points/day (positive slope)
        nonlinear_gap=20.0,
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
        traj_types=('prolonged_nonprogression', 'linear_increase', 'nonlinear'),
        label_map={'nonprogression': 'prolonged_nonprogression', 'linear': 'linear_increase', 'nonlinear': 'nonlinear'}
    )
    
    # PRE-COMPILE PyTensor functions
    precompile_pytensor_functions(pf_config)
    
    # Initialize and compute trajectories
    traj_model = BayesianTrajPS(cfg=pf_config)
    
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
        batch_probs.to_csv(f'/home/gaga/tamarw1/trajectory-modeling/results/mimic/ventilator/pf_trajectory_probs_bayes_batch_{i+1}.csv', index=False)
        trajectory_probs_list.append(batch_probs)
        
        gc.collect()
    
    trajectory_probs = pd.concat(trajectory_probs_list, ignore_index=True)
    
    print(f"   ✓ Computed {len(trajectory_probs):,} trajectory probabilities")
    
    # Rename columns
    column_map = {
        'trajtype_prolonged_nonprogression_prob': 'prob_stable',
        'trajtype_linear_increase_prob': 'prob_gradual_improvement',
        'trajtype_nonlinear_prob': 'prob_rapid_improvement'
    }
    
    trajectory_probs = trajectory_probs.rename(columns=column_map)
    
    # Merge with original data
    pf_with_probs = pf_ts_clean.merge(
        trajectory_probs[['hadm_id', 'time_day', 'prob_stable', 'prob_gradual_improvement', 'prob_rapid_improvement']],
        on=['hadm_id', 'time_day'],
        how='right'
    ).sort_values(by='time_days').drop_duplicates(subset=['hadm_id', 'time_day'], keep='last')

    # Dominant trajectory per time window
    pf_with_probs['dominant_traj'] = pf_with_probs[
        ['prob_stable', 'prob_gradual_improvement', 'prob_rapid_improvement']
    ].idxmax(axis=1).str.replace('prob_', '')

    print(f"\n   Trajectory Distribution:")
    for traj in ['stable', 'gradual_improvement', 'rapid_improvement']:
        subset = pf_with_probs[pf_with_probs['dominant_traj'] == traj]
        if len(subset) > 0:
            print(f"     {traj.replace('_', ' ').title():20s}: {len(subset):5,} ({100*len(subset)/len(pf_with_probs):5.1f}%)")
            if 'pf_ratio' in subset.columns:
                print(f"        Mean P/F ratio: {subset['pf_ratio'].mean():.2f}")
    
    # Save output
    pf_output = pf_with_probs[[
        'hadm_id', 'time_days', 'time_day',
        'pf_ratio', 'baseline_pf_ratio',
        'prob_stable', 'prob_gradual_improvement', 'prob_rapid_improvement'
    ]]
    pf_output.to_csv('/home/gaga/tamarw1/trajectory-modeling/results/mimic/ventilator/pf_trajectory_probs_bayes.csv', index=False)
    print(f"\n✓ Saved: results/mimic/ventilator/pf_trajectory_probs_bayes.csv")
    
    return pf_with_probs


def main():
    """Main execution."""
    print("\n" + "=" * 60)
    print("VENTILATOR WEANING - P/F RATIO TRAJECTORY ANALYSIS")
    print("=" * 60)
    
    pf_with_probs = process_pf_trajectories()
    
    print("\n" + "=" * 60)
    print("TRAJECTORY COMPUTATION COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated P/F ratio trajectory probabilities:")
    print(f"  Total timepoints: {len(pf_with_probs):,}")
    print(f"  Admissions: {pf_with_probs['hadm_id'].nunique():,}")


if __name__ == '__main__':
    main()
