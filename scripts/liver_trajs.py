"""
Compute trajectory probabilities for bilirubin in liver disease patients.

Bilirubin levels:
- Normal: <1.2 mg/dL
- Elevated: 1.2-3.0 mg/dL
- High: 3.0-12.0 mg/dL
- Severe (ACLF): ≥12.0 mg/dL

Trajectory interpretation:
- Increasing bilirubin = worsening hepatic function
- Stable bilirubin = steady state
- Decreasing bilirubin = improving liver function
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

# Limit OpenBLAS/MKL threading (prevents oversubscription with joblib)
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Force PyTensor to use C linker (more stable with parallel workers)
os.environ['PYTENSOR_FLAGS'] += ',cxx='

print(f"[Setup] PyTensor compile dir: ~/.pytensor_{job_id}")
# print(f"[Setup] Thread limits: OMP/MKL/OpenBLAS = 1")


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


def process_bilirubin_trajectories():
    """Process bilirubin trajectories for liver disease patients."""
    
    print(f"\n{'='*60}")
    print("Processing BILIRUBIN trajectories")
    print(f"{'='*60}")
    
    bili_ts = pd.read_csv('/home/gaga/tamarw1/trajectory-modeling/results/mimic/liver/bilirubin_timeseries.csv')
    
    bili_ts_clean = bili_ts.dropna(subset=['bilirubin']).sort_values(by=['hadm_id', 'time_days'])
    
    print(f"\n1. Input data:")
    print(f"   Rows: {len(bili_ts_clean):,}")
    print(f"   Patients: {bili_ts_clean['hadm_id'].nunique():,}")
    print(f"   Timepoints per patient: {bili_ts_clean.groupby('hadm_id')['time_days'].count().mean():.1f}")
    print(f"   Bilirubin range: {bili_ts_clean['bilirubin'].min():.2f} - {bili_ts_clean['bilirubin'].max():.2f} mg/dL")
    
    # Prepare data for BayesianTrajPS
    traj_input = bili_ts_clean[[
        'hadm_id',
        'time_days',
        'time_day',
        'bilirubin'
    ]].rename(columns={'bilirubin': 'lab_value'})
    
    # Bilirubin INCREASES = worsening (like creatinine, lactate, WBC)
    # Use pos_flags_from_traj with positive thresholds
    bili_config = BayesConfig(
        window_years=5.0,           # 5 days lookback
        df_basis=5,
        n_samples=200,
        tune=300,
        min_points_per_window=4,
        grid_freq=2,
        flat_thr=0.5,               # Stable: within ±0.5 mg/dL/day
        decline_thr=1.0,            # Gradual increase: 1-3 mg/dL/day
        nonlinear_gap=3.0,          # Rapid increase: >3 mg/dL deviation
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
        traj_types=('prolonged_nonprogression', 'linear_incline', 'nonlinear'),
    )
    
    print(f"\n2. BayesianTrajPS Configuration:")
    print(f"   Window: {bili_config.window_years} days")
    print(f"   Flat threshold: ±{bili_config.flat_thr} mg/dL/day (stable)")
    print(f"   Incline threshold: {bili_config.decline_thr} mg/dL/day (gradual increase)")
    print(f"   Nonlinear gap: {bili_config.nonlinear_gap} mg/dL (rapid increase)")
    print(f"   Chains: {bili_config.chains}")
    print(f"   Samples: {bili_config.n_samples} (warmup: {bili_config.tune})")
    
    # PRE-COMPILE PyTensor functions
    precompile_pytensor_functions(bili_config)
    
    # Initialize and compute trajectories
    traj_model = BayesianTrajPS(cfg=bili_config)
    
    print(f"\n3. Computing trajectory probabilities...")
    
    hadms = traj_input['hadm_id'].unique()
    npts = hadms.size
    n_batches = 10
    batch_size = npts // n_batches
    
    trajectory_probs_list = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < n_batches - 1 else npts
        subset_hadms = hadms[start_idx:end_idx]
        
        print(f"\n   Processing batch {i+1}/{n_batches} ({len(subset_hadms)} patients)...")
        
        batch_probs = traj_model.embed(traj_input[traj_input['hadm_id'].isin(subset_hadms)])
        batch_probs.to_csv(f'/home/gaga/tamarw1/trajectory-modeling/results/mimic/liver/bili_trajectory_probs_bayes_batch_{i+1}.csv', index=False)
        trajectory_probs_list.append(batch_probs)
        
        gc.collect()
    
    trajectory_probs = pd.concat(trajectory_probs_list, ignore_index=True)
    
    print(f"\n✓ Trajectory probabilities computed!")
    print(f"   Output shape: {trajectory_probs.shape}")
    print(f"   Columns: {list(trajectory_probs.columns)}")
    
    # Rename columns to match expected naming
    column_map = {
        'trajtype_prolonged_nonprogression_prob': 'prob_stable',
        'trajtype_linear_incline_prob': 'prob_gradual_increase',
        'trajtype_nonlinear_prob': 'prob_rapid_increase'
    }
    
    trajectory_probs = trajectory_probs.rename(columns=column_map)
    
    # Merge with original data
    bili_with_probs = bili_ts_clean.merge(
        trajectory_probs[['hadm_id', 'time_day', 'prob_stable', 'prob_gradual_increase', 'prob_rapid_increase']],
        on=['hadm_id', 'time_day'],
        how='right'
    ).sort_values(by='time_days').drop_duplicates(subset=['hadm_id', 'time_day'], keep='last')
    
    print(f"\n4. Merged trajectory probabilities:")
    print(f"   Rows: {len(bili_with_probs):,}")
    print(f"   Multiple time windows per patient: ✓")
    print(f"   Missing probabilities: {bili_with_probs['prob_stable'].isna().sum()}")
    
    # Dominant trajectory per time window
    bili_with_probs['dominant_traj'] = bili_with_probs[
        ['prob_stable', 'prob_gradual_increase', 'prob_rapid_increase']
    ].idxmax(axis=1).str.replace('prob_', '')
    
    print(f"\n5. Trajectory Distribution:")
    for traj in ['stable', 'gradual_increase', 'rapid_increase']:
        subset = bili_with_probs[bili_with_probs['dominant_traj'] == traj]
        if len(subset) > 0:
            print(f"   {traj.replace('_', ' ').title():20s}: {len(subset):5,} observations ({100*len(subset)/len(bili_with_probs):5.1f}%)")
            print(f"      Mean bilirubin: {subset['bilirubin'].mean():.2f} mg/dL")
    
    # Save output
    bili_output = bili_with_probs[[
        'hadm_id', 'time_days', 'time_day',
        'bilirubin', 'baseline_bilirubin', 'bili_fold_change',
        'prob_stable', 'prob_gradual_increase', 'prob_rapid_increase',
        'age', 'gender', 'hospital_expire_flag'
    ]]
    
    bili_output.to_csv('/home/gaga/tamarw1/trajectory-modeling/results/mimic/liver/bili_trajectory_probs_bayes.csv', index=False)
    print(f"\n✓ Saved: results/mimic/liver/bili_trajectory_probs_bayes.csv")
    
    return bili_with_probs


def main():
    """Main execution."""
    print("\n" + "=" * 60)
    print("LIVER FAILURE - BILIRUBIN TRAJECTORY ANALYSIS")
    print("=" * 60)
    
    bili_with_probs = process_bilirubin_trajectories()
    
    print("\n" + "=" * 60)
    print("TRAJECTORY COMPUTATION COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated bilirubin trajectory probabilities:")
    print(f"  Total timepoints: {len(bili_with_probs):,}")
    print(f"  Admissions: {bili_with_probs['hadm_id'].nunique():,}")
    print(f"\nInterpretation:")
    print(f"  - Stable: Bilirubin change ≤ ±0.5 mg/dL/day")
    print(f"  - Gradual increase: Bilirubin increasing 1-3 mg/dL/day")
    print(f"  - Rapid increase: Bilirubin increasing >3 mg/dL/day")
    print(f"\nIncreasing bilirubin = worsening hepatic function")


if __name__ == '__main__':
    main()
