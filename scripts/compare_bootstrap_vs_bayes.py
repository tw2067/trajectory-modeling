#!/usr/bin/env python3
"""
Compare Bootstrap vs Bayesian trajectory backends.

This script demonstrates the speed and similarity of both approaches.
"""

import pandas as pd
import numpy as np
import time
from traj_features.backends.bootstrap import BootstrapTrajPS, BootstrapConfig
from traj_features.backends.bayes import BayesianTrajPS, BayesConfig
from traj_features.backends.bayes.classify import pos_flags_from_traj


def generate_test_data(n_patients=100, n_obs_per_patient=20):
    """Generate synthetic P/F ratio data for testing."""
    data = []
    for pid in range(n_patients):
        # Generate trajectory with some noise
        times = np.sort(np.random.uniform(0, 7, n_obs_per_patient))
        
        # Random trajectory type
        traj_type = np.random.choice(['stable', 'improving', 'declining'])
        if traj_type == 'stable':
            values = 250 + np.random.normal(0, 15, n_obs_per_patient)
        elif traj_type == 'improving':
            values = 200 + times * 20 + np.random.normal(0, 15, n_obs_per_patient)
        else:
            values = 300 - times * 15 + np.random.normal(0, 15, n_obs_per_patient)
        
        for t, v in zip(times, values):
            data.append({
                'patientid': pid,
                'time_days': t,
                'time_day': int(t),
                'lab_value': v
            })
    
    return pd.DataFrame(data)


def run_bootstrap(data, config):
    """Run bootstrap backend and return timing."""
    model = BootstrapTrajPS(cfg=config)
    
    start = time.time()
    result = model.embed(data)
    elapsed = time.time() - start
    
    return result, elapsed


def run_bayesian(data, config):
    """Run Bayesian backend and return timing."""
    model = BayesianTrajPS(cfg=config)
    
    start = time.time()
    result = model.embed(data)
    elapsed = time.time() - start
    
    return result, elapsed


def compare_results(bootstrap_probs, bayes_probs):
    """Compare probability distributions from both methods."""
    # Merge on patient and time
    merged = bootstrap_probs.merge(
        bayes_probs,
        on=['patientid', 'time_day'],
        suffixes=('_boot', '_bayes')
    )
    
    # Compare probabilities
    prob_cols = ['trajtype_prolonged_nonprogression_prob', 
                 'trajtype_linear_increase_prob',
                 'trajtype_nonlinear_prob']
    
    print("\n📊 Probability Comparison:")
    for col in prob_cols:
        if f'{col}_boot' in merged.columns and f'{col}_bayes' in merged.columns:
            boot_vals = merged[f'{col}_boot']
            bayes_vals = merged[f'{col}_bayes']
            
            mae = np.mean(np.abs(boot_vals - bayes_vals))
            corr = np.corrcoef(boot_vals, bayes_vals)[0, 1]
            
            print(f"\n   {col}:")
            print(f"      Mean Absolute Error: {mae:.3f}")
            print(f"      Correlation: {corr:.3f}")
            print(f"      Bootstrap mean: {boot_vals.mean():.3f}")
            print(f"      Bayesian mean: {bayes_vals.mean():.3f}")


def main():
    print("="*80)
    print("Bootstrap vs Bayesian Trajectory Backends - Comparison")
    print("="*80)
    
    # Generate test data
    print("\n🔬 Generating synthetic test data...")
    data = generate_test_data(n_patients=50, n_obs_per_patient=15)
    print(f"   Patients: {data['patientid'].nunique()}")
    print(f"   Observations: {len(data)}")
    
    # Shared configuration
    common_params = {
        'window_years': 3.0,
        'min_points_per_window': 4,
        'grid_freq': 12,
        'flat_thr': 10.0,
        'decline_thr': 30.0,
        'nonlinear_gap': 20.0,
        'pids': 'patientid',
        'values': 'lab_value',
        'time_col': 'time_days',
        'windowing_col': 'time_day',
        'n_jobs': -1,
        'class_func': pos_flags_from_traj,
        'traj_types': ('prolonged_nonprogression', 'linear_increase', 'nonlinear'),
        'label_map': {
            'nonprogression': 'prolonged_nonprogression',
            'linear': 'linear_increase',
            'nonlinear': 'nonlinear'
        }
    }
    
    # Bootstrap configuration
    bootstrap_config = BootstrapConfig(
        **common_params,
        n_bootstrap=500,
        smoothing=None,
        progressbar=False
    )
    
    # Bayesian configuration (small for speed comparison)
    bayes_config = BayesConfig(
        **common_params,
        n_samples=100,  # Reduced for faster comparison
        tune=100,
        sampler='pymc',
        chains=2,
        cores=1,
        progressbar=False,
        use_gpu=False
    )
    
    # Run bootstrap
    print("\n⚡ Running Bootstrap backend...")
    boot_probs, boot_time = run_bootstrap(data, bootstrap_config)
    print(f"   ✓ Completed in {boot_time:.2f} seconds")
    print(f"   Windows processed: {len(boot_probs)}")
    
    # Run Bayesian
    print("\n🎲 Running Bayesian backend...")
    try:
        bayes_probs, bayes_time = run_bayesian(data, bayes_config)
        print(f"   ✓ Completed in {bayes_time:.2f} seconds")
        print(f"   Windows processed: {len(bayes_probs)}")
        
        # Compare
        print(f"\n⏱️  Speed Comparison:")
        print(f"   Bootstrap: {boot_time:.2f}s")
        print(f"   Bayesian:  {bayes_time:.2f}s")
        print(f"   Speedup:   {bayes_time / boot_time:.1f}x faster")
        
        compare_results(boot_probs, bayes_probs)
        
    except Exception as e:
        print(f"   ⚠️  Bayesian backend failed: {e}")
        print("\n   (This is normal if PyMC is not properly configured)")
    
    print("\n" + "="*80)
    print("✅ Comparison complete!")
    print("\nRecommendations:")
    print("   - Use Bootstrap for: Fast prototyping, large datasets, no GPU")
    print("   - Use Bayesian for: Full uncertainty quantification, small datasets")
    print("="*80)


if __name__ == '__main__':
    main()
