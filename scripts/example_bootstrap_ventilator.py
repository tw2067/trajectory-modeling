#!/usr/bin/env python3
"""
Example: Using Bootstrap trajectory backend for HiRiD ventilator analysis.

This is a drop-in replacement for the Bayesian backend that's much faster:
- No MCMC sampling or compilation
- Uses bootstrap resampling for uncertainty
- Typically 10-100x faster with comparable trajectory probabilities

Usage:
    python example_bootstrap_ventilator.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from traj_features.backends.bootstrap import BootstrapTrajPS, BootstrapConfig
from traj_features.backends.bayes.classify import pos_flags_from_traj

# Example configuration for P/F ratio trajectories (improvement-oriented)
config = BootstrapConfig(
    window_years=4.0,           # 4-day lookback window
    n_bootstrap=500,            # Number of bootstrap samples per window
    smoothing=None,             # Auto-select smoothing (or set to e.g., 1.0)
    min_points_per_window=4,    # Minimum observations required
    grid_freq=2,               # Evaluation grid size
    flat_thr=10.0,              # Stable threshold: ±10 P/F ratio/day
    decline_thr=30.0,           # Improvement threshold: >30 P/F ratio/day
    nonlinear_gap=20.0,         # Nonlinear detection threshold
    pids='patientid',
    values='lab_value',
    time_col='time_days',
    windowing_col='time_day',
    n_jobs=-1,                  # Use all available cores
    progressbar=True,
    class_func=pos_flags_from_traj,  # Improvement-oriented classification
    traj_types=('prolonged_nonprogression', 'linear_increase', 'nonlinear'),
    label_map={
        'nonprogression': 'prolonged_nonprogression',
        'linear': 'linear_increase',
        'nonlinear': 'nonlinear'
    }
)

def main():
    print("="*80)
    print("Bootstrap Trajectory Modeling - P/F Ratio (HiRiD Ventilator)")
    print("="*80)
    
    # Load data (adjust path as needed)
    input_path = '/home/gaga/tamarw1/trajectory-modeling/results/hirid/ventilator/pf_ratio_timeseries.csv'
    output_path = '/home/gaga/tamarw1/trajectory-modeling/results/hirid/ventilator/ventilator_trajectory_probs_bootstrap.csv'
    
    print(f"\n📊 Loading data from: {input_path}")
    pf_ts = pd.read_csv(input_path)
    
    print(f"   Patients: {pf_ts['patientid'].nunique():,}")
    print(f"   Measurements: {len(pf_ts):,}")
    
    # Prepare input
    traj_input = pf_ts[[
        'patientid',
        'time_days',
        'time_day',
        'lab_value'
    ]].dropna().sort_values(['patientid', 'time_days'])
    
    print(f"\n🔧 Configuration:")
    print(f"   Method: Bootstrap resampling")
    print(f"   Window: {config.window_years} days")
    print(f"   Bootstrap samples: {config.n_bootstrap}")
    print(f"   Thresholds: flat=±{config.flat_thr}, improvement=>{config.decline_thr}")
    print(f"   Parallel jobs: {config.n_jobs}")
    
    # Initialize model
    model = BootstrapTrajPS(cfg=config)
    
    print(f"\n⚡ Computing trajectory probabilities (bootstrap)...")
    trajectory_probs = model.embed(traj_input)
    
    print(f"\n✓ Computed {len(trajectory_probs):,} trajectory probabilities")
    print(f"   Output shape: {trajectory_probs.shape}")
    
    # Rename columns for consistency
    trajectory_probs = trajectory_probs.rename(columns={
        'trajtype_prolonged_nonprogression_prob': 'prob_stable',
        'trajtype_linear_increase_prob': 'prob_gradual_improvement',
        'trajtype_nonlinear_prob': 'prob_rapid_improvement'
    })
    
    # Merge with original data
    prob_cols = ['prob_stable', 'prob_gradual_improvement', 'prob_rapid_improvement']
    pf_with_probs = pf_ts.merge(
        trajectory_probs[['patientid', 'time_day', *prob_cols]],
        on=['patientid', 'time_day'],
        how='right'
    ).sort_values('time_days').drop_duplicates(['patientid', 'time_day'], keep='last')
    
    # Analyze distribution
    pf_with_probs['dominant_traj'] = pf_with_probs[prob_cols].idxmax(axis=1).str.replace('prob_', '')
    
    print(f"\n📊 Trajectory Distribution:")
    for traj in ['stable', 'gradual_improvement', 'rapid_improvement']:
        subset = pf_with_probs[pf_with_probs['dominant_traj'] == traj]
        if len(subset) > 0:
            print(f"   {traj.replace('_', ' ').title():25s}: {len(subset):5,} ({100*len(subset)/len(pf_with_probs):5.1f}%)")
            print(f"      Mean P/F ratio: {subset['lab_value'].mean():.1f}")
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_df = pf_with_probs[[
        'patientid', 'time_days', 'time_day',
        'lab_value', 'baseline_pf_ratio',
        *prob_cols
    ]]
    output_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved: {output_path}")
    print("="*80)

if __name__ == '__main__':
    main()
