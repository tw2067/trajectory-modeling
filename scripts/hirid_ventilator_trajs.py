#!/usr/bin/env python3
"""
Compute P/F ratio trajectory probabilities for HiRiD ventilator discontinuation cohort.

This script can be run via SLURM to utilize multiple CPUs for faster processing.

Usage:
    python hirid_ventilator_trajs.py --input <pf_ratio_timeseries.csv> --output <trajectory_probs.csv>
"""

import pandas as pd
import numpy as np
import argparse
import os
import gc
from pathlib import Path
from traj_ps.backends.bayes import BayesianTrajPS, BayesConfig
from traj_ps.backends.bayes.classify import pos_flags_from_traj

# Use job-specific compile directory to avoid lock contention
job_id = os.environ.get('SLURM_JOB_ID', 'local')
os.environ['PYTENSOR_FLAGS'] = f"base_compiledir={os.path.expanduser('~')}/.pytensor_{job_id},optimizer=fast_compile,exception_verbosity=high"

# Limit threading to prevent oversubscription
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Force PyTensor to use C linker (more stable with parallel workers)
os.environ['PYTENSOR_FLAGS'] += ',cxx='

print(f"[Setup] PyTensor compile dir: ~/.pytensor_{job_id}")
print(f"[Setup] Thread limits: OMP/MKL/OpenBLAS = 1")


def main():
    parser = argparse.ArgumentParser(description='Compute HiRiD ventilator trajectory probabilities')
    parser.add_argument('--input', type=str, 
                       default='../results/hirid/ventilator/pf_ratio_timeseries.csv',
                       help='Path to P/F ratio time series CSV file')
    parser.add_argument('--output', type=str,
                       default='../results/hirid/ventilator/ventilator_trajectory_probs_bayes.csv',
                       help='Path to save trajectory probabilities')
    parser.add_argument('--window-days', type=float, default=3.0,
                       help='Lookback window in days (default: 3.0, aligned with MIMIC ventilator)')
    parser.add_argument('--flat-thr', type=float, default=10.0,
                       help='Stable threshold for P/F ratio/day (near-zero change; default: 10.0)')
    parser.add_argument('--decline-thr', type=float, default=30.0,
                       help='Increase threshold for P/F ratio/day (default: 30.0, positive slope for improvement)')
    parser.add_argument('--n-batches', type=int, default=10,
                       help='Number of batches for processing (default: 10)')
    parser.add_argument('--sampler', type=str, choices=['pymc', 'numpyro', 'nutpie'], default='pymc',
                       help='Sampler backend for BayesianTrajPS (default: pymc)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("HiRiD Ventilator Trajectory Modeling (P/F Ratio)")
    print("="*80)
    
    # Load P/F ratio time series
    print(f"\n📊 Loading data from: {args.input}")
    pf_ts = pd.read_csv(args.input)
    
    print(f"   Patients: {pf_ts['patientid'].nunique():,}")
    print(f"   Measurements: {len(pf_ts):,}")
    print(f"   Mean measurements/patient: {len(pf_ts) / pf_ts['patientid'].nunique():.1f}")
    
    # Prepare data for BayesianTrajPS
    traj_input = pf_ts[[
        'patientid',
        'time_days',
        'time_day',
        'lab_value'
    ]].copy()
    
    traj_input = traj_input.dropna(subset=['lab_value']).sort_values(by=['patientid', 'time_days'])
    
    print(f"\n1. Input data:")
    print(f"   Rows: {len(traj_input):,}")
    print(f"   Patients: {traj_input['patientid'].nunique():,}")
    print(f"   Timepoints per patient: {traj_input.groupby('patientid')['time_days'].count().mean():.1f}")
    
    # Configure Bayesian trajectory model
    config = BayesConfig(
        window_years=args.window_days,
        df_basis=5,
        n_samples=200,
        tune=300,
        min_points_per_window=4,
        grid_freq=2,
        flat_thr=args.flat_thr,
        decline_thr=args.decline_thr,
        nonlinear_gap=20.0,
        pids='patientid',
        values='lab_value',
        time_col='time_days',
        windowing_col='time_day',
        use_gpu=False,
        sampler=args.sampler,
        target_accept=0.99,
        chains=4,
        n_jobs=-1,
        class_func=pos_flags_from_traj,
        traj_types=('prolonged_nonprogression', 'linear_increase', 'nonlinear'),
        label_map={'nonprogression': 'prolonged_nonprogression', 'linear': 'linear_increase', 'nonlinear': 'nonlinear'}
    )
    
    print(f"\n2. BayesianTrajPS Configuration:")
    print(f"   Window: {args.window_days} days")
    print(f"   Chains: {config.chains}")
    print(f"   Samples: {config.n_samples} (warmup: {config.tune})")
    print(f"   Stable threshold: ±{args.flat_thr} P/F ratio/day")
    print(f"   Improvement threshold: >{args.decline_thr} P/F ratio/day")
    
    # Initialize model
    traj_model = BayesianTrajPS(cfg=config)
    
    print(f"\n3. Computing trajectory probabilities...")
    
    # Process in batches to manage memory
    patients = traj_input['patientid'].unique()
    npts = patients.size
    n_batches = args.n_batches
    batch_size = npts // n_batches
    
    trajectory_probs_list = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < n_batches - 1 else npts
        subset_patients = patients[start_idx:end_idx]
        
        print(f"\n   Processing batch {i+1}/{n_batches} ({len(subset_patients)} patients)...")
        
        batch_probs = traj_model.embed(traj_input[traj_input['patientid'].isin(subset_patients)])
        trajectory_probs_list.append(batch_probs)
        
        # Clear memory after each batch
        gc.collect()
    
    trajectory_probs = pd.concat(trajectory_probs_list, ignore_index=True)
    
    print(f"\n✓ Trajectory probabilities computed!")
    print(f"   Output shape: {trajectory_probs.shape}")
    
    # Rename trajectory type columns to consistent names
    trajectory_probs = trajectory_probs.rename(columns={
        'trajtype_prolonged_nonprogression_prob': 'prob_stable',
        'trajtype_linear_increase_prob': 'prob_gradual_improvement',
        'trajtype_nonlinear_prob': 'prob_rapid_improvement'
    })
    
    # Merge with original data to add metadata
    prob_cols = ['prob_stable', 'prob_gradual_improvement', 'prob_rapid_improvement']
    pf_with_probs = pf_ts.merge(
        trajectory_probs[['patientid', 'time_day', *prob_cols]],
        on=['patientid', 'time_day'],
        how='right'
    ).sort_values(by='time_days').drop_duplicates(subset=['patientid', 'time_day'], keep='last')
    
    print(f"\n4. Merged trajectory probabilities:")
    print(f"   Rows: {len(pf_with_probs):,}")
    print(f"   Missing probabilities: {pf_with_probs['prob_stable'].isna().sum()}")
    
    # Dominant trajectory per time window
    pf_with_probs['dominant_traj'] = pf_with_probs[prob_cols].idxmax(axis=1).str.replace('prob_', '')
    
    print(f"\n5. Trajectory Distribution:")
    for traj in ['stable', 'gradual_improvement', 'rapid_improvement']:
        subset = pf_with_probs[pf_with_probs['dominant_traj'] == traj]
        if len(subset) > 0:
            print(f"   {traj.replace('_', ' ').title():20s}: {len(subset):5,} observations ({100*len(subset)/len(pf_with_probs):5.1f}%)")
            print(f"      Mean P/F ratio: {subset['lab_value'].mean():.2f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    trajectory_probs_full = pf_with_probs[[
        'patientid', 'time_days', 'time_day',
        'lab_value', 'baseline_pf_ratio',
        *prob_cols
    ]]
    
    trajectory_probs_full.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved: {output_path}")
    print("="*80)
    print("\n✅ Done! Continue in HiRiD_Ventilator_traj_probs.ipynb to load results and visualize")

if __name__ == '__main__':
    main()
