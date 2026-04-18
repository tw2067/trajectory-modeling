"""
Compute creatinine trajectory probabilities for HiRiD AKI cohort.

This script can be run via SLURM to utilize multiple CPUs for faster processing.

Usage:
    python hirid_aki_trajs.py --input <creat_timeseries.csv> --output <trajectory_probs.csv>
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
    parser = argparse.ArgumentParser(description='Compute HiRiD AKI trajectory probabilities')
    parser.add_argument('--input', type=str, 
                       default='/home/gaga/data/physionet/hirid/aki/creatinine_timeseries.csv',
                       help='Path to creatinine time series CSV file')
    parser.add_argument('--output', type=str,
                       default='/home/gaga/data/physionet/hirid/aki/aki_trajectory_probs_bayes.csv',
                       help='Path to save trajectory probabilities')
    parser.add_argument('--window-days', type=float, default=7.0,
                       help='Lookback window in days (default: 7.0)')
    parser.add_argument('--flat-thr', type=float, default=0.1,
                       help='Stable threshold in mg/dL (default: 0.1)')
    parser.add_argument('--decline-thr', type=float, default=0.3,
                       help='Decline threshold in mg/dL/day (default: 0.3)')
    parser.add_argument('--n-batches', type=int, default=10,
                       help='Number of batches for processing (default: 10)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("HiRiD AKI Trajectory Modeling")
    print("="*80)
    
    # Load creatinine time series
    print(f"\n📊 Loading data from: {args.input}")
    creat_ts = pd.read_csv(args.input)
    
    print(f"   Patients: {creat_ts['patientid'].nunique():,}")
    print(f"   Measurements: {len(creat_ts):,}")
    print(f"   Mean measurements/patient: {len(creat_ts) / creat_ts['patientid'].nunique():.1f}")
    
    # Prepare data for BayesianTrajPS
    traj_input = creat_ts[[
        'patientid',
        'time_days',
        'time_day',
        'creatinine'
    ]].rename(columns={
        'creatinine': 'lab_value'
    })
    
    traj_input = traj_input.dropna(subset=['lab_value']).sort_values(by=['patientid', 'time_days'])
    
    print(f"\n1. Input data:")
    print(f"   Rows: {len(traj_input):,}")
    print(f"   Patients: {traj_input['patientid'].nunique():,}")
    print(f"   Timepoints per patient: {traj_input.groupby('patientid')['time_days'].count().mean():.1f}")
    
    # Configure Bayesian trajectory model (matching MIMIC configuration)
    config = BayesConfig(
        window_years=args.window_days,
        df_basis=5,
        n_samples=200,
        tune=300,
        min_points_per_window=6,
        grid_freq=2,
        flat_thr=args.flat_thr,
        decline_thr=args.decline_thr,
        nonlinear_gap=0.5,
        pids='patientid',
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
    print(f"   Window: {args.window_days} days")
    print(f"   Chains: {config.chains}")
    print(f"   Samples: {config.n_samples} (warmup: {config.tune})")
    print(f"   Stable threshold: ±{args.flat_thr} mg/dL")
    print(f"   Decline threshold: >{args.decline_thr} mg/dL/day")
    
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
        'trajtype_linear_decline_prob': 'prob_gradual_increase',
        'trajtype_nonlinear_prob': 'prob_rapid_increase'
    })
    
    # Merge with original data to add metadata
    creat_with_probs = creat_ts.merge(
        trajectory_probs[['patientid', 'time_day', 'prob_stable', 'prob_gradual_increase', 'prob_rapid_increase']],
        on=['patientid', 'time_day'],
        how='right'
    ).sort_values(by='time_days').drop_duplicates(subset=['patientid', 'time_day'], keep='last')
    
    print(f"\n4. Merged trajectory probabilities:")
    print(f"   Rows: {len(creat_with_probs):,}")
    print(f"   Missing probabilities: {creat_with_probs['prob_stable'].isna().sum()}")
    
    # Dominant trajectory per time window
    creat_with_probs['dominant_traj'] = creat_with_probs[
        ['prob_stable', 'prob_gradual_increase', 'prob_rapid_increase']
    ].idxmax(axis=1).str.replace('prob_', '')
    
    print(f"\n5. Trajectory Distribution:")
    for traj in ['stable', 'gradual_increase', 'rapid_increase']:
        subset = creat_with_probs[creat_with_probs['dominant_traj'] == traj]
        if len(subset) > 0:
            print(f"   {traj.replace('_', ' ').title():20s}: {len(subset):5,} observations ({100*len(subset)/len(creat_with_probs):5.1f}%)")
            print(f"      Mean creatinine: {subset['creatinine'].mean():.2f} mg/dL")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    trajectory_probs_full = creat_with_probs[[
        'patientid', 'time_days', 'time_day',
        'creatinine', 'baseline_creatinine',
        'prob_stable', 'prob_gradual_increase', 'prob_rapid_increase'
    ]]
    
    trajectory_probs_full.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved: {output_path}")
    print("="*80)
    print("\n✅ Done! Continue in HiRiD_AKI_traj_probs.ipynb to load results and visualize")

if __name__ == '__main__':
    main()
