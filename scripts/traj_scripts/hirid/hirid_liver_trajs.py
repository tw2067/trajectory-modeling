#!/usr/bin/env python3
"""
Compute bilirubin trajectory probabilities for HiRiD liver failure cohort.

Uses the BayesianTrajPS embed pipeline (aligned with hirid_aki_trajs.py) with
job-safe PyTensor compile dirs and single-threaded BLAS to avoid contention.
"""

import argparse
import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

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


@dataclass
class BiomarkerSpec:
    input_path: Path
    output_path: Path
    baseline_candidates: List[str]
    config: BayesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compute HiRiD liver trajectory probabilities (bilirubin)')
    parser.add_argument('--input', type=str,
                        default='/home/gaga/data/physionet/hirid/liver/bilirubin_timeseries.csv',
                        help='Path to bilirubin time series CSV file')
    parser.add_argument('--output', type=str,
                        default='/home/gaga/data/physionet/hirid/liver/liver_trajectory_probs_bayes.csv',
                        help='Path to save trajectory probabilities')
    parser.add_argument('--window-days', type=float, default=5.0,
                        help='Lookback window in days (default: 5.0)')
    parser.add_argument('--flat-thr', type=float, default=10.0,
                        help='Stable threshold in µmol/L/day (default: 10.0)')
    parser.add_argument('--increase-thr', type=float, default=20.0,
                        help='Increase threshold in µmol/L/day (default: 20.0)')
    parser.add_argument('--n-batches', type=int, default=10,
                        help='Number of batches for processing (default: 10)')
    parser.add_argument('--sampler', type=str, choices=['pymc', 'numpyro', 'nutpie'], default='pymc',
                        help='Sampler backend for BayesianTrajPS (default: pymc)')
    return parser.parse_args()


def _coalesce_baseline(df: pd.DataFrame, candidates: List[str], target: str) -> Optional[str]:
    available = [c for c in candidates if c in df.columns]
    if not available:
        return None
    df[target] = df[available].bfill(axis=1).iloc[:, 0]
    return target


def _make_spec(args: argparse.Namespace) -> BiomarkerSpec:
    cfg = BayesConfig(
        window_years=args.window_days,
        df_basis=5,
        n_samples=200,
        tune=300,
        min_points_per_window=4,
        grid_freq=2,
        flat_thr=args.flat_thr,
        decline_thr=args.increase_thr,
        nonlinear_gap=50.0,
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
        traj_types=('stable', 'slow_decline', 'rapid_decline'),
        label_map={'nonprogression': 'stable', 'linear': 'slow_decline', 'nonlinear': 'rapid_decline'}
    )

    return BiomarkerSpec(
        input_path=Path(args.input),
        output_path=Path(args.output),
        baseline_candidates=['baseline_bilirubin', 'baseline_bilirubin_x', 'baseline_bilirubin_y'],
        config=cfg,
    )


def _compute_probs(spec: BiomarkerSpec, n_batches: int) -> pd.DataFrame:
    if not spec.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {spec.input_path}")

    print("=" * 80)
    print("Processing BILIRUBIN trajectories")
    print("=" * 80)
    print(f"Input:  {spec.input_path}")
    print(f"Output: {spec.output_path}")

    df = pd.read_csv(spec.input_path)
    df = df.rename(columns={'lab_value': 'bilirubin'})

    baseline_col = _coalesce_baseline(df, spec.baseline_candidates, 'baseline_bilirubin')

    df = df.dropna(subset=['bilirubin'])
    df = df.sort_values(by=['patientid', 'time_days'])
    df = df.drop_duplicates(subset=['patientid', 'time_days', 'bilirubin'])

    print(f"Patients: {df['patientid'].nunique():,}")
    print(f"Measurements: {len(df):,}")
    print(f"Mean measurements/patient: {len(df) / df['patientid'].nunique():.1f}")

    traj_input = df[['patientid', 'time_days', 'time_day', 'bilirubin']].rename(columns={'bilirubin': 'lab_value'})

    traj_model = BayesianTrajPS(cfg=spec.config)

    patients = traj_input['patientid'].unique()
    npts = patients.size
    batch_size = max(1, npts // n_batches)

    trajectory_probs_list = []

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < n_batches - 1 else npts
        subset_patients = patients[start_idx:end_idx]

        print(f"\nBatch {i + 1}/{n_batches} ({len(subset_patients)} patients)...")
        batch_probs = traj_model.embed(traj_input[traj_input['patientid'].isin(subset_patients)])
        trajectory_probs_list.append(batch_probs)
        gc.collect()

    trajectory_probs = pd.concat(trajectory_probs_list, ignore_index=True)

    trajectory_probs = trajectory_probs.rename(columns={
        'trajtype_stable_prob': 'prob_stable',
        'trajtype_slow_decline_prob': 'prob_gradual_increase',
        'trajtype_rapid_decline_prob': 'prob_rapid_increase'
    })

    prob_cols = ['prob_stable', 'prob_gradual_increase', 'prob_rapid_increase']
    merged = df.merge(
        trajectory_probs[['patientid', 'time_day', *prob_cols]],
        on=['patientid', 'time_day'],
        how='right'
    ).sort_values(by='time_days').drop_duplicates(subset=['patientid', 'time_day'], keep='last')

    merged['dominant_traj'] = merged[prob_cols].idxmax(axis=1).str.replace('prob_', '')

    print("\nTrajectory distribution:")
    for traj in ['stable', 'gradual_increase', 'rapid_increase']:
        subset = merged[merged['dominant_traj'] == traj]
        if len(subset) > 0:
            print(f"  {traj.replace('_', ' ').title():20s}: {len(subset):5,} ({100 * len(subset) / len(merged):5.1f}%)")

    output_cols = ['patientid', 'time_days', 'time_day', 'bilirubin']
    if baseline_col:
        output_cols.append(baseline_col)
    output_cols.extend(prob_cols)

    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    merged[output_cols].to_csv(spec.output_path, index=False)

    print(f"\nSaved: {spec.output_path} (shape: {merged[output_cols].shape})")
    return merged[output_cols]


def main() -> None:
    args = parse_args()
    spec = _make_spec(args)
    _compute_probs(spec, args.n_batches)
    print("\n" + "=" * 80)
    print("Liver trajectory computation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
