#!/usr/bin/env python3
"""
HiRiD sepsis trajectory probabilities for three biomarkers:
  - Lactate (perfusion)
  - WBC (infection/inflammation)
  - Platelets (coagulopathy)

The script mirrors the working setup in hirid_aki_trajs.py: it uses the
BayesianTrajPS embed function, job-specific PyTensor compile dirs, and
single-threaded BLAS to avoid contention on SLURM nodes.
"""

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add code directory to path
code_dir = Path(__file__).parent.parent / 'code'
sys.path.insert(0, str(code_dir))

from traj_features.backends.bayes import BayesianTrajPS, BayesConfig
from traj_features.backends.bayes.classify import flags_from_traj, pos_flags_from_traj


# Use job-specific compile directory to avoid lock contention
job_id = os.environ.get('SLURM_JOB_ID', 'local')
os.environ['PYTENSOR_FLAGS'] = (
    f"base_compiledir={os.path.expanduser('~')}/.pytensor_{job_id},"
    "optimizer=fast_compile,exception_verbosity=high"
)

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
    name: str
    input_path: Path
    output_path: Path
    value_col: str
    value_alias: str
    baseline_candidates: List[str]
    config: BayesConfig
    column_map: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compute HiRiD sepsis trajectory probabilities for lactate, WBC, and platelets'
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='../results/hirid/sepsis',
        help='Directory containing lactate_timeseries.csv, wbc_timeseries.csv, platelet_timeseries.csv'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='../results/hirid/sepsis',
        help='Directory to write trajectory probability CSVs'
    )

    parser.add_argument(
        '--window-days',
        type=float,
        default=3.0,
        help='Lookback window in days for all biomarkers (default: 3.0)'
    )

    parser.add_argument(
        '--n-batches',
        type=int,
        default=10,
        help='Number of patient batches (default: 10)'
    )

    parser.add_argument(
        '--sampler',
        type=str,
        choices=['pymc', 'numpyro', 'nutpie'],
        default='pymc',
        help='Sampler backend for BayesianTrajPS (default: pymc)'
    )

    return parser.parse_args()


def _coalesce_baseline(df: pd.DataFrame, candidates: List[str], target: str) -> Optional[str]:
    """Pick the first available baseline column and standardize its name."""
    available = [c for c in candidates if c in df.columns]
    if not available:
        return None
    df[target] = df[available].bfill(axis=1).iloc[:, 0]
    return target


def _make_configs(args: argparse.Namespace) -> Dict[str, BiomarkerSpec]:
    base_in = Path(args.input_dir)
    base_out = Path(args.output_dir)
    window = args.window_days
    sampler = args.sampler

    lactate_cfg = BayesConfig(
        window_years=window,
        df_basis=5,
        n_samples=300,
        tune=300,
        min_points_per_window=4,
        grid_freq=3,
        flat_thr=0.15,
        decline_thr=0.5,
        nonlinear_gap=0.6,
        pids='patientid',
        values='lab_value',
        time_col='time_days',
        windowing_col='time_day',
        use_gpu=False,
        sampler=sampler,
        target_accept=0.995,
        chains=4,
        n_jobs=-1,
        class_func=pos_flags_from_traj,
        traj_types=('stable', 'slow_decline', 'rapid_decline'),
        label_map={'nonprogression': 'stable', 'linear': 'slow_decline', 'nonlinear': 'rapid_decline'}
    )

    wbc_cfg = BayesConfig(
        window_years=window,
        df_basis=5,
        n_samples=300,
        tune=300,
        min_points_per_window=4,
        grid_freq=3,
        flat_thr=1.0,
        decline_thr=2.0,
        nonlinear_gap=3.0,
        pids='patientid',
        values='lab_value',
        time_col='time_days',
        windowing_col='time_day',
        use_gpu=False,
        sampler=sampler,
        target_accept=0.995,
        chains=4,
        n_jobs=-1,
        class_func=pos_flags_from_traj,
        traj_types=('stable', 'slow_decline', 'rapid_decline'),
        label_map={'nonprogression': 'stable', 'linear': 'slow_decline', 'nonlinear': 'rapid_decline'}
    )

    platelet_cfg = BayesConfig(
        window_years=window,
        df_basis=5,
        n_samples=300,
        tune=300,
        min_points_per_window=4,
        grid_freq=3,
        flat_thr=-20.0,
        decline_thr=-50.0,
        nonlinear_gap=30.0,
        pids='patientid',
        values='lab_value',
        time_col='time_days',
        windowing_col='time_day',
        use_gpu=False,
        sampler=sampler,
        target_accept=0.995,
        chains=4,
        n_jobs=-1,
        class_func=flags_from_traj,
        traj_types=('prolonged_nonprogression', 'linear_decline', 'nonlinear'),
        label_map={'nonprogression': 'prolonged_nonprogression', 'linear': 'linear_decline', 'nonlinear': 'nonlinear'}
    )

    return {
        'lactate': BiomarkerSpec(
            name='lactate',
            input_path=base_in / 'lactate_timeseries.csv',
            output_path=base_out / 'lactate_trajectory_probs_bayes.csv',
            value_col='lab_value',
            value_alias='lactate',
            baseline_candidates=['baseline_lactate'],
            config=lactate_cfg,
            column_map={
                'trajtype_stable_prob': 'prob_stable',
                'trajtype_slow_decline_prob': 'prob_gradual_increase',
                'trajtype_rapid_decline_prob': 'prob_rapid_increase'
            }
        ),
        'wbc': BiomarkerSpec(
            name='wbc',
            input_path=base_in / 'wbc_timeseries.csv',
            output_path=base_out / 'wbc_trajectory_probs_bayes.csv',
            value_col='lab_value',
            value_alias='wbc',
            baseline_candidates=['baseline_wbc', 'baseline_wbc_x', 'baseline_wbc_y'],
            config=wbc_cfg,
            column_map={
                'trajtype_stable_prob': 'prob_stable',
                'trajtype_slow_decline_prob': 'prob_gradual_increase',
                'trajtype_rapid_decline_prob': 'prob_rapid_increase'
            }
        ),
        'platelet': BiomarkerSpec(
            name='platelet',
            input_path=base_in / 'platelet_timeseries.csv',
            output_path=base_out / 'platelet_trajectory_probs_bayes.csv',
            value_col='lab_value',
            value_alias='platelet',
            baseline_candidates=['baseline_platelets', 'baseline_platelets_x', 'baseline_platelets_y'],
            config=platelet_cfg,
            column_map={
                'trajtype_prolonged_nonprogression_prob': 'prob_stable',
                'trajtype_linear_decline_prob': 'prob_gradual_increase',
                'trajtype_nonlinear_prob': 'prob_rapid_increase'
            }
        ),
    }


def _compute_biomarker_probs(spec: BiomarkerSpec, n_batches: int) -> pd.DataFrame:
    if not spec.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {spec.input_path}")

    print("=" * 80)
    print(f"Processing {spec.name.upper()} trajectories")
    print("=" * 80)
    print(f"Input:  {spec.input_path}")
    print(f"Output: {spec.output_path}")

    df = pd.read_csv(spec.input_path)

    # Standardize value and baseline columns
    df = df.rename(columns={spec.value_col: spec.value_alias})
    baseline_name = f"baseline_{spec.value_alias}"
    baseline_col = _coalesce_baseline(df, spec.baseline_candidates, baseline_name)

    df = df.dropna(subset=[spec.value_alias])
    df = df.sort_values(by=['patientid', 'time_days'])
    df = df.drop_duplicates(subset=['patientid', 'time_day', spec.value_alias])

    print(f"Patients: {df['patientid'].nunique():,}")
    print(f"Measurements: {len(df):,}")
    print(f"Mean measurements/patient: {len(df) / df['patientid'].nunique():.1f}")

    traj_input = df[['patientid', 'time_days', 'time_day', spec.value_alias]].rename(
        columns={spec.value_alias: 'lab_value'}
    )

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

    # Rename probabilities to consistent names
    trajectory_probs = trajectory_probs.rename(columns=spec.column_map)

    prob_cols = list(spec.column_map.values())
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

    output_cols = ['patientid', 'time_days', 'time_day', spec.value_alias]
    if baseline_col:
        output_cols.append(baseline_col)
    output_cols.extend(prob_cols)

    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    merged[output_cols].to_csv(spec.output_path, index=False)

    print(f"\nSaved: {spec.output_path} (shape: {merged[output_cols].shape})")
    return merged[output_cols]


def main() -> None:
    args = parse_args()
    configs = _make_configs(args)

    results = {}
    for biomarker in ['lactate', 'wbc', 'platelet']:
        results[biomarker] = _compute_biomarker_probs(configs[biomarker], args.n_batches)

    print("\n" + "=" * 80)
    print("All trajectory computations complete!")
    for biomarker, df in results.items():
        print(f"  {biomarker.title():10s}: {len(df):,} timepoints -> {configs[biomarker].output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
