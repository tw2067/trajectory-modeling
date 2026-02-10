#!/usr/bin/env python3
"""
HiRiD circulatory failure trajectory probabilities for three biomarkers:
  - Lactate
  - Heart Rate
  - Systolic BP
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
        description='Compute HiRiD circulatory failure trajectory probabilities for lactate, heart rate, and systolic BP'
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='../results/hirid/circulatory_failure',
        help='Directory containing lactate_timeseries.csv, heartrate_timeseries.csv, systolic_timeseries.csv'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='../results/hirid/circulatory_failure',
        help='Directory to write trajectory probability CSVs'
    )

    parser.add_argument(
        '--window-hours',
        type=float,
        default=12.0,
        help='Lookback window in hours for all biomarkers (default: 12.0)'
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
    available = [c for c in candidates if c in df.columns]
    if not available:
        return None
    df[target] = df[available].bfill(axis=1).iloc[:, 0]
    return target


def _make_configs(args: argparse.Namespace) -> Dict[str, BiomarkerSpec]:
    base_in = Path(args.input_dir)
    base_out = Path(args.output_dir)
    window = args.window_hours
    sampler = args.sampler

    lactate_cfg = BayesConfig(
        window_years=window,
        df_basis=5,
        n_samples=300,
        tune=300,
        min_points_per_window=4,
        grid_freq=3,
        flat_thr=0.04,
        decline_thr=0.15,
        nonlinear_gap=0.2,
        pids='patientid',
        values='lab_value',
        time_col='time_hours',
        windowing_col='time_hour',
        use_gpu=False,
        sampler=sampler,
        target_accept=0.995,
        chains=4,
        n_jobs=-1,
        class_func=pos_flags_from_traj,
        traj_types=('stable', 'gradual_increase', 'rapid_increase'),
        label_map={'nonprogression': 'stable', 'linear': 'gradual_increase', 'nonlinear': 'rapid_increase'}
    )

    hr_cfg = BayesConfig(
        window_years=window,
        df_basis=5,
        n_samples=300,
        tune=300,
        min_points_per_window=4,
        grid_freq=3,
        flat_thr=1.5,
        decline_thr=6.0,
        nonlinear_gap=10.0,
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
        traj_types=('stable', 'gradual_increase', 'rapid_increase'),
        label_map={'nonprogression': 'stable', 'linear': 'gradual_increase', 'nonlinear': 'rapid_increase'}
    )

    sbp_cfg = BayesConfig(
        window_years=window,
        df_basis=5,
        n_samples=300,
        tune=300,
        min_points_per_window=4,
        grid_freq=3,
        flat_thr=-1.5,
        decline_thr=-6.0,
        nonlinear_gap=8.0,
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
        traj_types=('stable', 'gradual_decline', 'rapid_decline'),
        label_map={'nonprogression': 'stable', 'linear': 'gradual_decline', 'nonlinear': 'rapid_decline'}
    )

    return {
        'lactate': BiomarkerSpec(
            name='lactate',
            input_path=base_in / 'lactate_timeseries.csv',
            output_path=base_out / 'lactate_trajectory_probs_bayes.csv',
            value_col='lactate',
            value_alias='lactate',
            baseline_candidates=['baseline_lactate'],
            config=lactate_cfg,
            column_map={
                'trajtype_stable_prob': 'prob_stable',
                'trajtype_gradual_increase_prob': 'prob_gradual_increase',
                'trajtype_rapid_increase_prob': 'prob_rapid_increase'
            }
        ),
        'heartrate': BiomarkerSpec(
            name='heartrate',
            input_path=base_in / 'heartrate_timeseries.csv',
            output_path=base_out / 'heartrate_trajectory_probs_bayes.csv',
            value_col='heartrate',
            value_alias='heartrate',
            baseline_candidates=['baseline_heartrate'],
            config=hr_cfg,
            column_map={
                'trajtype_stable_prob': 'prob_stable',
                'trajtype_gradual_increase_prob': 'prob_gradual_increase',
                'trajtype_rapid_increase_prob': 'prob_rapid_increase'
            }
        ),
        'systolic': BiomarkerSpec(
            name='systolic',
            input_path=base_in / 'systolic_timeseries.csv',
            output_path=base_out / 'systolic_trajectory_probs_bayes.csv',
            value_col='systolic',
            value_alias='systolic',
            baseline_candidates=['baseline_systolic'],
            config=sbp_cfg,
            column_map={
                'trajtype_stable_prob': 'prob_stable',
                'trajtype_gradual_decline_prob': 'prob_gradual_decline',
                'trajtype_rapid_decline_prob': 'prob_rapid_decline'
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

    df = df.rename(columns={spec.value_col: spec.value_alias})
    baseline_name = f"baseline_{spec.value_alias}"
    _coalesce_baseline(df, spec.baseline_candidates, baseline_name)

    df = df.dropna(subset=[spec.value_alias])
    df = df.sort_values(by=['patientid', 'time_hours'])
    df = df.drop_duplicates(subset=['patientid', 'time_hour', spec.value_alias])

    print(f"Patients: {df['patientid'].nunique():,}")
    print(f"Measurements: {len(df):,}")

    model = BayesianTrajPS(cfg=spec.config)

    patients = df['patientid'].unique()
    npts = patients.size
    batch_size = max(1, npts // n_batches)

    outputs = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < n_batches - 1 else npts
        subset_patients = patients[start_idx:end_idx]

        print(f"  Batch {i + 1}/{n_batches} ({len(subset_patients)} patients)...")

        batch_probs = model.embed(df[df['patientid'].isin(subset_patients)])
        outputs.append(batch_probs)

        gc.collect()

    trajectory_probs = pd.concat(outputs, ignore_index=True)
    trajectory_probs = trajectory_probs.rename(columns=spec.column_map)

    prob_cols = list(spec.column_map.values())
    trajectory_probs = trajectory_probs.rename(columns={'patientid': 'patientid'})

    probs_ts = df.merge(
        trajectory_probs[['patientid', 'time_hour'] + prob_cols],
        on=['patientid', 'time_hour'],
        how='left'
    )

    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    probs_ts.to_csv(spec.output_path, index=False)
    print(f"✓ Saved: {spec.output_path}")

    return trajectory_probs


def main():
    args = parse_args()
    specs = _make_configs(args)

    for spec in specs.values():
        _compute_biomarker_probs(spec, args.n_batches)


if __name__ == '__main__':
    main()
