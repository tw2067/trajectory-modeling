"""
Compute trajectory probabilities for ALL circulatory failure biomarkers (lactate, heartrate, systolic) in one unified run.

Usage:
    python circulatory_failure_trajs.py
"""

import pandas as pd
import numpy as np
import argparse
import os
import gc
from pathlib import Path
import sys

# Ensure compiled artifacts and matplotlib cache land in a writable location
JOB_ID = os.environ.get('SLURM_JOB_ID', 'local')
CACHE_ROOT = Path('/home/gaga/tamarw1')
PYTENSOR_CACHE = CACHE_ROOT / '.pytensor_cache' / JOB_ID
PYTENSOR_CACHE.mkdir(parents=True, exist_ok=True)
os.environ['PYTENSOR_FLAGS'] = f"compiledir={PYTENSOR_CACHE},base_compiledir={PYTENSOR_CACHE},optimizer=fast_compile,exception_verbosity=high"

MPL_CACHE = CACHE_ROOT / '.matplotlib'
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(MPL_CACHE))

# Limit threading
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.abspath('src'))

from traj_features.backends.bayes import BayesianTrajPS, BayesConfig
from traj_features.backends.bayes.classify import pos_flags_from_traj, flags_from_traj

import pymc as pm
from patsy import dmatrix


def precompile_pytensor_functions(config):
    """
    Pre-compile PyTensor functions with a dummy run to avoid lock contention.
    """
    print("\n[Precompilation] Compiling PyTensor functions...")

    dummy_data = pd.DataFrame({
        'patientid': [1] * 10,
        'time_days': np.linspace(0, 3, 10),
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
        print(f"  [Precompilation] ⚠️  Warning: Could not precompile: {e}")
        return False


BIOMARKERS = {
    'lactate': {
        'file': 'lactate_timeseries.csv',
        'value_col': 'lactate',
        'flat_thr': 0.04,
        'decline_thr': 0.15,
        'nonlinear_gap': 0.2,
        'class_func': pos_flags_from_traj,
        'traj_types': ('stable', 'gradual_increase', 'rapid_increase'),
        'label_map': {'nonprogression': 'stable', 'linear': 'gradual_increase', 'nonlinear': 'rapid_increase'},
    },
    'heartrate': {
        'file': 'heartrate_timeseries.csv',
        'value_col': 'heartrate',
        'flat_thr': 1.5,
        'decline_thr': 6.0,
        'nonlinear_gap': 10.0,
        'class_func': pos_flags_from_traj,
        'traj_types': ('stable', 'gradual_increase', 'rapid_increase'),
        'label_map': {'nonprogression': 'stable', 'linear': 'gradual_increase', 'nonlinear': 'rapid_increase'},
    },
    'systolic': {
        'file': 'systolic_timeseries.csv',
        'value_col': 'systolic',
        'flat_thr': -1.5,
        'decline_thr': -6.0,
        'nonlinear_gap': 8.0,
        'class_func': flags_from_traj,
        'traj_types': ('stable', 'gradual_decline', 'rapid_decline'),
        'label_map': {'nonprogression': 'stable', 'linear': 'gradual_decline', 'nonlinear': 'rapid_decline'},
    },
}


def compute_biomarker_trajectories(biomarker_name, config_dict, data_dir, window_hours, n_batches, cohort_patients=None):
    print("\n" + "=" * 80)
    print(f"Computing Trajectories: {biomarker_name.upper()}")
    print("=" * 80)

    ts_path = data_dir / config_dict['file']
    print(f"\n📊 Loading {biomarker_name} time series from: {ts_path}")

    if not ts_path.exists():
        print(f"   ⚠️  WARNING: File not found, skipping {biomarker_name}")
        return None

    ts_df = pd.read_csv(ts_path)
    n_patients = ts_df['stay_id'].nunique()
    n_rows = len(ts_df)

    if n_patients == 0 or n_rows == 0:
        print(f"   ⚠️  WARNING: No data found, skipping {biomarker_name}")
        return None

    print(f"   Patients: {n_patients:,}")
    print(f"   Measurements: {n_rows:,}")
    print(f"   Measurements/patient: {n_rows / n_patients:.1f}")

    value_col = config_dict['value_col']
    if value_col not in ts_df.columns:
        print(f"   ⚠️  WARNING: Column '{value_col}' not found, skipping {biomarker_name}")
        return None

    traj_input = ts_df[['stay_id', 'time_hours', 'time_hour', value_col]].copy()
    traj_input = traj_input.rename(columns={
        'stay_id': 'patientid',
        value_col: 'lab_value'
    })
    traj_input = traj_input.dropna(subset=['lab_value']).sort_values(by=['patientid', 'time_days'])
    if cohort_patients is not None:
        traj_input = traj_input[traj_input['patientid'].isin(cohort_patients)]

    config = BayesConfig(
        window_years=window_hours,
        df_basis=5,
        n_samples=200,
        tune=300,
        min_points_per_window=4,
        grid_freq=2,
        flat_thr=config_dict['flat_thr'],
        decline_thr=config_dict['decline_thr'],
        nonlinear_gap=config_dict['nonlinear_gap'],
        pids='patientid',
        values='lab_value',
        time_col='time_hours',
        windowing_col='time_hour',
        use_gpu=False,
        sampler='pymc',
        target_accept=0.99,
        chains=4,
        n_jobs=-1,
        class_func=config_dict['class_func'],
        traj_types=config_dict['traj_types'],
        label_map=config_dict['label_map'],
    )

    print(f"\nConfiguration:")
    print(f"   Window: {window_hours} hours")
    print(f"   Stable threshold: ±{config_dict['flat_thr']}")
    print(f"   Change threshold: {config_dict['decline_thr']}")
    print(f"   Nonlinear gap: {config_dict['nonlinear_gap']}")

    precompile_pytensor_functions(config)

    traj_model = BayesianTrajPS(cfg=config)

    patients = traj_input['patientid'].unique()
    npts = patients.size
    batch_size = npts // n_batches

    trajectory_probs_list = []

    print(f"\nComputing probabilities ({n_batches} batches)...")
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size if i < n_batches - 1 else npts
        subset_patients = patients[start_idx:end_idx]

        print(f"   Batch {i + 1}/{n_batches} ({len(subset_patients)} patients)...", end=' ', flush=True)

        try:
            batch_probs = traj_model.embed(traj_input[traj_input['patientid'].isin(subset_patients)])
            trajectory_probs_list.append(batch_probs)
            print("✓")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            continue

        gc.collect()

    if not trajectory_probs_list:
        print(f"\n   ⚠️  ERROR: All batches failed for {biomarker_name}")
        return None

    trajectory_probs = pd.concat(trajectory_probs_list, ignore_index=True)

    traj_type_names = config_dict['traj_types']
    prob_map = {
        f'trajtype_{traj_type_names[0]}_prob': f'{biomarker_name}_stable',
        f'trajtype_{traj_type_names[1]}_prob': f'{biomarker_name}_gradual',
        f'trajtype_{traj_type_names[2]}_prob': f'{biomarker_name}_rapid',
    }

    trajectory_probs = trajectory_probs.rename(columns=prob_map)
    trajectory_probs = trajectory_probs.rename(columns={'patientid': 'stay_id'})

    result_cols = ['stay_id', 'time_hour', f'{biomarker_name}_stable', f'{biomarker_name}_gradual', f'{biomarker_name}_rapid']
    result = trajectory_probs[result_cols].copy()

    prob_cols = [f'{biomarker_name}_stable', f'{biomarker_name}_gradual', f'{biomarker_name}_rapid']
    missing = result[prob_cols].isna().any(axis=1).sum()
    if missing > 0:
        print(f"   WARNING: Missing probabilities for {missing} rows in {biomarker_name} time series.")

    out_of_range = ((result[prob_cols] < 0) | (result[prob_cols] > 1)).any(axis=1).sum()
    if out_of_range > 0:
        print(f"   ERROR: {out_of_range} rows have probabilities outside [0, 1] for {biomarker_name}.")
        sys.exit(2)

    sum_probs = result[prob_cols].sum(axis=1)
    invalid_sum = (np.abs(sum_probs - 1.0) > 0.05).sum()
    if invalid_sum > 0:
        print(f"   WARNING: {invalid_sum} rows have probabilities not summing to ~1 for {biomarker_name}.")

    return result


def main():
    parser = argparse.ArgumentParser(description='Compute eICU circulatory failure trajectory probabilities')
    parser.add_argument('--data-dir', type=str, default='results/eicu/circulatory_failure',
                        help='Directory containing biomarker time series CSVs')
    parser.add_argument('--pred-dataset', type=str, default='results/eicu/circulatory_failure/circulatory_failure_prediction_dataset.csv',
                        help='Path to prediction dataset for merging')
    parser.add_argument('--merged-output', type=str, default=None,
                        help='Optional path to save prediction dataset merged with probabilities')
    parser.add_argument('--window-hours', type=float, default=12.0,
                        help='Lookback window in hours (default: 12.0)')
    parser.add_argument('--n-batches', type=int, default=8,
                        help='Number of batches for processing (default: 8)')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    biomarker_results = {}

    for biomarker, cfg in BIOMARKERS.items():
        traj_df = compute_biomarker_trajectories(
            biomarker_name=biomarker,
            config_dict=cfg,
            data_dir=data_dir,
            window_hours=args.window_hours,
            n_batches=args.n_batches,
        )
        if traj_df is not None:
            biomarker_results[biomarker] = traj_df

    if not biomarker_results:
        print("\n❌ ERROR: No biomarker trajectories were computed successfully")
        sys.exit(1)

    if args.pred_dataset and Path(args.pred_dataset).exists():
        print(f"\nLoading prediction dataset: {args.pred_dataset}")
        prediction_dataset = pd.read_csv(args.pred_dataset)
        print(f"   Samples: {len(prediction_dataset):,}")

        merged = prediction_dataset.copy()
        for biomarker, traj_df in biomarker_results.items():
            print(f"\nMerging {biomarker} trajectories...")
            merged = merged.merge(traj_df, on=['stay_id', 'time_hour'], how='left')

            prob_cols = [f"{biomarker}_stable", f"{biomarker}_gradual", f"{biomarker}_rapid"]
            missing = merged[prob_cols].isna().any(axis=1).sum()
            print(f"   Rows with {biomarker} probs: {len(merged) - missing:,} / {len(merged):,}")

        merged_path = Path(args.merged_output) if args.merged_output else data_dir / "circulatory_failure_prediction_dataset_with_probs.csv"
        merged.to_csv(merged_path, index=False)
        print(f"\n✓ Saved merged prediction dataset: {merged_path}")
    else:
        print("\nℹ️  Prediction dataset not found or not provided. Skipping merged output.")


if __name__ == '__main__':
    main()
