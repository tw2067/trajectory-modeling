#!/usr/bin/env python
"""
MIMIC Circulatory Failure Prediction Analysis
Compares model and feature configurations across single vs multi-biomarker trajectories and summary statistics.

Run from workspace root:
    python scripts/analysis/mimic/mimic_circulatory_failure_analysis.py

This analysis expects bootstrap trajectory probability features to already be
present in the prediction dataset and evaluates the multi-marker feature sets.
"""
import os
import sys
import logging
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def _pick_id_col(df):
    """Auto-detect ID column."""
    for col in ['hadm_id', 'stay_id', 'patientid']:
        if col in df.columns:
            return col
    raise ValueError('No ID column found')


def _pick_time_cols(df):
    """Auto-detect time columns."""
    if 'time_hours' in df.columns and 'time_hour' in df.columns:
        return 'time_hours', 'time_hour'
    if 'time_days' in df.columns and 'time_day' in df.columns:
        return 'time_days', 'time_day'
    if 'time_hour' in df.columns:
        return 'time_hour', 'time_hour'
    if 'time_day' in df.columns:
        return 'time_day', 'time_day'
    raise ValueError('No time columns found')


def _read_table(path: Path) -> pd.DataFrame:
    """Read table (parquet or CSV)."""
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_named_table(base_dir: Path, stem_name: str) -> pd.DataFrame:
    """Load named table (parquet-first fallback to CSV)."""
    candidates = [base_dir / f'{stem_name}.parquet', base_dir / f'{stem_name}.csv']
    for p in candidates:
        if p.exists():
            logger.info(f'  Loading: {p}')
            return _read_table(p)
    raise FileNotFoundError(f'Missing table for {stem_name}. Tried: {candidates}')


def load_data(base_dir: Path):
    """Load prediction dataset and biomarker timeseries."""
    logger.info(f'BASE_DIR: {base_dir}')

    # Load prediction dataset
    pred_stem_candidates = [
        'circulatory_failure_prediction_dataset_with_bootstrap_probs',
        'circulatory_failure_prediction_dataset_with_probs',
        'circulatory_failure_prediction_dataset',
    ]
    for _stem in pred_stem_candidates:
        try:
            dataset = _load_named_table(base_dir, _stem)
            pred_source_stem = _stem
            break
        except FileNotFoundError:
            continue
    else:
        raise FileNotFoundError('Could not find prediction dataset.')

    logger.info(f'Loaded prediction dataset ({pred_source_stem}): {len(dataset):,} samples')

    # Check for bootstrap probs
    expected_traj_cols = [
        'lactate_stable', 'lactate_gradual', 'lactate_rapid',
        'heartrate_stable', 'heartrate_gradual', 'heartrate_rapid',
        'systolic_stable', 'systolic_gradual', 'systolic_rapid',
    ]
    has_bootstrap_probs = all(c in dataset.columns for c in expected_traj_cols)
    logger.info(f'Bootstrap trajectory features present: {has_bootstrap_probs}')

    # Load biomarker timeseries
    lactate_ts = _load_named_table(base_dir, 'lactate_timeseries')
    heartrate_ts = _load_named_table(base_dir, 'heartrate_timeseries')
    systolic_ts = _load_named_table(base_dir, 'systolic_timeseries')

    logger.info(f'Lactate:   {len(lactate_ts):,} rows')
    logger.info(f'Heartrate: {len(heartrate_ts):,} rows')
    logger.info(f'Systolic:  {len(systolic_ts):,} rows')

    return dataset, lactate_ts, heartrate_ts, systolic_ts, has_bootstrap_probs


def biomarker_summary_stats(ts_df, value_col, id_col, windowing_col, time_col, lookback_hours=12):
    """Compute windowed summary statistics per biomarker."""
    ts_df = ts_df.copy()
    ts_df[time_col] = pd.to_numeric(ts_df[time_col], errors='coerce')
    ts_df[windowing_col] = pd.to_numeric(ts_df[windowing_col], errors='coerce')
    ts_df[value_col] = pd.to_numeric(ts_df[value_col], errors='coerce')
    ts_df = ts_df.dropna(subset=[time_col, windowing_col, value_col])

    trend_failures = 0

    summary_df_list = []
    for group, group_df in ts_df.groupby(id_col):
        group_df = group_df.sort_values(time_col)
        summary_list = []
        for current_time in group_df[windowing_col].unique():
            window_start = current_time - lookback_hours
            window_data = group_df[group_df[windowing_col].between(window_start, current_time, inclusive='both')]
            if len(window_data) > 0:
                value_mean = window_data[value_col].mean()
                value_max = window_data[value_col].max()
                value_min = window_data[value_col].min()
                value_change = window_data[value_col].iloc[-1] - window_data[value_col].iloc[0]

                if len(window_data) > 1:
                    x = window_data[time_col].to_numpy(dtype=float)
                    y = window_data[value_col].to_numpy(dtype=float)
                    finite = np.isfinite(x) & np.isfinite(y)
                    x = x[finite]
                    y = y[finite]

                    # polyfit can fail when x has <2 unique points or degenerate values
                    if len(x) > 1 and np.unique(x).size > 1:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore', np.exceptions.RankWarning)
                                value_linear_trend = float(np.polyfit(x, y, 1)[0])
                        except Exception:
                            value_linear_trend = 0.0
                            trend_failures += 1
                    else:
                        value_linear_trend = 0.0
                else:
                    value_linear_trend = 0.0

                value_std = window_data[value_col].std() if len(window_data) > 1 else 0
            else:
                value_mean = np.nan
                value_max = np.nan
                value_min = np.nan
                value_change = np.nan
                value_linear_trend = np.nan
                value_std = np.nan
            summary_list.append({
                id_col: group,
                windowing_col: current_time,
                f'{value_col}_mean_{lookback_hours}h': value_mean,
                f'{value_col}_max_{lookback_hours}h': value_max,
                f'{value_col}_min_{lookback_hours}h': value_min,
                f'{value_col}_change_{lookback_hours}h': value_change,
                f'{value_col}_trend_{lookback_hours}h': value_linear_trend,
                f'{value_col}_std_{lookback_hours}h': value_std
            })
        summary_df_list.append(pd.DataFrame(summary_list))

    if trend_failures > 0:
        logger.warning(
            f"{value_col}: trend calculation failed in {trend_failures:,} windows; set trend=0 for those windows"
        )

    return pd.concat(summary_df_list, ignore_index=True)


def prepare_features(dataset, id_col, windowing_col, lactate_ts, heartrate_ts, systolic_ts, lookback_hours=12):
    """Prepare feature sets for modeling."""
    key_cols = [id_col, windowing_col]

    # Load summary statistics
    logger.info(f'Computing summary statistics (lookback={lookback_hours} hours)...')
    lactate_time_col, _ = _pick_time_cols(lactate_ts)
    heartrate_time_col, _ = _pick_time_cols(heartrate_ts)
    systolic_time_col, _ = _pick_time_cols(systolic_ts)

    lactate_summary = biomarker_summary_stats(
        lactate_ts, 'lactate', id_col, windowing_col, lactate_time_col, lookback_hours
    )
    heartrate_summary = biomarker_summary_stats(
        heartrate_ts, 'heartrate', id_col, windowing_col, heartrate_time_col, lookback_hours
    )
    systolic_summary = biomarker_summary_stats(
        systolic_ts, 'systolic', id_col, windowing_col, systolic_time_col, lookback_hours
    )

    logger.info('Merging summary statistics into dataset...')
    dataset = dataset.merge(lactate_summary, on=key_cols, how='left')
    dataset = dataset.merge(heartrate_summary, on=key_cols, how='left')
    dataset = dataset.merge(systolic_summary, on=key_cols, how='left')

    # Prepare feature sets
    target_col = next((c for c in ['target_circulatory_failure', 'target_circ_failure', 'target_circulatory'] if c in dataset.columns), None)
    if target_col is None:
        raise ValueError('No target column found')

    if 'gender' in dataset.columns:
        gender_map = {'M': 1, 'F': 0}
        dataset['gender'] = dataset['gender'].map(gender_map)

    static_features = [c for c in ['age', 'gender'] if c in dataset.columns]
    exclude_keywords = ['stable', 'gradual', 'rapid', 'worsening', 'marker', 'trend', 'change', 'boot']
    dynamic_labs = [
        col for col in dataset.columns
        if any(col.startswith(prefix) for prefix in ['min_', 'mean_', 'max_'])
        and not any(keyword in col.lower() for keyword in exclude_keywords)
    ]
    dynamic_vitals = [
        col for col in dataset.columns
        if any(x in col for x in ['heart_rate', 'respiratory_rate', 'o2_sat', 'systolic', 'diastolic', 'mean_bp', 'temperature'])
    ]
    dynamic_features = list(dict.fromkeys(dynamic_labs + dynamic_vitals))
    static_dynamic_cols = static_features + dynamic_features

    lactate_traj = ['lactate_stable', 'lactate_gradual', 'lactate_rapid', 'lactate_worsening']
    heartrate_traj = ['heartrate_stable', 'heartrate_gradual', 'heartrate_rapid', 'heartrate_worsening']
    systolic_traj = ['systolic_stable', 'systolic_gradual', 'systolic_rapid', 'systolic_worsening']
    multi_traj = lactate_traj + heartrate_traj + systolic_traj + ['multi_marker_mean_worsening']

    lactate_summary_cols = [c for c in dataset.columns if c.startswith('lactate_') and c.endswith('h')]
    heartrate_summary_cols = [c for c in dataset.columns if c.startswith('heartrate_') and c.endswith('h')]
    systolic_summary_cols = [c for c in dataset.columns if c.startswith('systolic_') and c.endswith('h')]
    multi_summary = lactate_summary_cols + heartrate_summary_cols + systolic_summary_cols

    feature_sets = {
        'Single Marker Trajectory': lactate_traj,
        'Multi-Marker Trajectories': multi_traj,
        'Single Marker Summary Stats': lactate_summary_cols,
        'Multi-Marker Summary Stats': multi_summary,
        'Static Only': static_features,
        'Static + Single Marker Trajectory': static_features + lactate_traj,
        'Static + Multi-Marker Trajectories': static_features + multi_traj,
        'Static + Single Marker Summary Stats': static_features + lactate_summary_cols,
        'Static + Multi-Marker Summary Stats': static_features + multi_summary,
        'Static + Single Marker Trajectory + Summary': static_features + lactate_traj + lactate_summary_cols,
        'Static + Multi-Marker Trajectories + Summary': static_features + multi_traj + multi_summary,
    }

    if len(dynamic_features) > 0:
        feature_sets['Static + Dynamic'] = static_dynamic_cols
        feature_sets['Static + Dynamic + Single Marker Trajectory'] = static_dynamic_cols + lactate_traj
        feature_sets['Static + Dynamic + Multi-Marker Trajectories'] = static_dynamic_cols + multi_traj
        feature_sets['Static + Dynamic + Single Marker Summary Stats'] = static_dynamic_cols + lactate_summary_cols
        feature_sets['Static + Dynamic + Multi-Marker Summary Stats'] = static_dynamic_cols + multi_summary
        feature_sets['Static + Dynamic + Single Marker Trajectory + Summary'] = static_dynamic_cols + lactate_traj + lactate_summary_cols
        feature_sets['Static + Dynamic + Multi-Marker Trajectories + Summary'] = static_dynamic_cols + multi_traj + multi_summary

    for set_name, cols in list(feature_sets.items()):
        feature_sets[set_name] = [c for c in cols if c in dataset.columns]

    logger.info(f'Feature sets prepared: {len(feature_sets)} total')

    return dataset, target_col, feature_sets


def run_cv(dataset, target_col, id_col, feature_sets, n_repeats=5, n_folds=5, seed=920):
    """Run cross-validation with all models and feature sets."""
    dataset_clean = dataset.dropna(subset=[target_col]).copy()
    y = dataset_clean[target_col]
    groups = dataset_clean[id_col]

    traj_cols = [c for c in dataset_clean.columns if any(k in c for k in ['_stable', '_gradual', '_rapid', '_worsening'])]
    summary_cols = [c for c in dataset_clean.columns if c.endswith('h') and any(prefix in c for prefix in ['lactate_', 'heartrate_', 'systolic_'])]

    dataset_clean.loc[:, traj_cols] = dataset_clean[traj_cols].fillna(0)
    dataset_clean.loc[:, summary_cols] = dataset_clean[summary_cols].fillna(0)

    models_to_evaluate = {
        'XGBoost': lambda pos_weight, seed: XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            scale_pos_weight=pos_weight,
            random_state=seed,
            eval_metric='logloss'
        ),
        'Logistic Regression': lambda pos_weight, seed: LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=seed,
            solver='lbfgs'
        ),
        'Random Forest': lambda pos_weight, seed: RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight='balanced',
            random_state=seed,
            n_jobs=4
        ),
        'Gradient Boosting': lambda pos_weight, seed: HistGradientBoostingClassifier(
            max_bins=225,
            max_depth=3,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=seed
        )
    }

    fold_rows = []

    logger.info(f"Starting CV: models={len(models_to_evaluate)}, feature_sets={len(feature_sets)}, repeats={n_repeats}, folds={n_folds}")

    for model_name, model_fn in models_to_evaluate.items():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'=' * 80}")

        for feature_set_name, feature_cols in feature_sets.items():
            logger.info(f"  -> Feature set: {feature_set_name} ({len(feature_cols)} features)")

            fold_metrics = {'roc_auc': [], 'avg_precision': []}

            for repeat in range(n_repeats):
                shuffle_idx = np.random.RandomState(seed=seed + repeat).permutation(len(dataset_clean))
                dataset_repeat = dataset_clean.iloc[shuffle_idx].reset_index(drop=True)
                y_repeat = y.iloc[shuffle_idx].reset_index(drop=True)
                groups_repeat = groups.iloc[shuffle_idx].reset_index(drop=True)
                gkf = GroupKFold(n_splits=n_folds)

                rep_auc = []
                rep_aupr = []

                for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(dataset_repeat, y_repeat, groups_repeat), start=1):
                    X_train = dataset_repeat.iloc[train_idx][feature_cols]
                    X_test = dataset_repeat.iloc[test_idx][feature_cols]
                    y_train = y_repeat.iloc[train_idx]
                    y_test = y_repeat.iloc[test_idx]

                    imputer = SimpleImputer(strategy='median')
                    X_train_imputed = imputer.fit_transform(X_train)
                    X_test_imputed = imputer.transform(X_test)

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_imputed)
                    X_test_scaled = scaler.transform(X_test_imputed)

                    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
                    model = model_fn(scale_pos_weight, seed + repeat)
                    model.fit(X_train_scaled, y_train)

                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

                    auc = roc_auc_score(y_test, y_pred_proba)
                    aupr = average_precision_score(y_test, y_pred_proba)

                    rep_auc.append(auc)
                    rep_aupr.append(aupr)

                    fold_metrics['roc_auc'].append(auc)
                    fold_metrics['avg_precision'].append(aupr)

                    fold_rows.append({
                        'Model': model_name,
                        'Feature Set': feature_set_name,
                        'repeat': repeat + 1,
                        'fold': fold_idx,
                        'ROC-AUC': auc,
                        'AUPR': aupr,
                    })

                logger.info(
                    f"     repeat {repeat + 1:02d}/{n_repeats}: "
                    f"AUROC={np.mean(rep_auc):.4f} | AUPR={np.mean(rep_aupr):.4f}"
                )

            logger.info(
                f"     done {feature_set_name}: "
                f"AUROC={np.mean(fold_metrics['roc_auc']):.4f}±{np.std(fold_metrics['roc_auc']):.4f} | "
                f"AUPR={np.mean(fold_metrics['avg_precision']):.4f}±{np.std(fold_metrics['avg_precision']):.4f}"
            )

    logger.info('\n✓ Finished model comparisons')

    results_df = pd.DataFrame(fold_rows)

    summary_df = (
        results_df.groupby(['Model', 'Feature Set'], as_index=False)
        .agg(
            ROC_AUC_mean=('ROC-AUC', 'mean'),
            ROC_AUC_std=('ROC-AUC', 'std'),
            AUPR_mean=('AUPR', 'mean'),
            AUPR_std=('AUPR', 'std'),
            n_folds=('ROC-AUC', 'count'),
        )
        .sort_values(['Model', 'ROC_AUC_mean'], ascending=[True, False])
    )
    summary_df['ROC-AUC'] = summary_df['ROC_AUC_mean']
    summary_df['ROC-AUC std'] = summary_df['ROC_AUC_std']
    summary_df['AUPR'] = summary_df['AUPR_mean']
    summary_df['AUPR std'] = summary_df['AUPR_std']

    baseline_metrics = (
        results_df[results_df['Feature Set'] == 'Baseline'][['Model', 'repeat', 'fold', 'ROC-AUC', 'AUPR']]
        .rename(columns={'ROC-AUC': 'Baseline ROC-AUC', 'AUPR': 'Baseline AUPR'})
    )

    delta_df = results_df.merge(
        baseline_metrics,
        on=['Model', 'repeat', 'fold'],
        how='left',
        validate='many_to_one',
    )
    delta_df['Delta ROC-AUC'] = delta_df['ROC-AUC'] - delta_df['Baseline ROC-AUC']
    delta_df['Delta AUPR'] = delta_df['AUPR'] - delta_df['Baseline AUPR']

    delta_summary_df = (
        delta_df[delta_df['Feature Set'] != 'Baseline']
        .groupby(['Model', 'Feature Set'], as_index=False)
        .agg(
            Delta_ROC_AUC_mean=('Delta ROC-AUC', 'mean'),
            Delta_ROC_AUC_std=('Delta ROC-AUC', 'std'),
            Delta_AUPR_mean=('Delta AUPR', 'mean'),
            Delta_AUPR_std=('Delta AUPR', 'std'),
            n_folds=('Delta ROC-AUC', 'count'),
        )
        .sort_values(['Model', 'Delta_ROC_AUC_mean'], ascending=[True, False])
    )

    return results_df, summary_df, delta_summary_df


def save_results(results_df, summary_df, delta_summary_df, output_dir):
    """Save fold-level and aggregated results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results_path = output_dir / 'mimic_circulatory_failure_fold_results.parquet'
    output_path = output_dir / 'mimic_circulatory_failure_summary.csv'
    delta_path = output_dir / 'mimic_circulatory_failure_delta_summary.csv'

    results_df.to_parquet(fold_results_path, index=False)
    summary_df.to_csv(output_path, index=False)
    delta_summary_df.to_csv(delta_path, index=False)

    logger.info(f'Results saved to: {fold_results_path}')
    logger.info(f'Results saved to: {output_path}')
    logger.info(f'Results saved to: {delta_path}')

    return summary_df


def main(args):
    """Main analysis pipeline."""
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        logger.error(f'Base directory does not exist: {base_dir}')
        sys.exit(1)

    output_dir = Path(args.output_dir)

    logger.info('Loading data...')
    dataset, lactate_ts, heartrate_ts, systolic_ts, has_bootstrap_probs = load_data(base_dir)

    logger.info('Preparing features...')
    id_col = _pick_id_col(dataset)
    _, windowing_col = _pick_time_cols(dataset)

    if not has_bootstrap_probs:
        raise ValueError('Bootstrap probabilities not found in dataset. Please run bootstrap pipeline first.')

    # Add worsening columns
    dataset['lactate_worsening'] = dataset['lactate_gradual'] + dataset['lactate_rapid']
    dataset['heartrate_worsening'] = dataset['heartrate_gradual'] + dataset['heartrate_rapid']
    dataset['systolic_worsening'] = dataset['systolic_gradual'] + dataset['systolic_rapid']
    dataset['multi_marker_mean_worsening'] = dataset[['lactate_worsening', 'heartrate_worsening', 'systolic_worsening']].mean(axis=1)

    dataset, target_col, feature_sets = prepare_features(
        dataset, id_col, windowing_col, lactate_ts, heartrate_ts, systolic_ts, args.lookback_hours
    )

    logger.info('Running cross-validation...')
    results_df, summary_df, delta_summary_df = run_cv(
        dataset, target_col, id_col, feature_sets,
        n_repeats=args.n_repeats, n_folds=args.n_folds, seed=args.seed
    )

    logger.info('Saving results...')
    summary_df = save_results(results_df, summary_df, delta_summary_df, output_dir)

    logger.info('Analysis complete!')
    logger.info(f'\nTop 10 configurations:\n{summary_df.head(10)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIMIC Circulatory Failure Analysis')
    parser.add_argument(
        '--base-dir',
        default=os.path.join(os.environ.get("TRAJ_DATA_ROOT", "/home/gaga/data/physionet"), "mimic", "circulatory_failure"),
        help='Base directory containing data files'
    )
    parser.add_argument(
        '--output-dir',
        default='./results/mimic',
        help='Output directory for results'
    )
    parser.add_argument(
        '--lookback-hours',
        type=int,
        default=12,
        help='Lookback window for summary statistics (hours)'
    )
    parser.add_argument(
        '--n-repeats',
        type=int,
        default=5,
        help='Number of CV repeats'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of CV folds'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=920,
        help='Random seed'
    )

    args = parser.parse_args()
    main(args)
