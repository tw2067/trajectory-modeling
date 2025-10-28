import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy.stats import pearsonr
from sklearn.metrics import log_loss, f1_score


def compare_trajectories(
    predicted_features: pd.DataFrame,
    ground_truth: dict,
    dynamic_df: Optional[pd.DataFrame] = None,
    predicted_trajectories: Optional[pd.DataFrame] = None,
    metric: str = "mse"
) -> dict:
    metrics: dict = {}

    try:
        if 'true_slopes' in ground_truth:
            metrics.update(_compare_slopes(predicted_features, ground_truth))
    except Exception:
        pass
    try:
        if 'true_intercepts' in ground_truth:
            metrics.update(_compare_intercepts(predicted_features, ground_truth))
    except Exception:
        pass

    # Full-trajectory metrics (R^2 / MSE / MAE)
    if dynamic_df is not None and predicted_trajectories is not None and not predicted_trajectories.empty:
        metrics.update(_compare_full_trajectories(dynamic_df, predicted_trajectories, feature="eGFR"))

    # Trajectory-type probability metrics (Bayes)
    if 'true_types' in ground_truth:
        prob_cols = [c for c in predicted_features.columns if c.startswith("trajtype_") and c.endswith("_prob")]
        if prob_cols:
            metrics.update(_compare_type_probs(predicted_features[['pid'] + prob_cols], ground_truth['true_types']))

    # Overall score selection
    if metric == "r_squared" and 'trajectory_r_squared' in metrics:
        metrics['overall_score'] = metrics['trajectory_r_squared']
    elif metric == "mse" and 'trajectory_mse' in metrics:
        metrics['overall_score'] = metrics['trajectory_mse']
    else:
        # default mixed
        metrics['overall_score'] = np.nanmean([
            metrics.get('slope_mse', np.nan),
            metrics.get('intercept_mse', np.nan)
        ])

    return metrics


def _compare_full_trajectories(
    dynamic_df: pd.DataFrame,
    predicted_trajectories: pd.DataFrame,
    feature: str = "eGFR"
) -> dict:
    """
    Compare predicted trajectory time series to observed values.
    Expects:
      - dynamic_df columns: pid, time, feature_name, value
      - predicted_trajectories columns: pid, time, predicted_value
    """
    true_df = dynamic_df[dynamic_df['feature_name'] == feature].copy()

    needed_true = {'pid', 'time', 'value'}
    needed_pred = {'pid', 'time', 'predicted_value'}
    if not needed_true.issubset(true_df.columns) or not needed_pred.issubset(predicted_trajectories.columns):
        return {
            'trajectory_r_squared': np.nan,
            'trajectory_mse': np.nan,
            'trajectory_mae': np.nan,
            'n_trajectory_points': 0
        }

    merged = true_df.merge(
        predicted_trajectories[['pid', 'time', 'predicted_value']],
        on=['pid', 'time'],
        how='inner'
    )
    if merged.empty:
        return {
            'trajectory_r_squared': np.nan,
            'trajectory_mse': np.nan,
            'trajectory_mae': np.nan,
            'n_trajectory_points': 0
        }

    y_true = merged['value'].to_numpy(dtype=float)
    y_pred = merged['predicted_value'].to_numpy(dtype=float)

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        'trajectory_r_squared': r2,
        'trajectory_mse': mse,
        'trajectory_mae': mae,
        'n_trajectory_points': int(len(merged))
    }


def _compare_slopes(
    predicted_features: pd.DataFrame,
    ground_truth: dict
) -> dict:
    """Compare predicted slopes to ground truth slopes."""
    true_slopes = ground_truth['true_slopes']
    
    # Find slope column in predicted_features
    slope_col = _find_column(predicted_features, ['slope', 'eGFR_slope', 'trajectory_slope'])
    
    if slope_col is None:
        return {
            'slope_mse': np.nan,
            'slope_mae': np.nan,
            'slope_correlation': np.nan,
            'slope_rmse': np.nan
        }
    
    # Align predictions with ground truth
    pred_slopes = []
    true_slopes_aligned = []
    
    for pid in predicted_features['pid'].unique():
        if pid in true_slopes:
            pred_val = predicted_features.loc[predicted_features['pid'] == pid, slope_col].iloc[0]
            true_val = true_slopes[pid]
            
            # Handle heterogeneous case (dict of slopes)
            if isinstance(true_val, dict):
                true_val = true_val.get('b1', 0)  # Use linear term
            
            pred_slopes.append(pred_val)
            true_slopes_aligned.append(true_val)
    
    pred_slopes = np.array(pred_slopes)
    true_slopes_aligned = np.array(true_slopes_aligned)
    
    # Compute metrics
    mse = np.mean((pred_slopes - true_slopes_aligned) ** 2)
    mae = np.mean(np.abs(pred_slopes - true_slopes_aligned))
    rmse = np.sqrt(mse)
    
    if len(pred_slopes) > 1:
        corr, _ = pearsonr(pred_slopes, true_slopes_aligned)
    else:
        corr = np.nan
    
    return {
        'slope_mse': mse,
        'slope_mae': mae,
        'slope_rmse': rmse,
        'slope_correlation': corr,
        'n_slopes_compared': len(pred_slopes)
    }


def _compare_intercepts(
    predicted_features: pd.DataFrame,
    ground_truth: dict
) -> dict:
    """Compare predicted intercepts/baselines to ground truth."""
    true_intercepts = ground_truth['true_intercepts']
    
    # Find intercept/baseline column
    intercept_col = _find_column(predicted_features, [
        'intercept', 'baseline', 'eGFR_baseline', 'eGFR_intercept'
    ])
    
    if intercept_col is None:
        return {
            'intercept_mse': np.nan,
            'intercept_mae': np.nan,
            'intercept_correlation': np.nan,
            'intercept_rmse': np.nan
        }
    
    # Align
    pred_intercepts = []
    true_intercepts_aligned = []
    
    for pid in predicted_features['pid'].unique():
        if pid in true_intercepts:
            pred_val = predicted_features.loc[predicted_features['pid'] == pid, intercept_col].iloc[0]
            true_val = true_intercepts[pid]
            
            pred_intercepts.append(pred_val)
            true_intercepts_aligned.append(true_val)
    
    pred_intercepts = np.array(pred_intercepts)
    true_intercepts_aligned = np.array(true_intercepts_aligned)
    
    # Metrics
    mse = np.mean((pred_intercepts - true_intercepts_aligned) ** 2)
    mae = np.mean(np.abs(pred_intercepts - true_intercepts_aligned))
    rmse = np.sqrt(mse)
    
    if len(pred_intercepts) > 1:
        corr, _ = pearsonr(pred_intercepts, true_intercepts_aligned)
    else:
        corr = np.nan
    
    return {
        'intercept_mse': mse,
        'intercept_mae': mae,
        'intercept_rmse': rmse,
        'intercept_correlation': corr,
        'n_intercepts_compared': len(pred_intercepts)
    }


def _compare_type_probs(pred_probs: pd.DataFrame, true_types: Dict[Any, Any]) -> dict:
    """
    Compare per-patient type probabilities to true generating types.
    Expects pred_probs columns: pid, trajtype_<label>_prob...
    true_types: dict pid -> label (same labels as columns after 'trajtype_' and before '_prob')
    """
    # Build label list from columns
    labels = [c[len("trajtype_"):-len("_prob")] for c in pred_probs.columns if c.startswith("trajtype_") and c.endswith("_prob")]
    if not labels:
        return {'type_acc': np.nan, 'type_logloss': np.nan, 'type_brier': np.nan, 'type_macro_f1': np.nan, 'n_types_compared': 0}

    # Align rows with truth
    rows, y_true_idx, proba_mat = [], [], []
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    for _, row in pred_probs.iterrows():
        pid = row['pid']
        if pid not in true_types:
            continue
        ylab = true_types[pid]
        if ylab not in label_to_idx:
            continue
        y_true_idx.append(label_to_idx[ylab])
        proba_mat.append([float(row[f"trajtype_{lab}_prob"]) if f"trajtype_{lab}_prob" in row else 0.0 for lab in labels])
        rows.append(pid)

    if not proba_mat:
        return {'type_acc': np.nan, 'type_logloss': np.nan, 'type_brier': np.nan, 'type_macro_f1': np.nan, 'n_types_compared': 0}

    import numpy as np
    proba = np.clip(np.asarray(proba_mat, dtype=float), 1e-12, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)

    y_true_idx = np.asarray(y_true_idx, dtype=int)
    y_pred_idx = proba.argmax(axis=1)

    acc = float(np.mean(y_pred_idx == y_true_idx))
    try:
        ll = float(log_loss(y_true_idx, proba, labels=list(range(len(labels)))))
    except Exception:
        ll = np.nan
    # Brier (multi-class): mean squared error between one-hot and probs
    onehot = np.eye(len(labels))[y_true_idx]
    brier = float(np.mean((onehot - proba) ** 2))
    try:
        macro_f1 = float(f1_score(y_true_idx, y_pred_idx, average="macro"))
    except Exception:
        macro_f1 = np.nan

    return {'type_acc': acc, 'type_logloss': ll, 'type_brier': brier, 'type_macro_f1': macro_f1, 'n_types_compared': int(len(rows))}


def _find_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Find first matching column name from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def compute_treatment_effect_recovery(
    predicted_features: pd.DataFrame,
    ground_truth: dict,
    treatment_col: str = 'treatment'
) -> dict:
    """
    Evaluate how well the model recovers the treatment effect on trajectories.
    
    Compares:
    - True treatment effect on slope
    - Estimated treatment effect (difference in mean slopes between treated/control)
    
    Parameters
    ----------
    predicted_features : pd.DataFrame
        Features with pid, treatment, and trajectory parameters.
    ground_truth : dict
        Contains 'treatment_effect_on_slope'.
    treatment_col : str
        Column name for treatment indicator.
    
    Returns
    -------
    metrics : dict
        - true_effect: Ground truth treatment effect
        - estimated_effect: Difference in mean predicted slopes
        - effect_bias: Absolute difference
        - effect_recovery_rate: estimated / true (should be ~1.0)
    """
    true_effect = ground_truth.get('treatment_effect_on_slope', np.nan)
    
    # Find slope column
    slope_col = _find_column(predicted_features, ['slope', 'eGFR_slope', 'trajectory_slope'])
    
    if slope_col is None or treatment_col not in predicted_features.columns:
        return {
            'true_effect': true_effect,
            'estimated_effect': np.nan,
            'effect_bias': np.nan,
            'effect_recovery_rate': np.nan
        }
    
    # Compute mean slope for treated vs control
    treated_slopes = predicted_features.loc[predicted_features[treatment_col] == 1, slope_col]
    control_slopes = predicted_features.loc[predicted_features[treatment_col] == 0, slope_col]
    
    estimated_effect = treated_slopes.mean() - control_slopes.mean()
    effect_bias = abs(estimated_effect - true_effect)
    
    if true_effect != 0:
        recovery_rate = estimated_effect / true_effect
    else:
        recovery_rate = np.nan
    
    return {
        'true_effect': true_effect,
        'estimated_effect': estimated_effect,
        'effect_bias': effect_bias,
        'effect_recovery_rate': recovery_rate,
        'n_treated': len(treated_slopes),
        'n_control': len(control_slopes)
    }