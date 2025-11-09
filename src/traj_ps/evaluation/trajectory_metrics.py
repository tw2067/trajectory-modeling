import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy.stats import pearsonr
from sklearn.metrics import log_loss, f1_score, confusion_matrix


def compare_trajectories(
    predicted_features: pd.DataFrame,
    ground_truth: dict,
    dynamic_df: Optional[pd.DataFrame] = None,
    predicted_trajectories: Optional[pd.DataFrame] = None,
    metric: str = "mse"
) -> dict:
    """
    Compare predicted trajectory features to ground truth.
    
    For slope/intercept backends (deep, gam):
        - Compute slope/intercept errors
        - Compute trajectory reconstruction R²
    
    For probability backends (bayes):
        - Compute classification metrics (accuracy, log-loss, etc.)
    """
    metrics: dict = {}

    # Detect backend type by checking columns
    slope_candidates = ['slope', 'eGFR_slope', 'trajectory_slope']
    has_slopes = any(c in predicted_features.columns for c in slope_candidates)
    has_probs = any(col.startswith('trajtype_') and col.endswith('_prob') 
                    for col in predicted_features.columns)
    
    if has_probs:
        # Bayes backend: compute classification metrics
        metrics.update(_compute_classification_metrics(predicted_features, ground_truth))
    
    if has_slopes:
        # Deep/GAM backend: compute regression metrics
        metrics.update(_compute_regression_metrics(
            predicted_features, ground_truth, dynamic_df, predicted_trajectories
        ))
    
    return metrics


def _compute_classification_metrics(
    predicted_features: pd.DataFrame,
    ground_truth: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute trajectory-type classification metrics for Bayes backend."""
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, f1_score
    
    # Extract true labels from ground truth
    if 'trajectory_types' not in ground_truth:
        return {
            'type_acc': np.nan,
            'type_logloss': np.nan,
            'type_brier': np.nan,
            'type_macro_f1': np.nan,
            'type_confusion': None,
        }
    
    true_types = ground_truth['trajectory_types']  # Series: pid -> type label
    
    # Get probability columns
    prob_cols = [c for c in predicted_features.columns 
                 if c.startswith('trajtype_') and c.endswith('_prob')]
    
    if not prob_cols:
        return {
            'type_acc': np.nan,
            'type_logloss': np.nan,
            'type_brier': np.nan,
            'type_macro_f1': np.nan,
            'type_confusion': None,
        }
    
    # Extract class labels from column names
    class_labels = sorted([c.replace('trajtype_', '').replace('_prob', '') for c in prob_cols])

    # Convert true_types to DataFrame if it's a dict or Series
    if isinstance(true_types, dict):
        true_types_df = pd.DataFrame.from_dict(true_types, orient='index', columns=['true_type'])
        true_types_df['pid'] = true_types_df.index
        true_types_df = true_types_df.reset_index(drop=True)
    elif isinstance(true_types, pd.Series):
        true_types_df = true_types.reset_index()
        true_types_df.columns = ['pid', 'true_type']
    else:
        # Assume it's already a DataFrame
        true_types_df = true_types
    
    print(f"[DEBUG] prob_cols={prob_cols}")
    print(f"[DEBUG] has trajectory_types={ 'trajectory_types' in ground_truth }")
    
    # Merge predictions with true labels
    merged = predicted_features[['pid'] + prob_cols].merge(
        true_types[['pid', 'true_type']],
        on='pid',
        how='inner'
    )

    print(f"[DEBUG] merged rows for classification={len(merged)}")
    
    if merged.empty:
        print(f"[WARNING] No matching PIDs between predictions and ground truth.")
        return {
            'type_acc': np.nan,
            'type_logloss': np.nan,
            'type_brier': np.nan,
            'type_macro_f1': np.nan,
            'type_confusion': None,
        }
    
    # Predicted class (argmax)
    y_pred = merged[prob_cols].values.argmax(axis=1)
    y_pred_labels = [class_labels[i] for i in y_pred]
    
    # True class
    y_true_labels = merged['true_type'].values
    
    # Compute metrics
    acc = accuracy_score(y_true_labels, y_pred_labels)
    
    # For log-loss and brier, need consistent label encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(class_labels)
    y_true_encoded = le.transform(y_true_labels)
    
    # Probability matrix (ensure column order matches class_labels)
    y_prob = merged[[f'trajtype_{label}_prob' for label in class_labels]].values

    # Clip probabilities to avoid log(0)
    y_prob = np.clip(y_prob, 1e-12, 1.0)
    # Renormalize to ensure they sum to 1
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    try:
        logloss = log_loss(y_true_encoded, y_prob)
    except Exception as e:
        print(f"[WARNING] Log-loss computation failed: {e}")
        logloss = np.nan
    
    try:
        # Brier score (multi-class): average over classes
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true_encoded, classes=range(len(class_labels)))
        brier = np.mean([brier_score_loss(y_true_bin[:, i], y_prob[:, i]) 
                        for i in range(len(class_labels))])
    except Exception as e:
        print(f"[WARNING] Brier score computation failed: {e}")
        brier = np.nan
    
    try:
        macro_f1 = f1_score(y_true_labels, y_pred_labels, average='macro')
    except Exception as e:
        print(f"[WARNING] F1-score computation failed: {e}")
        macro_f1 = np.nan

    # Confusion matrix
    try:
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=class_labels)
        cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
        cm_df.index.name = 'True'
        cm_df.columns.name = 'Predicted'
    except Exception as e:
        print(f"[WARNING] Confusion matrix computation failed: {e}")
        cm_df = None
    
    return {
        'type_acc': acc,
        'type_logloss': logloss,
        'type_brier': brier,
        'type_macro_f1': macro_f1,
        'type_confusion': cm_df,
        'n_types_compared': len(merged)
    }


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


def _compute_regression_metrics(
    predicted_features: pd.DataFrame,
    ground_truth: Dict[str, Any],
    dynamic_df: Optional[pd.DataFrame],
    predicted_trajectories: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """
    Compute slope/intercept regression metrics for Deep/GAM backends.
    
    Metrics include:
    - Trajectory reconstruction (R², MSE, MAE) if trajectories provided
    - Slope comparison (MSE, MAE, correlation)
    - Intercept comparison (MSE, MAE, correlation)
    """
    metrics = {}
    
    # 1. Compare full trajectories (if available)
    if dynamic_df is not None and predicted_trajectories is not None and not predicted_trajectories.empty:
        traj_metrics = _compare_full_trajectories(
            dynamic_df=dynamic_df,
            predicted_trajectories=predicted_trajectories,
            feature="eGFR"
        )
        metrics.update(traj_metrics)
    else:
        # No trajectory reconstruction available
        metrics.update({
            'trajectory_r_squared': np.nan,
            'trajectory_mse': np.nan,
            'trajectory_mae': np.nan,
            'n_trajectory_points': 0
        })
    
    # 2. Compare slopes
    if 'true_slopes' in ground_truth:
        slope_metrics = _compare_slopes(predicted_features, ground_truth)
        metrics.update(slope_metrics)
    else:
        metrics.update({
            'slope_mse': np.nan,
            'slope_mae': np.nan,
            'slope_rmse': np.nan,
            'slope_correlation': np.nan,
            'n_slopes_compared': 0
        })
    
    # 3. Compare intercepts/baselines
    if 'true_intercepts' in ground_truth:
        intercept_metrics = _compare_intercepts(predicted_features, ground_truth)
        metrics.update(intercept_metrics)
    else:
        metrics.update({
            'intercept_mse': np.nan,
            'intercept_mae': np.nan,
            'intercept_rmse': np.nan,
            'intercept_correlation': np.nan,
            'n_intercepts_compared': 0
        })
    
    return metrics


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