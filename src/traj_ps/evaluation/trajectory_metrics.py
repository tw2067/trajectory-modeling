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

    primary_feature = ground_truth.get('primary_feature', 'eGFR')

    # Detect backend type by checking columns
    slope_candidates = ['slope', f'{primary_feature}_slope', 'trajectory_slope']
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
    from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, f1_score, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    
    # Extract true labels from ground truth
    if 'trajectory_types' not in ground_truth:
        return {
            'type_acc': np.nan,
            'type_logloss': np.nan,
            'type_brier': np.nan,
            'type_macro_f1': np.nan,
            'type_confusion': None,
            'n_types_compared': 0
        }

    true_types = ground_truth['trajectory_types']
    
    # Get probability columns from predictions
    prob_cols = [c for c in predicted_features.columns 
                 if c.startswith('trajtype_') and c.endswith('_prob')]
    
    if not prob_cols:
        print(f"[WARNING] No trajtype probability columns found in predictions")
        print(f"[DEBUG] Available columns: {predicted_features.columns.tolist()}")
        return {
            'type_acc': np.nan,
            'type_logloss': np.nan,
            'type_brier': np.nan,
            'type_macro_f1': np.nan,
            'type_confusion': None,
            'n_types_compared': 0
        }
    
    # Extract predicted class labels from column names
    pred_class_labels = sorted([c.replace('trajtype_', '').replace('_prob', '') for c in prob_cols])
    
    # Convert true_types to DataFrame with standardized format
    if isinstance(true_types, dict):
        true_types_df = pd.DataFrame([
            {'pid': pid, 'true_type': str(ttype)}
            for pid, ttype in true_types.items()
        ])
    elif isinstance(true_types, pd.Series):
        true_types_df = pd.DataFrame({
            'pid': true_types.index,
            'true_type': true_types.values.astype(str)
        })
    elif isinstance(true_types, pd.DataFrame):
        if 'true_type' not in true_types.columns:
            # Assume structure: index=pid, single column is type
            true_types_df = pd.DataFrame({
                'pid': true_types.index,
                'true_type': true_types.iloc[:, 0].astype(str)
            })
        else:
            true_types_df = true_types.copy()
            true_types_df['true_type'] = true_types_df['true_type'].astype(str)
    else:
        print(f"[ERROR] Unexpected trajectory_types format: {type(true_types)}")
        return {
            'type_acc': np.nan,
            'type_logloss': np.nan,
            'type_brier': np.nan,
            'type_macro_f1': np.nan,
            'type_confusion': None,
            'n_types_compared': 0
        }
    
    # Ensure pid column exists and is correct type
    if 'pid' not in true_types_df.columns:
        print(f"[ERROR] 'pid' column missing from true_types_df")
        return {
            'type_acc': np.nan,
            'type_logloss': np.nan,
            'type_brier': np.nan,
            'type_macro_f1': np.nan,
            'type_confusion': None,
            'n_types_compared': 0
        }
    
    # Merge predictions with true labels
    merged = predicted_features[['pid'] + prob_cols].merge(
        true_types_df[['pid', 'true_type']],
        on='pid',
        how='inner'
    )
    
    if merged.empty:
        print(f"[WARNING] No matching PIDs between predictions and ground truth.")
        print(f"[DEBUG] Predicted PIDs sample: {predicted_features['pid'].head().tolist()}")
        print(f"[DEBUG] True PIDs sample: {true_types_df['pid'].head().tolist()}")
        return {
            'type_acc': np.nan,
            'type_logloss': np.nan,
            'type_brier': np.nan,
            'type_macro_f1': np.nan,
            'type_confusion': None,
            'n_types_compared': 0
        }
    
    # Get unique true labels from merged data
    true_class_labels = sorted(merged['true_type'].unique())
    
    # Create union of predicted and true labels
    all_class_labels = sorted(set(pred_class_labels) | set(true_class_labels))
    
    print(f"[DEBUG] Predicted labels: {pred_class_labels}")
    print(f"[DEBUG] True labels: {true_class_labels}")
    print(f"[DEBUG] All labels: {all_class_labels}")
    
    # Check for label mismatch
    if set(pred_class_labels) != set(true_class_labels):
        print(f"[WARNING] Label mismatch detected!")
        print(f"[WARNING] Predicted but not in truth: {set(pred_class_labels) - set(true_class_labels)}")
        print(f"[WARNING] In truth but not predicted: {set(true_class_labels) - set(pred_class_labels)}")
    
    # Ensure probability columns exist for all classes
    for label in all_class_labels:
        prob_col = f'trajtype_{label}_prob'
        if prob_col not in merged.columns:
            merged[prob_col] = 0.0
            print(f"[WARNING] Added missing probability column: {prob_col} (set to 0.0)")
    
    # Update prob_cols to include all labels
    prob_cols = [f'trajtype_{label}_prob' for label in all_class_labels]
    
    # Get probability matrix
    y_prob = merged[prob_cols].values
    
    # Normalize probabilities (handle cases where they don't sum to 1)
    row_sums = y_prob.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)  # Avoid division by zero
    y_prob = y_prob / row_sums
    
    # Clip for numerical stability
    y_prob = np.clip(y_prob, 1e-12, 1.0)
    
    # Predicted class (argmax)
    y_pred_idx = y_prob.argmax(axis=1)
    y_pred_labels = [all_class_labels[i] for i in y_pred_idx]
    
    # True labels
    y_true_labels = merged['true_type'].values
    
    # Encode labels consistently
    le = LabelEncoder()
    le.fit(all_class_labels)
    
    try:
        y_true_encoded = le.transform(y_true_labels)
        y_pred_encoded = le.transform(y_pred_labels)
    except ValueError as e:
        print(f"[ERROR] Label encoding failed: {e}")
        print(f"[DEBUG] Sample true values: {y_true_labels[:5]}")
        print(f"[DEBUG] Sample pred values: {y_pred_labels[:5]}")
        return {
            'type_acc': np.nan,
            'type_logloss': np.nan,
            'type_brier': np.nan,
            'type_macro_f1': np.nan,
            'type_confusion': None,
            'n_types_compared': 0
        }
    
    # Compute metrics
    acc = accuracy_score(y_true_encoded, y_pred_encoded)
    
    try:
        logloss = log_loss(y_true_encoded, y_prob, labels=list(range(len(all_class_labels))))
    except Exception as e:
        print(f"[WARNING] Log-loss computation failed: {e}")
        logloss = np.nan
    
    try:
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true_encoded, classes=range(len(all_class_labels)))
        brier = np.mean([brier_score_loss(y_true_bin[:, i], y_prob[:, i]) 
                        for i in range(len(all_class_labels))])
    except Exception as e:
        print(f"[WARNING] Brier score computation failed: {e}")
        brier = np.nan
    
    try:
        macro_f1 = f1_score(y_true_encoded, y_pred_encoded, average='macro')
    except Exception as e:
        print(f"[WARNING] F1-score computation failed: {e}")
        macro_f1 = np.nan

    # Confusion matrix (use string labels for readability)
    try:
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=all_class_labels)
        cm_df = pd.DataFrame(cm, index=all_class_labels, columns=all_class_labels)
        cm_df.index.name = 'True'
        cm_df.columns.name = 'Predicted'
        
        print("\n[DEBUG] Confusion Matrix:")
        print(cm_df)
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
    primary_feature = ground_truth.get('primary_feature', 'eGFR')
    
    # Find slope column - try feature-specific name first, then generic
    slope_col = _find_column(predicted_features, [
        f'{primary_feature}_slope',
        'slope',
        'trajectory_slope'
    ])
    
    if slope_col is None:
        print(f"[WARNING] No slope column found for feature '{primary_feature}'")
        print(f"[DEBUG] Available columns: {predicted_features.columns.tolist()}")
        return {
            'slope_mse': np.nan,
            'slope_mae': np.nan,
            'slope_correlation': np.nan,
            'slope_rmse': np.nan,
            'n_slopes_compared': 0
        }
    
    print(f"[DEBUG] Using slope column: {slope_col}")
    
    # Align predictions with ground truth
    pred_slopes = []
    true_slopes_aligned = []
    
    for pid in predicted_features['pid'].unique():
        if pid in true_slopes:
            pred_row = predicted_features[predicted_features['pid'] == pid]
            if pred_row.empty:
                continue
                
            pred_val = pred_row[slope_col].iloc[0]
            
            # Skip NaN predictions
            if not np.isfinite(pred_val):
                continue
                
            true_val = true_slopes[pid]
            
            # Handle different slope formats
            if isinstance(true_val, dict):
                # Priority order: mean (nonprogression), linear (mixed), b1 (nonlinear)
                if 'mean' in true_val:
                    # Nonprogression scenario with local slopes
                    true_val = true_val['mean']
                elif 'linear' in true_val:
                    # Mixed scenario stores linear term in 'linear' key
                    true_val = true_val['linear']
                elif 'b1' in true_val:
                    # Pure nonlinear scenario stores linear term in 'b1' key
                    true_val = true_val['b1']
                else:
                    # Fallback: take first numeric value
                    print(f"[WARNING] Unexpected slope dict keys for {pid}: {true_val.keys()}")
                    true_val = next((v for v in true_val.values() if isinstance(v, (int, float))), 0)
            
            pred_slopes.append(pred_val)
            true_slopes_aligned.append(true_val)
    
    if len(pred_slopes) == 0:
        print(f"[WARNING] No valid slope comparisons could be made")
        return {
            'slope_mse': np.nan,
            'slope_mae': np.nan,
            'slope_rmse': np.nan,
            'slope_correlation': np.nan,
            'n_slopes_compared': 0
        }
    
    pred_slopes = np.array(pred_slopes)
    true_slopes_aligned = np.array(true_slopes_aligned)
    
    # Debug: print sample of extracted slopes
    print(f"[DEBUG] Sample true slopes (first 5): {true_slopes_aligned[:5]}")
    print(f"[DEBUG] Sample pred slopes (first 5): {pred_slopes[:5]}")
    
    # Compute metrics
    mse = np.mean((pred_slopes - true_slopes_aligned) ** 2)
    mae = np.mean(np.abs(pred_slopes - true_slopes_aligned))
    rmse = np.sqrt(mse)
    
    # Compute correlation with proper handling of constant arrays
    if len(pred_slopes) > 1:
        # Check if either array is constant (no variance)
        pred_std = np.std(pred_slopes)
        true_std = np.std(true_slopes_aligned)
        
        print(f"[DEBUG] Pred slopes std: {pred_std:.6f}, True slopes std: {true_std:.6f}")
        
        if pred_std < 1e-10 or true_std < 1e-10:
            # At least one array is constant
            if pred_std < 1e-10 and true_std < 1e-10:
                # Both constant - perfect agreement if values match
                if np.abs(pred_slopes[0] - true_slopes_aligned[0]) < 1e-6:
                    corr = 1.0
                    print(f"[DEBUG] Both predictions and truth are constant at {pred_slopes[0]:.4f}")
                else:
                    corr = np.nan
                    print(f"[WARNING] Both arrays constant but different values: pred={pred_slopes[0]:.4f}, true={true_slopes_aligned[0]:.4f}")
            elif pred_std < 1e-10:
                corr = np.nan
                print(f"[WARNING] Predicted slopes are constant at {pred_slopes[0]:.4f} (std={pred_std:.6f})")
            else:
                corr = np.nan
                print(f"[WARNING] True slopes are constant at {true_slopes_aligned[0]:.4f} (std={true_std:.6f})")
        else:
            # Both arrays have variance, compute correlation
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    corr, p_value = pearsonr(pred_slopes, true_slopes_aligned)
                    
                    # Check if correlation is significant
                    if not np.isfinite(corr):
                        corr = np.nan
                        print(f"[WARNING] Correlation computation returned non-finite value")
                    elif len(pred_slopes) >= 3 and p_value > 0.05:
                        print(f"[INFO] Correlation not significant: r={corr:.4f}, p={p_value:.4f}")
            except Exception as e:
                corr = np.nan
                print(f"[WARNING] Correlation computation failed: {e}")
    else:
        corr = np.nan
        print(f"[INFO] Only 1 slope comparison - correlation not computable")
    
    # FIX: Format correlation separately
    corr_str = f"{corr:.4f}" if np.isfinite(corr) else "N/A"
    print(f"[DEBUG] Compared {len(pred_slopes)} slopes: MSE={mse:.4f}, MAE={mae:.4f}, Corr={corr_str}")
    
    return {
        'slope_mse': float(mse),
        'slope_mae': float(mae),
        'slope_rmse': float(rmse),
        'slope_correlation': float(corr) if np.isfinite(corr) else np.nan,
        'n_slopes_compared': len(pred_slopes)
    }


def _compare_intercepts(
    predicted_features: pd.DataFrame,
    ground_truth: dict
) -> dict:
    """Compare predicted intercepts/baselines to ground truth."""
    true_intercepts = ground_truth['true_intercepts']
    primary_feature = ground_truth.get('primary_feature', 'eGFR')
    
    # Find intercept/baseline column - try feature-specific name first
    intercept_col = _find_column(predicted_features, [
        f'{primary_feature}_intercept',
        f'{primary_feature}_baseline',
        'intercept',
        'baseline'
    ])
    
    if intercept_col is None:
        print(f"[WARNING] No intercept column found for feature '{primary_feature}'")
        print(f"[DEBUG] Available columns: {predicted_features.columns.tolist()}")
        return {
            'intercept_mse': np.nan,
            'intercept_mae': np.nan,
            'intercept_correlation': np.nan,
            'intercept_rmse': np.nan,
            'n_intercepts_compared': 0
        }
    
    print(f"[DEBUG] Using intercept column: {intercept_col}")
    
    # Align
    pred_intercepts = []
    true_intercepts_aligned = []
    
    for pid in predicted_features['pid'].unique():
        if pid in true_intercepts:
            pred_row = predicted_features[predicted_features['pid'] == pid]
            if pred_row.empty:
                continue
                
            pred_val = pred_row[intercept_col].iloc[0]
            
            # Skip NaN predictions
            if not np.isfinite(pred_val):
                continue
                
            true_val = true_intercepts[pid]
            
            pred_intercepts.append(pred_val)
            true_intercepts_aligned.append(true_val)
    
    if len(pred_intercepts) == 0:
        print(f"[WARNING] No valid intercept comparisons could be made")
        return {
            'intercept_mse': np.nan,
            'intercept_mae': np.nan,
            'intercept_rmse': np.nan,
            'intercept_correlation': np.nan,
            'n_intercepts_compared': 0
        }
    
    pred_intercepts = np.array(pred_intercepts)
    true_intercepts_aligned = np.array(true_intercepts_aligned)
    
    # Metrics
    mse = np.mean((pred_intercepts - true_intercepts_aligned) ** 2)
    mae = np.mean(np.abs(pred_intercepts - true_intercepts_aligned))
    rmse = np.sqrt(mse)
    
    # Compute correlation with proper handling
    if len(pred_intercepts) > 1:
        pred_std = np.std(pred_intercepts)
        true_std = np.std(true_intercepts_aligned)
        
        if pred_std < 1e-10 or true_std < 1e-10:
            if pred_std < 1e-10 and true_std < 1e-10:
                if np.abs(pred_intercepts[0] - true_intercepts_aligned[0]) < 1e-6:
                    corr = 1.0
                    print(f"[DEBUG] Both predictions and truth are constant at {pred_intercepts[0]:.4f}")
                else:
                    corr = np.nan
                    print(f"[WARNING] Both arrays constant but different: pred={pred_intercepts[0]:.4f}, true={true_intercepts_aligned[0]:.4f}")
            elif pred_std < 1e-10:
                corr = np.nan
                print(f"[WARNING] Predicted intercepts are constant at {pred_intercepts[0]:.4f}")
            else:
                corr = np.nan
                print(f"[WARNING] True intercepts are constant at {true_intercepts_aligned[0]:.4f}")
        else:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    corr, p_value = pearsonr(pred_intercepts, true_intercepts_aligned)
                    
                    if not np.isfinite(corr):
                        corr = np.nan
                        print(f"[WARNING] Correlation computation returned non-finite value")
                    elif len(pred_intercepts) >= 3 and p_value > 0.05:
                        print(f"[INFO] Correlation not significant: r={corr:.4f}, p={p_value:.4f}")
            except Exception as e:
                corr = np.nan
                print(f"[WARNING] Correlation computation failed: {e}")
    else:
        corr = np.nan
        print(f"[INFO] Only 1 intercept comparison - correlation not computable")
    
    # FIX: Format correlation separately
    corr_str = f"{corr:.4f}" if np.isfinite(corr) else "N/A"
    print(f"[DEBUG] Compared {len(pred_intercepts)} intercepts: MSE={mse:.4f}, MAE={mae:.4f}, Corr={corr_str}")
    
    return {
        'intercept_mse': float(mse),
        'intercept_mae': float(mae),
        'intercept_rmse': float(rmse),
        'intercept_correlation': float(corr) if np.isfinite(corr) else np.nan,
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

    primary_feature = ground_truth.get('primary_feature', 'eGFR')

    # 1. Compare full trajectories (if available)
    if dynamic_df is not None and predicted_trajectories is not None and not predicted_trajectories.empty:
        traj_metrics = _compare_full_trajectories(
            dynamic_df=dynamic_df,
            predicted_trajectories=predicted_trajectories,
            feature=primary_feature
        )
        metrics.update(traj_metrics)
    else:
        # No trajectory reconstruction available
        print("[DEBUG] Skipping trajectory comparison")
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
        print("[DEBUG] No true_slopes in ground_truth, skipping")
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
        print("[DEBUG] No true_intercepts in ground_truth, skipping")
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


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """
    Find first matching column name from candidates list.
    Returns None if no match found.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def compute_treatment_effect_recovery(
    predicted_features: pd.DataFrame,
    ground_truth: dict,
    treatment_col: str = 'treatment'
) -> dict:
    """Evaluate how well the model recovers the treatment effect on trajectories."""
    true_effect = ground_truth.get('treatment_effect_on_slope', np.nan)
    primary_feature = ground_truth.get('primary_feature', 'eGFR')
    
    # Find slope column - try feature-specific first
    slope_col = _find_column(predicted_features, [
        f'{primary_feature}_slope',
        'slope',
        'trajectory_slope'
    ])
    
    if slope_col is None or treatment_col not in predicted_features.columns:
        print(f"[WARNING] Cannot compute treatment effect: slope_col={slope_col}, has_treatment={treatment_col in predicted_features.columns}")
        return {
            'true_effect': true_effect,
            'estimated_effect': np.nan,
            'effect_bias': np.nan,
            'effect_recovery_rate': np.nan,
            'n_treated': 0,
            'n_control': 0
        }
    
    # Filter out NaN slopes
    valid_data = predicted_features[predicted_features[slope_col].notna()].copy()
    
    # Compute mean slope for treated vs control
    treated_slopes = valid_data.loc[valid_data[treatment_col] == 1, slope_col]
    control_slopes = valid_data.loc[valid_data[treatment_col] == 0, slope_col]
    
    if len(treated_slopes) == 0 or len(control_slopes) == 0:
        print(f"[WARNING] Insufficient data: n_treated={len(treated_slopes)}, n_control={len(control_slopes)}")
        return {
            'true_effect': true_effect,
            'estimated_effect': np.nan,
            'effect_bias': np.nan,
            'effect_recovery_rate': np.nan,
            'n_treated': len(treated_slopes),
            'n_control': len(control_slopes)
        }
    
    estimated_effect = treated_slopes.mean() - control_slopes.mean()
    effect_bias = abs(estimated_effect - true_effect)
    
    if abs(true_effect) > 1e-6:
        recovery_rate = estimated_effect / true_effect
    else:
        recovery_rate = np.nan
    
    print(f"[DEBUG] Treatment effect: true={true_effect:.4f}, estimated={estimated_effect:.4f}, rate={recovery_rate:.4f}")
    
    return {
        'true_effect': float(true_effect),
        'estimated_effect': float(estimated_effect),
        'effect_bias': float(effect_bias),
        'effect_recovery_rate': float(recovery_rate),
        'n_treated': int(len(treated_slopes)),
        'n_control': int(len(control_slopes))
    }