"""
Evaluate trajectory features in downstream prediction tasks.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def evaluate_downstream_prediction(
    features: pd.DataFrame,
    outcomes: pd.Series,
    feature_cols: list[str],
    model_type: Literal['logistic', 'rf', 'xgboost'] = 'logistic',
    cv_folds: int = 5,
    random_state: int = 42
) -> dict:
    """
    Evaluate trajectory features in a downstream prediction task.
    
    Parameters
    ----------
    features : DataFrame
        Trajectory features and other covariates
    outcomes : Series
        Binary outcome to predict
    feature_cols : list of str
        Feature columns to use
    model_type : str
        'logistic', 'rf', or 'xgboost'
    cv_folds : int
        Number of cross-validation folds
    random_state : int
        Random seed
        
    Returns
    -------
    dict
        Evaluation metrics:
        - 'auroc': Area under ROC curve
        - 'auprc': Area under precision-recall curve
        - 'brier': Brier score
        - 'cv_scores': Cross-validation scores
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    X = features[feature_cols].values
    y = outcomes.values
    
    # Remove any NaNs
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(y) == 0:
        return {'error': 'No valid samples'}
    
    # Select model
    if model_type == 'logistic':
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    elif model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            max_depth=5
        )
    elif model_type == 'xgboost':
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=100,
                random_state=random_state,
                max_depth=3,
                eval_metric='logloss'
            )
        except ImportError:
            print("XGBoost not installed, falling back to logistic regression")
            model = LogisticRegression(random_state=random_state, max_iter=1000)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Cross-validation
    cv_scores = cross_val_score(
        model, X, y,
        cv=cv_folds,
        scoring='roc_auc'
    )
    
    # Fit on full data for other metrics
    model.fit(X, y)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    results = {
        'model_type': model_type,
        'n_samples': len(y),
        'n_features': X.shape[1],
        'cv_scores': cv_scores,
        'mean_cv_auroc': cv_scores.mean(),
        'std_cv_auroc': cv_scores.std(),
        'auroc': roc_auc_score(y, y_pred_proba),
        'auprc': average_precision_score(y, y_pred_proba),
        'brier': brier_score_loss(y, y_pred_proba)
    }
    
    return results


def compare_feature_sets(
    data: pd.DataFrame,
    outcome_col: str,
    feature_sets: dict[str, list[str]],
    model_type: str = 'logistic',
    cv_folds: int = 5
) -> pd.DataFrame:
    """
    Compare multiple feature sets (e.g., Bayesian vs Bootstrap trajectories).
    
    Parameters
    ----------
    data : DataFrame
        Combined dataset with all features
    outcome_col : str
        Name of outcome column
    feature_sets : dict
        Mapping of feature set name to list of feature columns
        Example: {
            'bayesian_traj': ['prob_stable_bayes', 'prob_increase_bayes'],
            'bootstrap_traj': ['prob_stable_boot', 'prob_increase_boot'],
            'baseline': ['age', 'sex']
        }
    model_type : str
        Model to use
    cv_folds : int
        Number of CV folds
        
    Returns
    -------
    DataFrame
        Comparison results for each feature set
    """
    results = []
    
    for name, features in feature_sets.items():
        print(f"\nEvaluating: {name}")
        print(f"  Features: {len(features)}")
        
        result = evaluate_downstream_prediction(
            features=data,
            outcomes=data[outcome_col],
            feature_cols=features,
            model_type=model_type,
            cv_folds=cv_folds
        )
        result['feature_set'] = name
        results.append(result)
    
    return pd.DataFrame(results)


def print_prediction_comparison(results_df: pd.DataFrame) -> None:
    """
    Pretty-print prediction comparison results.
    
    Parameters
    ----------
    results_df : DataFrame
        Output from compare_feature_sets
    """
    print("\n" + "="*70)
    print("Downstream Prediction Performance Comparison")
    print("="*70)
    
    # Sort by AUROC
    results_df = results_df.sort_values('mean_cv_auroc', ascending=False)
    
    print(f"\n{'Feature Set':<25} {'CV AUROC':>12} {'AUPRC':>10} {'Brier':>10}")
    print("-" * 70)
    
    for _, row in results_df.iterrows():
        print(
            f"{row['feature_set']:<25} "
            f"{row['mean_cv_auroc']:>7.3f} ± {row['std_cv_auroc']:<.3f} "
            f"{row['auprc']:>10.3f} "
            f"{row['brier']:>10.3f}"
        )
    
    print("="*70)
