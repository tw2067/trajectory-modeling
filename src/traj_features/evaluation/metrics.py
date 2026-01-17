"""
Metrics for comparing trajectory extraction methods.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal


def compare_trajectory_assignments(
    method1_probs: pd.DataFrame,
    method2_probs: pd.DataFrame,
    prob_cols: list[str] = None,
    merge_on: list[str] = None
) -> dict:
    """
    Compare trajectory probability assignments from two methods.
    
    Parameters
    ----------
    method1_probs : DataFrame
        Trajectory probabilities from first method (e.g., Bayesian)
    method2_probs : DataFrame
        Trajectory probabilities from second method (e.g., Bootstrap)
    prob_cols : list of str, optional
        Probability column names to compare
        Default: ['prob_stable', 'prob_gradual_improvement', 'prob_rapid_improvement']
    merge_on : list of str, optional
        Columns to merge on (e.g., ['patientid', 'time_day'])
        
    Returns
    -------
    dict
        Comparison metrics:
        - 'mae': Mean absolute error per probability type
        - 'correlation': Correlation per probability type
        - 'agreement_rate': % of windows with same dominant trajectory
        - 'kappa': Cohen's kappa for dominant trajectory agreement
    """
    if prob_cols is None:
        prob_cols = ['prob_stable', 'prob_gradual_improvement', 'prob_rapid_improvement']
    
    if merge_on is None:
        # Infer merge columns
        common_cols = set(method1_probs.columns) & set(method2_probs.columns)
        id_cols = [c for c in common_cols if c not in prob_cols]
        merge_on = id_cols
    
    # Merge datasets
    merged = method1_probs.merge(
        method2_probs,
        on=merge_on,
        suffixes=('_m1', '_m2')
    )
    
    results = {
        'mae': {},
        'correlation': {},
        'n_windows': len(merged)
    }
    
    # Compute metrics per probability type
    for col in prob_cols:
        col_m1 = f'{col}_m1'
        col_m2 = f'{col}_m2'
        
        if col_m1 in merged.columns and col_m2 in merged.columns:
            vals_m1 = merged[col_m1].values
            vals_m2 = merged[col_m2].values
            
            # Mean absolute error
            mae = np.mean(np.abs(vals_m1 - vals_m2))
            results['mae'][col] = mae
            
            # Correlation
            if len(vals_m1) > 1:
                corr = np.corrcoef(vals_m1, vals_m2)[0, 1]
                results['correlation'][col] = corr
    
    # Dominant trajectory agreement
    prob_cols_m1 = [f'{c}_m1' for c in prob_cols if f'{c}_m1' in merged.columns]
    prob_cols_m2 = [f'{c}_m2' for c in prob_cols if f'{c}_m2' in merged.columns]
    
    if prob_cols_m1 and prob_cols_m2:
        dominant_m1 = merged[prob_cols_m1].idxmax(axis=1).str.replace('_m1', '')
        dominant_m2 = merged[prob_cols_m2].idxmax(axis=1).str.replace('_m2', '')
        
        agreement = (dominant_m1 == dominant_m2).mean()
        results['agreement_rate'] = agreement
        
        # Cohen's kappa
        from sklearn.metrics import cohen_kappa_score
        results['kappa'] = cohen_kappa_score(dominant_m1, dominant_m2)
    
    return results


def trajectory_agreement_score(
    probs1: np.ndarray,
    probs2: np.ndarray,
    method: Literal['mae', 'kl_div', 'js_div'] = 'mae'
) -> float:
    """
    Compute agreement between two probability distributions.
    
    Parameters
    ----------
    probs1, probs2 : array
        Probability vectors (should sum to 1)
    method : str
        'mae': Mean absolute error
        'kl_div': KL divergence
        'js_div': Jensen-Shannon divergence
        
    Returns
    -------
    float
        Agreement score (lower is more similar)
    """
    if method == 'mae':
        return np.mean(np.abs(probs1 - probs2))
    
    elif method == 'kl_div':
        # KL(p1 || p2)
        epsilon = 1e-10
        p1 = np.clip(probs1, epsilon, 1.0)
        p2 = np.clip(probs2, epsilon, 1.0)
        return np.sum(p1 * np.log(p1 / p2))
    
    elif method == 'js_div':
        # Jensen-Shannon divergence (symmetric KL)
        epsilon = 1e-10
        p1 = np.clip(probs1, epsilon, 1.0)
        p2 = np.clip(probs2, epsilon, 1.0)
        m = 0.5 * (p1 + p2)
        kl1 = np.sum(p1 * np.log(p1 / m))
        kl2 = np.sum(p2 * np.log(p2 / m))
        return 0.5 * (kl1 + kl2)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def summarize_comparison(comparison_results: dict) -> None:
    """
    Pretty-print comparison results.
    
    Parameters
    ----------
    comparison_results : dict
        Output from compare_trajectory_assignments
    """
    print("\n" + "="*60)
    print("Trajectory Method Comparison")
    print("="*60)
    
    print(f"\nWindows compared: {comparison_results['n_windows']:,}")
    
    print("\n📊 Probability Agreement (MAE):")
    for prob_type, mae in comparison_results['mae'].items():
        print(f"   {prob_type:35s}: {mae:.3f}")
    
    print("\n📈 Correlation:")
    for prob_type, corr in comparison_results['correlation'].items():
        print(f"   {prob_type:35s}: {corr:.3f}")
    
    if 'agreement_rate' in comparison_results:
        print(f"\n✅ Dominant Trajectory Agreement: {comparison_results['agreement_rate']:.1%}")
        print(f"   Cohen's Kappa: {comparison_results['kappa']:.3f}")
    
    print("="*60)
