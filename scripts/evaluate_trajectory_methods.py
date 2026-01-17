#!/usr/bin/env python3
"""
Comprehensive evaluation: Compare trajectory extraction methods.

Evaluates both trajectory assignment agreement and downstream prediction performance.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from traj_features.evaluation.metrics import compare_trajectory_assignments, summarize_comparison
from traj_features.evaluation.prediction import compare_feature_sets, print_prediction_comparison


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate and compare trajectory extraction methods'
    )
    parser.add_argument('--bayesian-probs', type=str, required=True,
                       help='Path to Bayesian trajectory probabilities CSV')
    parser.add_argument('--bootstrap-probs', type=str, required=True,
                       help='Path to Bootstrap trajectory probabilities CSV')
    parser.add_argument('--outcomes', type=str, required=True,
                       help='Path to outcomes CSV (must have patientid and outcome column)')
    parser.add_argument('--outcome-col', type=str, default='outcome',
                       help='Name of binary outcome column')
    parser.add_argument('--merge-on', type=str, nargs='+', default=['patientid', 'time_day'],
                       help='Columns to merge datasets on')
    parser.add_argument('--output', type=str, default='results/evaluation_report.csv',
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Trajectory Method Evaluation Framework")
    print("="*80)
    
    # Load data
    print(f"\n📁 Loading data...")
    bayes_probs = pd.read_csv(args.bayesian_probs)
    boot_probs = pd.read_csv(args.bootstrap_probs)
    outcomes = pd.read_csv(args.outcomes)
    
    print(f"   Bayesian: {len(bayes_probs):,} windows")
    print(f"   Bootstrap: {len(boot_probs):,} windows")
    print(f"   Outcomes: {len(outcomes):,} patients")
    
    # Part 1: Compare trajectory assignments
    print("\n" + "="*80)
    print("PART 1: Trajectory Assignment Comparison")
    print("="*80)
    
    comparison = compare_trajectory_assignments(
        method1_probs=bayes_probs,
        method2_probs=boot_probs,
        merge_on=args.merge_on
    )
    
    summarize_comparison(comparison)
    
    # Part 2: Downstream prediction evaluation
    print("\n" + "="*80)
    print("PART 2: Downstream Prediction Performance")
    print("="*80)
    
    # Merge trajectory features with outcomes
    # Take last observation per patient
    bayes_last = bayes_probs.sort_values('time_day').groupby('patientid').last().reset_index()
    boot_last = boot_probs.sort_values('time_day').groupby('patientid').last().reset_index()
    
    # Rename columns to distinguish methods
    prob_cols = ['prob_stable', 'prob_gradual_improvement', 'prob_rapid_improvement']
    bayes_last = bayes_last.rename(columns={c: f'{c}_bayes' for c in prob_cols})
    boot_last = boot_last.rename(columns={c: f'{c}_boot' for c in prob_cols})
    
    # Merge all data
    data = outcomes.merge(bayes_last, on='patientid', how='inner')
    data = data.merge(boot_last, on='patientid', how='inner')
    
    print(f"\nMerged dataset: {len(data):,} patients")
    
    # Define feature sets
    feature_sets = {
        'Bayesian Trajectories': [f'{c}_bayes' for c in prob_cols],
        'Bootstrap Trajectories': [f'{c}_boot' for c in prob_cols],
    }
    
    # Evaluate
    results = compare_feature_sets(
        data=data,
        outcome_col=args.outcome_col,
        feature_sets=feature_sets,
        model_type='logistic',
        cv_folds=5
    )
    
    print_prediction_comparison(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    
    print(f"\n✅ Evaluation results saved: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
