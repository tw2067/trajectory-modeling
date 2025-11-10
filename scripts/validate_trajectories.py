import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from traj_ps.evaluation.extractors import extract_trajectory_features
import traceback


"""
Validate trajectory extraction across backends.

For each backend (deep/bayes/gam):
1. Simulate data with known trajectory patterns
2. Fit backend model
3. Extract trajectory features
4. Compare to ground truth
5. Report metrics

Usage:
    python scripts/validate_trajectories.py --backend deep --scenario linear_decline
    python scripts/validate_trajectories.py --backend bayes --scenario nonlinear
    python scripts/validate_trajectories.py --backend gam --scenario heterogeneous
    python scripts/validate_trajectories.py --all  # Test all backends on all scenarios
"""

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from traj_ps.data.simulate import simulate_with_known_trajectories
from traj_ps.data.aggregate import make_bins, DEFAULT_BIN_W
from traj_ps.data.counting import build_counting_process
from traj_ps.evaluation.trajectory_metrics import (
    compare_trajectories,
    compute_treatment_effect_recovery
)


def validate_deep_backend(dyn_df, sta_df, ground_truth, config):
    """Validate deep backend using per-patient OLS features as proxy."""
    print("  [Deep] Extracting OLS trajectory features...")
    pred_features, pred_trajectories = extract_trajectory_features(
        "deep", dyn_df, feature="eGFR", config=config
    )
    metrics = compare_trajectories(
        pred_features, ground_truth, 
        dynamic_df=dyn_df, 
        predicted_trajectories=pred_trajectories
    )
    effect_metrics = compute_treatment_effect_recovery(
        pred_features.merge(sta_df[['pid', 'treatment']], on='pid'),
        ground_truth
    )
    return {**metrics, **effect_metrics}


def validate_bayes_backend(dyn_df, sta_df, ground_truth, config):
    """Validate Bayesian backend using per-patient OLS features as proxy."""
    print("  [Bayes] Extracting trajectory-type probabilities (CPU)...")
    # Configure Bayes with GPU sampler
    from traj_ps.backends.bayes.model import BayesConfig
    import os
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    print(f"  [Bayes] Using CPU with n_jobs={n_cpus}")
    
    bayes_cfg = BayesConfig(
        sampler="pymc",      # Use JAX/NumPyro for GPU
        use_gpu=True,
        chains=4,               # Single chain per window
        n_jobs=-1,               # Sequential to avoid GPU contention
        n_samples=400,          # Reduce for speed
        tune=700,
        target_accept=0.99,
        progressbar=False,
    )
    
    # Pass config to extractor
    config = config or {}
    config['bayes_cfg'] = bayes_cfg
    
    try:
        pred_features, pred_trajectories = extract_trajectory_features(
        "bayes", dyn_df, feature="eGFR", config=config
    )

        metrics = compare_trajectories(
        pred_features, ground_truth,
        dynamic_df=dyn_df,
        predicted_trajectories=pred_trajectories
     )
    
        effect_metrics = compute_treatment_effect_recovery(
        pred_features.merge(sta_df[['pid', 'treatment']], on='pid'),
        ground_truth
        )
        return {**metrics, **effect_metrics}

    except Exception as e:
        print(f"  [ERROR] Exception during Bayes validation: {e}")
        traceback.print_exc()
        return {}
    


def validate_gam_backend(dyn_df, sta_df, ground_truth, config):
    """Validate GAM backend using per-patient OLS features as proxy."""
    print("  [GAM] Extracting OLS trajectory features...")
    pred_features, pred_trajectories = extract_trajectory_features(
        "gam", dyn_df, feature="eGFR", config=config
    )
    metrics = compare_trajectories(
        pred_features, ground_truth,
        dynamic_df=dyn_df,
        predicted_trajectories=pred_trajectories
    )
    effect_metrics = compute_treatment_effect_recovery(
        pred_features.merge(sta_df[['pid', 'treatment']], on='pid'),
        ground_truth
    )
    return {**metrics, **effect_metrics}

def run_validation(backend: str, scenario: str, n_patients: int = 200, seed: int = 920):
    """
    Run validation for one backend on one scenario.
    
    Parameters
    ----------
    backend : str
        'deep', 'bayes', or 'gam'
    scenario : str
        'linear_decline', 'nonlinear', or 'heterogeneous'
    n_patients : int
        Number of patients to simulate
    seed : int
        Random seed
    
    Returns
    -------
    results : dict
        Validation metrics
    """
    print(f"\n{'='*70}")
    print(f"Validating: {backend.upper()} on {scenario}")
    print(f"{'='*70}")
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / f"{backend}.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Simulate data with known trajectories
    print(f"\n1. Simulating {n_patients} patients with {scenario} pattern...")
    dyn_df, sta_df, ground_truth = simulate_with_known_trajectories(
        n_pat=n_patients,
        scenario=scenario,
        treatment_effect_on_slope=0.5,
        seed=seed
    )
    
    print(f"   - Dynamic observations: {len(dyn_df)}")
    print(f"   - Features: {dyn_df['feature_name'].unique().tolist()}")
    print(f"   - Treatment effect on slope: {ground_truth['treatment_effect_on_slope']}")
    
    # Validate backend
    print(f"\n2. Validating {backend} backend...")
    
    try:
        if backend == "deep":
            results = validate_deep_backend(dyn_df, sta_df, ground_truth, config)
        elif backend == "bayes":
            results = validate_bayes_backend(dyn_df, sta_df, ground_truth, config)
        elif backend == "gam":
            results = validate_gam_backend(dyn_df, sta_df, ground_truth, config)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    except Exception as e:
        print(f"   Error in {backend} backend: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Print results
    print(f"\n3. Results:")

    if backend == "bayes":
        # Bayes returns trajectory-type probabilities, not slope/intercept
        print(f"   Trajectory-Type Classification Metrics:")
        print(f"     - Accuracy:         {results.get('type_acc', np.nan):.4f}")
        print(f"     - Log-loss:         {results.get('type_logloss', np.nan):.4f}")
        print(f"     - Brier Score:      {results.get('type_brier', np.nan):.4f}")
        print(f"     - Macro-F1:         {results.get('type_macro_f1', np.nan):.4f}")
        
        if 'type_confusion' in results:
            print(f"\n   Confusion Matrix:")
            print(results['type_confusion'])
        
        # Treatment effect recovery (if available)
        if 'estimated_effect' in results:
            print(f"\n   Treatment Effect Recovery:")
            print(f"     - True effect:      {results.get('true_effect', np.nan):.4f}")
            print(f"     - Estimated effect: {results.get('estimated_effect', np.nan):.4f}")
            print(f"     - Bias:            {results.get('effect_bias', np.nan):.4f}")
            print(f"     - Recovery rate:   {results.get('effect_recovery_rate', np.nan):.4f}")
    else:
        # Deep/GAM return slope/intercept features
        print(f"   Trajectory Reconstruction Metrics:")
        print(f"     - R-squared: {results.get('trajectory_r_squared', np.nan):.4f}")
        print(f"     - MSE:       {results.get('trajectory_mse', np.nan):.4f}")
        print(f"     - MAE:       {results.get('trajectory_mae', np.nan):.4f}")
        
        print(f"\n   Slope Metrics:")
        print(f"     - MSE:         {results.get('slope_mse', np.nan):.4f}")
        print(f"     - MAE:         {results.get('slope_mae', np.nan):.4f}")
        print(f"     - Correlation: {results.get('slope_correlation', np.nan):.4f}")
        
        print(f"\n   Intercept Metrics:")
        print(f"     - MSE:         {results.get('intercept_mse', np.nan):.4f}")
        print(f"     - MAE:         {results.get('intercept_mae', np.nan):.4f}")
        print(f"     - Correlation: {results.get('intercept_correlation', np.nan):.4f}")
        
        print(f"\n   Treatment Effect Recovery:")
        print(f"     - True effect:      {results.get('true_effect', np.nan):.4f}")
        print(f"     - Estimated effect: {results.get('estimated_effect', np.nan):.4f}")
        print(f"     - Bias:            {results.get('effect_bias', np.nan):.4f}")
        print(f"     - Recovery rate:   {results.get('effect_recovery_rate', np.nan):.4f}")
    
    # Pass/Fail criteria - backend-specific
    print(f"\n4. Validation Status:")
    
    checks = []
    
    if backend == "bayes":
        # For Bayesian backend, check trajectory-type classification accuracy
        type_acc = results.get('type_acc', 0)
        if type_acc > 0.5:  # Better than random for 3 classes
            print(f"   ✓ Trajectory-type accuracy > 0.5 ({type_acc:.3f})")
            checks.append(True)
        else:
            print(f"   ✗ Trajectory-type accuracy ≤ 0.5 ({type_acc:.3f})")
            checks.append(False)
        
        # Check log-loss is reasonable
        logloss = results.get('type_logloss', float('inf'))
        if logloss < 1.1:  # -log(1/3) ≈ 1.099 for random guessing
            print(f"   ✓ Log-loss < 1.1 (better than random, {logloss:.3f})")
            checks.append(True)
        else:
            print(f"   ✗ Log-loss ≥ 1.1 ({logloss:.3f})")
            checks.append(False)
    else:
        # Deep/GAM: check slope/intercept recovery
        if scenario == "nonlinear":
            # R-squared should be > 0.3
            r2 = results.get('trajectory_r_squared', 0)
            if r2 > 0.3:
                print(f"   ✓ R-squared > 0.3 ({r2:.3f})")
                checks.append(True)
            else:
                print(f"   ✗ R-squared ≤ 0.3 ({r2:.3f})")
                checks.append(False)
        else:
            # Slope correlation should be > 0.5
            slope_corr = results.get('slope_correlation', 0)
            if slope_corr > 0.5:
                print(f"   ✓ Slope correlation > 0.5 ({slope_corr:.3f})")
                checks.append(True)
            else:
                print(f"   ✗ Slope correlation ≤ 0.5 ({slope_corr:.3f})")
                checks.append(False)
            
            # Effect recovery should be within 50% of true effect
            recovery_rate = results.get('effect_recovery_rate', 0)
            if 0.5 <= recovery_rate <= 1.5:
                print(f"   ✓ Treatment effect recovered (rate: {recovery_rate:.2f})")
                checks.append(True)
            else:
                print(f"   ✗ Treatment effect not recovered (rate: {recovery_rate:.2f})")
                checks.append(False)
    
    if all(checks):
        print(f"\n   ✅ PASSED: {backend} on {scenario}")
    else:
        print(f"\n   ❌ FAILED: {backend} on {scenario}")
    
    return results


def print_summary_table(results_summary_):
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    summary_df = pd.DataFrame(results_summary_)
    if not summary_df.empty:
        bayes_rows = summary_df[summary_df['backend'] == 'bayes']
        other_rows = summary_df[summary_df['backend'] != 'bayes']

        cols = ['backend', 'scenario']
        if 'slope_correlation' in other_rows.columns:
            cols.append('slope_correlation')
        if 'effect_recovery_rate' in other_rows.columns:
            cols.append('effect_recovery_rate')
        
        if not other_rows.empty:
            print("\nDeep/GAM Backends (Slope/Intercept):")
            print(other_rows[cols])
        
        if not bayes_rows.empty:
            print("\nBayes Backend (Trajectory-Type Classification):")
            cols = ['backend', 'scenario']
            if 'type_acc' in bayes_rows.columns:
                cols.append('type_acc')
            if 'type_logloss' in bayes_rows.columns:
                cols.append('type_logloss')
            if 'type_macro_f1' in bayes_rows.columns:
                cols.append('type_macro_f1')
            print(bayes_rows[cols])



def main():
    parser = argparse.ArgumentParser(description="Validate trajectory extraction backends")
    parser.add_argument(
        "--backend",
        choices=["deep", "bayes", "gam"],
        help="Backend to validate"
    )
    parser.add_argument(
        "--scenario",
        choices=["linear_decline", "nonlinear", "heterogeneous"],
        default="linear_decline",
        help="Trajectory scenario to test"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all backends on all scenarios"
    )

    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Test all scenarios for the specified backend"
    )

    parser.add_argument(
        "--all-backends",
        action="store_true",
        help="Test all backends for the specified scenario"
    )

    parser.add_argument(
        "--n-patients",
        type=int,
        default=200,
        help="Number of patients to simulate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=920,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Run all combinations
        backends = ["deep", "bayes", "gam"]
        scenarios = ["linear_decline", "nonlinear", "heterogeneous"]
        
        results_summary = []
        
        for backend in backends:
            for scenario in scenarios:
                result = run_validation(
                    backend=backend,
                    scenario=scenario,
                    n_patients=args.n_patients,
                    seed=args.seed
                )
                
                if result:
                    results_summary.append({
                        'backend': backend,
                        'scenario': scenario,
                        **result
                    })
        # Print summary table
        print_summary_table(results_summary)

    elif args.all_scenarios:
        results_summary = []

        # Run all scenarios for specified backend
        backends = [args.backend]
        scenarios = ["linear_decline", "nonlinear", "heterogeneous"]
        
        for scenario in scenarios:
            result = run_validation(
                backend=args.backend,
                scenario=scenario,
                n_patients=args.n_patients,
                seed=args.seed
            )
            if result:
                results_summary.append({
                    'backend': args.backend,
                    'scenario': scenario,
                    **result
                })
        # Print summary table
        print_summary_table(results_summary)

    elif args.all_backends:
        results_summary = []

        # Run all backends for specified scenario
        backends = ["deep", "bayes", "gam"]
        scenarios = [args.scenario]
        
        for backend in backends:
            result = run_validation(
                backend=backend,
                scenario=args.scenario,
                n_patients=args.n_patients,
                seed=args.seed
            )
            if result:
                results_summary.append({
                    'backend': backend,
                    'scenario': args.scenario,
                    **result
                })
        # Print summary table
        print_summary_table(results_summary)
               
        
    else:
        if not args.backend:
            parser.error("--backend is required (or use --all)")
        
        run_validation(
            backend=args.backend,
            scenario=args.scenario,
            n_patients=args.n_patients,
            seed=args.seed
        )


if __name__ == "__main__":
    main()