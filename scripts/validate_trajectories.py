import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from traj_ps.evaluation.extractors import extract_trajectory_features


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
    print("  [Bayes] Extracting trajectory-type probabilities (GPU)...")
    # Configure Bayes with GPU sampler
    from traj_ps.backends.bayes.model import BayesConfig
    
    bayes_cfg = BayesConfig(
        sampler="nutpie",      # Use JAX/NumPyro for GPU
        use_gpu=True,
        chains=4,               # Single chain per window
        n_jobs=4,               # Sequential to avoid GPU contention
        n_samples=500,          # Reduce for speed
        tune=500,
        progressbar=False,
    )
    
    # Pass config to extractor
    config = config or {}
    config['bayes_cfg'] = bayes_cfg
    config['n_jobs'] = 4  # Force sequential execution
    
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
    print(f"   Trajectory R-squared: {results.get('trajectory_r_squared', np.nan):.4f}")
    print(f"   Trajectory MSE:       {results.get('trajectory_mse', np.nan):.4f}")
    print(f"   Trajectory MAE:       {results.get('trajectory_mae', np.nan):.4f}")
    print(f"   Slope Metrics:")
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

    if backend == "bayes":
        print(f"\n   Trajectory-type (Bayes) Metrics:")
        print(f"     - Accuracy:    {results.get('type_acc', float('nan')):.4f}")
        print(f"     - Log-loss:    {results.get('type_logloss', float('nan')):.4f}")
        print(f"     - Brier:       {results.get('type_brier', float('nan')):.4f}")
        print(f"     - Macro-F1:    {results.get('type_macro_f1', float('nan')):.4f}")
    
    # Pass/Fail criteria
    print(f"\n4. Validation Status:")
    
    checks = []
    
    if backend == "bayes":
        # For Bayesian backend, check trajectory-type classification accuracy
        if results.get('type_acc', 0) > 0.6:
            print(f"   ✓ Trajectory-type accuracy > 0.6")
            checks.append(True)
        else:
            print(f"   ✗ Trajectory-type accuracy ≤ 0.6")
            checks.append(False)
    else:
        if scenario == "nonlinear":
            # R-squared should be > 0.3
            if results.get('trajectory_r_squared', 0) > 0.3:
                print(f"   ✓ R-squared > 0.3")
                checks.append(True)
            else:
                print(f"   ✗ R-squared ≤ 0.3")
                checks.append(False)
        else:
            # Slope correlation should be > 0.5 (reasonable correlation)
            if results.get('slope_correlation', 0) > 0.5:
                print(f"   ✓ Slope correlation > 0.5")
                checks.append(True)
            else:
                print(f"   ✗ Slope correlation ≤ 0.5")
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
        print(f"\n\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}\n")
        
        summary_df = pd.DataFrame(results_summary)
        print(summary_df[['backend', 'scenario', 'slope_correlation', 'effect_recovery_rate']])
        
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