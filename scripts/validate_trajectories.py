import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from traj_ps.evaluation.extractors import extract_trajectory_features
from traj_ps.config import DiseaseConfig
import traceback
import gc
import importlib

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


def clear_cache():
    """Clear Python module cache and garbage collect."""
    # Clear imported modules related to backends
    modules_to_reload = [
        'traj_ps.backends.bayes.model',
        'traj_ps.backends.bayes.pipeline',
        'traj_ps.backends.bayes.classify',
        'traj_ps.backends.deep.model',
        'traj_ps.backends.gam.model',
    ]
    
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
    
    # Force garbage collection
    gc.collect()
    
    # Clear JAX cache if using GPU
    try:
        import jax
        jax.clear_caches()
        print("  [Cache] Cleared JAX cache")
    except ImportError:
        pass
    
    # Clear PyMC cache
    try:
        import pymc as pm
        # PyMC doesn't have explicit cache clearing, but gc should help
        print("  [Cache] Garbage collected PyMC objects")
    except ImportError:
        pass
    
    print("  [Cache] Cleared module cache and collected garbage")


def validate_deep_backend(dyn_df, sta_df, ground_truth, config, disease_cfg):
    """
    Validate deep backend by:
    1. Training GRU-D model on trajectory data
    2. Extracting predictions from trained model
    3. Evaluating against ground truth
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from traj_ps.backends.deep.model import DeepPSDual
    from traj_ps.data.dual_prep import prepare_samples_dual
    from traj_ps.data.collate import dual_pad_collate
    
    primary_feature = disease_cfg.primary_feature.name
    
    print(f"\n{'='*80}")
    print(f"  [Deep] TRAINING AND VALIDATING DEEP BACKEND")
    print(f"{'='*80}\n")
    
    # ===== STEP 1: Prepare Training Data =====
    print(f"  [Deep] Step 1: Preparing data for {primary_feature}...")
    
    # Define features
    embed_feats = [primary_feature]  # Primary feature for embedding
    
    # Get secondary features for aggregation
    if disease_cfg.secondary_features:
        agg_feats = [f.name for f in disease_cfg.secondary_features[:3]]
    else:
        # Default aggregation features if none specified
        agg_feats = []
    
    print(f"  [Deep] Embed features: {embed_feats}")
    print(f"  [Deep] Aggregation features: {agg_feats}")
    
    # Prepare dual timeline samples
    bin_width = 1/12  # 1 month bins (standard for CKD/HIV)
    
    try:
        samples = prepare_samples_dual(
            dynamic=dyn_df,
            static=sta_df,
            embed_feats=embed_feats,
            agg_feats=agg_feats,
            bin_w=bin_width
        )
    except Exception as e:
        print(f"  [ERROR] Failed to prepare samples: {e}")
        import traceback
        traceback.print_exc()
        return {
            'slope_mse': np.nan,
            'slope_mae': np.nan,
            'slope_correlation': np.nan,
            'slope_rmse': np.nan,
            'n_slopes_compared': 0,
            'error': str(e)
        }
    
    print(f"  [Deep] Prepared {len(samples)} samples")
    
    if len(samples) == 0:
        print(f"  [ERROR] No samples prepared")
        return {
            'slope_mse': np.nan,
            'slope_mae': np.nan,
            'slope_correlation': np.nan,
            'slope_rmse': np.nan,
            'n_slopes_compared': 0,
            'error': 'No samples'
        }
    
    # Create train/val split
    train_size = int(0.8 * len(samples))
    val_size = len(samples) - train_size
    
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]
    
    train_loader = DataLoader(
        train_samples,
        batch_size=min(32, train_size),
        shuffle=True,
        collate_fn=dual_pad_collate
    )
    
    val_loader = DataLoader(
        val_samples,
        batch_size=min(32, val_size),
        shuffle=False,
        collate_fn=dual_pad_collate
    )
    
    print(f"  [Deep] Train: {train_size} samples, Val: {val_size} samples")
    
    # ===== STEP 2: Initialize Model =====
    print(f"\n  [Deep] Step 2: Initializing model...")
    
    # Get dimensions from first batch
    first_batch = next(iter(train_loader))
    p_seq = first_batch['X_raw'].shape[-1]      # Sequential feature dimension
    p_std = first_batch['STD_agg'].shape[-1]     # Aggregated feature dimension
    p_static = first_batch['Z'].shape[-1]        # Static feature dimension
    
    h_dim = 64  # Hidden dimension
    head_hidden = 32  # Head hidden dimension
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DeepPSDual(
        p_seq=p_seq,           
        p_std=p_std,          
        p_static=p_static,    
        h=h_dim,
        head_hidden=head_hidden
    ).to(device)
    
    print(f"  [Deep] Model: p_seq={p_seq}, p_std={p_std}, p_static={p_static}, h={h_dim}")
    print(f"  [Deep] Device: {device}")
    print(f"  [Deep] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ===== STEP 3: Train Model =====
    print(f"\n  [Deep] Step 3: Training model...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Simple reconstruction loss for validation
    n_epochs = 20
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Create a reconstruction layer (will be reused)
    recon_layer = nn.Linear(h_dim, p_seq).to(device)
    recon_optimizer = torch.optim.Adam(recon_layer.parameters(), lr=1e-3)
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        recon_layer.train()
        train_loss = 0.0
        n_train_batches = 0
        
        for batch in train_loader:
            X_raw = batch['X_raw'].to(device)
            M_raw = batch['M_raw'].to(device)
            DT_raw = batch['DT_raw'].to(device)
            STD_agg = batch['STD_agg'].to(device)
            Z = batch['Z'].to(device)
            idx_map = batch['idx_map']
            
            optimizer.zero_grad()
            recon_optimizer.zero_grad()
            
            # Forward pass - note: model returns (eta, H_raw, H_agg, mask_sel)
            eta, H_raw, H_agg, mask_sel = model(
                X_raw, M_raw, DT_raw, STD_agg, Z, idx_map
            )
            
            # Reconstruction loss: predict observed values
            # Use hidden states to reconstruct input
            recon = recon_layer(H_raw)  # (B, Tr, p_seq)
            
            # MSE on observed values only
            loss = ((recon - X_raw) ** 2 * M_raw).sum() / (M_raw.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            recon_optimizer.step()
            
            train_loss += loss.item()
            n_train_batches += 1
        
        avg_train_loss = train_loss / n_train_batches
        
        # Validation
        model.eval()
        recon_layer.eval()
        val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                X_raw = batch['X_raw'].to(device)
                M_raw = batch['M_raw'].to(device)
                DT_raw = batch['DT_raw'].to(device)
                STD_agg = batch['STD_agg'].to(device)
                Z = batch['Z'].to(device)
                idx_map = batch['idx_map']
                
                eta, H_raw, H_agg, mask_sel = model(
                    X_raw, M_raw, DT_raw, STD_agg, Z, idx_map
                )
                
                recon = recon_layer(H_raw)
                loss = ((recon - X_raw) ** 2 * M_raw).sum() / (M_raw.sum() + 1e-8)
                
                val_loss += loss.item()
                n_val_batches += 1
        
        avg_val_loss = val_loss / n_val_batches
        
        print(f"  [Deep] Epoch {epoch+1}/{n_epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model states
            best_model_state = model.state_dict()
            best_recon_state = recon_layer.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [Deep] Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    recon_layer.load_state_dict(best_recon_state)
    
    # ===== STEP 4: Extract Predictions =====
    print(f"\n  [Deep] Step 4: Extracting predictions...")
    
    model.eval()
    recon_layer.eval()
    
    all_predictions = []
    all_trajectories = []
    
    # Process all samples
    full_loader = DataLoader(samples, batch_size=32, collate_fn=dual_pad_collate, shuffle=False)
    
    with torch.no_grad():
        for batch in full_loader:
            X_raw = batch['X_raw'].to(device)
            M_raw = batch['M_raw'].to(device)
            DT_raw = batch['DT_raw'].to(device)
            STD_agg = batch['STD_agg'].to(device)
            Z = batch['Z'].to(device)
            idx_map = batch['idx_map']
            pids = batch['pid']
            raw_times = batch['raw_time']  # List of arrays (unpadded)
            
            # Forward pass
            eta, H_raw, H_agg, mask_sel = model(
                X_raw, M_raw, DT_raw, STD_agg, Z, idx_map
            )
            
            # Project to feature space for trajectory reconstruction
            pred_vals = recon_layer(H_raw)  # (B, Tr, p_seq)
            
            # Extract per-patient features
            for i, pid in enumerate(pids):
                # Get ORIGINAL unpadded length for this patient
                original_length = len(raw_times[i])
                
                # Get mask for UNPADDED portion only
                mask = M_raw[i, :original_length, 0].cpu().numpy().astype(bool)
                
                if mask.sum() >= 2:
                    # Get times for observed points (from original unpadded array)
                    times = raw_times[i][mask]  # ✓ Now both have same length
                    
                    # Get predicted values at observed times (from unpadded portion)
                    vals = pred_vals[i, :original_length, 0][mask].cpu().numpy()
                    
                    # Fit linear model to extract slope/intercept
                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression()
                    lr.fit(times.reshape(-1, 1), vals)
                    
                    slope = float(lr.coef_[0])
                    intercept = float(lr.intercept_)
                    
                    all_predictions.append({
                        'pid': pid,
                        f'{primary_feature}_slope': slope,
                        f'{primary_feature}_intercept': intercept
                    })
                    
                    # Store trajectory points
                    for t, v in zip(times, vals):
                        all_trajectories.append({
                            'pid': pid,
                            'time': float(t),
                            'predicted_value': float(v)
                        })
    
    pred_features = pd.DataFrame(all_predictions)
    pred_trajectories = pd.DataFrame(all_trajectories)
    
    print(f"  [Deep] Extracted features for {len(pred_features)} patients")
    print(f"  [Deep] Extracted {len(pred_trajectories)} trajectory points")
    
    if pred_features.empty:
        print(f"  [ERROR] No predictions extracted")
        return {
            'slope_mse': np.nan,
            'slope_mae': np.nan,
            'slope_correlation': np.nan,
            'slope_rmse': np.nan,
            'n_slopes_compared': 0,
            'error': 'No predictions'
        }
    
    # ===== STEP 5: Evaluate =====
    print(f"\n  [Deep] Step 5: Evaluating...")
    
    # Add detailed diagnostics
    print_detailed_diagnostics(pred_features, ground_truth, dyn_df, "deep")
    
    # Compute metrics
    metrics = compare_trajectories(
        pred_features, 
        ground_truth, 
        dynamic_df=dyn_df, 
        predicted_trajectories=pred_trajectories
    )
    
    # Add treatment effect recovery
    effect_metrics = compute_treatment_effect_recovery(
        pred_features.merge(sta_df[['pid', 'treatment']], on='pid', how='left'),
        ground_truth
    )
    
    # Merge metrics
    all_metrics = {**metrics, **effect_metrics}
    
    # Print results
    print(f"\n   Trajectory Reconstruction Metrics:")
    print(f"     - R-squared:       {all_metrics.get('trajectory_r_squared', np.nan):.4f}")
    print(f"     - MSE:             {all_metrics.get('trajectory_mse', np.nan):.4f}")
    print(f"     - MAE:             {all_metrics.get('trajectory_mae', np.nan):.4f}")
    
    print(f"\n   Slope Metrics:")
    print(f"     - MSE:             {all_metrics.get('slope_mse', np.nan):.4f}")
    print(f"     - MAE:             {all_metrics.get('slope_mae', np.nan):.4f}")
    print(f"     - RMSE:            {all_metrics.get('slope_rmse', np.nan):.4f}")
    print(f"     - Correlation:     {all_metrics.get('slope_correlation', np.nan):.4f}")
    print(f"     - N compared:      {all_metrics.get('n_slopes_compared', 0)}")
    
    print(f"\n   Intercept Metrics:")
    print(f"     - MSE:             {all_metrics.get('intercept_mse', np.nan):.4f}")
    print(f"     - MAE:             {all_metrics.get('intercept_mae', np.nan):.4f}")
    print(f"     - RMSE:            {all_metrics.get('intercept_rmse', np.nan):.4f}")
    print(f"     - Correlation:     {all_metrics.get('intercept_correlation', np.nan):.4f}")
    print(f"     - N compared:      {all_metrics.get('n_intercepts_compared', 0)}")
    
    print(f"\n   Treatment Effect Recovery:")
    print(f"     - True effect:     {all_metrics.get('true_effect', np.nan):.4f}")
    print(f"     - Estimated effect:{all_metrics.get('estimated_effect', np.nan):.4f}")
    print(f"     - Bias:            {all_metrics.get('effect_bias', np.nan):.4f}")
    print(f"     - Recovery rate:   {all_metrics.get('effect_recovery_rate', np.nan):.4f}")
    
    print(f"\n{'='*80}")
    print(f"  [Deep] VALIDATION COMPLETE")
    print(f"{'='*80}\n")
    
    return all_metrics


def validate_bayes_backend(dyn_df, sta_df, ground_truth, config, disease_cfg):
    """Validate Bayesian backend using trajectory classification."""
    print("  [Bayes] Extracting trajectory-type probabilities...")
    
    # Configure Bayes with settings
    from traj_ps.backends.bayes.model import BayesConfig
    import os
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    print(f"  [Bayes] Using CPU with n_jobs={n_cpus}")

    primary_feature = disease_cfg.primary_feature.name
    
    bayes_cfg = BayesConfig(
        sampler="pymc",
        use_gpu=False,
        chains=4,
        n_jobs=n_cpus,
        n_samples=400,
        tune=700,
        target_accept=0.99,
        progressbar=False,
    )
    
    # Pass config with disease_cfg embedded
    config = config or {}
    config['bayes_cfg'] = bayes_cfg
    
    try:
        pred_features, pred_trajectories = extract_trajectory_features(
            backend="bayes",
            dynamic_df=dyn_df,
            feature=primary_feature,
            config=config,
            disease_cfg=disease_cfg  # IMPORTANT: Pass disease config
        )

         # Add detailed diagnostics
        print_detailed_diagnostics(pred_features, ground_truth, dyn_df, "bayes")
        
        # Compute classification metrics only (no treatment effect recovery)
        metrics = compare_trajectories(
            pred_features, 
            ground_truth,
            dynamic_df=dyn_df,
            predicted_trajectories=pred_trajectories
        )
        
        # Print results
        print("\n   Classification Metrics:")
        print(f"     - Accuracy:    {metrics.get('type_acc', np.nan):.4f}")
        print(f"     - Log-loss:    {metrics.get('type_logloss', np.nan):.4f}")
        print(f"     - Brier Score: {metrics.get('type_brier', np.nan):.4f}")
        print(f"     - Macro F1:    {metrics.get('type_macro_f1', np.nan):.4f}")
        
        if metrics.get('type_confusion') is not None:
            print("\n   Confusion Matrix:")
            print(metrics['type_confusion'])

        return metrics

    except Exception as e:
        print(f"  [ERROR] Exception during Bayes validation: {e}")
        traceback.print_exc()
        return {}
    


def validate_gam_backend(dyn_df, sta_df, ground_truth, config, disease_cfg):
    """Validate GAM backend using per-window spline features."""
    primary_feature = disease_cfg.primary_feature.name
    
    print(f"  [GAM] Extracting trajectory features for {primary_feature}...")
    
    # Use the extractor dispatcher
    pred_features, pred_trajectories = extract_trajectory_features(
        backend="gam",
        dynamic_df=dyn_df,
        feature=primary_feature,
        config=config,
        disease_cfg=disease_cfg
    )
    
    if pred_features.empty:
        print("  [ERROR] No features extracted from GAM backend")
        return {
            'slope_mse': np.nan,
            'slope_mae': np.nan,
            'slope_correlation': np.nan,
            'slope_rmse': np.nan,
            'n_slopes_compared': 0
        }

    # Add detailed diagnostics
    print_detailed_diagnostics(pred_features, ground_truth, dyn_df, "gam")

    metrics = compare_trajectories(
        pred_features, ground_truth,
        dynamic_df=dyn_df,
        predicted_trajectories=pred_trajectories
    )
    effect_metrics = compute_treatment_effect_recovery(
        pred_features.merge(sta_df[['pid', 'treatment']], on='pid'),
        ground_truth
    )
    
    # Merge metrics
    all_metrics = {**metrics, **effect_metrics}
    
    # Print results
    print("\n   Trajectory Reconstruction Metrics:")
    print(f"     - R-squared:       {all_metrics.get('trajectory_r_squared', np.nan):.4f}")
    print(f"     - MSE:             {all_metrics.get('trajectory_mse', np.nan):.4f}")
    print(f"     - MAE:             {all_metrics.get('trajectory_mae', np.nan):.4f}")
    print(f"     - RMSE:            {all_metrics.get('trajectory_rmse', np.nan):.4f}")
    
    print(f"\n   Slope Metrics:")
    print(f"     - MSE:             {all_metrics.get('slope_mse', np.nan):.4f}")
    print(f"     - MAE:             {all_metrics.get('slope_mae', np.nan):.4f}")
    print(f"     - RMSE:            {all_metrics.get('slope_rmse', np.nan):.4f}")
    print(f"     - Correlation:     {all_metrics.get('slope_correlation', np.nan):.4f}")
    print(f"     - N compared:      {all_metrics.get('n_slopes_compared', 0)}")
    
    print(f"\n   Intercept Metrics:")
    print(f"     - MSE:             {all_metrics.get('intercept_mse', np.nan):.4f}")
    print(f"     - MAE:             {all_metrics.get('intercept_mae', np.nan):.4f}")
    print(f"     - RMSE:            {all_metrics.get('intercept_rmse', np.nan):.4f}")
    print(f"     - Correlation:     {all_metrics.get('intercept_correlation', np.nan):.4f}")
    print(f"     - N compared:      {all_metrics.get('n_intercepts_compared', 0)}")
    
    print(f"\n   Treatment Effect Recovery:")
    print(f"     - True effect:     {all_metrics.get('true_effect', np.nan):.4f}")
    print(f"     - Estimated effect:{all_metrics.get('estimated_effect', np.nan):.4f}")
    print(f"     - Bias:            {all_metrics.get('effect_bias', np.nan):.4f}")
    print(f"     - Absolute bias:   {all_metrics.get('effect_abs_bias', np.nan):.4f}")
    print(f"     - Recovery rate:   {all_metrics.get('effect_recovery_rate', np.nan):.4f}")
    
    return all_metrics

def run_validation(backend: str, scenario: str, disease_cfg: DiseaseConfig, n_patients: int = 200, seed: int = 920):
    """
    Run validation for one backend on one scenario.
    
    Parameters
    ----------
    backend : str
        'deep', 'bayes', or 'gam'
    scenario : str
        'linear_decline', 'nonlinear', 'nonprogression', 'heterogeneous', or 'mixed'
    disease_cfg : DiseaseConfig
        Disease-specific configuration
    n_patients : int
        Number of patients to simulate
    seed : int
        Random seed
    
    Returns
    -------
    results : dict
        Validation metrics
    """
    clear_cache()

    print(f"\n{'='*70}")
    print(f"Validating: {backend.upper()} on {scenario} ({disease_cfg.name.upper()})")
    print(f"{'='*70}")
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / f"{backend}.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Set treatment effect based on scenario
    # Heterogeneous needs large effect for responders to cross thresholds
    # Other scenarios use moderate effects
    if scenario == "heterogeneous":
        # Strong effect: allows 30% of treated to reach nonprogression
        # For decreasing features (eGFR, MMSE): need to add ~2 units to slope
        # For increasing features (CD4): need to reduce worsening by ~2 units
        treatment_effect = 2.0
        print(f"   Using LARGE treatment effect ({treatment_effect}) for heterogeneous responses")
    elif scenario == "mixed":
        # Moderate effect for mixed populations
        treatment_effect = 0.8
        print(f"   Using MODERATE treatment effect ({treatment_effect}) for mixed populations")
    elif scenario == "nonprogression":
        # Small effect - patients already stable
        treatment_effect = 0.3
        print(f"   Using SMALL treatment effect ({treatment_effect}) for stable patients")
    else:
        # Default for linear_decline and nonlinear
        treatment_effect = 0.5
        print(f"   Using DEFAULT treatment effect ({treatment_effect})")
    
    # Simulate data with known trajectories
    print(f"\n1. Simulating {n_patients} patients with {scenario} pattern...")
    dyn_df, sta_df, ground_truth = simulate_with_known_trajectories(
        n_pat=n_patients,
        scenario=scenario,
        treatment_effect_on_slope=treatment_effect,
        disease=disease_cfg.name,
        disease_cfg=disease_cfg,
        seed=seed
    )
    
    print(f"   - Dynamic observations: {len(dyn_df)}")
    print(f"   - Features: {dyn_df['feature_name'].unique().tolist()}")
    print(f"   - Treatment effect on slope: {ground_truth['treatment_effect_on_slope']}")
    
    # Validate backend
    print(f"\n2. Validating {backend} backend...")
    
    try:
        if backend == "deep":
            results = validate_deep_backend(dyn_df, sta_df, ground_truth, config, disease_cfg)
        elif backend == "bayes":
            results = validate_bayes_backend(dyn_df, sta_df, ground_truth, config, disease_cfg)
        elif backend == "gam":
            results = validate_gam_backend(dyn_df, sta_df, ground_truth, config, disease_cfg)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    except Exception as e:
        print(f"   Error in {backend} backend: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Pass/Fail criteria - backend and scenario specific
    print(f"\n3. Validation Status:")
    
    checks = []
    
    if backend == "bayes":
        # For Bayesian backend, check trajectory-type classification accuracy
        type_acc = results.get('type_acc', 0)
        
        # Adjust threshold based on scenario complexity
        if scenario == "mixed":
            # 3-class problem with 30/50/20 split
            min_acc = 0.40  # Better than random (33%) but not too strict
            print(f"   Target: Accuracy > {min_acc} (mixed 3-class problem)")
        elif scenario == "heterogeneous":
            # Harder: heterogeneous responses create overlapping distributions
            min_acc = 0.35  # Slightly better than random
            print(f"   Target: Accuracy > {min_acc} (heterogeneous responses)")
        elif scenario == "nonprogression":
            # Easier: all patients stable
            min_acc = 0.60  # Should be high
            print(f"   Target: Accuracy > {min_acc} (homogeneous stable)")
        elif scenario == "linear_decline":
            # Easier: all patients declining
            min_acc = 0.60
            print(f"   Target: Accuracy > {min_acc} (homogeneous decline)")
        else:  # nonlinear
            min_acc = 0.50
            print(f"   Target: Accuracy > {min_acc} (nonlinear patterns)")
        
        if type_acc > min_acc:
            print(f"   ✓ Trajectory-type accuracy > {min_acc:.2f} ({type_acc:.3f})")
            checks.append(True)
        else:
            print(f"   ✗ Trajectory-type accuracy ≤ {min_acc:.2f} ({type_acc:.3f})")
            checks.append(False)
        
        # Check log-loss is reasonable (not worse than uniform guessing)
        logloss = results.get('type_logloss', float('inf'))
        n_classes = len(results.get('type_confusion', [[]])) if results.get('type_confusion') is not None else 3
        uniform_logloss = -np.log(1/n_classes)  # e.g., -log(1/3) ≈ 1.099
        
        if logloss < uniform_logloss * 1.2:  # Allow 20% slack
            print(f"   ✓ Log-loss < {uniform_logloss*1.2:.3f} (better than uniform, actual: {logloss:.3f})")
            checks.append(True)
        else:
            print(f"   ✗ Log-loss ≥ {uniform_logloss*1.2:.3f} (actual: {logloss:.3f})")
            checks.append(False)
    else:
        # Deep/GAM: check slope/intercept recovery
        if scenario in ["nonlinear", "mixed"]:
            # For complex patterns, check overall trajectory fit
            r2 = results.get('trajectory_r_squared', 0)
            min_r2 = 0.3 if scenario == "nonlinear" else 0.25
            
            if r2 > min_r2:
                print(f"   ✓ R-squared > {min_r2} ({r2:.3f})")
                checks.append(True)
            else:
                print(f"   ✗ R-squared ≤ {min_r2} ({r2:.3f})")
                checks.append(False)
        else:
            # For linear patterns, check slope correlation
            slope_corr = results.get('slope_correlation', 0)
            
            # Adjust threshold based on scenario
            if scenario == "heterogeneous":
                min_corr = 0.3  # Lower: heterogeneous responses are harder
                print(f"   Target: Slope correlation > {min_corr} (heterogeneous)")
            elif scenario == "nonprogression":
                min_corr = 0.2  # Very low variance makes correlation unreliable
                print(f"   Target: Slope correlation > {min_corr} (low variance)")
            else:
                min_corr = 0.5  # Standard threshold
                print(f"   Target: Slope correlation > {min_corr}")
            
            if np.isfinite(slope_corr) and slope_corr > min_corr:
                print(f"   ✓ Slope correlation > {min_corr} ({slope_corr:.3f})")
                checks.append(True)
            else:
                corr_str = f"{slope_corr:.3f}" if np.isfinite(slope_corr) else "N/A"
                print(f"   ✗ Slope correlation ≤ {min_corr} ({corr_str})")
                checks.append(False)
             
            # Treatment effect recovery - more lenient for heterogeneous
            recovery_rate = results.get('effect_recovery_rate', 0)
            
            if scenario == "heterogeneous":
                # Heterogeneous: expect different effect sizes per subgroup
                # Just check we recovered SOME effect in right direction
                est_effect = results.get('estimated_effect', 0)
                true_effect = results.get('true_effect', 0)
                
                if np.sign(est_effect) == np.sign(true_effect) and abs(est_effect) > 0.1:
                    print(f"   ✓ Treatment effect direction correct (est: {est_effect:.3f}, true: {true_effect:.3f})")
                    checks.append(True)
                else:
                    print(f"   ✗ Treatment effect direction wrong (est: {est_effect:.3f}, true: {true_effect:.3f})")
                    checks.append(False)
            else:
                # Standard scenarios: expect recovery rate in reasonable range
                if 0.4 <= recovery_rate <= 1.6:  # More lenient: 40-160%
                    print(f"   ✓ Treatment effect recovered (rate: {recovery_rate:.2f})")
                    checks.append(True)
                else:
                    print(f"   ✗ Treatment effect not recovered (rate: {recovery_rate:.2f})")
                    checks.append(False)
    
    # Overall pass/fail
    if all(checks):
        print(f"\n   ✅ PASSED: {backend} on {scenario}")
        results['passed'] = True
    elif len(checks) > 0 and sum(checks) >= len(checks) * 0.5:  # At least 50% of checks passed
        print(f"\n   ⚠️  PARTIAL PASS: {backend} on {scenario} ({sum(checks)}/{len(checks)} checks)")
        results['passed'] = False
        results['partial_pass'] = True
    else:
        print(f"\n   ❌ FAILED: {backend} on {scenario} ({sum(checks)}/{len(checks)} checks)")
        results['passed'] = False
    
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


def print_detailed_diagnostics(pred_features, ground_truth, dyn_df, backend):
    """Print detailed diagnostics about predictions and ground truth."""
    print(f"\n{'='*70}")
    print(f"DETAILED DIAGNOSTICS - {backend.upper()}")
    print(f"{'='*70}")
    
    primary_feature = ground_truth.get('primary_feature', 'eGFR')
    
    # 1. Ground truth statistics
    print(f"\n1. Ground Truth Statistics:")
    true_slopes = ground_truth['true_slopes']
    true_intercepts = ground_truth['true_intercepts']
    
    # Extract numeric slopes (handle dict/nested cases)
    numeric_slopes = []
    for pid, slope in true_slopes.items():
        if isinstance(slope, dict):
            numeric_slopes.append(slope.get('mean', slope.get('linear', slope.get('b1', 0))))
        else:
            numeric_slopes.append(slope)
    
    numeric_intercepts = list(true_intercepts.values())
    
    print(f"   True Slopes:")
    print(f"     - Count:  {len(numeric_slopes)}")
    print(f"     - Mean:   {np.mean(numeric_slopes):.4f}")
    print(f"     - Std:    {np.std(numeric_slopes):.4f}")
    print(f"     - Min:    {np.min(numeric_slopes):.4f}")
    print(f"     - Max:    {np.max(numeric_slopes):.4f}")
    print(f"     - Median: {np.median(numeric_slopes):.4f}")
    
    print(f"\n   True Intercepts:")
    print(f"     - Count:  {len(numeric_intercepts)}")
    print(f"     - Mean:   {np.mean(numeric_intercepts):.4f}")
    print(f"     - Std:    {np.std(numeric_intercepts):.4f}")
    print(f"     - Min:    {np.min(numeric_intercepts):.4f}")
    print(f"     - Max:    {np.max(numeric_intercepts):.4f}")
    
    # 2. Trajectory type distribution
    if 'trajectory_types' in ground_truth:
        traj_types = ground_truth['trajectory_types']
        type_counts = pd.Series(list(traj_types.values())).value_counts()
        print(f"\n   Trajectory Type Distribution:")
        for ttype, count in type_counts.items():
            print(f"     - {ttype:30s}: {count:3d} ({100*count/len(traj_types):.1f}%)")
    
    # 3. Predicted features statistics
    print(f"\n2. Predicted Features Statistics:")
    
    if backend == "bayes":
        # Bayes returns trajectory probabilities
        prob_cols = [c for c in pred_features.columns if c.endswith('_prob')]
        print(f"   Trajectory Probability Columns: {prob_cols}")
        
        for col in prob_cols:
            values = pred_features[col].values
            print(f"\n   {col}:")
            print(f"     - Mean:   {np.mean(values):.4f}")
            print(f"     - Std:    {np.std(values):.4f}")
            print(f"     - Min:    {np.min(values):.4f}")
            print(f"     - Max:    {np.max(values):.4f}")
        
        # Predicted trajectory types (argmax of probabilities)
        if prob_cols:
            pred_types = pred_features[prob_cols].idxmax(axis=1).str.replace('trajtype_', '').str.replace('_prob', '')
            pred_type_counts = pred_types.value_counts()
            print(f"\n   Predicted Trajectory Type Distribution:")
            for ttype, count in pred_type_counts.items():
                print(f"     - {ttype:30s}: {count:3d} ({100*count/len(pred_types):.1f}%)")
    else:
        # Deep/GAM return slopes and intercepts
        slope_col = f'{primary_feature}_slope' if f'{primary_feature}_slope' in pred_features.columns else 'slope'
        intercept_col = f'{primary_feature}_intercept' if f'{primary_feature}_intercept' in pred_features.columns else 'intercept'
        
        if slope_col in pred_features.columns:
            pred_slopes = pred_features[slope_col].values
            print(f"   Predicted Slopes (column: {slope_col}):")
            print(f"     - Count:    {len(pred_slopes)}")
            print(f"     - Mean:     {np.mean(pred_slopes):.4f}")
            print(f"     - Std:      {np.std(pred_slopes):.4f}")
            print(f"     - Min:      {np.min(pred_slopes):.4f}")
            print(f"     - Max:      {np.max(pred_slopes):.4f}")
            print(f"     - Median:   {np.median(pred_slopes):.4f}")
            print(f"     - Variance: {np.var(pred_slopes):.6f}")
            
            # Check if predictions are constant
            if np.std(pred_slopes) < 1e-6:
                print(f"     ⚠️  WARNING: Predicted slopes are nearly constant!")
                print(f"     ⚠️  All predictions ≈ {pred_slopes[0]:.6f}")
        
        if intercept_col in pred_features.columns:
            pred_intercepts = pred_features[intercept_col].values
            print(f"\n   Predicted Intercepts (column: {intercept_col}):")
            print(f"     - Count:    {len(pred_intercepts)}")
            print(f"     - Mean:     {np.mean(pred_intercepts):.4f}")
            print(f"     - Std:      {np.std(pred_intercepts):.4f}")
            print(f"     - Min:      {np.min(pred_intercepts):.4f}")
            print(f"     - Max:      {np.max(pred_intercepts):.4f}")
    
    # 4. Data availability
    print(f"\n3. Data Availability:")
    print(f"   Dynamic data shape: {dyn_df.shape}")
    print(f"   Features in dynamic data: {dyn_df['feature_name'].unique()}")
    
    primary_data = dyn_df[dyn_df['feature_name'] == primary_feature]
    print(f"   Primary feature ({primary_feature}) observations: {len(primary_data)}")
    
    obs_per_patient = primary_data.groupby('pid').size()
    print(f"   Observations per patient:")
    print(f"     - Mean:   {obs_per_patient.mean():.2f}")
    print(f"     - Median: {obs_per_patient.median():.2f}")
    print(f"     - Min:    {obs_per_patient.min()}")
    print(f"     - Max:    {obs_per_patient.max()}")
    
    # 5. Sample predictions vs truth
    print(f"\n4. Sample Predictions (first 10 patients):")
    
    if backend == "bayes":
        # Show top predicted type and probabilities
        for pid in list(pred_features['pid'].head(10)):
            if pid in traj_types:
                true_type = traj_types[pid]
                pred_row = pred_features[pred_features['pid'] == pid]
                
                if not pred_row.empty and prob_cols:
                    probs = pred_row[prob_cols].values[0]
                    pred_type_idx = np.argmax(probs)
                    pred_type = prob_cols[pred_type_idx].replace('trajtype_', '').replace('_prob', '')
                    max_prob = probs[pred_type_idx]
                    
                    match = "✓" if pred_type == true_type else "✗"
                    print(f"   {pid}: true={true_type:20s}, pred={pred_type:20s} (p={max_prob:.3f}) {match}")
    else:
        # Show slope/intercept comparison
        for pid in list(pred_features['pid'].head(10)):
            if pid in true_slopes and pid in true_intercepts:
                pred_row = pred_features[pred_features['pid'] == pid]
                
                if not pred_row.empty:
                    true_slope = true_slopes[pid]
                    if isinstance(true_slope, dict):
                        true_slope = true_slope.get('mean', true_slope.get('linear', true_slope.get('b1', 0)))
                    
                    true_int = true_intercepts[pid]
                    
                    pred_slope = pred_row[slope_col].values[0] if slope_col in pred_row.columns else np.nan
                    pred_int = pred_row[intercept_col].values[0] if intercept_col in pred_row.columns else np.nan
                    
                    print(f"   {pid}: slope: true={true_slope:7.3f}, pred={pred_slope:7.3f} | "
                          f"intercept: true={true_int:7.2f}, pred={pred_int:7.2f}")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Validate trajectory extraction backends")
    parser.add_argument(
        "--scenario",
        choices=["linear_decline", "nonlinear", "heterogeneous", "nonprogression", "mixed", "all"],
        default="linear_decline",
        help="Trajectory scenario to test"
    )

    parser.add_argument('--backend', type=str, default='all',
                       choices=['all', 'bayes', 'gam', 'deep'],
                       help='Which backend to test')
    

    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all backends on all scenarios"
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

    parser.add_argument('--disease', type=str, default='all',
                       choices=['all', 'ckd', 'hiv', 'alzheimers', 'parkinsons'],
                       help='Which disease to test')
    
    args = parser.parse_args()

    if args.all or args.scenario == 'all':
        test_scenarios = ['linear_decline', 'nonlinear', 'nonprogression', 'heterogeneous', 'mixed']
    else:
        test_scenarios = [args.scenario]

    test_configs = {
        'ckd': (DiseaseConfig.for_ckd(), test_scenarios),
        'hiv': (DiseaseConfig.for_hiv(), test_scenarios),
        'alzheimers': (DiseaseConfig.for_alzheimers(), test_scenarios),
        'parkinsons': (DiseaseConfig.for_parkinsons(), test_scenarios),
    }

    # Filter by disease
    if args.disease != 'all':
        test_configs = {args.disease: test_configs[args.disease]}
    
    # Run tests
    results = {}
    
    for disease_name, (disease_cfg, scenarios) in test_configs.items():
        print(f"\n{'='*80}")
        print(f"Testing {disease_name.upper()}")
        print(f"  Primary feature: {disease_cfg.primary_feature.name}")
        print(f"  Direction: {disease_cfg.primary_feature.direction}")
        print(f"  Higher better: {disease_cfg.primary_feature.higher_better}")
        print(f"  Trajectory types: {disease_cfg.trajectory_types}")
        print(f"{'='*80}")
        
        for scenario in scenarios:
            print(f"\n{'-'*80}")
            print(f"Scenario: {scenario}")
            print(f"{'-'*80}")
            
            # Generate data
            dyn_df, sta_df, ground_truth = simulate_with_known_trajectories(
                n_pat=60,
                scenario=scenario,
                disease=disease_name,
                disease_cfg=disease_cfg,
                seed=920
            )
            
            # Test backends
            if args.backend == 'all' or args.backend == 'bayes':
                results[f'{disease_name}_{scenario}_bayes'] = validate_bayes_backend(
                    dyn_df, sta_df, ground_truth, {}, disease_cfg
                )
            
            if args.backend == 'all' or args.backend == 'gam':
                results[f'{disease_name}_{scenario}_gam'] = validate_gam_backend(
                    dyn_df, sta_df, ground_truth, {}, disease_cfg
                )
            
            if args.backend == 'all' or args.backend == 'deep':
                results[f'{disease_name}_{scenario}_deep'] = validate_deep_backend(
                    dyn_df, sta_df, ground_truth, {}, disease_cfg
                )
    
    # Print summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result.get('passed', False) else "✗ FAILED"
        print(f"{test_name:40s} {status}")
    
    # Overall pass rate
    total = len(results)
    passed = sum(1 for r in results.values() if r.get('passed', False))
    print(f"\nOverall: {passed}/{total} tests passed ({100*passed/total:.1f}%)")


if __name__ == "__main__":
    main()