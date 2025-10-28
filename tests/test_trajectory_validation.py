import pytest
import numpy as np
import pandas as pd
from traj_ps.data.simulate import simulate_with_known_trajectories
from traj_ps.evaluation.trajectory_metrics import (
    compare_trajectories,
    compute_treatment_effect_recovery
)


def test_simulate_linear_decline():
    """Should generate linear decline trajectories with known parameters."""
    dyn, sta, truth = simulate_with_known_trajectories(
        n_pat=50,
        scenario="linear_decline",
        treatment_effect_on_slope=0.5,
        seed=42
    )
    
    assert truth['scenario'] == "linear_decline"
    assert truth['treatment_effect_on_slope'] == 0.5
    assert len(truth['true_slopes']) == 50
    assert len(truth['true_intercepts']) == 50
    assert 'eGFR' in truth['feature_params']


def test_simulate_nonlinear():
    """Should generate nonlinear (quadratic) trajectories."""
    dyn, sta, truth = simulate_with_known_trajectories(
        n_pat=30,
        scenario="nonlinear",
        treatment_effect_on_slope=0.3,
        seed=42
    )
    
    assert truth['scenario'] == "nonlinear"
    assert 'HbA1c' in truth['feature_params']
    assert truth['feature_params']['HbA1c']['model'] == 'quadratic'


def test_simulate_heterogeneous():
    """Should generate heterogeneous treatment effects."""
    dyn, sta, truth = simulate_with_known_trajectories(
        n_pat=40,
        scenario="heterogeneous",
        treatment_effect_on_slope=0.4,
        seed=42
    )
    
    assert truth['scenario'] == "heterogeneous"
    assert 'heterogeneity_factor' in truth['feature_params']['eGFR']


def test_compare_trajectories_perfect_prediction():
    """Perfect predictions should have zero error."""
    # Simulate data
    dyn, sta, truth = simulate_with_known_trajectories(
        n_pat=20, scenario="linear_decline", seed=42
    )
    
    # Create "perfect" predictions matching ground truth
    pred_features = pd.DataFrame({
        'pid': list(truth['true_slopes'].keys()),
        'eGFR_slope': list(truth['true_slopes'].values()),
        'eGFR_intercept': list(truth['true_intercepts'].values())
    })
    
    metrics = compare_trajectories(pred_features, truth, metric="mse")
    
    assert metrics['slope_mse'] < 1e-10  # Should be ~0
    assert metrics['intercept_mse'] < 1e-10


def test_compare_trajectories_noisy_prediction():
    """Noisy predictions should have non-zero but reasonable error."""
    dyn, sta, truth = simulate_with_known_trajectories(
        n_pat=50, scenario="linear_decline", seed=42
    )
    
    # Add noise to true values
    pred_features = pd.DataFrame({
        'pid': list(truth['true_slopes'].keys()),
        'eGFR_slope': [s + np.random.normal(0, 0.5) for s in truth['true_slopes'].values()],
        'eGFR_intercept': [i + np.random.normal(0, 2) for i in truth['true_intercepts'].values()]
    })
    
    metrics = compare_trajectories(pred_features, truth, metric="mse")
    
    assert metrics['slope_mse'] > 0
    assert metrics['slope_correlation'] > 0.6, f"Got {metrics['slope_correlation']} correlation"


def test_treatment_effect_recovery():
    """Should detect if treatment effect is recovered."""
    dyn, sta, truth = simulate_with_known_trajectories(
        n_pat=100,
        scenario="linear_decline",
        treatment_effect_on_slope=0.5,
        seed=42
    )
    
    # Create predictions with correct treatment effect
    pred_features = []
    for pid in sta['pid']:
        is_treated = sta.loc[sta['pid'] == pid, 'treatment'].iloc[0]
        true_slope = truth['true_slopes'][pid]
        
        pred_features.append({
            'pid': pid,
            'treatment': is_treated,
            'eGFR_slope': true_slope + np.random.normal(0, 0.1)
        })
    
    pred_df = pd.DataFrame(pred_features)
    
    effect_metrics = compute_treatment_effect_recovery(pred_df, truth)
    
    assert abs(effect_metrics['true_effect'] - 0.5) < 1e-6
    assert abs(effect_metrics['effect_bias']) < 0.2  # Allow some noise
    assert 0.8 < effect_metrics['effect_recovery_rate'] < 1.2


def test_trajectory_metrics_missing_columns():
    """Should handle missing columns gracefully."""
    truth = {
        'true_slopes': {'P001': -2.0},
        'true_intercepts': {'P001': 60.0}
    }
    
    # Features missing slope column
    pred_features = pd.DataFrame({
        'pid': ['P001'],
        'some_other_feature': [1.0]
    })
    
    metrics = compare_trajectories(pred_features, truth)
    
    assert np.isnan(metrics['slope_mse'])
    assert np.isnan(metrics['intercept_mse'])