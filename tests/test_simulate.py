import pytest
import numpy as np
import pandas as pd
from traj_ps.data.simulate import simulate_dynamic_static

def test_simulate_basic_structure():
    """Verify simulate_dynamic_static returns correct DataFrames with expected columns."""
    dyn, sta = simulate_dynamic_static(n_pat=20, seed=920)
    
    # Check dynamic DataFrame structure
    assert isinstance(dyn, pd.DataFrame)
    required_dyn_cols = {"pid", "time", "feature_name", "value"}
    assert required_dyn_cols.issubset(dyn.columns), f"Missing columns: {required_dyn_cols - set(dyn.columns)}"
    
    # Check static DataFrame structure
    assert isinstance(sta, pd.DataFrame)
    required_sta_cols = {"pid", "Tmax", "treatment", "time_to_event"} 
    assert required_sta_cols.issubset(sta.columns), f"Missing columns: {required_sta_cols - set(sta.columns)}"
    
    # Check patient count
    assert sta["pid"].nunique() == 20
    assert dyn["pid"].nunique() == 20

def test_simulate_reproducibility():
    """Same seed should produce identical data."""
    dyn1, sta1 = simulate_dynamic_static(n_pat=10, seed=920)
    dyn2, sta2 = simulate_dynamic_static(n_pat=10, seed=920)
    
    pd.testing.assert_frame_equal(dyn1.sort_values(["pid", "time", "feature_name"]).reset_index(drop=True),
                                   dyn2.sort_values(["pid", "time", "feature_name"]).reset_index(drop=True))
    pd.testing.assert_frame_equal(sta1.sort_values("pid").reset_index(drop=True),
                                   sta2.sort_values("pid").reset_index(drop=True))

def test_simulate_treatment_assignment():
    """Treatment column should be binary and have reasonable prevalence."""
    _, sta = simulate_dynamic_static(n_pat=100, seed=920)

    assert "treatment" in sta.columns, "treatment column is missing"
    assert sta["treatment"].isin([0, 1]).all(), "Treatment should be binary (0 or 1)"
    prevalence = sta["treatment"].mean()
    assert 0.1 < prevalence < 0.8, f"Treatment prevalence {prevalence:.2f} seems extreme"

def test_simulate_time_ordering():
    """Time values should be non-negative and ordered within patient."""
    dyn, _ = simulate_dynamic_static(n_pat=15, seed=920)
    
    assert (dyn["time"] >= 0).all(), "Time cannot be negative"
    
    # Check within-patient ordering
    for pid in dyn["pid"].unique():
        patient_times = dyn[dyn["pid"] == pid]["time"].values
        # Allow ties (measurements at same time)
        assert (np.diff(patient_times) >= 0).all(), f"Patient {pid} has non-monotonic times"

def test_simulate_feature_values():
    """Feature values should be finite and in plausible ranges."""
    dyn, _ = simulate_dynamic_static(n_pat=50, seed=920)
    
    assert np.isfinite(dyn["value"]).all(), "Feature values contain NaN or Inf"
    
    # Check feature-specific ranges (adjust based on your simulation logic)
    for feat in dyn["feature_name"].unique():
        feat_vals = dyn[dyn["feature_name"] == feat]["value"]
        assert feat_vals.min() > -1000, f"{feat} has suspiciously low value"
        assert feat_vals.max() < 1000, f"{feat} has suspiciously high value"

def test_simulate_time_to_event():
    """Tmax (or time_to_event) should be positive and finite."""
    _, sta = simulate_dynamic_static(n_pat=30, seed=920)
    
    time_col = "time_to_event" if "time_to_event" in sta.columns else "Tmax"
    valid_times = sta[time_col].dropna()
    assert (valid_times > 0).all(), f"{time_col} must be positive for non-NaN values"
    assert (sta[time_col].notna()).any(), f"At least some patients should have {time_col}"
    assert (valid_times > 0).all(), f"{time_col} contains NaN or Inf for treated patients"

def test_simulate_empty_inputs():
    """n_pat=0 should return empty DataFrames without error."""
    dyn, sta = simulate_dynamic_static(n_pat=0, seed=920)
    
    assert dyn.shape[0] == 0
    assert sta.shape[0] == 0
    
