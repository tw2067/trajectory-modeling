import pytest
import numpy as np
import pandas as pd
from traj_ps.data.simulate import simulate_dynamic_static
from traj_ps.data.aggregate import DEFAULT_BIN_W
from traj_ps.data.counting import aggregate_covariates_to_bins

def test_aggregate_basic_output():
    """aggregate_covariates_to_bins returns agg_df and bin_map."""
    dyn, sta = simulate_dynamic_static(n_pat=10, seed=920)
    agg_df, bin_map = aggregate_covariates_to_bins(dyn, sta, agg_features=["SBP"], bin_width=DEFAULT_BIN_W)
    
    assert isinstance(agg_df, pd.DataFrame)
    assert isinstance(bin_map, dict), "bin_map should be dict of {pid: array of bin edges}"
    assert "pid" in agg_df.columns
    assert "start" in agg_df.columns
    assert "stop" in agg_df.columns

def test_aggregate_bin_alignment():
    """All aggregated records should align to valid bins."""
    dyn, sta = simulate_dynamic_static(n_pat=15, seed=920)
    agg_df, bin_map = aggregate_covariates_to_bins(dyn, sta, agg_features=["SBP", "MedA"], bin_width=1.0)
    
    # Check every patient has bin edges
    for pid in agg_df["pid"].unique():
        assert pid in bin_map, f"Missing bin_map for {pid}"
        assert len(bin_map[pid]) > 0, f"Empty bin array for {pid}"

def test_aggregate_bin_coverage():
    """bin_map should cover [0, Tmax) for each patient."""
    dyn, sta = simulate_dynamic_static(n_pat=20, seed=920)
    _, bin_map = aggregate_covariates_to_bins(dyn, sta, agg_features=["eGFR"], bin_width=0.5)
    
    for pid in sta["pid"]:
        patient_bins = bin_map[pid]
        patient_tmax = sta.loc[sta["pid"] == pid, "Tmax"].iloc[0]
        
        # First bin should start at 0
        assert patient_bins[0] == 0.0
        # Last bin edge should be close to Tmax (within one bin width)
        assert patient_bins[-1] >= patient_tmax - 0.5, \
        f"Patient {pid}: last bin {patient_bins[-1]} too far from Tmax {patient_tmax}"

def test_aggregate_empty_features():
    """Empty agg_features list should return minimal agg_df."""
    dyn, sta = simulate_dynamic_static(n_pat=5, seed=920)
    agg_df, bin_map = aggregate_covariates_to_bins(dyn, sta, agg_features=[], bin_width=1.0)
    
    # Should still have bin_map, but agg_df may have only pid/bin_id
    assert len(bin_map) > 0, "bin_map dict should have entries"
    assert {"pid", "start", "stop"}.issubset(agg_df.columns)

def test_aggregate_missing_feature():
    """Requesting non-existent feature should handle gracefully or raise clear error."""
    dyn, sta = simulate_dynamic_static(n_pat=5, seed=920)
    
    # Your implementation includes missing features as NaN columns
    agg_df, _ = aggregate_covariates_to_bins(dyn, sta, agg_features=["NonExistentFeature"], bin_width=1.0)
    # Feature column exists but should be all NaN
    assert "NonExistentFeature" in agg_df.columns
    assert agg_df["NonExistentFeature"].isna().all(), "Missing feature should be all NaN"

def test_aggregate_bin_width_consistency():
    """Bins should have consistent width (within floating point tolerance)."""
    dyn, sta = simulate_dynamic_static(n_pat=10, seed=920)
    bin_width = 0.25
    _, bin_map = aggregate_covariates_to_bins(dyn, sta, agg_features=["HbA1c"], bin_width=bin_width)

    # bin_map is dict[pid -> array of bin edges]
    for pid, bins in bin_map.items():
        if len(bins) > 1:
            widths = np.diff(bins)
            # All widths should equal bin_width
            assert np.allclose(widths, bin_width, atol=1e-9), f"Patient {pid} has inconsistent bin widths: {widths}"
    