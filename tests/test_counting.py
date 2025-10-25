import pytest
import numpy as np
import pandas as pd
from traj_ps.data.simulate import simulate_dynamic_static
from traj_ps.data.aggregate import DEFAULT_BIN_W
from traj_ps.data.counting import build_counting_process, aggregate_covariates_to_bins

def test_counting_basic_structure():
    """build_counting_process returns DataFrame with required columns."""
    dyn, sta = simulate_dynamic_static(n_pat=10, seed=920)
    agg_df, bin_map = aggregate_covariates_to_bins(dyn, sta, agg_features=["SBP"], bin_width=DEFAULT_BIN_W)
    
    counting = build_counting_process(static_df=sta, agg_df=agg_df, traj_aligned_df=pd.DataFrame(), id_col="pid")
    
    required_cols = {"pid", "start", "stop", "treatment"}
    assert required_cols.issubset(counting.columns), f"Missing: {required_cols - set(counting.columns)}"

def test_counting_time_intervals():
    """start < stop for all rows, and intervals should be contiguous per patient."""
    dyn, sta = simulate_dynamic_static(n_pat=15, seed=920)
    agg_df, _ = aggregate_covariates_to_bins(dyn, sta, agg_features=["eGFR"], bin_width=0.5)
    counting = build_counting_process(static_df=sta, agg_df=agg_df, traj_aligned_df=pd.DataFrame(), id_col="pid")
    
    assert (counting["start"] < counting["stop"]).all(), "start must be < stop"
    
    # Check contiguity: next interval start == previous interval stop
    for pid in counting["pid"].unique():
        patient_rows = counting[counting["pid"] == pid].sort_values("start")
        starts = patient_rows["start"].values
        stops = patient_rows["stop"].values
        if len(starts) > 1:
            assert np.allclose(starts[1:], stops[:-1], atol=1e-9), f"Non-contiguous intervals for pid={pid}"

def test_counting_patient_coverage():
    """Each patient should have intervals covering [0, time_to_event)."""
    dyn, sta = simulate_dynamic_static(n_pat=20, seed=920)
    agg_df, _ = aggregate_covariates_to_bins(dyn, sta, agg_features=["HbA1c"], bin_width=1.0)
    counting = build_counting_process(static_df=sta, agg_df=agg_df, traj_aligned_df=pd.DataFrame(), id_col="pid")
    
    for pid in sta["pid"]:
        if sta.loc[sta["pid"] == pid, "time_to_event"].isna().all():
            time_col = "Tmax"
        else:
            time_col = "time_to_event"
        patient_counting = counting[counting["pid"] == pid].sort_values("start")
        tte = sta.loc[sta["pid"] == pid, time_col].iloc[0]
        
        assert patient_counting["start"].iloc[0] == 0.0, f"Patient {pid} doesn't start at 0"
        assert patient_counting["stop"].iloc[-1] >= tte, f"Patient {pid} coverage ends before time_to_event"

def test_counting_treatment_column():
    """treatment column should be binary and match static_df."""
    dyn, sta = simulate_dynamic_static(n_pat=10, seed=920)
    agg_df, _ = aggregate_covariates_to_bins(dyn, sta, agg_features=["SBP"], bin_width=DEFAULT_BIN_W)
    counting = build_counting_process(static_df=sta, agg_df=agg_df, traj_aligned_df=pd.DataFrame(), id_col="pid")
    
    assert counting["treatment"].isin([0, 1]).all()
    
    # Treatment column should be time-varying:
    # For treated patients (static treatment=1), some bins are 0 (before treatment), some are 1 (after)
    # For untreated patients (static treatment=0), all bins should be 0
    for pid in counting["pid"].unique():
        patient_rows = counting[counting["pid"] == pid]
        static_trt = sta.loc[sta["pid"] == pid, "treatment"].iloc[0]

        if static_trt == 0:
            # Untreated patients should have treatment=0 in all intervals
            assert (patient_rows["treatment"] == 0).all(), f"Untreated patient {pid} has treatment=1 intervals"
        else:
            # Treated patients may have both 0 and 1 (time-varying treatment)
            # At least one interval should be 1 (after treatment starts)
            assert (patient_rows["treatment"] == 1).any(), f"Treated patient {pid} has no treatment=1 intervals"

def test_counting_empty_agg():
    """Passing empty agg_df should still produce valid counting process."""
    dyn, sta = simulate_dynamic_static(n_pat=5, seed=920)
    agg_df = pd.DataFrame(columns=["pid", "start", "stop", "treatment"])
    
    counting = build_counting_process(static_df=sta, agg_df=agg_df, traj_aligned_df=pd.DataFrame(), id_col="pid")
    
    assert counting.shape[0] == 0, "Empty agg_df should produce empty result"
    assert {"pid", "start", "stop", "treatment"}.issubset(counting.columns)

def test_counting_single_observation_patient():
    """Patient with single time point should have one interval."""
    sta = pd.DataFrame({
        "pid": [99],
        "treatment": [1],
        "time_to_event": [5.0],
        "Tmax": [5.0],
        "age": [50]
    })
    dyn = pd.DataFrame({
        "pid": [99, 99],
        "time": [0.0, 0.0],
        "feature_name": ["eGFR", "SBP"],
        "value": [60.0, 120.0],
        "treated": [0, 0]
    })
    agg_df, _ = aggregate_covariates_to_bins(dyn, sta, agg_features=["SBP"], bin_width=1.0)
    counting = build_counting_process(static_df=sta, agg_df=agg_df, traj_aligned_df=pd.DataFrame(), id_col="pid")
    
    patient_rows = counting[counting["pid"] == 99]
    assert patient_rows.shape[0] >= 1, "Should have at least one interval"
    assert patient_rows["start"].min() == 0.0
    # Bins may extend slightly beyond Tmax due to bin width
    assert patient_rows["stop"].max() >= 5.0, "Should cover at least up to Tmax"
    assert patient_rows["stop"].max() <= 6.0, "Should not extend too far beyond Tmax"
