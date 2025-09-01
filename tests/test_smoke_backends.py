import numpy as np
import pandas as pd
from traj_ps.data.simulate import simulate_dynamic_static
from traj_ps.data.aggregate import DEFAULT_BIN_W
from traj_ps.data.counting import aggregate_covariates_to_bins, build_counting_process

from lifelines import CoxTimeVaryingFitter

from .conftest import skip_if_no_torch, skip_if_no_pymc, skip_if_no_pygam

# ---------- Deep ----------
@skip_if_no_torch
def test_deep_smoke():
    import torch
    from torch.utils.data import DataLoader
    from traj_ps.data.dual_prep import prepare_samples_dual
    from traj_ps.data.dataset import DualTimelineDS
    from traj_ps.data.collate import dual_pad_collate
    from traj_ps.backends.deep.model import DeepPSDual, cox_binned_partial_lik

    dyn, sta = simulate_dynamic_static(n_pat=10)
    samples = prepare_samples_dual(
        dynamic=dyn, static=sta,
        embed_feats=["eGFR","HbA1c"], agg_feats=["SBP","MedA"], bin_w=DEFAULT_BIN_W
    )
    ds = DualTimelineDS(samples)
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=dual_pad_collate)

    p_seq  = samples[0]["X_raw"].shape[1]
    p_std  = samples[0]["STD_agg"].shape[1]
    p_stat = samples[0]["Z"].shape[0]
    model = DeepPSDual(p_seq, p_std, p_stat, h=16, head_hidden=16)

    b = next(iter(dl))
    eta, _, H_agg, mask_sel = model(b["X_raw"], b["M_raw"], b["DT_raw"], b["STD_agg"], b["Z"], b["idx_map"])
    y = b["y_event"] * mask_sel
    r = b["at_risk"] * mask_sel
    loss = cox_binned_partial_lik(eta, y, r)
    assert np.isfinite(loss.item())

# ---------- Bayesian ----------
@skip_if_no_pymc
def test_bayes_smoke():
    from traj_ps.backends.bayes import BayesianTrajPS, BayesConfig
    from traj_ps.data.counting import align_traj_probs_to_bins

    dyn, sta = simulate_dynamic_static(n_pat=8)
    embed_labs = ["eGFR"]
    agg_feats  = ["SBP","MedA"]

    # build PS bins
    agg_df, bin_map = aggregate_covariates_to_bins(dyn, sta, agg_features=agg_feats, bin_width=DEFAULT_BIN_W)

    # compute posterior feature probabilities on tiny settings for speed
    cfg = BayesConfig(window_years=2.0, df_basis=4, n_samples=50, tune=50,
                      min_points_per_window=3, grid_freq=6)
    bayes = BayesianTrajPS(cfg)

    lab = "eGFR"
    lab_long = dyn.loc[dyn["feature_name"] == lab, ["pid","time","value"]] \
                 .rename(columns={"pid":"patient_id","value":"lab_value"})
    probs = bayes.embed(lab_long)
    assert {"patient_id","time"}.issubset(probs.columns)

    # align and build counting-process DF
    probs = probs.rename(columns={c: f"{c}__{lab}" for c in probs.columns if c.startswith("traj_prob_")})
    traj_aligned = align_traj_probs_to_bins({lab: probs}, bin_map, pid_col="patient_id", time_col="time")
    counting = build_counting_process(static_df=sta, agg_df=agg_df, traj_aligned_df=traj_aligned, id_col="pid")

    # fit Cox TVF and ensure PS exists
    ctv = CoxTimeVaryingFitter()
    ctv.fit(counting, id_col="pid", start_col="start", stop_col="stop", event_col="treatment")
    counting["ps"] = ctv.predict_partial_hazard(counting)
    assert "ps" in counting.columns and np.isfinite(counting["ps"]).all()

# ---------- GAM ----------
@skip_if_no_pygam
def test_gam_smoke():
    from traj_ps.backends.gam import GAMTrajPS, GAMConfig
    from traj_ps.backends.gam.align import align_gam_betas_to_bins

    dyn, sta = simulate_dynamic_static(n_pat=8)
    embed_labs = ["eGFR"]
    agg_feats  = ["SBP","MedA"]

    # PS bins + aggregated covariates
    agg_df, bin_map = aggregate_covariates_to_bins(dyn, sta, agg_features=agg_feats, bin_width=DEFAULT_BIN_W)

    # Build a lab-wide DF for eGFR only
    lab = "eGFR"
    lab_wide = dyn.loc[dyn["feature_name"] == lab, ["pid","time","value"]] \
                 .rename(columns={"pid":"patient_id","value": lab})

    gam = GAMTrajPS(GAMConfig(
        window_years=2.0, n_splines_range=(3,4), lam_grid=(0.3,1.0),
        min_points_per_window=5, standardize_y=True, cv_splits=0,
        max_iter=2000, spline_order=3, n_jobs=1, verbose=0
    ))

    cov = gam.embed(lab_wide)
    assert {"patient_id","time"}.issubset(cov.columns)

    aligned = align_gam_betas_to_bins(cov, bin_map, pid_col="patient_id", time_col="time")
    counting = build_counting_process(static_df=sta, agg_df=agg_df, traj_aligned_df=aligned, id_col="pid")

    ctv = CoxTimeVaryingFitter()
    ctv.fit(counting, id_col="pid", start_col="start", stop_col="stop", event_col="treatment")
    counting["ps"] = ctv.predict_partial_hazard(counting)
    assert "ps" in counting.columns and np.isfinite(counting["ps"]).all()
