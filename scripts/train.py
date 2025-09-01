import argparse, os, yaml
import pandas as pd
import numpy as np

# ---------- shared data helpers ----------
from traj_ps.data.simulate import simulate_dynamic_static
from traj_ps.data.counting import (
    aggregate_covariates_to_bins,
    align_traj_probs_to_bins,       # for Bayesian posterior probs
    build_counting_process,
)
from traj_ps.data.aggregate import DEFAULT_BIN_W

# ---------- deep backend ----------
import torch
from torch.utils.data import DataLoader
from traj_ps.data.dual_prep import prepare_samples_dual
from traj_ps.data.dataset import DualTimelineDS
from traj_ps.data.collate import dual_pad_collate
from traj_ps.backends.deep.model import DeepPSDual, cox_binned_partial_lik

# ---------- bayesian backend ----------
from traj_ps.backends.bayes import BayesianTrajPS, BayesConfig

# ---------- gam backend ----------
from traj_ps.backends.gam import GAMTrajPS, GAMConfig
from traj_ps.backends.gam.align import align_gam_betas_to_bins  # LOCF align of GAM betas to bins


# ------------------------------
# utility: load YAML or defaults
# ------------------------------
def _load_yaml(path, fallback: dict):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return dict(fallback)


# ------------------------------
# DEEP: GRU-D (raw) -> gather -> Cox-binned head
# ------------------------------
def run_deep(train_cfg, data_cfg, save_dir="artifacts"):
    dyn, sta = simulate_dynamic_static(n_pat=data_cfg.get("n_patients", 120))
    samples = prepare_samples_dual(
        dynamic=dyn, static=sta,
        embed_feats=data_cfg["embed_features"],
        agg_feats=data_cfg["agg_features"],
        bin_w=float(data_cfg.get("bin_width", DEFAULT_BIN_W))
    )

    idx = torch.randperm(len(samples)).tolist()
    split = int(0.8 * len(idx))
    tr, va = idx[:split], idx[split:]
    from operator import itemgetter
    train_ds = DualTimelineDS(itemgetter(*tr)(samples) if len(tr) > 1 else [samples[tr[0]]])
    val_ds   = DualTimelineDS(itemgetter(*va)(samples) if len(va) > 1 else [samples[va[0]]])

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,  collate_fn=dual_pad_collate)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=dual_pad_collate)

    p_seq  = samples[0]["X_raw"].shape[1]
    p_std  = samples[0]["STD_agg"].shape[1]
    p_stat = samples[0]["Z"].shape[0]

    device = torch.device("cuda" if (train_cfg.get("device","auto")=="auto" and torch.cuda.is_available()) else "cpu")
    model  = DeepPSDual(p_seq=p_seq, p_std=p_std, p_static=p_stat,
                        h=train_cfg["hidden"], head_hidden=train_cfg["head_hidden"]).to(device)
    optim  = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    def run_epoch(loader, train=True):
        model.train(train)
        total = n = 0
        for b in loader:
            for k in ("X_raw","M_raw","DT_raw","STD_agg","Z","y_event","at_risk","idx_map"):
                b[k] = b[k].to(device)
            eta, _, H_agg, mask_sel = model(b["X_raw"], b["M_raw"], b["DT_raw"], b["STD_agg"], b["Z"], b["idx_map"])
            y = b["y_event"] * mask_sel
            r = b["at_risk"] * mask_sel
            loss = cox_binned_partial_lik(eta, y, r)
            if train:
                optim.zero_grad(); loss.backward(); optim.step()
            total += float(loss.item()); n += 1
        return total / max(1, n)

    for ep in range(1, int(train_cfg["epochs"]) + 1):
        tr_loss = run_epoch(train_loader, True)
        va_loss = run_epoch(val_loader, False)
        print(f"epoch {ep:02d}  train {tr_loss:.4f}  valid {va_loss:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "p_seq": p_seq, "p_std": p_std, "p_static": p_stat,
                "h": train_cfg["hidden"], "head_hidden": train_cfg["head_hidden"]},
               os.path.join(save_dir, "deep_cox_binned.pt"))
    print(f"Saved model to {os.path.join(save_dir,'deep_cox_binned.pt')}")


# ------------------------------
# BAYES: posterior feature probs -> align -> CoxTVF
# ------------------------------
def run_bayes(train_cfg, data_cfg, save_dir="artifacts"):
    dyn, sta = simulate_dynamic_static(n_pat=data_cfg.get("n_patients", 120))
    embed_labs = list(data_cfg["embed_features"])   # e.g., ["eGFR","HbA1c"]
    agg_feats  = list(data_cfg["agg_features"])     # e.g., ["SBP","MedA"]
    bin_w      = float(data_cfg.get("bin_width", DEFAULT_BIN_W))

    # aggregated covariates (PS grid)
    agg_df, bin_map = aggregate_covariates_to_bins(dyn, sta, agg_features=agg_feats, bin_width=bin_w)

    # compute Bayesian trajectory probabilities for each lab separately (tidy long)
    bayes = BayesianTrajPS(BayesConfig(
        window_years=train_cfg.get("window_years", 2.0),
        df_basis=train_cfg.get("df_basis", 5),
        n_samples=train_cfg.get("n_samples", 1000),
        tune=train_cfg.get("tune", 1000),
        min_points_per_window=train_cfg.get("min_points_per_window", 5),
        grid_freq=train_cfg.get("grid_freq", 12),
        flat_thr=train_cfg.get("flat_thr", -1.0),
        decline_thr=train_cfg.get("decline_thr", -2.0),
        nonlinear_gap=train_cfg.get("nonlinear_gap", 3.0),
        pids="patient_id", values="lab_value", time_col="time",
    ))

    traj_probs_by_lab = {}
    for lab in embed_labs:
        lab_long = dyn.loc[dyn["feature_name"] == lab, ["pid","time","value"]].rename(
            columns={"pid":"patient_id","value":"lab_value"}
        )
        probs = bayes.embed(lab_long)  # ['patient_id','time','traj_prob_*']
        # suffix with lab name to keep them distinct when merging
        prob_cols = [c for c in probs.columns if c.startswith("traj_prob_")]
        probs = probs.rename(columns={c: f"{c}__{lab}" for c in prob_cols})
        traj_probs_by_lab[lab] = probs

    # align posterior probs to bin starts via LOCF
    traj_aligned = align_traj_probs_to_bins(traj_probs_by_lab, bin_map, pid_col="patient_id", time_col="time")

    # build counting-process DF & fit CoxTVF (Lu-style)
    counting = build_counting_process(static_df=sta, agg_df=agg_df, traj_aligned_df=traj_aligned, id_col="pid")

    from lifelines import CoxTimeVaryingFitter
    ctv = CoxTimeVaryingFitter()
    ctv.fit(counting, id_col="pid", start_col="start", stop_col="stop", event_col="treatment")
    counting["ps"] = ctv.predict_partial_hazard(counting)

    os.makedirs(save_dir, exist_ok=True)
    ctv.save(os.path.join(save_dir, "bayes_cox_tvf.pkl"))
    counting.to_parquet(os.path.join(save_dir, "bayes_ps.parquet"), index=False)
    print(f"Saved CoxTVF to {os.path.join(save_dir,'bayes_cox_tvf.pkl')} and PS to {os.path.join(save_dir,'bayes_ps.parquet')}")


# ------------------------------
# GAM: windowed GAM betas -> align -> CoxTVF
# ------------------------------
def run_gam(train_cfg, data_cfg, save_dir="artifacts"):
    dyn, sta = simulate_dynamic_static(n_pat=data_cfg.get("n_patients", 120))
    embed_labs = list(data_cfg["embed_features"])   # labs to encode via GAM betas
    agg_feats  = list(data_cfg["agg_features"])     # additional aggregated covariates
    bin_w      = float(data_cfg.get("bin_width", DEFAULT_BIN_W))

    # PS bins + aggregated covariates
    agg_df, bin_map = aggregate_covariates_to_bins(dyn, sta, agg_features=agg_feats, bin_width=bin_w)

    # compute GAM covariates PER LAB (avoid wide-table NaN issues), then outer-merge on [patient_id,time]
    cfg = GAMConfig(
        window_years=train_cfg.get("window_years", 2.0),
        n_splines_range=tuple(train_cfg.get("n_splines_range", (4,6))),
        lam_grid=tuple(train_cfg.get("lam_grid", (0.3, 1.0, 3.0))),
        min_points_per_window=train_cfg.get("min_points_per_window", 6),
        standardize_y=train_cfg.get("standardize_y", True),
        cv_splits=train_cfg.get("cv_splits", 0),
        max_iter=train_cfg.get("max_iter", 5000),
        spline_order=train_cfg.get("spline_order", 3),
        n_jobs=train_cfg.get("n_jobs", -1),
        verbose=train_cfg.get("verbose", 5),
        pca_components=train_cfg.get("pca_components", None),
    )
    gam = GAMTrajPS(cfg)

    cov_merged = None
    for lab in embed_labs:
        lab_wide = dyn.loc[dyn["feature_name"] == lab, ["pid","time","value"]] \
                     .rename(columns={"pid":"patient_id", "value": lab})
        # embed() expects columns ['patient_id','time', <lab>]
        cov_lab = gam.embed(lab_wide)
        cov_merged = cov_lab if cov_merged is None else cov_merged.merge(
            cov_lab, on=["patient_id","time"], how="outer"
        )

    cov_merged = cov_merged.sort_values(["patient_id","time"]).reset_index(drop=True)

    # align GAM covariates to bin starts via LOCF
    gam_aligned = align_gam_betas_to_bins(cov_merged, bin_map, pid_col="patient_id", time_col="time")

    # counting-process DF & CoxTVF
    counting = build_counting_process(static_df=sta, agg_df=agg_df, traj_aligned_df=gam_aligned, id_col="pid")

    from lifelines import CoxTimeVaryingFitter
    ctv = CoxTimeVaryingFitter()
    ctv.fit(counting, id_col="pid", start_col="start", stop_col="stop", event_col="treatment")
    counting["ps"] = ctv.predict_partial_hazard(counting)

    os.makedirs(save_dir, exist_ok=True)
    ctv.save(os.path.join(save_dir, "gam_cox_tvf.pkl"))
    counting.to_parquet(os.path.join(save_dir, "gam_ps.parquet"), index=False)
    print(f"Saved CoxTVF to {os.path.join(save_dir,'gam_cox_tvf.pkl')} and PS to {os.path.join(save_dir,'gam_ps.parquet')}")


# ------------------------------
# main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["deep","bayes","gam"], default="bayes")
    ap.add_argument("--data_cfg", type=str, default="configs/data.yaml")
    ap.add_argument("--train_cfg", type=str, default=None)
    ap.add_argument("--save_dir", type=str, default="artifacts")
    args = ap.parse_args()

    # defaults for each backend
    deep_defaults = dict(epochs=5, batch_size=32, lr=1e-3, weight_decay=1e-4, hidden=64, head_hidden=64, device="auto")
    bayes_defaults = dict(window_years=2.0, df_basis=5, n_samples=1000, tune=1000,
                          min_points_per_window=5, grid_freq=12, flat_thr=-1.0, decline_thr=-2.0, nonlinear_gap=3.0)
    gam_defaults = dict(window_years=2.0, n_splines_range=(4,6), lam_grid=(0.3,1.0,3.0),
                        min_points_per_window=6, standardize_y=True, cv_splits=0, max_iter=5000,
                        spline_order=3, n_jobs=-1, verbose=5, pca_components=None)

    data_cfg  = _load_yaml(args.data_cfg, fallback=dict(
        seed=920, n_patients=120, bin_width=1/12, embed_features=["eGFR","HbA1c"], agg_features=["SBP","MedA"]
    ))
    train_cfg = _load_yaml(args.train_cfg, fallback=dict())

    if args.backend == "deep":
        deep_cfg = {**deep_defaults, **train_cfg}
        run_deep(deep_cfg, data_cfg, save_dir=args.save_dir)

    elif args.backend == "bayes":
        bayes_cfg = {**bayes_defaults, **train_cfg}
        run_bayes(bayes_cfg, data_cfg, save_dir=args.save_dir)

    elif args.backend == "gam":  # gam
        gam_cfg = {**gam_defaults, **train_cfg}
        run_gam(gam_cfg, data_cfg, save_dir=args.save_dir)

    else:
        raise ValueError(f"Unknown backend: {args.backend}")


if __name__ == "__main__":
    main()
