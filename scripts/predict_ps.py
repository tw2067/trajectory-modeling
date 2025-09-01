import argparse, os, yaml, numpy as np, pandas as pd

# shared data helpers (bins + counting design)
from traj_ps.data.aggregate import DEFAULT_BIN_W
from traj_ps.data.counting import (
    aggregate_covariates_to_bins,
    align_traj_probs_to_bins,   # bayes alignment
    build_counting_process,
)

# simulation (only for demo if paths not provided)
from traj_ps.data.simulate import simulate_dynamic_static

# ---------- deep ----------
import torch
from torch.utils.data import DataLoader
from traj_ps.data.dual_prep import prepare_samples_dual
from traj_ps.data.dataset import DualTimelineDS
from traj_ps.data.collate import dual_pad_collate
from traj_ps.backends.deep.model import DeepPSDual
from traj_ps.inference.predict_ps import predict_ps as predict_ps_deep

# ---------- bayesian ----------
from traj_ps.backends.bayes import BayesianTrajPS, BayesConfig

# ---------- gam ----------
from traj_ps.backends.gam import GAMTrajPS, GAMConfig
from traj_ps.backends.gam.align import align_gam_betas_to_bins


def _load_yaml(path, fallback):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return dict(fallback)


def load_or_simulate(dynamic_path, static_path, n_patients=120):
    if dynamic_path and static_path and os.path.exists(dynamic_path) and os.path.exists(static_path):
        return pd.read_parquet(dynamic_path), pd.read_parquet(static_path)
    return simulate_dynamic_static(n_pat=n_patients)


def run_deep(args, data_cfg):
    dyn, sta = load_or_simulate(args.dynamic, args.static, data_cfg.get("n_patients", 120))
    samples = prepare_samples_dual(
        dynamic=dyn, static=sta,
        embed_feats=data_cfg["embed_features"],
        agg_feats=data_cfg["agg_features"],
        bin_w=float(data_cfg.get("bin_width", DEFAULT_BIN_W))
    )
    ds = DualTimelineDS(samples)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=dual_pad_collate)

    ckpt = torch.load(args.model_path, map_location="cpu")
    model = DeepPSDual(ckpt["p_seq"], ckpt["p_std"], ckpt["p_static"], ckpt["h"], ckpt["head_hidden"])
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    df_ps, Hmap = predict_ps_deep(model, dl, return_embeddings=True)
    df_ps.to_parquet(args.out, index=False)
    print(f"[deep] wrote PS to {args.out}")
    if args.export_embeddings:
        # flatten Hmap to long form
        rows=[]
        for pid, H in Hmap.items():
            T, D = H.shape
            for t in range(T):
                rows.append({"patient_id": pid, "t_idx": t, **{f"h{j+1}": float(H[t,j]) for j in range(D)}})
        embs = pd.DataFrame(rows)
        emb_path = os.path.splitext(args.out)[0] + "_embeddings.parquet"
        embs.to_parquet(emb_path, index=False)
        print(f"[deep] wrote embeddings to {emb_path}")


def run_bayes(args, data_cfg, train_cfg):
    dyn, sta = load_or_simulate(args.dynamic, args.static, data_cfg.get("n_patients", 120))
    embed_labs = list(data_cfg["embed_features"])
    agg_feats  = list(data_cfg["agg_features"])
    bin_w      = float(data_cfg.get("bin_width", DEFAULT_BIN_W))

    agg_df, bin_map = aggregate_covariates_to_bins(dyn, sta, agg_features=agg_feats, bin_width=bin_w)

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
        lab_long = dyn.loc[dyn["feature_name"] == lab, ["pid","time","value"]] \
                     .rename(columns={"pid":"patient_id","value":"lab_value"})
        probs = bayes.embed(lab_long)  # uses posterior feature probability method (Monte Carlo feature proportion)
        prob_cols = [c for c in probs.columns if c.startswith("traj_prob_")]
        probs = probs.rename(columns={c: f"{c}__{lab}" for c in prob_cols})
        traj_probs_by_lab[lab] = probs

    traj_aligned = align_traj_probs_to_bins(traj_probs_by_lab, bin_map, pid_col="patient_id", time_col="time")
    counting = build_counting_process(static_df=sta, agg_df=agg_df, traj_aligned_df=traj_aligned, id_col="pid")

    # load saved CoxTimeVaryingFitter and score
    from lifelines import CoxTimeVaryingFitter
    ctv = CoxTimeVaryingFitter(); ctv.load(args.model_path)
    counting["ps"] = ctv.predict_partial_hazard(counting)
    counting.to_parquet(args.out, index=False)
    print(f"[bayes] wrote PS to {args.out}")


def run_gam(args, data_cfg, train_cfg):
    dyn, sta = load_or_simulate(args.dynamic, args.static, data_cfg.get("n_patients", 120))
    embed_labs = list(data_cfg["embed_features"])
    agg_feats  = list(data_cfg["agg_features"])
    bin_w      = float(data_cfg.get("bin_width", DEFAULT_BIN_W))

    agg_df, bin_map = aggregate_covariates_to_bins(dyn, sta, agg_features=agg_feats, bin_width=bin_w)

    gam = GAMTrajPS(GAMConfig(
        window_years=train_cfg.get("window_years", 2.0),
        n_splines_range=tuple(train_cfg.get("n_splines_range", (4,6))),
        lam_grid=tuple(train_cfg.get("lam_grid", (0.3,1.0,3.0))),
        min_points_per_window=train_cfg.get("min_points_per_window", 6),
        standardize_y=train_cfg.get("standardize_y", True),
        cv_splits=train_cfg.get("cv_splits", 0),
        max_iter=train_cfg.get("max_iter", 5000),
        spline_order=train_cfg.get("spline_order", 3),
        n_jobs=train_cfg.get("n_jobs", -1),
        verbose=train_cfg.get("verbose", 5),
        pca_components=train_cfg.get("pca_components", None),
    ))

    # embed per lab, then outer-merge on [patient_id,time]
    cov = None
    for lab in embed_labs:
        lab_wide = dyn.loc[dyn["feature_name"] == lab, ["pid","time","value"]] \
                     .rename(columns={"pid":"patient_id", "value": lab})
        cov_lab = gam.embed(lab_wide)          # robust adaptive GAM betas per window
        cov = cov_lab if cov is None else cov.merge(cov_lab, on=["patient_id","time"], how="outer")
    cov = cov.sort_values(["patient_id","time"]).reset_index(drop=True)

    gam_aligned = align_gam_betas_to_bins(cov, bin_map, pid_col="patient_id", time_col="time")
    counting = build_counting_process(static_df=sta, agg_df=agg_df, traj_aligned_df=gam_aligned, id_col="pid")

    # load saved CoxTVF and score
    from lifelines import CoxTimeVaryingFitter
    ctv = CoxTimeVaryingFitter(); ctv.load(args.model_path)
    counting["ps"] = ctv.predict_partial_hazard(counting)
    counting.to_parquet(args.out, index=False)
    print(f"[gam] wrote PS to {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["deep","bayes","gam"], required=True)
    ap.add_argument("--data_cfg", type=str, default="configs/data.yaml")
    ap.add_argument("--train_cfg", type=str, default=None)
    ap.add_argument("--dynamic", type=str, default=None)   # optional parquet path
    ap.add_argument("--static", type=str, default=None)    # optional parquet path
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=32)  # deep only
    ap.add_argument("--export_embeddings", action="store_true")
    args = ap.parse_args()

    data_cfg = _load_yaml(args.data_cfg, fallback=dict(
        seed=920, n_patients=120, bin_width=1/12, embed_features=["eGFR","HbA1c"], agg_features=["SBP","MedA"]
    ))
    train_cfg = _load_yaml(args.train_cfg, fallback=dict())

    if args.backend == "deep":
        run_deep(args, data_cfg)
    elif args.backend == "bayes":
        run_bayes(args, data_cfg, train_cfg)
    elif args.backend == "gam" :
        run_gam(args, data_cfg, train_cfg)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")


if __name__ == "__main__":
    main()
