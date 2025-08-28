from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Optional
import matplotlib.pyplot as plt
import sys
from deep_trajs import *
from deep_trajs_preprocessing import *
from deep_trajs_train import *

SEED = 920
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# -----------------------------
# 1) Simulation: dynamic + static tables
# -----------------------------
@dataclass
class SimConfig:
    n_patients: int = 200
    followup_min: float = 2.0   # years
    followup_max: float = 6.0
    visit_rate_mean: float = 6.0  # events/year (per feature; features interleave)
    visit_rate_cv: float = 0.6
    features: Tuple[str, ...] = ("eGFR","Creatinine","SBP","DBP","HbA1c","MedA","MedB")
    # which features are binary indicators (e.g., meds)
    binary_features: Tuple[str, ...] = ("MedA","MedB")

def simulate_dynamic_static(cfg: SimConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      dynamic_df: columns [pid, time, feature_name, value, treated]
      static_df : columns [pid, age, sex, cci]
    Notes:
      - time in years (float), irregular per feature
      - 'treated' is cumulative: 0 until first treatment time, then 1 afterwards
    """
    dyn_rows = []
    stat_rows = []
    shape = 1.0 / (cfg.visit_rate_cv**2)
    scale = cfg.visit_rate_mean / shape

    for i in range(cfg.n_patients):
        pid = f"P{i:05d}"
        T = rng.uniform(cfg.followup_min, cfg.followup_max)
        age = int(rng.integers(45, 85))
        sex = int(rng.integers(0, 2))
        cci = int(rng.poisson(2))
        stat_rows.append((pid, age, sex, cci, float(T)))

        # latent processes for continuous features
        def egfr_f(t):
            slope = -rng.uniform(1.0, 4.0)
            base = rng.uniform(60, 120)
            return np.clip(base + slope*t + rng.normal(0,3,size=t.size), 5, 200)

        def cr_from_gfr(g):
            return np.clip(1.2*(90/np.maximum(g,5)), 0.3, 8) + rng.normal(0,0.2,size=g.size)

        def sbp_f(t):
            return np.clip(120 + 5*np.sin(2*np.pi*t) + rng.normal(0,8,size=t.size), 90, 200)

        def dbp_f(t):
            return np.clip(75 + 3*np.cos(2*np.pi*t/1.5) + rng.normal(0,6,size=t.size), 50, 120)

        def hba1c_f(t):
            base = rng.uniform(6.0, 8.5)
            drift = rng.normal(0, 0.1) * t
            return np.clip(base + drift + 0.2*np.sin(2*np.pi*t), 5, 14)

        latent_map = {
            "eGFR": egfr_f,
            "Creatinine": lambda t: cr_from_gfr(egfr_f(t)),
            "SBP": sbp_f,
            "DBP": dbp_f,
            "HbA1c": hba1c_f,
        }

        # generate irregular times for each feature independently
        feat_times: Dict[str, np.ndarray] = {}
        for f in cfg.features:
            # meds are state toggles; others are measurements
            rate = rng.gamma(shape, scale)
            times, t = [], 0.0
            while True:
                t += rng.exponential(1.0 / max(rate, 1e-6))
                if t >= T: break
                times.append(t)
            if len(times) == 0:
                times = [rng.uniform(0.1, T)]
            feat_times[f] = np.array(times, dtype=float)

        # simulate med indicators (state changes)
        med_state = {m: 0 for m in cfg.binary_features}
        for m in cfg.binary_features:
            for t in feat_times[m]:
                if rng.random() < 0.4:  # 40% chance to toggle at an event time
                    med_state[m] = 1 - med_state[m]
                dyn_rows.append((pid, float(t), m, float(med_state[m]), 0))

        # simulate continuous values
        for f, ts in feat_times.items():
            if f in cfg.binary_features:
                continue
            vals = latent_map.get(f, lambda x: rng.normal(0,1,size=x.size))(ts)
            for t, v in zip(ts, vals):
                dyn_rows.append((pid, float(t), f, float(v), 0))

        # simulate treatment start hazard based on current states
        # build a combined timeline of all observed times
        all_times = np.unique(np.concatenate([feat_times[f] for f in cfg.features]))
        treated_flag = 0
        treat_start_time = None
        medA_series = {t: 0 for t in all_times}
        # reconstruct MedA state over time (last value carried forward)
        mA_times = np.sort(feat_times["MedA"])
        state = 0
        for t in all_times:
            # update state if t is an event time for MedA (we recorded values in dyn_rows)
            # approximate by nearest <= t
            prior = mA_times[mA_times<=t]
            if prior.size>0:
                # find last recorded value in dyn_rows for MedA at time prior[-1]
                pass
            medA_series[t] = state

        # simple hazard using latent eGFR trend + MedA (proxy)
        # approximate eGFR at grid times using linear interp of measured eGFR
        egfr_obs = np.array([row[3] for row in dyn_rows if row[0]==pid and row[2]=="eGFR"])
        egfr_t   = np.array([row[1] for row in dyn_rows if row[0]==pid and row[2]=="eGFR"])
        if egfr_t.size >= 2:
            v = np.interp(all_times, egfr_t, egfr_obs)
        else:
            v = np.full(all_times.shape, np.median(egfr_obs) if egfr_obs.size>0 else 90.0)
        low_gfr = np.maximum(0.0, 90.0 - v)
        score = -3.2 + 0.03*low_gfr + 0.2*sex + 0.05*cci
        p_bin = 1 - np.exp(-np.exp(score))  # discrete-time hazard

        for t, p in zip(all_times, p_bin):
            if treated_flag==0 and rng.random() < p:
                treated_flag = 1
                treat_start_time = float(t)
                break

        # append treated=1 after treat_start_time to dynamic table (carry as indicator)
        if treat_start_time is not None:
            for idx, (ppid, t, name, val, tr) in enumerate(dyn_rows):
                if ppid==pid and t>=treat_start_time:
                    dyn_rows[idx] = (ppid, t, name, val, 1)
        # ensure at least one record with treated=1 at treat start time (for alignment)
        # (optional; downstream code uses aggregated timeline)
    dynamic_df = pd.DataFrame(dyn_rows, columns=["pid","time","feature_name","value","treated"])
    static_df  = pd.DataFrame(stat_rows, columns=["pid","age","sex","cci","followup_T"])
    return dynamic_df.sort_values(["pid","time","feature_name"]).reset_index(drop=True), \
           static_df.sort_values("pid").reset_index(drop=True)

# -----------------------------
# 2) Aggregation into PS time grid
# -----------------------------
def make_time_bins(T_max: float, bin_width: float) -> np.ndarray:
    # bins edges [0, T_max] with fixed width
    n = int(np.ceil(T_max/bin_width))
    edges = np.linspace(0.0, n*bin_width, n+1)
    return edges

def aggregate_dynamic(
    dynamic_df: pd.DataFrame,
    static_df: pd.DataFrame,
    agg_features: Iterable[str],
    bin_width: float = 1/12  # monthly (years)
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Aggregates selected features into bins for PS covariates.
    Returns:
      agg_df: rows per (pid, bin_idx) with aggregated covariates and 'treated' (already initiated by start of bin)
      bin_times: dict pid -> array of bin start times (used later for alignment)
    Aggregation: mean within bin for continuous; last-observation-carried-forward (LOCF) for binaries.
    """
    agg_features = list(agg_features)
    bin_times: Dict[str, np.ndarray] = {}
    rows = []
    for pid, g in dynamic_df.groupby("pid"):
        T = float(static_df.loc[static_df.pid==pid, "followup_T"].iloc[0])
        edges = make_time_bins(T, bin_width)
        starts = edges[:-1]
        bin_times[pid] = starts

        # determine treatment start time (first time with treated==1)
        tr_times = g.loc[g["treated"]==1, "time"]
        treat_start_time = float(tr_times.min()) if not tr_times.empty else None

        for b, t0 in enumerate(starts):
            t1 = edges[b+1]
            sl = g[(g["time"]>=t0) & (g["time"]<t1)]
            rec = {"pid": pid, "bin_idx": b, "time": float(t0)}
            # aggregated features
            for feat in agg_features:
                gl = sl[sl["feature_name"]==feat]
                if feat.startswith("Med") or gl["value"].dropna().isin([0,1]).all():
                    # binary: take last value in bin; if none, carry from previous bin
                    if not gl.empty:
                        val = float(gl.sort_values("time")["value"].iloc[-1])
                        rec[feat] = val
                    else:
                        # carry forward later after building DataFrame
                        rec[feat] = np.nan
                else:
                    # continuous: mean within bin
                    rec[feat] = float(gl["value"].mean()) if not gl.empty else np.nan
            # treated indicator at **start** of the bin
            rec["treated"] = int(0 if (treat_start_time is None or t0 < treat_start_time) else 1)
            rows.append(rec)

    agg_df = pd.DataFrame(rows).sort_values(["pid","time"]).reset_index(drop=True)
    # fill forward binary features per patient
    for feat in agg_features:
        if feat.startswith("Med"):
            agg_df[feat] = agg_df.groupby("pid")[feat].ffill().fillna(0.0)
        else:
            # fill with last observation carried forward, else overall median
            agg_df[feat] = agg_df.groupby("pid")[feat].apply(lambda s: s.ffill()).reset_index(level=0, drop=True)
            agg_df[feat] = agg_df[feat].fillna(agg_df[feat].median())
    return agg_df, bin_times

# -----------------------------
# 3) Build GRU inputs from non-aggregated sequences (per embedded feature)
#     - last valid time aligns to each aggregated bin start
#     - sequences truncated to times <= bin start
#     - if no prior data for a feature, use mask=0 and filler value (mean)
# -----------------------------
def build_targets_from_treated(treated_series: np.ndarray, include_event_bin_in_risk: bool = True):
    """
    treated_series : (T,) int {0/1}, value at the **start** of each bin (monotone nondecreasing)
    Returns:
      y_event : (T,) one-hot at first bin where treated switches to 1 (else all zeros if censored)
      at_risk : (T,) 1 while untreated at start of bin; 0 after event (or after T if censored)
                If include_event_bin_in_risk=True, at_risk[event_idx] = 1 (common in discrete-time setups).
    """
    T = int(len(treated_series))
    y_event = np.zeros(T, dtype=float)
    at_risk = np.zeros(T, dtype=float)

    # find first switch to 1
    idxs = np.flatnonzero(treated_series == 1)
    event_idx = int(idxs[0]) if idxs.size > 0 else None

    if event_idx is None:
        # censored: at-risk for all bins
        at_risk[:] = 1.0
        return y_event, at_risk

    # event present
    y_event[event_idx] = 1.0
    if include_event_bin_in_risk:
        at_risk[:event_idx + 1] = 1.0
    else:
        at_risk[:event_idx] = 1.0
    return y_event, at_risk


def build_sequences_for_bins(
    dynamic_df: pd.DataFrame,
    static_df: pd.DataFrame,
    bin_times: Dict[str, np.ndarray],
    embed_features: List[str],
    agg_df: pd.DataFrame,
) -> List[Dict]:
    """
    Returns a list of per-patient samples with keys:
      'pid','time','X','M','DT','STD','Z','y_event','at_risk'
    where:
      - 'time' are the aggregated bin start times for that patient
      - 'X','M','DT' are built from irregular per-feature histories up to each bin time
      - 'STD' are the aggregated features from agg_df for that bin (you can choose them)
      - 'treated' from agg_df defines y_event/at_risk (first bin with treated==1 is event)
    """
    # Decide what goes into STD: here we include all agg_df features except treated/pid/bin_idx/time
    std_features = [c for c in agg_df.columns if c not in ("pid","bin_idx","time","treated")]
    # map per pid -> aggregated rows
    agg_map = {pid: g.sort_values("time").reset_index(drop=True) for pid, g in agg_df.groupby("pid")}

    samples = []
    for pid, g_dyn in dynamic_df.groupby("pid"):
        times_bins = bin_times[pid]
        g_agg = agg_map[pid]
        T_bins = len(times_bins)

        # build event targets from aggregated treated flag
        treated_series = g_agg["treated"].to_numpy().astype(int)
        treated_series = np.maximum.accumulate(treated_series)
        # first bin where treated==1
        event_idx = int(np.argmax(treated_series==1)) if treated_series.any() else None
        y_event, at_risk = build_targets_from_treated(
            treated_series,
            include_event_bin_in_risk=True  # matches your earlier loss setup
        )
        for t_idx in range(T_bins):
            if event_idx is None:
                at_risk[t_idx] = 1.0
            else:
                at_risk[t_idx] = 1.0 if t_idx <= event_idx else 0.0
                y_event[t_idx]  = 1.0 if t_idx == event_idx else 0.0

        # build GRU inputs X,M,DT **at the aggregated bin grid**, but using history up to each bin
        # We will **carry last observation** per feature at each bin time; mask=1 if observed before or at bin, else 0
        X = np.zeros((T_bins, len(embed_features)), float)
        M = np.zeros_like(X)
        DT = np.ones((T_bins, 1), float)

        # pre-extract per embedded feature event times and values
        hist = {}
        for j, feat in enumerate(embed_features):
            gf = g_dyn[g_dyn["feature_name"]==feat].sort_values("time")
            hist[feat] = (gf["time"].to_numpy(float), gf["value"].to_numpy(float))

        last_obs_time = np.zeros(len(embed_features), float)
        for t_idx, t0 in enumerate(times_bins):
            for j, feat in enumerate(embed_features):
                ft, fv = hist[feat]
                if ft.size == 0 or ft[ft <= t0].size == 0:
                    # no history yet → leave X=0, M=0, DT=1
                    continue
                # last observed value at or before t0
                k = np.searchsorted(ft, t0, side="right") - 1
                X[t_idx, j] = float(fv[k])
                M[t_idx, j] = 1.0
                # time since last obs
                DT[t_idx, 0] = float(max(1e-6, t0 - float(ft[k])))
                last_obs_time[j] = float(ft[k])

        # STD from aggregated features at the same bin index
        STD = g_agg[std_features].to_numpy(float)
        Z = static_df.loc[static_df.pid==pid, ["age","sex","cci"]].iloc[0].to_numpy(float)

        samples.append({
            "pid": pid,
            "time": times_bins.astype(float),
            "X": X, "M": M, "DT": DT,
            "STD": STD, "Z": Z,
            "y_event": y_event, "at_risk": at_risk
        })
    return samples

# -----------------------------
# 4) Convenience driver
# -----------------------------
def simulate_and_prepare_pipeline(
    embed_features: List[str],
    agg_features: List[str],
    bin_width: float = 1/12,
    cfg: Optional[SimConfig] = None
):
    cfg = cfg or SimConfig()
    dynamic_df, static_df = simulate_dynamic_static(cfg)
    # aggregation includes only selected agg_features; others ignored for STD
    dyn_subset = dynamic_df[dynamic_df["feature_name"].isin(set(embed_features) | set(agg_features) | set(cfg.binary_features))]
    agg_df, bin_times = aggregate_dynamic(dyn_subset, static_df, agg_features=agg_features, bin_width=bin_width)
    samples = build_sequences_for_bins(dyn_subset, static_df, bin_times, embed_features=embed_features, agg_df=agg_df)
    return dynamic_df, static_df, agg_df, samples

def build_targets_from_treated(treated_series: np.ndarray, include_event_bin_in_risk: bool = True):
    """
    treated_series : (T,) int {0/1}, value at the **start** of each bin (monotone nondecreasing)
    Returns:
      y_event : (T,) one-hot at first bin where treated switches to 1 (else all zeros if censored)
      at_risk : (T,) 1 while untreated at start of bin; 0 after event (or after T if censored)
                If include_event_bin_in_risk=True, at_risk[event_idx] = 1 (common in discrete-time setups).
    """
    T = int(len(treated_series))
    y_event = np.zeros(T, dtype=float)
    at_risk = np.zeros(T, dtype=float)

    # find first switch to 1
    idxs = np.flatnonzero(treated_series == 1)
    event_idx = int(idxs[0]) if idxs.size > 0 else None

    if event_idx is None:
        # censored: at-risk for all bins
        at_risk[:] = 1.0
        return y_event, at_risk

    # event present
    y_event[event_idx] = 1.0
    if include_event_bin_in_risk:
        at_risk[:event_idx + 1] = 1.0
    else:
        at_risk[:event_idx] = 1.0
    return y_event, at_risk

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 3:
        print("program requires path to results dir and identifier for results version")
        sys.exit()

    res_dir = args[1]
    res_ver = args[2]

    embed_feats = ["eGFR", "HbA1c"]  # go to GRU (sequence)
    agg_feats = ["Creatinine", "SBP", "MedA", "eGFR", "HbA1c"]  # aggregated covariates

    dynamic_df, static_df, agg_df, samples = simulate_and_prepare_pipeline(
        embed_features=embed_feats,
        agg_features=agg_feats,
        bin_width= 1/12,  # monthly bins
        cfg=SimConfig(n_patients=200)
    )
    print(dynamic_df.head())
    print(agg_df.head())
    print(samples[0].keys(), samples[0]["X"].shape, samples[0]["STD"].shape)

    dynamic_df.to_csv(f"{res_dir}/simulated_dynamic_{res_ver}.csv", index=False)
    static_df.to_csv(f"{res_dir}/simulated_static_{res_ver}.csv", index=False)
    agg_df.to_csv(f"{res_dir}/simulated_aggregated_{res_ver}.csv", index=False)
    pd.DataFrame(samples).to_csv(f"{res_dir}/simulated_samples_{res_ver}.csv")
    p_seq = samples[0]["X"].shape[1]
    p_std = samples[0]["STD"].shape[1]
    p_static = samples[0]["Z"].shape[0]

    # split
    idx = np.random.permutation(len(samples))
    train_idx, val_idx = idx[:160], idx[160:]

    train_ds = TVPropensityDataset([samples[i] for i in train_idx])  # list of dicts with the schema above
    val_ds = TVPropensityDataset([samples[i] for i in val_idx])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=pad_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=pad_collate, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepPS(p_seq=p_seq, p_std=p_std, p_static=p_static, h=64, head_hidden=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_hist, val_hist = training_loop(train_loader, val_loader, model, device, opt, 25)
    df_ps, hmap = predict_ps(model, val_loader, device, group_by="time", return_embeddings=True)


    # plot training process
    plt.plot(range(1, 26), train_hist, label="train", color="#598e4d")
    plt.plot(range(1, 26), val_hist, label="validation", color="#d37860")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Simulation Training Loss")
    plt.savefig(f"{res_dir}/sim_train_history_{res_ver}.png")

    # save model's output
    time_map = {}
    for batch in val_loader:
        for pid, t_arr in zip(batch["pid"], batch["time"]):
            time_map[pid] = t_arr
    df_h = Hmap_to_df(hmap, time_map)

    df_ps.to_csv(f"{res_dir}/sim_ps_{res_ver}.csv")
    df_h.to_csv(f"{res_dir}/sim_embeddings_{res_ver}.csv")
