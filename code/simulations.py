import numpy as np
import pandas as pd
import os
from bayesian_trajs import compute_time_varying_trajectory_covariates_parallel, pos_flags_from_traj
from gam_trajs import compute_gam_beta_covariates_adaptive, compute_gam_beta_covariates_adaptive_parallel
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

SEED = 920
rng = np.random.default_rng(SEED)

# ==== 1) Simulate longitudinal data for GAM testing ====
def simulate_cohort_for_gam(n_patients=300, min_follow=3.0, max_follow=8.0,
                            visit_rate_mean=4.0, visit_rate_cv=0.6,
                            missing_prob=0.10, seed=SEED):
    rng = np.random.default_rng(seed)

    def _visit_times(T, rate):
        t, out = 0.0, []
        while True:
            t += rng.exponential(1.0 / max(rate, 1e-6))
            if t >= T:
                break
            out.append(t)
        return np.array(out)

    def _choose_type():
        return rng.choice(["linear", "nonlinear", "nonprogression", "fluctuating"],
                          p=[0.35, 0.30, 0.25, 0.10])

    def _egfr(t, kind, base):
        if kind == "linear":
            slope = -rng.uniform(1.0, 4.0)
            return base + slope*t
        if kind == "nonlinear":
            k = -rng.uniform(0.5, 2.0)
            return base + k*(t**1.5)
        if kind == "nonprogression":
            te = rng.uniform(0.6, 0.9) * (t.max() + 1e-9)
            slope2 = -rng.uniform(5.0, 12.0)
            return np.where(t < te, base, base + slope2*(t-te))
        # fluctuating
        amp = rng.uniform(4.0, 10.0)
        trend = -rng.uniform(0.5, 2.0)
        period = rng.uniform(0.8, 1.5)
        return base + trend*t + amp*np.sin(2*np.pi*t/period)

    def _cr_from_gfr(g):
        return np.clip(1.2 * (90.0 / np.maximum(g, 5.0)), 0.3, 8.0)

    def _a1c(t):
        base = rng.uniform(6.2, 8.2)
        drift = rng.normal(0.0, 0.15) * t
        seasonal = 0.2 * np.sin(2*np.pi*t/1.0 + rng.uniform(0, 2*np.pi))
        return np.clip(base + drift + seasonal, 5.5, 12.5)

    lab_rows, patient_rows = [], []

    # gamma-mixed Poisson rate across patients
    shape = 1.0 / (visit_rate_cv**2)
    scale = visit_rate_mean / shape

    for i in range(1, n_patients+1):
        pid = f"P{i:05d}"
        T = rng.uniform(min_follow, max_follow)
        rate = rng.gamma(shape, scale)

        t = _visit_times(T, rate)
        if t.size < 5:
            # ensure at least a few visits
            t = np.sort(np.r_[t, rng.uniform(0.1, T, size=5 - t.size)])

        typ = _choose_type()
        base = rng.uniform(60, 120)
        egfr_lat = _egfr(t, typ, base)

        # observations (with noise and missingness)
        egfr_obs = np.clip(egfr_lat + rng.normal(0, 3.0, size=t.size), 5.0, 200.0)
        cr_obs   = np.clip(_cr_from_gfr(egfr_lat) + rng.normal(0, 0.2, size=t.size), 0.3, 12.0)
        a1c_obs  = np.clip(_a1c(t) + rng.normal(0, 0.3, size=t.size), 4.5, 15.0)

        for ti, e, c, h in zip(t, egfr_obs, cr_obs, a1c_obs):
            if rng.random() > missing_prob:
                lab_rows.append((pid, float(ti), "eGFR", float(e)))
            if rng.random() > missing_prob:
                lab_rows.append((pid, float(ti), "Creatinine", float(c)))
            if rng.random() > missing_prob:
                lab_rows.append((pid, float(ti), "HbA1c", float(h)))

        patient_rows.append((pid, typ, float(T)))

    lab_df = pd.DataFrame(lab_rows, columns=["patient_id","time","lab_test","lab_value"]) \
             .sort_values(["patient_id","time","lab_test"]).reset_index(drop=True)
    patients_df = pd.DataFrame(patient_rows, columns=["patient_id","egfr_traj_type","followup_years"])
    return lab_df, patients_df

# ==== 2) Compute GAM β-covariates on sliding 2y windows ====
# Use your previously-defined function:
# from your_module import compute_gam_beta_covariates_adaptive

def build_testing_features(lab_df, patients_df,
                           labs=("eGFR","Creatinine","HbA1c"),
                           window_years=2.0,
                           n_splines_range=(3,6),
                           lam_grid=(0.3,1.0,3.0),
                           min_points_per_window=6):
    gam_covs = compute_gam_beta_covariates_adaptive_parallel(
        lab_df=lab_df,
        window_years=window_years,
        labs=list(labs),
        n_splines_range=n_splines_range,   # adaptive small range
        lam_grid=lam_grid,
        min_points_per_window=min_points_per_window,
        standardize_y=True,
        cv_splits=0,                       # use pyGAM GCV; set to 3/5 for CV
    )

    # For a clean supervised test, take the **latest available window** per patient
    last_win = gam_covs.sort_values(["patient_id","time"]).groupby("patient_id").tail(1)

    # Merge the true eGFR trajectory type (label)
    data = last_win.merge(patients_df[["patient_id","egfr_traj_type"]], on="patient_id", how="left")

    # X = all beta columns (and optionally the n_splines flags); y = type
    beta_cols = [c for c in data.columns if "_beta_" in c]
    flag_cols = [c for c in data.columns if c.endswith("_n_splines")]  # optional
    X = data[beta_cols + flag_cols].to_numpy()
    y = data["egfr_traj_type"].to_numpy()

    return data, X, y, beta_cols, flag_cols

# ==== 3) Evaluate: do β-covariates predict true trajectory type? ====
def evaluate_gam_betas(X, y, model="logit"):
    """
    Simple baseline: multinomial logistic reg or linear SVM on β-features.
    Returns mean accuracy via stratified 5-fold CV.
    """
    if model == "logit":
        clf = LogisticRegression(max_iter=2000, multi_class="multinomial", solver="lbfgs", n_jobs=1, random_state=SEED)
    elif model == "linear-svm":
        from sklearn.svm import LinearSVC
        clf = LinearSVC(random_state=SEED)
    else:
        raise ValueError("model must be 'logit' or 'linear-svm'")

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", clf)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1)
    return float(np.mean(acc)), float(np.std(acc))


def simulate_lab_data(n_patients=100, lab_tests=["eGFR"], seed=920):
    np.random.seed(seed)
    records = []

    for pid in range(1, n_patients + 1):
        n_measurements = np.random.randint(6, 25)
        times = np.sort(np.cumsum(np.random.exponential(scale=0.4, size=n_measurements)))
        times = np.round(times, 2)

        for lab in lab_tests:
            baseline = np.random.uniform(60, 120)
            progression_type = np.random.choice(["linear", "nonlinear", "stable", "fluctuating"])

            for t in times:
                if progression_type == "linear":
                    value = baseline - 2.5 * t + np.random.normal(0, 1)
                elif progression_type == "nonlinear":
                    value = baseline - 1.2 * t**1.5 + np.random.normal(0, 2)
                elif progression_type == "stable":
                    value = baseline + np.random.normal(0, 3)
                elif progression_type == "fluctuating":
                    value = baseline + 10 * np.sin(t) + np.random.normal(0, 5)

                value = max(value, 5)
                records.append({
                    "patient_id": f"P{pid:04d}",
                    "time": t,
                    "lab_test": lab,
                    "lab_value": value
                })

    return pd.DataFrame(records)


if __name__ == '__main__':

    # sim_labs = simulate_lab_data()
    # traj_covs = compute_time_varying_trajectory_covariates_parallel(sim_labs, min_points_per_window=6, n_samples=400, tune=400, n_jobs=-1, verbose=5)
    # traj_covs.to_csv('test_res/traj_covs_sim.csv')
    # print(traj_covs)
    lab_df, patients_df = simulate_cohort_for_gam(n_patients=400)


    # 2) build β-features from latest 2y window
    data, X, y, beta_cols, flag_cols = build_testing_features(
        lab_df, patients_df,
        labs=("eGFR", "Creatinine", "HbA1c"),
        window_years=2.0,
        n_splines_range=(4, 6),
        lam_grid=(0.3, 1.0, 3.0),
        min_points_per_window=6
    )

    # 3) quick supervised sanity check
    mean_acc, sd_acc = evaluate_gam_betas(X, y, model="logit")
    print(f"[GAM β-covariates] 5-fold CV accuracy predicting true eGFR type: {mean_acc:.3f} ± {sd_acc:.3f}")
    print(f"n patients used: {X.shape[0]}, n features: {X.shape[1]}")
    print("Example feature block columns (first 10):", beta_cols[:10])



