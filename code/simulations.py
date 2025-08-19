import numpy as np
import pandas as pd
import os
from bayesian_trajs import compute_time_varying_trajectory_covariates_parallel, pos_flags_from_traj

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

    updrs3_scores = pd.read_csv('../data/ppmi/parsed/updrs3_parsed.csv')
    updrs3_on_scores = pd.read_csv('../data/ppmi/parsed/updrs3_on_parsed.csv')

    traj_covs_off = compute_time_varying_trajectory_covariates_parallel(updrs3_scores,
                                                                        window_years=9,
                                                                        flat_thr= 0.5,
                                                                        decline_thr=1,
                                                                        nonlinear_gap=2,
                                                                        min_points_per_window=8,
                                                                        grid_freq=4,
                                                                        n_samples=400,
                                                                        tune=500,
                                                                        pids='PATNO',
                                                                        values='updrs3_score',
                                                                        class_func=pos_flags_from_traj,
                                                                        verbose=5)
    traj_covs_off.to_csv('test_res/updrs3_off_covs.csv', index=False)

    traj_covs_on = compute_time_varying_trajectory_covariates_parallel(updrs3_on_scores,
                                                                        window_years=9,
                                                                        flat_thr=0.5,
                                                                        decline_thr=1,
                                                                        nonlinear_gap=2,
                                                                        min_points_per_window=8,
                                                                        grid_freq=4,
                                                                        n_samples=400,
                                                                        tune=500,
                                                                        pids='PATNO',
                                                                        values='updrs3_score_on',
                                                                        class_func=pos_flags_from_traj,
                                                                        verbose=5)
    traj_covs_on.to_csv('test_res/updrs3_on_covs.csv', index=False)

