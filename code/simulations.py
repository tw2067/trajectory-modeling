import numpy as np
import pandas as pd
from bayesian_trajs import compute_time_varying_trajectory_covariates_parallel

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
    sim_labs = simulate_lab_data()
    traj_covs = compute_time_varying_trajectory_covariates_parallel(sim_labs, n_samples=200, n_jobs=-1, verbose=5)
    traj_covs.to_csv('test_res/traj_covs_sim.csv')
    print(traj_covs)
    # TODO: do simulations

