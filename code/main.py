import numpy as np
import pandas as pd
from bayesian_trajs import compute_time_varying_trajectory_covariates_parallel, pos_flags_from_traj

if __name__ == '__main__':
    updrs3_scores = pd.read_csv('../data/ppmi/parsed/updrs3_parsed.csv')
    updrs3_on_scores = pd.read_csv('../data/ppmi/parsed/updrs3_on_parsed.csv')

    traj_covs_off = compute_time_varying_trajectory_covariates_parallel(updrs3_scores,
                                                                        window_years=9,
                                                                        flat_thr=0.5,
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