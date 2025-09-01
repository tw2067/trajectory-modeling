import numpy as np
import pandas as pd
from bayesian_trajs import compute_time_varying_trajectory_covariates_parallel, pos_flags_from_traj, mmse_flag_from_traj
from gam_trajs import compute_gam_beta_covariates_adaptive_parallel

if __name__ == '__main__':
    mmse_scores = pd.read_csv('../data/adni/parsed/mmse_parsed.csv')

    traj_covs_mmse = compute_time_varying_trajectory_covariates_parallel(mmse_scores,
                                                                        window_years=9,
                                                                        flat_thr=-0.3,
                                                                        decline_thr=-1.5,
                                                                        nonlinear_gap=1,
                                                                        min_points_per_window=8,
                                                                        grid_freq=4,
                                                                        n_samples=400,
                                                                        tune=500,
                                                                        pids='PTID',
                                                                        values='MMSE',
                                                                        class_func=mmse_flag_from_traj,
                                                                        traj_types=('prolonged_nonprogression',
                                                                                    'slow_linear_decline',
                                                                                    'fast_linear_decline',
                                                                                    'nonlinear'),
                                                                        verbose=5)
    traj_covs_mmse.to_csv('test_res/mmse_covs.csv', index=False)

    gam_betas_mmse = compute_gam_beta_covariates_adaptive_parallel(mmse_scores,
                                                                 window_years=9,
                                                                 min_points_per_window=8,
                                                                 labs=['MMSE'],
                                                                 pids='PTID')
    gam_betas_mmse.to_csv('test_res/gam_betas_mmse.csv', index=False)


    # updrs3_scores = pd.read_csv('../data/ppmi/parsed/updrs3_parsed.csv')
    # updrs3_on_scores = pd.read_csv('../data/ppmi/parsed/updrs3_on_parsed.csv')

    # gam_betas_off = compute_gam_beta_covariates_adaptive_parallel(updrs3_scores,
    #                                                               window_years=9,
    #                                                               min_points_per_window=8,
    #                                                               labs=['updrs3_score'],
    #                                                               pids='PATNO')
    # gam_betas_off.to_csv('test_res/gam_betas_off.csv', index=False)
    #
    # gam_betas_on = compute_gam_beta_covariates_adaptive_parallel(updrs3_on_scores,
    #                                                              window_years=9,
    #                                                              min_points_per_window=8,
    #                                                              labs=['updrs3_score_on'],
    #                                                              pids='PATNO')
    # gam_betas_on.to_csv('test_res/gam_betas_on.csv', index=False)


    # traj_covs_off = compute_time_varying_trajectory_covariates_parallel(updrs3_scores,
    #                                                                     window_years=9,
    #                                                                     flat_thr=0.5,
    #                                                                     decline_thr=1,
    #                                                                     nonlinear_gap=2,
    #                                                                     min_points_per_window=8,
    #                                                                     grid_freq=4,
    #                                                                     n_samples=400,
    #                                                                     tune=500,
    #                                                                     pids='PATNO',
    #                                                                     values='updrs3_score',
    #                                                                     class_func=pos_flags_from_traj,
    #                                                                     verbose=5)
    # traj_covs_off.to_csv('test_res/updrs3_off_covs.csv', index=False)
    #
    # traj_covs_on = compute_time_varying_trajectory_covariates_parallel(updrs3_on_scores,
    #                                                                    window_years=9,
    #                                                                    flat_thr=0.5,
    #                                                                    decline_thr=1,
    #                                                                    nonlinear_gap=2,
    #                                                                    min_points_per_window=8,
    #                                                                    grid_freq=4,
    #                                                                    n_samples=400,
    #                                                                    tune=500,
    #                                                                    pids='PATNO',
    #                                                                    values='updrs3_score_on',
    #                                                                    class_func=pos_flags_from_traj,
    #                                                                    verbose=5)
    # traj_covs_on.to_csv('test_res/updrs3_on_covs.csv', index=False)