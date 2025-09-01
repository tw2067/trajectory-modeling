from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from lifelines import CoxTimeVaryingFitter

from .features import compute_gam_beta_covariates_adaptive_parallel, pca_reduce_gam_betas
from .align import align_gam_betas_to_bins

@dataclass
class GAMConfig:
    window_years: float = 2.0
    labs: list[str] | None = None        # columns in wide lab table (besides pid/time)
    n_splines_range: tuple[int,int] = (4,6)
    lam_grid: tuple[float,...] = (0.3, 1.0, 3.0)
    min_points_per_window: int = 6
    standardize_y: bool = True
    cv_splits: int = 0
    max_iter: int = 5000
    spline_order: int = 3
    n_jobs: int = -1
    verbose: int = 5
    # PCA (optional compression of beta blocks)
    pca_components: int | None = None

class GAMTrajPS:
    name = "gam"
    def __init__(self, cfg: GAMConfig | None = None):
        self.cfg = cfg or GAMConfig()
        self.ctv_: CoxTimeVaryingFitter | None = None

    # Step 1: extract time-varying covariates (GAM betas on sliding windows)
    def embed(self, lab_wide_df: pd.DataFrame) -> pd.DataFrame:
        """
        lab_wide_df columns: ['patient_id','time', <lab1>, <lab2>, ...]
        Returns: ['patient_id','time', <lab>_beta_*, <lab>_n_splines, <lab>_lam, ...]
        """
        cov = compute_gam_beta_covariates_adaptive_parallel(
            lab_df=lab_wide_df,
            window_years=self.cfg.window_years,
            labs=self.cfg.labs,
            n_splines_range=self.cfg.n_splines_range,
            lam_grid=self.cfg.lam_grid,
            min_points_per_window=self.cfg.min_points_per_window,
            standardize_y=self.cfg.standardize_y,
            cv_splits=self.cfg.cv_splits,
            max_iter=self.cfg.max_iter,
            spline_order=self.cfg.spline_order,
            n_jobs=self.cfg.n_jobs,
            verbose=self.cfg.verbose,
            pids="patient_id",
            time_col="time",
            robust=True,  # use robust fallback path by default
        )
        if self.cfg.pca_components:
            cov = pca_reduce_gam_betas(cov, labs=self.cfg.labs or
                                       sorted(set(c.split("_beta_")[0] for c in cov.columns if "_beta_" in c)),
                                       n_components=self.cfg.pca_components)
        return cov

    # Step 2: align those covariates to PS bin starts (LOCF)
    def align_to_bins(self, gam_cov_df: pd.DataFrame, bin_map: dict[str, list[float]]) -> pd.DataFrame:
        return align_gam_betas_to_bins(gam_cov_df, bin_map, pid_col="patient_id", time_col="time")

    # Step 3: fit Cox TV PS on the counting-process design
    def fit(self, counting_process_df: pd.DataFrame) -> "GAMTrajPS":
        ctv = CoxTimeVaryingFitter()
        ctv.fit(counting_process_df, id_col="pid", start_col="start", stop_col="stop", event_col="treatment")
        self.ctv_ = ctv
        return self

    # Step 4: score PS
    def ps(self, counting_process_df: pd.DataFrame) -> pd.DataFrame:
        if self.ctv_ is None:
            raise RuntimeError("Call fit(...) first.")
        out = counting_process_df.copy()
        out["ps"] = self.ctv_.predict_partial_hazard(out)
        return out

    # (optional) persistence
    def save(self, path: str) -> None:
        if self.ctv_ is not None:
            self.ctv_.save(path)
    @classmethod
    def load(cls, path: str) -> "GAMTrajPS":
        obj = cls(); ctv = CoxTimeVaryingFitter(); ctv.load(path); obj.ctv_ = ctv; return obj
