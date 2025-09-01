# src/traj_ps/evaluation/diagnostics.py
from __future__ import annotations
import numpy as np
import arviz as az

def check_rhat(idata: az.InferenceData, threshold: float = 1.01) -> dict:
    rhat = az.rhat(idata, method="rank")
    over = np.array(rhat.to_array()) > threshold
    summary = {
        "max_rhat": float(np.nanmax(rhat.to_array())),
        "n_over": int(np.sum(over)),
        "threshold": threshold,
    }
    return summary

def ess_summary(idata: az.InferenceData) -> dict:
    ess = az.ess(idata, method="bulk")
    return {
        "min_ess": float(np.nanmin(ess.to_array())),
        "median_ess": float(np.nanmedian(ess.to_array())),
    }
