# src/traj_ps/evaluation/benchmarks.py
from __future__ import annotations
import numpy as np, pandas as pd
from lifelines.utils import concordance_index

def concordance_by_interval(df: pd.DataFrame) -> float:
    """
    df: counting-process rows with columns ['pid','start','stop','treatment','ps'].
    We compute a simple c-index using stop times and ps as risk score.
    """
    if not {"stop","ps","treatment"}.issubset(df.columns):
        raise ValueError("df must have columns stop, ps, treatment")
    return float(concordance_index(event_times=df["stop"].values,
                                   predicted_scores=df["ps"].values,
                                   event_observed=df["treatment"].values))

def brier_score_binary(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    y_true = np.asarray(y_true, float); p_hat = np.asarray(p_hat, float)
    return float(np.mean((p_hat - y_true) ** 2))
