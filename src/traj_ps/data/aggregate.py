import numpy as np, pandas as pd
DEFAULT_BIN_W = 1/12
DEFAULT_FEATURES = ("eGFR","Creatinine","SBP","DBP","HbA1c","MedA")

def make_bins(Tmax: float, bin_w: float = DEFAULT_BIN_W):
    n = int(np.ceil(Tmax/bin_w))
    edges = np.linspace(0.0, n*bin_w, n+1)
    return edges[:-1]
