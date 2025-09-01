# src/traj_ps/inference/export_embed.py
from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict

def embedding_map_to_df(H_map: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows=[]
    for pid, H in H_map.items():
        T, D = H.shape
        for t in range(T):
            rows.append({"pid": pid, "t_idx": t, **{f"h{j+1}": float(H[t,j]) for j in range(D)}})
    return pd.DataFrame(rows)

def save_embeddings(H_map: Dict[str, np.ndarray], path: str) -> None:
    embedding_map_to_df(H_map).to_parquet(path, index=False)
