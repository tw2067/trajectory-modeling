import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TVPropensityDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def pad_2d(arr, D, t_max, fill=0.0):
    T = arr.shape[0]
    out = np.full((t_max, D), fill, dtype=float)
    out[:T] = arr
    return out


def pad_collate(batch):
    """Pad to max T in batch. Padded steps are masked out (at_risk=0)."""
    B = len(batch)
    T_max = max(len(b["time"]) for b in batch)

    # infer dims
    p_seq  = batch[0]["X"].shape[1]
    p_std  = batch[0]["STD"].shape[1] if batch[0]["STD"].ndim==2 else 0
    p_stat = batch[0]["Z"].shape[0]

    X    = np.stack([pad_2d(b["X"], p_seq, T_max, 0.0) for b in batch])
    M    = np.stack([pad_2d(b["M"], p_seq, T_max, 0.0) for b in batch])
    DT   = np.stack([pad_2d(b["DT"], 1, T_max, 0.0) for b in batch])
    STD  = np.stack([pad_2d(b["STD"], p_std, T_max, 0.0) for b in batch]) if p_std>0 else np.zeros((B,T_max,0), float)
    Z    = np.stack([b["Z"] for b in batch])                                       # (B, p_static)
    yev  = np.stack([np.pad(b["y_event"].astype(float), (0, T_max-len(b["y_event"]))) for b in batch])
    risk = np.stack([np.pad(b["at_risk"].astype(float), (0, T_max-len(b["at_risk"]))) for b in batch])

    time = [b["time"] for b in batch]   # keep original (per patient) time arrays (length varies)
    pid  = [b["pid"]  for b in batch]

    out = {
        "X": torch.tensor(X,   dtype=torch.float32),
        "M": torch.tensor(M,   dtype=torch.float32),
        "DT":torch.tensor(DT,  dtype=torch.float32),
        "STD":torch.tensor(STD,dtype=torch.float32),
        "Z": torch.tensor(Z,   dtype=torch.float32),
        "y_event": torch.tensor(yev,  dtype=torch.float32),
        "at_risk": torch.tensor(risk, dtype=torch.float32),
        "pid": pid,
        "time": time,  # list of np arrays
    }
    return out


