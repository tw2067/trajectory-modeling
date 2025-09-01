import numpy as np, torch

def dual_pad_collate(batch):
    B=len(batch)
    Tr=max(len(b["raw_time"]) for b in batch)
    Ta=max(len(b["agg_time"]) for b in batch)
    p_seq=batch[0]["X_raw"].shape[1]
    p_std=batch[0]["STD_agg"].shape[1]

    def pad2(a, T, D, val=0.0):
        out = np.full((T,D), val, float); out[:a.shape[0],:a.shape[1]]=a; return out
    def pad1(a, T, val=0.0):
        out = np.full((T,), val, float); out[:a.shape[0]]=a; return out
    def pad_idx(a, T):
        out = np.full((T,), -1, int); out[:a.shape[0]]=a; return out

    Xr   = np.stack([pad2(b["X_raw"],  Tr, p_seq, 0.0) for b in batch])
    Mr   = np.stack([pad2(b["M_raw"],  Tr, p_seq, 0.0) for b in batch])
    DTr  = np.stack([pad2(b["DT_raw"], Tr, 1,     0.0) for b in batch])
    STDa = np.stack([pad2(b["STD_agg"],Ta, p_std, 0.0) for b in batch])
    Za   = np.stack([b["Z"] for b in batch])
    ye   = np.stack([pad1(b["y_event"], Ta, 0.0) for b in batch])
    ar   = np.stack([pad1(b["at_risk"], Ta, 0.0) for b in batch])
    idxm = np.stack([pad_idx(b["idx_map"], Ta) for b in batch])

    return {
        "pid": [b["pid"] for b in batch],
        "raw_time": [b["raw_time"] for b in batch],
        "agg_time": [b["agg_time"] for b in batch],
        "X_raw": torch.tensor(Xr, dtype=torch.float32),
        "M_raw": torch.tensor(Mr, dtype=torch.float32),
        "DT_raw":torch.tensor(DTr,dtype=torch.float32),
        "STD_agg":torch.tensor(STDa,dtype=torch.float32),
        "Z": torch.tensor(Za, dtype=torch.float32),
        "y_event": torch.tensor(ye, dtype=torch.float32),
        "at_risk": torch.tensor(ar, dtype=torch.float32),
        "idx_map": torch.tensor(idxm, dtype=torch.int64),
    }
