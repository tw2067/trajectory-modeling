# src/traj_ps/backends/deep/encoders/meanpool.py
from __future__ import annotations
import torch, torch.nn as nn

class MeanPool(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, H, mask=None):
        if mask is None: return H.mean(dim=1)
        w = mask.float().mean(dim=-1, keepdim=True)  # [B,T,1]
        w = torch.clamp(w, min=1e-6)
        return (H * w).sum(dim=1) / w.sum(dim=1)
