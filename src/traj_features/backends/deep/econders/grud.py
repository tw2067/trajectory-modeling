# src/traj_ps/backends/deep/encoders/grud.py
from __future__ import annotations
import torch, torch.nn as nn

class GRUImputeEncoder(nn.Module):
    """
    A light encoder: concatenate [x, mask, delta] → GRU → hidden.
    Shapes:
      x:     [B, T, D]
      mask:  [B, T, D]   (1 if observed else 0)
      delta: [B, T, 1]   (time since last observation, precomputed)
    """
    def __init__(self, d_in: int, h: int):
        super().__init__()
        self.gru = nn.GRU(input_size=d_in*2 + 1, hidden_size=h, batch_first=True)

    def forward(self, x, mask, delta):
        z = torch.cat([x, mask, delta], dim=-1)  # [B,T,2D+1]
        H, _ = self.gru(z)
        return H  # [B,T,h]
