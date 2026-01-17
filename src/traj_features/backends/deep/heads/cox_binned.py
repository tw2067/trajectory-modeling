# src/traj_ps/backends/deep/heads/cox_binned.py
from __future__ import annotations
import torch, torch.nn as nn

class CoxBinnedHead(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, 1)
        )
    def forward(self, X):  # X: [N, d_in]
        return self.net(X).squeeze(-1)  # eta

def cox_binned_partial_lik(eta, y_event, at_risk, eps: float = 1e-8):
    """
    eta:     [N] log-risk per interval
    y_event: [N] {0,1} event indicator within interval
    at_risk: [N] {0,1} risk-set membership in interval
    """
    # logsumexp over risk set
    eta = eta + torch.log(at_risk + eps)  # mask invalid intervals
    log_denom = torch.logsumexp(eta, dim=0)
    numer = (y_event * eta).sum()
    return -(numer - log_denom)
