import torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRUDCell(nn.Module):
    def __init__(self, p, h):
        super().__init__()
        self.gamma_x = nn.Linear(p, p)      # decay for inputs
        self.gamma_h = nn.Linear(p, h)      # decay for hidden
        self.gru = nn.GRUCell(p*2, h)       # [imputed x || mask] -> h

        self.register_parameter('x_mean', nn.Parameter(torch.zeros(p), requires_grad=False))

    def forward(self, x, m, dt, h):
        # exponential decays as in GRU-D
        delta = torch.exp(-torch.relu(dt))             # (B,1) or (B,p)
        x_hat = m * x + (1 - m) * (self.x_mean)        # mean-impute missing
        x_decay = m * x + (1 - m) * (delta * x_hat + (1 - delta) * self.x_mean)
        h = delta * h + (1 - delta) * torch.tanh(self.gamma_h(x_hat))
        h = self.gru(torch.cat([x_decay, m], dim=-1), h)
        return h

class CoxRNN(nn.Module):
    def __init__(self, p, h, s):
        super().__init__()
        self.cell = GRUDCell(p, h)
        self.head = nn.Sequential(nn.Linear(h + s, 1, bias=False))  # log-hazard score η_t

    def forward(self, X, M, DT, L, S):
        """
        X,M,DT: padded (B,T,p), (B,T,p), (B,T,1); L: lengths (B,); S: static (B,s)
        Returns η_t for each time step (masked beyond length).
        """
        B,T,p = X.shape
        h = torch.zeros(B, self.cell.gru.hidden_size, device=X.device)
        etas = []
        for t in range(T):
            h = self.cell(X[:,t], M[:,t], DT[:,t], h)
            hs = torch.cat([h, S], dim=-1)
            etas.append(self.head(hs))        # (B,1)
        eta = torch.cat(etas, dim=1)          # (B,T)
        # mask padded steps
        mask = torch.arange(T, device=X.device)[None,:] < L[:,None]
        return eta, mask

def cox_partial_ll(eta, times, events, mask):
    """
    eta: (B,T) score at each observed time (use last step per subject or per-unique event time)
    times: (B,) event/censor time index; events: (B,) {0,1}; mask: (B,T)
    This is a compact discrete approximation; for exact continuous-time you’d align risk sets by calendar time.
    """
    # take the last valid eta per subject
    idx = (torch.arange(eta.size(1))[None,:].to(eta.device) == (times[:,None]-1))
    eta_last = (eta * idx).sum(dim=1)  # (B,)
    # risk set: everyone with time >= event time
    # build matrix of risk membership
    T = eta.size(1)
    t_event = times - 1
    risk = (t_event[:,None] <= torch.arange(T, device=eta.device)[None,:]).float()
    # sum exp(eta) over risk sets at their last step
    exp_eta = torch.exp(eta)
    denom = (exp_eta * risk).sum(dim=1) + 1e-8
    ll = (events * (eta_last - torch.log(denom))).sum()
    return -ll


class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, d=1, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=d*(k-1), dilation=d),
            nn.ReLU(),
            nn.BatchNorm1d(c_out),
            nn.Dropout(p),
            nn.Conv1d(c_out, c_out, k, padding=d*(k-1), dilation=d),
            nn.ReLU(),
            nn.BatchNorm1d(c_out),
        )
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in!=c_out else nn.Identity()
    def forward(self, x):  # x: (B,C,T)
        y = self.net(x)
        return y + self.down(x)

class CoxTCN(nn.Module):
    def __init__(self, p, s, h=64, blocks=3):
        super().__init__()
        layers = []
        c = p
        for b in range(blocks):
            layers += [TCNBlock(c, h, d=2**b, p=0.1)]
            c = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(h + s, 1, bias=False)

    def forward(self, X, S):  # X: (B,T,p) -> (B,C=p,T)
        Xc = X.transpose(1,2)
        H = self.backbone(Xc).transpose(1,2)    # (B,T,h)
        # per-time hazard score
        Srep = S[:,None,:].expand(-1,H.size(1),-1)
        eta = self.head(torch.cat([H, Srep], dim=-1)).squeeze(-1)  # (B,T)
        return eta
