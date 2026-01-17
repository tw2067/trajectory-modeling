import torch, torch.nn as nn, torch.nn.functional as F

class GRUDCell(nn.Module):
    def __init__(self, p, h):
        super().__init__()
        self.h=h; self.p=p
        self.x_mean = nn.Parameter(torch.zeros(p), requires_grad=False)
        self.decay_h = nn.Linear(p, h)
        self.gru = nn.GRUCell(2*p, h)
    def forward(self, x, m, dt, h):
        delta = torch.exp(-F.softplus(dt))
        x_hat = m*x + (1-m)*self.x_mean
        x_dec = m*x + (1-m)*(delta*x_hat + (1-delta)*self.x_mean)
        h_til = torch.tanh(self.decay_h(x_hat))
        h = delta*h + (1-delta)*h_til
        h = self.gru(torch.cat([x_dec, m], dim=-1), h)
        return h

class DeepPSDual(nn.Module):
    def __init__(self, p_seq, p_std, p_static, h=64, head_hidden=64):
        super().__init__()
        self.cell = GRUDCell(p_seq, h)
        self.head = nn.Sequential(
            nn.Linear(h + p_std + p_static, head_hidden),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(head_hidden, 1, bias=False)
        )
        self.h=h; self.p_static=p_static

    def forward(self, X_raw, M_raw, DT_raw, STD_agg, Z, idx_map):
        B, Tr, p = X_raw.shape
        Ta = STD_agg.shape[1]
        h = X_raw.new_zeros(B, self.h)
        H_raw=[]
        for t in range(Tr):
            h = self.cell(X_raw[:,t], M_raw[:,t], DT_raw[:,t], h)
            H_raw.append(h)
        H_raw = torch.stack(H_raw, dim=1)  # (B,Tr,h)

        mask_sel = (idx_map >= 0)
        idx_safe = torch.clamp(idx_map, min=0)
        b_idx = torch.arange(B, device=X_raw.device)[:,None].expand(B,Ta)
        H_agg = H_raw[b_idx, idx_safe] * mask_sel.unsqueeze(-1)
        Zrep = Z[:,None,:].expand(-1,Ta,-1)
        eta = self.head(torch.cat([H_agg, STD_agg, Zrep], dim=-1)).squeeze(-1)
        return eta, H_raw, H_agg, mask_sel.float()

def cox_binned_partial_lik(eta, y_event, at_risk, eps=1e-8):
    exp_eta = torch.exp(eta)
    denom_t = (exp_eta * at_risk).sum(dim=0, keepdim=True)
    ll = (y_event * (eta - torch.log(denom_t + eps))).sum()
    return -(ll / y_event.sum().clamp_min(1.0))
