import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

############################# end-to-end version #####################################
# --------- GRU-D–style cell (handles mask & time gaps) ----------
class GRUDCell(nn.Module):
    def __init__(self, p, h):
        super().__init__()
        self.p, self.h = p, h
        self.x_mean = nn.Parameter(torch.zeros(p), requires_grad=False)
        self.decay_x = nn.Linear(p, p)   # to modulate imputation toward means
        self.decay_h = nn.Linear(p, h)   # to modulate hidden carry
        self.gru = nn.GRUCell(2*p, h)    # [x_imputed, mask] -> h

    def forward(self, x, m, dt, h):
        # dt: (B,1); m: (B,p) {0/1}; x: (B,p)
        # exponential decay with nonlinearity for stability
        delta = torch.exp(-F.softplus(dt))                # (B,1)
        # mean-impute missing values, then decay toward mean
        x_hat = m * x + (1 - m) * self.x_mean
        x_decay = m * x + (1 - m) * (delta * x_hat + (1 - delta) * self.x_mean)
        # hidden decay toward a learned bias from current observed pattern
        h_tilde = torch.tanh(self.decay_h(x_hat))         # (B,h)
        h = delta * h + (1 - delta) * h_tilde
        # GRU update with mask concatenated
        h = self.gru(torch.cat([x_decay, m], dim=-1), h)
        return h

# --------- full model: per-time embeddings + PS head ----------
class DeepPS(nn.Module):
    def __init__(self, p_seq, p_std, p_static, h=64, head_hidden=64):
        """
        p_seq   : # trajectory vars (labs/vitals) modeled by GRU-D
        p_std   : # time-varying 'standard' covariates (meds, raw labs, vitals,...)
        p_static: # static covariates (age, sex, CCI, etc.)
        """
        super().__init__()
        self.cell = GRUDCell(p_seq, h)
        self.mlp = nn.Sequential(
            nn.Linear(h + p_std + p_static, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, 1, bias=False)  # logit/log-hazard per time
        )

    def forward(self, X, M, DT, STD, Z):
        """
        X   : (B,T,p_seq)       trajectory inputs
        M   : (B,T,p_seq)       0/1 mask for X
        DT  : (B,T,1)           time since last obs (bin units)
        STD : (B,T,p_std)       additional time-varying covariates
        Z   : (B,p_static)      static covariates (broadcast across time)
        returns:
          eta   (B,T)           per-time score (logit/log-hazard)
          H     (B,T,h)         per-time trajectory embeddings (covariates you can export)
        """
        B,T,_ = X.shape
        h = X.new_zeros(B, self.cell.h)
        H = []
        for t in range(T):
            h = self.cell(X[:,t], M[:,t], DT[:,t], h)
            H.append(h)
        H = torch.stack(H, dim=1)                 # (B,T,h)
        Zrep = Z[:,None,:].expand(-1,T,-1)        # (B,T,p_static)
        eta = self.mlp(torch.cat([H, STD, Zrep], dim=-1)).squeeze(-1)  # (B,T)
        return eta, H

# --------- losses that use ALL time points while at risk ----------
def discrete_time_hazard_loss(logits, y_event, at_risk_mask, pos_weight=None):
    """
    logits       : (B,T) real-valued; sigmoid(logits)=hazard per bin
    y_event      : (B,T) 0/1; exactly one '1' at the bin where treatment starts (else all 0 if censored)
    at_risk_mask : (B,T) 0/1; 1 up to and including the event bin (or all observed bins if censored)
    pos_weight   : optional scalar to balance rare events
    """
    # BCEWithLogits with masking over at-risk person-time
    if pos_weight is None:
        loss = F.binary_cross_entropy_with_logits(logits, y_event, reduction='none')
    else:
        # weighted positive class if events are rare
        loss = F.binary_cross_entropy_with_logits(logits, y_event, reduction='none', pos_weight=torch.as_tensor(pos_weight, device=logits.device))
    return (loss * at_risk_mask).sum() / (at_risk_mask.sum().clamp_min(1.0))

def cox_binned_partial_lik(eta, y_event, at_risk_mask, eps=1e-8):
    """
    eta          : (B,T) score at each bin (acts like log-hazard)
    y_event      : (B,T) 0/1 event indicator (1 at the bin that starts treatment)
    at_risk_mask : (B,T) 0/1 risk set membership per bin (1 if untreated at start of bin)
    Implements:  - sum_{bins t with events} [ sum_i y_{it} * (eta_{it} - log sum_j exp(eta_{jt}) * R_{jt}) ]
    Fully vectorized over t.
    """
    exp_eta = torch.exp(eta)                                     # (B,T)
    denom_t = (exp_eta * at_risk_mask).sum(dim=0, keepdim=True)  # (1,T)
    # Only bins with at least one event contribute; y_event zeroes others
    ll = (y_event * (eta - torch.log(denom_t + eps))).sum()
    return -ll / (y_event.sum().clamp_min(1.0))                  # normalize by #events

# --------- utility: export per-time embeddings as covariates ----------
@torch.no_grad()
def export_time_covariates(model: DeepPS, batch):
    """
    Returns a dict with:
      'H'    : (B,T,h) deep trajectory covariates
      'eta'  : (B,T)   per-time score (for PS)
      'p_t'  : (B,T)   per-time hazard prob (if using discrete-time loss)
    """
    eta, H = model(batch['X'], batch['M'], batch['DT'], batch['STD'], batch['Z'])
    p_t = torch.sigmoid(eta)
    return {'H': H, 'eta': eta, 'p_t': p_t}

@torch.no_grad()
def predict_propensity_discrete(model, batch, to_cpu=True):
    eta, H = model(batch['X'], batch['M'], batch['DT'], batch['STD'], batch['Z'])
    p_t = torch.sigmoid(eta)                           # (B,T)
    # zero-out (or NaN) bins where the person is not at-risk
    p_t = p_t * batch['at_risk_mask']
    if to_cpu:
        p_t = p_t.cpu().numpy(); H = H.cpu().numpy()
    return p_t, H

@torch.no_grad()
def predict_propensity_cox_binned(model, batch):
    eta, H = model(batch['X'], batch['M'], batch['DT'], batch['STD'], batch['Z'])  # (B,T)
    # mask out non–at-risk person-time
    eta = eta.masked_fill(batch['at_risk_mask']==0, float('-inf'))
    # softmax across the *batch* per time bin -> risk-set probability at each t
    ps = torch.softmax(eta.transpose(0,1), dim=1).transpose(0,1)  # (B,T)
    return ps.cpu().numpy(), H.cpu().numpy()

@torch.no_grad()
def predict_propensity_cox_binned(model, batch):
    eta, H = model(batch['X'], batch['M'], batch['DT'], batch['STD'], batch['Z'])  # (B,T)
    # mask out non–at-risk person-time
    eta = eta.masked_fill(batch['at_risk_mask']==0, float('-inf'))
    # softmax across the *batch* per time bin -> risk-set probability at each t
    ps = torch.softmax(eta.transpose(0,1), dim=1).transpose(0,1)  # (B,T)
    return ps.cpu().numpy(), H.cpu().numpy()


def predict_ps(model, loader, device ,group_by="time", return_embeddings=False):
    model.eval()
    rows = []
    H_map = {} if return_embeddings else None
    for batch in loader:
        for k in ("X","M","DT","STD","Z","y_event","at_risk"):
            batch[k] = batch[k].to(device)
        eta, H = model(batch["X"], batch["M"], batch["DT"], batch["STD"], batch["Z"])  # (B,T)
        eta_np  = eta.cpu().detach().numpy()
        risk_np = batch["at_risk"].cpu().detach().numpy()

        for i, pid in enumerate(batch["pid"]):
            t_arr = np.asarray(batch["time"][i]); Ti = len(t_arr)
            for t_idx in range(Ti):
                rows.append({
                    "patient_id": pid,
                    "t_idx": t_idx,
                    "time": float(t_arr[t_idx]),
                    "eta": float(eta_np[i, t_idx]),
                    "at_risk": bool(risk_np[i, t_idx])
                })
            if return_embeddings:
                H_map[pid] = H[i, :Ti, :].cpu().detach().numpy()

    df = pd.DataFrame(rows)
    # mask non–at-risk -> -inf so softmax -> 0
    eta_masked = np.where(df["at_risk"].values, df["eta"].values, -np.inf)

    # softmax within each time bin across all at-risk subjects
    df["ps"] = 0.0
    key = "time" if group_by == "time" else "t_idx"
    for _, idx in df.groupby(key).indices.items():
        v = eta_masked[idx]
        if np.isfinite(v).any():
            m = np.max(v[np.isfinite(v)])
            e = np.exp(np.where(np.isfinite(v), v - m, -np.inf))
            s = e.sum()
            df.loc[df.index[idx], "ps"] = e / s if s > 0 else 0.0
        else:
            df.loc[df.index[idx], "ps"] = 0.0

    return (df.sort_values(["patient_id", key]).reset_index(drop=True),
            H_map) if return_embeddings else df.sort_values(["patient_id", key]).reset_index(drop=True)


def Hmap_to_df(H_map, time_map=None):
    """
    Convert H_map {pid: (T,h)} into long DataFrame.
    Optionally merge true time arrays if you have time_map {pid: array(T)}.
    """
    rows = []
    for pid, H in H_map.items():
        T, h = H.shape
        for t in range(T):
            row = {"patient_id": pid, "t_idx": t}
            # add time if available
            if time_map is not None:
                row["time"] = float(time_map[pid][t])
            # add embedding dims
            for j in range(h):
                row[f"H{j}"] = H[t, j]
            rows.append(row)
    return pd.DataFrame(rows)


def verify_batch_schema(batch):
    """Raise clear errors if a batch dict is missing keys or has wrong shapes/dtypes."""
    required = ["X","M","DT","STD","Z","y_event","at_risk","pid","time"]
    for k in required:
        if k not in batch:
            raise KeyError(f"Missing key '{k}' in batch")

    X, M, DT, STD, Z = batch["X"], batch["M"], batch["DT"], batch["STD"], batch["Z"]
    y_event, at_risk  = batch["y_event"], batch["at_risk"]

    # type checks
    for k,t in [("X",X),("M",M),("DT",DT),("STD",STD),("Z",Z),("y_event",y_event),("at_risk",at_risk)]:
        if not hasattr(t, "dtype"):
            raise TypeError(f"'{k}' must be a torch.Tensor")
        if t.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"'{k}' must be float tensor; got {t.dtype}")

    # shape checks
    B,T,p_seq = X.shape
    assert M.shape == (B,T,p_seq), f"M shape {M.shape} != {(B,T,p_seq)}"
    assert DT.shape == (B,T,1),    f"DT shape {DT.shape} != {(B,T,1)}"
    assert y_event.shape == (B,T), f"y_event shape {y_event.shape} != {(B,T)}"
    assert at_risk.shape  == (B,T),f"at_risk shape {at_risk.shape}  != {(B,T)}"
    assert Z.shape[0] == B and Z.ndim == 2, f"Z shape must be (B,p_static), got {Z.shape}"
    # STD can be (B,T,0) if you have no extra time-varying covariates:
    assert STD.shape[0] == B and STD.shape[1] == T, f"STD first two dims must be (B,T,·), got {STD.shape}"

    # pid/time lists
    if not isinstance(batch["pid"], (list, tuple)): raise TypeError("'pid' must be a list/tuple")
    if not isinstance(batch["time"], (list, tuple)): raise TypeError("'time' must be a list/tuple")
    if len(batch["pid"]) != B or len(batch["time"]) != B:
        raise ValueError("'pid'/'time' length must equal batch size")
    # per-item time arrays must have length ≤ T
    for i, t_arr in enumerate(batch["time"]):
        if len(t_arr) > T: raise ValueError(f"time[{i}] length {len(t_arr)} > padded T {T}")

    return True


################################ task agnostic ########################################
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


