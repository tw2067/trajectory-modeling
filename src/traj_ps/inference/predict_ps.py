import numpy as np, pandas as pd, torch

@torch.no_grad()
def predict_ps(model, loader, device=None, return_embeddings=False, group_by="time"):
    device = device or next(model.parameters()).device
    model.eval()
    rows=[]; Hmap = {} if return_embeddings else None
    for batch in loader:
        for k in ("X_raw","M_raw","DT_raw","STD_agg","Z","y_event","at_risk","idx_map"):
            batch[k]=batch[k].to(device)
        eta, H_raw, H_agg, mask_sel = model(batch["X_raw"], batch["M_raw"], batch["DT_raw"],
                                            batch["STD_agg"], batch["Z"], batch["idx_map"])
        eta_np = eta.cpu().numpy()
        risk_np= (batch["at_risk"]*mask_sel).cpu().numpy()
        for i,pid in enumerate(batch["pid"]):
            t = np.asarray(batch["agg_time"][i]); Ti=len(t)
            for k in range(Ti):
                rows.append({"patient_id":pid,"t_idx":k,"time":float(t[k]),
                             "eta":float(eta_np[i,k]), "at_risk":bool(risk_np[i,k])})
            if return_embeddings:
                Hmap[pid] = H_agg[i,:Ti,:].cpu().numpy()
    df = pd.DataFrame(rows)
    df["eta_masked"] = np.where(df["at_risk"], df["eta"], -np.inf)
    key = "time" if group_by=="time" else "t_idx"
    df["ps"]=0.0
    for _, idx in df.groupby(key).indices.items():
        v = df.loc[idx,"eta_masked"].to_numpy(float)
        if np.isfinite(v).any():
            m = np.max(v[np.isfinite(v)]); e = np.exp(np.where(np.isfinite(v), v-m, -np.inf)); s=e.sum()
            df.loc[idx,"ps"] = e/s if s>0 else 0.0
    return (df.drop(columns=["eta_masked"]).sort_values(["patient_id",key]).reset_index(drop=True),
            Hmap) if return_embeddings else df.drop(columns=["eta_masked"]).sort_values(["patient_id",key]).reset_index(drop=True)
