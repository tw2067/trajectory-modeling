import numpy as np
import torch
from deep_trajs import *
from deep_trajs_preprocessing import *



def _run_epoch(loader, model, device, opt, train=True):
    model.train(train)
    total = 0.0; n = 0
    for batch in loader:
        verify_batch_schema(batch)
        for k in ("X","M","DT","STD","Z","y_event","at_risk"):
            batch[k] = batch[k].to(device)
        eta, H = model(batch["X"], batch["M"], batch["DT"], batch["STD"], batch["Z"])  # (B,T), (B,T,h)
        loss = cox_binned_partial_lik(eta, batch["y_event"], batch["at_risk"])
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        total += float(loss.item()); n += 1
    return total / max(1,n)

def training_loop(train_loader, val_loader, model, device, opt, n_epochs):
    tr_hist, val_hist = [], []
    for epoch in range(n_epochs):
        tr = _run_epoch(train_loader, model, device, opt,True)
        va = _run_epoch(val_loader, model, device, opt,False)
        tr_hist.append(tr)
        val_hist.append(va)
        print(f"epoch {epoch + 1:02d}  train {tr:.4f}  valid {va:.4f}")

    return tr_hist, val_hist



# # training loop
# model = DeepPS(p_seq=P_seq, p_std=P_std, p_static=P_static, h=64, head_hidden=64).to(device)
# opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
#
# for batch in loader:
#     for k in batch: batch[k] = batch[k].to(device)
#     eta, H = model(batch['X'], batch['M'], batch['DT'], batch['STD'], batch['Z'])
#     loss = discrete_time_hazard_loss(eta, batch['y_event'], batch['at_risk_mask'])
#     opt.zero_grad(); loss.backward(); opt.step()