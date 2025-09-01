import numpy as np, pandas as pd
from .aggregate import make_bins, DEFAULT_BIN_W

def build_targets_from_treated(treated_series, include_event_bin=True):
    T=len(treated_series); y=np.zeros(T,float); r=np.zeros(T,float)
    treated_series = np.maximum.accumulate(np.asarray(treated_series, int))
    idx = np.flatnonzero(treated_series==1)
    if idx.size==0: r[:]=1.0; return y,r
    e=int(idx[0]); y[e]=1.0; r[:e+(1 if include_event_bin else 0)]=1.0; return y,r

def prepare_samples_dual(dynamic, static, embed_feats, agg_feats, bin_w=DEFAULT_BIN_W):
    samples=[]
    for pid, gpid in dynamic.groupby("pid"):
        Tmax = float(static.loc[static.pid==pid, "Tmax"].iloc[0])
        agg_time = make_bins(Tmax, bin_w)
        # treated at bin start:
        t1 = gpid.loc[gpid.treated==1,"time"]
        t_start = float(t1.min()) if not t1.empty else None
        treated_bins = (agg_time >= (t_start if t_start is not None else 1e18)).astype(int)
        y_event, at_risk = build_targets_from_treated(treated_bins, True)

        # STD (simple LOCF for meds, mean-in-bin for others)
        STD = np.zeros((len(agg_time), len(agg_feats)), float)
        for j,feat in enumerate(agg_feats):
            g = gpid[gpid.feature_name==feat].sort_values("time")
            t = g["time"].to_numpy(float); v = g["value"].to_numpy(float)
            if t.size==0:
                STD[:,j]=0.0; continue
            idx = np.searchsorted(t, agg_time, side="right")-1; idx[idx<0]=0
            STD[:,j]=v[np.clip(idx,0,len(v)-1)]

        # raw union times for embed feats
        raw_times=[]
        per={}
        for f in embed_feats:
            gf=gpid[gpid.feature_name==f].sort_values("time")
            tf=gf["time"].to_numpy(float); vf=gf["value"].to_numpy(float)
            per[f]=(tf,vf); raw_times.append(tf)
        raw_time = np.unique(np.concatenate([rt for rt in raw_times if rt.size>0])) if len(raw_times)>0 else np.array([0.0])

        p_seq = len(embed_feats)
        X=np.zeros((len(raw_time), p_seq), float)
        M=np.zeros_like(X); DT=np.ones((len(raw_time),1), float)
        for j,f in enumerate(embed_feats):
            tf,vf = per[f]
            if tf.size==0: continue
            pos = np.searchsorted(tf, raw_time, side="right")-1
            for k,p in enumerate(pos):
                if p>=0:
                    X[k,j]=vf[p]; M[k,j]=1.0; DT[k,0]=max(1e-6, raw_time[k]-tf[p])

        idx_map = np.searchsorted(raw_time, agg_time, side="right")-1
        idx_map[idx_map<0] = -1
        Z = static.loc[static.pid==pid, ["age","sex","cci"]].iloc[0].to_numpy(float)

        samples.append({
            "pid": pid,
            "raw_time": raw_time.astype(float),
            "agg_time": agg_time.astype(float),
            "X_raw": X.astype(float),
            "M_raw": M.astype(float),
            "DT_raw":DT.astype(float),
            "STD_agg":STD.astype(float),
            "Z": Z,
            "y_event": y_event.astype(float),
            "at_risk": at_risk.astype(float),
            "idx_map": idx_map.astype(int),
        })
    return samples
