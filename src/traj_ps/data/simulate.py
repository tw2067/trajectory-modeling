import numpy as np, pandas as pd
DEFAULT_FEATURES = ("eGFR","Creatinine","SBP","DBP","HbA1c","MedA")

def simulate_dynamic_static(n_pat=120, features=DEFAULT_FEATURES):
    rows_dyn, rows_sta = [], []
    rng = np.random.default_rng(920)
    for i in range(n_pat):
        pid = f"P{i:05d}"
        T = rng.uniform(2.0, 4.0)
        age, sex, cci = int(rng.integers(45,85)), int(rng.integers(0,2)), int(rng.poisson(2))
        rows_sta.append((pid, age, sex, cci, float(T)))

        def pp(rate):
            t, out = .0, []
            while True:
                t += rng.exponential(1/max(rate,1e-6))
                if t >= T: break
                out.append(t)
            return np.array(out) if out else np.array([rng.uniform(.1,T)])
        t_e, t_h, t_s, t_m = pp(6), pp(5), pp(4), pp(3)

        eg = np.clip(90 - 3.0*t_e + rng.normal(0,3,size=t_e.size), 5, 200)
        hb = np.clip(7.5 + 0.1*t_h + 0.2*np.sin(2*np.pi*t_h) + rng.normal(0,.3,size=t_h.size), 5, 14)
        sb = np.clip(120 + 5*np.sin(2*np.pi*t_s) + rng.normal(0,8,size=t_s.size), 90, 200)
        med=0
        for tm in t_m:
            if rng.random()<0.4: med=1-med
            rows_dyn.append((pid, float(tm), "MedA", float(med), 0))
        for t,v in zip(t_e,eg): rows_dyn.append((pid,float(t),"eGFR",float(v),0))
        for t,v in zip(t_h,hb): rows_dyn.append((pid,float(t),"HbA1c",float(v),0))
        for t,v in zip(t_s,sb): rows_dyn.append((pid,float(t),"SBP",float(v),0))

        grid = np.arange(0.0, T, 0.1)
        # LOCF approximations
        def locf(tt, vv, g):
            if tt.size==0: return np.zeros_like(g)
            idx = np.searchsorted(tt, g, side="right")-1; idx[idx<0]=0
            return vv[np.clip(idx,0,vv.size-1)]
        eg_loc = locf(t_e, eg, grid)
        score = -3.2 + 0.04*np.maximum(0, 90-eg_loc) + 0.2*sex + 0.05*cci
        p = 1 - np.exp(-np.exp(score))
        treated=0; t_start=None
        for gg,pp in zip(grid,p):
            if treated==0 and rng.random()<pp:
                treated=1; t_start=float(gg); break
        if t_start is not None:
            for k,(ppid,tt,fn,val,tr) in enumerate(rows_dyn):
                if ppid==pid and tt>=t_start:
                    rows_dyn[k]=(ppid,tt,fn,val,1)

    dynamic = pd.DataFrame(rows_dyn, columns=["pid","time","feature_name","value","treated"]) \
                .sort_values(["pid","time","feature_name"]).reset_index(drop=True)
    static  = pd.DataFrame(rows_sta, columns=["pid","age","sex","cci","Tmax"]).sort_values("pid")
    return dynamic, static
