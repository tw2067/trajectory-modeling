import argparse, os, pandas as pd, yaml
from traj_ps.data.simulate import simulate_dynamic_static
from traj_ps.data.dual_prep import prepare_samples_dual
from traj_ps.data.aggregate import DEFAULT_FEATURES

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--config", type=str, default="configs/data.yaml")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    dynamic, static = simulate_dynamic_static(
        n_pat=cfg.get("n_patients", 120),
        features=DEFAULT_FEATURES
    )
    dynamic.to_parquet(os.path.join(args.out, "dynamic.parquet"))
    static.to_parquet(os.path.join(args.out,  "static.parquet"))
    print("Wrote dynamic/static to", args.out)

if __name__ == "__main__":
    main()
