import pandas as pd
from pathlib import Path

paths = [
    '/home/gaga/data/physionet/mimic/sepsis/sepsis_prediction_dataset_with_bootstrap_probs.csv',
    '/home/gaga/data/physionet/mimic/liver/liver_prediction_dataset_with_bootstrap_probs.csv',
    '/home/gaga/data/physionet/mimic/ventilator/ventilator_prediction_dataset_with_bootstrap_probs.csv',
    '/home/gaga/data/physionet/eicu/sepsis/sepsis_prediction_dataset_with_bootstrap_probs.csv',
    '/home/gaga/data/physionet/eicu/liver/liver_prediction_dataset_with_bootstrap_probs.csv',
    '/home/gaga/data/physionet/eicu/ventilator/ventilator_prediction_dataset_with_bootstrap_probs.csv',
]

for p in paths:
    print(f"\n=== {p} ===")
    path = Path(p)
    if not path.exists():
        print("missing")
        continue

    cols = pd.read_csv(path, nrows=0).columns.tolist()
    traj = [c for c in cols if ('stable' in c or 'gradual' in c or 'rapid' in c or c.startswith('prob_'))]
    boot = [c for c in cols if 'boot' in c.lower()]
    print("traj-like:", traj[:15])
    print("boot:", boot[:15])
