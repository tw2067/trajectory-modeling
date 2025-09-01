# src/traj_ps/config.py
from __future__ import annotations
import os, yaml
from dataclasses import dataclass, asdict
from typing import Dict, Any

def load_yaml(path: str | None, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Load YAML if it exists; otherwise return fallback dict copy."""
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return dict(fallback)

@dataclass
class DataDefaults:
    seed: int = 920
    n_patients: int = 120
    bin_width: float = 1/12  # monthly
    embed_features: tuple[str,...] = ("eGFR","HbA1c")
    agg_features:   tuple[str,...] = ("SBP","MedA")

def load_configs(data_cfg: str | None, train_cfg: str | None,
                 data_fallback: Dict[str, Any] | None = None,
                 train_fallback: Dict[str, Any] | None = None):
    data = load_yaml(data_cfg, fallback=(data_fallback or asdict(DataDefaults())))
    train = load_yaml(train_cfg, fallback=(train_fallback or {}))
    return data, train
