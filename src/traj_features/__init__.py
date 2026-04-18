"""
Trajectory Features: Extract temporal biomarker trajectories for clinical prediction.

Backends:
- Bayesian: MCMC-based posterior sampling (precise, slow)
- Bootstrap: Fast resampling with splines (10-100x faster)

Usage:
    from traj_features.backends.bootstrap import BootstrapTrajPS, BootstrapConfig
    from traj_features.backends.bayes import BayesianTrajPS, BayesConfig
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.2.0"

if TYPE_CHECKING:
    from .backends.bayes import BayesianTrajPS, BayesConfig
    from .backends.bootstrap import BootstrapTrajPS, BootstrapConfig

__all__ = [
    "BayesianTrajPS",
    "BayesConfig",
    "BootstrapTrajPS",
    "BootstrapConfig",
]


def __getattr__(name: str):
    if name in {"BayesianTrajPS", "BayesConfig"}:
        from .backends.bayes import BayesianTrajPS, BayesConfig
        return {"BayesianTrajPS": BayesianTrajPS, "BayesConfig": BayesConfig}[name]
    if name in {"BootstrapTrajPS", "BootstrapConfig"}:
        from .backends.bootstrap import BootstrapTrajPS, BootstrapConfig
        return {"BootstrapTrajPS": BootstrapTrajPS, "BootstrapConfig": BootstrapConfig}[name]
    raise AttributeError(name)

# Backward compatibility: make traj_ps an alias to traj_features
import sys
sys.modules['traj_ps'] = sys.modules[__name__]
