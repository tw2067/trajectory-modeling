"""
Trajectory Features: Extract temporal biomarker trajectories for clinical prediction.

Backends:
- Bayesian: MCMC-based posterior sampling (precise, slow)
- Bootstrap: Fast resampling with splines (10-100x faster)

Usage:
    from traj_features.backends.bootstrap import BootstrapTrajPS, BootstrapConfig
    from traj_features.backends.bayes import BayesianTrajPS, BayesConfig
"""

__version__ = "0.2.0"

# Import main classes for convenience
from .backends.bayes import BayesianTrajPS, BayesConfig
from .backends.bootstrap import BootstrapTrajPS, BootstrapConfig

__all__ = [
    "BayesianTrajPS",
    "BayesConfig",
    "BootstrapTrajPS",
    "BootstrapConfig",
]

# Backward compatibility: make traj_ps an alias to traj_features
import sys
sys.modules['traj_ps'] = sys.modules[__name__]
