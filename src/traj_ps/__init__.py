"""
Backward compatibility module: traj_ps → traj_features

This package has been renamed to `traj_features`.
All imports from `traj_ps` are forwarded to `traj_features` for backward compatibility.

Migration:
    OLD: from traj_ps.backends.bayes import BayesianTrajPS
    NEW: from traj_features.backends.bayes import BayesianTrajPS
"""

import warnings

warnings.warn(
    "traj_ps has been renamed to traj_features. "
    "Please update your imports. "
    "traj_ps will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Forward all imports to traj_features
from traj_features import *  # noqa: F401, F403
from traj_features import __version__  # noqa: F401

__all__ = ["interfaces"]
