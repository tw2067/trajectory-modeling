# Trajectory Features: Temporal Biomarker Trajectory Extraction

Extract rich temporal features from longitudinal biomarker data for clinical prediction tasks.

## Backends

- **Bayesian** (MCMC): Full posterior distributions via spline-based Bayesian sampling (slow, precise)
- **Bootstrap**: Fast resampling-based trajectories with splines (10-100x faster, comparable accuracy)

## Core Functionality

Classify longitudinal biomarker patterns into trajectory types:
- **Stable/Nonprogression**: Minimal change over time
- **Linear increase/decline**: Consistent slope
- **Nonlinear/Rapid change**: Accelerating patterns

## Installation

```bash
# Bayesian backend (MCMC)
pip install -e .[bayes]

# Bootstrap backend (fast)
pip install -e .[bootstrap]

# Evaluation tools
pip install -e .[evaluation]

# Everything
pip install -e .[all]
```

## Quick Start

### Extract Trajectory Features

```python
from traj_features.backends.bootstrap import BootstrapTrajPS, BootstrapConfig
from traj_features.backends.bayes.classify import pos_flags_from_traj

# Configure (example: P/F ratio improvement trajectories)
config = BootstrapConfig(
    window_years=3.0,
    n_bootstrap=500,
    flat_thr=10.0,
    decline_thr=30.0,
    class_func=pos_flags_from_traj
)

# Extract features
model = BootstrapTrajPS(cfg=config)
trajectory_probs = model.embed(longitudinal_data)
# Returns: DataFrame with prob_stable, prob_gradual_improvement, prob_rapid_improvement
```

### Clinical Dataset Examples

```bash
# AKI (serum creatinine trajectories)
python scripts/hirid_aki_trajs.py --backend bootstrap

# Sepsis (lactate, WBC, platelets)
python scripts/hirid_sepsis_trajs.py --backend bootstrap

# Ventilator weaning (P/F ratio)
python scripts/hirid_ventilator_trajs.py --backend bootstrap

# Compare backends
python scripts/compare_bootstrap_vs_bayes.py
```

## `.gitignore`
```gitignore
__pycache__/
*.pyc
.venv/
.env
/data/
logs/
*.pt
```
