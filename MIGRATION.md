# Migration Guide: traj_ps → traj_features

## Overview

The project has been refactored to focus on **trajectory feature extraction** for prediction tasks, removing the propensity score emphasis.

## What Changed

### Package Rename
- **Old**: `traj_ps`
- **New**: `traj_features`

### Focus Shift
- **Before**: Trajectory-based propensity scores (treatment modeling)
- **After**: Temporal biomarker trajectory features (general prediction)

### New Backends
- ✅ **Bayesian** (MCMC): Retained from original
- ✅ **Bootstrap** (NEW): Fast resampling alternative (10-100x faster)
- ❌ **Deep** (GRU-D): Removed (out of scope)
- ❌ **GAM**: Removed (use Bootstrap instead)

## Backward Compatibility

**Phase 1** (Current): Package supports both names
```python
# Both work
from traj_features.backends.bayes import BayesianTrajPS
from traj_ps.backends.bayes import BayesianTrajPS  # Still works via alias
```

**Phase 2** (Future): `traj_ps` deprecated with warnings

**Phase 3** (Final): `traj_ps` removed

## Migration Steps

### 1. Update Imports

**Old:**
```python
from traj_ps.backends.bayes import BayesianTrajPS, BayesConfig
```

**New:**
```python
from traj_features.backends.bayes import BayesianTrajPS, BayesConfig
from traj_features.backends.bootstrap import BootstrapTrajPS, BootstrapConfig
```

### 2. Update Installation

**Old:**
```bash
pip install -e .[bayes,deep,gam]
```

**New:**
```bash
pip install -e .[bayes,bootstrap,evaluation]
```

### 3. Update Config Files

**Old configs:** `configs/bayes.yaml`, `configs/deep.yaml`, `configs/gam.yaml`

**New configs:** `configs/bayes.yaml`, `configs/bootstrap.yaml`

### 4. Update Scripts

Replace PS-specific scripts with trajectory extraction:

**Old:**
```bash
python scripts/train.py --backend bayes
python scripts/predict_ps.py --backend bayes
```

**New:**
```bash
python scripts/hirid_ventilator_trajs.py --backend bootstrap
python scripts/evaluate_trajectory_methods.py
```

## New Features

### 1. Bootstrap Backend
Fast alternative to Bayesian MCMC:
```python
from traj_features.backends.bootstrap import BootstrapTrajPS, BootstrapConfig

config = BootstrapConfig(
    window_years=3.0,
    n_bootstrap=500,
    flat_thr=10.0,
    decline_thr=30.0
)

model = BootstrapTrajPS(cfg=config)
features = model.embed(data)  # Much faster!
```

### 2. Evaluation Framework
Compare methods and evaluate prediction performance:
```python
from traj_features.evaluation import compare_trajectory_assignments

results = compare_trajectory_assignments(
    bayesian_probs,
    bootstrap_probs
)
```

```bash
python scripts/evaluate_trajectory_methods.py \
    --bayesian-probs results/bayes.csv \
    --bootstrap-probs results/bootstrap.csv \
    --outcomes data/outcomes.csv
```

### 3. Clinical Dataset Pipelines
Standardized extraction scripts:
- `hirid_aki_trajs.py` (AKI prediction)
- `hirid_sepsis_trajs.py` (Sepsis trajectories)
- `hirid_liver_trajs.py` (Liver failure)
- `hirid_ventilator_trajs.py` (Ventilator weaning)

## Removed Features

### Deep Learning Backend
- **Removed**: GRU-D, TCN, Transformer models
- **Reason**: Out of scope for trajectory feature extraction
- **Alternative**: Use extracted features with your own deep learning models

### GAM Backend
- **Removed**: GAM spline fitting
- **Reason**: Superseded by Bootstrap backend
- **Alternative**: Use Bootstrap (faster, more flexible)

### Propensity Score Modeling
- **Removed**: Direct Cox model integration for treatment propensity
- **Reason**: Too specific; trajectories useful for many tasks
- **Alternative**: Use trajectory features in any downstream model

## Quick Reference

### Common Tasks

**Extract features (fast):**
```python
from traj_features.backends.bootstrap import BootstrapTrajPS, BootstrapConfig
model = BootstrapTrajPS(BootstrapConfig(window_years=3.0, n_bootstrap=500))
features = model.embed(data)
```

**Extract features (precise):**
```python
from traj_features.backends.bayes import BayesianTrajPS, BayesConfig
model = BayesianTrajPS(BayesConfig(window_years=3.0, n_samples=200))
features = model.embed(data)
```

**Compare methods:**
```bash
python scripts/compare_bootstrap_vs_bayes.py
```

**Evaluate on clinical task:**
```bash
python scripts/evaluate_trajectory_methods.py \
    --bayesian-probs results/bayes.csv \
    --bootstrap-probs results/bootstrap.csv \
    --outcomes data/outcomes.csv
```

## Questions?

See updated `README.md` for full documentation.
