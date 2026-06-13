# Trajectory Modeling — Improvement Plan

**Status:** In progress
**Last updated:** 2026-06-13

---

## Planned Steps

| # | Area | Task | Status |
|---|------|------|--------|
| 1 | Bayes bug | Fix numpyro/JAX sampler overwrite bug in `sampling.py` | ✅ 0273f35 |
| 2 | Dead code | Remove temp files, untrack `__pycache__`, dedup launcher generator | ✅ 5b35c1c |
| 3 | Preprocessing | Fix deprecated `fillna(method=)` with `.ffill()` (9 files) | ✅ d74cd55 |
| 4 | Organization | Move `src/eicu_loader.py` → `scripts/eicu_preprocessing/` | ✅ 255d30f |
| 5 | Bayes perf | Add `batch_size`/`chain_method` to `BayesConfig`; auto-select nutpie | ✅ 2584306 |
| 6 | Testing | Smoke test suite (`tests/test_smoke_backends.py`, 15 tests) | ✅ 888ae51 |
| 7 | Organization | Replace hardcoded `/home/gaga/` paths with env-var config (54 files) | ✅ 1abf0a0 |
| 8 | Preprocessing | Standardize column naming across all preprocessing scripts | ✅ (see below) |
| 9 | Dead code | Complete script migration: delete old dirs, promote new structure | ⚠️ deferred |
| 10 | Organization | Add `src/analysis/` as proper installed package | ✅ already done |

---

## Decision Log

### Step 8 — Column naming audit findings

After auditing all preprocessing outputs, the convention is **intentionally consistent by condition type**, not by dataset:

- **Circulatory failure** (MIMIC, eICU, HiRID): `time_hours` (float) + `time_hour` (int) — hourly resolution because CF is an acute condition
- **AKI, liver, sepsis, ventilator** (MIMIC, eICU): `time_days` (float) + `time_day` (int) — daily resolution because these conditions evolve over days
- **Value columns** (`lactate`, `heartrate`, `systolic`) are named consistently across all datasets for the same biomarker

`pick_time_cols()` in `analysis_utils.py` auto-detects the pair at runtime, so no manual configuration is needed.

**Fix applied:** removed dead `time_unit` field from `get_dataset_config()`. The field was set to "hours" for all datasets but was never read by `analysis_template.py` and was factually wrong for non-CF tasks. Replaced with a docstring explaining the task-level split.

Note: MIMIC CF timeseries outputs include an extra `time_days` column (float days, redundant since hours is present). Harmless — `pick_time_cols()` returns hours when both are present.

### Step 9 — Script migration: deferred

Current state:
- `scripts/traj_scripts/` — actual code for all trajectory scripts (still active)
- `scripts/trajectory/` — runpy wrappers pointing into `traj_scripts/`; most SLURM scripts call these
- `scripts/*_preprocessing/` — actual preprocessing code (active)
- `scripts/preprocess/` — runpy wrappers pointing into `*_preprocessing/`; not yet used by SLURM scripts

4 SLURM scripts still reference `traj_scripts/` directly (the `*_bayes_trajectories.py` scripts, for which wrappers were never created in `scripts/trajectory/`):
- `scripts/slurm/eicu/run_eicu_circulatory_failure_bayes_trajectories.slurm`
- `scripts/slurm/mimic/run_mimic_circulatory_failure_bayes_gpu_test.slurm`
- `scripts/slurm/mimic/run_mimic_circulatory_failure_bayes_trajectories.slurm`
- `scripts/slurm/hirid/run_hirid_circulatory_failure_bayes_trajectories.slurm`

**Recommended next step:** move actual code from `traj_scripts/` directly into `scripts/trajectory/` (rename and update imports), then delete the wrappers and the old dir. Deferred because it touches many files and requires verifying all `sys.path` and relative imports.

### Step 10 — `src/analysis/` package

Already properly discoverable: `pyproject.toml` uses `[tool.setuptools.packages.find] where = ["src"]`, which picks up both `traj_features` and `analysis` automatically. `src/analysis/__init__.py` exists (empty). Confirmed by `pytest tests/test_smoke_backends.py::test_analysis_imports`.
