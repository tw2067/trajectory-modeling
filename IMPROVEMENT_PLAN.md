# Trajectory Modeling — Improvement Plan

**Status:** In planning
**Last updated:** 2026-06-13

---

## Planned Steps

| # | Area | Task | Status |
|---|------|------|--------|
| 1 | Bayes bug | Fix numpyro/JAX sampler overwrite bug in `sampling.py` | ✅ 0273f35 |
| 2 | Dead code | Remove temp files, untrack `__pycache__`, dedup launcher generator | ✅ 5b35c1c |
| 3 | Dead code | Complete script migration: delete old dirs, promote new structure | pending (needs inspection) |
| 4 | Preprocessing | Fix deprecated `fillna(method=)` with `.ffill()` (9 files) | ✅ d74cd55 |
| 5 | Organization | Move `src/eicu_loader.py` → `scripts/eicu_preprocessing/` | ✅ 255d30f |
| 6 | Bayes perf | Add `batch_size`/`chain_method` to `BayesConfig`; auto-select nutpie | ✅ 2584306 |
| 7 | Preprocessing | Replace hardcoded `/home/gaga/` paths with env-var config | needs inspection |
| 8 | Preprocessing | Standardize column naming across all preprocessing scripts | needs inspection |
| 9 | Dead code | Complete script migration: delete old dirs, promote new structure | needs inspection |
| 10 | Organization | Add `src/analysis/` as proper installed package | needs inspection |

---

## Decision Log

*(Decisions made during execution will be recorded here)*
