# Trajectory Modeling — Improvement Plan

**Status:** In planning
**Last updated:** 2026-06-13

---

## Planned Steps

| # | Area | Task | Status |
|---|------|------|--------|
| 1 | Bayes bug | Fix numpyro/JAX sampler overwrite bug in `sampling.py` | pending |
| 2 | Dead code | Remove temp files, untrack `__pycache__`, dedup launcher generator | pending |
| 3 | Dead code | Complete script migration: delete old dirs, promote new structure | pending |
| 4 | Preprocessing | Fix deprecated `fillna(method=)` with `.ffill()` | pending |
| 5 | Preprocessing | Replace hardcoded `/home/gaga/` paths with env-var config | pending |
| 6 | Bayes perf | Remove DEBUG prints, expose `batch_size`/`chain_method` in `BayesConfig` | pending |
| 7 | Bayes perf | Add nutpie as recommended default, document sampler tradeoffs | pending |
| 8 | Preprocessing | Standardize column naming across all preprocessing scripts | pending |
| 9 | Organization | Move `src/eicu_loader.py` into proper location | pending |
| 10 | Organization | Add `src/analysis/` as proper installed package | pending |

---

## Decision Log

*(Decisions made during execution will be recorded here)*
