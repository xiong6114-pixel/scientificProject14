# Compare Folder To 50-Point Main Chain

This note keeps the old MATLAB-aligned compare sandbox intact while marking the files that correspond to the current 50-point typed-seeded pipeline.

## Entry Points

| compare file | 50-point main-chain file | note |
| --- | --- | --- |
| `runme.py` | `../run_ev_typed_mogabka_seeded.py`, `../mogabka_seeded.py`, `../get_mofcn.py` | `compare/runme.py` now runs the new 50-point typed multi-algorithm compare entry. |
| `runme_50_typed_compare.py` | `../run_ev_typed_mogabka_seeded.py`, `../mogabka_seeded.py` | New compare-side typed runner for `MOGABKA + HNSGA-II+KMeans + NSGA-II + NSGA-III + MOEA/D`. |
| `runme_50_typed_seeded.py` | `../run me.py` and `../run_ev_typed_mogabka_seeded.py` | New compare-side shim for the current typed-seeded pipeline. |
| `runme_matlab_aligned.py` | `../run_ev_typed_mogabka_seeded.py` | Legacy compare entry for old MATLAB-aligned multi-algorithm experiments. |

## Objective And Parameters

| compare file | 50-point main-chain file | note |
| --- | --- | --- |
| `get_parameter.py` | `../get_parameter.py` | Legacy compare code still uses `mu=20`; main chain uses `mu=40`. Do not mix them. |
| `cal_obj_1_debug.py` | `../cal_obj_typed_py.py` and `../get_mofcn.py` | Old compare objective is untyped single-count encoding; main chain is typed `3*M` encoding with infeasible penalty handling. |
| `waiting_time.py` | `../cal_obj_typed_py.py` | Queue logic moved into typed station analysis (`mmc_wq`, rho checks, per-type service rates). |

## Initialization And Seed Injection

| compare file | 50-point main-chain file | note |
| --- | --- | --- |
| `init_encoding.py` | `../initialization.py` | Old compare init builds MATLAB-style sparse single-vector individuals. |
| `init_encoding.py` | `../ev_typed_nn_seed_adapter.py` | Typed packing/repair moved here for `3*M` decision vectors. |
| none | `../nn_seed_injection_50.py` | The current strict PCC seed builder only exists in the main chain. |

## MOGABKA Core

| compare file | 50-point main-chain file | note |
| --- | --- | --- |
| `mobkaga_matlab_aligned.py` | `../mogabka_seeded.py` | Both are MOGABKA-family code, but the main-chain version adds integer typed encoding, seed injection, safer archive metrics, and archive PF exports. |
| `mobka_metrics.py` | `../hv.py` and `../coverage.py` | Compare keeps MATLAB-style lightweight metrics; main chain uses the project metric stack. |
| `compare_init_matlab_python.py` and other `compare_*` scripts | none | These stay as legacy MATLAB regression checks and are not part of the typed 50-point runtime. |

## Deep-Upgraded Compare Algorithms

These compare files now have a typed branch that activates when settings include `_typed_problem`:

- `hnsga2_kmeans_typed.py`
- `nsga2_matlab_aligned.py`
- `nsga3_matlab_aligned.py`
- `moead_matlab_aligned.py`

That typed branch uses:

- `mu=40` from `../get_parameter.py`
- 50-point runtime loading through the current checkpoint/sample path
- typed `3*M` decision vectors
- shared typed initialization and typed mutation/crossover

`hnsga2_kmeans_typed.py` is the HNSGA-II + K-Means paper baseline. By default it keeps K-Means demand-aware initialization even when the other baselines share a random initial population. Set `HNSGA_USE_SHARED_INIT=1` to force the same shared initialization for ablation, and set `HNSGA_KMEANS_SEED_COUNT` to tune how many K-Means seeds enter its initial population.

The original MATLAB-aligned branch still exists for old regression tests.

## Algorithms With No Current Main-Chain Replacement

These files are still compare-only today:

- `imobka_matlab_aligned.py`
- `mobka_stage1.py`
- `mobka_stage2.py`
- `mobka_stage3.py`
- `moead_matlab_aligned.py`
- `nsga2_matlab_aligned.py`
- `nsga3_matlab_aligned.py`
- `run_mobka_incremental_experiment.py`
- `run_moead_imobka_nsga3_small.py`
- `run_nsga2_small_experiment.py`

They still target the old MATLAB-aligned problem definition and have not been ported to `mu=40 + 50-point + typed seed`.

## Practical Rule

- Use `compare/runme.py` when you want the current 50-point typed multi-algorithm compare run.
- Use `compare/runme_50_typed_seeded.py` when you only want the current 50-point seeded MOGABKA quick run.
- Use `compare/runme_matlab_aligned.py` and the `compare_*` scripts only when you are intentionally reproducing the old MATLAB-aligned sandbox.

`runme.py` / `runme_50_typed_compare.py` support `COMPARE_INIT_MODE`:

- `all_random`
- `all_shared_seeded_init`
- `mobka_only_seeded`
- `all_modes` for a full sweep and summary table
