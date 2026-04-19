from __future__ import annotations

import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_PROJECT_DIR))

from coverage import Coverage
from gd import GD
from hv import HV
from igd import IGD
from mogabka_seeded import MOGABKA
from moead_matlab_aligned import MOEAD_function
from nsga2_matlab_aligned import NSGA2_funciton
from nsga3_matlab_aligned import NSGA3_funciton
from typed_compare_utils import (
    activated_ev_context,
    build_mogabka_seed_config,
    build_typed_compare_problem,
    build_typed_initial_population,
    merge_fronts_and_extract_reference,
)


MODE_ALL_RANDOM = "all_random"
MODE_ALL_SHARED_SEEDED = "all_shared_seeded_init"
MODE_MOBKA_ONLY_SEEDED = "mobka_only_seeded"
MODE_ALL = "all_modes"
VALID_MODES = {
    MODE_ALL_RANDOM,
    MODE_ALL_SHARED_SEEDED,
    MODE_MOBKA_ONLY_SEEDED,
    MODE_ALL,
}


def _safe_metric(metric_fn, front: np.ndarray, ref: np.ndarray, empty_value: float) -> float:
    front = np.asarray(front, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if front.ndim == 1 and front.size > 0:
        front = front.reshape(1, -1)
    if ref.ndim == 1 and ref.size > 0:
        ref = ref.reshape(1, -1)
    if front.size == 0 or ref.size == 0:
        return float(empty_value)
    return float(metric_fn(front, ref))


def _safe_hv(front: np.ndarray, ref: np.ndarray) -> float:
    if np.asarray(front).size == 0 or np.asarray(ref).size == 0:
        return 0.0
    return float(HV(front, ref)[0])


def _safe_coverage(front_a: np.ndarray, front_b: np.ndarray) -> float:
    front_a = np.asarray(front_a, dtype=float)
    front_b = np.asarray(front_b, dtype=float)
    if front_a.size == 0 or front_b.size == 0:
        return 0.0
    return float(Coverage(front_b, front_a))


def _plot_front(ax, front: np.ndarray, label: str, marker: str) -> None:
    front = np.asarray(front, dtype=float)
    if front.size == 0:
        return
    ax.scatter(front[:, 0], front[:, 1], s=26, label=label, marker=marker)


def _normalize_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    aliases = {
        "random": MODE_ALL_RANDOM,
        "seeded": MODE_ALL_SHARED_SEEDED,
        "shared_seeded": MODE_ALL_SHARED_SEEDED,
        "mobka_seeded": MODE_MOBKA_ONLY_SEEDED,
        "mobka_only": MODE_MOBKA_ONLY_SEEDED,
        "sweep": MODE_ALL,
        "all": MODE_ALL,
    }
    mode = aliases.get(mode, mode)
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported COMPARE_INIT_MODE={mode!r}. Valid modes: {sorted(VALID_MODES)}")
    return mode


def _mode_filename_tag(mode: str) -> str:
    return {
        MODE_ALL_RANDOM: "all_random",
        MODE_ALL_SHARED_SEEDED: "all_shared_seeded_init",
        MODE_MOBKA_ONLY_SEEDED: "mobka_only_seeded",
    }[mode]


def _mode_description(mode: str) -> str:
    return {
        MODE_ALL_RANDOM: "all algorithms start from the same random typed population",
        MODE_ALL_SHARED_SEEDED: "all algorithms start from the same shared typed population that already contains NN seeds",
        MODE_MOBKA_ONLY_SEEDED: "NSGA-II/III/MOEA-D use shared random init, while MOGABKA keeps NN-seeded initialization",
    }[mode]


def _build_shared_initial_for_mode(problem, mode: str, popnum: int, init_nn_seed_count: int, base_seed: int):
    rng_init = np.random.RandomState(base_seed)
    shared_seed_count = 0 if mode in (MODE_ALL_RANDOM, MODE_MOBKA_ONLY_SEEDED) else init_nn_seed_count
    shared_pop, shared_obj = build_typed_initial_population(
        problem=problem,
        pop_size=popnum,
        rng=rng_init,
        init_nn_seed_count=shared_seed_count,
    )
    return shared_pop, shared_obj, shared_seed_count


def _run_mogabka_for_mode(problem, mode: str, popnum: int, max_iter: int, init_nn_seed_count: int, base_seed: int, shared_pop: np.ndarray):
    seed_cfg = build_mogabka_seed_config(problem, init_nn_seed_count)

    if mode == MODE_ALL_RANDOM:
        seed_cfg["initial_individuals"] = shared_pop.copy()
        seed_cfg["builder_fn"] = None
        seed_cfg["init_nn_seed_count"] = 0
    elif mode == MODE_ALL_SHARED_SEEDED:
        seed_cfg["initial_individuals"] = shared_pop.copy()
        seed_cfg["builder_fn"] = None
        seed_cfg["init_nn_seed_count"] = 0
    elif mode == MODE_MOBKA_ONLY_SEEDED:
        seed_cfg["initial_individuals"] = None
        seed_cfg["builder_fn"] = problem.seed_builder
        seed_cfg["init_nn_seed_count"] = init_nn_seed_count
    else:
        raise ValueError(f"Unsupported mode for MOGABKA: {mode}")

    with activated_ev_context(problem):
        fitness, pop, ture_pf, result = MOGABKA(
            Max_iter=max_iter,
            SearchAgents_no=popnum,
            FUN="EV_TYPED_CS",
            dim=problem.dim,
            numObj=2,
            lb=problem.lb_vec,
            ub=problem.ub_vec,
            seed=base_seed,
            seed_injection_config=seed_cfg,
            problem_context=problem.problem_context,
        )

    front = np.asarray(result.get("archive_pf_fitness", ture_pf), dtype=float)
    return {
        "fitness": np.asarray(fitness, dtype=float),
        "pop": np.asarray(pop, dtype=float),
        "front": front,
        "result": result,
    }


def _write_mode_outputs(
    out_dir: Path,
    mode: str,
    reference_pf: np.ndarray,
    fronts: dict[str, np.ndarray],
    traces: dict[str, np.ndarray],
    metrics_rows: list[list[object]],
    coverage_matrix: np.ndarray,
) -> None:
    tag = _mode_filename_tag(mode)

    for algo_key, front in fronts.items():
        np.savetxt(out_dir / f"py_runme50_typed_front_{algo_key}_{tag}.csv", np.asarray(front, dtype=float), delimiter=",", fmt="%.17g")

    np.savetxt(out_dir / f"py_runme50_typed_reference_pf_{tag}.csv", np.asarray(reference_pf, dtype=float), delimiter=",", fmt="%.17g")

    for algo_key, gen_metrics in traces.items():
        np.savetxt(
            out_dir / f"py_runme50_typed_{algo_key}_generation_metrics_{tag}.csv",
            np.asarray(gen_metrics, dtype=float),
            delimiter=",",
            fmt="%.17g",
        )

    with open(out_dir / f"py_runme50_typed_metrics_{tag}.csv", "w", encoding="utf-8") as f:
        f.write("mode,algorithm,nd_count,hv,gd,igd\n")
        for row in metrics_rows:
            f.write(f"{mode},{row[0]},{int(row[1])},{row[2]:.17g},{row[3]:.17g},{row[4]:.17g}\n")

    labels = [str(row[0]) for row in metrics_rows]
    with open(out_dir / f"py_runme50_typed_coverage_matrix_{tag}.csv", "w", encoding="utf-8") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, label in enumerate(labels):
            values = ",".join(f"{coverage_matrix[i, j]:.17g}" for j in range(len(labels)))
            f.write(f"{label},{values}\n")

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    _plot_front(ax1, fronts["mogabka"], "MOGABKA", "o")
    _plot_front(ax1, fronts["nsga2"], "NSGA-II", "s")
    _plot_front(ax1, fronts["nsga3"], "NSGA-III", "^")
    _plot_front(ax1, fronts["moead"], "MOEA/D", "x")
    if reference_pf.size > 0:
        ax1.scatter(reference_pf[:, 0], reference_pf[:, 1], s=18, label="Reference PF", marker=".", alpha=0.5)
    ax1.set_xlabel("obj1")
    ax1.set_ylabel("obj2")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(out_dir / f"py_runme50_typed_front_compare_{tag}.png", dpi=180)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    hv_vals = np.asarray([float(row[2]) for row in metrics_rows], dtype=float)
    ax2.bar(labels, hv_vals)
    ax2.set_ylabel("HV")
    fig2.tight_layout()
    fig2.savefig(out_dir / f"py_runme50_typed_hv_bar_{tag}.png", dpi=180)

    plt.close(fig1)
    plt.close(fig2)


def _run_one_mode(
    problem,
    out_dir: Path,
    mode: str,
    max_iter: int,
    popnum: int,
    init_nn_seed_count: int,
    base_seed: int,
) -> dict:
    print(f"\n=== Mode: {mode} ===")
    print(_mode_description(mode))

    shared_pop, shared_obj, shared_seed_count = _build_shared_initial_for_mode(
        problem=problem,
        mode=mode,
        popnum=popnum,
        init_nn_seed_count=init_nn_seed_count,
        base_seed=base_seed,
    )
    print(f"shared population built: size={shared_pop.shape[0]}, shared_seed_count={shared_seed_count}")

    shared_settings = {
        "_typed_problem": problem,
        "maxgen": max_iter,
        "popnum": popnum,
        "init_nn_seed_count": init_nn_seed_count,
        "obj_manager": np.asarray(shared_obj, dtype=float).copy(),
        "sol_manager": [shared_pop[i, :].copy() for i in range(shared_pop.shape[0])],
    }

    print("running MOGABKA...", flush=True)
    mogabka_out = _run_mogabka_for_mode(
        problem=problem,
        mode=mode,
        popnum=popnum,
        max_iter=max_iter,
        init_nn_seed_count=init_nn_seed_count,
        base_seed=base_seed,
        shared_pop=shared_pop,
    )
    front_mogabka = mogabka_out["front"]

    print("running NSGA-II...", flush=True)
    front_nsga2, trace_nsga2 = NSGA2_funciton(
        settings=dict(shared_settings),
        rng=np.random.RandomState(base_seed),
        return_trace=True,
    )

    print("running NSGA-III...", flush=True)
    front_nsga3, trace_nsga3 = NSGA3_funciton(
        settings=dict(shared_settings),
        rng=np.random.RandomState(base_seed),
        return_trace=True,
    )

    print("running MOEA/D...", flush=True)
    front_moead, trace_moead = MOEAD_function(
        settings=dict(shared_settings),
        rng=np.random.RandomState(base_seed),
        return_trace=True,
    )

    reference_pf = merge_fronts_and_extract_reference(
        [front_mogabka, front_nsga2, front_nsga3, front_moead],
        invalid_penalty=problem.invalid_penalty,
    )

    labels = ["MOGABKA", "NSGA2", "NSGA3", "MOEAD"]
    fronts = {
        "mogabka": front_mogabka,
        "nsga2": front_nsga2,
        "nsga3": front_nsga3,
        "moead": front_moead,
    }
    traces = {
        "nsga2": trace_nsga2.generation_metrics,
        "nsga3": trace_nsga3.generation_metrics,
        "moead": trace_moead.generation_metrics,
    }

    metrics_rows = []
    ordered_fronts = [front_mogabka, front_nsga2, front_nsga3, front_moead]
    for label, front in zip(labels, ordered_fronts):
        metrics_rows.append(
            [
                label,
                int(np.asarray(front).shape[0]),
                _safe_hv(front, reference_pf),
                _safe_metric(GD, front, reference_pf, np.inf),
                _safe_metric(IGD, front, reference_pf, np.inf),
            ]
        )

    coverage_matrix = np.zeros((len(ordered_fronts), len(ordered_fronts)), dtype=float)
    for i in range(len(ordered_fronts)):
        for j in range(len(ordered_fronts)):
            coverage_matrix[i, j] = 1.0 if i == j else _safe_coverage(ordered_fronts[i], ordered_fronts[j])

    _write_mode_outputs(
        out_dir=out_dir,
        mode=mode,
        reference_pf=reference_pf,
        fronts=fronts,
        traces=traces,
        metrics_rows=metrics_rows,
        coverage_matrix=coverage_matrix,
    )

    print("mode summary:")
    for row in metrics_rows:
        print(f"{row[0]}: nd={int(row[1])} hv={row[2]:.6f} gd={row[3]:.6f} igd={row[4]:.6f}")

    return {
        "mode": mode,
        "reference_pf": reference_pf,
        "metrics_rows": metrics_rows,
        "coverage_matrix": coverage_matrix,
        "fronts": fronts,
    }


def _write_sweep_summary(out_dir: Path, all_results: list[dict]) -> None:
    rows = []
    for result in all_results:
        mode = result["mode"]
        for algo, nd_count, hv_val, gd_val, igd_val in result["metrics_rows"]:
            rows.append([mode, algo, int(nd_count), float(hv_val), float(gd_val), float(igd_val)])

    with open(out_dir / "py_runme50_typed_mode_sweep_metrics.csv", "w", encoding="utf-8") as f:
        f.write("mode,algorithm,nd_count,hv,gd,igd\n")
        for row in rows:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]:.17g},{row[4]:.17g},{row[5]:.17g}\n")

    modes = [r["mode"] for r in all_results]
    algos = [str(row[0]) for row in all_results[0]["metrics_rows"]]
    hv_by_mode_algo = np.zeros((len(modes), len(algos)), dtype=float)
    for i, result in enumerate(all_results):
        for j, row in enumerate(result["metrics_rows"]):
            hv_by_mode_algo[i, j] = float(row[2])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(modes))
    width = 0.18
    for j, algo in enumerate(algos):
        ax.bar(x + (j - 1.5) * width, hv_by_mode_algo[:, j], width=width, label=algo)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=12)
    ax.set_ylabel("HV")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "py_runme50_typed_mode_sweep_hv.png", dpi=180)
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    max_iter = int(os.environ.get("MAX_ITER", "30"))
    popnum = int(os.environ.get("SEARCH_AGENTS_NO", "50"))
    init_nn_seed_count = int(os.environ.get("INIT_NN_SEED_COUNT", "12"))
    seed_device = os.environ.get("SEED_DEVICE", "cpu")
    base_seed = int(os.environ.get("COMPARE_SEED", "42"))
    mode = _normalize_mode(os.environ.get("COMPARE_INIT_MODE", MODE_ALL_SHARED_SEEDED))

    print("building typed compare problem...", flush=True)
    problem = build_typed_compare_problem(seed_device=seed_device)
    print(f"runtime source: {problem.source_desc}")
    print(f"checkpoint: {problem.ckpt_path.name}")
    print(f"parameter: {problem.parameter.tolist()}")
    print(f"popnum={popnum}, max_iter={max_iter}, init_nn_seed_count={init_nn_seed_count}, mode={mode}")

    modes_to_run = (
        [MODE_ALL_RANDOM, MODE_ALL_SHARED_SEEDED, MODE_MOBKA_ONLY_SEEDED]
        if mode == MODE_ALL
        else [mode]
    )

    all_results = []
    for mode_name in modes_to_run:
        result = _run_one_mode(
            problem=problem,
            out_dir=out_dir,
            mode=mode_name,
            max_iter=max_iter,
            popnum=popnum,
            init_nn_seed_count=init_nn_seed_count,
            base_seed=base_seed,
        )
        all_results.append(result)

    if len(all_results) == 1:
        result = all_results[0]
        print("\nCoverage matrix C(A,B) = fraction of B dominated by A")
        print("rows/cols:", [str(row[0]) for row in result["metrics_rows"]])
        print(np.array2string(result["coverage_matrix"], precision=4, suppress_small=False))
    else:
        _write_sweep_summary(out_dir, all_results)
        print("\nMode sweep summary:")
        for result in all_results:
            best = max(result["metrics_rows"], key=lambda row: float(row[2]))
            print(f"{result['mode']}: best_hv={best[0]} ({float(best[2]):.6f})")

    print("\nSaved typed compare outputs to:", out_dir)


if __name__ == "__main__":
    main()
