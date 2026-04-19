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
from hv import HV
from imobka_matlab_aligned import IMOBKA_funciton
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


def _safe_hv(front: np.ndarray, ref: np.ndarray) -> float:
    front = np.asarray(front, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if front.size == 0 or ref.size == 0:
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
    ax.scatter(front[:, 0], front[:, 1], s=24, label=label, marker=marker)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    max_iter = int(os.environ.get("MAX_ITER", "30"))
    popnum = int(os.environ.get("SEARCH_AGENTS_NO", "50"))
    init_nn_seed_count = int(os.environ.get("INIT_NN_SEED_COUNT", "12"))
    seed_device = os.environ.get("SEED_DEVICE", "cpu")
    base_seed = int(os.environ.get("COMPARE_SEED", "42"))

    problem = build_typed_compare_problem(seed_device=seed_device)
    print(f"runtime source: {problem.source_desc}")
    print(f"checkpoint: {problem.ckpt_path.name}")
    print(f"parameter: {problem.parameter.tolist()}")
    print(f"popnum={popnum}, max_iter={max_iter}, init_nn_seed_count={init_nn_seed_count}")

    rng_init = np.random.RandomState(base_seed)
    shared_random_pop, shared_random_obj = build_typed_initial_population(
        problem=problem,
        pop_size=popnum,
        rng=rng_init,
        init_nn_seed_count=0,
    )

    baseline_settings = {
        "_typed_problem": problem,
        "maxgen": max_iter,
        "popnum": popnum,
        "init_nn_seed_count": 0,
        "obj_manager": np.asarray(shared_random_obj, dtype=float).copy(),
        "sol_manager": [shared_random_pop[i, :].copy() for i in range(shared_random_pop.shape[0])],
    }

    print("running IMOGABKA-seed...", flush=True)
    seed_cfg = build_mogabka_seed_config(problem, init_nn_seed_count)
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
    _ = fitness, pop
    front_imogabka_seed = np.asarray(result.get("archive_pf_fitness", ture_pf), dtype=float)

    print("running MOBKA...", flush=True)
    front_mobka, trace_mobka = IMOBKA_funciton(
        settings=dict(baseline_settings),
        rng=np.random.RandomState(base_seed),
        return_trace=True,
    )

    print("running NSGA-II...", flush=True)
    front_nsga2, trace_nsga2 = NSGA2_funciton(
        settings=dict(baseline_settings),
        rng=np.random.RandomState(base_seed),
        return_trace=True,
    )

    print("running NSGA-III...", flush=True)
    front_nsga3, trace_nsga3 = NSGA3_funciton(
        settings=dict(baseline_settings),
        rng=np.random.RandomState(base_seed),
        return_trace=True,
    )

    print("running MOEA/D...", flush=True)
    front_moead, trace_moead = MOEAD_function(
        settings=dict(baseline_settings),
        rng=np.random.RandomState(base_seed),
        return_trace=True,
    )

    labels = ["IMOGABKA-seed", "MOBKA", "NSGA2", "NSGA3", "MOEAD"]
    fronts = [front_imogabka_seed, front_mobka, front_nsga2, front_nsga3, front_moead]
    front_map = {
        "imogabka_seed": front_imogabka_seed,
        "mobka": front_mobka,
        "nsga2": front_nsga2,
        "nsga3": front_nsga3,
        "moead": front_moead,
    }

    reference_pf = merge_fronts_and_extract_reference(fronts, invalid_penalty=problem.invalid_penalty)

    metrics_rows = []
    for label, front in zip(labels, fronts):
        metrics_rows.append([label, int(np.asarray(front).shape[0]), _safe_hv(front, reference_pf)])

    coverage_matrix = np.zeros((len(fronts), len(fronts)), dtype=float)
    for i in range(len(fronts)):
        for j in range(len(fronts)):
            coverage_matrix[i, j] = 1.0 if i == j else _safe_coverage(fronts[i], fronts[j])

    np.savetxt(out_dir / "real_compare_front_imogabka_seed.csv", np.asarray(front_imogabka_seed, dtype=float), delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / "real_compare_front_mobka.csv", np.asarray(front_mobka, dtype=float), delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / "real_compare_front_nsga2.csv", np.asarray(front_nsga2, dtype=float), delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / "real_compare_front_nsga3.csv", np.asarray(front_nsga3, dtype=float), delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / "real_compare_front_moead.csv", np.asarray(front_moead, dtype=float), delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / "real_compare_reference_pf.csv", np.asarray(reference_pf, dtype=float), delimiter=",", fmt="%.17g")

    with open(out_dir / "real_compare_hv.csv", "w", encoding="utf-8") as f:
        f.write("algorithm,nd_count,hv\n")
        for row in metrics_rows:
            f.write(f"{row[0]},{int(row[1])},{float(row[2]):.17g}\n")

    with open(out_dir / "real_compare_coverage.csv", "w", encoding="utf-8") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, label in enumerate(labels):
            values = ",".join(f"{coverage_matrix[i, j]:.17g}" for j in range(len(labels)))
            f.write(f"{label},{values}\n")

    np.savetxt(out_dir / "real_compare_mobka_trace_archive_size.csv", trace_mobka.archive_size_history, delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / "real_compare_nsga2_generation_metrics.csv", trace_nsga2.generation_metrics, delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / "real_compare_nsga3_generation_metrics.csv", trace_nsga3.generation_metrics, delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / "real_compare_moead_generation_metrics.csv", trace_moead.generation_metrics, delimiter=",", fmt="%.17g")

    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_front(ax, front_imogabka_seed, "IMOGABKA-seed", "o")
    _plot_front(ax, front_mobka, "MOBKA", "d")
    _plot_front(ax, front_nsga2, "NSGA-II", "s")
    _plot_front(ax, front_nsga3, "NSGA-III", "^")
    _plot_front(ax, front_moead, "MOEA/D", "x")
    if reference_pf.size > 0:
        ax.scatter(reference_pf[:, 0], reference_pf[:, 1], s=14, label="Reference PF", marker=".", alpha=0.5)
    ax.set_xlabel("obj1")
    ax.set_ylabel("obj2")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "real_compare_fronts.png", dpi=180)
    plt.close(fig)

    print("\nHV summary:")
    for row in metrics_rows:
        print(f"{row[0]}: nd={int(row[1])} hv={float(row[2]):.6f}")

    print("\nCoverage matrix C(A,B) = fraction of B dominated by A")
    print("rows/cols:", labels)
    print(np.array2string(coverage_matrix, precision=4, suppress_small=False))
    print("\nSaved real-problem compare outputs to:", out_dir)


if __name__ == "__main__":
    main()
