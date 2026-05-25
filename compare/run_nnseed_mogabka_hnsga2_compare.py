from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_PROJECT_DIR))

from coverage import Coverage
from hnsga2_kmeans_typed import HNSGA2_KMeans_funciton
from hv import HV
from imobka_matlab_aligned import IMOBKA_funciton
from mogabka_seeded import MOGABKA
from nsga2_matlab_aligned import NSGA2_funciton
from nsga3_matlab_aligned import NSGA3_funciton
from typed_compare_utils import (
    activated_ev_context,
    build_mogabka_seed_config,
    build_typed_compare_problem,
    build_typed_initial_population,
    merge_fronts_and_extract_reference,
)


OUT_DIR = Path(__file__).resolve().parent


def _safe_hv(front: np.ndarray, ref: np.ndarray) -> float:
    front = np.asarray(front, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if front.size == 0 or ref.size == 0:
        return 0.0

    fixed_ref = os.environ.get("HV_REF_POINT", "").strip()
    if fixed_ref:
        parts = [float(x.strip()) for x in fixed_ref.split(",") if x.strip()]
        if len(parts) != 2:
            raise ValueError("HV_REF_POINT must be 'obj1,obj2' for 2D experiments.")
        return _hv_fixed_ref_2d(front, np.asarray(parts, dtype=float))

    return float(HV(front, ref)[0])


def _hv_fixed_ref_2d(front: np.ndarray, ref_point: np.ndarray) -> float:
    front = np.asarray(front, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float).reshape(2)
    if front.ndim == 1:
        front = front.reshape(1, -1)
    if front.shape[1] != 2:
        raise ValueError("fixed-reference HV currently supports only 2 objectives.")

    mask = np.all(np.isfinite(front), axis=1) & np.all(front <= ref_point.reshape(1, -1), axis=1)
    pts = front[mask, :]
    if pts.shape[0] == 0:
        return 0.0

    keep = np.ones(pts.shape[0], dtype=bool)
    for i in range(pts.shape[0]):
        for j in range(pts.shape[0]):
            if i == j:
                continue
            if np.all(pts[j, :] <= pts[i, :]) and np.any(pts[j, :] < pts[i, :]):
                keep[i] = False
                break
    nd = pts[keep, :]
    if nd.shape[0] == 0:
        return 0.0

    nd = nd[np.argsort(nd[:, 0]), :]
    hv = 0.0
    curr_y = float(ref_point[1])
    for x, y in nd:
        if y < curr_y:
            hv += max(float(ref_point[0]) - float(x), 0.0) * (curr_y - float(y))
            curr_y = float(y)
    return float(hv)


def _safe_coverage(front_a: np.ndarray, front_b: np.ndarray) -> float:
    front_a = np.asarray(front_a, dtype=float)
    front_b = np.asarray(front_b, dtype=float)
    if front_a.size == 0 or front_b.size == 0:
        return 0.0
    return float(Coverage(front_b, front_a))


def _front_snapshot(front: np.ndarray) -> dict[str, float]:
    front = np.asarray(front, dtype=float)
    if front.size == 0:
        return {
            "nd_count": 0.0,
            "best_obj1": float("inf"),
            "best_obj2": float("inf"),
            "mean_obj1": float("inf"),
            "mean_obj2": float("inf"),
        }
    if front.ndim == 1:
        front = front.reshape(1, -1)
    return {
        "nd_count": float(front.shape[0]),
        "best_obj1": float(np.min(front[:, 0])),
        "best_obj2": float(np.min(front[:, 1])),
        "mean_obj1": float(np.mean(front[:, 0])),
        "mean_obj2": float(np.mean(front[:, 1])),
    }


def _plot_front(ax, front: np.ndarray, label: str, marker: str) -> None:
    front = np.asarray(front, dtype=float)
    if front.size == 0:
        return
    if front.ndim == 1:
        front = front.reshape(1, -1)
    ax.scatter(front[:, 0], front[:, 1], s=24, label=label, marker=marker)


def _run_mogabka(
    problem,
    max_iter: int,
    popnum: int,
    base_seed: int,
    seed_cfg: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    with activated_ev_context(problem):
        _, _, true_pf, result = MOGABKA(
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
    front = np.asarray(result.get("archive_pf_fitness", true_pf), dtype=float)
    return front, result


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    sample_idx = int(os.environ.get("RUNTIME_SAMPLE_INDEX", "1286"))
    os.environ["RUNTIME_SAMPLE_INDEX"] = str(sample_idx)

    max_iter = int(os.environ.get("MAX_ITER", "200"))
    popnum = int(os.environ.get("SEARCH_AGENTS_NO", "50"))
    init_nn_seed_count = int(os.environ.get("INIT_NN_SEED_COUNT", "12"))
    seed_device = os.environ.get("SEED_DEVICE", "cpu")
    base_seed = int(os.environ.get("COMPARE_SEED", "42"))
    hnsga_kmeans_seed_count = int(os.environ.get("HNSGA_KMEANS_SEED_COUNT", max(4, min(12, popnum // 3))))

    if "HV_REF_POINT" not in os.environ:
        os.environ["HV_REF_POINT"] = "200,500000000"
    if "VARIATE_RATE" not in os.environ:
        os.environ["VARIATE_RATE"] = "0.8"
    if "CROSS_RATE" not in os.environ:
        os.environ["CROSS_RATE"] = "0.3"
    if "ARCHIVE_SIZE" not in os.environ:
        os.environ["ARCHIVE_SIZE"] = "300"

    problem = build_typed_compare_problem(seed_device=seed_device)
    problem.problem_context["variate_rate"] = float(os.environ["VARIATE_RATE"])
    problem.problem_context["cross_rate"] = float(os.environ["CROSS_RATE"])
    problem.problem_context["archive_size"] = int(os.environ["ARCHIVE_SIZE"])

    print(f"runtime source: {problem.source_desc}")
    print(f"checkpoint: {problem.ckpt_path.name}")
    print(f"parameter: {problem.parameter.tolist()}")
    print(
        "settings: "
        f"sample_idx={sample_idx}, popnum={popnum}, max_iter={max_iter}, "
        f"init_nn_seed_count={init_nn_seed_count}, seed={base_seed}, "
        f"cross_rate={os.environ['CROSS_RATE']}, variate_rate={os.environ['VARIATE_RATE']}, "
        f"archive_size={os.environ['ARCHIVE_SIZE']}, hv_ref={os.environ['HV_REF_POINT']}"
    )

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
        "variate_rate": float(os.environ["VARIATE_RATE"]),
        "cross_rate": float(os.environ["CROSS_RATE"]),
        "obj_manager": np.asarray(shared_random_obj, dtype=float).copy(),
        "sol_manager": [shared_random_pop[i, :].copy() for i in range(shared_random_pop.shape[0])],
    }

    runs: list[tuple[str, str, np.ndarray, float, Any]] = []

    print("running NNseed-MOGABKA...", flush=True)
    t0 = time.perf_counter()
    nnseed_cfg = build_mogabka_seed_config(problem, init_nn_seed_count)
    front_nnseed_mogabka, result_nnseed = _run_mogabka(problem, max_iter, popnum, base_seed, nnseed_cfg)
    runs.append(("NNseed-MOGABKA", "nnseed_mogabka", front_nnseed_mogabka, time.perf_counter() - t0, result_nnseed))

    print("running MOGABKA...", flush=True)
    t0 = time.perf_counter()
    mogabka_cfg = {
        "enabled": True,
        "init_enabled": True,
        "init_nn_seed_count": 0,
        "initial_individuals": shared_random_pop.copy(),
        "builder_fn": None,
        "reinject_enabled": False,
    }
    front_mogabka, result_mogabka = _run_mogabka(problem, max_iter, popnum, base_seed, mogabka_cfg)
    runs.append(("MOGABKA", "mogabka", front_mogabka, time.perf_counter() - t0, result_mogabka))

    print("running HNSGA2-KMeans...", flush=True)
    t0 = time.perf_counter()
    hnsga_settings = dict(baseline_settings)
    hnsga_settings["force_kmeans_initial"] = True
    hnsga_settings["kmeans_seed_count"] = hnsga_kmeans_seed_count
    front_hnsga, trace_hnsga = HNSGA2_KMeans_funciton(
        settings=hnsga_settings,
        rng=np.random.RandomState(base_seed),
        return_trace=True,
    )
    runs.append(("HNSGA2-KMeans", "hnsga2_kmeans", np.asarray(front_hnsga, dtype=float), time.perf_counter() - t0, trace_hnsga))

    print("running NSGA2...", flush=True)
    t0 = time.perf_counter()
    front_nsga2, trace_nsga2 = NSGA2_funciton(
        settings=dict(baseline_settings),
        rng=np.random.RandomState(base_seed),
        return_trace=True,
    )
    runs.append(("NSGA2", "nsga2", np.asarray(front_nsga2, dtype=float), time.perf_counter() - t0, trace_nsga2))

    print("running NSGA3...", flush=True)
    t0 = time.perf_counter()
    front_nsga3, trace_nsga3 = NSGA3_funciton(
        settings=dict(baseline_settings),
        rng=np.random.RandomState(base_seed),
        return_trace=True,
    )
    runs.append(("NSGA3", "nsga3", np.asarray(front_nsga3, dtype=float), time.perf_counter() - t0, trace_nsga3))

    print("running MOBKA...", flush=True)
    t0 = time.perf_counter()
    front_mobka, trace_mobka = IMOBKA_funciton(
        settings=dict(baseline_settings),
        rng=np.random.RandomState(base_seed),
        return_trace=True,
    )
    runs.append(("MOBKA", "mobka", np.asarray(front_mobka, dtype=float), time.perf_counter() - t0, trace_mobka))

    labels = [label for label, _, _, _, _ in runs]
    fronts = [front for _, _, front, _, _ in runs]
    reference_pf = merge_fronts_and_extract_reference(fronts, invalid_penalty=problem.invalid_penalty)

    prefix = f"nnseed_mogabka_hnsga2_sample{sample_idx}"
    np.savetxt(OUT_DIR / f"{prefix}_reference_pf.csv", np.asarray(reference_pf, dtype=float), delimiter=",", fmt="%.17g")
    for _, key, front, _, _ in runs:
        np.savetxt(OUT_DIR / f"{prefix}_front_{key}.csv", np.asarray(front, dtype=float), delimiter=",", fmt="%.17g")

    metrics_rows: list[dict[str, Any]] = []
    for label, key, front, elapsed_sec, _ in runs:
        snap = _front_snapshot(front)
        metrics_rows.append(
            {
                "sample_idx": sample_idx,
                "algorithm": label,
                "front_key": key,
                "nd_count": int(snap["nd_count"]),
                "hv": _safe_hv(front, reference_pf),
                "best_obj1": snap["best_obj1"],
                "best_obj2": snap["best_obj2"],
                "mean_obj1": snap["mean_obj1"],
                "mean_obj2": snap["mean_obj2"],
                "elapsed_sec": float(elapsed_sec),
            }
        )
    _write_csv(
        OUT_DIR / f"{prefix}_metrics.csv",
        metrics_rows,
        [
            "sample_idx",
            "algorithm",
            "front_key",
            "nd_count",
            "hv",
            "best_obj1",
            "best_obj2",
            "mean_obj1",
            "mean_obj2",
            "elapsed_sec",
        ],
    )

    coverage_matrix = np.zeros((len(fronts), len(fronts)), dtype=float)
    for i in range(len(fronts)):
        for j in range(len(fronts)):
            coverage_matrix[i, j] = 1.0 if i == j else _safe_coverage(fronts[i], fronts[j])

    with (OUT_DIR / f"{prefix}_coverage.csv").open("w", encoding="utf-8") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, label in enumerate(labels):
            values = ",".join(f"{coverage_matrix[i, j]:.17g}" for j in range(len(labels)))
            f.write(f"{label},{values}\n")

    trace_map = {
        "hnsga2_kmeans": trace_hnsga.generation_metrics,
        "nsga2": trace_nsga2.generation_metrics,
        "nsga3": trace_nsga3.generation_metrics,
        "mobka_archive_size": trace_mobka.archive_size_history,
        "nnseed_mogabka_hv": result_nnseed.get("HV", np.empty(0)),
        "mogabka_hv": result_mogabka.get("HV", np.empty(0)),
    }
    for key, arr in trace_map.items():
        np.savetxt(OUT_DIR / f"{prefix}_trace_{key}.csv", np.asarray(arr, dtype=float), delimiter=",", fmt="%.17g")

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    markers = ["o", "P", "D", "s", "^", "d"]
    for (label, _, front, _, _), marker in zip(runs, markers):
        _plot_front(ax, front, label, marker)
    if reference_pf.size > 0:
        ax.scatter(reference_pf[:, 0], reference_pf[:, 1], s=14, label="Reference PF", marker=".", alpha=0.5)
    ax.set_xlabel("obj1")
    ax.set_ylabel("obj2")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{prefix}_fronts.png", dpi=180)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8.5, 5.2))
    hv_vals = [float(row["hv"]) for row in metrics_rows]
    x = np.arange(len(labels))
    ax2.bar(x, hv_vals)
    ax2.set_ylabel("HV")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=18, ha="right")
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / f"{prefix}_hv_bar.png", dpi=180)
    plt.close(fig2)

    print("\nHV summary:")
    for row in sorted(metrics_rows, key=lambda r: -float(r["hv"])):
        print(
            f"{row['algorithm']}: hv={float(row['hv']):.6f}, "
            f"nd={int(row['nd_count'])}, elapsed={float(row['elapsed_sec']):.2f}s"
        )

    print("\nCoverage matrix C(A,B) = fraction of B dominated by A")
    print("rows/cols:", labels)
    print(np.array2string(coverage_matrix, precision=4, suppress_small=False))
    print("\nSaved outputs with prefix:", OUT_DIR / prefix)


if __name__ == "__main__":
    main()
