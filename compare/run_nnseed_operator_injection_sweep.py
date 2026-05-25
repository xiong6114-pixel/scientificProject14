from __future__ import annotations

import csv
from contextlib import nullcontext, redirect_stdout
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _hv_fixed_ref_2d(front: np.ndarray, ref_point: np.ndarray) -> float:
    front = np.asarray(front, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float).reshape(2)
    if front.size == 0:
        return 0.0
    if front.ndim == 1:
        front = front.reshape(1, -1)

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


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_front(ax, front: np.ndarray, label: str, marker: str) -> None:
    front = np.asarray(front, dtype=float)
    if front.size == 0:
        return
    if front.ndim == 1:
        front = front.reshape(1, -1)
    ax.scatter(front[:, 0], front[:, 1], s=24, label=label, marker=marker)


def _configure_problem_context(problem, cross_rate: float, variate_rate: float, archive_size: int) -> None:
    problem.problem_context["cross_rate"] = float(cross_rate)
    problem.problem_context["variate_rate"] = float(variate_rate)
    problem.problem_context["archive_size"] = int(archive_size)


def _run_mogabka_front(
    problem,
    max_iter: int,
    popnum: int,
    seed: int,
    seed_cfg: dict[str, Any] | None,
    log_path: Path | None = None,
):
    stream_context = nullcontext()
    log_handle = None
    if log_path is not None:
        log_handle = log_path.open("w", encoding="utf-8")
        stream_context = redirect_stdout(log_handle)
    try:
        with stream_context:
            with activated_ev_context(problem):
                _, _, true_pf, result = MOGABKA(
                    Max_iter=max_iter,
                    SearchAgents_no=popnum,
                    FUN="EV_TYPED_CS",
                    dim=problem.dim,
                    numObj=2,
                    lb=problem.lb_vec,
                    ub=problem.ub_vec,
                    seed=seed,
                    seed_injection_config=seed_cfg,
                    problem_context=problem.problem_context,
                )
    finally:
        if log_handle is not None:
            log_handle.close()
    return np.asarray(result.get("archive_pf_fitness", true_pf), dtype=float), result


def _run_nnseed_worker(args: dict[str, Any]) -> dict[str, Any]:
    sample_idx = int(args["sample_idx"])
    seed = int(args["seed"])
    run_id = int(args["run_id"])
    max_iter = int(args["max_iter"])
    popnum = int(args["popnum"])
    init_nn_seed_count = int(args["init_nn_seed_count"])
    seed_device = str(args["seed_device"])
    cross_rate = float(args["cross_rate"])
    variate_rate = float(args["variate_rate"])
    archive_size = int(args["archive_size"])
    reinject_generations = tuple(int(x) for x in args["reinject_generations"])
    reinject_count = int(args["reinject_count"])
    hv_ref = np.asarray(args["hv_ref"], dtype=float).reshape(2)

    os.environ["RUNTIME_SAMPLE_INDEX"] = str(sample_idx)
    os.environ["CROSS_RATE"] = str(cross_rate)
    os.environ["VARIATE_RATE"] = str(variate_rate)
    os.environ["ARCHIVE_SIZE"] = str(archive_size)

    problem = build_typed_compare_problem(seed_device=seed_device)
    _configure_problem_context(problem, cross_rate, variate_rate, archive_size)

    seed_cfg = build_mogabka_seed_config(problem, init_nn_seed_count)
    seed_cfg["reinject_enabled"] = True
    seed_cfg["reinject_generations"] = reinject_generations
    seed_cfg["reinject_count"] = reinject_count

    t0 = time.perf_counter()
    log_path = Path(args["out_dir"]) / f"{str(args['prefix'])}_run{run_id:02d}_seed{seed}_nnseed_mogabka.log"
    front, result = _run_mogabka_front(problem, max_iter, popnum, seed, seed_cfg, log_path=log_path)
    elapsed_sec = time.perf_counter() - t0

    snap = _front_snapshot(front)
    out_dir = Path(args["out_dir"])
    run_prefix = str(args["prefix"]) + f"_run{run_id:02d}_seed{seed}"
    np.savetxt(out_dir / f"{run_prefix}_front_nnseed_mogabka.csv", front, delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / f"{run_prefix}_trace_nnseed_mogabka_hv.csv", np.asarray(result.get("HV", np.empty(0)), dtype=float), delimiter=",", fmt="%.17g")

    return {
        "run_id": run_id,
        "seed": seed,
        "algorithm": "NNseed-MOGABKA",
        "nd_count": int(snap["nd_count"]),
        "hv": _hv_fixed_ref_2d(front, hv_ref),
        "best_obj1": snap["best_obj1"],
        "best_obj2": snap["best_obj2"],
        "mean_obj1": snap["mean_obj1"],
        "mean_obj2": snap["mean_obj2"],
        "elapsed_sec": elapsed_sec,
        "front_path": str(out_dir / f"{run_prefix}_front_nnseed_mogabka.csv"),
    }


def _run_best_seed_compare(
    *,
    sample_idx: int,
    best_seed: int,
    best_run_id: int,
    prefix: str,
    max_iter: int,
    popnum: int,
    init_nn_seed_count: int,
    seed_device: str,
    cross_rate: float,
    variate_rate: float,
    archive_size: int,
    reinject_generations: tuple[int, ...],
    reinject_count: int,
    hv_ref: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    os.environ["RUNTIME_SAMPLE_INDEX"] = str(sample_idx)
    problem = build_typed_compare_problem(seed_device=seed_device)
    _configure_problem_context(problem, cross_rate, variate_rate, archive_size)

    rng_init = np.random.RandomState(best_seed)
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
        "variate_rate": float(variate_rate),
        "cross_rate": float(cross_rate),
        "obj_manager": np.asarray(shared_random_obj, dtype=float).copy(),
        "sol_manager": [shared_random_pop[i, :].copy() for i in range(shared_random_pop.shape[0])],
    }

    runs: list[tuple[str, str, np.ndarray, float, Any]] = []

    print(f"best seed compare: running NNseed-MOGABKA seed={best_seed}...", flush=True)
    t0 = time.perf_counter()
    nnseed_cfg = build_mogabka_seed_config(problem, init_nn_seed_count)
    nnseed_cfg["reinject_enabled"] = True
    nnseed_cfg["reinject_generations"] = reinject_generations
    nnseed_cfg["reinject_count"] = reinject_count
    best_prefix = f"{prefix}_best_run{best_run_id:02d}_seed{best_seed}"
    front_nnseed, result_nnseed = _run_mogabka_front(
        problem,
        max_iter,
        popnum,
        best_seed,
        nnseed_cfg,
        log_path=OUT_DIR / f"{best_prefix}_nnseed_mogabka.log",
    )
    runs.append(("NNseed-MOGABKA", "nnseed_mogabka", front_nnseed, time.perf_counter() - t0, result_nnseed))

    print("best seed compare: running MOGABKA random init...", flush=True)
    t0 = time.perf_counter()
    mogabka_cfg = {
        "enabled": True,
        "init_enabled": True,
        "init_nn_seed_count": 0,
        "initial_individuals": shared_random_pop.copy(),
        "builder_fn": None,
        "reinject_enabled": False,
    }
    front_mogabka, result_mogabka = _run_mogabka_front(
        problem,
        max_iter,
        popnum,
        best_seed,
        mogabka_cfg,
        log_path=OUT_DIR / f"{best_prefix}_mogabka.log",
    )
    runs.append(("MOGABKA", "mogabka", front_mogabka, time.perf_counter() - t0, result_mogabka))

    print("best seed compare: running HNSGA2-KMeans random init...", flush=True)
    t0 = time.perf_counter()
    hnsga_settings = dict(baseline_settings)
    hnsga_settings["force_kmeans_initial"] = False
    front_hnsga, trace_hnsga = HNSGA2_KMeans_funciton(
        settings=hnsga_settings,
        rng=np.random.RandomState(best_seed),
        return_trace=True,
    )
    runs.append(("HNSGA2-KMeans", "hnsga2_kmeans", np.asarray(front_hnsga, dtype=float), time.perf_counter() - t0, trace_hnsga))

    print("best seed compare: running NSGA2 random init...", flush=True)
    t0 = time.perf_counter()
    front_nsga2, trace_nsga2 = NSGA2_funciton(
        settings=dict(baseline_settings),
        rng=np.random.RandomState(best_seed),
        return_trace=True,
    )
    runs.append(("NSGA2", "nsga2", np.asarray(front_nsga2, dtype=float), time.perf_counter() - t0, trace_nsga2))

    print("best seed compare: running NSGA3 random init...", flush=True)
    t0 = time.perf_counter()
    front_nsga3, trace_nsga3 = NSGA3_funciton(
        settings=dict(baseline_settings),
        rng=np.random.RandomState(best_seed),
        return_trace=True,
    )
    runs.append(("NSGA3", "nsga3", np.asarray(front_nsga3, dtype=float), time.perf_counter() - t0, trace_nsga3))

    print("best seed compare: running MOBKA random init...", flush=True)
    t0 = time.perf_counter()
    front_mobka, trace_mobka = IMOBKA_funciton(
        settings=dict(baseline_settings),
        rng=np.random.RandomState(best_seed),
        return_trace=True,
    )
    runs.append(("MOBKA", "mobka", np.asarray(front_mobka, dtype=float), time.perf_counter() - t0, trace_mobka))

    labels = [label for label, _, _, _, _ in runs]
    fronts = [front for _, _, front, _, _ in runs]
    reference_pf = merge_fronts_and_extract_reference(fronts, invalid_penalty=problem.invalid_penalty)
    np.savetxt(OUT_DIR / f"{best_prefix}_reference_pf.csv", reference_pf, delimiter=",", fmt="%.17g")
    metrics_rows: list[dict[str, Any]] = []
    for label, key, front, elapsed_sec, _ in runs:
        np.savetxt(OUT_DIR / f"{best_prefix}_front_{key}.csv", front, delimiter=",", fmt="%.17g")
        snap = _front_snapshot(front)
        metrics_rows.append(
            {
                "run_id": best_run_id,
                "seed": best_seed,
                "algorithm": label,
                "front_key": key,
                "nd_count": int(snap["nd_count"]),
                "hv": _hv_fixed_ref_2d(front, hv_ref),
                "best_obj1": snap["best_obj1"],
                "best_obj2": snap["best_obj2"],
                "mean_obj1": snap["mean_obj1"],
                "mean_obj2": snap["mean_obj2"],
                "elapsed_sec": float(elapsed_sec),
            }
        )

    _write_csv(
        OUT_DIR / f"{best_prefix}_metrics.csv",
        metrics_rows,
        [
            "run_id",
            "seed",
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
    with (OUT_DIR / f"{best_prefix}_coverage.csv").open("w", encoding="utf-8") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, label in enumerate(labels):
            values = ",".join(f"{coverage_matrix[i, j]:.17g}" for j in range(len(labels)))
            f.write(f"{label},{values}\n")

    trace_map = {
        "nnseed_mogabka_hv": runs[0][4].get("HV", np.empty(0)),
        "mogabka_hv": runs[1][4].get("HV", np.empty(0)),
        "hnsga2_kmeans": trace_hnsga.generation_metrics,
        "nsga2": trace_nsga2.generation_metrics,
        "nsga3": trace_nsga3.generation_metrics,
        "mobka_archive_size": trace_mobka.archive_size_history,
    }
    for key, arr in trace_map.items():
        np.savetxt(OUT_DIR / f"{best_prefix}_trace_{key}.csv", np.asarray(arr, dtype=float), delimiter=",", fmt="%.17g")

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    markers = ["o", "P", "D", "s", "^", "d"]
    for (label, _, front, _, _), marker in zip(runs, markers):
        _plot_front(ax, front, label, marker)
    if reference_pf.size > 0:
        ax.scatter(reference_pf[:, 0], reference_pf[:, 1], s=14, label="Reference PF", marker=".", alpha=0.5)
    ax.set_xlabel("obj1")
    ax.set_ylabel("obj2")
    ax.set_title(f"Best NNseed-MOGABKA Run: run={best_run_id}, seed={best_seed}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{best_prefix}_fronts.png", dpi=220)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(labels))
    hv_vals = [float(row["hv"]) for row in metrics_rows]
    ax2.bar(x, hv_vals)
    ax2.set_ylabel("HV")
    ax2.set_title(f"HV on Best NNseed-MOGABKA Seed={best_seed}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=18, ha="right")
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / f"{best_prefix}_hv_bar.png", dpi=220)
    plt.close(fig2)

    return metrics_rows, coverage_matrix


def main() -> None:
    sample_idx = int(os.environ.get("RUNTIME_SAMPLE_INDEX", "1286"))
    max_iter = int(os.environ.get("MAX_ITER", "200"))
    popnum = int(os.environ.get("SEARCH_AGENTS_NO", "50"))
    init_nn_seed_count = int(os.environ.get("INIT_NN_SEED_COUNT", "12"))
    seed_device = os.environ.get("SEED_DEVICE", "cpu")
    base_seed = int(os.environ.get("COMPARE_SEED", "42"))
    run_count = int(os.environ.get("RUN_COUNT", "10"))
    max_workers = int(os.environ.get("MAX_WORKERS", "2"))
    cross_rate = float(os.environ.get("CROSS_RATE", "0.3"))
    variate_rate = float(os.environ.get("VARIATE_RATE", "0.8"))
    archive_size = int(os.environ.get("ARCHIVE_SIZE", "300"))
    reinject_generations = tuple(_parse_int_list(os.environ.get("REINJECT_GENERATIONS", "20,60")))
    reinject_count = int(os.environ.get("REINJECT_COUNT", "4"))
    hv_ref = np.asarray([float(x) for x in os.environ.get("HV_REF_POINT", "200,500000000").split(",")], dtype=float)
    prefix = os.environ.get("OUTPUT_PREFIX", f"nnseed_operator_injection_10runs_sample{sample_idx}")

    raw_seeds = os.environ.get("RUN_SEEDS", "").strip()
    seeds = _parse_int_list(raw_seeds) if raw_seeds else [base_seed + i for i in range(run_count)]
    if len(seeds) != run_count:
        raise ValueError(f"RUN_COUNT={run_count} but got {len(seeds)} seeds: {seeds}")

    print(
        "sweep settings: "
        f"sample_idx={sample_idx}, run_count={run_count}, seeds={seeds}, "
        f"max_iter={max_iter}, popnum={popnum}, init_nn_seed_count={init_nn_seed_count}, "
        f"cross_rate={cross_rate}, variate_rate={variate_rate}, archive_size={archive_size}, "
        f"reinject_generations={reinject_generations}, reinject_count={reinject_count}, "
        f"hv_ref={hv_ref.tolist()}, max_workers={max_workers}"
    )

    worker_args = [
        {
            "sample_idx": sample_idx,
            "seed": seed,
            "run_id": i + 1,
            "max_iter": max_iter,
            "popnum": popnum,
            "init_nn_seed_count": init_nn_seed_count,
            "seed_device": seed_device,
            "cross_rate": cross_rate,
            "variate_rate": variate_rate,
            "archive_size": archive_size,
            "reinject_generations": reinject_generations,
            "reinject_count": reinject_count,
            "hv_ref": hv_ref.tolist(),
            "out_dir": str(OUT_DIR),
            "prefix": prefix,
        }
        for i, seed in enumerate(seeds)
    ]

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_run_nnseed_worker, args): args for args in worker_args}
        for future in as_completed(future_map):
            args = future_map[future]
            row = future.result()
            rows.append(row)
            print(
                f"finished run={row['run_id']} seed={row['seed']} "
                f"hv={float(row['hv']):.6f} nd={row['nd_count']} "
                f"elapsed={float(row['elapsed_sec']):.2f}s",
                flush=True,
            )

    rows.sort(key=lambda r: int(r["run_id"]))
    _write_csv(
        OUT_DIR / f"{prefix}_sweep_metrics.csv",
        rows,
        [
            "run_id",
            "seed",
            "algorithm",
            "nd_count",
            "hv",
            "best_obj1",
            "best_obj2",
            "mean_obj1",
            "mean_obj2",
            "elapsed_sec",
            "front_path",
        ],
    )

    best_row = max(rows, key=lambda r: float(r["hv"]))
    best_seed = int(best_row["seed"])
    best_run_id = int(best_row["run_id"])
    print(
        f"\nBest NNseed-MOGABKA: run={best_run_id}, seed={best_seed}, "
        f"hv={float(best_row['hv']):.6f}, nd={best_row['nd_count']}"
    )

    best_metrics, coverage_matrix = _run_best_seed_compare(
        sample_idx=sample_idx,
        best_seed=best_seed,
        best_run_id=best_run_id,
        prefix=prefix,
        max_iter=max_iter,
        popnum=popnum,
        init_nn_seed_count=init_nn_seed_count,
        seed_device=seed_device,
        cross_rate=cross_rate,
        variate_rate=variate_rate,
        archive_size=archive_size,
        reinject_generations=reinject_generations,
        reinject_count=reinject_count,
        hv_ref=hv_ref,
    )

    print("\nBest-seed compare HV summary:")
    for row in sorted(best_metrics, key=lambda r: -float(r["hv"])):
        print(
            f"{row['algorithm']}: hv={float(row['hv']):.6f}, "
            f"nd={int(row['nd_count'])}, elapsed={float(row['elapsed_sec']):.2f}s"
        )

    print("\nCoverage matrix for best seed:")
    print(np.array2string(coverage_matrix, precision=4, suppress_small=False))
    print(f"\nTotal elapsed_sec={time.perf_counter() - t0:.2f}")
    print("Saved outputs with prefix:", OUT_DIR / prefix)


if __name__ == "__main__":
    main()
