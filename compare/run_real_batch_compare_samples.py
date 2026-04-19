from __future__ import annotations

import csv
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

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


OUT_DIR = Path(__file__).resolve().parent
POI_KMEANS_DIR = ROOT_PROJECT_DIR.parent / "poi kmeans"
DEFAULT_DEBUG_MAT_PATH = POI_KMEANS_DIR / "dataset_v3_50_debug.mat"
DEFAULT_META_CSV = POI_KMEANS_DIR / "meta_samples_50.csv"


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

    order = np.argsort(nd[:, 0])
    nd = nd[order, :]

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
            "conv_best_obj1": float("inf"),
            "conv_best_obj2": float("inf"),
            "conv_mean_obj1": float("inf"),
            "conv_mean_obj2": float("inf"),
        }

    if front.ndim == 1:
        front = front.reshape(1, -1)

    return {
        "nd_count": float(front.shape[0]),
        "conv_best_obj1": float(np.min(front[:, 0])),
        "conv_best_obj2": float(np.min(front[:, 1])),
        "conv_mean_obj1": float(np.mean(front[:, 0])),
        "conv_mean_obj2": float(np.mean(front[:, 1])),
    }


def _load_meta_lookup(meta_csv: Path) -> dict[int, dict[str, str]]:
    lookup: dict[int, dict[str, str]] = {}
    with meta_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            lookup[idx] = row
    return lookup


def _sample_info(meta_lookup: dict[int, dict[str, str]], sample_idx: int) -> dict[str, Any]:
    row = meta_lookup[int(sample_idx)]
    return {
        "source_file": row["source_file"],
        "scale_km": float(row["scale_km"]),
        "offset_id": int(float(row["offset_id"])),
        "cell_rank": int(float(row["cell_rank"])),
        "cell_lon_center": float(row["cell_lon_center"]),
        "cell_lat_center": float(row["cell_lat_center"]),
        "group_tag": row.get("group_tag", ""),
    }


def _run_single_sample(
    sample_idx: int,
    meta_lookup: dict[int, dict[str, str]],
    debug_mat_path: Path,
    seed_device: str,
    popnum: int,
    max_iter: int,
    init_nn_seed_count: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    os.environ["RUNTIME_DEBUG_MAT_PATH"] = str(debug_mat_path)
    os.environ["RUNTIME_SAMPLE_INDEX"] = str(sample_idx)

    info = _sample_info(meta_lookup, sample_idx)
    problem = build_typed_compare_problem(seed_device=seed_device)

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
        "variate_rate": float(os.environ.get("VARIATE_RATE", "0.3")),
        "cross_rate": float(os.environ.get("CROSS_RATE", "0.8")),
        "neighbor_size": int(os.environ.get("MOEAD_NEIGHBOR_SIZE", max(2, int(round(0.1 * popnum))))),
        "obj_manager": np.asarray(shared_random_obj, dtype=float).copy(),
        "sol_manager": [shared_random_pop[i, :].copy() for i in range(shared_random_pop.shape[0])],
    }

    runs: list[tuple[str, np.ndarray, float]] = []

    t0 = time.perf_counter()
    seed_cfg = build_mogabka_seed_config(problem, init_nn_seed_count)
    with activated_ev_context(problem):
        _, _, ture_pf, result = MOGABKA(
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
    front_imogabka_seed = np.asarray(result.get("archive_pf_fitness", ture_pf), dtype=float)
    runs.append(("IMOGABKA-seed", front_imogabka_seed, time.perf_counter() - t0))

    t0 = time.perf_counter()
    front_mobka, _ = IMOBKA_funciton(
        settings=dict(baseline_settings),
        rng=np.random.RandomState(base_seed),
        return_trace=True,
    )
    runs.append(("MOBKA", np.asarray(front_mobka, dtype=float), time.perf_counter() - t0))

    t0 = time.perf_counter()
    front_nsga2, _ = NSGA2_funciton(
        settings=dict(baseline_settings),
        rng=np.random.RandomState(base_seed),
        return_trace=True,
    )
    runs.append(("NSGA2", np.asarray(front_nsga2, dtype=float), time.perf_counter() - t0))

    t0 = time.perf_counter()
    front_nsga3, _ = NSGA3_funciton(
        settings=dict(baseline_settings),
        rng=np.random.RandomState(base_seed),
        return_trace=True,
    )
    runs.append(("NSGA3", np.asarray(front_nsga3, dtype=float), time.perf_counter() - t0))

    t0 = time.perf_counter()
    front_moead, _ = MOEAD_function(
        settings=dict(baseline_settings),
        rng=np.random.RandomState(base_seed),
        return_trace=True,
    )
    runs.append(("MOEAD", np.asarray(front_moead, dtype=float), time.perf_counter() - t0))

    algo_order = [algorithm for algorithm, _, _ in runs]
    fronts = [front for _, front, _ in runs]
    reference_pf = merge_fronts_and_extract_reference(
        fronts,
        invalid_penalty=problem.invalid_penalty,
    )
    coverage_rows: list[dict[str, Any]] = []
    for i, algo_a in enumerate(algo_order):
        for j, algo_b in enumerate(algo_order):
            coverage_rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "algorithm_a": algo_a,
                    "algorithm_b": algo_b,
                    "coverage": 1.0 if i == j else _safe_coverage(fronts[i], fronts[j]),
                }
            )

    out_rows: list[dict[str, Any]] = []
    for algorithm, front, elapsed_sec in runs:
        snap = _front_snapshot(front)
        out_rows.append(
            {
                "sample_idx": int(sample_idx),
                "source_file": info["source_file"],
                "scale_km": info["scale_km"],
                "offset_id": info["offset_id"],
                "cell_rank": info["cell_rank"],
                "cell_lon_center": info["cell_lon_center"],
                "cell_lat_center": info["cell_lat_center"],
                "group_tag": info["group_tag"],
                "algorithm": algorithm,
                "hv": _safe_hv(front, reference_pf),
                "nd_count": int(snap["nd_count"]),
                "converge_best_obj1": snap["conv_best_obj1"],
                "converge_best_obj2": snap["conv_best_obj2"],
                "converge_mean_obj1": snap["conv_mean_obj1"],
                "converge_mean_obj2": snap["conv_mean_obj2"],
                "elapsed_sec": float(elapsed_sec),
            }
        )
    return out_rows, coverage_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_by_algorithm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_algo: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_algo.setdefault(str(row["algorithm"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for algo, algo_rows in by_algo.items():
        hv_vals = [float(r["hv"]) for r in algo_rows]
        nd_vals = [int(r["nd_count"]) for r in algo_rows]
        best1_vals = [float(r["converge_best_obj1"]) for r in algo_rows]
        best2_vals = [float(r["converge_best_obj2"]) for r in algo_rows]
        mean1_vals = [float(r["converge_mean_obj1"]) for r in algo_rows]
        mean2_vals = [float(r["converge_mean_obj2"]) for r in algo_rows]
        sec_vals = [float(r["elapsed_sec"]) for r in algo_rows]
        summary_rows.append(
            {
                "algorithm": algo,
                "runs": len(algo_rows),
                "hv_mean": float(np.mean(hv_vals)),
                "hv_median": float(np.median(hv_vals)),
                "hv_std": float(np.std(hv_vals)),
                "nd_count_mean": float(np.mean(nd_vals)),
                "best_obj1_mean": float(np.mean(best1_vals)),
                "best_obj2_mean": float(np.mean(best2_vals)),
                "mean_obj1_mean": float(np.mean(mean1_vals)),
                "mean_obj2_mean": float(np.mean(mean2_vals)),
                "elapsed_sec_mean": float(np.mean(sec_vals)),
                "elapsed_sec_median": float(statistics.median(sec_vals)),
            }
        )
    summary_rows.sort(key=lambda r: (-float(r["hv_mean"]), str(r["algorithm"])))
    return summary_rows


def _build_hv_pivot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sample_map: dict[int, dict[str, Any]] = {}
    algo_order = ["IMOGABKA-seed", "MOBKA", "NSGA2", "NSGA3", "MOEAD"]
    for row in rows:
        sample_idx = int(row["sample_idx"])
        item = sample_map.setdefault(
            sample_idx,
            {
                "sample_idx": sample_idx,
                "source_file": row["source_file"],
                "offset_id": row["offset_id"],
                "cell_rank": row["cell_rank"],
                "cell_lon_center": row["cell_lon_center"],
                "cell_lat_center": row["cell_lat_center"],
            },
        )
        item[f"hv_{row['algorithm']}"] = float(row["hv"])
        item[f"nd_{row['algorithm']}"] = int(row["nd_count"])
    out = [sample_map[k] for k in sorted(sample_map.keys())]
    for item in out:
        hv_pairs = [(algo, item.get(f"hv_{algo}", float("-inf"))) for algo in algo_order]
        hv_pairs.sort(key=lambda kv: (-float(kv[1]), kv[0]))
        item["best_hv_algorithm"] = hv_pairs[0][0]
        item["best_hv"] = float(hv_pairs[0][1])
    return out


def _aggregate_coverage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pair_map: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row["algorithm_a"]), str(row["algorithm_b"]))
        pair_map.setdefault(key, []).append(float(row["coverage"]))

    out: list[dict[str, Any]] = []
    for (algo_a, algo_b), vals in sorted(pair_map.items()):
        out.append(
            {
                "algorithm_a": algo_a,
                "algorithm_b": algo_b,
                "coverage_mean": float(np.mean(vals)),
                "coverage_median": float(np.median(vals)),
                "coverage_std": float(np.std(vals)),
                "runs": len(vals),
            }
        )
    return out


def main() -> None:
    sample_start = int(os.environ.get("SAMPLE_START", "1030"))
    sample_end = int(os.environ.get("SAMPLE_END", "1067"))
    max_iter = int(os.environ.get("MAX_ITER", "50"))
    popnum = int(os.environ.get("SEARCH_AGENTS_NO", "50"))
    init_nn_seed_count = int(os.environ.get("INIT_NN_SEED_COUNT", "12"))
    seed_device = os.environ.get("SEED_DEVICE", "cpu")
    base_seed = int(os.environ.get("COMPARE_SEED", "42"))
    debug_mat_path = Path(os.environ.get("RUNTIME_DEBUG_MAT_PATH", str(DEFAULT_DEBUG_MAT_PATH)))
    meta_csv = Path(os.environ.get("META_CSV_PATH", str(DEFAULT_META_CSV)))

    if not debug_mat_path.is_file():
        raise FileNotFoundError(f"debug mat not found: {debug_mat_path}")
    if not meta_csv.is_file():
        raise FileNotFoundError(f"meta csv not found: {meta_csv}")

    meta_lookup = _load_meta_lookup(meta_csv)

    all_rows: list[dict[str, Any]] = []
    all_coverage_rows: list[dict[str, Any]] = []
    total_samples = sample_end - sample_start + 1
    batch_t0 = time.perf_counter()

    for offset, sample_idx in enumerate(range(sample_start, sample_end + 1), start=1):
        print(f"[{offset}/{total_samples}] running sample_idx={sample_idx} ...", flush=True)
        rows, coverage_rows = _run_single_sample(
            sample_idx=sample_idx,
            meta_lookup=meta_lookup,
            debug_mat_path=debug_mat_path,
            seed_device=seed_device,
            popnum=popnum,
            max_iter=max_iter,
            init_nn_seed_count=init_nn_seed_count,
            base_seed=base_seed,
        )
        all_rows.extend(rows)
        all_coverage_rows.extend(coverage_rows)

        hv_map = {row["algorithm"]: float(row["hv"]) for row in rows}
        hv_sorted = sorted(hv_map.items(), key=lambda kv: (-kv[1], kv[0]))
        print(
            "    HV ranking: "
            + ", ".join(f"{algo}={hv:.4f}" for algo, hv in hv_sorted),
            flush=True,
        )

    detailed_path = OUT_DIR / f"real_batch_samples_{sample_start}_{sample_end}_detailed.csv"
    summary_path = OUT_DIR / f"real_batch_samples_{sample_start}_{sample_end}_summary_by_algorithm.csv"
    hv_pivot_path = OUT_DIR / f"real_batch_samples_{sample_start}_{sample_end}_hv_pivot.csv"
    coverage_long_path = OUT_DIR / f"real_batch_samples_{sample_start}_{sample_end}_coverage_long.csv"
    coverage_summary_path = OUT_DIR / f"real_batch_samples_{sample_start}_{sample_end}_coverage_summary.csv"

    detailed_fields = [
        "sample_idx",
        "source_file",
        "scale_km",
        "offset_id",
        "cell_rank",
        "cell_lon_center",
        "cell_lat_center",
        "group_tag",
        "algorithm",
        "hv",
        "nd_count",
        "converge_best_obj1",
        "converge_best_obj2",
        "converge_mean_obj1",
        "converge_mean_obj2",
        "elapsed_sec",
    ]
    _write_csv(detailed_path, all_rows, detailed_fields)

    coverage_fields = ["sample_idx", "algorithm_a", "algorithm_b", "coverage"]
    _write_csv(coverage_long_path, all_coverage_rows, coverage_fields)

    summary_rows = _aggregate_by_algorithm(all_rows)
    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    if summary_fields:
        _write_csv(summary_path, summary_rows, summary_fields)

    coverage_summary_rows = _aggregate_coverage_rows(all_coverage_rows)
    coverage_summary_fields = list(coverage_summary_rows[0].keys()) if coverage_summary_rows else []
    if coverage_summary_fields:
        _write_csv(coverage_summary_path, coverage_summary_rows, coverage_summary_fields)

    hv_pivot_rows = _build_hv_pivot(all_rows)
    pivot_fields = [
        "sample_idx",
        "source_file",
        "offset_id",
        "cell_rank",
        "cell_lon_center",
        "cell_lat_center",
        "hv_IMOGABKA-seed",
        "hv_MOBKA",
        "hv_NSGA2",
        "hv_NSGA3",
        "hv_MOEAD",
        "nd_IMOGABKA-seed",
        "nd_MOBKA",
        "nd_NSGA2",
        "nd_NSGA3",
        "nd_MOEAD",
        "best_hv_algorithm",
        "best_hv",
    ]
    _write_csv(hv_pivot_path, hv_pivot_rows, pivot_fields)

    print("\nsummary by algorithm:")
    for row in summary_rows:
        print(
            f"  {row['algorithm']}: hv_mean={row['hv_mean']:.6f}, "
            f"hv_median={row['hv_median']:.6f}, nd_count_mean={row['nd_count_mean']:.2f}"
        )

    elapsed = time.perf_counter() - batch_t0
    print(f"\nsaved detailed rows to: {detailed_path}")
    print(f"saved summary rows to:  {summary_path}")
    print(f"saved hv pivot to:      {hv_pivot_path}")
    print(f"saved coverage rows to: {coverage_long_path}")
    print(f"saved coverage avg to:  {coverage_summary_path}")
    print(f"total elapsed_sec={elapsed:.2f}")


if __name__ == "__main__":
    main()
