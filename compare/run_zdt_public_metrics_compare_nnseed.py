from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


ROOT_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_PROJECT_DIR))

from gd import GD
from igd import IGD
from mogabka_seeded import MOGABKA
from spacing import Spacing
from coverage import Coverage
from hv import HV
from compare.run_zdt_public_metrics_compare import (
    _dedup_front,
    _problem_spec,
    _run_mobka_zdt,
    _run_moead_zdt,
    _run_nsga2_zdt,
    _run_nsga3_zdt,
)
from compare.zdt_metrics_plot import zdt_true_pf
from compare.zdt_nn_seed import make_zdt_nn_seed_builder


def _safe_gd(front: np.ndarray, true_pf: np.ndarray) -> float:
    if front.shape[0] == 0 or true_pf.shape[0] == 0:
        return float("inf")
    return float(GD(front, true_pf))


def _safe_igd(front: np.ndarray, true_pf: np.ndarray) -> float:
    if front.shape[0] == 0 or true_pf.shape[0] == 0:
        return float("inf")
    return float(IGD(front, true_pf))


def _safe_spacing(front: np.ndarray, true_pf: np.ndarray) -> float:
    if front.shape[0] <= 1:
        return 0.0
    return float(Spacing(front, true_pf))


def _safe_hv(front: np.ndarray, true_pf: np.ndarray) -> float:
    if front.shape[0] == 0 or true_pf.shape[0] == 0:
        return 0.0
    return float(HV(front, true_pf)[0])


def _safe_coverage(front_a: np.ndarray, front_b: np.ndarray) -> float:
    if front_a.shape[0] == 0 or front_b.shape[0] == 0:
        return 0.0
    return float(Coverage(front_b, front_a))


def _run_nnseed_imogabka_zdt(
    prob_id: int,
    popnum: int,
    max_iter: int,
    base_seed: int,
) -> np.ndarray:
    dim, lb, ub = _problem_spec(prob_id)
    seed_device = os.environ.get("SEED_DEVICE", "cpu")
    init_nn_seed_count = int(os.environ.get("INIT_NN_SEED_COUNT", "12"))
    tail_noise_std = float(os.environ.get("ZDT_NNSEED_TAIL_NOISE_STD", "0.01" if prob_id != 4 else "0.03"))

    seed_builder = make_zdt_nn_seed_builder(
        prob_id=prob_id,
        dim=dim,
        lb_vec=lb,
        ub_vec=ub,
        device=seed_device,
        train_seed=base_seed,
        tail_noise_std=tail_noise_std,
    )
    seed_cfg = {
        "enabled": True,
        "init_enabled": True,
        "init_nn_seed_count": init_nn_seed_count,
        "builder_fn": seed_builder,
        "reinject_enabled": False,
        "reinject_generations": (20, 60),
        "reinject_count": 4,
    }
    problem_context = {
        "debug_seed_injection": bool(int(os.environ.get("DEBUG_SEED_INJECTION", "0"))),
    }

    _, _, ture_pf, result = MOGABKA(
        Max_iter=max_iter,
        SearchAgents_no=popnum,
        FUN=f"ZDT{prob_id}",
        dim=dim,
        numObj=2,
        lb=lb,
        ub=ub,
        seed=base_seed,
        seed_injection_config=seed_cfg,
        problem_context=problem_context,
    )
    front = np.asarray(result.get("archive_pf_fitness", ture_pf), dtype=np.float64)
    return _dedup_front(front)


def _plot_problem_fronts(out_png: Path, prob_id: int, true_pf: np.ndarray, front_map: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(true_pf[:, 0], true_pf[:, 1], "k.", markersize=3, label="True PF")

    markers = {
        "NN-seed IMOGABKA": "o",
        "MOBKA": "d",
        "NSGA-II": "s",
        "NSGA-III": "^",
        "MOEA/D": "x",
    }
    for label, front in front_map.items():
        if front.size == 0:
            continue
        ax.scatter(front[:, 0], front[:, 1], s=18, marker=markers[label], label=label)

    ax.set_title(f"ZDT{prob_id} Public Benchmark (NN-seed IMOGABKA)")
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    max_iter = int(os.environ.get("MAX_ITER", "50"))
    popnum = int(os.environ.get("SEARCH_AGENTS_NO", "50"))
    base_seed = int(os.environ.get("COMPARE_SEED", "42"))
    problems = [int(x.strip()) for x in os.environ.get("ZDT_PROBLEMS", "1,2,3,4").split(",") if x.strip()]

    algo_specs: list[tuple[str, Callable[[int, int, int, np.random.RandomState | int], np.ndarray]]] = [
        ("NN-seed IMOGABKA", lambda prob_id, pop, iters, seed_or_rng: _run_nnseed_imogabka_zdt(prob_id, pop, iters, int(seed_or_rng))),
        ("MOBKA", lambda prob_id, pop, iters, seed_or_rng: _run_mobka_zdt(prob_id, pop, iters, seed_or_rng)),
        ("NSGA-II", lambda prob_id, pop, iters, seed_or_rng: _run_nsga2_zdt(prob_id, pop, iters, seed_or_rng)),
        ("NSGA-III", lambda prob_id, pop, iters, seed_or_rng: _run_nsga3_zdt(prob_id, pop, iters, seed_or_rng)),
        ("MOEA/D", lambda prob_id, pop, iters, seed_or_rng: _run_moead_zdt(prob_id, pop, iters, seed_or_rng)),
    ]

    metrics_rows: list[list[object]] = []
    mean_rows: list[list[object]] = []

    for prob_id in problems:
        true_pf = zdt_true_pf(prob_id, 1000)
        front_map: dict[str, np.ndarray] = {}

        print(f"running ZDT{prob_id} with popnum={popnum}, max_iter={max_iter}")
        for algo_idx, (label, runner) in enumerate(algo_specs):
            algo_seed = base_seed + prob_id * 100 + algo_idx
            print(f"  {label}...", flush=True)
            if label == "NN-seed IMOGABKA":
                front = runner(prob_id, popnum, max_iter, algo_seed)
            else:
                front = runner(prob_id, popnum, max_iter, np.random.RandomState(algo_seed))

            front = _dedup_front(front)
            front_map[label] = front
            file_label = label.lower().replace("/", "").replace("-", "").replace(" ", "_")
            np.savetxt(
                out_dir / f"zdt{prob_id}_front_{file_label}.csv",
                front,
                delimiter=",",
                fmt="%.17g",
            )

            metrics_rows.append(
                [
                    f"ZDT{prob_id}",
                    label,
                    int(front.shape[0]),
                    _safe_gd(front, true_pf),
                    _safe_igd(front, true_pf),
                    _safe_spacing(front, true_pf),
                ]
            )

        _plot_problem_fronts(out_dir / f"zdt{prob_id}_public_compare_nnseed.png", prob_id, true_pf, front_map)

        with open(out_dir / f"zdt{prob_id}_hv_coverage_aux_nnseed.csv", "w", encoding="utf-8") as f:
            labels = list(front_map.keys())
            f.write("metric,algorithm,value\n")
            for label in labels:
                f.write(f"hv,{label},{_safe_hv(front_map[label], true_pf):.17g}\n")
            for a in labels:
                for b in labels:
                    value = 1.0 if a == b else _safe_coverage(front_map[a], front_map[b])
                    f.write(f"coverage_{a}_over_{b},{a},{value:.17g}\n")

    metrics_arr = {}
    for row in metrics_rows:
        metrics_arr.setdefault(row[1], []).append(row)

    for label, rows in metrics_arr.items():
        gd_vals = np.array([float(r[3]) for r in rows], dtype=np.float64)
        igd_vals = np.array([float(r[4]) for r in rows], dtype=np.float64)
        spacing_vals = np.array([float(r[5]) for r in rows], dtype=np.float64)
        mean_rows.append(
            [
                label,
                float(np.mean(gd_vals)),
                float(np.mean(igd_vals)),
                float(np.mean(spacing_vals)),
            ]
        )

    with open(out_dir / "zdt_public_metrics_nnseed.csv", "w", encoding="utf-8") as f:
        f.write("problem,algorithm,nd_count,gd,igd,spacing\n")
        for row in metrics_rows:
            f.write(
                f"{row[0]},{row[1]},{int(row[2])},{float(row[3]):.17g},{float(row[4]):.17g},{float(row[5]):.17g}\n"
            )

    with open(out_dir / "zdt_public_metrics_nnseed_mean.csv", "w", encoding="utf-8") as f:
        f.write("algorithm,mean_gd,mean_igd,mean_spacing\n")
        for row in mean_rows:
            f.write(f"{row[0]},{float(row[1]):.17g},{float(row[2]):.17g},{float(row[3]):.17g}\n")

    print("\nZDT nn-seed metric summary:")
    for row in mean_rows:
        print(
            f"{row[0]}: mean_gd={float(row[1]):.6g} "
            f"mean_igd={float(row[2]):.6g} mean_spacing={float(row[3]):.6g}"
        )
    print("\nSaved ZDT nn-seed benchmark outputs to:", out_dir)


if __name__ == "__main__":
    main()
