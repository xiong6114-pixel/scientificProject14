from __future__ import annotations

import math
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
from compare.mobka_common import (
    Particle,
    cgrid,
    delete_one_rep_member,
    deter_dom,
    determine_domination,
    dominates_particle,
    fgrid,
    rep_to_obj_matrix,
    select_leader,
)
from compare.ndsort_crowding import fast_non_dominated_sort_with_crowding
from compare.nsga3_matlab_aligned import fast_non_dominated_sort_with_reference_points
from compare.zdt_metrics_plot import zdt_true_pf


def _problem_spec(prob_id: int) -> tuple[int, np.ndarray, np.ndarray]:
    dim_default = int(os.environ.get("ZDT_DIM", "30"))
    zdt4_dim = int(os.environ.get("ZDT4_DIM", "10"))

    if prob_id == 4:
        dim = zdt4_dim
        lb = np.full(dim, -5.0, dtype=np.float64)
        ub = np.full(dim, 5.0, dtype=np.float64)
        lb[0] = 0.0
        ub[0] = 1.0
        return dim, lb, ub

    dim = dim_default
    lb = np.zeros(dim, dtype=np.float64)
    ub = np.ones(dim, dtype=np.float64)
    return dim, lb, ub


def _zdt_cost(x: np.ndarray, prob_id: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    d = x.size
    f1 = np.float64(x[0])

    if prob_id == 1:
        g = np.float64(1.0 + 9.0 * np.sum(x[1:]) / (d - 1))
        f2 = np.float64(g * (1.0 - np.sqrt(f1 / g)))
    elif prob_id == 2:
        g = np.float64(1.0 + 9.0 * np.sum(x[1:]) / (d - 1))
        f2 = np.float64(g * (1.0 - (f1 / g) ** 2))
    elif prob_id == 3:
        g = np.float64(1.0 + 9.0 * np.sum(x[1:]) / (d - 1))
        ratio = np.float64(f1 / g)
        f2 = np.float64(g * (1.0 - np.sqrt(ratio) - ratio * np.sin(10.0 * np.pi * f1)))
    elif prob_id == 4:
        g = np.float64(1.0 + 10.0 * (d - 1) + np.sum(x[1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[1:])))
        f2 = np.float64(g * (1.0 - np.sqrt(f1 / g)))
    else:
        raise ValueError(f"Unsupported ZDT problem: {prob_id}")

    return np.array([f1, f2], dtype=np.float64)


def _sample_uniform(pop_size: int, lb: np.ndarray, ub: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    u = rng.rand(pop_size, lb.size)
    return (u * (ub - lb).reshape(1, -1) + lb.reshape(1, -1)).astype(np.float64)


def _reset_mutation(sol: np.ndarray, lb: np.ndarray, ub: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    child = np.asarray(sol, dtype=np.float64).reshape(-1).copy()
    idx = int(rng.randint(0, child.size))
    child[idx] = np.float64(lb[idx] + rng.rand() * (ub[idx] - lb[idx]))
    return child


def _single_dim_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    child = np.asarray(a, dtype=np.float64).reshape(-1).copy()
    parent = np.asarray(b, dtype=np.float64).reshape(-1)
    idx = int(rng.randint(0, child.size))
    child[idx] = parent[idx]
    return child


def _dedup_front(front: np.ndarray) -> np.ndarray:
    front = np.asarray(front, dtype=np.float64)
    if front.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    unique = np.unique(front, axis=0)
    return unique[np.argsort(unique[:, 0]), :]


def _extract_first_front(objs: np.ndarray) -> np.ndarray:
    objs = np.asarray(objs, dtype=np.float64)
    if objs.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    fronts, _ = fast_non_dominated_sort_with_crowding(objs)
    if len(fronts) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return _dedup_front(objs[fronts[0], :])


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


def _run_mobka_zdt(
    prob_id: int,
    popnum: int,
    max_iter: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    dim, lb, ub = _problem_spec(prob_id)

    pop = []
    init_pop = _sample_uniform(popnum, lb, ub, rng)
    for i in range(popnum):
        pos = init_pop[i, :].copy()
        cost = _zdt_cost(pos, prob_id)
        pop.append(
            Particle(
                Position=pos.copy(),
                Cost=cost.copy(),
                BestPosition=pos.copy(),
                BestCost=cost.copy(),
            )
        )

    n_rep = 200
    n_grid = 6
    alpha = 0.1
    beta = 1.5
    gamma = 2
    p = 0.8

    pop = deter_dom(pop)
    rep = [pp for pp in pop if not pp.IsDominated]
    grid = cgrid(rep, n_grid, alpha)
    for i in range(len(rep)):
        rep[i] = fgrid(rep[i], grid)

    for it in range(1, max_iter + 1):
        for i in range(popnum):
            leader = select_leader(rep, beta, rng)

            r = np.float64(rng.rand())
            n = np.float64(0.05 * np.exp(-2 * (it / max_iter) ** 2))

            if p < r:
                candidate = pop[i].Position + n * (1 + np.sin(r)) * pop[i].Position
            else:
                candidate = pop[i].Position * (n * (2 * rng.rand(dim) - 1) + 1)

            m = np.float64(2 * np.sin(r + np.pi / 2))
            s_idx = int(rng.randint(0, popnum))
            cauchy_value = np.tan((rng.rand(dim) - 0.5) * np.pi)

            if dominates_particle(pop[i], pop[s_idx]):
                candidate = candidate + cauchy_value * (pop[i].BestPosition - leader.Position)
            else:
                candidate = candidate + cauchy_value * (leader.Position - m * pop[i].BestPosition)

            candidate = np.clip(candidate, lb, ub)
            pop[i].Position = candidate
            pop[i].Cost = _zdt_cost(candidate, prob_id)

        pop = deter_dom(pop)
        rep = rep + [pp for pp in pop if not pp.IsDominated]
        rep = determine_domination(rep)
        rep = [pp for pp in rep if not pp.IsDominated]

        grid = cgrid(rep, n_grid, alpha)
        for i in range(len(rep)):
            rep[i] = fgrid(rep[i], grid)

        if len(rep) > n_rep:
            extra = len(rep) - n_rep
            for _ in range(extra):
                rep = delete_one_rep_member(rep, gamma, rng)

    return _dedup_front(rep_to_obj_matrix(rep))


def _run_nsga2_zdt(
    prob_id: int,
    popnum: int,
    max_iter: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    _, lb, ub = _problem_spec(prob_id)
    population = _sample_uniform(popnum, lb, ub, rng)
    pop_objs = np.vstack([_zdt_cost(population[i, :], prob_id) for i in range(popnum)]).astype(np.float64)

    variate_rate = np.float64(0.3)
    cross_rate = np.float64(0.8)

    for _ in range(max_iter):
        obj_manager = pop_objs.copy()
        sol_manager = [population[i, :].copy() for i in range(population.shape[0])]

        for i in range(popnum):
            if rng.rand() < variate_rate:
                new_sol = _reset_mutation(population[i, :], lb, ub, rng)
                obj_manager = np.vstack([obj_manager, _zdt_cost(new_sol, prob_id).reshape(1, 2)])
                sol_manager.append(new_sol)

            if rng.rand() < cross_rate:
                rand_id = int(rng.randint(0, popnum))
                new_sol = _single_dim_crossover(population[i, :], population[rand_id, :], rng)
                if rng.rand() < 0.25:
                    new_sol = _reset_mutation(new_sol, lb, ub, rng)
                obj_manager = np.vstack([obj_manager, _zdt_cost(new_sol, prob_id).reshape(1, 2)])
                sol_manager.append(new_sol)

        fronts, crowding_dist = fast_non_dominated_sort_with_crowding(obj_manager)
        new_population = []
        new_objs = np.empty((0, 2), dtype=np.float64)
        for front in fronts:
            idx = np.argsort(crowding_dist[front])[::-1]
            for j in idx:
                sol_id = int(front[j])
                new_population.append(sol_manager[sol_id].copy())
                new_objs = np.vstack([new_objs, obj_manager[sol_id, :].reshape(1, 2)])
                if len(new_population) >= popnum:
                    break
            if len(new_population) >= popnum:
                break

        population = np.vstack([new_population[i].reshape(1, -1) for i in range(len(new_population))]).astype(np.float64)
        pop_objs = new_objs.copy()

    return _extract_first_front(pop_objs)


def _run_nsga3_zdt(
    prob_id: int,
    popnum: int,
    max_iter: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    _, lb, ub = _problem_spec(prob_id)
    population = _sample_uniform(popnum, lb, ub, rng)
    pop_objs = np.vstack([_zdt_cost(population[i, :], prob_id) for i in range(popnum)]).astype(np.float64)

    variate_rate = np.float64(0.3)
    cross_rate = np.float64(0.8)

    for _ in range(max_iter):
        obj_manager = pop_objs.copy()
        sol_manager = [population[i, :].copy() for i in range(population.shape[0])]

        for i in range(popnum):
            if rng.rand() < variate_rate:
                new_sol = _reset_mutation(population[i, :], lb, ub, rng)
                obj_manager = np.vstack([obj_manager, _zdt_cost(new_sol, prob_id).reshape(1, 2)])
                sol_manager.append(new_sol)

            if rng.rand() < cross_rate:
                rand_id = int(rng.randint(0, popnum))
                new_sol = _single_dim_crossover(population[i, :], population[rand_id, :], rng)
                if rng.rand() < 0.25:
                    new_sol = _reset_mutation(new_sol, lb, ub, rng)
                obj_manager = np.vstack([obj_manager, _zdt_cost(new_sol, prob_id).reshape(1, 2)])
                sol_manager.append(new_sol)

        fronts, niche_score = fast_non_dominated_sort_with_reference_points(obj_manager, max(obj_manager.shape[0], popnum))
        new_population = []
        new_objs = np.empty((0, 2), dtype=np.float64)
        for front in fronts:
            idx = np.argsort(niche_score[front])[::-1]
            for j in idx:
                sol_id = int(front[j])
                new_population.append(sol_manager[sol_id].copy())
                new_objs = np.vstack([new_objs, obj_manager[sol_id, :].reshape(1, 2)])
                if len(new_population) >= popnum:
                    break
            if len(new_population) >= popnum:
                break

        population = np.vstack([new_population[i].reshape(1, -1) for i in range(len(new_population))]).astype(np.float64)
        pop_objs = new_objs.copy()

    return _extract_first_front(pop_objs)


def _run_moead_zdt(
    prob_id: int,
    popnum: int,
    max_iter: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    _, lb, ub = _problem_spec(prob_id)
    population = _sample_uniform(popnum, lb, ub, rng)
    pop_objs = np.vstack([_zdt_cost(population[i, :], prob_id) for i in range(popnum)]).astype(np.float64)

    t = max(2, int(np.round(0.1 * popnum)))
    t = min(t, popnum)
    variate_rate = np.float64(0.3)
    cross_rate = np.float64(0.8)

    weights = np.empty((0, 2), dtype=np.float64)
    for i in range(popnum):
        w1 = np.float64(i / max(popnum - 1, 1))
        w2 = np.float64(1 - w1)
        weights = np.vstack([weights, np.array([[w1, w2]], dtype=np.float64)])

    neighbor_idx = np.zeros((popnum, t), dtype=np.int64)
    for i in range(popnum):
        distances = np.linalg.norm(weights - weights[i, :].reshape(1, -1), axis=1)
        neighbor_idx[i, :] = np.argsort(distances)[:t]

    z = np.min(pop_objs, axis=0)
    for _ in range(max_iter):
        for i in range(popnum):
            b = neighbor_idx[i, :]
            p1 = int(b[int(rng.randint(0, t))])
            p2 = int(b[int(rng.randint(0, t))])

            if rng.rand() < cross_rate:
                new_sol = _single_dim_crossover(population[p1, :], population[p2, :], rng)
            else:
                new_sol = population[i, :].copy()

            if rng.rand() < variate_rate:
                new_sol = _reset_mutation(new_sol, lb, ub, rng)

            new_sol = np.clip(new_sol, lb, ub)
            new_obj = _zdt_cost(new_sol, prob_id)
            z = np.minimum(z, new_obj)
            obj_scale = np.maximum(np.ptp(pop_objs, axis=0), 1.0)

            for j in range(t):
                k = int(b[j])
                old_fitness = np.max(weights[k, :] * np.abs((pop_objs[k, :] - z) / obj_scale))
                new_fitness = np.max(weights[k, :] * np.abs((new_obj - z) / obj_scale))
                if new_fitness < old_fitness:
                    population[k, :] = new_sol.copy()
                    pop_objs[k, :] = new_obj.copy()

    return _extract_first_front(pop_objs)


def _run_imogabka_zdt(
    prob_id: int,
    popnum: int,
    max_iter: int,
    seed: int,
) -> np.ndarray:
    dim, lb, ub = _problem_spec(prob_id)
    _, _, ture_pf, result = MOGABKA(
        Max_iter=max_iter,
        SearchAgents_no=popnum,
        FUN=f"ZDT{prob_id}",
        dim=dim,
        numObj=2,
        lb=lb,
        ub=ub,
        seed=seed,
    )
    front = np.asarray(result.get("archive_pf_fitness", ture_pf), dtype=np.float64)
    return _dedup_front(front)


def _plot_problem_fronts(out_png: Path, prob_id: int, true_pf: np.ndarray, front_map: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(true_pf[:, 0], true_pf[:, 1], "k.", markersize=3, label="True PF")

    markers = {
        "IMOGABKA": "o",
        "MOBKA": "d",
        "NSGA-II": "s",
        "NSGA-III": "^",
        "MOEA/D": "x",
    }
    for label, front in front_map.items():
        if front.size == 0:
            continue
        ax.scatter(front[:, 0], front[:, 1], s=18, marker=markers[label], label=label)

    ax.set_title(f"ZDT{prob_id} Public Benchmark")
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
        ("IMOGABKA", lambda prob_id, pop, iters, seed_or_rng: _run_imogabka_zdt(prob_id, pop, iters, int(seed_or_rng))),
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
            if label == "IMOGABKA":
                front = runner(prob_id, popnum, max_iter, algo_seed)
            else:
                front = runner(prob_id, popnum, max_iter, np.random.RandomState(algo_seed))

            front = _dedup_front(front)
            front_map[label] = front
            np.savetxt(
                out_dir / f"zdt{prob_id}_front_{label.lower().replace('/', '').replace('-', '').replace(' ', '_')}.csv",
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

        _plot_problem_fronts(out_dir / f"zdt{prob_id}_public_compare.png", prob_id, true_pf, front_map)

        with open(out_dir / f"zdt{prob_id}_hv_coverage_aux.csv", "w", encoding="utf-8") as f:
            labels = list(front_map.keys())
            f.write("metric,algorithm,value\n")
            for label in labels:
                f.write(f"hv,{label},{_safe_hv(front_map[label], true_pf):.17g}\n")
            for a in labels:
                for b in labels:
                    f.write(f"coverage_{a}_over_{b},{a},{(1.0 if a == b else _safe_coverage(front_map[a], front_map[b])):.17g}\n")

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

    with open(out_dir / "zdt_public_metrics.csv", "w", encoding="utf-8") as f:
        f.write("problem,algorithm,nd_count,gd,igd,spacing\n")
        for row in metrics_rows:
            f.write(
                f"{row[0]},{row[1]},{int(row[2])},{float(row[3]):.17g},{float(row[4]):.17g},{float(row[5]):.17g}\n"
            )

    with open(out_dir / "zdt_public_metrics_mean.csv", "w", encoding="utf-8") as f:
        f.write("algorithm,mean_gd,mean_igd,mean_spacing\n")
        for row in mean_rows:
            f.write(f"{row[0]},{float(row[1]):.17g},{float(row[2]):.17g},{float(row[3]):.17g}\n")

    print("\nZDT metric summary:")
    for row in mean_rows:
        print(
            f"{row[0]}: mean_gd={float(row[1]):.6g} "
            f"mean_igd={float(row[2]):.6g} mean_spacing={float(row[3]):.6g}"
        )
    print("\nSaved ZDT public benchmark outputs to:", out_dir)


if __name__ == "__main__":
    main()
