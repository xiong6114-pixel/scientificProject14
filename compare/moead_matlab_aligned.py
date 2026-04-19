from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from cal_obj_1_debug import cal_obj_1_with_debug
from get_parameter import get_parameter
from nsga2_ops import cross, variate
from typed_compare_utils import (
    build_typed_initial_population,
    evaluate_typed_solution,
    extract_feasible_first_front,
    generation_metrics_row,
    is_feasible_obj,
    typed_station_crossover,
    typed_station_mutation,
)


@dataclass
class MOEADTrace:
    generation_metrics: np.ndarray


def _eval_obj(sol: np.ndarray, demand_points_info: np.ndarray, charge_points_info: np.ndarray, parameter: np.ndarray) -> Tuple[np.float64, np.float64]:
    f, _ = cal_obj_1_with_debug(sol, demand_points_info, charge_points_info, parameter)
    obj1 = np.float64(f[0, 0])
    obj2 = np.float64(f[1, 0])
    if (not np.isfinite(obj1)) or (not np.isfinite(obj2)):
        return np.float64(-1.0), np.float64(-1.0)
    return obj1, obj2


def _dedup_rows_keep_order(x: np.ndarray) -> np.ndarray:
    out: List[np.ndarray] = []
    for i in range(x.shape[0]):
        row = x[i, :]
        seen = False
        for r in out:
            if r[0] == row[0] and r[1] == row[1]:
                seen = True
                break
        if not seen:
            out.append(row.copy())
    if len(out) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(out).astype(np.float64)


def MOEAD_function(settings: Dict[str, Any], rng: np.random.RandomState | None = None, return_trace: bool = False) -> np.ndarray | Tuple[np.ndarray, MOEADTrace]:
    """MATLAB moead/MOEAD_function.m aligned implementation."""
    if "_typed_problem" in settings:
        return _MOEAD_function_typed(settings=settings, rng=rng, return_trace=return_trace)

    if rng is None:
        rng = np.random.RandomState(2)

    demand_points_info = np.asarray(np.loadtxt(settings["data1"], delimiter=","), dtype=np.float64)
    charge_points_info = np.asarray(np.loadtxt(settings["data2"], delimiter=","), dtype=np.float64)

    demand_points_num = int(demand_points_info.shape[0])
    charge_points_num = int(charge_points_info.shape[0])

    parameter = get_parameter()

    max_iter_num = int(settings["maxgen"])
    pop_num = int(settings["popnum"])
    t = int(np.round(0.1 * pop_num))
    variate_rate = np.float64(0.3)
    cross_rate = np.float64(0.8)

    weights = np.empty((0, 2), dtype=np.float64)
    for i in range(pop_num):
        w1 = np.float64(i / (pop_num - 1))
        w2 = np.float64(1 - w1)
        weights = np.vstack([weights, np.array([[w1 * 1000.0, w2]], dtype=np.float64)])

    neighbor_idx = np.zeros((pop_num, t), dtype=np.int64)
    for i in range(pop_num):
        distances = np.zeros(pop_num, dtype=np.float64)
        for j in range(pop_num):
            distances[j] = np.linalg.norm(weights[i, :] - weights[j, :])
        sorted_idx_0b = np.argsort(distances)
        neighbor_idx[i, :] = sorted_idx_0b[:t]

    z = np.array([np.inf, np.inf], dtype=np.float64)

    population = [np.asarray(s, dtype=np.float64).reshape(-1).copy() for s in settings["sol_manager"]]
    pop_objs = np.asarray(settings["obj_manager"], dtype=np.float64).copy()

    gen_rows: List[np.ndarray] = []

    for iter_num in range(1, max_iter_num + 1):
        for i in range(pop_num):
            b = neighbor_idx[i, :]

            p1_0b = int(b[int(rng.randint(0, t))])
            p2_0b = int(b[int(rng.randint(0, t))])

            if rng.rand() < cross_rate:
                new_sol = cross(population[p1_0b], charge_points_num, population[p2_0b], rng)
            else:
                new_sol = population[i].copy()

            if rng.rand() < variate_rate:
                new_sol = variate(new_sol, charge_points_num, demand_points_num, rng)

            new_obj1, new_obj2 = _eval_obj(new_sol, demand_points_info, charge_points_info, parameter)
            if new_obj1 == -1.0 or new_obj2 == -1.0:
                continue

            z = np.minimum(z, np.array([new_obj1, new_obj2], dtype=np.float64))

            for j in range(t):
                k = int(b[j])
                old_fitness = np.max(weights[k, :] * np.abs(pop_objs[k, :] - z))
                new_fitness = np.max(weights[k, :] * np.abs(np.array([new_obj1, new_obj2], dtype=np.float64) - z))
                if new_fitness < old_fitness:
                    population[k] = new_sol.copy()
                    pop_objs[k, :] = np.array([new_obj1, new_obj2], dtype=np.float64)

        nd_count = float(_dedup_rows_keep_order(pop_objs).shape[0])
        best1 = float(np.min(pop_objs[:, 0])) if pop_objs.size > 0 else float("inf")
        best2 = float(np.min(pop_objs[:, 1])) if pop_objs.size > 0 else float("inf")
        mean1 = float(np.mean(pop_objs[:, 0])) if pop_objs.size > 0 else float("inf")
        mean2 = float(np.mean(pop_objs[:, 1])) if pop_objs.size > 0 else float("inf")
        gen_rows.append(np.array([iter_num, best1, best2, mean1, mean2, nd_count], dtype=np.float64))

    final_obj = _dedup_rows_keep_order(pop_objs)

    if return_trace:
        trace = MOEADTrace(generation_metrics=np.vstack(gen_rows) if len(gen_rows) > 0 else np.empty((0, 6), dtype=np.float64))
        return final_obj, trace
    return final_obj


def _MOEAD_function_typed(
    settings: Dict[str, Any],
    rng: np.random.RandomState | None = None,
    return_trace: bool = False,
) -> np.ndarray | Tuple[np.ndarray, MOEADTrace]:
    if rng is None:
        rng = np.random.RandomState(2)

    problem = settings["_typed_problem"]
    invalid_penalty = float(problem.invalid_penalty)

    max_iter_num = int(settings["maxgen"])
    pop_num = int(settings["popnum"])
    neighbor_size = int(settings.get("neighbor_size", np.round(0.1 * pop_num)))
    t = max(2, neighbor_size)
    t = min(t, pop_num)
    variate_rate = np.float64(settings.get("variate_rate", 0.3))
    cross_rate = np.float64(settings.get("cross_rate", 0.8))

    weights = np.empty((0, 2), dtype=np.float64)
    for i in range(pop_num):
        w1 = np.float64(i / max(pop_num - 1, 1))
        w2 = np.float64(1 - w1)
        weights = np.vstack([weights, np.array([[w1, w2]], dtype=np.float64)])

    neighbor_idx = np.zeros((pop_num, t), dtype=np.int64)
    for i in range(pop_num):
        distances = np.zeros(pop_num, dtype=np.float64)
        for j in range(pop_num):
            distances[j] = np.linalg.norm(weights[i, :] - weights[j, :])
        neighbor_idx[i, :] = np.argsort(distances)[:t]

    if "obj_manager" in settings and "sol_manager" in settings:
        population = [np.asarray(s, dtype=np.float64).reshape(-1).copy() for s in settings["sol_manager"]]
        pop_objs = np.asarray(settings["obj_manager"], dtype=np.float64).copy()
    else:
        init_nn_seed_count = int(settings.get("init_nn_seed_count", min(12, pop_num)))
        pop, obj = build_typed_initial_population(
            problem=problem,
            pop_size=pop_num,
            rng=rng,
            init_nn_seed_count=init_nn_seed_count,
        )
        population = [pop[i, :].copy() for i in range(pop.shape[0])]
        pop_objs = np.asarray(obj, dtype=np.float64).copy()

    z = np.min(pop_objs, axis=0)
    gen_rows: List[np.ndarray] = []

    for iter_num in range(1, max_iter_num + 1):
        for i in range(pop_num):
            b = neighbor_idx[i, :]
            p1 = int(b[int(rng.randint(0, t))])
            p2 = int(b[int(rng.randint(0, t))])

            if rng.rand() < cross_rate:
                new_sol = typed_station_crossover(population[p1], population[p2], problem, rng)
            else:
                new_sol = population[i].copy()

            if rng.rand() < variate_rate:
                new_sol = typed_station_mutation(new_sol, problem, rng)

            new_obj = evaluate_typed_solution(new_sol, problem)
            if not is_feasible_obj(new_obj, invalid_penalty):
                continue

            z = np.minimum(z, new_obj)
            obj_scale = np.maximum(np.ptp(pop_objs, axis=0), 1.0)

            for j in range(t):
                k = int(b[j])
                old_fitness = np.max(weights[k, :] * np.abs((pop_objs[k, :] - z) / obj_scale))
                new_fitness = np.max(weights[k, :] * np.abs((new_obj - z) / obj_scale))
                if new_fitness < old_fitness:
                    population[k] = new_sol.copy()
                    pop_objs[k, :] = new_obj.copy()

        gen_rows.append(generation_metrics_row(iter_num, pop_objs, invalid_penalty))

    pop_matrix = np.vstack([sol.reshape(1, -1) for sol in population]).astype(np.float64) if len(population) > 0 else None
    _, final_obj = extract_feasible_first_front(
        pop=pop_matrix,
        objs=pop_objs,
        invalid_penalty=invalid_penalty,
    )

    if return_trace:
        trace = MOEADTrace(generation_metrics=np.vstack(gen_rows) if len(gen_rows) > 0 else np.empty((0, 6), dtype=np.float64))
        return final_obj, trace
    return final_obj
