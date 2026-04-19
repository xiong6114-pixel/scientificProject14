from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from cal_obj_1_debug import cal_obj_1_with_debug
from get_parameter import get_parameter
from init_encoding import init_sol
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
class NSGA3Trace:
    generation_metrics: np.ndarray


def _eval_obj(sol: np.ndarray, demand_points_info: np.ndarray, charge_points_info: np.ndarray, parameter: np.ndarray) -> Tuple[np.float64, np.float64]:
    f, _ = cal_obj_1_with_debug(sol, demand_points_info, charge_points_info, parameter)
    obj1 = np.float64(f[0, 0])
    obj2 = np.float64(f[1, 0])
    if (not np.isfinite(obj1)) or (not np.isfinite(obj2)):
        return np.float64(-1.0), np.float64(-1.0)
    return obj1, obj2


def _dominates(indiv1: np.ndarray, indiv2: np.ndarray) -> bool:
    return bool(np.all(indiv1 <= indiv2) and np.any(indiv1 < indiv2))


def _generate_reference_points(obj1_bound: np.ndarray, obj2_bound: np.ndarray, num_ref_points: int) -> np.ndarray:
    # MATLAB uses linspace(..., sqrt(num_ref_points)); non-integer sqrt behavior is implementation-defined.
    n_side = int(np.round(np.sqrt(num_ref_points)))
    n_side = max(n_side, 1)
    obj1_grid, obj2_grid = np.meshgrid(
        np.linspace(obj1_bound[0], obj1_bound[1], n_side, dtype=np.float64),
        np.linspace(obj2_bound[0], obj2_bound[1], n_side, dtype=np.float64),
    )
    return np.column_stack([obj1_grid.reshape(-1), obj2_grid.reshape(-1)]).astype(np.float64)


def _assign_reference_points(front: np.ndarray, ref_points: np.ndarray, niche_count: np.ndarray, count: np.ndarray, front_id_0b: np.ndarray) -> np.ndarray:
    for i in range(front.shape[0]):
        distances = np.linalg.norm(ref_points - front[i, :], axis=1)
        min_index_0b = int(np.argmin(distances))
        niche_count[min_index_0b] = niche_count[min_index_0b] + 1

    for i in range(front.shape[0]):
        distances = np.linalg.norm(ref_points - front[i, :], axis=1)
        min_index_0b = int(np.argmin(distances))
        count[int(front_id_0b[i])] = -niche_count[min_index_0b]

    return count


def fast_non_dominated_sort_with_reference_points(population: np.ndarray, num_ref_points: int) -> Tuple[List[np.ndarray], np.ndarray]:
    population = np.asarray(population, dtype=np.float64)

    n = population.shape[0]
    s: List[List[int]] = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int64)
    fronts: List[List[int]] = [[]]

    obj1_bound = np.array([np.min(population[:, 0]), np.max(population[:, 0])], dtype=np.float64)
    obj2_bound = np.array([np.min(population[:, 1]), np.max(population[:, 1])], dtype=np.float64)

    ref_points = _generate_reference_points(obj1_bound, obj2_bound, num_ref_points)

    # MATLAB count initialized with input num_ref_points (caller passes length(population)).
    count = np.zeros(num_ref_points, dtype=np.float64)
    num_ref_points_real = ref_points.shape[0]

    for i in range(n):
        s[i] = []
        n_dom[i] = 0
        for j in range(n):
            if _dominates(population[i, :], population[j, :]):
                s[i].append(j)
            elif _dominates(population[j, :], population[i, :]):
                n_dom[i] += 1

        if n_dom[i] == 0:
            rank[i] = 1
            fronts[0].append(i)

    k = 0
    while k < len(fronts) and len(fronts[k]) > 0:
        q: List[int] = []
        for i in fronts[k]:
            for j in s[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    rank[j] = k + 2
                    q.append(j)
        k += 1
        fronts.append(q)

    out_fronts = [np.asarray(f, dtype=np.int64) for f in fronts[:k]]

    for front in out_fronts:
        niche_count = np.zeros(num_ref_points_real, dtype=np.float64)
        count = _assign_reference_points(population[front, :], ref_points, niche_count, count, front)

    return out_fronts, count


def NSGA3_funciton(settings: Dict[str, Any], rng: np.random.RandomState | None = None, return_trace: bool = False) -> np.ndarray | Tuple[np.ndarray, NSGA3Trace]:
    """MATLAB NSGAIII/NSGA3_funciton.m aligned implementation."""
    if "_typed_problem" in settings:
        return _NSGA3_function_typed(settings=settings, rng=rng, return_trace=return_trace)

    if rng is None:
        rng = np.random.RandomState(2)

    demand_points_info = np.asarray(np.loadtxt(settings["data1"], delimiter=","), dtype=np.float64)
    charge_points_info = np.asarray(np.loadtxt(settings["data2"], delimiter=","), dtype=np.float64)

    demand_points_num = int(demand_points_info.shape[0])
    charge_points_num = int(charge_points_info.shape[0])

    all_demand = int(np.round(np.sum(demand_points_info[:, 2])))

    parameter = get_parameter()

    max_iter_num = int(settings["maxgen"])
    pop_num = int(settings["popnum"])
    variate_rate = np.float64(0.3)
    cross_rate = np.float64(0.8)

    obj_manager = np.empty((0, 2), dtype=np.float64)
    sol_manager: List[np.ndarray] = []

    for _ in range(pop_num):
        curr_sol = init_sol(charge_points_num, demand_points_num, rng)
        obj_1, obj_2 = _eval_obj(curr_sol, demand_points_info, charge_points_info, parameter)
        while obj_1 == -1.0 and obj_2 == -1.0:
            curr_sol = init_sol(charge_points_num, demand_points_num, rng)
            obj_1, obj_2 = _eval_obj(curr_sol, demand_points_info, charge_points_info, parameter)
        obj_manager = np.vstack([obj_manager, np.array([[obj_1, obj_2]], dtype=np.float64)])
        sol_manager.append(curr_sol)

    gen_rows: List[np.ndarray] = []

    for iter_num in range(1, max_iter_num + 1):
        for i in range(pop_num):
            if rng.rand() < variate_rate:
                new_sol = variate(sol_manager[i], charge_points_num, all_demand, rng)
                obj_1, obj_2 = _eval_obj(new_sol, demand_points_info, charge_points_info, parameter)
                if obj_1 != -1.0 or obj_2 != -1.0:
                    obj_manager = np.vstack([obj_manager, np.array([[obj_1, obj_2]], dtype=np.float64)])
                    sol_manager.append(new_sol)

            if rng.rand() < cross_rate:
                rand_id_1b = int(rng.randint(1, pop_num + 1))
                new_sol = cross(sol_manager[i], charge_points_num, sol_manager[rand_id_1b - 1], rng)
                obj_1, obj_2 = _eval_obj(new_sol, demand_points_info, charge_points_info, parameter)
                if obj_1 != -1.0 or obj_2 != -1.0:
                    obj_manager = np.vstack([obj_manager, np.array([[obj_1, obj_2]], dtype=np.float64)])
                    sol_manager.append(new_sol)

        fronts_tmp, _ = fast_non_dominated_sort_with_reference_points(obj_manager, obj_manager.shape[0])
        nd_count = float(fronts_tmp[0].size if len(fronts_tmp) > 0 else 0)
        best1 = float(np.min(obj_manager[:, 0])) if obj_manager.size > 0 else float("inf")
        best2 = float(np.min(obj_manager[:, 1])) if obj_manager.size > 0 else float("inf")
        mean1 = float(np.mean(obj_manager[:, 0])) if obj_manager.size > 0 else float("inf")
        mean2 = float(np.mean(obj_manager[:, 1])) if obj_manager.size > 0 else float("inf")
        gen_rows.append(np.array([iter_num, best1, best2, mean1, mean2, nd_count], dtype=np.float64))

        if iter_num == max_iter_num:
            break

        fronts, crowding_dist = fast_non_dominated_sort_with_reference_points(obj_manager, obj_manager.shape[0])

        new_sol_manager: List[np.ndarray] = []
        new_obj_manager = np.empty((0, 2), dtype=np.float64)

        for front in fronts:
            crowd: List[np.float64] = []
            recorder: List[int] = []
            for sol_id in front:
                recorder.append(int(sol_id))
                crowd.append(np.float64(crowding_dist[int(sol_id)]))

            idx = np.argsort(np.asarray(crowd, dtype=np.float64))[::-1]
            for j in range(front.size):
                curr_sol_id = recorder[int(idx[j])]
                new_sol_manager.append(sol_manager[curr_sol_id])
                new_obj_manager = np.vstack([new_obj_manager, np.array([[obj_manager[curr_sol_id, 0], obj_manager[curr_sol_id, 1]]], dtype=np.float64)])

        sol_manager = []
        obj_manager = np.empty((0, 2), dtype=np.float64)
        for i in range(pop_num):
            sol_manager.append(new_sol_manager[i])
            obj_manager = np.vstack([obj_manager, new_obj_manager[i, :].reshape(1, 2)])

    fronts, _ = fast_non_dominated_sort_with_reference_points(obj_manager, obj_manager.shape[0])
    final_rows: List[np.ndarray] = []

    if len(fronts) > 0:
        first_front = fronts[0]
        for j in range(first_front.size):
            sol_id = int(first_front[j])
            if j > 0:
                find_flag = False
                for k in range(len(final_rows)):
                    if obj_manager[sol_id, 0] == final_rows[k][0] and obj_manager[sol_id, 1] == final_rows[k][1]:
                        find_flag = True
                        break
                if find_flag:
                    continue
            final_rows.append(np.array([obj_manager[sol_id, 0], obj_manager[sol_id, 1]], dtype=np.float64))

    final_obj = np.vstack(final_rows) if len(final_rows) > 0 else np.empty((0, 2), dtype=np.float64)

    if return_trace:
        trace = NSGA3Trace(generation_metrics=np.vstack(gen_rows) if len(gen_rows) > 0 else np.empty((0, 6), dtype=np.float64))
        return final_obj, trace
    return final_obj


def _NSGA3_function_typed(
    settings: Dict[str, Any],
    rng: np.random.RandomState | None = None,
    return_trace: bool = False,
) -> np.ndarray | Tuple[np.ndarray, NSGA3Trace]:
    if rng is None:
        rng = np.random.RandomState(2)

    problem = settings["_typed_problem"]
    invalid_penalty = float(problem.invalid_penalty)

    max_iter_num = int(settings["maxgen"])
    pop_num = int(settings["popnum"])
    variate_rate = np.float64(settings.get("variate_rate", 0.3))
    cross_rate = np.float64(settings.get("cross_rate", 0.8))

    if "obj_manager" in settings and "sol_manager" in settings:
        obj_manager = np.asarray(settings["obj_manager"], dtype=np.float64).copy()
        sol_manager = [np.asarray(s, dtype=np.float64).reshape(-1).copy() for s in settings["sol_manager"]]
    else:
        init_nn_seed_count = int(settings.get("init_nn_seed_count", min(12, pop_num)))
        pop, obj = build_typed_initial_population(
            problem=problem,
            pop_size=pop_num,
            rng=rng,
            init_nn_seed_count=init_nn_seed_count,
        )
        obj_manager = np.asarray(obj, dtype=np.float64).copy()
        sol_manager = [pop[i, :].copy() for i in range(pop.shape[0])]

    gen_rows: List[np.ndarray] = []

    for iter_num in range(1, max_iter_num + 1):
        for i in range(pop_num):
            if rng.rand() < variate_rate:
                new_sol = typed_station_mutation(sol_manager[i], problem, rng)
                obj = evaluate_typed_solution(new_sol, problem)
                if is_feasible_obj(obj, invalid_penalty):
                    obj_manager = np.vstack([obj_manager, obj.reshape(1, 2)])
                    sol_manager.append(new_sol)

            if rng.rand() < cross_rate:
                rand_id = int(rng.randint(0, pop_num))
                new_sol = typed_station_crossover(sol_manager[i], sol_manager[rand_id], problem, rng)
                if rng.rand() < 0.25:
                    new_sol = typed_station_mutation(new_sol, problem, rng)
                obj = evaluate_typed_solution(new_sol, problem)
                if is_feasible_obj(obj, invalid_penalty):
                    obj_manager = np.vstack([obj_manager, obj.reshape(1, 2)])
                    sol_manager.append(new_sol)

        gen_rows.append(generation_metrics_row(iter_num, obj_manager, invalid_penalty))

        if iter_num == max_iter_num:
            break

        fronts, crowding_dist = fast_non_dominated_sort_with_reference_points(obj_manager, max(obj_manager.shape[0], pop_num))

        new_sol_manager: List[np.ndarray] = []
        new_obj_manager = np.empty((0, 2), dtype=np.float64)

        for front in fronts:
            crowd = [np.float64(crowding_dist[int(sol_id)]) for sol_id in front]
            recorder = [int(sol_id) for sol_id in front]
            idx = np.argsort(np.asarray(crowd, dtype=np.float64))[::-1]
            for j in range(front.size):
                curr_sol_id = recorder[int(idx[j])]
                new_sol_manager.append(sol_manager[curr_sol_id])
                new_obj_manager = np.vstack([new_obj_manager, obj_manager[curr_sol_id, :].reshape(1, 2)])

        sol_manager = []
        obj_manager = np.empty((0, 2), dtype=np.float64)
        for i in range(min(pop_num, len(new_sol_manager))):
            sol_manager.append(new_sol_manager[i])
            obj_manager = np.vstack([obj_manager, new_obj_manager[i, :].reshape(1, 2)])

    final_pop, final_obj = extract_feasible_first_front(
        pop=np.vstack([sol.reshape(1, -1) for sol in sol_manager]).astype(np.float64) if len(sol_manager) > 0 else None,
        objs=obj_manager,
        invalid_penalty=invalid_penalty,
    )
    _ = final_pop

    if return_trace:
        trace = NSGA3Trace(generation_metrics=np.vstack(gen_rows) if len(gen_rows) > 0 else np.empty((0, 6), dtype=np.float64))
        return final_obj, trace
    return final_obj
