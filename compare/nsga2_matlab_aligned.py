from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from cal_obj_1_debug import cal_obj_1_with_debug
from get_parameter import get_parameter
from init_encoding import init_sol
from ndsort_crowding import fast_non_dominated_sort_with_crowding
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
class NSGA2Trace:
    generation_metrics: np.ndarray
    final_front_sorted: np.ndarray


def _eval_obj_as_nsga2(
    sol: np.ndarray,
    demand_points_info: np.ndarray,
    charge_points_info: np.ndarray,
    parameter: np.ndarray,
) -> Tuple[np.float64, np.float64]:
    """Adapter: migrated cal_obj_1 returns [inf;inf] for infeasible; NSGAII expects -1,-1."""
    f, _ = cal_obj_1_with_debug(sol, demand_points_info, charge_points_info, parameter)
    obj1 = np.float64(f[0, 0])
    obj2 = np.float64(f[1, 0])
    if (not np.isfinite(obj1)) or (not np.isfinite(obj2)):
        return np.float64(-1.0), np.float64(-1.0)
    return obj1, obj2


def build_initial_population_for_nsga2(
    settings: Dict[str, Any],
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    """Build settings.obj_manager and settings.sol_manager equivalent to runme/mutiRun init block."""
    if "_typed_problem" in settings:
        problem = settings["_typed_problem"]
        pop_num = int(settings["popnum"])
        init_nn_seed_count = int(settings.get("init_nn_seed_count", min(12, pop_num)))
        pop, obj = build_typed_initial_population(
            problem=problem,
            pop_size=pop_num,
            rng=rng,
            init_nn_seed_count=init_nn_seed_count,
        )

        out = dict(settings)
        out["obj_manager"] = np.asarray(obj, dtype=np.float64)
        out["sol_manager"] = [pop[i, :].copy() for i in range(pop.shape[0])]
        return out

    demand_points_info = np.asarray(np.loadtxt(settings["data1"], delimiter=","), dtype=np.float64)
    charge_points_info = np.asarray(np.loadtxt(settings["data2"], delimiter=","), dtype=np.float64)

    demand_points_num = int(demand_points_info.shape[0])
    charge_points_num = int(charge_points_info.shape[0])

    parameter = get_parameter()
    pop_num = int(settings["popnum"])

    obj_manager = np.empty((0, 2), dtype=np.float64)
    sol_manager: List[np.ndarray] = []

    for _ in range(pop_num):
        curr_sol = init_sol(charge_points_num, demand_points_num, rng)
        obj_1, obj_2 = _eval_obj_as_nsga2(curr_sol, demand_points_info, charge_points_info, parameter)

        while obj_1 == -1.0 and obj_2 == -1.0:
            curr_sol = init_sol(charge_points_num, demand_points_num, rng)
            obj_1, obj_2 = _eval_obj_as_nsga2(curr_sol, demand_points_info, charge_points_info, parameter)

        obj_manager = np.vstack([obj_manager, np.array([[obj_1, obj_2]], dtype=np.float64)])
        sol_manager.append(curr_sol)

    out = dict(settings)
    out["obj_manager"] = obj_manager
    out["sol_manager"] = sol_manager
    return out


def _compute_generation_metrics(gen: int, obj_manager: np.ndarray) -> np.ndarray:
    fronts, _ = fast_non_dominated_sort_with_crowding(obj_manager)
    nd_count = fronts[0].size if len(fronts) > 0 else 0

    best_obj1 = np.min(obj_manager[:, 0]) if obj_manager.size > 0 else np.inf
    best_obj2 = np.min(obj_manager[:, 1]) if obj_manager.size > 0 else np.inf
    mean_obj1 = np.mean(obj_manager[:, 0]) if obj_manager.size > 0 else np.inf
    mean_obj2 = np.mean(obj_manager[:, 1]) if obj_manager.size > 0 else np.inf

    return np.array([gen, best_obj1, best_obj2, mean_obj1, mean_obj2, float(nd_count)], dtype=np.float64)


def NSGA2_funciton(
    settings: Dict[str, Any],
    rng: np.random.RandomState | None = None,
    return_trace: bool = False,
) -> np.ndarray | Tuple[np.ndarray, NSGA2Trace]:
    """MATLAB NSGAII/NSGA2_funciton.m aligned implementation.

    Required settings keys:
    - data1, data2, maxgen, popnum, obj_manager, sol_manager

    Returns:
    - final_obj (same as MATLAB function output)
    - optionally trace for analysis script
    """
    if "_typed_problem" in settings:
        return _NSGA2_function_typed(settings=settings, rng=rng, return_trace=return_trace)

    if rng is None:
        rng = np.random.RandomState(2)

    demand_points_info = np.asarray(np.loadtxt(settings["data1"], delimiter=","), dtype=np.float64)
    charge_points_info = np.asarray(np.loadtxt(settings["data2"], delimiter=","), dtype=np.float64)

    demand_points_num = int(demand_points_info.shape[0])
    charge_points_num = int(charge_points_info.shape[0])

    parameter = get_parameter()

    max_iter_num = int(settings["maxgen"])
    pop_num = int(settings["popnum"])
    variate_rate = np.float64(0.3)
    cross_rate = np.float64(0.8)

    obj_manager = np.asarray(settings["obj_manager"], dtype=np.float64).copy()
    sol_manager = [np.asarray(s, dtype=np.float64).reshape(-1).copy() for s in settings["sol_manager"]]

    generation_metrics: List[np.ndarray] = []

    for iter_num in range(1, max_iter_num + 1):
        for i in range(pop_num):
            if rng.rand() < variate_rate:
                # MATLAB NSGA2_funciton passes demand_points_num to variate() 3rd arg.
                new_sol = variate(sol_manager[i], charge_points_num, demand_points_num, rng)
                obj_1, obj_2 = _eval_obj_as_nsga2(new_sol, demand_points_info, charge_points_info, parameter)
                if obj_1 != -1.0 or obj_2 != -1.0:
                    obj_manager = np.vstack([obj_manager, np.array([[obj_1, obj_2]], dtype=np.float64)])
                    sol_manager.append(new_sol)

            if rng.rand() < cross_rate:
                rand_id_1b = int(rng.randint(1, pop_num + 1))
                new_sol = cross(sol_manager[i], charge_points_num, sol_manager[rand_id_1b - 1], rng)
                obj_1, obj_2 = _eval_obj_as_nsga2(new_sol, demand_points_info, charge_points_info, parameter)
                if obj_1 != -1.0 or obj_2 != -1.0:
                    obj_manager = np.vstack([obj_manager, np.array([[obj_1, obj_2]], dtype=np.float64)])
                    sol_manager.append(new_sol)

        # trace before break/select to match generation checkpoint expectation.
        generation_metrics.append(_compute_generation_metrics(iter_num, obj_manager))

        if iter_num == max_iter_num:
            break

        fronts, crowding_dist = fast_non_dominated_sort_with_crowding(obj_manager)

        new_sol_manager: List[np.ndarray] = []
        new_obj_manager = np.empty((0, 2), dtype=np.float64)

        for front in fronts:
            crowd: List[np.float64] = []
            recorder: List[int] = []

            for sol_id in front:
                recorder.append(int(sol_id))
                crowd.append(np.float64(crowding_dist[int(sol_id)]))

            idx = np.argsort(np.asarray(crowd, dtype=np.float64))[::-1]
            for j in range(len(front)):
                curr_sol_id = recorder[int(idx[j])]
                new_sol_manager.append(sol_manager[curr_sol_id])
                new_obj_manager = np.vstack(
                    [new_obj_manager, np.array([[obj_manager[curr_sol_id, 0], obj_manager[curr_sol_id, 1]]], dtype=np.float64)]
                )

        sol_manager = []
        obj_manager = np.empty((0, 2), dtype=np.float64)
        for i in range(pop_num):
            sol_manager.append(new_sol_manager[i])
            obj_manager = np.vstack([obj_manager, new_obj_manager[i, :].reshape(1, 2)])

    fronts, _ = fast_non_dominated_sort_with_crowding(obj_manager)

    final_obj_rows: List[np.ndarray] = []
    if len(fronts) > 0:
        first_front = fronts[0]
        for j in range(first_front.size):
            sol_id = int(first_front[j])
            if j > 0:
                find_flag = False
                for k in range(len(final_obj_rows)):
                    if obj_manager[sol_id, 0] == final_obj_rows[k][0] and obj_manager[sol_id, 1] == final_obj_rows[k][1]:
                        find_flag = True
                        break
                if find_flag:
                    continue
            final_obj_rows.append(np.array([obj_manager[sol_id, 0], obj_manager[sol_id, 1]], dtype=np.float64))

    final_obj = np.vstack(final_obj_rows) if len(final_obj_rows) > 0 else np.empty((0, 2), dtype=np.float64)

    if final_obj.shape[0] > 0:
        order = np.argsort(final_obj[:, 0])
        final_front_sorted = final_obj[order, :]
    else:
        final_front_sorted = final_obj.copy()

    trace = NSGA2Trace(
        generation_metrics=np.vstack(generation_metrics) if len(generation_metrics) > 0 else np.empty((0, 6), dtype=np.float64),
        final_front_sorted=final_front_sorted,
    )

    if return_trace:
        return final_obj, trace
    return final_obj


def _NSGA2_function_typed(
    settings: Dict[str, Any],
    rng: np.random.RandomState | None = None,
    return_trace: bool = False,
) -> np.ndarray | Tuple[np.ndarray, NSGA2Trace]:
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
        seeded = build_initial_population_for_nsga2(settings, rng)
        obj_manager = np.asarray(seeded["obj_manager"], dtype=np.float64).copy()
        sol_manager = [np.asarray(s, dtype=np.float64).reshape(-1).copy() for s in seeded["sol_manager"]]

    generation_metrics: List[np.ndarray] = []

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

        generation_metrics.append(generation_metrics_row(iter_num, obj_manager, invalid_penalty))

        if iter_num == max_iter_num:
            break

        fronts, crowding_dist = fast_non_dominated_sort_with_crowding(obj_manager)

        new_sol_manager: List[np.ndarray] = []
        new_obj_manager = np.empty((0, 2), dtype=np.float64)

        for front in fronts:
            crowd = [np.float64(crowding_dist[int(sol_id)]) for sol_id in front]
            recorder = [int(sol_id) for sol_id in front]
            idx = np.argsort(np.asarray(crowd, dtype=np.float64))[::-1]
            for j in range(len(front)):
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

    if final_obj.shape[0] > 0:
        final_front_sorted = final_obj[np.argsort(final_obj[:, 0]), :]
    else:
        final_front_sorted = final_obj.copy()

    trace = NSGA2Trace(
        generation_metrics=np.vstack(generation_metrics) if len(generation_metrics) > 0 else np.empty((0, 6), dtype=np.float64),
        final_front_sorted=final_front_sorted,
    )

    if return_trace:
        return final_obj, trace
    return final_obj
