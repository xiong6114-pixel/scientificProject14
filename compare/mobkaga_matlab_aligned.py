from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from cal_obj_1_debug import cal_obj_1_with_debug
from get_parameter import get_parameter
from ndsort_crowding import fast_non_dominated_sort_with_crowding
from nsga2_ops import cross, variate


def _eval_obj_as_mobkaga(
    sol: np.ndarray,
    demand_points_info: np.ndarray,
    charge_points_info: np.ndarray,
    parameter: np.ndarray,
) -> Tuple[np.float64, np.float64]:
    f, _ = cal_obj_1_with_debug(sol, demand_points_info, charge_points_info, parameter)
    obj1 = np.float64(f[0, 0])
    obj2 = np.float64(f[1, 0])
    if (not np.isfinite(obj1)) or (not np.isfinite(obj2)):
        return np.float64(-1.0), np.float64(-1.0)
    return obj1, obj2


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def _calculate_crowding_distance(objs: np.ndarray) -> np.ndarray:
    num_solutions = int(objs.shape[0])
    num_objs = int(objs.shape[1])
    distances = np.zeros(num_solutions, dtype=np.float64)

    if num_solutions == 0:
        return distances

    for m in range(num_objs):
        sorted_indices = np.argsort(objs[:, m])
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf

        if num_solutions <= 2:
            continue

        denom = np.max(objs[:, m]) - np.min(objs[:, m])
        for i in range(1, num_solutions - 1):
            distances[sorted_indices[i]] = distances[sorted_indices[i]] + (
                (objs[sorted_indices[i + 1], m] - objs[sorted_indices[i - 1], m]) / denom
            )
    return distances


def _calculate_objs(
    archive: np.ndarray,
    demand_points_info: np.ndarray,
    charge_points_info: np.ndarray,
    parameter: np.ndarray,
) -> np.ndarray:
    if archive.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    objs = np.zeros((archive.shape[0], 2), dtype=np.float64)
    for i in range(archive.shape[0]):
        obj1, obj2 = _eval_obj_as_mobkaga(archive[i, :], demand_points_info, charge_points_info, parameter)
        objs[i, :] = np.array([obj1, obj2], dtype=np.float64)
    return objs


def update_archive(
    archive: np.ndarray,
    particles: np.ndarray,
    archive_size: int,
    demand_points_info: np.ndarray,
    charge_points_info: np.ndarray,
    parameter: np.ndarray,
) -> np.ndarray:
    new_solutions_rows: List[np.ndarray] = []
    new_objs_rows: List[np.ndarray] = []

    for i in range(particles.shape[0]):
        obj1, obj2 = _eval_obj_as_mobkaga(particles[i, :], demand_points_info, charge_points_info, parameter)
        if obj1 != -1.0 or obj2 != -1.0:
            new_solutions_rows.append(particles[i, :].reshape(1, -1))
            new_objs_rows.append(np.array([[obj1, obj2]], dtype=np.float64))

    if len(new_solutions_rows) > 0:
        new_solutions = np.vstack(new_solutions_rows).astype(np.float64)
        new_objs = np.vstack(new_objs_rows).astype(np.float64)
    else:
        new_solutions = np.empty((0, particles.shape[1]), dtype=np.float64)
        new_objs = np.empty((0, 2), dtype=np.float64)

    archive_solutions = np.vstack([archive, new_solutions]).astype(np.float64)
    archive_objs = np.vstack([_calculate_objs(archive, demand_points_info, charge_points_info, parameter), new_objs]).astype(np.float64)

    is_dominated = np.zeros(archive_objs.shape[0], dtype=bool)
    for i in range(archive_objs.shape[0]):
        for j in range(archive_objs.shape[0]):
            if i != j and _dominates(archive_objs[j, :], archive_objs[i, :]):
                is_dominated[i] = True
                break

    archive_solutions = archive_solutions[~is_dominated, :]
    archive_objs = archive_objs[~is_dominated, :]

    if archive_solutions.shape[0] > archive_size:
        distances = _calculate_crowding_distance(archive_objs)
        sorted_indices = np.argsort(distances)[::-1]
        archive_solutions = archive_solutions[sorted_indices[:archive_size], :]

    return archive_solutions


def _levy(dim: int, rng: np.random.RandomState) -> np.ndarray:
    beta = np.float64(1.1)
    num = math.gamma(1 + float(beta)) * np.sin(np.pi * float(beta) / 2)
    den = math.gamma((1 + float(beta)) / 2) * float(beta) * 2 ** ((float(beta) - 1) / 2)
    sigma_u = (num / den) ** (1 / float(beta))

    u = rng.normal(0.0, sigma_u, size=(dim,))
    v = rng.normal(0.0, 1.0, size=(dim,))
    z = u / (np.abs(v) ** (1 / float(beta)))
    return z.astype(np.float64)


def MOBKAGA_function(
    settings: Dict[str, Any],
    rng: np.random.RandomState | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """MATLAB MOBKAGA/MOBKAGA_function.m aligned implementation."""
    if rng is None:
        rng = np.random.RandomState(2)

    demand_points_info = np.asarray(np.loadtxt(settings["data1"], delimiter=","), dtype=np.float64)
    charge_points_info = np.asarray(np.loadtxt(settings["data2"], delimiter=","), dtype=np.float64)

    demand_points_num = int(demand_points_info.shape[0])
    charge_points_num = int(charge_points_info.shape[0])

    parameter = get_parameter()

    archive_size = 100
    max_iter_num = int(settings["maxgen"])
    pop_num = int(settings["popnum"])
    variate_rate = np.float64(0.3)
    cross_rate = np.float64(0.8)
    p = np.float64(0.8)
    p_levy = np.float64(0.5)

    obj_manager = np.asarray(settings["obj_manager"], dtype=np.float64).copy()
    sol_manager = [np.asarray(s, dtype=np.float64).reshape(-1).copy() for s in settings["sol_manager"]]

    archive = np.empty((0, charge_points_num), dtype=np.float64)

    for iter_num in range(1, max_iter_num + 1):
        new_sol = sol_manager[-1].copy()

        for i in range(pop_num):
            if rng.rand() < variate_rate:
                # MATLAB variate() is called with demand_points_num as upper bound here.
                new_sol = variate(sol_manager[i], charge_points_num, demand_points_num, rng)
                obj_1, obj_2 = _eval_obj_as_mobkaga(new_sol, demand_points_info, charge_points_info, parameter)
                if obj_1 != -1.0 or obj_2 != -1.0:
                    obj_manager = np.vstack([obj_manager, np.array([[obj_1, obj_2]], dtype=np.float64)])
                    sol_manager.append(new_sol)

            if rng.rand() < cross_rate:
                rand_id_1b = int(rng.randint(1, pop_num + 1))
                # Explicit 1-based to 0-based mapping.
                new_sol = cross(sol_manager[i], charge_points_num, sol_manager[rand_id_1b - 1], rng)
                obj_1, obj_2 = _eval_obj_as_mobkaga(new_sol, demand_points_info, charge_points_info, parameter)
                if obj_1 != -1.0 or obj_2 != -1.0:
                    obj_manager = np.vstack([obj_manager, np.array([[obj_1, obj_2]], dtype=np.float64)])
                    sol_manager.append(new_sol)

            leader_1b = int(rng.randint(1, len(sol_manager) + 1))
            leader = sol_manager[leader_1b - 1]

            r = np.float64(rng.rand())
            n = np.float64(0.05 * np.exp(-2 * (iter_num / max_iter_num) ** 2))

            if p < r:
                delta_sol = n * (1 + np.sin(r)) * leader
            else:
                delta_sol = sol_manager[i] * n * (2 * rng.rand(demand_points_num) - 1)

            delta_sol = np.maximum(np.float64(-10.0), np.minimum(np.float64(10.0), delta_sol))
            new_sol = np.round(sol_manager[i] + delta_sol)
            new_sol[new_sol < 0] = 0
            new_sol[new_sol > demand_points_num] = demand_points_num

            obj_1, obj_2 = _eval_obj_as_mobkaga(new_sol, demand_points_info, charge_points_info, parameter)
            if obj_1 != -1.0 or obj_2 != -1.0:
                obj_manager = np.vstack([obj_manager, np.array([[obj_1, obj_2]], dtype=np.float64)])
                sol_manager.append(new_sol)

            _m = np.float64(2 * np.sin(r + np.pi / 2))
            _s_1b = int(rng.randint(1, pop_num + 1))
            ori_value = rng.rand(demand_points_num)
            cauchy_value = np.tan((ori_value - 0.5) * np.pi)

            delta_sol = cauchy_value * (new_sol - leader)
            delta_sol = np.maximum(np.float64(-10.0), np.minimum(np.float64(10.0), delta_sol))
            new_sol = np.round(new_sol + delta_sol)
            new_sol[new_sol < 0] = 0
            new_sol[new_sol > demand_points_num] = demand_points_num

            obj_1, obj_2 = _eval_obj_as_mobkaga(new_sol, demand_points_info, charge_points_info, parameter)
            if obj_1 != -1.0 or obj_2 != -1.0:
                obj_manager = np.vstack([obj_manager, np.array([[obj_1, obj_2]], dtype=np.float64)])
                sol_manager.append(new_sol)

        # Keep MATLAB behavior: new_sol may be reused when rand >= Plevy.
        new_sol_levy = new_sol.copy()
        for i in range(len(sol_manager)):
            if rng.rand() < p_levy:
                l = _levy(demand_points_num, rng)
                new_sol_levy = np.round(sol_manager[i] + sol_manager[i] * l)
                new_sol_levy[new_sol_levy > demand_points_num] = demand_points_num
                new_sol_levy[new_sol_levy < 0] = 0

            obj_1, obj_2 = _eval_obj_as_mobkaga(new_sol_levy, demand_points_info, charge_points_info, parameter)
            if obj_1 != -1.0 or obj_2 != -1.0:
                obj_manager = np.vstack([obj_manager, np.array([[obj_1, obj_2]], dtype=np.float64)])
                sol_manager.append(new_sol_levy)

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
            for j in range(front.size):
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

        pop = np.vstack([sol.reshape(1, -1) for sol in sol_manager]).astype(np.float64)
        archive = update_archive(
            archive=archive,
            particles=pop,
            archive_size=archive_size,
            demand_points_info=demand_points_info,
            charge_points_info=charge_points_info,
            parameter=parameter,
        )

    archive_obj = np.empty((0, 2), dtype=np.float64)
    archive_sol = np.empty((0, charge_points_num), dtype=np.float64)
    for i in range(archive.shape[0]):
        obj1, obj2 = _eval_obj_as_mobkaga(archive[i, :], demand_points_info, charge_points_info, parameter)
        if obj1 != -1.0 or obj2 != -1.0:
            archive_obj = np.vstack([archive_obj, np.array([[obj1, obj2]], dtype=np.float64)])
            archive_sol = np.vstack([archive_sol, archive[i, :].reshape(1, -1)])

    return archive_obj, archive_sol

