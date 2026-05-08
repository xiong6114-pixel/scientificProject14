from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np

try:
    from sklearn.cluster import Birch, KMeans
except Exception:  # pragma: no cover - exercised only on minimal environments.
    Birch = None
    KMeans = None

from cal_obj_1_debug import cal_obj_1_with_debug
from ev_typed_nn_seed_adapter import split_total_into_types
from get_parameter import get_parameter
from init_encoding import init_sol
from ndsort_crowding import fast_non_dominated_sort_with_crowding
from nsga2_ops import cross, variate
from typed_compare_utils import (
    evaluate_typed_solution,
    extract_feasible_first_front,
    generation_metrics_row,
    is_feasible_obj,
    random_typed_solution,
    sanitize_typed_solution,
    typed_station_crossover,
    typed_station_mutation,
)


@dataclass
class HNSGA2KMeansTrace:
    generation_metrics: np.ndarray
    final_front_sorted: np.ndarray
    cluster_count_history: np.ndarray


def HNSGA2_KMeans_funciton(
    settings: Dict[str, Any],
    rng: np.random.RandomState | None = None,
    return_trace: bool = False,
) -> np.ndarray | Tuple[np.ndarray, HNSGA2KMeansTrace]:
    """HNSGA-II + K-Means baseline for the compare sandbox.

    The typed branch is the intended 50-point EV baseline.  It uses K-Means to
    build demand-aware initial solutions, then runs an NSGA-II loop whose parent
    selection combines BIRCH clustering, roulette-wheel sampling, and binary
    tournament selection.
    """
    if "_typed_problem" in settings:
        return _run_typed_hnsga2_kmeans(settings=settings, rng=rng, return_trace=return_trace)
    return _run_legacy_hnsga2_kmeans(settings=settings, rng=rng, return_trace=return_trace)


def HNSGA2KMeans_function(
    settings: Dict[str, Any],
    rng: np.random.RandomState | None = None,
    return_trace: bool = False,
):
    return HNSGA2_KMeans_funciton(settings=settings, rng=rng, return_trace=return_trace)


def build_initial_population_for_hnsga2_kmeans(
    settings: Dict[str, Any],
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    if "_typed_problem" in settings:
        problem = settings["_typed_problem"]
        pop_num = int(settings["popnum"])
        kmeans_seed_count = _default_kmeans_seed_count(pop_num)
        pop, obj = _build_kmeans_guided_typed_initial_population(
            problem=problem,
            pop_size=pop_num,
            rng=rng,
            kmeans_seed_count=int(settings.get("kmeans_seed_count", kmeans_seed_count)),
        )
        out = dict(settings)
        out["obj_manager"] = np.asarray(obj, dtype=np.float64)
        out["sol_manager"] = [pop[i, :].copy() for i in range(pop.shape[0])]
        return out

    demand_points_info = np.asarray(np.loadtxt(settings["data1"], delimiter=","), dtype=np.float64)
    charge_points_info = np.asarray(np.loadtxt(settings["data2"], delimiter=","), dtype=np.float64)
    parameter = get_parameter()
    pop_num = int(settings["popnum"])
    charge_points_num = int(charge_points_info.shape[0])
    demand_points_num = int(demand_points_info.shape[0])
    kmeans_seed_count = int(settings.get("kmeans_seed_count", _default_kmeans_seed_count(pop_num)))

    obj_manager = np.empty((0, 2), dtype=np.float64)
    sol_manager: List[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()

    def _try_add(sol: np.ndarray) -> bool:
        key = tuple(np.asarray(sol, dtype=float).round().astype(int).tolist())
        if key in seen:
            return False
        obj_1, obj_2 = _eval_obj_as_nsga2(sol, demand_points_info, charge_points_info, parameter)
        if obj_1 == -1.0 and obj_2 == -1.0:
            return False
        seen.add(key)
        sol_manager.append(np.asarray(sol, dtype=np.float64).copy())
        nonlocal obj_manager
        obj_manager = np.vstack([obj_manager, np.array([[obj_1, obj_2]], dtype=np.float64)])
        return True

    for _ in range(min(pop_num, max(0, kmeans_seed_count))):
        _try_add(_make_legacy_kmeans_seed(demand_points_info, charge_points_info, rng))

    attempts = 0
    while len(sol_manager) < pop_num and attempts < max(300, 100 * pop_num):
        attempts += 1
        if len(sol_manager) > 0 and rng.rand() < 0.6:
            base = sol_manager[int(rng.randint(0, len(sol_manager)))].copy()
            if rng.rand() < 0.5:
                cand = variate(base, charge_points_num, demand_points_num, rng)
            else:
                mate = sol_manager[int(rng.randint(0, len(sol_manager)))]
                cand = cross(base, charge_points_num, mate, rng)
        else:
            cand = init_sol(charge_points_num, demand_points_num, rng)
        _try_add(cand)

    if len(sol_manager) == 0:
        raise RuntimeError("Failed to build any HNSGA-II + K-Means initial solution.")

    while len(sol_manager) < pop_num:
        idx = len(sol_manager) % len(sol_manager)
        sol_manager.append(sol_manager[idx].copy())
        obj_manager = np.vstack([obj_manager, obj_manager[idx, :].reshape(1, 2)])

    out = dict(settings)
    out["obj_manager"] = obj_manager[:pop_num].copy()
    out["sol_manager"] = [sol_manager[i].copy() for i in range(pop_num)]
    return out


def _run_typed_hnsga2_kmeans(
    settings: Dict[str, Any],
    rng: np.random.RandomState | None,
    return_trace: bool,
) -> np.ndarray | Tuple[np.ndarray, HNSGA2KMeansTrace]:
    if rng is None:
        rng = np.random.RandomState(2)

    problem = settings["_typed_problem"]
    invalid_penalty = float(problem.invalid_penalty)
    max_iter_num = int(settings["maxgen"])
    pop_num = int(settings["popnum"])
    variate_rate = float(settings.get("variate_rate", 0.3))
    cross_rate = float(settings.get("cross_rate", 0.8))
    selection_n_clusters = int(settings.get("selection_n_clusters", max(2, int(round(np.sqrt(pop_num))))))

    use_given_initial = (
        "obj_manager" in settings
        and "sol_manager" in settings
        and not bool(settings.get("force_kmeans_initial", False))
    )

    if use_given_initial:
        obj_manager = np.asarray(settings["obj_manager"], dtype=np.float64).copy()
        sol_manager = [sanitize_typed_solution(np.asarray(s, dtype=float), problem) for s in settings["sol_manager"]]
    else:
        init_settings = build_initial_population_for_hnsga2_kmeans(settings, rng)
        obj_manager = np.asarray(init_settings["obj_manager"], dtype=np.float64).copy()
        sol_manager = [np.asarray(s, dtype=np.float64).reshape(-1).copy() for s in init_settings["sol_manager"]]

    generation_metrics: List[np.ndarray] = []
    cluster_count_history: List[float] = []

    for iter_num in range(1, max_iter_num + 1):
        ranks, crowding, labels, weights = _build_selection_state(
            obj_manager,
            rng=rng,
            selection_n_clusters=selection_n_clusters,
        )
        cluster_count_history.append(float(np.unique(labels).size))

        offspring_sols: List[np.ndarray] = []
        offspring_objs: List[np.ndarray] = []
        attempts = 0
        max_attempts = max(200, 20 * pop_num)
        while len(offspring_sols) < pop_num and attempts < max_attempts:
            attempts += 1
            p1 = _select_hnsga_parent(labels, weights, ranks, crowding, rng)
            p2 = _select_hnsga_parent(labels, weights, ranks, crowding, rng)

            child = sol_manager[p1].copy()
            if rng.rand() < cross_rate:
                child = typed_station_crossover(child, sol_manager[p2], problem, rng)
            if rng.rand() < variate_rate:
                child = typed_station_mutation(child, problem, rng)

            child = sanitize_typed_solution(child, problem)
            obj = evaluate_typed_solution(child, problem)
            if is_feasible_obj(obj, invalid_penalty):
                offspring_sols.append(child)
                offspring_objs.append(obj)

        while len(offspring_sols) < pop_num:
            base = sol_manager[int(rng.randint(0, len(sol_manager)))] if len(sol_manager) > 0 else random_typed_solution(problem, rng)
            child = typed_station_mutation(base, problem, rng)
            obj = evaluate_typed_solution(child, problem)
            offspring_sols.append(child)
            offspring_objs.append(obj)

        combined_pop = np.vstack(
            [np.vstack([s.reshape(1, -1) for s in sol_manager]), np.vstack([s.reshape(1, -1) for s in offspring_sols])]
        )
        combined_obj = np.vstack([obj_manager, np.vstack([o.reshape(1, 2) for o in offspring_objs])])
        selected_idx = _environmental_select_indices(combined_obj, pop_num)
        sol_manager = [combined_pop[i, :].copy() for i in selected_idx]
        obj_manager = combined_obj[selected_idx, :].copy()

        generation_metrics.append(generation_metrics_row(iter_num, obj_manager, invalid_penalty))

    final_pop, final_obj = extract_feasible_first_front(
        pop=np.vstack([sol.reshape(1, -1) for sol in sol_manager]) if len(sol_manager) > 0 else None,
        objs=obj_manager,
        invalid_penalty=invalid_penalty,
    )
    _ = final_pop

    final_front_sorted = final_obj[np.argsort(final_obj[:, 0]), :] if final_obj.shape[0] > 0 else final_obj.copy()
    trace = HNSGA2KMeansTrace(
        generation_metrics=np.vstack(generation_metrics) if generation_metrics else np.empty((0, 6), dtype=np.float64),
        final_front_sorted=final_front_sorted,
        cluster_count_history=np.asarray(cluster_count_history, dtype=np.float64),
    )
    if return_trace:
        return final_obj, trace
    return final_obj


def _run_legacy_hnsga2_kmeans(
    settings: Dict[str, Any],
    rng: np.random.RandomState | None,
    return_trace: bool,
) -> np.ndarray | Tuple[np.ndarray, HNSGA2KMeansTrace]:
    if rng is None:
        rng = np.random.RandomState(2)

    if "obj_manager" not in settings or "sol_manager" not in settings:
        settings = build_initial_population_for_hnsga2_kmeans(settings, rng)

    demand_points_info = np.asarray(np.loadtxt(settings["data1"], delimiter=","), dtype=np.float64)
    charge_points_info = np.asarray(np.loadtxt(settings["data2"], delimiter=","), dtype=np.float64)
    parameter = get_parameter()

    demand_points_num = int(demand_points_info.shape[0])
    charge_points_num = int(charge_points_info.shape[0])
    max_iter_num = int(settings["maxgen"])
    pop_num = int(settings["popnum"])
    variate_rate = float(settings.get("variate_rate", 0.3))
    cross_rate = float(settings.get("cross_rate", 0.8))
    selection_n_clusters = int(settings.get("selection_n_clusters", max(2, int(round(np.sqrt(pop_num))))))

    obj_manager = np.asarray(settings["obj_manager"], dtype=np.float64).copy()
    sol_manager = [np.asarray(s, dtype=np.float64).reshape(-1).copy() for s in settings["sol_manager"]]

    generation_metrics: List[np.ndarray] = []
    cluster_count_history: List[float] = []

    for iter_num in range(1, max_iter_num + 1):
        ranks, crowding, labels, weights = _build_selection_state(
            obj_manager,
            rng=rng,
            selection_n_clusters=selection_n_clusters,
        )
        cluster_count_history.append(float(np.unique(labels).size))

        offspring_sols: List[np.ndarray] = []
        offspring_objs: List[np.ndarray] = []
        attempts = 0
        while len(offspring_sols) < pop_num and attempts < max(200, 20 * pop_num):
            attempts += 1
            p1 = _select_hnsga_parent(labels, weights, ranks, crowding, rng)
            p2 = _select_hnsga_parent(labels, weights, ranks, crowding, rng)
            child = sol_manager[p1].copy()
            if rng.rand() < cross_rate:
                child = cross(child, charge_points_num, sol_manager[p2], rng)
            if rng.rand() < variate_rate:
                child = variate(child, charge_points_num, demand_points_num, rng)
            obj_1, obj_2 = _eval_obj_as_nsga2(child, demand_points_info, charge_points_info, parameter)
            if obj_1 != -1.0 or obj_2 != -1.0:
                offspring_sols.append(child)
                offspring_objs.append(np.array([obj_1, obj_2], dtype=np.float64))

        if len(offspring_sols) == 0:
            offspring_sols = [sol_manager[int(rng.randint(0, len(sol_manager)))].copy()]
            offspring_objs = [obj_manager[int(rng.randint(0, obj_manager.shape[0])), :].copy()]

        while len(offspring_sols) < pop_num:
            idx = len(offspring_sols) % len(offspring_sols)
            offspring_sols.append(offspring_sols[idx].copy())
            offspring_objs.append(offspring_objs[idx].copy())

        combined_pop = np.vstack(
            [np.vstack([s.reshape(1, -1) for s in sol_manager]), np.vstack([s.reshape(1, -1) for s in offspring_sols])]
        )
        combined_obj = np.vstack([obj_manager, np.vstack([o.reshape(1, 2) for o in offspring_objs])])
        selected_idx = _environmental_select_indices(combined_obj, pop_num)
        sol_manager = [combined_pop[i, :].copy() for i in selected_idx]
        obj_manager = combined_obj[selected_idx, :].copy()

        generation_metrics.append(_legacy_generation_metrics_row(iter_num, obj_manager))

    final_obj = _legacy_first_front(obj_manager)
    final_front_sorted = final_obj[np.argsort(final_obj[:, 0]), :] if final_obj.shape[0] > 0 else final_obj.copy()
    trace = HNSGA2KMeansTrace(
        generation_metrics=np.vstack(generation_metrics) if generation_metrics else np.empty((0, 6), dtype=np.float64),
        final_front_sorted=final_front_sorted,
        cluster_count_history=np.asarray(cluster_count_history, dtype=np.float64),
    )
    if return_trace:
        return final_obj, trace
    return final_obj


def _default_kmeans_seed_count(pop_num: int) -> int:
    return max(4, min(12, max(1, int(pop_num) // 3)))


def _build_kmeans_guided_typed_initial_population(
    problem,
    pop_size: int,
    rng: np.random.RandomState,
    kmeans_seed_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    pop_size = int(pop_size)
    kmeans_seed_count = max(0, int(kmeans_seed_count))
    sols: list[np.ndarray] = []
    objs: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()

    def _key(sol: np.ndarray) -> tuple[int, ...]:
        return tuple(np.round(sol).astype(int).tolist())

    def _try_add(sol: np.ndarray, require_feasible: bool = True) -> bool:
        sol = sanitize_typed_solution(sol, problem)
        key = _key(sol)
        if key in seen:
            return False
        obj = evaluate_typed_solution(sol, problem)
        feasible = is_feasible_obj(obj, problem.invalid_penalty)
        if require_feasible and not feasible:
            return False
        seen.add(key)
        sols.append(sol)
        objs.append(obj)
        return True

    for _ in range(min(kmeans_seed_count, pop_size)):
        _try_add(_make_typed_kmeans_seed(problem, rng), require_feasible=True)

    attempts = 0
    while len(sols) < pop_size and attempts < max(300, 100 * pop_size):
        attempts += 1
        if len(sols) > 0 and rng.rand() < 0.85:
            base = sols[int(rng.randint(0, len(sols)))]
            cand = typed_station_mutation(base, problem, rng)
            if len(sols) > 1 and rng.rand() < 0.4:
                mate = sols[int(rng.randint(0, len(sols)))]
                cand = typed_station_crossover(cand, mate, problem, rng)
        else:
            cand = random_typed_solution(problem, rng)
        _try_add(cand, require_feasible=True)

    while len(sols) < pop_size and len(sols) > 0:
        idx = len(sols) % len(sols)
        sols.append(sols[idx].copy())
        objs.append(objs[idx].copy())

    if len(sols) == 0:
        raise RuntimeError("Failed to build any typed HNSGA-II + K-Means seed.")

    return (
        np.vstack([s.reshape(1, -1) for s in sols[:pop_size]]).astype(np.float64),
        np.vstack([o.reshape(1, 2) for o in objs[:pop_size]]).astype(np.float64),
    )


def _make_typed_kmeans_seed(problem, rng: np.random.RandomState) -> np.ndarray:
    m = int(problem.charge_points_info.shape[0])
    demand_xy = np.asarray(problem.demand_points_info[:, :2], dtype=np.float64)
    charge_xy = np.asarray(problem.charge_points_info[:, :2], dtype=np.float64)
    demand_weight = _demand_weights(problem.demand_points_info)
    type_nmax = np.asarray(problem.problem_context["type_nmax"], dtype=int).reshape(3)
    type_prob = np.asarray(problem.problem_context["type_prob"], dtype=float).reshape(3)
    total_cap = int(np.sum(type_nmax))

    k_low = max(2, min(5, m))
    k_high = max(k_low, min(m, max(8, int(round(0.45 * m)))))
    n_clusters = int(rng.randint(k_low, k_high + 1))

    labels, centers = _fit_kmeans(demand_xy, demand_weight, n_clusters, rng)
    selected = _nearest_unique_charge_indices(centers, charge_xy)
    if selected.size == 0:
        selected = rng.choice(m, size=min(n_clusters, m), replace=False)

    cluster_demand = np.bincount(labels, weights=demand_weight, minlength=n_clusters).astype(np.float64)
    positive = cluster_demand[cluster_demand > 0]
    demand_scale = float(np.percentile(positive, 75)) if positive.size > 0 else 1.0
    demand_scale = max(demand_scale, 1.0)

    n_slow = np.zeros(m, dtype=int)
    n_fast = np.zeros(m, dtype=int)
    n_ultra = np.zeros(m, dtype=int)

    for cluster_id, station_id in enumerate(selected):
        rel = float(cluster_demand[min(cluster_id, cluster_demand.size - 1)] / demand_scale)
        rel = max(0.15, min(1.4, rel))
        jitter = float(rng.uniform(0.8, 1.2))
        target_total = int(np.ceil(min(total_cap, max(1.0, rel * jitter * total_cap * 0.65))))
        counts = split_total_into_types(
            total_piles=target_total,
            nmax_vec=type_nmax,
            type_prob=type_prob,
            stochastic=True,
        )
        j = int(station_id)
        n_slow[j], n_fast[j], n_ultra[j] = counts.astype(int).tolist()

    return sanitize_typed_solution(np.concatenate([n_slow, n_fast, n_ultra], axis=0).astype(float), problem)


def _make_legacy_kmeans_seed(
    demand_points_info: np.ndarray,
    charge_points_info: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    m = int(charge_points_info.shape[0])
    demand_num = int(demand_points_info.shape[0])
    demand_xy = np.asarray(demand_points_info[:, :2], dtype=np.float64)
    charge_xy = np.asarray(charge_points_info[:, :2], dtype=np.float64)
    demand_weight = _demand_weights(demand_points_info)

    k_low = max(1, min(3, m))
    k_high = max(k_low, min(m, max(4, int(round(0.45 * m)))))
    n_clusters = int(rng.randint(k_low, k_high + 1))
    labels, centers = _fit_kmeans(demand_xy, demand_weight, n_clusters, rng)
    selected = _nearest_unique_charge_indices(centers, charge_xy)

    cluster_demand = np.bincount(labels, weights=demand_weight, minlength=n_clusters).astype(np.float64)
    positive = cluster_demand[cluster_demand > 0]
    demand_scale = float(np.percentile(positive, 75)) if positive.size > 0 else 1.0
    demand_scale = max(demand_scale, 1.0)

    sol = np.zeros(m, dtype=np.float64)
    for cluster_id, station_id in enumerate(selected):
        rel = float(cluster_demand[min(cluster_id, cluster_demand.size - 1)] / demand_scale)
        target = int(round(max(1.0, min(demand_num + 50, rel * 0.5 * (demand_num + 50)))))
        sol[int(station_id)] = np.float64(max(1, target))
    return sol


def _fit_kmeans(
    points: np.ndarray,
    weights: np.ndarray,
    n_clusters: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    n_clusters = max(1, min(int(n_clusters), points.shape[0]))
    if points.shape[0] == 0:
        return np.zeros(0, dtype=int), np.empty((0, 2), dtype=np.float64)
    if KMeans is None or points.shape[0] < n_clusters:
        chosen = rng.choice(points.shape[0], size=n_clusters, replace=False)
        labels = np.arange(points.shape[0], dtype=int) % n_clusters
        return labels, points[chosen, :]

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=int(rng.randint(0, 2**31 - 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            km.fit(points, sample_weight=np.asarray(weights, dtype=np.float64))
        except TypeError:
            km.fit(points)
    return np.asarray(km.labels_, dtype=int), np.asarray(km.cluster_centers_, dtype=np.float64)


def _nearest_unique_charge_indices(centers: np.ndarray, charge_xy: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    charge_xy = np.asarray(charge_xy, dtype=np.float64)
    if centers.size == 0 or charge_xy.size == 0:
        return np.empty(0, dtype=int)

    selected: list[int] = []
    used: set[int] = set()
    for center in centers:
        dist = np.sum((charge_xy - center.reshape(1, -1)) ** 2, axis=1)
        for idx in np.argsort(dist):
            j = int(idx)
            if j not in used:
                used.add(j)
                selected.append(j)
                break
    return np.asarray(selected, dtype=int)


def _demand_weights(demand_points_info: np.ndarray) -> np.ndarray:
    demand_points_info = np.asarray(demand_points_info, dtype=np.float64)
    if demand_points_info.ndim == 2 and demand_points_info.shape[1] >= 3:
        weights = np.asarray(demand_points_info[:, 2], dtype=np.float64)
        if np.any(np.isfinite(weights) & (weights > 0)):
            return np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    return np.ones(demand_points_info.shape[0], dtype=np.float64)


def _build_selection_state(
    obj_manager: np.ndarray,
    rng: np.random.RandomState,
    selection_n_clusters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obj_manager = np.asarray(obj_manager, dtype=np.float64)
    fronts, crowding = fast_non_dominated_sort_with_crowding(obj_manager)
    ranks = np.full(obj_manager.shape[0], len(fronts) + 1, dtype=np.int64)
    for rank_id, front in enumerate(fronts, start=1):
        ranks[front] = rank_id
    labels = _cluster_objectives_for_selection(obj_manager, selection_n_clusters, rng)
    weights = _roulette_weights(ranks, crowding, obj_manager)
    return ranks, crowding, labels, weights


def _cluster_objectives_for_selection(
    obj_manager: np.ndarray,
    selection_n_clusters: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    obj_manager = np.asarray(obj_manager, dtype=np.float64)
    n = int(obj_manager.shape[0])
    if n <= 2:
        return np.zeros(n, dtype=int)

    x = obj_manager.copy()
    finite = np.all(np.isfinite(x), axis=1)
    if np.any(finite):
        finite_vals = x[finite, :]
        high = np.max(finite_vals, axis=0) + np.maximum(np.ptp(finite_vals, axis=0), 1.0)
        x[~finite, :] = high.reshape(1, -1)
    else:
        return np.zeros(n, dtype=int)

    denom = np.ptp(x, axis=0)
    denom[denom <= 0] = 1.0
    x = (x - np.min(x, axis=0).reshape(1, -1)) / denom.reshape(1, -1)

    n_clusters = max(1, min(int(selection_n_clusters), n))
    if n_clusters == 1:
        return np.zeros(n, dtype=int)

    if Birch is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = Birch(n_clusters=n_clusters, threshold=0.35).fit_predict(x)
            return np.asarray(labels, dtype=int)
        except Exception:
            pass

    if KMeans is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = KMeans(
                    n_clusters=n_clusters,
                    n_init=10,
                    random_state=int(rng.randint(0, 2**31 - 1)),
                ).fit_predict(x)
            return np.asarray(labels, dtype=int)
        except Exception:
            pass

    return np.arange(n, dtype=int) % n_clusters


def _roulette_weights(ranks: np.ndarray, crowding: np.ndarray, obj_manager: np.ndarray) -> np.ndarray:
    ranks = np.asarray(ranks, dtype=np.float64)
    crowding = np.asarray(crowding, dtype=np.float64)
    obj_manager = np.asarray(obj_manager, dtype=np.float64)

    finite_crowd = crowding[np.isfinite(crowding)]
    crowd = crowding.copy()
    if finite_crowd.size == 0:
        crowd[:] = 1.0
    else:
        max_finite = float(np.max(finite_crowd))
        crowd[~np.isfinite(crowd)] = max_finite + 1.0
        min_c = float(np.min(crowd))
        ptp_c = float(np.ptp(crowd))
        crowd = np.ones_like(crowd) if ptp_c <= 0 else 1.0 + (crowd - min_c) / ptp_c

    weights = (1.0 / np.maximum(ranks, 1.0)) * crowd
    feasible = np.all(np.isfinite(obj_manager), axis=1)
    weights = np.where(feasible, weights, 1e-12)
    if not np.any(weights > 0):
        weights = np.ones_like(weights, dtype=np.float64)
    return weights.astype(np.float64)


def _select_hnsga_parent(
    labels: np.ndarray,
    weights: np.ndarray,
    ranks: np.ndarray,
    crowding: np.ndarray,
    rng: np.random.RandomState,
) -> int:
    labels = np.asarray(labels, dtype=int)
    unique = np.unique(labels)
    if unique.size == 0:
        return 0

    if unique.size == 1:
        cluster_a = unique[0]
        cluster_b = unique[0]
    else:
        sizes = np.asarray([np.sum(labels == u) for u in unique], dtype=np.float64)
        probs = np.sqrt(np.maximum(sizes, 1.0))
        probs = probs / np.sum(probs)
        chosen = rng.choice(unique, size=2, replace=False, p=probs)
        cluster_a, cluster_b = int(chosen[0]), int(chosen[1])

    idx_a = _roulette_pick(np.where(labels == cluster_a)[0], weights, rng)
    idx_b = _roulette_pick(np.where(labels == cluster_b)[0], weights, rng)
    return _binary_tournament(idx_a, idx_b, ranks, crowding, rng)


def _roulette_pick(indices: np.ndarray, weights: np.ndarray, rng: np.random.RandomState) -> int:
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return int(rng.randint(0, weights.size))
    local_w = np.asarray(weights[indices], dtype=np.float64)
    total = float(np.sum(local_w))
    if total <= 0 or not np.isfinite(total):
        return int(indices[int(rng.randint(0, indices.size))])
    threshold = float(rng.rand() * total)
    cumsum = np.cumsum(local_w)
    return int(indices[int(np.searchsorted(cumsum, threshold, side="right").clip(0, indices.size - 1))])


def _binary_tournament(
    idx_a: int,
    idx_b: int,
    ranks: np.ndarray,
    crowding: np.ndarray,
    rng: np.random.RandomState,
) -> int:
    if ranks[idx_a] < ranks[idx_b]:
        return int(idx_a)
    if ranks[idx_b] < ranks[idx_a]:
        return int(idx_b)
    crowd_a = crowding[idx_a]
    crowd_b = crowding[idx_b]
    if not np.isfinite(crowd_a):
        crowd_a = np.finfo(np.float64).max
    if not np.isfinite(crowd_b):
        crowd_b = np.finfo(np.float64).max
    if crowd_a > crowd_b:
        return int(idx_a)
    if crowd_b > crowd_a:
        return int(idx_b)
    return int(idx_a if rng.rand() < 0.5 else idx_b)


def _environmental_select_indices(obj_manager: np.ndarray, pop_num: int) -> np.ndarray:
    fronts, crowding = fast_non_dominated_sort_with_crowding(obj_manager)
    selected: list[int] = []
    for front in fronts:
        if len(selected) + front.size <= pop_num:
            selected.extend([int(i) for i in front])
            continue
        remaining = pop_num - len(selected)
        if remaining <= 0:
            break
        order = np.argsort(crowding[front])[::-1]
        selected.extend([int(front[int(i)]) for i in order[:remaining]])
        break
    if len(selected) < pop_num:
        missing = pop_num - len(selected)
        all_idx = np.arange(obj_manager.shape[0], dtype=int)
        unused = [int(i) for i in all_idx if int(i) not in set(selected)]
        selected.extend(unused[:missing])
    return np.asarray(selected[:pop_num], dtype=int)


def _eval_obj_as_nsga2(
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


def _legacy_generation_metrics_row(gen: int, obj_manager: np.ndarray) -> np.ndarray:
    fronts, _ = fast_non_dominated_sort_with_crowding(obj_manager)
    nd_count = fronts[0].size if len(fronts) > 0 else 0
    return np.array(
        [
            gen,
            float(np.min(obj_manager[:, 0])) if obj_manager.size > 0 else np.inf,
            float(np.min(obj_manager[:, 1])) if obj_manager.size > 0 else np.inf,
            float(np.mean(obj_manager[:, 0])) if obj_manager.size > 0 else np.inf,
            float(np.mean(obj_manager[:, 1])) if obj_manager.size > 0 else np.inf,
            float(nd_count),
        ],
        dtype=np.float64,
    )


def _legacy_first_front(obj_manager: np.ndarray) -> np.ndarray:
    fronts, _ = fast_non_dominated_sort_with_crowding(obj_manager)
    if len(fronts) == 0 or fronts[0].size == 0:
        return np.empty((0, 2), dtype=np.float64)
    front = obj_manager[fronts[0], :].copy()
    if front.shape[0] > 1:
        _, unique_idx = np.unique(front, axis=0, return_index=True)
        front = front[np.sort(unique_idx), :]
    return front[np.argsort(front[:, 0]), :]
