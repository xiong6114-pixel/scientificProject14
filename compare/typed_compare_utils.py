from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import torch


ROOT_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_PROJECT_DIR))

from cal_obj_typed_py import cal_obj_typed, build_default_type_info, build_distance_matrix
from ev_typed_nn_seed_adapter import (
    build_ev_typed_problem_context,
    split_total_into_types,
    typed_repair_solution,
)
from get_mofcn import clear_ev_context, set_ev_context
from get_parameter import get_parameter
from nn_seed_injection_50 import SeedDecodeConfig, make_pcc_seed_builder
from run_ev_typed_mogabka_seeded import DEFAULT_CKPT_PATH, _load_runtime_inputs


@dataclass
class TypedCompareProblem:
    source_desc: str
    ckpt_path: Path
    checkpoint_available: bool
    demand_points_info: np.ndarray
    charge_points_info: np.ndarray
    x_raw_flat: np.ndarray
    parameter: np.ndarray
    type_info: object
    dist_matrix: np.ndarray
    problem_context: dict
    lb_vec: np.ndarray
    ub_vec: np.ndarray
    dim: int
    invalid_penalty: float
    seed_builder: Callable[..., np.ndarray] | None


def build_typed_compare_problem(
    seed_device: str = "cpu",
    invalid_penalty: float = 1e12,
    type_prob: tuple[float, float, float] = (0.50, 0.35, 0.15),
    stochastic_split: bool = False,
) -> TypedCompareProblem:
    ckpt_path = Path(os.environ.get("CKPT_PATH", str(DEFAULT_CKPT_PATH)))
    ckpt = None
    expected_points = None
    expected_in_dim = None
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        expected_points = int(ckpt["n_points"])
        feat_dim = int(ckpt["feat_dim"])
        expected_in_dim = int(ckpt.get("in_dim", expected_points * feat_dim))

    demand_points_info, charge_points_info, x_raw_flat, source_desc = _load_runtime_inputs(
        expected_points=expected_points,
        expected_in_dim=expected_in_dim,
    )

    parameter = get_parameter()
    type_info = build_default_type_info(
        parameter=parameter,
        nmax=np.array([20, 20, 10], dtype=int),
    )

    m = int(charge_points_info.shape[0])
    dist_matrix = build_distance_matrix(
        demand_points_info[:, 0:2],
        charge_points_info[:, 0:2],
    )

    problem_context = build_ev_typed_problem_context(
        x_raw_flat=x_raw_flat,
        charge_points_num=m,
        cal_obj_fn=cal_obj_typed,
        demand_points_info=demand_points_info,
        charge_points_info=charge_points_info,
        parameter=parameter,
        typeInfo=type_info,
        type_prob=type_prob,
        fallback_index=0,
        stochastic_split=stochastic_split,
        dist_matrix=dist_matrix,
        invalid_penalty=invalid_penalty,
        debug_seed_injection=False,
        debug_metrics=False,
    )

    seed_builder = None
    if ckpt is not None:
        decode_cfg = SeedDecodeConfig(
            num_seeds=12,
            count_radius=2,
            top_margin=4,
            stochastic_ratio=0.75,
            priority_temperature=1.0,
            capacity_temperature=1.0,
            min_k=1,
            max_k=m,
        )
        seed_builder = make_pcc_seed_builder(
            ckpt_path=str(ckpt_path),
            decode_cfg=decode_cfg,
            device=seed_device,
        )

    ub_vec = np.concatenate(
        [
            np.full(m, type_info.nmax[0], dtype=float),
            np.full(m, type_info.nmax[1], dtype=float),
            np.full(m, type_info.nmax[2], dtype=float),
        ]
    )
    lb_vec = np.zeros(3 * m, dtype=float)

    return TypedCompareProblem(
        source_desc=source_desc,
        ckpt_path=ckpt_path,
        checkpoint_available=ckpt is not None,
        demand_points_info=np.asarray(demand_points_info, dtype=float),
        charge_points_info=np.asarray(charge_points_info, dtype=float),
        x_raw_flat=np.asarray(x_raw_flat, dtype=np.float32),
        parameter=np.asarray(parameter, dtype=float),
        type_info=type_info,
        dist_matrix=np.asarray(dist_matrix, dtype=float),
        problem_context=problem_context,
        lb_vec=lb_vec,
        ub_vec=np.asarray(ub_vec, dtype=float),
        dim=int(3 * m),
        invalid_penalty=float(invalid_penalty),
        seed_builder=seed_builder,
    )


@contextmanager
def activated_ev_context(problem: TypedCompareProblem):
    set_ev_context(problem.problem_context)
    try:
        yield
    finally:
        clear_ev_context()


def build_mogabka_seed_config(
    problem: TypedCompareProblem,
    init_nn_seed_count: int,
) -> dict:
    enabled = problem.seed_builder is not None and int(init_nn_seed_count) > 0
    return {
        "enabled": enabled,
        "init_enabled": enabled,
        "init_nn_seed_count": int(init_nn_seed_count) if enabled else 0,
        "builder_fn": problem.seed_builder if enabled else None,
        "reinject_enabled": False,
        "reinject_generations": (20, 60),
        "reinject_count": 4,
    }


def sanitize_typed_solution(sol: np.ndarray, problem: TypedCompareProblem) -> np.ndarray:
    sol = np.asarray(sol, dtype=float).reshape(-1).copy()
    if sol.size != problem.dim:
        raise ValueError(f"solution dim mismatch: expected {problem.dim}, got {sol.size}")

    sol = np.clip(sol, problem.lb_vec, problem.ub_vec)
    sol = np.round(sol)
    sol = typed_repair_solution(sol, context=problem.problem_context)
    sol = np.clip(sol, problem.lb_vec, problem.ub_vec)
    return np.round(sol).astype(float)


def evaluate_typed_solution(sol: np.ndarray, problem: TypedCompareProblem) -> np.ndarray:
    sol = sanitize_typed_solution(sol, problem)
    evaluate_fn = problem.problem_context["evaluate_fn"]
    obj = evaluate_fn(sol, problem.problem_context)
    if obj is None:
        return np.full(2, problem.invalid_penalty, dtype=float)
    obj = np.asarray(obj, dtype=float).reshape(-1)
    if obj.size != 2 or (not np.all(np.isfinite(obj))):
        return np.full(2, problem.invalid_penalty, dtype=float)
    return obj


def is_feasible_obj(obj: np.ndarray, invalid_penalty: float) -> bool:
    obj = np.asarray(obj, dtype=float).reshape(-1)
    return bool(obj.size == 2 and np.all(np.isfinite(obj)) and np.all(obj < float(invalid_penalty)))


def typed_station_crossover(
    sol: np.ndarray,
    parent: np.ndarray,
    problem: TypedCompareProblem,
    rng: np.random.RandomState,
) -> np.ndarray:
    m = problem.charge_points_info.shape[0]
    child = sanitize_typed_solution(sol, problem)
    parent = sanitize_typed_solution(parent, problem)

    n_take = 1 if m == 1 else int(rng.randint(1, min(4, m) + 1))
    station_ids = rng.choice(m, size=n_take, replace=False)
    for j in np.asarray(station_ids, dtype=int):
        child[j] = parent[j]
        child[m + j] = parent[m + j]
        child[2 * m + j] = parent[2 * m + j]
    return sanitize_typed_solution(child, problem)


def typed_station_mutation(
    sol: np.ndarray,
    problem: TypedCompareProblem,
    rng: np.random.RandomState,
) -> np.ndarray:
    sol = sanitize_typed_solution(sol, problem)
    m = problem.charge_points_info.shape[0]
    type_nmax = np.asarray(problem.problem_context["type_nmax"], dtype=int).reshape(3)
    type_prob = np.asarray(problem.problem_context["type_prob"], dtype=float).reshape(3)
    total_cap = int(np.sum(type_nmax))

    n_slow = sol[:m].astype(int)
    n_fast = sol[m : 2 * m].astype(int)
    n_ultra = sol[2 * m :].astype(int)

    n_change = 1 if m == 1 else int(rng.randint(1, min(4, m) + 1))
    station_ids = rng.choice(m, size=n_change, replace=False)

    for j in np.asarray(station_ids, dtype=int):
        current = np.array([n_slow[j], n_fast[j], n_ultra[j]], dtype=int)
        mode = float(rng.rand())

        if mode < 0.55:
            delta = rng.randint(-2, 3, size=3)
            proposal = np.clip(current + delta, 0, type_nmax)
            if proposal.sum() == 0 and current.sum() > 0:
                proposal = current.copy()
                kk = int(rng.randint(0, 3))
                if proposal[kk] < type_nmax[kk]:
                    proposal[kk] += 1
        elif mode < 0.85:
            target_total = int(current.sum() + rng.randint(-6, 7))
            target_total = max(0, min(total_cap, target_total))
            proposal = split_total_into_types(
                total_piles=target_total,
                nmax_vec=type_nmax,
                type_prob=type_prob,
                stochastic=True,
            )
        else:
            if current.sum() == 0 or rng.rand() < 0.65:
                target_total = int(rng.randint(1, total_cap + 1))
                proposal = split_total_into_types(
                    total_piles=target_total,
                    nmax_vec=type_nmax,
                    type_prob=type_prob,
                    stochastic=True,
                )
            else:
                proposal = np.zeros(3, dtype=int)

        n_slow[j], n_fast[j], n_ultra[j] = proposal.astype(int).tolist()

    mutated = np.concatenate([n_slow, n_fast, n_ultra], axis=0).astype(float)
    return sanitize_typed_solution(mutated, problem)


def random_typed_solution(problem: TypedCompareProblem, rng: np.random.RandomState) -> np.ndarray:
    m = problem.charge_points_info.shape[0]
    type_nmax = np.asarray(problem.problem_context["type_nmax"], dtype=int).reshape(3)
    type_prob = np.asarray(problem.problem_context["type_prob"], dtype=float).reshape(3)
    total_cap = int(np.sum(type_nmax))

    n_slow = np.zeros(m, dtype=int)
    n_fast = np.zeros(m, dtype=int)
    n_ultra = np.zeros(m, dtype=int)

    k_low = max(4, min(10, m))
    k_high = max(k_low, min(m, max(12, m // 2)))
    built_k = int(rng.randint(k_low, k_high + 1))
    station_ids = rng.choice(m, size=built_k, replace=False)

    for j in np.asarray(station_ids, dtype=int):
        total = int(rng.randint(1, total_cap + 1))
        counts = split_total_into_types(
            total_piles=total,
            nmax_vec=type_nmax,
            type_prob=type_prob,
            stochastic=True,
        )
        n_slow[j], n_fast[j], n_ultra[j] = counts.astype(int).tolist()

    sol = np.concatenate([n_slow, n_fast, n_ultra], axis=0).astype(float)
    return sanitize_typed_solution(sol, problem)


def build_typed_initial_population(
    problem: TypedCompareProblem,
    pop_size: int,
    rng: np.random.RandomState,
    init_nn_seed_count: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    pop_size = int(pop_size)
    init_nn_seed_count = max(0, int(init_nn_seed_count))

    sols: list[np.ndarray] = []
    objs: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()

    def _key(sol: np.ndarray) -> tuple[int, ...]:
        return tuple(np.round(sol).astype(int).tolist())

    def _try_add(sol: np.ndarray, require_feasible: bool) -> bool:
        sol = sanitize_typed_solution(sol, problem)
        key = _key(sol)
        if key in seen:
            return False

        obj = evaluate_typed_solution(sol, problem)
        feasible = is_feasible_obj(obj, problem.invalid_penalty)
        if require_feasible and (not feasible):
            return False

        seen.add(key)
        sols.append(sol)
        objs.append(obj)
        return True

    if problem.seed_builder is not None and init_nn_seed_count > 0:
        state = np.random.get_state()
        np.random.seed(int(rng.randint(0, 2**31 - 1)))
        try:
            seed_pop = problem.seed_builder(
                init_nn_seed_count,
                problem.dim,
                problem.lb_vec,
                problem.ub_vec,
                problem.problem_context,
            )
        finally:
            np.random.set_state(state)

        seed_pop = np.asarray(seed_pop, dtype=float)
        if seed_pop.ndim == 1:
            seed_pop = seed_pop.reshape(1, -1)
        for i in range(seed_pop.shape[0]):
            if len(sols) >= pop_size:
                break
            _try_add(seed_pop[i, :], require_feasible=True)

    attempts = 0
    max_attempts = max(200, 80 * pop_size)
    while len(sols) < pop_size and attempts < max_attempts:
        attempts += 1
        if len(sols) > 0 and rng.rand() < 0.85:
            base = sols[int(rng.randint(0, len(sols)))].copy()
            cand = typed_station_mutation(base, problem, rng)
            if len(sols) > 1 and rng.rand() < 0.35:
                mate = sols[int(rng.randint(0, len(sols)))]
                cand = typed_station_crossover(cand, mate, problem, rng)
        else:
            cand = random_typed_solution(problem, rng)
        _try_add(cand, require_feasible=True)

    fallback_attempts = 0
    while len(sols) < pop_size and fallback_attempts < max(100, 30 * pop_size):
        fallback_attempts += 1
        if len(sols) > 0:
            base = sols[int(rng.randint(0, len(sols)))].copy()
            cand = typed_station_mutation(base, problem, rng)
        else:
            cand = random_typed_solution(problem, rng)
        _try_add(cand, require_feasible=False)

    if len(sols) == 0:
        raise RuntimeError("Failed to build any typed initial solution.")

    if len(sols) < pop_size:
        base_sols = [sol.copy() for sol in sols]
        base_objs = [obj.copy() for obj in objs]
        while len(sols) < pop_size:
            idx = len(sols) % len(base_sols)
            sols.append(base_sols[idx].copy())
            objs.append(base_objs[idx].copy())

    pop = np.vstack([np.asarray(sol, dtype=float).reshape(1, -1) for sol in sols[:pop_size]])
    obj = np.vstack([np.asarray(f, dtype=float).reshape(1, -1) for f in objs[:pop_size]])
    return pop, obj


def first_front_mask(objs: np.ndarray, invalid_penalty: float) -> np.ndarray:
    objs = np.asarray(objs, dtype=float)
    if objs.ndim != 2:
        raise ValueError("objs must be 2D")
    n = objs.shape[0]
    keep = np.zeros(n, dtype=bool)
    feasible = np.all(np.isfinite(objs), axis=1) & np.all(objs < float(invalid_penalty), axis=1)
    feasible_idx = np.where(feasible)[0]
    if feasible_idx.size == 0:
        return keep

    keep[feasible_idx] = True
    for i in feasible_idx:
        if not keep[i]:
            continue
        for j in feasible_idx:
            if i == j or (not keep[i]):
                continue
            if np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                keep[i] = False
                break
    return keep


def extract_feasible_first_front(
    pop: np.ndarray | None,
    objs: np.ndarray,
    invalid_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    objs = np.asarray(objs, dtype=float)
    mask = first_front_mask(objs, invalid_penalty)
    front_obj = objs[mask]
    if front_obj.size == 0:
        obj_dim = objs.shape[1] if objs.ndim == 2 else 0
        pop_dim = 0 if pop is None else np.asarray(pop).shape[1]
        return np.empty((0, pop_dim), dtype=float), np.empty((0, obj_dim), dtype=float)

    if pop is None:
        front_pop = np.empty((front_obj.shape[0], 0), dtype=float)
    else:
        pop_arr = np.asarray(pop, dtype=float)
        front_pop = pop_arr[mask]

    if front_obj.shape[0] > 1:
        _, unique_idx = np.unique(front_obj, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        front_obj = front_obj[unique_idx]
        if front_pop.size > 0:
            front_pop = front_pop[unique_idx]

    order = np.argsort(front_obj[:, 0])
    front_obj = front_obj[order]
    if front_pop.size > 0:
        front_pop = front_pop[order]
    return front_pop, front_obj


def generation_metrics_row(
    gen: int,
    objs: np.ndarray,
    invalid_penalty: float,
) -> np.ndarray:
    objs = np.asarray(objs, dtype=float)
    feasible = objs[np.all(np.isfinite(objs), axis=1) & np.all(objs < float(invalid_penalty), axis=1)]
    if feasible.shape[0] == 0:
        return np.array([gen, np.inf, np.inf, np.inf, np.inf, 0.0], dtype=float)

    nd_count = float(np.sum(first_front_mask(feasible, invalid_penalty)))
    return np.array(
        [
            gen,
            float(np.min(feasible[:, 0])),
            float(np.min(feasible[:, 1])),
            float(np.mean(feasible[:, 0])),
            float(np.mean(feasible[:, 1])),
            nd_count,
        ],
        dtype=float,
    )


def merge_fronts_and_extract_reference(fronts: list[np.ndarray], invalid_penalty: float) -> np.ndarray:
    rows = []
    for front in fronts:
        arr = np.asarray(front, dtype=float)
        if arr.ndim == 1 and arr.size > 0:
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            continue
        keep = np.all(np.isfinite(arr), axis=1) & np.all(arr < float(invalid_penalty), axis=1)
        if np.any(keep):
            rows.append(arr[keep])

    if len(rows) == 0:
        return np.empty((0, 2), dtype=float)

    merged = np.vstack(rows)
    _, ref = extract_feasible_first_front(None, merged, invalid_penalty)
    return ref
