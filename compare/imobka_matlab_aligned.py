from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from mobka_common import (
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
from mobka_stage1 import MOBKATrace, run_mobka_stage1
from typed_compare_utils import (
    build_typed_initial_population,
    extract_feasible_first_front,
    sanitize_typed_solution,
    evaluate_typed_solution,
)


def IMOBKA_funciton(
    settings: Dict[str, Any],
    rng: np.random.RandomState | None = None,
    return_trace: bool = False,
) -> np.ndarray | Tuple[np.ndarray, MOBKATrace]:
    """MATLAB mobka/IMOBKA_funciton.m aligned.

    NOTE: In the provided MATLAB file, active logic is effectively same as MOBKA core loop.
    The function name inside that file is also `MOBKA_funciton`.
    """
    if "_typed_problem" in settings:
        return _run_typed_mobka(settings=settings, rng=rng, return_trace=return_trace)
    return run_mobka_stage1(settings=settings, rng=rng, return_trace=return_trace)


# Keep MATLAB-file internal signature alias.
def MOBKA_funciton(settings: Dict[str, Any], rng: np.random.RandomState | None = None, return_trace: bool = False):
    return IMOBKA_funciton(settings=settings, rng=rng, return_trace=return_trace)


def _run_typed_mobka(
    settings: Dict[str, Any],
    rng: np.random.RandomState | None = None,
    return_trace: bool = False,
) -> np.ndarray | Tuple[np.ndarray, MOBKATrace]:
    if rng is None:
        rng = np.random.RandomState(2)

    problem = settings["_typed_problem"]
    maxgen = int(settings["maxgen"])
    sizepop = int(settings["popnum"])
    init_nn_seed_count = int(settings.get("init_nn_seed_count", 0))

    if "sol_manager" in settings and "obj_manager" in settings:
        pop: list[Particle] = []
        for i in range(sizepop):
            pos = sanitize_typed_solution(np.asarray(settings["sol_manager"][i], dtype=float), problem)
            costv = np.asarray(settings["obj_manager"][i], dtype=float).reshape(-1)
            pop.append(
                Particle(
                    Position=pos.copy(),
                    Cost=costv.copy(),
                    BestPosition=pos.copy(),
                    BestCost=costv.copy(),
                )
            )
    else:
        init_pop, init_obj = build_typed_initial_population(
            problem=problem,
            pop_size=sizepop,
            rng=rng,
            init_nn_seed_count=init_nn_seed_count,
        )
        pop = []
        for i in range(sizepop):
            pos = init_pop[i, :].copy()
            costv = init_obj[i, :].copy()
            pop.append(
                Particle(
                    Position=pos.copy(),
                    Cost=costv.copy(),
                    BestPosition=pos.copy(),
                    BestCost=costv.copy(),
                )
            )

    n_rep = 200
    n_grid = 6
    alpha = 0.1
    beta = 1.5
    gamma = 2
    p = 0.8

    lb_vec = np.asarray(problem.lb_vec, dtype=float).reshape(-1)
    ub_vec = np.asarray(problem.ub_vec, dtype=float).reshape(-1)
    dim = int(problem.dim)

    pop = deter_dom(pop)
    rep = [pp for pp in pop if not pp.IsDominated]
    grid = cgrid(rep, n_grid, alpha)
    for i in range(len(rep)):
        rep[i] = fgrid(rep[i], grid)

    archive_size_hist = []
    nd_hist = []
    hv_hist = []

    for it in range(1, maxgen + 1):
        for i in range(sizepop):
            leader = select_leader(rep, beta, rng)

            r = np.float64(rng.rand())
            n = np.float64(0.05 * np.exp(-2 * (it / maxgen) ** 2))

            if p < r:
                candidate = pop[i].Position + n * (1 + np.sin(r)) * pop[i].Position
            else:
                candidate = pop[i].Position * (n * (2 * rng.rand(dim) - 1) + 1)

            m = np.float64(2 * np.sin(r + np.pi / 2))
            s_1b = int(rng.randint(1, sizepop + 1))
            ori_value = rng.rand(dim)
            cauchy_value = np.tan((ori_value - 0.5) * np.pi)

            if dominates_particle(pop[i], pop[s_1b - 1]):
                candidate = candidate + cauchy_value * (pop[i].BestPosition - leader.Position)
            else:
                candidate = candidate + cauchy_value * (leader.Position - m * pop[i].BestPosition)

            candidate = np.clip(candidate, lb_vec, ub_vec)
            candidate = sanitize_typed_solution(candidate, problem)
            pop[i].Position = candidate
            pop[i].Cost = evaluate_typed_solution(candidate, problem)

        pop = deter_dom(pop)
        rep = rep + [pp for pp in pop if not pp.IsDominated]
        rep = determine_domination(rep)
        rep = [pp for pp in rep if not pp.IsDominated]

        if len(rep) == 0:
            archive_size_hist.append(0.0)
            nd_hist.append(0.0)
            hv_hist.append(0.0)
            continue

        grid = cgrid(rep, n_grid, alpha)
        for i in range(len(rep)):
            rep[i] = fgrid(rep[i], grid)

        if len(rep) > n_rep:
            extra = len(rep) - n_rep
            for _ in range(extra):
                rep = delete_one_rep_member(rep, gamma, rng)

        rep_obj = rep_to_obj_matrix(rep)
        _, feasible_front = extract_feasible_first_front(None, rep_obj, problem.invalid_penalty)
        archive_size_hist.append(np.float64(len(rep)))
        nd_hist.append(np.float64(feasible_front.shape[0]))
        hv_hist.append(np.float64(feasible_front.shape[0]))

    rep_obj = rep_to_obj_matrix(rep)
    _, final_obj = extract_feasible_first_front(None, rep_obj, problem.invalid_penalty)

    if return_trace:
        trace = MOBKATrace(
            archive_size_history=np.asarray(archive_size_hist, dtype=np.float64),
            nd_count_history=np.asarray(nd_hist, dtype=np.float64),
            hv_history=np.asarray(hv_hist, dtype=np.float64),
        )
        return final_obj, trace
    return final_obj
