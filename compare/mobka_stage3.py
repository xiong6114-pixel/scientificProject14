from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from mobka_common import (
    Particle,
    cgrid,
    delete_one_rep_member,
    deter_dom,
    determine_domination,
    dominates_cost,
    dominates_particle,
    fgrid,
    rep_to_obj_matrix,
    select_leader,
)
from mobka_metrics import hv_cir_like
from mobka_stage1 import build_initial_particles_mobka, _cost_fun_factory


@dataclass
class MOBKAStage3Trace:
    archive_size_history: np.ndarray
    nd_count_history: np.ndarray
    hv_history: np.ndarray
    ext_archive_size_history: np.ndarray


def levy(dim: int, rng: np.random.RandomState) -> np.ndarray:
    beta = np.float64(1.1)
    num = math.gamma(1 + float(beta)) * np.sin(np.pi * float(beta) / 2)
    den = math.gamma((1 + float(beta)) / 2) * float(beta) * 2 ** ((float(beta) - 1) / 2)
    sigma_u = (num / den) ** (1 / float(beta))

    u = rng.normal(0, sigma_u, size=(dim,))
    v = rng.normal(0, 1, size=(dim,))
    z = u / (np.abs(v) ** (1 / float(beta)))
    return z.astype(np.float64)


def _nondominated_filter(objs: np.ndarray) -> np.ndarray:
    n = objs.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if dominates_cost(objs[j], objs[i]):
                keep[i] = False
                break
    return keep


def _crowding_distance(objs: np.ndarray) -> np.ndarray:
    n = objs.shape[0]
    m = objs.shape[1]
    if n == 0:
        return np.empty((0,), dtype=np.float64)
    if n == 1:
        return np.array([np.inf], dtype=np.float64)

    d = np.zeros(n, dtype=np.float64)
    for mm in range(m):
        idx = np.argsort(objs[:, mm])
        d[idx[0]] = np.inf
        d[idx[-1]] = np.inf
        denom = np.max(objs[:, mm]) - np.min(objs[:, mm])
        if denom == 0:
            continue
        for k in range(1, n - 1):
            d[idx[k]] = d[idx[k]] + (objs[idx[k + 1], mm] - objs[idx[k - 1], mm]) / denom
    return d


def _update_external_archive(
    archive_solutions: np.ndarray,
    archive_objs: np.ndarray,
    candidate_solutions: np.ndarray,
    cost_fun,
    archive_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    new_solutions = []
    new_objs = []

    for i in range(candidate_solutions.shape[0]):
        sol = candidate_solutions[i, :]
        obj = cost_fun(sol)
        if np.all(np.isfinite(obj)):
            new_solutions.append(sol.reshape(1, -1))
            new_objs.append(obj.reshape(1, -1))

    if len(new_solutions) > 0:
        new_solutions_m = np.vstack(new_solutions)
        new_objs_m = np.vstack(new_objs)
    else:
        new_solutions_m = np.empty((0, candidate_solutions.shape[1]), dtype=np.float64)
        new_objs_m = np.empty((0, 2), dtype=np.float64)

    if archive_solutions.size == 0:
        merged_solutions = new_solutions_m
        merged_objs = new_objs_m
    else:
        merged_solutions = np.vstack([archive_solutions, new_solutions_m])
        merged_objs = np.vstack([archive_objs, new_objs_m])

    if merged_objs.size == 0:
        return merged_solutions, merged_objs

    keep = _nondominated_filter(merged_objs)
    merged_solutions = merged_solutions[keep, :]
    merged_objs = merged_objs[keep, :]

    if merged_solutions.shape[0] > archive_size:
        cd = _crowding_distance(merged_objs)
        order = np.argsort(cd)[::-1][:archive_size]
        merged_solutions = merged_solutions[order, :]
        merged_objs = merged_objs[order, :]

    return merged_solutions, merged_objs


def run_mobka_stage3_levy_archive(
    settings: Dict[str, Any],
    rng: np.random.RandomState | None = None,
    return_trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, MOBKAStage3Trace]:
    """Stage3 = Stage2 + Levy flight + secondary external archive."""
    if rng is None:
        rng = np.random.RandomState(2)

    pop, settings2 = build_initial_particles_mobka(settings, rng)

    demand_points_info = settings2["_demand_points_info"]
    charge_points_info = settings2["_charge_points_info"]
    parameter = settings2["_parameter"]
    cost_fun = _cost_fun_factory(demand_points_info, charge_points_info, parameter)

    demand_points_num = int(demand_points_info.shape[0])

    maxgen = int(settings["maxgen"])
    n_pop = int(settings["popnum"])

    n_rep = 200
    n_grid = 6
    alpha = 0.1
    beta = 1.5
    gamma = 2
    p = 0.8
    pc = 0.8
    pm = 0.3
    plevy = 0.5

    vmin = np.float64(0.0)
    vmax = np.float64(demand_points_num)

    ext_archive_size = 300
    ext_archive_solutions = np.empty((0, demand_points_num), dtype=np.float64)
    ext_archive_objs = np.empty((0, 2), dtype=np.float64)

    pop = deter_dom(pop)
    rep = [pp for pp in pop if not pp.IsDominated]
    grid = cgrid(rep, n_grid, alpha)
    for i in range(len(rep)):
        rep[i] = fgrid(rep[i], grid)

    archive_size_hist: List[np.float64] = []
    nd_hist: List[np.float64] = []
    hv_hist: List[np.float64] = []
    ext_archive_size_hist: List[np.float64] = []

    for it in range(1, maxgen + 1):
        offspring: List[Particle] = []

        for i in range(n_pop):
            leader = select_leader(rep, beta, rng)

            r = np.float64(rng.rand())
            n_step = np.float64(0.05 * np.exp(-2 * (it / maxgen) ** 2))
            n_var = demand_points_num

            if p < r:
                pop[i].Position = pop[i].Position + n_step * (1 + np.sin(r)) * (leader.Position - pop[i].Position)
            else:
                pop[i].Position = pop[i].Position + n_step * (2 * rng.rand(n_var) - 1) * pop[i].Position

            m = np.float64(2 * np.sin(r + np.pi / 2))
            s_1b = int(rng.randint(1, n_pop + 1))
            ori_value = rng.rand(n_var)
            cauchy_value = np.tan((ori_value - 0.5) * np.pi)

            if dominates_particle(pop[i], pop[s_1b - 1]):
                pop[i].Position = pop[i].Position + cauchy_value * (pop[i].BestPosition - leader.Position)
            else:
                pop[i].Position = pop[i].Position + cauchy_value * (leader.Position - m * pop[i].BestPosition)

            pop[i].Position = np.maximum(pop[i].Position, vmin)
            pop[i].Position = np.minimum(pop[i].Position, vmax)
            pop[i].Position = np.round(pop[i].Position)
            pop[i].Cost = cost_fun(pop[i].Position)

            if dominates_particle(pop[i], Particle(pop[i].BestPosition, pop[i].BestCost, pop[i].BestPosition, pop[i].BestCost)):
                pop[i].BestPosition = pop[i].Position.copy()
                pop[i].BestCost = pop[i].Cost.copy()
            elif not dominates_particle(Particle(pop[i].BestPosition, pop[i].BestCost, pop[i].BestPosition, pop[i].BestCost), pop[i]):
                if rng.rand() < 0.5:
                    pop[i].BestPosition = pop[i].Position.copy()
                    pop[i].BestCost = pop[i].Cost.copy()

            if rng.rand() < pm:
                child_pos = pop[i].Position.copy()
                sigma = np.float64(0.1 * (vmax - vmin))
                child_pos = child_pos + sigma * rng.randn(n_var)
                child_pos = np.maximum(child_pos, vmin)
                child_pos = np.minimum(child_pos, vmax)
                child_pos = np.round(child_pos)
                child_cost = cost_fun(child_pos)
                offspring.append(Particle(child_pos, child_cost, child_pos.copy(), child_cost.copy()))

            if rng.rand() < pc:
                mate_idx_1b = int(rng.randint(1, n_pop + 1))
                mate = pop[mate_idx_1b - 1]
                alpha_c = rng.rand(n_var)
                child_pos = alpha_c * pop[i].Position + (1 - alpha_c) * mate.Position
                child_pos = np.maximum(child_pos, vmin)
                child_pos = np.minimum(child_pos, vmax)
                child_pos = np.round(child_pos)
                child_cost = cost_fun(child_pos)
                offspring.append(Particle(child_pos, child_cost, child_pos.copy(), child_cost.copy()))

            # Levy flight improvement
            if rng.rand() < plevy:
                child_pos = pop[i].Position.copy()
                lvec = levy(n_var, rng)
                step = np.float64(0.01) * lvec
                child_pos = child_pos + step
                child_pos = np.maximum(child_pos, vmin)
                child_pos = np.minimum(child_pos, vmax)
                child_pos = np.round(child_pos)
                child_cost = cost_fun(child_pos)
                offspring.append(Particle(child_pos, child_cost, child_pos.copy(), child_cost.copy()))

        all_pop = pop + offspring
        all_pop = deter_dom(all_pop)

        rep = rep + [pp for pp in all_pop if not pp.IsDominated]
        rep = determine_domination(rep)
        rep = [pp for pp in rep if not pp.IsDominated]

        grid = cgrid(rep, n_grid, alpha)
        for i in range(len(rep)):
            rep[i] = fgrid(rep[i], grid)

        if len(rep) > n_rep:
            extra = len(rep) - n_rep
            for _ in range(extra):
                rep = delete_one_rep_member(rep, gamma, rng)

        nd_pop = [pp for pp in all_pop if not pp.IsDominated]
        if len(nd_pop) >= n_pop:
            idx = rng.permutation(len(nd_pop))[:n_pop]
            pop = [nd_pop[int(i)] for i in idx]
        else:
            pop = nd_pop
            dom_pop = [pp for pp in all_pop if pp.IsDominated]
            need = n_pop - len(pop)
            if need > 0 and len(dom_pop) > 0:
                idx2 = rng.permutation(len(dom_pop))[: min(need, len(dom_pop))]
                pop = pop + [dom_pop[int(i)] for i in idx2]

        all_positions = np.vstack([pp.Position.reshape(1, -1) for pp in all_pop])
        ext_archive_solutions, ext_archive_objs = _update_external_archive(
            ext_archive_solutions,
            ext_archive_objs,
            all_positions,
            cost_fun,
            ext_archive_size,
        )

        rep_obj = rep_to_obj_matrix(rep)
        archive_size_hist.append(np.float64(len(rep)))
        nd_hist.append(np.float64(rep_obj.shape[0]))
        hv_hist.append(hv_cir_like(rep_obj))
        ext_archive_size_hist.append(np.float64(ext_archive_solutions.shape[0]))

    final_obj = rep_to_obj_matrix(rep)

    if return_trace:
        trace = MOBKAStage3Trace(
            archive_size_history=np.asarray(archive_size_hist, dtype=np.float64),
            nd_count_history=np.asarray(nd_hist, dtype=np.float64),
            hv_history=np.asarray(hv_hist, dtype=np.float64),
            ext_archive_size_history=np.asarray(ext_archive_size_hist, dtype=np.float64),
        )
        return final_obj, trace

    return final_obj
