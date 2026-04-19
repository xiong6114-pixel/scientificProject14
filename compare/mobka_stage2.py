from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

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
from mobka_metrics import hv_cir_like
from mobka_stage1 import build_initial_particles_mobka, _cost_fun_factory


@dataclass
class MOBKAStage2Trace:
    archive_size_history: np.ndarray
    nd_count_history: np.ndarray
    hv_history: np.ndarray


def run_mobka_stage2_cross_mutation(
    settings: Dict[str, Any],
    rng: np.random.RandomState | None = None,
    return_trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, MOBKAStage2Trace]:
    """Stage2 = Pure MOBKA + GA crossover/mutation (IMOBKA incremental)."""
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

    vmin = np.float64(0.0)
    vmax = np.float64(demand_points_num)

    pop = deter_dom(pop)
    rep = [pp for pp in pop if not pp.IsDominated]
    grid = cgrid(rep, n_grid, alpha)
    for i in range(len(rep)):
        rep[i] = fgrid(rep[i], grid)

    archive_size_hist: List[np.float64] = []
    nd_hist: List[np.float64] = []
    hv_hist: List[np.float64] = []

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

            # Best update copied from IMOBKA_ZDT logic
            if dominates_particle(pop[i], Particle(pop[i].BestPosition, pop[i].BestCost, pop[i].BestPosition, pop[i].BestCost)):
                pop[i].BestPosition = pop[i].Position.copy()
                pop[i].BestCost = pop[i].Cost.copy()
            elif not dominates_particle(Particle(pop[i].BestPosition, pop[i].BestCost, pop[i].BestPosition, pop[i].BestCost), pop[i]):
                if rng.rand() < 0.5:
                    pop[i].BestPosition = pop[i].Position.copy()
                    pop[i].BestCost = pop[i].Cost.copy()

            # Mutation improvement
            if rng.rand() < pm:
                child_pos = pop[i].Position.copy()
                sigma = np.float64(0.1 * (vmax - vmin))
                child_pos = child_pos + sigma * rng.randn(n_var)
                child_pos = np.maximum(child_pos, vmin)
                child_pos = np.minimum(child_pos, vmax)
                child_pos = np.round(child_pos)
                child_cost = cost_fun(child_pos)
                offspring.append(
                    Particle(
                        Position=child_pos,
                        Cost=child_cost,
                        BestPosition=child_pos.copy(),
                        BestCost=child_cost.copy(),
                    )
                )

            # Crossover improvement
            if rng.rand() < pc:
                mate_idx_1b = int(rng.randint(1, n_pop + 1))
                mate = pop[mate_idx_1b - 1]
                alpha_c = rng.rand(n_var)
                child_pos = alpha_c * pop[i].Position + (1 - alpha_c) * mate.Position
                child_pos = np.maximum(child_pos, vmin)
                child_pos = np.minimum(child_pos, vmax)
                child_pos = np.round(child_pos)
                child_cost = cost_fun(child_pos)
                offspring.append(
                    Particle(
                        Position=child_pos,
                        Cost=child_cost,
                        BestPosition=child_pos.copy(),
                        BestCost=child_cost.copy(),
                    )
                )

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

        rep_obj = rep_to_obj_matrix(rep)
        archive_size_hist.append(np.float64(len(rep)))
        nd_hist.append(np.float64(rep_obj.shape[0]))
        hv_hist.append(hv_cir_like(rep_obj))

    final_obj = rep_to_obj_matrix(rep)

    if return_trace:
        trace = MOBKAStage2Trace(
            archive_size_history=np.asarray(archive_size_hist, dtype=np.float64),
            nd_count_history=np.asarray(nd_hist, dtype=np.float64),
            hv_history=np.asarray(hv_hist, dtype=np.float64),
        )
        return final_obj, trace

    return final_obj
