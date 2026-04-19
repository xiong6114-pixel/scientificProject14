from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from cal_obj_1_debug import cal_obj_1_with_debug
from get_parameter import get_parameter
from init_encoding import init_sol
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


@dataclass
class MOBKATrace:
    archive_size_history: np.ndarray
    nd_count_history: np.ndarray
    hv_history: np.ndarray


def _cost_fun_factory(
    demand_points_info: np.ndarray,
    charge_points_info: np.ndarray,
    parameter: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    def _cost(x: np.ndarray) -> np.ndarray:
        f, _ = cal_obj_1_with_debug(x, demand_points_info, charge_points_info, parameter)
        return f.reshape(-1).astype(np.float64)

    return _cost


def build_initial_particles_mobka(
    settings: Dict[str, Any],
    rng: np.random.RandomState,
) -> Tuple[List[Particle], Dict[str, Any]]:
    demand_points_info = np.asarray(np.loadtxt(settings["data1"], delimiter=","), dtype=np.float64)
    charge_points_info = np.asarray(np.loadtxt(settings["data2"], delimiter=","), dtype=np.float64)

    demand_points_num = int(demand_points_info.shape[0])
    charge_points_num = int(charge_points_info.shape[0])

    parameter = get_parameter()
    cost_fun = _cost_fun_factory(demand_points_info, charge_points_info, parameter)

    sizepop = int(settings["popnum"])

    pop: List[Particle] = []

    if "sol_manager" in settings and "obj_manager" in settings:
        for i in range(sizepop):
            pos = np.asarray(settings["sol_manager"][i], dtype=np.float64).reshape(-1)
            cost = np.asarray(settings["obj_manager"][i], dtype=np.float64).reshape(-1)
            if cost.size == 2:
                costv = cost.copy()
            else:
                costv = cost_fun(pos)
            pop.append(
                Particle(
                    Position=pos.copy(),
                    Cost=costv.copy(),
                    BestPosition=pos.copy(),
                    BestCost=costv.copy(),
                )
            )
    else:
        for _ in range(sizepop):
            while True:
                pos = init_sol(charge_points_num, demand_points_num, rng)
                costv = cost_fun(pos)
                if np.all(np.isfinite(costv)):
                    break
            pop.append(
                Particle(
                    Position=pos.copy(),
                    Cost=costv.copy(),
                    BestPosition=pos.copy(),
                    BestCost=costv.copy(),
                )
            )

    out_settings = dict(settings)
    out_settings["_demand_points_info"] = demand_points_info
    out_settings["_charge_points_info"] = charge_points_info
    out_settings["_parameter"] = parameter
    return pop, out_settings


def run_mobka_stage1(
    settings: Dict[str, Any],
    rng: np.random.RandomState | None = None,
    return_trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, MOBKATrace]:
    """Pure MOBKA stage aligned with mobka/MOBKA_funciton.m core loop."""
    if rng is None:
        rng = np.random.RandomState(2)

    pop, settings2 = build_initial_particles_mobka(settings, rng)

    demand_points_info = settings2["_demand_points_info"]
    charge_points_info = settings2["_charge_points_info"]
    parameter = settings2["_parameter"]
    cost_fun = _cost_fun_factory(demand_points_info, charge_points_info, parameter)

    demand_points_num = int(demand_points_info.shape[0])

    maxgen = int(settings["maxgen"])
    sizepop = int(settings["popnum"])

    n_rep = 200
    n_grid = 6
    alpha = 0.1
    beta = 1.5
    gamma = 2
    p = 0.8

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

    # Random call order is kept explicit to match MATLAB sequence.
    for it in range(1, maxgen + 1):
        for i in range(sizepop):
            leader = select_leader(rep, beta, rng)

            r = np.float64(rng.rand())
            n = np.float64(0.05 * np.exp(-2 * (it / maxgen) ** 2))

            # nVar in MATLAB code equals demand_points_num.
            n_var = demand_points_num
            if p < r:
                pop[i].Position = pop[i].Position + n * (1 + np.sin(r)) * pop[i].Position
            else:
                pop[i].Position = pop[i].Position * (n * (2 * rng.rand(n_var) - 1) + 1)

            m = np.float64(2 * np.sin(r + np.pi / 2))
            s_1b = int(rng.randint(1, sizepop + 1))
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

        rep_obj = rep_to_obj_matrix(rep)
        archive_size_hist.append(np.float64(len(rep)))
        nd_hist.append(np.float64(rep_obj.shape[0]))
        hv_hist.append(hv_cir_like(rep_obj))

    final_obj = rep_to_obj_matrix(rep)

    if return_trace:
        trace = MOBKATrace(
            archive_size_history=np.asarray(archive_size_hist, dtype=np.float64),
            nd_count_history=np.asarray(nd_hist, dtype=np.float64),
            hv_history=np.asarray(hv_hist, dtype=np.float64),
        )
        return final_obj, trace

    return final_obj
