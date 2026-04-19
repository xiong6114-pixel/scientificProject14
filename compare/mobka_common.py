from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Particle:
    Position: np.ndarray
    Cost: np.ndarray
    BestPosition: np.ndarray
    BestCost: np.ndarray
    IsDominated: bool = False
    GridIndex: int = 0
    GridSubIndex: np.ndarray | None = None


def dominates_cost(x: np.ndarray, y: np.ndarray) -> bool:
    return bool(np.all(x <= y) and np.any(x < y))


def dominates_particle(x: Particle, y: Particle) -> bool:
    return dominates_cost(x.Cost, y.Cost)


def deter_dom(pop: List[Particle]) -> List[Particle]:
    n_pop = len(pop)
    for i in range(n_pop):
        pop[i].IsDominated = False

    for i in range(n_pop - 1):
        for j in range(i + 1, n_pop):
            if dominates_particle(pop[i], pop[j]):
                pop[j].IsDominated = True
            if dominates_particle(pop[j], pop[i]):
                pop[i].IsDominated = True
    return pop


def determine_domination(pop: List[Particle]) -> List[Particle]:
    return deter_dom(pop)


def cgrid(rep: List[Particle], n_grid: int, alpha: float) -> list[dict[str, np.ndarray]]:
    c = np.column_stack([p.Cost for p in rep]).astype(np.float64)

    cmin = np.min(c, axis=1)
    cmax = np.max(c, axis=1)

    # Keep MATLAB-like warning-off behavior for inf/nan arithmetic.
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        dc = cmax - cmin
    cmin = cmin - alpha * dc
    cmax = cmax + alpha * dc

    n_obj = c.shape[0]
    grid: list[dict[str, np.ndarray]] = []

    for j in range(n_obj):
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            cj = np.linspace(cmin[j], cmax[j], n_grid + 1, dtype=np.float64)
        lb = np.concatenate([np.array([-np.inf], dtype=np.float64), cj])
        ub = np.concatenate([cj, np.array([np.inf], dtype=np.float64)])
        grid.append({"LB": lb, "UB": ub})

    return grid


def fgrid(particle: Particle, grid: list[dict[str, np.ndarray]]) -> Particle:
    n_obj = particle.Cost.size
    n_grid = grid[0]["LB"].size

    gsi = np.zeros(n_obj, dtype=np.int64)
    for j in range(n_obj):
        ub = grid[j]["UB"]
        # MATLAB find(...,1,'first') returns 1-based
        matches = np.where(particle.Cost[j] < ub)[0]
        if matches.size == 0:
            # MATLAB empty-find assignment tolerance: keep as 0 instead of raising.
            gsi[j] = 0
        else:
            idx0 = int(matches[0])
            gsi[j] = idx0 + 1

    gi = int(gsi[0])
    for j in range(1, n_obj):
        gi = gi - 1
        gi = n_grid * gi
        gi = gi + int(gsi[j])

    particle.GridSubIndex = gsi
    particle.GridIndex = gi
    return particle


def roulette_wheel_selection(p: np.ndarray, rng: np.random.RandomState) -> int:
    r = float(rng.rand())
    c = np.cumsum(p)
    idx0 = int(np.where(r <= c)[0][0])
    # MATLAB output is 1-based index.
    return idx0 + 1


def select_leader(rep: List[Particle], beta: float, rng: np.random.RandomState) -> Particle:
    gi = np.array([p.GridIndex for p in rep], dtype=np.int64)
    oc = np.unique(gi)

    n = np.zeros(oc.size, dtype=np.float64)
    for k in range(oc.size):
        n[k] = np.float64(np.sum(gi == oc[k]))

    p = np.exp(-beta * n)
    p = p / np.sum(p)

    sci_1b = roulette_wheel_selection(p, rng)
    sc = int(oc[sci_1b - 1])

    scm_0b = np.where(gi == sc)[0]
    smi_1b = int(rng.randint(1, scm_0b.size + 1))
    sm_0b = int(scm_0b[smi_1b - 1])

    return rep[sm_0b]


def delete_one_rep_member(rep: List[Particle], gamma: float, rng: np.random.RandomState) -> List[Particle]:
    gi = np.array([p.GridIndex for p in rep], dtype=np.int64)
    oc = np.unique(gi)

    n = np.zeros(oc.size, dtype=np.float64)
    for k in range(oc.size):
        n[k] = np.float64(np.sum(gi == oc[k]))

    p = np.exp(gamma * n)
    p = p / np.sum(p)

    sci_1b = roulette_wheel_selection(p, rng)
    sc = int(oc[sci_1b - 1])

    scm_0b = np.where(gi == sc)[0]
    smi_1b = int(rng.randint(1, scm_0b.size + 1))
    sm_0b = int(scm_0b[smi_1b - 1])

    out = rep.copy()
    del out[sm_0b]
    return out


def rep_to_obj_matrix(rep: List[Particle]) -> np.ndarray:
    if len(rep) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack([p.Cost.reshape(1, -1) for p in rep]).astype(np.float64)


def clone_particle(p: Particle) -> Particle:
    return Particle(
        Position=p.Position.copy(),
        Cost=p.Cost.copy(),
        BestPosition=p.BestPosition.copy(),
        BestCost=p.BestCost.copy(),
        IsDominated=bool(p.IsDominated),
        GridIndex=int(p.GridIndex),
        GridSubIndex=(None if p.GridSubIndex is None else p.GridSubIndex.copy()),
    )
