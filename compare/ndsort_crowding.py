from __future__ import annotations

from typing import List, Tuple

import numpy as np


def dominates(indiv1: np.ndarray, indiv2: np.ndarray) -> bool:
    """MATLAB fast_non_dominated_sort_with_crowding.m dominates() equivalent."""
    return bool(np.all(indiv1 <= indiv2) and np.any(indiv1 < indiv2))


def calculate_crowding_distance(front: np.ndarray, m: int) -> np.ndarray:
    """MATLAB calculate_crowding_distance() equivalent."""
    n = front.shape[0]
    distance = np.zeros(n, dtype=np.float64)

    if n == 0:
        return distance
    if n == 1:
        distance[0] = np.inf
        return distance

    for mm in range(m):
        idx = np.argsort(front[:, mm])
        distance[idx[0]] = np.inf
        distance[idx[-1]] = np.inf

        denom = np.max(front[:, mm]) - np.min(front[:, mm]) + np.finfo(np.float64).eps
        for j in range(1, n - 1):
            distance[idx[j]] = np.float64(
                distance[idx[j]] + (front[idx[j + 1], mm] - front[idx[j - 1], mm]) / denom
            )

    return distance


def fast_non_dominated_sort_with_crowding(population: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """MATLAB NSGAII/fast_non_dominated_sort_with_crowding.m equivalent.

    Index note:
    - MATLAB returns 1-based front indices.
    - Python here returns 0-based indices.
    """
    population = np.asarray(population, dtype=np.float64)

    n = population.shape[0]
    m = population.shape[1]

    s: List[List[int]] = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int64)
    crowding_distance = np.zeros(n, dtype=np.float64)
    fronts: List[List[int]] = [[]]

    for i in range(n):
        s[i] = []
        n_dom[i] = 0
        for j in range(n):
            if dominates(population[i, :], population[j, :]):
                s[i].append(j)
            elif dominates(population[j, :], population[i, :]):
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
        crowding_distance[front] = calculate_crowding_distance(population[front, :], m)

    return out_fronts, crowding_distance
