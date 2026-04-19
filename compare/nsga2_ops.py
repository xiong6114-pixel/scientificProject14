from __future__ import annotations

import numpy as np


def variate(sol: np.ndarray, charge_points_num: int, all_demand: int, rng: np.random.RandomState) -> np.ndarray:
    """MATLAB NSGAII/variate.m equivalent."""
    new_sol = np.asarray(sol, dtype=np.float64).copy()

    # MATLAB: rand_charge_point = randi([1, charge_points_num]); (1-based)
    rand_charge_point_1b = int(rng.randint(1, charge_points_num + 1))
    new_sol[rand_charge_point_1b - 1] = np.float64(rng.randint(0, all_demand + 1))

    return new_sol


def cross(sol: np.ndarray, charge_points_num: int, parent: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """MATLAB NSGAII/cross.m equivalent."""
    new_sol = np.asarray(sol, dtype=np.float64).copy()
    parent = np.asarray(parent, dtype=np.float64).reshape(-1)

    rand_charge_point_1b = int(rng.randint(1, charge_points_num + 1))
    new_sol[rand_charge_point_1b - 1] = parent[rand_charge_point_1b - 1]

    return new_sol
