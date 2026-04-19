from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class IndividualKeyFields:
    """Key fields for one individual (decoded from sol vector)."""

    active_count: int
    active_indices_1based: np.ndarray
    active_values: np.ndarray


def init_sol(charge_points_num: int, demand_points_num: int, rng: np.random.RandomState) -> np.ndarray:
    """Equivalent to MATLAB init_sol(charge_points_num, demand_points_num).

    MATLAB logic:
    1) binary_vars = rand(1, charge_points_num) < 0.3
    2) initial_sol = zeros(1, charge_points_num)
    3) for i=1:charge_points_num, if binary_vars(i)==1:
           initial_sol(i) = randi([1, demand_points_num+50])

    Returns
    - shape: (charge_points_num,), dtype float64
    - value range per element: 0 or integer in [1, demand_points_num+50]
    """
    p = np.float64(0.3)

    # Keep random call order aligned with MATLAB: first one rand vector, then per-active randi.
    binary_vars = (rng.rand(charge_points_num) < p)

    initial_sol = np.zeros(charge_points_num, dtype=np.float64)
    for i0 in range(charge_points_num):
        if bool(binary_vars[i0]):
            initial_sol[i0] = np.float64(rng.randint(1, demand_points_num + 50 + 1))

    return initial_sol


def decode_sol_vector(sol: np.ndarray) -> IndividualKeyFields:
    """Decode one sol vector using MATLAB semantics.

    Index mapping:
    - MATLAB active index i in [1..J]
    - Python index i0 = i-1 in [0..J-1]
    """
    sol = np.asarray(sol, dtype=np.float64).reshape(-1)

    active_i0 = np.where(sol > 0)[0]
    active_indices_1based = (active_i0 + 1).astype(np.int64)
    active_values = sol[active_i0].astype(np.float64)

    return IndividualKeyFields(
        active_count=int(active_i0.size),
        active_indices_1based=active_indices_1based,
        active_values=active_values,
    )


def init_population_solutions(
    charge_points_num: int,
    demand_points_num: int,
    n_individuals: int,
    seed: int,
) -> List[np.ndarray]:
    """Conservative MATLAB-cell equivalent: list of 1-D sol vectors."""
    rng = np.random.RandomState(seed)
    sols: List[np.ndarray] = []
    for _ in range(n_individuals):
        sols.append(init_sol(charge_points_num, demand_points_num, rng))
    return sols


def stack_population_to_matrix(sols: List[np.ndarray]) -> np.ndarray:
    """Utility for baseline comparison/export only.

    MATLAB cell -> Python list[np.ndarray].
    This function converts list to a 2D matrix (n_individuals, charge_points_num)
    without changing encoding semantics.
    """
    if len(sols) == 0:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack([np.asarray(s, dtype=np.float64).reshape(1, -1) for s in sols])


def build_key_field_padded_matrices(sols: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build padded matrices for stable CSV-level MATLAB/Python comparison.

    Returns
    - active_count_vec: shape (n,)
    - active_idx_pad: shape (n, max_active), padded by -1
    - active_val_pad: shape (n, max_active), padded by NaN
    """
    decoded = [decode_sol_vector(s) for s in sols]
    n = len(decoded)
    max_active = max((d.active_count for d in decoded), default=0)

    active_count_vec = np.zeros(n, dtype=np.float64)
    active_idx_pad = np.full((n, max_active), -1.0, dtype=np.float64)
    active_val_pad = np.full((n, max_active), np.nan, dtype=np.float64)

    for i in range(n):
        d = decoded[i]
        active_count_vec[i] = np.float64(d.active_count)
        if d.active_count > 0:
            active_idx_pad[i, : d.active_count] = d.active_indices_1based.astype(np.float64)
            active_val_pad[i, : d.active_count] = d.active_values.astype(np.float64)

    return active_count_vec, active_idx_pad, active_val_pad
