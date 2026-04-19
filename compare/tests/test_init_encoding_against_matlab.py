from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cal_obj_1_debug import cal_obj_1_with_debug
from get_parameter import get_parameter
from init_encoding import (
    build_key_field_padded_matrices,
    init_population_solutions,
    stack_population_to_matrix,
)


BASELINE_DIR = Path(__file__).resolve().parents[1] / "baseline_matlab"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_csv(name: str) -> np.ndarray:
    p = BASELINE_DIR / name
    if not p.exists():
        pytest.skip(f"Missing MATLAB baseline: {p}. Run matlab/export_init_encoding_baseline.m")
    return np.asarray(np.loadtxt(p, delimiter=","), dtype=np.float64)


def test_init_population_shape_range() -> None:
    sols = init_population_solutions(charge_points_num=12, demand_points_num=7, n_individuals=5, seed=2)
    mat = stack_population_to_matrix(sols)

    assert mat.shape == (5, 12)
    assert mat.dtype == np.float64

    # Value range: 0 or integer in [1, demand_points_num+50]
    assert np.all((mat == 0) | ((mat >= 1) & (mat <= 57)))


def test_init_population_against_matlab_baseline() -> None:
    seed = int(_load_csv("init_seed.csv").reshape(-1)[0])
    charge_points_num = int(_load_csv("init_charge_points_num.csv").reshape(-1)[0])
    demand_points_num = int(_load_csv("init_demand_points_num.csv").reshape(-1)[0])

    mat_solutions = _load_csv("mat_init_solutions.csv")
    if mat_solutions.ndim == 1:
        mat_solutions = mat_solutions.reshape(1, -1)

    py_sols = init_population_solutions(
        charge_points_num=charge_points_num,
        demand_points_num=demand_points_num,
        n_individuals=mat_solutions.shape[0],
        seed=seed,
    )
    py_solutions = stack_population_to_matrix(py_sols)

    py_active_count, py_idx_pad, py_val_pad = build_key_field_padded_matrices(py_sols)
    mat_active_count = _load_csv("mat_init_active_count.csv").reshape(-1)
    mat_idx_pad = _load_csv("mat_init_active_idx_pad.csv")
    mat_val_pad = _load_csv("mat_init_active_val_pad.csv")

    assert np.max(np.abs(py_solutions - mat_solutions)) <= 0.0
    assert np.max(np.abs(py_active_count - mat_active_count)) <= 0.0
    assert np.max(np.abs(py_idx_pad - mat_idx_pad)) <= 0.0

    both_nan = np.isnan(py_val_pad) & np.isnan(mat_val_pad)
    diff = np.abs(np.where(both_nan, 0.0, py_val_pad - mat_val_pad))
    assert np.max(diff) <= 0.0


def test_init_individuals_are_readable_by_cal_obj_chain() -> None:
    # Requirement: each initialized individual should be consumable by cal_obj chain.
    demand = np.asarray(np.loadtxt(PROJECT_ROOT / "demand_points_info.csv", delimiter=","), dtype=np.float64)
    charge = np.asarray(np.loadtxt(PROJECT_ROOT / "charge_points_info.csv", delimiter=","), dtype=np.float64)
    parameter = get_parameter()

    sols = init_population_solutions(
        charge_points_num=charge.shape[0],
        demand_points_num=demand.shape[0],
        n_individuals=5,
        seed=2,
    )

    for sol in sols:
        obj, dbg = cal_obj_1_with_debug(sol=sol, demand_points_info=demand, charge_points_info=charge, parameter=parameter)
        assert obj.shape == (2, 1)
        assert isinstance(dbg, dict)
        assert dbg["assignment_matrix"].shape == (demand.shape[0], charge.shape[0])
