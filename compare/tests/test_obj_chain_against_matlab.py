from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cal_obj_1_debug import cal_obj_1_with_debug
from get_parameter import get_parameter
from waiting_time import compute_p0_and_wj


BASELINE_DIR = Path(__file__).resolve().parents[1] / "baseline_matlab"


def _load(name: str) -> np.ndarray:
    p = BASELINE_DIR / name
    if not p.exists():
        pytest.skip(f"Missing MATLAB baseline: {p}. Run matlab/export_cal_obj_chain_debug.m")
    data = np.loadtxt(p, delimiter=",")
    return np.asarray(data, dtype=np.float64)


def test_get_parameter_value_and_dtype() -> None:
    p = get_parameter()
    assert p.dtype == np.float64
    assert p.shape == (5,)
    assert np.max(np.abs(p - np.array([0.0, 20.0, 40.0, 12000.0, 5.0], dtype=np.float64))) <= 1e-12


def test_waiting_time_against_matlab_baseline() -> None:
    beta_j = _load("mat_beta_j.csv").reshape(-1)
    p0 = _load("mat_P0.csv").reshape(-1)
    wj = _load("mat_W_j.csv").reshape(-1)
    sol = _load("input_sol.csv").reshape(-1)
    parameter = _load("input_parameter.csv").reshape(-1)

    mu = parameter[1]
    for j in range(sol.size):
        if sol[j] > 0 and beta_j[j] > 0 and np.isfinite(p0[j]) and np.isfinite(wj[j]) and p0[j] > 0:
            py_p0, py_wj = compute_p0_and_wj(beta_j=beta_j[j], n_j=int(sol[j]), mu=mu)
            abs_err_p0 = np.abs(py_p0 - p0[j])
            rel_err_p0 = abs_err_p0 / (np.abs(p0[j]) + 1e-30)
            abs_err_wj = np.abs(py_wj - wj[j])
            rel_err_wj = abs_err_wj / (np.abs(wj[j]) + 1e-30)
            assert abs_err_p0 <= 1e-10
            assert rel_err_p0 <= 1e-10
            assert abs_err_wj <= 1e-10
            assert rel_err_wj <= 1e-10


def test_obj_chain_against_matlab_baseline() -> None:
    sol = _load("input_sol.csv").reshape(-1)
    demand = _load("input_demand_points_info.csv")
    charge = _load("input_charge_points_info.csv")
    parameter = _load("input_parameter.csv").reshape(-1)

    mat_assignment = _load("mat_assignment_matrix.csv")
    mat_dispatch = _load("mat_dispatch_1based.csv").reshape(-1)
    mat_ej = _load("mat_E_j.csv").reshape(-1)
    mat_beta = _load("mat_beta_j.csv").reshape(-1)
    mat_p0 = _load("mat_P0.csv").reshape(-1)
    mat_wj = _load("mat_W_j.csv").reshape(-1)
    mat_obj = _load("mat_obj.csv")
    if mat_obj.ndim == 1:
        mat_obj = mat_obj.reshape(-1, 1)

    py_obj, py_debug = cal_obj_1_with_debug(sol, demand, charge, parameter)

    assert np.max(np.abs(py_debug["assignment_matrix"] - mat_assignment)) <= 0.0
    assert np.max(np.abs(py_debug["dispatch_1based"].astype(np.float64) - mat_dispatch)) <= 0.0
    assert np.max(np.abs(py_debug["E_j"] - mat_ej)) <= 1e-12
    assert np.max(np.abs(py_debug["beta_j"] - mat_beta)) <= 1e-12
    assert np.max(np.abs(py_debug["P0"] - mat_p0)) <= 1e-10
    assert np.max(np.abs(py_debug["W_j"] - mat_wj)) <= 1e-10
    assert np.max(np.abs(py_obj - mat_obj)) <= 1e-10
