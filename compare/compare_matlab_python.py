from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from cal_obj_1_debug import cal_obj_1_with_debug, save_debug_to_csv


def _load_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    return np.asarray(data, dtype=np.float64)


def _load_csv_or_mat(bdir: Path, csv_name: str, mat_var_name: str) -> np.ndarray:
    csv_path = bdir / csv_name
    if csv_path.exists():
        return _load_csv(csv_path)

    mat_path = bdir / "mat_debug_dump.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing both {csv_path} and {mat_path}")

    try:
        from scipy.io import loadmat
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "scipy is required to read .mat baseline when csv is absent."
        ) from exc

    data = loadmat(mat_path)
    if mat_var_name not in data:
        raise KeyError(f"Variable '{mat_var_name}' not found in {mat_path}")
    return np.asarray(data[mat_var_name], dtype=np.float64)


def _first_mismatch(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> tuple[tuple[int, ...], float, float, float] | None:
    if a.shape != b.shape:
        return (), np.nan, np.nan, np.inf

    for idx in np.ndindex(a.shape):
        av = float(a[idx])
        bv = float(b[idx])
        abs_err = abs(av - bv)
        tol = atol + rtol * abs(bv)
        if not (abs_err <= tol):
            return idx, av, bv, abs_err
    return None


def _compare_one(name: str, py: np.ndarray, matlab: np.ndarray, atol: float, rtol: float) -> bool:
    mismatch = _first_mismatch(py, matlab, atol=atol, rtol=rtol)
    if mismatch is None:
        print(f"[OK] {name}: shape={py.shape}, atol={atol}, rtol={rtol}")
        return True

    idx, pyv, mv, abs_err = mismatch
    if idx == ():
        print(f"[FAIL] {name}: shape mismatch, py={py.shape}, matlab={matlab.shape}")
    else:
        idx_1b = tuple(i + 1 for i in idx)
        print(
            f"[FAIL] {name}: first mismatch at idx0={idx}, idx1={idx_1b}, "
            f"py={pyv:.17g}, matlab={mv:.17g}, abs_err={abs_err:.3e}, "
            f"tol={atol + rtol * abs(mv):.3e}"
        )
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare MATLAB and Python cal_obj_1 debug chain outputs.")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "baseline_matlab",
        help="Directory containing MATLAB exported input/output csv files.",
    )
    parser.add_argument(
        "--dump-python",
        action="store_true",
        help="Also dump Python intermediate outputs to baseline dir as py_*.csv",
    )
    args = parser.parse_args()

    bdir = args.baseline_dir

    required = [
        "input_sol.csv",
        "input_demand_points_info.csv",
        "input_charge_points_info.csv",
        "input_parameter.csv",
    ]
    missing = [x for x in required if not (bdir / x).exists()]
    if missing:
        print("Missing MATLAB baseline files:")
        for x in missing:
            print(f"  - {bdir / x}")
        print("Run matlab/export_cal_obj_chain_debug.m first.")
        return 2

    sol = _load_csv(bdir / "input_sol.csv").reshape(-1)
    demand = _load_csv(bdir / "input_demand_points_info.csv")
    charge = _load_csv(bdir / "input_charge_points_info.csv")
    parameter = _load_csv(bdir / "input_parameter.csv").reshape(-1)

    py_obj, py_debug = cal_obj_1_with_debug(sol, demand, charge, parameter)

    if args.dump_python:
        save_debug_to_csv(py_debug, py_obj, str(bdir))

    mat_assignment = _load_csv_or_mat(bdir, "mat_assignment_matrix.csv", "assignment_matrix")
    mat_dispatch = _load_csv_or_mat(bdir, "mat_dispatch_1based.csv", "dispatch").reshape(-1)
    mat_ej = _load_csv_or_mat(bdir, "mat_E_j.csv", "E_j").reshape(-1)
    mat_beta = _load_csv_or_mat(bdir, "mat_beta_j.csv", "beta_j").reshape(-1)
    mat_p0 = _load_csv_or_mat(bdir, "mat_P0.csv", "P0").reshape(-1)
    mat_wj = _load_csv_or_mat(bdir, "mat_W_j.csv", "w_j_list").reshape(-1)
    mat_obj = _load_csv_or_mat(bdir, "mat_obj.csv", "f")
    if mat_obj.ndim == 1:
        mat_obj = mat_obj.reshape(-1, 1)

    checks: Iterable[tuple[str, np.ndarray, np.ndarray, float, float]] = [
        ("assignment_matrix", py_debug["assignment_matrix"], mat_assignment, 0.0, 0.0),
        ("dispatch_1based", py_debug["dispatch_1based"].astype(np.float64), mat_dispatch, 0.0, 0.0),
        ("E_j", py_debug["E_j"], mat_ej, 1e-12, 1e-12),
        ("beta_j", py_debug["beta_j"], mat_beta, 1e-12, 1e-12),
        ("P0", py_debug["P0"], mat_p0, 1e-10, 1e-10),
        ("W_j", py_debug["W_j"], mat_wj, 1e-10, 1e-10),
        ("obj", py_obj, mat_obj, 1e-10, 1e-10),
    ]

    all_ok = True
    for name, py, mat, atol, rtol in checks:
        ok = _compare_one(name, np.asarray(py, dtype=np.float64), np.asarray(mat, dtype=np.float64), atol, rtol)
        all_ok = all_ok and ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
