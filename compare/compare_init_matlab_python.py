from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from init_encoding import (
    build_key_field_padded_matrices,
    init_population_solutions,
    stack_population_to_matrix,
)


def _load_csv(path: Path) -> np.ndarray:
    return np.asarray(np.loadtxt(path, delimiter=","), dtype=np.float64)


def _first_mismatch(a: np.ndarray, b: np.ndarray, atol: float, rtol: float):
    if a.shape != b.shape:
        return (), np.nan, np.nan, np.inf
    for idx in np.ndindex(a.shape):
        av = float(a[idx])
        bv = float(b[idx])
        if np.isnan(av) and np.isnan(bv):
            continue
        abs_err = abs(av - bv)
        tol = atol + rtol * abs(bv)
        if not (abs_err <= tol):
            return idx, av, bv, abs_err
    return None


def _compare(name: str, py: np.ndarray, mat: np.ndarray, atol: float = 0.0, rtol: float = 0.0) -> bool:
    mm = _first_mismatch(py, mat, atol, rtol)
    if mm is None:
        print(f"[OK] {name}: shape={py.shape}")
        return True

    idx, pyv, mv, abs_err = mm
    if idx == ():
        print(f"[FAIL] {name}: shape mismatch py={py.shape}, matlab={mat.shape}")
    else:
        idx_1b = tuple(i + 1 for i in idx)
        print(
            f"[FAIL] {name}: first mismatch idx0={idx}, idx1={idx_1b}, "
            f"py={pyv:.17g}, matlab={mv:.17g}, abs_err={abs_err:.3e}, tol={atol + rtol * abs(mv):.3e}"
        )
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare MATLAB vs Python init/encoding outputs.")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "baseline_matlab",
    )
    args = parser.parse_args()

    bdir = args.baseline_dir
    required = [
        "init_seed.csv",
        "init_charge_points_num.csv",
        "init_demand_points_num.csv",
        "mat_init_solutions.csv",
        "mat_init_active_count.csv",
        "mat_init_active_idx_pad.csv",
        "mat_init_active_val_pad.csv",
    ]
    missing = [x for x in required if not (bdir / x).exists()]
    if missing:
        print("Missing init baseline files:")
        for x in missing:
            print(f"  - {bdir / x}")
        print("Run matlab/export_init_encoding_baseline.m first.")
        return 2

    seed = int(_load_csv(bdir / "init_seed.csv").reshape(-1)[0])
    charge_points_num = int(_load_csv(bdir / "init_charge_points_num.csv").reshape(-1)[0])
    demand_points_num = int(_load_csv(bdir / "init_demand_points_num.csv").reshape(-1)[0])

    mat_solutions = _load_csv(bdir / "mat_init_solutions.csv")
    if mat_solutions.ndim == 1:
        mat_solutions = mat_solutions.reshape(1, -1)

    n_individuals = int(mat_solutions.shape[0])

    py_sols = init_population_solutions(
        charge_points_num=charge_points_num,
        demand_points_num=demand_points_num,
        n_individuals=n_individuals,
        seed=seed,
    )
    py_solutions = stack_population_to_matrix(py_sols)
    py_active_count, py_idx_pad, py_val_pad = build_key_field_padded_matrices(py_sols)

    mat_active_count = _load_csv(bdir / "mat_init_active_count.csv").reshape(-1)
    mat_idx_pad = _load_csv(bdir / "mat_init_active_idx_pad.csv")
    mat_val_pad = _load_csv(bdir / "mat_init_active_val_pad.csv")

    ok = True
    ok &= _compare("init_solutions", py_solutions, mat_solutions, atol=0.0, rtol=0.0)
    ok &= _compare("active_count", py_active_count, mat_active_count, atol=0.0, rtol=0.0)
    ok &= _compare("active_idx_pad", py_idx_pad, mat_idx_pad, atol=0.0, rtol=0.0)
    ok &= _compare("active_val_pad", py_val_pad, mat_val_pad, atol=0.0, rtol=0.0)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
