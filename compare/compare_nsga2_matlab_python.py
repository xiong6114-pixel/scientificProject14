from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nsga2_matlab_aligned import NSGA2_funciton, build_initial_population_for_nsga2


def _load_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    return np.asarray(data, dtype=np.float64)


def _first_mismatch(a: np.ndarray, b: np.ndarray, atol: float, rtol: float):
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


def _compare(name: str, py: np.ndarray, mat: np.ndarray, atol: float, rtol: float) -> bool:
    mm = _first_mismatch(py, mat, atol, rtol)
    if mm is None:
        print(f"[OK] {name}: shape={py.shape}, atol={atol}, rtol={rtol}")
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
    parser = argparse.ArgumentParser(description="Compare MATLAB vs Python NSGA-II small experiment.")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "baseline_matlab",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()

    bdir = args.baseline_dir

    required = [
        "nsga2_small_seed.csv",
        "nsga2_small_maxgen.csv",
        "nsga2_small_popnum.csv",
        "mat_nsga2_small_final_obj.csv",
        "mat_nsga2_small_front_sorted.csv",
        "mat_nsga2_small_generation_metrics.csv",
    ]
    missing = [x for x in required if not (bdir / x).exists()]
    if missing:
        print("Missing MATLAB NSGA-II baseline files:")
        for x in missing:
            print(f"  - {bdir / x}")
        print("Run matlab/export_nsga2_baseline_small.m first.")
        return 2

    seed = int(_load_csv(bdir / "nsga2_small_seed.csv").reshape(-1)[0])
    maxgen = int(_load_csv(bdir / "nsga2_small_maxgen.csv").reshape(-1)[0])
    popnum = int(_load_csv(bdir / "nsga2_small_popnum.csv").reshape(-1)[0])

    settings = {
        "data1": str(args.project_root / "demand_points_info_10.csv"),
        "data2": str(args.project_root / "charge_points_info_10.csv"),
        "maxgen": maxgen,
        "popnum": popnum,
    }

    rng = np.random.RandomState(seed)
    settings = build_initial_population_for_nsga2(settings=settings, rng=rng)
    py_final_obj, py_trace = NSGA2_funciton(settings=settings, rng=rng, return_trace=True)

    if py_final_obj.shape[0] > 0:
        py_front_sorted = py_final_obj[np.argsort(py_final_obj[:, 0]), :]
    else:
        py_front_sorted = py_final_obj.copy()

    mat_final_obj = _load_csv(bdir / "mat_nsga2_small_final_obj.csv")
    if mat_final_obj.ndim == 1:
        mat_final_obj = mat_final_obj.reshape(1, -1)

    mat_front_sorted = _load_csv(bdir / "mat_nsga2_small_front_sorted.csv")
    if mat_front_sorted.ndim == 1:
        mat_front_sorted = mat_front_sorted.reshape(1, -1)

    mat_metrics = _load_csv(bdir / "mat_nsga2_small_generation_metrics.csv")
    if mat_metrics.ndim == 1:
        mat_metrics = mat_metrics.reshape(1, -1)

    ok = True

    # 1) Non-dominated count
    py_nd_count = np.array([float(py_final_obj.shape[0])], dtype=np.float64)
    mat_nd_count = np.array([float(mat_final_obj.shape[0])], dtype=np.float64)
    ok &= _compare("non_dominated_count", py_nd_count, mat_nd_count, atol=0.0, rtol=0.0)

    # 2) Front distribution (sorted by obj1)
    ok &= _compare("front_sorted", py_front_sorted, mat_front_sorted, atol=1e-8, rtol=1e-8)

    # 3) Generation best/mean trends
    ok &= _compare("generation_metrics", py_trace.generation_metrics, mat_metrics, atol=1e-8, rtol=1e-8)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
