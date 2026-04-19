from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from imobka_matlab_aligned import IMOBKA_funciton
from moead_matlab_aligned import MOEAD_function
from nsga2_matlab_aligned import build_initial_population_for_nsga2
from nsga3_matlab_aligned import NSGA3_funciton


def _load(path: Path) -> np.ndarray:
    d = np.asarray(np.loadtxt(path, delimiter=","), dtype=np.float64)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    return d


def _compare(name: str, py: np.ndarray, mat: np.ndarray, atol: float = 1e-8) -> bool:
    if py.shape != mat.shape:
        print(f"[FAIL] {name}: shape py={py.shape} mat={mat.shape}")
        return False
    if py.size == 0:
        print(f"[OK] {name}: both empty")
        return True
    err = np.abs(py - mat)
    mx = float(np.max(err))
    if mx <= atol:
        print(f"[OK] {name}: max_abs_err={mx:.3e}")
        return True
    idx = np.unravel_index(int(np.argmax(err)), err.shape)
    idx1 = tuple(i + 1 for i in idx)
    print(f"[FAIL] {name}: idx0={idx}, idx1={idx1}, py={py[idx]:.17g}, mat={mat[idx]:.17g}, err={mx:.3e}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=Path(__file__).resolve().parent / "baseline_matlab")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    b = args.baseline_dir
    req = [
        "mat_moead_small_final_obj.csv",
        "mat_imobka_small_final_obj.csv",
        "mat_nsga3_small_final_obj.csv",
    ]
    miss = [x for x in req if not (b / x).exists()]
    if miss:
        print("Missing baseline files:")
        for x in miss:
            print(f"  - {b / x}")
        print("Run export_moead_baseline_small.m / export_imobka_baseline_small.m / export_nsga3_baseline_small.m first.")
        return 2

    settings = {
        "data1": str(args.project_root / "demand_points_info_10.csv"),
        "data2": str(args.project_root / "charge_points_info_10.csv"),
        "maxgen": 8,
        "popnum": 20,
    }

    st = build_initial_population_for_nsga2(settings, np.random.RandomState(2))
    py_moead = MOEAD_function(st, rng=np.random.RandomState(2), return_trace=False)
    py_imobka = IMOBKA_funciton(st, rng=np.random.RandomState(2), return_trace=False)
    py_nsga3 = NSGA3_funciton(settings, rng=np.random.RandomState(2), return_trace=False)

    mat_moead = _load(b / "mat_moead_small_final_obj.csv")
    mat_imobka = _load(b / "mat_imobka_small_final_obj.csv")
    mat_nsga3 = _load(b / "mat_nsga3_small_final_obj.csv")

    ok = True
    ok &= _compare("MOEAD final_obj", py_moead, mat_moead, atol=1e-8)
    ok &= _compare("IMOBKA final_obj", py_imobka, mat_imobka, atol=1e-8)
    ok &= _compare("NSGA3 final_obj", py_nsga3, mat_nsga3, atol=1e-8)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
