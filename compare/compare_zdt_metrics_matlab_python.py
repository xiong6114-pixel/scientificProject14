from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from zdt_metrics_plot import calc_coverage, calc_hv_2d, select_front_points, zdt_cost, zdt_true_pf


def _load(path: Path) -> np.ndarray:
    return np.asarray(np.loadtxt(path, delimiter=","), dtype=np.float64)


def _cmp(name: str, a: np.ndarray, b: np.ndarray, tol: float = 1e-12) -> bool:
    if a.shape != b.shape:
        print(f"[FAIL] {name}: shape mismatch py={a.shape} mat={b.shape}")
        return False
    err = np.max(np.abs(a - b)) if a.size > 0 else 0.0
    if err <= tol:
        print(f"[OK] {name}: max_abs_err={err:.3e}")
        return True
    idx = np.unravel_index(np.argmax(np.abs(a - b)), a.shape)
    print(f"[FAIL] {name}: idx={idx}, py={a[idx]:.17g}, mat={b[idx]:.17g}, err={err:.3e}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=Path(__file__).resolve().parent / "baseline_matlab")
    args = parser.parse_args()

    b = args.baseline_dir
    req = [
        "mat_zdt_f1.csv",
        "mat_zdt_f2.csv",
        "mat_zdt_f3.csv",
        "mat_zdt_f4.csv",
        "mat_zdt_pf1.csv",
        "mat_zdt_pf4.csv",
        "mat_zdt_hv_example.csv",
        "mat_zdt_cov_ab.csv",
        "mat_zdt_cov_ba.csv",
        "mat_zdt_select_front.csv",
    ]
    miss = [x for x in req if not (b / x).exists()]
    if miss:
        print("Missing baseline files. Run matlab/export_zdt_metrics_baseline.m first.")
        for x in miss:
            print(f"  - {b / x}")
        return 2

    x = np.zeros(30, dtype=np.float64)
    py_f1 = zdt_cost(x, 1)
    py_f2 = zdt_cost(x, 2)
    py_f3 = zdt_cost(x, 3)
    py_f4 = zdt_cost(np.zeros(10, dtype=np.float64), 4)

    py_pf1 = zdt_true_pf(1, 5)
    py_pf4 = zdt_true_pf(4, 5)

    py_hv = np.array([calc_hv_2d(np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float64), np.array([1.0, 1.0], dtype=np.float64))], dtype=np.float64)
    py_cab = np.array([calc_coverage(np.array([[1, 1], [2, 2]], dtype=np.float64), np.array([[1.5, 1.5], [0.5, 2.5]], dtype=np.float64))], dtype=np.float64)
    py_cba = np.array([calc_coverage(np.array([[1.5, 1.5], [0.5, 2.5]], dtype=np.float64), np.array([[1, 1], [2, 2]], dtype=np.float64))], dtype=np.float64)

    p2 = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.3, 0.7], [0.7, 0.3]], dtype=np.float64)
    py_sel = select_front_points(p2, 3)

    ok = True
    ok &= _cmp("zdt_f1", py_f1, _load(b / "mat_zdt_f1.csv").reshape(-1), 1e-12)
    ok &= _cmp("zdt_f2", py_f2, _load(b / "mat_zdt_f2.csv").reshape(-1), 1e-12)
    ok &= _cmp("zdt_f3", py_f3, _load(b / "mat_zdt_f3.csv").reshape(-1), 1e-12)
    ok &= _cmp("zdt_f4", py_f4, _load(b / "mat_zdt_f4.csv").reshape(-1), 1e-12)
    ok &= _cmp("zdt_pf1", py_pf1, _load(b / "mat_zdt_pf1.csv"), 1e-12)
    ok &= _cmp("zdt_pf4", py_pf4, _load(b / "mat_zdt_pf4.csv"), 1e-12)
    ok &= _cmp("zdt_hv_example", py_hv, _load(b / "mat_zdt_hv_example.csv").reshape(-1), 1e-12)
    ok &= _cmp("zdt_cov_ab", py_cab, _load(b / "mat_zdt_cov_ab.csv").reshape(-1), 1e-12)
    ok &= _cmp("zdt_cov_ba", py_cba, _load(b / "mat_zdt_cov_ba.csv").reshape(-1), 1e-12)
    ok &= _cmp("zdt_select_front", py_sel, _load(b / "mat_zdt_select_front.csv"), 1e-12)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
