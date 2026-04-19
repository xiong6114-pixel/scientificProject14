from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def zdt_cost(x: np.ndarray, prob_id: int) -> np.ndarray:
    """Equivalent to MATLAB mobka/ZDT_cost.m.

    Returns column-like vector [f1; f2] as shape (2,).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    d = x.size

    if d < 2:
        raise ValueError("x must have at least 2 variables for ZDT problems.")

    if prob_id == 1:
        f1 = np.float64(x[0])
        g = np.float64(1.0 + 9.0 * np.sum(x[1:]) / (d - 1))
        f2 = np.float64(g * (1.0 - np.sqrt(f1 / g)))
    elif prob_id == 2:
        f1 = np.float64(x[0])
        g = np.float64(1.0 + 9.0 * np.sum(x[1:]) / (d - 1))
        f2 = np.float64(g * (1.0 - (f1 / g) ** 2))
    elif prob_id == 3:
        f1 = np.float64(x[0])
        g = np.float64(1.0 + 9.0 * np.sum(x[1:]) / (d - 1))
        f2 = np.float64(g * (1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)))
    elif prob_id == 4:
        f1 = np.float64(x[0])
        g = np.float64(1.0 + 10.0 * (d - 1) + np.sum(x[1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[1:])))
        f2 = np.float64(g * (1.0 - np.sqrt(f1 / g)))
    else:
        raise ValueError("Unknown ZDT problem ID.")

    return np.array([f1, f2], dtype=np.float64)


def zdt_true_pf(prob_id: int, n_points: int) -> np.ndarray:
    """Equivalent to MATLAB mobka/ZDT_truePF.m. Output shape: (n_points,2)."""
    x1 = np.linspace(0.0, 1.0, n_points, dtype=np.float64)

    if prob_id == 1:
        f1 = x1
        g = np.float64(1.0)
        f2 = g * (1.0 - np.sqrt(f1 / g))
    elif prob_id == 2:
        f1 = x1
        g = np.float64(1.0)
        f2 = g * (1.0 - (f1 / g) ** 2)
    elif prob_id == 3:
        f1 = x1
        g = np.float64(1.0)
        f2 = g * (1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1))
    elif prob_id == 4:
        # MATLAB comments: ZDT4 PF same as ZDT1 formula.
        f1 = x1
        g = np.float64(1.0)
        f2 = g * (1.0 - np.sqrt(f1 / g))
    else:
        raise ValueError("Unknown ZDT problem ID.")

    return np.column_stack([f1, f2]).astype(np.float64)


def zdt_hv_reference(prob_id: int) -> np.ndarray:
    """Reference points aligned with mobka/run_ZDT_IMOBKA_vs_MOBKA.m."""
    if prob_id == 4:
        return np.array([1.1, 150.0], dtype=np.float64)
    return np.array([1.1, 1.1], dtype=np.float64)


def calc_hv_2d(p: np.ndarray, ref: np.ndarray) -> np.float64:
    """Equivalent to MATLAB mobka/calc_hv_2d.m.

    Minimization case. ref must be worse than all points.
    """
    p = np.asarray(p, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)

    if p.size == 0:
        return np.float64(0.0)

    keep = (p[:, 0] <= ref[0]) & (p[:, 1] <= ref[1])
    p = p[keep, :]

    if p.size == 0:
        return np.float64(0.0)

    p = p[np.argsort(p[:, 0]), :]

    hv = np.float64(0.0)
    prev_f1 = np.float64(ref[0])
    for i in range(p.shape[0] - 1, -1, -1):
        f1 = np.float64(p[i, 0])
        f2 = np.float64(p[i, 1])
        hv = np.float64(hv + (prev_f1 - f1) * (ref[1] - f2))
        prev_f1 = f1

    return hv


def calc_coverage(a: np.ndarray, b: np.ndarray) -> np.float64:
    """Equivalent to MATLAB mobka/calc_coverage.m.

    Direction is C(A,B): fraction of B dominated by A.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.size == 0 or b.size == 0:
        return np.float64(0.0)

    dominated = 0
    for i in range(b.shape[0]):
        bv = b[i, :]
        le = (a <= bv)
        lt = (a < bv)
        is_dom = np.any(np.all(le, axis=1) & np.any(lt, axis=1))
        if bool(is_dom):
            dominated += 1

    return np.float64(dominated / b.shape[0])


def select_front_points(p: np.ndarray, k: int) -> np.ndarray:
    """Equivalent to helper in run_ZDT_IMOBKA_vs_MOBKA.m."""
    p = np.asarray(p, dtype=np.float64)
    if p.size == 0:
        return p

    p = p[np.argsort(p[:, 0]), :]
    n = p.shape[0]
    if n <= k:
        return p

    idx_1b = np.round(np.linspace(1.0, float(n), k)).astype(np.int64)
    idx_1b[idx_1b < 1] = 1
    idx_1b[idx_1b > n] = n

    idx_0b = np.unique(idx_1b - 1)
    return p[idx_0b, :]


def prepare_pareto_plot_data(true_pf: np.ndarray, front_a: np.ndarray, front_b: np.ndarray, n_plot_a: int, n_plot_b: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return plotting data positions. Style can differ, positions must match."""
    t = np.asarray(true_pf, dtype=np.float64)
    a = select_front_points(np.asarray(front_a, dtype=np.float64), n_plot_a)
    b = select_front_points(np.asarray(front_b, dtype=np.float64), n_plot_b)
    return t, a, b


def plot_pareto_front(true_pf: np.ndarray, front_a: np.ndarray, front_b: np.ndarray, title: str, out_png: Path | None = None):
    """Plot fronts. Point coordinates are exactly the passed arrays."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(true_pf[:, 0], true_pf[:, 1], "k.", markersize=4, label="TruePF")
    ax.plot(front_a[:, 0], front_a[:, 1], "bo", markersize=4, label="A")
    ax.plot(front_b[:, 0], front_b[:, 1], "r.", markersize=8, label="B")
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True)
    if out_png is not None:
        fig.savefig(out_png, dpi=180)
    return fig, ax
