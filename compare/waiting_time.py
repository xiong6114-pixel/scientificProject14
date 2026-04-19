from __future__ import annotations

import math
import numpy as np


def factorial_float64(n: int) -> np.float64:
    """MATLAB-like double factorial behavior: n>170 overflows to inf in double."""
    if n < 0:
        return np.float64(np.nan)
    if n > 170:
        return np.float64(np.inf)
    return np.float64(float(math.factorial(n)))


def compute_p0_and_wj(beta_j: float, n_j: int, mu: float) -> tuple[np.float64, np.float64]:
    """Exact formula order as in mobka/cal_obj_1.m."""
    beta_j = np.float64(beta_j)
    mu = np.float64(mu)
    n_j = int(n_j)

    p_0 = np.float64(0.0)
    for k in range(0, n_j):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            first = np.float64((beta_j / mu) ** k) / factorial_float64(k)
            second_num = np.float64(n_j) * np.float64((beta_j / mu) ** n_j)
            second_den = factorial_float64(n_j) * np.float64(n_j - (beta_j / mu))
            second = np.float64(second_num / second_den)
        p_0 = np.float64(p_0 + first + second)

    if p_0 <= 0:
        return np.float64(np.inf), np.float64(np.inf)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        p0_prob = np.float64(p_0 ** np.float64(-1.0))
        upper = np.float64(n_j) * np.float64((beta_j / mu) ** (n_j + 1))
        down = np.float64(beta_j) * factorial_float64(n_j) * np.float64(n_j - (beta_j / mu)) ** np.float64(2.0)
        w_j = np.float64((upper / down) * p0_prob)

    return p0_prob, w_j
