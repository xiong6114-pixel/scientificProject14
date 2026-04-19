import math
from itertools import combinations

import numpy as np


def UniformPoint(N, M):
    """
    MATLAB equivalent:
    [W, N] = UniformPoint(N, M)

    Returns
    -------
    W : np.ndarray, shape=(N_new, M)
    N : np.ndarray, shape=()  (0-d scalar array), equal to W.shape[0]
    """
    N_in = int(N)
    M = int(M)

    H1 = 1
    while math.comb(H1 + M, M - 1) <= N_in:
        H1 = H1 + 1

    W = _nchoosek_1_to_n(H1 + M - 1, M - 1) - np.tile(np.arange(0, M - 1), (_rows(H1 + M - 1, M - 1), 1)) - 1
    W = (np.hstack([W, np.zeros((W.shape[0], 1)) + H1]) - np.hstack([np.zeros((W.shape[0], 1)), W])) / H1

    if H1 < M:
        H2 = 0
        while math.comb(H1 + M - 1, M - 1) + math.comb(H2 + M, M - 1) <= N_in:
            H2 = H2 + 1

        if H2 > 0:
            W2 = (
                _nchoosek_1_to_n(H2 + M - 1, M - 1)
                - np.tile(np.arange(0, M - 1), (_rows(H2 + M - 1, M - 1), 1))
                - 1
            )
            W2 = (np.hstack([W2, np.zeros((W2.shape[0], 1)) + H2]) - np.hstack([np.zeros((W2.shape[0], 1)), W2])) / H2
            W = np.vstack([W, W2 / 2 + 1 / (2 * M)])

    W = np.maximum(W, 1e-6)
    N_out = np.array(W.shape[0], dtype=int)
    return W.astype(float, copy=False), N_out


def _nchoosek_1_to_n(n, k):
    data = list(combinations(range(1, n + 1), k))
    return np.asarray(data, dtype=float)


def _rows(n, k):
    return math.comb(n, k)


if __name__ == "__main__":
    # Minimal test
    W, N_out = UniformPoint(10, 3)
    print("W.shape =", W.shape)
    print("N =", N_out)
    print("W (first 5 rows) =")
    print(W[:5])
