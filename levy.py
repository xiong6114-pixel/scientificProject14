import math
import os

import numpy as np


def levy(dim):
    """
    MATLAB equivalent of levy.m
    Generates a Levy step vector of length dim.
    """
    n = 1
    m = int(dim)
    beta = float(os.environ.get("LEVY_BETA", "1.1"))

    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)  # Numerator
    den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)  # Denominator
    sigma_u = (num / den) ** (1 / beta)  # Standard deviation

    u = np.random.normal(0.0, sigma_u, size=(n, m))
    v = np.random.normal(0.0, 1.0, size=(n, m))

    z = u / (np.abs(v) ** (1 / beta))
    return z.ravel()


if __name__ == "__main__":
    # Minimal test
    z = levy(5)
    print("z:", z)
    print("shape:", z.shape)
    print("len:", len(z))
