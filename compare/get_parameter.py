from __future__ import annotations

import numpy as np


def get_parameter() -> np.ndarray:
    """Equivalent to MATLAB get_parameter.m in project root."""
    parameter = np.array([0.0, 20.0, 40.0, 12000.0, 5.0], dtype=np.float64)
    return parameter
