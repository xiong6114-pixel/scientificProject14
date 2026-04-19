from __future__ import annotations

import os

import numpy as np


def get_parameter() -> np.ndarray:
    """Equivalent to MATLAB get_parameter.m in project root."""
    parameter = np.array(
        [
            float(os.environ.get("PARAM_TE", "0.0")),
            float(os.environ.get("PARAM_MU", "40.0")),
            float(os.environ.get("PARAM_V", "40.0")),
            float(os.environ.get("PARAM_CJ", "12000.0")),
            float(os.environ.get("PARAM_CO", "5.0")),
        ],
        dtype=np.float64,
    )
    return parameter
