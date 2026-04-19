from __future__ import annotations

import math
import numpy as np


def get_dist(latitude1: float, longitude1: float, latitude2: float, longitude2: float) -> np.float64:
    """Equivalent to MATLAB mobka/get_dist.m."""
    latitude1 = np.float64(latitude1) * np.float64(math.pi / 180.0)
    latitude2 = np.float64(latitude2) * np.float64(math.pi / 180.0)
    longitude1 = np.float64(longitude1) * np.float64(math.pi / 180.0)
    longitude2 = np.float64(longitude2) * np.float64(math.pi / 180.0)

    a = np.float64(latitude1 - latitude2)
    b = np.float64(longitude1 - longitude2)

    distance = np.float64(
        math.asin(math.sqrt(math.sin(a / 2.0) ** 2 + math.cos(latitude1) * math.cos(latitude2) * math.sin(b / 2.0) ** 2))
        * 2.0
    )

    radius = np.float64(6378.137)
    distance = np.float64(distance * radius)

    # MATLAB: distance = round(distance * 10000) / 10000;
    distance = np.float64(np.round(distance * np.float64(10000.0)) / np.float64(10000.0))
    return distance
