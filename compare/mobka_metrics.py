from __future__ import annotations

import numpy as np


def hv_cir_like(final_obj: np.ndarray) -> np.float64:
    """Aligned with project hv_cir.m style (2 objectives, minimization)."""
    final_obj = np.asarray(final_obj, dtype=np.float64)
    if final_obj.size == 0:
        return np.float64(0.0)

    obj1_ref = np.float64(200.0)
    obj2_ref = np.float64(500000000.0)

    sorted_by_obj1 = final_obj[np.argsort(final_obj[:, 0]), :]
    whole_area_x = np.float64(obj1_ref - sorted_by_obj1[0, 0])

    sorted_by_obj2 = final_obj[np.argsort(final_obj[:, 1]), :]
    whole_area_y = np.float64(obj2_ref - sorted_by_obj2[0, 1])

    whole_area = np.float64(whole_area_x * whole_area_y)
    num = final_obj.shape[0]
    for i in range(num - 1):
        whole_area = np.float64(
            whole_area
            - (sorted_by_obj2[i + 1, 0] - sorted_by_obj2[i, 0]) * (sorted_by_obj2[i, 1] - sorted_by_obj2[-1, 1])
        )
    return whole_area


def coverage(a: np.ndarray, b: np.ndarray) -> np.float64:
    """Coverage(A,B): fraction of points in B dominated by A."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.size == 0 or b.size == 0:
        return np.float64(0.0)

    dominated = 0
    for i in range(b.shape[0]):
        bv = b[i, :]
        found = False
        for j in range(a.shape[0]):
            av = a[j, :]
            if np.all(av <= bv) and np.any(av < bv):
                found = True
                break
        if found:
            dominated += 1

    return np.float64(dominated / b.shape[0])
