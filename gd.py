import numpy as np

try:
    from scipy.spatial.distance import cdist
except Exception:
    cdist = None


def GD(PopObj, PF) -> float:
    """
    MATLAB equivalent:
    Distance = min(pdist2(PopObj, PF), [], 2)
    Score    = norm(Distance) / length(Distance)
    """
    distance, score = _gd_distance_score(PopObj, PF)
    return float(score)


def _gd_distance_score(PopObj, PF):
    pop_obj = np.asarray(PopObj, dtype=float)
    pf = np.asarray(PF, dtype=float)

    if pop_obj.ndim != 2 or pf.ndim != 2:
        raise ValueError("PopObj and PF must both be 2D arrays.")
    if pop_obj.shape[1] != pf.shape[1]:
        raise ValueError("PopObj and PF must have the same number of columns.")
    if pop_obj.shape[0] == 0 or pf.shape[0] == 0:
        raise ValueError("PopObj and PF must be non-empty.")

    if cdist is not None:
        dist_matrix = cdist(pop_obj, pf)
    else:
        # NumPy broadcasting fallback for pairwise Euclidean distances.
        diff = pop_obj[:, None, :] - pf[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))

    distance = np.min(dist_matrix, axis=1)
    score = np.linalg.norm(distance) / distance.size
    return distance, float(score)


if __name__ == "__main__":
    # Minimal test and MATLAB-alignment check values
    pop_obj = np.array([[0.0, 0.0], [1.0, 1.0]])
    pf = np.array([[0.0, 1.0], [1.0, 0.0]])

    distance, score = _gd_distance_score(pop_obj, pf)
    print("Distance =", distance)
    print("Score =", score)
