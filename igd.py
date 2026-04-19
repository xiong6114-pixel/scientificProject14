import numpy as np

try:
    from scipy.spatial.distance import cdist
except Exception:  # scipy not available
    cdist = None


def IGD(PopObj, PF) -> float:
    """
    MATLAB equivalent:
    Distance = min(pdist2(PF, PopObj), [], 2)
    Score    = mean(Distance)
    """
    pop_obj = np.asarray(PopObj, dtype=float)
    pf = np.asarray(PF, dtype=float)

    if pop_obj.ndim != 2 or pf.ndim != 2:
        raise ValueError("PF and PopObj must both be 2D arrays.")
    if pop_obj.shape[1] != pf.shape[1]:
        raise ValueError("PF and PopObj must have the same number of columns.")
    if pop_obj.shape[0] == 0 or pf.shape[0] == 0:
        raise ValueError("PF and PopObj must be non-empty.")

    if cdist is not None:
        dist_matrix = cdist(pf, pop_obj)
    else:
        # NumPy broadcasting fallback for pairwise Euclidean distance.
        diff = pf[:, None, :] - pop_obj[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))

    distance = np.min(dist_matrix, axis=1)
    score = np.mean(distance)
    return float(score)


if __name__ == "__main__":
    # Minimal test
    pop_obj = np.array([[0.0, 0.0], [1.0, 1.0]])
    pf = np.array([[0.0, 1.0], [1.0, 0.0]])
    print("IGD =", IGD(pop_obj, pf))
