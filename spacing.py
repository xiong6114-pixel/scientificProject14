import numpy as np


def Spacing(PopObj, PF) -> float:
    """
    MATLAB equivalent of:
        Distance = pdist2(PopObj,PopObj,'cityblock');
        Distance(logical(eye(size(Distance,1)))) = inf;
        Score    = std(min(Distance,[],2));

    Note:
    - PF is not used in the current implementation, but is kept for
      interface consistency with MATLAB: Spacing(PopObj, PF).
    """
    _ = PF  # Keep interface consistent with MATLAB; intentionally unused.

    pop_obj = np.asarray(PopObj, dtype=float)
    if pop_obj.ndim != 2:
        raise ValueError("PopObj must be a 2D array.")
    if pop_obj.shape[0] == 0:
        raise ValueError("PopObj must be non-empty.")

    # Manhattan (cityblock) distance matrix: pdist2(PopObj, PopObj, 'cityblock')
    diff = np.abs(pop_obj[:, None, :] - pop_obj[None, :, :])
    distance = np.sum(diff, axis=2)

    # Set diagonal to inf
    np.fill_diagonal(distance, np.inf)

    # min(Distance, [], 2)  <=> row-wise min
    nearest = np.min(distance, axis=1)

    # MATLAB std default uses sample std (N-1 normalization), so ddof=1.
    score = np.std(nearest, ddof=1)
    return float(score)


if __name__ == "__main__":
    # Minimal test
    pop_obj = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=float,
    )
    pf = np.array([[0.0, 0.0]], dtype=float)  # unused, kept for signature compatibility
    score = Spacing(pop_obj, pf)
    print("Score =", score)
