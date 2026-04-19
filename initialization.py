import numpy as np


def initialization(SearchAgents_no, dim, ub, lb, rng=None, seed=None):
    """
    Python equivalent of MATLAB initialization(SearchAgents_no, dim, ub, lb).

    Parameters
    ----------
    SearchAgents_no : int
    dim : int
    ub : scalar or array-like of length dim
    lb : scalar or array-like of length dim
    rng : numpy.random.Generator or numpy.random.RandomState, optional
        External RNG. If provided, seed is ignored.
    seed : int, optional
        Seed used only when rng is None.

    Returns
    -------
    Positions : np.ndarray, shape=(SearchAgents_no, dim)
    """
    search_agents_no = int(SearchAgents_no)
    dim = int(dim)

    ub_arr = np.asarray(ub, dtype=float)
    lb_arr = np.asarray(lb, dtype=float)
    local_rng = _resolve_rng(rng=rng, seed=seed)

    # Match MATLAB Boundary_no = size(ub,2) behavior for scalar vs vector use.
    boundary_no = 1 if ub_arr.size == 1 else ub_arr.size

    if boundary_no == 1:
        if lb_arr.size != 1:
            raise ValueError("ub and lb must both be scalars, or both be vectors of length dim.")
        u = _rand((search_agents_no, dim), rng=local_rng)
        positions = u * (float(ub_arr.reshape(-1)[0]) - float(lb_arr.reshape(-1)[0])) + float(lb_arr.reshape(-1)[0])
        return np.asarray(positions, dtype=float)

    # If each variable has a different lb and ub: keep per-dimension loop logic.
    ub_vec = ub_arr.reshape(-1)
    lb_vec = lb_arr.reshape(-1)
    if ub_vec.size != dim or lb_vec.size != dim:
        raise ValueError("When ub/lb are vectors, both must have length dim.")

    positions = np.empty((search_agents_no, dim), dtype=float)
    for i in range(dim):
        ub_i = ub_vec[i]
        lb_i = lb_vec[i]
        u = _rand((search_agents_no, 1), rng=local_rng)
        positions[:, i] = (u * (ub_i - lb_i) + lb_i).reshape(-1)
    return positions


def _resolve_rng(rng=None, seed=None):
    if rng is not None:
        if isinstance(rng, np.random.Generator):
            return rng
        if isinstance(rng, np.random.RandomState):
            return rng
        raise TypeError("rng must be numpy.random.Generator or numpy.random.RandomState.")

    if seed is not None:
        return np.random.RandomState(seed)

    return None


def _rand(shape, rng=None):
    if isinstance(rng, np.random.Generator):
        return rng.random(shape)
    if isinstance(rng, np.random.RandomState):
        return rng.rand(*shape)

    # Default logic: same style as MATLAB rand via NumPy global RNG call.
    return np.random.rand(*shape)


if __name__ == "__main__":
    # Minimal tests
    p1 = initialization(4, 3, 1.0, 0.0, seed=42)  # scalar bounds
    print("scalar bounds -> shape:", p1.shape)
    print("within [0,1]:", bool(np.all((p1 >= 0.0) & (p1 <= 1.0))))

    ub = np.array([1.0, 5.0, 10.0])
    lb = np.array([0.0, -5.0, 2.0])
    p2 = initialization(4, 3, ub, lb, seed=7)  # vector bounds
    print("vector bounds -> shape:", p2.shape)
    print("per-dim lower check:", bool(np.all(p2 >= lb)))
    print("per-dim upper check:", bool(np.all(p2 <= ub)))
