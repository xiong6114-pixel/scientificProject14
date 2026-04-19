import numpy as np


def cross(sol, charge_points_num, parent, rng=None, seed=None):
    """
    MATLAB equivalent:
    new_sol = sol;
    rand_charge_point = randi([1, charge_points_num]);
    new_sol(rand_charge_point) = parent(rand_charge_point);

    Returns
    -------
    new_sol : np.ndarray, shape=(D,)
    """
    sol_arr = np.asarray(sol, dtype=float).ravel()
    parent_arr = np.asarray(parent, dtype=float).ravel()
    if sol_arr.shape != parent_arr.shape:
        raise ValueError("sol and parent must have the same shape.")

    cp = int(charge_points_num)
    if cp < 1 or cp > sol_arr.size:
        raise ValueError("charge_points_num must satisfy 1 <= charge_points_num <= len(sol).")

    if rng is not None:
        if isinstance(rng, np.random.Generator):
            rand_charge_point = int(rng.integers(1, cp + 1))
        elif isinstance(rng, np.random.RandomState):
            rand_charge_point = int(rng.randint(1, cp + 1))
        else:
            raise TypeError("rng must be numpy.random.Generator or numpy.random.RandomState.")
    elif seed is not None:
        rs = np.random.RandomState(seed)
        rand_charge_point = int(rs.randint(1, cp + 1))
    else:
        rand_charge_point = int(np.random.randint(1, cp + 1))

    new_sol = sol_arr.copy()
    new_sol[rand_charge_point - 1] = parent_arr[rand_charge_point - 1]
    return new_sol


if __name__ == "__main__":
    # Minimal test
    sol = np.array([10, 20, 30, 40, 50], dtype=float)
    parent = np.array([1, 2, 3, 4, 5], dtype=float)
    charge_points_num = 5
    seed = 123

    # Reproduce the selected index for display (same RNG path as function when using seed)
    rs = np.random.RandomState(seed)
    modified_pos_1based = int(rs.randint(1, charge_points_num + 1))

    new_sol = cross(sol, charge_points_num, parent, seed=seed)
    print("modified position (1-based):", modified_pos_1based)
    print("new_sol:", new_sol)
