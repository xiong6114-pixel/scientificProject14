import numpy as np


def variate(sol, charge_points_num, all_demand):
    """
    MATLAB equivalent of variate.m (current effective behavior).

    Effective path:
    1) Pick one random position in [1, charge_points_num]
    2) Replace that element with a uniform random number in [0, 1]
    3) Return immediately

    Notes:
    - all_demand is intentionally kept in the signature for compatibility.
    - Return type is a 1D numpy array.
    """
    _ = all_demand  # Kept for signature compatibility; unused in current logic.

    new_sol = np.asarray(sol, dtype=float).ravel().copy()
    cp = int(charge_points_num)
    if cp < 1 or cp > new_sol.size:
        raise ValueError("charge_points_num must satisfy 1 <= charge_points_num <= len(sol).")

    rand_charge_point = int(np.random.randint(1, cp + 1))  # MATLAB randi([1, charge_points_num])
    new_sol[rand_charge_point - 1] = float(np.random.rand())  # MATLAB rand * (1 - 0) + 0
    return new_sol

    # Unreachable legacy MATLAB code (kept as documentation only):
    # if rand() > 0.5
    #     new_sol(rand_charge_point) = new_sol(rand_charge_point) + 1;
    # else
    #     new_sol(rand_charge_point) = new_sol(rand_charge_point) - 1;
    # end
    # if new_sol(rand_charge_point) < 0
    #     new_sol(rand_charge_point) = 0;
    # end


if __name__ == "__main__":
    # Minimal test
    np.random.seed(123)
    sol = np.array([10, 20, 30, 40, 50], dtype=float)
    out = variate(sol, 5, all_demand=None)

    changed_idx = np.where(out != sol)[0]
    print("output:", out)
    print("changed positions (0-based):", changed_idx)
