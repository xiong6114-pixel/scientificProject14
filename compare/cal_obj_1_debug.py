from __future__ import annotations

from typing import Any

import numpy as np

from get_dist import get_dist
from waiting_time import compute_p0_and_wj


def cal_obj_1_with_debug(
    sol: np.ndarray,
    demand_points_info: np.ndarray,
    charge_points_info: np.ndarray,
    parameter: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Equivalent to MATLAB mobka/cal_obj_1.m with debug hooks.

    Returns
    - f: shape (2,1), float64
    - debug: dict with intermediate variables
      assignment_matrix: (I,J) float64 one-hot assignment
      dispatch_1based: (I,) int64
      E_j: (J,) float64 assigned demand-point counts per station
      beta_j: (J,) float64 demand sums per station
      P0: (J,) float64 (queue empty probability for active station; 0 for inactive)
      W_j: (J,) float64 waiting time for each station (0 for inactive)
      obj1: scalar float64
      obj2: scalar float64
    """
    sol = np.asarray(sol, dtype=np.float64).reshape(-1)
    demand_points_info = np.asarray(demand_points_info, dtype=np.float64)
    charge_points_info = np.asarray(charge_points_info, dtype=np.float64)
    parameter = np.asarray(parameter, dtype=np.float64).reshape(-1)

    obj1 = np.float64(0.0)
    obj2 = np.float64(0.0)

    i_num = int(demand_points_info.shape[0])
    j_num = int(sol.size)

    assignment_matrix = np.zeros((i_num, j_num), dtype=np.float64)
    dispatch_1based = np.zeros(i_num, dtype=np.int64)

    # MATLAB: chosen_charge_points holds 1-based index.
    chosen_charge_points_1b: list[int] = []
    for j_1b in range(1, j_num + 1):
        if sol[j_1b - 1] > 0:
            chosen_charge_points_1b.append(j_1b)

    for i_1b in range(1, i_num + 1):
        best_dist = np.float64(99999999.0)
        best_charge_point_1b = -1

        for chosen_point_id_1b in chosen_charge_points_1b:
            # Explicit 1-based -> 0-based conversion.
            i0 = i_1b - 1
            j0 = chosen_point_id_1b - 1
            curr_dist = get_dist(
                demand_points_info[i0, 0],
                demand_points_info[i0, 1],
                charge_points_info[j0, 0],
                charge_points_info[j0, 1],
            )
            if curr_dist < best_dist and curr_dist <= parameter[4]:
                best_dist = curr_dist
                best_charge_point_1b = chosen_point_id_1b

        if best_charge_point_1b == -1:
            debug = {
                "assignment_matrix": assignment_matrix,
                "dispatch_1based": dispatch_1based,
                "E_j": np.sum(assignment_matrix, axis=0, dtype=np.float64),
                "beta_j": np.zeros(j_num, dtype=np.float64),
                "P0": np.zeros(j_num, dtype=np.float64),
                "W_j": np.zeros(j_num, dtype=np.float64),
                "obj1": np.float64(np.inf),
                "obj2": np.float64(np.inf),
            }
            return np.array([[np.inf], [np.inf]], dtype=np.float64), debug

        dispatch_1based[i_1b - 1] = best_charge_point_1b
        assignment_matrix[i_1b - 1, best_charge_point_1b - 1] = np.float64(1.0)

    e_j = np.sum(assignment_matrix, axis=0, dtype=np.float64)

    beta_j = np.zeros(j_num, dtype=np.float64)
    p0_j = np.zeros(j_num, dtype=np.float64)
    w_j_list = np.zeros(j_num, dtype=np.float64)

    for j_1b in range(1, j_num + 1):
        if sol[j_1b - 1] > 0:
            n_j = int(sol[j_1b - 1])
            beta = np.float64(0.0)

            for i_1b in range(1, i_num + 1):
                if dispatch_1based[i_1b - 1] == j_1b:
                    beta = np.float64(beta + demand_points_info[i_1b - 1, 2])

            beta_j[j_1b - 1] = beta

            if beta == 0:
                debug = {
                    "assignment_matrix": assignment_matrix,
                    "dispatch_1based": dispatch_1based,
                    "E_j": e_j,
                    "beta_j": beta_j,
                    "P0": p0_j,
                    "W_j": w_j_list,
                    "obj1": np.float64(np.inf),
                    "obj2": np.float64(np.inf),
                }
                return np.array([[np.inf], [np.inf]], dtype=np.float64), debug

            p0, w_j = compute_p0_and_wj(beta_j=beta, n_j=n_j, mu=parameter[1])
            p0_j[j_1b - 1] = p0
            w_j_list[j_1b - 1] = w_j

            # Keep MATLAB failure condition scope: only explicit p0<=0 is rejected.
            if p0 <= 0:
                debug = {
                    "assignment_matrix": assignment_matrix,
                    "dispatch_1based": dispatch_1based,
                    "E_j": e_j,
                    "beta_j": beta_j,
                    "P0": p0_j,
                    "W_j": w_j_list,
                    "obj1": np.float64(np.inf),
                    "obj2": np.float64(np.inf),
                }
                return np.array([[np.inf], [np.inf]], dtype=np.float64), debug

    for i_1b in range(1, i_num + 1):
        chosen_point_id_1b = int(dispatch_1based[i_1b - 1])
        i0 = i_1b - 1
        j0 = chosen_point_id_1b - 1

        curr_dist = get_dist(
            demand_points_info[i0, 0],
            demand_points_info[i0, 1],
            charge_points_info[j0, 0],
            charge_points_info[j0, 1],
        )
        curr_time = np.float64(curr_dist / parameter[2])
        obj1 = np.float64(obj1 + max(w_j_list[j0] + curr_time - parameter[0], np.float64(0.0)) * demand_points_info[i0, 2])

    for j_1b in range(1, j_num + 1):
        if sol[j_1b - 1] > 0:
            obj2 = np.float64(obj2 + sol[j_1b - 1] * parameter[3] + charge_points_info[j_1b - 1, 2])

    f = np.array([[obj1], [obj2]], dtype=np.float64)
    debug = {
        "assignment_matrix": assignment_matrix,
        "dispatch_1based": dispatch_1based,
        "E_j": e_j,
        "beta_j": beta_j,
        "P0": p0_j,
        "W_j": w_j_list,
        "obj1": obj1,
        "obj2": obj2,
    }
    return f, debug


def save_debug_to_csv(debug: dict[str, Any], obj: np.ndarray, out_dir: str) -> None:
    out = out_dir
    np.savetxt(f"{out}/py_assignment_matrix.csv", debug["assignment_matrix"], delimiter=",", fmt="%.17g")
    np.savetxt(f"{out}/py_dispatch_1based.csv", debug["dispatch_1based"].reshape(1, -1), delimiter=",", fmt="%d")
    np.savetxt(f"{out}/py_E_j.csv", debug["E_j"].reshape(1, -1), delimiter=",", fmt="%.17g")
    np.savetxt(f"{out}/py_beta_j.csv", debug["beta_j"].reshape(1, -1), delimiter=",", fmt="%.17g")
    np.savetxt(f"{out}/py_P0.csv", debug["P0"].reshape(1, -1), delimiter=",", fmt="%.17g")
    np.savetxt(f"{out}/py_W_j.csv", debug["W_j"].reshape(1, -1), delimiter=",", fmt="%.17g")
    np.savetxt(f"{out}/py_obj.csv", obj, delimiter=",", fmt="%.17g")
