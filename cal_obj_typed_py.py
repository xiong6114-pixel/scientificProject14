from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from get_dist import get_dist
from get_parameter import get_parameter


@dataclass
class TypeInfo:
    nmax: np.ndarray
    mu: np.ndarray
    cost_per_pile: np.ndarray


def _build_default_type_info(parameter: np.ndarray, nmax: Optional[np.ndarray] = None) -> TypeInfo:
    base_mu = float(parameter[1])
    base_cost = float(parameter[3])
    if nmax is None:
        nmax = np.array([40, 40, 40], dtype=int)
    return TypeInfo(
        nmax=np.asarray(nmax, dtype=int).reshape(3),
        mu=np.array([0.6 * base_mu, 1.0 * base_mu, 1.8 * base_mu], dtype=float),
        cost_per_pile=np.array([0.8 * base_cost, 1.0 * base_cost, 1.6 * base_cost], dtype=float),
    )


def build_distance_matrix(demand_xy: np.ndarray, charge_xy: np.ndarray) -> np.ndarray:
    n = demand_xy.shape[0]
    m = charge_xy.shape[0]
    d = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            d[i, j] = get_dist(
                float(demand_xy[i, 0]),
                float(demand_xy[i, 1]),
                float(charge_xy[j, 0]),
                float(charge_xy[j, 1]),
            )
    return d


def mmc_wq(lam: float, mu: float, c: int) -> float:
    a = lam / mu
    rho = lam / (c * mu)
    if rho >= 1.0:
        return math.inf

    s = 0.0
    for k in range(c):
        s += math.exp(k * math.log(a) - math.lgamma(k + 1.0))

    term = math.exp(c * math.log(a) - math.lgamma(c + 1.0)) * (1.0 / (1.0 - rho))
    p0 = 1.0 / (s + term)
    pw = term * p0
    lq = pw * rho / (1.0 - rho)
    return lq / lam


def _analyze_typed_solution(
    sol: np.ndarray,
    demand_points_info: np.ndarray,
    charge_points_info: np.ndarray,
    parameter: np.ndarray,
    type_info: TypeInfo,
    dist_matrix: Optional[np.ndarray],
) -> dict:
    n = demand_points_info.shape[0]
    m = charge_points_info.shape[0]

    result = {
        "feasible": False,
        "reason": "",
        "obj": np.array([math.inf, math.inf], dtype=float),
        "chosen": np.empty((0,), dtype=int),
        "dispatch": np.empty((0,), dtype=int),
        "best_distance": np.empty((0,), dtype=float),
        "station_table": np.empty((0, 8), dtype=float),
        "n_slow": np.zeros(m, dtype=int),
        "n_fast": np.zeros(m, dtype=int),
        "n_ultra": np.zeros(m, dtype=int),
        "n_total": np.zeros(m, dtype=int),
        "lambda_vec": np.zeros(m, dtype=float),
        "mu_eff": np.zeros(m, dtype=float),
        "service_capacity": np.zeros(m, dtype=float),
        "rho": np.full(m, np.nan, dtype=float),
        "wq": np.full(m, np.nan, dtype=float),
        "unreachable_demand_idx": np.empty((0,), dtype=int),
        "bad_station_idx": np.empty((0,), dtype=int),
        "total_lambda": 0.0,
        "total_service_capacity": 0.0,
        "service_gap": 0.0,
    }

    te = float(parameter[0])
    v = float(parameter[2])
    max_dist = float(parameter[4])

    sol = np.asarray(sol).reshape(-1)
    if sol.size != 3 * m:
        result["reason"] = f"dimension_mismatch: expected {3 * m}, got {sol.size}"
        return result

    sol = np.round(sol).astype(int)
    sol[sol < 0] = 0

    n_slow = sol[0:m]
    n_fast = sol[m : 2 * m]
    n_ultra = sol[2 * m : 3 * m]
    n_total = n_slow + n_fast + n_ultra
    chosen = np.where(n_total > 0)[0]

    result["n_slow"] = n_slow
    result["n_fast"] = n_fast
    result["n_ultra"] = n_ultra
    result["n_total"] = n_total
    result["chosen"] = chosen

    if chosen.size == 0:
        result["reason"] = "no_built_station"
        return result

    if dist_matrix is None:
        dist_matrix = build_distance_matrix(demand_points_info[:, 0:2], charge_points_info[:, 0:2])

    d_sub = dist_matrix[:, chosen].copy()
    d_sub[d_sub > max_dist] = math.inf
    loc = np.argmin(d_sub, axis=1)
    best = d_sub[np.arange(n), loc]
    unreachable = np.where(~np.isfinite(best))[0]
    result["best_distance"] = best
    result["unreachable_demand_idx"] = unreachable

    if unreachable.size > 0:
        result["reason"] = f"unreachable_demand: {unreachable.size}"
        return result

    dispatch = chosen[loc]
    demand_w = demand_points_info[:, 2].astype(float)
    lambda_vec = np.bincount(dispatch, weights=demand_w, minlength=m).astype(float)

    mu_eff = np.zeros(m, dtype=float)
    service_capacity = np.zeros(m, dtype=float)
    rho = np.full(m, np.nan, dtype=float)
    wq = np.full(m, np.nan, dtype=float)
    station_rows = []
    bad_station_idx = []

    for j in chosen:
        lam = float(lambda_vec[j])
        c = int(n_total[j])

        if lam <= 0:
            bad_station_idx.append(j)
            station_rows.append([j, lam, c, math.nan, math.nan, math.nan, 1.0, 0.0])
            continue

        if c <= 0:
            bad_station_idx.append(j)
            station_rows.append([j, lam, c, math.nan, math.nan, math.nan, 2.0, 0.0])
            continue

        mu_eff_j = float(
            (n_slow[j] * type_info.mu[0] + n_fast[j] * type_info.mu[1] + n_ultra[j] * type_info.mu[2]) / c
        )
        mu_eff[j] = mu_eff_j

        if mu_eff_j <= 0:
            bad_station_idx.append(j)
            station_rows.append([j, lam, c, mu_eff_j, math.nan, math.nan, 3.0, 0.0])
            continue

        service_j = c * mu_eff_j
        rho_j = lam / service_j
        service_capacity[j] = service_j
        rho[j] = rho_j

        if rho_j >= 1.0:
            bad_station_idx.append(j)
            station_rows.append([j, lam, c, mu_eff_j, service_j, rho_j, 4.0, 0.0])
            continue

        wq_j = mmc_wq(lam, mu_eff_j, c)
        wq[j] = wq_j
        if (not np.isfinite(wq_j)) or wq_j < 0:
            bad_station_idx.append(j)
            station_rows.append([j, lam, c, mu_eff_j, service_j, rho_j, 5.0, wq_j])
            continue

        station_rows.append([j, lam, c, mu_eff_j, service_j, rho_j, 0.0, wq_j])

    result["dispatch"] = dispatch
    result["lambda_vec"] = lambda_vec
    result["mu_eff"] = mu_eff
    result["service_capacity"] = service_capacity
    result["total_lambda"] = float(np.sum(lambda_vec))
    result["total_service_capacity"] = float(np.sum(service_capacity))
    result["service_gap"] = float(result["total_lambda"] - result["total_service_capacity"])
    result["rho"] = rho
    result["wq"] = wq
    result["station_table"] = np.asarray(station_rows, dtype=float) if station_rows else np.empty((0, 8), dtype=float)
    result["bad_station_idx"] = np.asarray(bad_station_idx, dtype=int)

    if len(bad_station_idx) > 0:
        bad_codes = result["station_table"][:, 6].astype(int)
        if np.any(bad_codes == 4):
            result["reason"] = f"overloaded_station_rho_ge_1: {int(np.sum(bad_codes == 4))}"
        elif np.any(bad_codes == 1):
            result["reason"] = f"zero_assigned_demand_station: {int(np.sum(bad_codes == 1))}"
        elif np.any(bad_codes == 5):
            result["reason"] = f"invalid_waiting_time_station: {int(np.sum(bad_codes == 5))}"
        elif np.any(bad_codes == 3):
            result["reason"] = f"nonpositive_mu_station: {int(np.sum(bad_codes == 3))}"
        else:
            result["reason"] = f"invalid_station: {len(bad_station_idx)}"
        return result

    travel_time = dist_matrix[np.arange(n), dispatch] / v
    penalty = np.maximum(wq[dispatch] + travel_time - te, 0.0)
    obj1 = float(np.sum(penalty * demand_w))
    built = (n_total > 0).astype(float)
    obj2 = float(
        np.sum(
            built
            * (
                charge_points_info[:, 2]
                + n_slow * type_info.cost_per_pile[0]
                + n_fast * type_info.cost_per_pile[1]
                + n_ultra * type_info.cost_per_pile[2]
            )
        )
    )

    result["feasible"] = True
    result["reason"] = "ok"
    result["obj"] = np.array([obj1, obj2], dtype=float)
    return result


def cal_obj_typed(
    sol: np.ndarray,
    demand_points_info: np.ndarray,
    charge_points_info: np.ndarray,
    parameter: Optional[np.ndarray] = None,
    type_info: Optional[TypeInfo] = None,
    dist_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return objective vector [obj1, obj2].

    Invalid solutions return [inf, inf] (aligned with typed MATLAB logic).
    """
    if parameter is None:
        parameter = get_parameter()
    parameter = np.asarray(parameter, dtype=float)

    if type_info is None:
        type_info = _build_default_type_info(parameter)

    analysis = _analyze_typed_solution(
        sol=sol,
        demand_points_info=demand_points_info,
        charge_points_info=charge_points_info,
        parameter=parameter,
        type_info=type_info,
        dist_matrix=dist_matrix,
    )
    return np.asarray(analysis["obj"], dtype=float)


def diagnose_cal_obj_typed(
    sol: np.ndarray,
    demand_points_info: np.ndarray,
    charge_points_info: np.ndarray,
    parameter: Optional[np.ndarray] = None,
    type_info: Optional[TypeInfo] = None,
    dist_matrix: Optional[np.ndarray] = None,
    top_k: int = 20,
    print_fn=print,
) -> dict:
    if parameter is None:
        parameter = get_parameter()
    parameter = np.asarray(parameter, dtype=float)

    if type_info is None:
        type_info = _build_default_type_info(parameter)

    analysis = _analyze_typed_solution(
        sol=sol,
        demand_points_info=demand_points_info,
        charge_points_info=charge_points_info,
        parameter=parameter,
        type_info=type_info,
        dist_matrix=dist_matrix,
    )

    if print_fn is not None:
        print_fn("=== cal_obj_typed diagnostics ===")
        print_fn(f"feasible: {analysis['feasible']}")
        print_fn(f"reason: {analysis['reason']}")
        print_fn(f"objective: {analysis['obj']}")
        print_fn(f"built_station_count: {analysis['chosen'].size}")
        print_fn(f"total_lambda: {analysis['total_lambda']:.2f}")
        print_fn(f"total_service_capacity: {analysis['total_service_capacity']:.2f}")
        print_fn(f"service_gap(lambda-service): {analysis['service_gap']:.2f}")

        if analysis["unreachable_demand_idx"].size > 0:
            print_fn(f"unreachable_demand_count: {analysis['unreachable_demand_idx'].size}")
            print_fn(f"unreachable_demand_sample: {analysis['unreachable_demand_idx'][:top_k].tolist()}")

        station_table = analysis["station_table"]
        if station_table.shape[0] > 0:
            valid_rho = station_table[:, 5]
            order = np.argsort(-np.nan_to_num(valid_rho, nan=-1.0))
            print_fn("top stations by rho:")
            header = "station  lambda  c  mu_eff  service  rho  code  wq"
            print_fn(header)
            for row in station_table[order[: min(top_k, station_table.shape[0])]]:
                print_fn(
                    f"{int(row[0]):>7}  {row[1]:>7.2f}  {int(row[2]):>2}  {row[3]:>6.2f}  "
                    f"{row[4]:>7.2f}  {row[5]:>5.3f}  {int(row[6])}  {row[7]:>8.4f}"
                )

            bad = station_table[station_table[:, 6] > 0]
            if bad.shape[0] > 0:
                print_fn(f"bad_station_count: {bad.shape[0]}")
                print_fn("bad station code meaning: 1=lam<=0, 2=c<=0, 3=mu_eff<=0, 4=rho>=1, 5=wq invalid")

    return analysis


def build_default_type_info(parameter: Optional[np.ndarray] = None, nmax: Optional[np.ndarray] = None) -> TypeInfo:
    if parameter is None:
        parameter = get_parameter()
    return _build_default_type_info(np.asarray(parameter, dtype=float), nmax=nmax)
